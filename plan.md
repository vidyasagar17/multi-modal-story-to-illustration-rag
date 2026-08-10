# Implementation Plan

Build plan for the multi-modal story-to-illustration agent described in [README.md](README.md).
The goal is an open-weights pipeline: story text in, illustrated storybook out, with a RAG
memory that keeps characters, settings, and art style consistent across pages.

## Guiding Constraints

- **Open weights only.** Every model in the pipeline must have downloadable weights under a
  permissive license. Hosted *inference* of those weights (Hugging Face Inference Providers,
  fal) is allowed; closed models are not. The test is "could I self-host this later?", not
  "am I running it right now?"
- **Swappable backends.** LLM, image model, embedder, and vector store each sit behind a
  small interface so a hosted setup (HF router + fal) and a local one (Ollama/vLLM + SD)
  run the same code.
- **Deterministic where possible.** Seeds, model IDs, and prompts are recorded per page so
  a run can be reproduced or a single page re-rendered without redoing the book.
- **Fail soft at the page level only.** A failed page produces a placeholder and a logged
  error, not a dead run. This is a product requirement about long batch jobs — it is not
  licence to wrap every call in a try/except. Everything below the page loop fails loudly.
- **Build the simplest thing that works, one step at a time.** Each phase below ships a
  working increment with evidence that it works before the next one starts. No speculative
  abstraction: a protocol earns its second implementation when a second backend is actually
  needed, not before.
- **`uv` for everything Python.** `uv run`, `uv add`, `uv sync`. No bare `python3`, `pip
  install`, or `python -m`.

See [CLAUDE.md](CLAUDE.md) for the full working conventions this plan assumes.

## Proposed Layout

```
storyillus/
  __init__.py
  config.py            # dataclass config + YAML loading, model/backend selection
  models.py            # Page, ScenePlan, MemoryRecord, PageResult dataclasses
  ingest/
    gutenberg.py       # DONE: catalog search, fetch, boilerplate strip, -> .md
    loader.py          # .txt / .md -> raw text (.epub only if a real story needs it)
    chapters.py        # split a fetched .md on its `## ` headings -> Chapter list
    paginate.py        # split into story pages (chapter, blank-line, or token budget)
  llm/
    base.py            # DONE: LLMBackend protocol + the complete_json wrapper
    fake.py            # DONE: FakeLLM — scripted replies, records prompts
    openai_compat.py   # one client: base_url + api_key + model_id
                       # covers HF router, vLLM, and Ollama's /v1 endpoint
  imagegen/
    base.py            # DONE: ImageBackend protocol: generate(prompt, negative, seed) -> PIL
    fake.py            # DONE: FakeImage — flat colour keyed by seed
    diffusers_local.py # Z-Image Turbo / SDXL via diffusers
  memory/
    store.py           # VectorStore protocol: add(), query(), persist()
    chroma_store.py    # the one implementation; a second arrives when something needs it
    embedder.py        # sentence-transformers wrapper
    schema.py          # record kinds: character, setting, style, scene
  agent/
    condense.py        # step 1: page -> condensed narrative beats
    retrieve.py        # step 2: build retrieval query, fetch context
    prompt_builder.py  # step 3: condensed + context -> image prompt
    illustrate.py      # step 4: call image backend
    update_memory.py   # step 5: extract new entities, write back
    graph.py           # orchestration wiring the five steps per page
  render/
    book.py            # assemble pages + images into HTML and PDF
  api/
    server.py          # FastAPI: search, fetch, submit run, poll status, serve book
    jobs.py            # run registry on disk; illustration is a background job
  web/
    static/            # single-page UI, served by the same process
  cli.py               # `uv run storyillus run story.md --config configs/hosted.yaml`
configs/
  hosted.yaml          # HF router LLM + fal images — the default on an 8 GB M3
  local.yaml           # Ollama + Z-Image Turbo, CPU/MPS friendly
  workstation.yaml     # vLLM + Qwen-Image locally
tests/
  ...
books/                 # gitignored: fetched .md files
.cache/                # gitignored: pg_catalog.csv, run artifacts
examples/
  short_story.txt
```

## Core Data Model

```python
@dataclass(frozen=True)
class Book:            # a catalog entry, before any text is downloaded
    id: int                    # Gutenberg ebook number
    title: str
    author: str
    language: str

@dataclass
class Chapter:         # a `## ` section of a fetched book — the unit of illustration
    index: int
    heading: str
    text: str

@dataclass
class Page:            # one unit of story text
    index: int
    text: str

@dataclass
class ScenePlan:       # the LLM's structured read of a page
    summary: str               # 2-4 vivid sentences
    characters: list[str]      # names present in this scene
    setting: str
    mood: str
    key_visual: str            # the single moment worth drawing

@dataclass
class MemoryRecord:    # what lives in the vector store
    kind: Literal["character", "setting", "style", "scene"]
    name: str
    description: str           # canonical visual description
    first_seen_page: int
    embedding_text: str        # what actually gets embedded

@dataclass
class PageResult:
    page: Page
    plan: ScenePlan
    retrieved: list[MemoryRecord]
    image_prompt: str
    negative_prompt: str
    seed: int
    image_path: Path | None
    error: str | None
```

`ScenePlan` is requested from the LLM as JSON with a schema in the prompt, parsed with a
retry-on-invalid-JSON wrapper (two retries, then fall back to a summary-only plan).

Provider-native structured output (`response_format: json_schema`) is not relied on —
support varies across the providers behind the HF router. Schema-in-prompt plus parse plus
retry is the portable path. The same wrapper also retries `429` and `503` with exponential
backoff, since rate limits and cold starts are normal operating conditions on hosted
inference, not failures.

## Scope Decisions

Two questions came out of adding book search and a UI. Both are settled here so the phases
below have something concrete to build against; both are cheap to revisit.

**A chapter is the unit of illustration, not a book.** Frankenstein's plain text is 438,841
characters — roughly 110k tokens, which paginates to 200+ pages and therefore 200+ diffusion
renders, over an hour of GPU time per book. Illustrating a whole novel on request is not an
interactive operation. So `storyillus` fetches a whole book but illustrates a selected
chapter range, defaulting to one chapter. `Chapter` exists as a first-class type for exactly
this reason. Agent-chosen "key scenes across the whole book" is a Phase 5 option, not the
default — it costs an extra full-text LLM pass before a single image is drawn.

**The frontend is a local single-user app.** The README's mission says no data leaves the
user's machine, and real accounts contradict that. So: a server bound to localhost, no auth,
no user table. "User info" means locally stored preferences — art style, default config,
recently fetched books — not an identity. Nothing in the data model gains an owner field.
Multi-user hosting would change the licence, storage, and privacy story all at once, and is
out of scope until someone actually asks to deploy it.

## Consistency Strategy

This is the part that makes or breaks the project, so it gets explicit rules rather than
"just retrieve some context":

1. **Character sheets are write-once, amend-rarely.** The first time a character appears,
   the LLM writes a canonical visual description (age, build, hair, clothing, distinctive
   features). Later pages retrieve that sheet verbatim and are instructed not to
   contradict it. Amendments are only allowed when the story text explicitly changes
   something (a costume change, an injury), and are appended as dated revisions.
2. **Style block is global and constant.** One art-style string (medium, palette, line
   quality, lighting) is generated once from the whole story's tone and prepended to every
   image prompt. It is never retrieved — it's always present.
3. **Retrieval is scoped, not fuzzy.** For each page: fetch character records by exact name
   match on the names in `ScenePlan.characters`, plus top-k semantic matches for the
   setting, plus the previous page's scene record for visual continuity. Semantic search
   alone drifts; name-keyed lookup is what actually holds characters steady.
4. **Seed discipline.** A per-book base seed plus a per-character offset keeps recurring
   subjects visually closer across pages.

## Phases

Phase 1a landed early — book acquisition was built before the text pipeline because it
supplies the input everything else is developed against. Phases are ordered by dependency,
not by the order they happened to get done.

### Phase 0 — Scaffolding — *done*
- DONE: `pyproject.toml` (hatchling, `storyillus` script entry point), `ruff` (line-length
  100), `pytest`, `uv.lock`, package skeleton.
- DONE: `models.py` — `Book`, `Chapter`, `Page`, `ScenePlan`, `MemoryRecord`, `PageResult`.
  `Book` moved here out of `ingest/gutenberg.py`: ingest creates it, but the CLI, renderer,
  and web API all consume it, so no single stage should own it.
- DONE: `config.py` — `LLMConfig` / `ImageConfig` / `Config`, YAML loading, token
  resolution. Ships `configs/hosted.yaml` and `configs/local.yaml`.
- DONE: `LLMBackend` and `ImageBackend` protocols plus the `FakeLLM` / `FakeImage` pair, so
  the suite runs with no models installed.
- Credentials read from `.env` only (already gitignored) — never from YAML, never committed.
  Config carries `base_url` and `model_id`; the token is looked up by variable name.
- **The HF token variable is `hf_token`, lowercase.** Env vars are case-sensitive, and
  `huggingface_hub` only auto-discovers the uppercase `HF_TOKEN`, so the token is read
  explicitly and passed to the client rather than picked up implicitly. One place does this
  — `config.py` — and every backend receives it as an argument. A test asserts that setting
  `HF_TOKEN` instead of `hf_token` does *not* satisfy the config.
- Missing token fails at config load with a clear message naming the variable, not deep in
  the first API call.
- **`complete_json` is a function over a backend, not a protocol method.** The plan first
  listed it as the second method on `LLMBackend`. Parse-and-retry is identical for every
  backend, so making it a method would mean every implementation reimplementing or
  inheriting it. The protocol is `complete()` alone; `complete_json(llm, prompt)` wraps any
  backend. It raises `LLMJSONError` on give-up rather than inventing a value — the
  summary-only fallback is `condense.py`'s policy to make, not the parser's.
- **429/503 backoff moved to Phase 1b.** It belongs in the HTTP client, where a status code
  exists; a `FakeLLM` has none. Nothing in Phase 0 can exercise it.
- **`Qwen-Image-2.0` has no weights on the Hub.** `models.md` names it the primary image
  pick, but `huggingface.co/api/models/Qwen/Qwen-Image-2.0` 404s and the `Qwen` org lists
  only `Qwen-Image`, `-2512`, `-Edit`, `-Edit-2509`, `-Edit-2511`. If it is API-only it
  fails the open-weights constraint outright. `configs/hosted.yaml` therefore pins
  `Qwen/Qwen-Image-Edit-2511` (verified, Apache 2.0), which `models.md` already wants for
  every panel after the first. Revisit in Phase 3.
- **Done when:** ✅ `env -u hf_token uv run pytest -q` passes — 59 tests, 0.10s, no weights,
  no tokens, no network. `uv run ruff check .` clean. Both shipped configs load, and
  `configs/hosted.yaml` resolves a real token out of `.env`.

### Phase 1a — Book acquisition — *done, minus chapter splitting*
The user names a book, it is found, downloaded, and written as markdown. This is what makes
"what can we illustrate?" answerable — the catalog is Gutenberg's ~75k public-domain texts
rather than a folder the user has to fill themselves.

- DONE: `ingest/gutenberg.py` — catalog search, `fetch_text`, `strip_boilerplate`,
  `to_markdown`, `download`. CLI: `uv run storyillus fetch "<title or author>"`, with
  `--first` for non-interactive use. 6 offline tests cover parsing and formatting.
- **Search runs against Gutenberg's official `pg_catalog.csv`** (~21 MB, downloaded once to
  `.cache/`), not the gutendex.com JSON API. The CSV was chosen after gutendex returned 403
  to non-browser clients. Two further reasons keep it the right call even when gutendex is
  reachable: gutendex answered a single search in **19.3 seconds** in testing, and a local
  CSV means search is offline, instant, and un-rate-limited after first use.
- Boilerplate markers are real and stable — verified `*** START OF THE PROJECT GUTENBERG
  EBOOK … ***` / `*** END … ***` in the live text of book 84. Both the modern `THE` and
  archaic `THIS` wordings are matched.
- Only `Type=Text` rows are searched, so audiobook entries don't shadow the text editions.
- DONE: `ingest/chapters.py` — `split_chapters`, `parse_frontmatter`, `parse_selection`,
  `select`. CLI: `uv run storyillus chapters <book.md> [--select 1-3]`.
- **A heading with no body is not a chapter.** Gutenberg prints its table of contents as one
  chapter name per line, so the download step promotes all of them to `## ` headings —
  Frankenstein yielded 56 chapters, 28 phantoms ahead of the 28 real ones. Requiring body
  text separates them without guessing where the contents block ends, and also discards bare
  section dividers ("Part 1" immediately followed by "Chapter 1"). This was found by running
  the command on a real book, not by the unit tests; both cases are now covered.
- Selection defaults to **one chapter**, so an unqualified run costs minutes, not hours.
  Out-of-range and malformed specs fail with the valid range named.
- **Known limitation:** chapter detection is a regex over line-start `Chapter|Part|Book|Act|
  Canto|Letter` + numeral. It works on conventionally formatted books (verified against
  Frankenstein's `Letter 1`…`Chapter 1` structure) and will miss books with named or unnumbered
  chapters. Those degrade to a single `(whole text)` chapter rather than failing. The fix, if it
  becomes a real problem, is to fetch the `text/html` edition instead and read its `<h1>`/`<h2>`
  tags — Gutenberg's HTML carries editorial structure the plain text has thrown away. Not worth
  doing until a book we care about actually breaks.
- **Done when:** ✅ `uv run storyillus fetch "frankenstein"` writes a clean
  `books/frankenstein-or-the-modern-prometheus.md` (75,128 words), and
  `storyillus chapters` lists its 28 chapters from `Letter 1` (1,198 words) to
  `Chapter 24` (8,237 words).

### Phase 1b — Text pipeline (no images)
- Loader + paginator, `condense.py` producing valid `ScenePlan`s against a real hosted model
  via the HF router (an instruction-tuned Qwen is the starting pick — big enough that JSON
  adherence is the pipeline's problem to get right, not the model's).
- CLI subcommand `uv run storyillus plan story.txt` dumping the plans as JSON.
- **Done when:** a 10-page story yields 10 well-formed `ScenePlan`s.

### Phase 2 — Memory layer
- Embedder, `VectorStore` protocol, Chroma implementation only, record schema.
- `update_memory.py` entity extraction + `retrieve.py` scoped retrieval.
- **Done when:** running Phase 1b over a story populates a store where querying a
  character's name returns that character's sheet as the top hit.

### Phase 3 — Image generation
- Diffusers backend (SD 1.5 first — fastest to iterate), prompt builder that fuses style
  block + retrieved context + `key_visual`, negative prompt defaults.
- **Done when:** the full five-step loop renders every page of the example story.

### Phase 4 — Assembly
- HTML book renderer (text page + facing illustration), PDF export.
- Run manifest JSON: model IDs, seeds, prompts, timings per page.
- `--pages 3,7` flag to re-render individual pages against an existing run.
- **Done when:** one command turns `examples/short_story.txt` into a viewable PDF.

### Phase 5 — Quality & scale
- SDXL config, optional IP-Adapter / reference-image conditioning for stronger character
  identity, batch and resume support for long stories.
- Optional whole-book mode: an LLM pass that picks the N most illustration-worthy scenes
  across a full novel, instead of illustrating a chapter exhaustively.
- Consistency evaluation (below).

### Phase 6 — Web frontend
Last, deliberately. Until Phase 4 there is no finished artifact to display, and a UI built
over a data model that is still moving gets written twice. Everything here wraps functions
that already exist and are exercised by the CLI.

- `api/server.py` — HTTP over the existing pipeline: `GET /search?q=`, `POST /fetch`,
  `GET /books`, `GET /books/{id}/chapters`, `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/book`.
- `api/jobs.py` — a run registry on disk. Illustration takes minutes to hours, so a request
  submits a job and returns a run id immediately; the UI polls. No request ever blocks on a
  diffusion loop.
- `web/static/` — one page: search box → results list → chapter picker → style choice →
  progress view where illustrations appear as they finish → the Phase 4 book, reusing the
  same HTML renderer rather than a second one.
- Server binds to localhost and is started by `uv run storyillus serve`.
- **Done when:** a user types "frankenstein" into a browser, picks a chapter, and watches an
  illustrated chapter assemble without touching a terminal.

## Evaluation

Subjective output needs at least some measurable signal:

- **Character consistency:** CLIP image embeddings of the cropped main character across
  pages; report mean pairwise cosine similarity. Track it as a number across config changes.
- **Prompt faithfulness:** CLIP similarity between each image and its own image prompt.
- **Style consistency:** embedding variance across all pages of a book — lower is better.
- **Human spot-check:** a 5-page rubric (character recognizable / setting correct / style
  matches / scene matches text) filled in per config change.

## Risks

| Risk | Mitigation |
|---|---|
| Character drift despite RAG | Name-keyed retrieval + fixed sheets; escalate to IP-Adapter in Phase 5 |
| Local image gen is slow | SD 1.5 default, batch overnight runs, cache by prompt+seed hash |
| LLM returns malformed JSON | Schema in prompt, retry wrapper, summary-only fallback |
| Hosted model ID deprecated by provider | Pin exact `org/model` in config and in the run manifest; a 404 fails loudly rather than silently substituting a different model |
| Rate limits / cold starts on hosted inference | Backoff retry on 429/503; `FakeLLM` keeps the test suite network-free |
| Vector store lock-in | `VectorStore` protocol keeps Chroma at arm's length; write the second implementation the day a second store is actually needed, not before |
| Memory poisoning by bad extraction | Write-once sheets; amendments require explicit textual evidence |
| A novel is 100× the size of the test story | Chapter is the unit of illustration, not the book; `--chapter` selection required before any full-length text enters the pipeline |
| Gutenberg blocks the client for bulk fetching | Catalog cached after first download, books cached by id, descriptive `User-Agent` set, one request per book — never crawl |
| Chapter regex misses unconventional books | Falls back to whole-text pagination; escalate to the HTML edition's heading tags if it bites |
| Gutenberg URL layout varies by book age | `fetch_text` tries `/ebooks/{id}.txt.utf-8` then `/files/{id}/{id}-0.txt`, and fails loudly naming both |

## Open Questions

- Page granularity *within* a chapter: fixed token budget, paragraph groups, or LLM-chosen
  scene breaks? (The chapter boundary itself is settled — see Scope Decisions.)
- Should the style block be user-specifiable, or always LLM-derived from the story?
- One illustration per page, or let the agent decide which pages deserve art?
- Does the memory store persist across chapters of the same book? It should — character
  sheets built in chapter 1 are exactly what chapter 2 needs — but that makes the store
  book-scoped rather than run-scoped, which Phase 2 has to account for.
- License for the repo (README lists it as TBD). The output is derived from public-domain
  texts, so nothing blocks a permissive choice.

Resolved: chapter-level scope and local single-user hosting — see Scope Decisions.

## Immediate Next Steps

1. ~~`models.py`~~ — done.
2. ~~`chapters.py` + selection~~ — done.
3. ~~Phase 0: `config.py`, the two protocols, `FakeLLM` / `FakeImage`~~ — done.
4. **`condense.py` + the `openai_compat` backend** pointed at the HF router, with 429/503
   backoff. Eyeball `ScenePlan` quality on Frankenstein chapter 5 (2,204 words — a real
   chapter, small enough to iterate on) before any image code exists. This is the first
   code that spends a token, so it is also where `configs/hosted.yaml`'s pinned model id
   gets verified against a live call.
5. **`paginate.py`** — split a selected chapter into illustration-sized pages. Deferred until
   `condense.py` shows how much text a good `ScenePlan` actually wants.

`examples/short_story.txt` is no longer needed as the primary fixture — chapter 1 of a
fetched Gutenberg book is a better one, since it's the actual input shape the product takes.
Keep a short synthetic story only if the test suite needs a tiny deterministic input.
