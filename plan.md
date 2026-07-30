# Implementation Plan

Build plan for the multi-modal story-to-illustration agent described in [README.md](README.md).
The goal is a fully local pipeline: story text in, illustrated storybook out, with a RAG
memory that keeps characters, settings, and art style consistent across pages.

## Guiding Constraints

- **Open-source only.** No proprietary APIs anywhere in the pipeline. Anything that
  can't run locally doesn't go in.
- **Swappable backends.** LLM, image model, embedder, and vector store each sit behind a
  small interface so a laptop (Ollama + SD 1.5) and a workstation (vLLM + SDXL) run the
  same code.
- **Deterministic where possible.** Seeds, model IDs, and prompts are recorded per page so
  a run can be reproduced or a single page re-rendered without redoing the book.
- **Fail soft.** A failed page produces a placeholder and a logged error, not a dead run.

## Proposed Layout

```
storyillus/
  __init__.py
  config.py            # dataclass config + YAML loading, model/backend selection
  models.py            # Page, ScenePlan, MemoryRecord, PageResult dataclasses
  ingest/
    loader.py          # .txt / .md / .epub -> raw text
    paginate.py        # split into story pages (chapter, blank-line, or token budget)
  llm/
    base.py            # LLMBackend protocol: complete(), complete_json()
    ollama.py
    vllm.py            # OpenAI-compatible HTTP client pointed at a local vLLM server
  imagegen/
    base.py            # ImageBackend protocol: generate(prompt, negative, seed) -> PIL
    diffusers_local.py # SD 1.5 / SDXL via diffusers
  memory/
    store.py           # VectorStore protocol: add(), query(), persist()
    chroma_store.py
    faiss_store.py
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
  cli.py               # `storyillus run story.txt --config configs/local.yaml`
configs/
  local.yaml           # small models, CPU/MPS friendly
  workstation.yaml     # SDXL + larger LLM
tests/
  ...
examples/
  short_story.txt
```

## Core Data Model

```python
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

### Phase 0 — Scaffolding
- `pyproject.toml` (uv or pip), package skeleton, `ruff` + `pytest`, config loading.
- `LLMBackend` and `ImageBackend` protocols plus a `FakeLLM` / `FakeImage` pair used by
  every test so the suite runs with no models installed.
- **Done when:** `pytest` passes on a machine with zero model weights.

### Phase 1 — Text pipeline (no images)
- Loader + paginator, `condense.py` producing valid `ScenePlan`s against a real Ollama model.
- CLI subcommand `storyillus plan story.txt` dumping the plans as JSON.
- **Done when:** a 10-page story yields 10 well-formed `ScenePlan`s.

### Phase 2 — Memory layer
- Embedder, `VectorStore` protocol, Chroma implementation, record schema.
- `update_memory.py` entity extraction + `retrieve.py` scoped retrieval.
- **Done when:** running Phase 1 over a story populates a store where querying a
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
- Consistency evaluation (below).

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
| Vector store lock-in | `VectorStore` protocol with two implementations from the start |
| Memory poisoning by bad extraction | Write-once sheets; amendments require explicit textual evidence |

## Open Questions

- Page granularity: fixed token budget, paragraph groups, or LLM-chosen scene breaks?
- Should the style block be user-specifiable, or always LLM-derived from the story?
- One illustration per page, or let the agent decide which pages deserve art?
- License for the repo (README lists it as TBD).

## Immediate Next Steps

1. Pick the packaging tool and land Phase 0 scaffolding.
2. Write `examples/short_story.txt` (~10 pages, 2-3 recurring characters) as the fixture
   everything is developed against.
3. Implement `condense.py` + the Ollama backend and eyeball the `ScenePlan` quality before
   any image code exists.
