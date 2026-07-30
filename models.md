# Model Selection — Image Generation for Comic Strips

Decision record for the image-generation half of the pipeline. Companion to
[plan.md](plan.md). Researched July 2026.

---

## Verdict

**Primary: the Qwen-Image family — `Qwen-Image-2.0` for generation, `Qwen-Image-Edit-2511`
for every panel after the first.**

Not one model. Comics need two jobs done, and they are different jobs:

| Job | Model | Why |
|---|---|---|
| Establishing panel + character sheets | `Qwen-Image-2.0` | Best-in-class in-image text, native 2K, Apache 2.0 |
| Every subsequent panel | `Qwen-Image-Edit-2511` | Takes up to 3 reference images → character stays locked |

**Local draft tier: `Z-Image Turbo` (6B, Apache 2.0)** — the only genuinely good model that
fits your 8 GB M3.

---

## What Comic Strips Actually Demand

Ranked by how badly failure hurts. This ordering is why the verdict lands where it does:

1. **Character consistency across panels.** A storybook can survive slight drift between
   pages. A comic cannot — panels sit side by side on one page, and readers compare them
   directly. This is the hardest requirement and the one that eliminates most models.
2. **Legible in-image text.** Speech bubbles, captions, sound effects. This is the
   capability that separates "comic" from "sequence of illustrations," and most image
   models are genuinely bad at it.
3. **Style lock.** Consistent line weight, palette, shading, and inking across the strip.
4. **Panel-appropriate composition.** Varied shot framing (wide / medium / close) that
   reads as sequential art rather than a series of portraits.
5. **Cost and throughput.** A 20-panel strip re-rendered 30 times during prompt tuning is
   600 images.

Note that requirements 1 and 2 are the *comic-specific* ones. Generic "which model makes
the prettiest picture" benchmarks weight neither, which is why arena rankings are a poor
guide for this project.

---

## The Field

| Model | Params | License | In-image text | Char. consistency | Fits 8 GB? |
|---|---|---|---|---|---|
| **Qwen-Image-2.0** | ~20B class, lighter arch | **Apache 2.0** | **Best open** | via Edit variant | No |
| **Qwen-Image-Edit-2511** | 20B | **Apache 2.0** | Strong | **Best open (3 refs)** | No |
| Qwen-Image-2512 | 20B MMDiT + 8.3B VL encoder | Apache 2.0 | Excellent | via Edit | Q2_K only, poor |
| FLUX.2 [dev] | 32B | **Non-commercial** | Good, degrades >30 chars | Good | No (~19 GB Q4) |
| FLUX.2 [klein] 9B | 9B + 8B Qwen3 encoder | **Non-commercial** | Good | **Excellent** | No |
| FLUX.2 [klein] 4B | 4B | **Apache 2.0** | Fair | Good | Marginal |
| **Z-Image Turbo** | 6B S³-DiT | **Apache 2.0** | Good (EN/CN) | Weak — no ref variant | **Yes** |
| Illustrious XL / NoobAI XL | SDXL 2.6B | Open (SDXL lineage) | Poor | LoRA-dependent | **Yes** |
| SDXL + IP-Adapter | 2.6B | Open | Poor | Moderate | **Yes** |

Timeline for orientation: FLUX.2 [dev] shipped 25 Nov 2025, Qwen-Image-2512 on 31 Dec
2025, Qwen-Image-2.0 on 10 Feb 2026. Qwen-Image-2.0 is the current head of that line —
lighter architecture, faster inference, and typography instructions up to 1k tokens.

---

## Why Qwen Wins This Specific Problem

**1. It is the only top-tier family that is actually Apache 2.0.**

This is not a footnote for your project — your README's mission is open models. Look at
what the alternatives require you to accept: FLUX.2 [dev] is non-commercial. FLUX.2
[klein] **9B** is non-commercial; only the weaker 4B is Apache. If your agent is ever
demoed publicly, taught from, or built on by someone else, the FLUX line puts a license
question on every output. Qwen does not. You can train on it, fine-tune it, sell what you
build, and distribute derivatives.

**2. It is decisively the best open model at text in images — the #1 comic-specific need.**

Independent benchmarking consistently puts Qwen ahead here. FLUX renders short strings
with good kerning but accuracy falls off after roughly 30 characters, which is
*exactly* the length of a normal line of comic dialogue. Qwen was explicitly trained for
complex text rendering and handles multi-element layout and non-Latin scripts. Qwen-Image-2.0
pushes further into professional typography.

If you want dialogue rendered *into* the panel rather than composited as an overlay
afterward, this alone decides the question. (See "Open Question" below — you may not
want that, and it changes the weighting.)

**3. `Edit-2511` attacks panel-to-panel consistency directly.**

It accepts **up to three reference images simultaneously** and fuses them into one
coherent composition while preserving identity. That maps onto your comic pipeline almost
too neatly:

```
ref 1 = character sheet (who)
ref 2 = setting plate    (where)
ref 3 = previous panel   (visual continuity)
→ new panel, same character, same world
```

This is a far stronger mechanism than the text-only RAG retrieval in plan.md. Your vector
store stops being just a text-context provider and starts storing **reference image paths**
alongside descriptions. That is a real upgrade to the architecture, and it's the single
biggest change this research implies.

**4. The customization path is open and cheap.** Covered in detail below — Qwen LoRA
training runs about $1–2 per character on fal.

---

## The Honest Counter-Case

I don't want to oversell this. Three real arguments against the pick:

**FLUX.2 [klein] 9B beats Qwen at image editing.** It ranks as the top open-weights image
editing model — reportedly surpassing even BFL's own 32B FLUX.2 [dev] — while being
step-distilled to 4 inference steps and running in under half a second. It is faster,
smaller, and arguably better at the consistency job. **If you decide the non-commercial
license is acceptable for a course project, klein 9B is a legitimate primary choice** and I
would not argue hard against it. The license is the reason I don't lead with it.

**Z-Image Turbo ranks #1 open on the Artificial Analysis Image Arena** — above FLUX.2
[dev], HunyuanImage 3.0, and Qwen-Image — at 6B and Apache 2.0. But arena rank measures
single-image aesthetic appeal, not the two things comics need. It has no multi-reference
editing variant, so panel-to-panel consistency would fall back entirely to LoRAs and seeds.
Excellent model; wrong shape for this job.

**Qwen is the expensive option.** `Edit-2511` runs about **$0.037/image** on fal versus
FLUX schnell at $0.0006 — roughly 60× more. A 20-panel strip costs ~$0.74 per full render
instead of ~$0.012. Still cheap in absolute terms, but it changes how casually you can
iterate. Mitigation is the two-tier setup below.

---

## Two-Tier Architecture

Your 8 GB M3 cannot run Qwen locally (20B needs ~14 GB even at Q4_K_M; Q2_K technically
fits 8 GB but quality collapses). So:

**Tier 1 — Local draft (`Z-Image Turbo`, 6B, Apache 2.0)**
Runs on your machine — 16 GB is the comfortable figure but 8 GB successes are reported.
Eight inference steps. Use it to check composition, framing, and prompt sanity for free
while iterating. Expect a few seconds per image on a 4090; slower on M3, benchmark it.

**Tier 2 — Cloud final (`Qwen-Image-2.0` + `Edit-2511` on fal)**
Character-locked, text-rendered, publication-quality panels. Only run when the draft looks
right.

This maps cleanly onto the `ImageBackend` protocol already in plan.md — two implementations,
one config flag, `--draft` / `--final`. No architectural change needed.

---

## Weight Recommendations

You asked specifically about changing weights. Three distinct levers, in descending order
of impact on comic quality:

### 1. Character LoRA — the biggest lever (train this)

Reference images get you most of the way; a trained LoRA gets you the rest. For any
character appearing in more than ~10 panels, train one.

| Parameter | Value |
|---|---|
| Dataset | 20–25 images (community sweet spot; 30 clean beats 150 messy) |
| Steps | 1500–2500 |
| Cost on fal | $0.00095/step → **~$1.40–2.40 per character** |
| Local training | Possible from 6 GB VRAM — not your M3, but worth knowing |
| Tooling | `ai-toolkit`, or fal's `qwen-image-edit-2511-trainer` |

The bootstrap problem: you need 20 images of a character that doesn't exist yet. Solution —
generate a character sheet with `Qwen-Image-2.0` (same character, varied angles/poses/
expressions), hand-cull to the best 20–25, train on those. This is a one-time cost per
book, and it is what makes a *comic* rather than a slideshow.

### 2. Style LoRA — pick one, apply to every panel

This makes plan.md's abstract "style block" concrete. Rather than hoping a text description
holds the look steady, a style LoRA enforces it at the weights level.

Starting points to evaluate:
- **RealComic (Qwen)** — comic-native, trained for the target base
- **American Cartoon Style (Qwen)** — cleaner cartoon register
- **Retro Comic FLUX / PULPKHOR (klein 9B)** — pulp/vintage with halftone dots, distressed
  paper, print artifacts; genuine period texture rather than a filter overlay
- **Western Comic Pony (SDXL)** — semi-realistic 2.5D, if you go the SDXL route

Trigger vocabulary worth knowing when prompting: *cel shading, inked lines, posterized,
halftone texture*.

### 3. Lightning LoRA — cost and speed

A 4-step distillation for `Edit-2511`, reportedly reaching 1920×1080 in seconds. Use it for
the draft pass; drop it for finals. Meaningfully cuts the per-image cost problem.

### If you want manga/anime instead of Western comic

Change the whole stack, don't fight Qwen with LoRAs. **Illustrious XL** or **NoobAI XL**
(SDXL-derived, ~2.6 GB) run comfortably on your M3 and sit on the largest stylized LoRA
library that exists. Pony still has the biggest raw LoRA count; Illustrious has ~30% of it
but cleaner line work, better anatomy, and better named-character accuracy. NoobAI adds a
more webtoon-leaning aesthetic. Their weakness is in-image text — bad enough that you'd
composite speech bubbles programmatically.

That's a real fork in the road, and it depends on a question only you can answer.

---

## Panel Layout — Recommendation

Generate **one image per panel**, then composite the page (gutters, borders, bubbles) in
code with PIL.

Page-first generation exists and integrated scene composition has real advantages, but
per-panel gives you what this project needs: individually re-rollable panels, per-panel
reference conditioning for consistency, and deterministic gutters. Regenerating panel 4
should not disturb panels 1–3. Layout is a solved 2D problem — don't spend model capacity
on it.

---

## Cost Model — 20-Panel Strip

| Pass | Setup | Cost |
|---|---|---|
| Draft iteration | Z-Image Turbo, local | **$0** |
| Final render | Qwen Edit-2511 @ $0.037 | **~$0.74** |
| Character LoRAs (one-time, 2 chars) | fal, 2000 steps each | **~$3–5** |
| **First complete strip** | | **~$4–6** |
| **Each re-render after** | | **~$0.74** |

---

## Rejected, With Reasons

- **FLUX.2 [dev]** — non-commercial license, 32B, ~19 GB at Q4. Beaten by its own klein 9B
  at editing. No path to your hardware and no license upside.
- **SDXL + IP-Adapter** — runs locally, mature ecosystem, but in-image text is poor and
  consistency is markedly weaker than native multi-reference conditioning. Viable 2024
  answer; superseded.
- **SD 1.5** — plan.md's original default. Keeping it only as a smoke-test backend.
- **Nano Banana Pro / FLUX Pro / Kontext Max** — API-only, closed weights. Off-mission.
  (Note: several "best open source model" listicles include FLUX Kontext Pro/Max. They are
  not open source. Treat those rankings with suspicion.)

---

## Confidence and What to Verify

Being straight about evidence quality: model specs, licenses, and release dates here are
well-corroborated. **Pricing and quality rankings come substantially from vendor pages,
comparison blogs, and SEO-driven listicles**, several of which contradicted each other and
at least one of which miscategorized closed models as open source. Benchmark arenas measure
aesthetic preference, not comic-specific capability.

Before committing real effort, verify by hand:

1. Render the **same character in 5 panels** via `Edit-2511` with a fixed reference. This is
   the whole thesis — if consistency doesn't hold, the verdict changes.
2. Render a panel with a **20-word speech bubble** and check legibility at that length.
3. Confirm **current fal pricing** — image pricing moves fast.
4. Benchmark **Z-Image Turbo on your actual M3** before relying on the local draft tier;
   8 GB reports are anecdotal.

---

## Open Question for You

**Should dialogue be rendered by the model, or composited as vector text afterward?**

It materially changes the ranking above:

- **Model-rendered** — organic, hand-lettered feel, integrated with art. Qwen's text
  advantage becomes decisive, and the verdict above stands firmly.
- **Code-composited** — perfectly legible, editable, re-translatable, zero spelling errors.
  Text rendering drops out as a criterion, and **FLUX.2 [klein] 9B or Z-Image Turbo become
  much stronger candidates** on speed and cost.

My lean is **composited** for a first version — guaranteed legibility, and you can always
move lettering into the model later. But it's a taste call about what you want the output
to feel like, and I'd rather you make it than assume it.

---

## Sources

- [Best Open Source Models for Comics and Manga 2026 — SiliconFlow](https://www.siliconflow.com/articles/en/best-open-source-models-for-comics-and-manga)
- [Qwen Image 2512 vs SDXL vs FLUX text-in-image benchmark — WaveSpeed](https://wavespeed.ai/blog/posts/qwen-2512-vs-sdxl-flux-text-benchmark/)
- [Qwen Image 2.0 vs FLUX vs Nano Banana Pro — WaveSpeed](https://wavespeed.ai/blog/posts/blog-qwen-image-2-0-vs-flux-nano-banana-pro-comparison/)
- [Qwen-Image-Edit-2511 — Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Qwen-Image-Edit-2511 ComfyUI workflow — ComfyUI Docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511)
- [Qwen-Image-2.0 launch — AlternativeTo](https://alternativeto.net/news/2026/2/alibaba-launches-qwen-image-2-0-with-improved-typography-rendering-and-lighter-architecture)
- [Qwen Image VRAM requirements — WillItRunAI](https://willitrunai.com/image-models/qwen-image)
- [FLUX.2 [klein] — Black Forest Labs](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)
- [FLUX.2-klein-9B — Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- [FLUX.2 self-hosting: VRAM and quantization](https://tinytiny.tools/en/blog/flux-2-self-hosting)
- [Z-Image-Turbo model info — SiliconFlow](https://www.siliconflow.com/models/z-image-turbo)
- [Model Rundown: Z-Image Turbo, Qwen 2512, Flux.2 Dev — Diffusion Doodles](https://medium.com/diffusion-doodles/model-rundown-z-image-turbo-qwen-image-2512-edit-2511-flux-2-dev-fc787f5e87ad)
- [Pony vs Illustrious XL in 2026](https://aiofm.info/en/guides/pony-vs-illustrious)
- [Qwen Image LoRA training tutorial — SECourses](https://github.com/FurkanGozukara/Stable-Diffusion/wiki/Qwen-Image-Models-Training-0-to-Hero-Level-Tutorial-LoRA-and-Fine-Tuning-Base-and-Edit-Model)
- [Qwen Image Edit 2511 Trainer — fal.ai](https://fal.ai/models/fal-ai/qwen-image-edit-2511-trainer)
- [Qwen Image Edit 2511 API — fal.ai](https://fal.ai/models/fal-ai/qwen-image-edit-2511)
- [RealComic LoRA (Qwen) — Civitai](https://civitai.com/models/1757495/realcomic)
- [American Cartoon Style (Qwen) — Civitai](https://civitai.com/models/1679852/american-cartoon-style)
- [Retro comic PULPKHOR (FLUX.2-klein-9B)](https://civarchive.com/models/2413450?modelVersionId=2713511)
