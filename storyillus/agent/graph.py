"""Orchestration: wires retrieve -> prompt_builder -> illustrate -> update_memory for one page.

Condense (step 1) and name canonicalization already happened upstream (the `plan` and
`canonicalize` CLI commands) — canonicalization needs every page's plan at once to resolve
aliases, which doesn't fit a per-page loop. This wires the remaining four steps, threading the
previous page's scene record forward for continuity.

Two different fail-soft policies meet here, deliberately different:
- The image call is wrapped narrowly (nothing else) — any failure becomes a `PageResult` with
  `image_path=None` and `error` set. plan.md's guiding constraint, literally: "a failed page
  produces a placeholder and a logged error, not a dead run."
- `update_memory()` is *not* wrapped: its `APIStatusError` (a depleted LLM quota — the same
  failure mode `_plan_pages`/`_canonicalize_document`/`_remember_document` already handle)
  propagates to the caller, which stops the whole run there. A systemic quota problem isn't a
  page-local rendering hiccup. `update_memory` still runs even when the image itself failed —
  a character's sheet shouldn't be held hostage by one bad render.

Reference-image conditioning (Phase 5): `context` (already fetched for the prompt) and
`previous_scene` may carry a `reference_image` — an actual earlier render, not just a text
description. `_collect_references()` gathers whichever exist (primary character, then the
previous panel, then a setting) before the image call, so a recurring character's real
likeness carries forward instead of being re-imagined from words each time. No reference
images yet (the very first page of a run) means plain text-to-image, unchanged from Phase 3.
"""

import time
from pathlib import Path

from storyillus.agent.illustrate import illustrate, page_seed
from storyillus.agent.prompt_builder import build_prompt
from storyillus.agent.retrieve import retrieve
from storyillus.agent.update_memory import update_memory
from storyillus.imagegen.base import ImageBackend
from storyillus.llm.base import LLMBackend
from storyillus.memory.embedder import Embedder
from storyillus.memory.store import VectorStore
from storyillus.models import MemoryRecord, Page, PageResult, ScenePlan


def illustrate_page(
    llm: LLMBackend,
    image_backend: ImageBackend,
    store: VectorStore,
    embedder: Embedder,
    plan: ScenePlan,
    *,
    chapter_index: int,
    page_index: int,
    style_block: str,
    base_seed: int,
    previous_scene: MemoryRecord | None,
    out_dir: Path,
) -> tuple[PageResult, MemoryRecord | None]:
    """Returns this page's result, plus its scene record for the caller to pass as the next
    call's `previous_scene` — no `get_by_name` lookup involved."""
    started = time.monotonic()
    context = retrieve(store, embedder, plan, previous_scene=previous_scene)
    prompt, negative = build_prompt(style_block, context, plan)
    seed = page_seed(base_seed, plan.characters)
    references = _collect_references(context, previous_scene)

    image_path, error = None, None
    try:
        image = illustrate(image_backend, prompt, negative, seed, references)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Chapter-qualified: `page_index` resets to 1 in every chapter, so the bare page
        # number alone collided across chapters and silently overwrote earlier renders —
        # caught by a live run over a real multi-chapter document, not by the unit tests.
        image_path = out_dir / f"ch{chapter_index:03d}-page{page_index:03d}.png"
        image.save(image_path)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: plan.md's guiding constraint
        error = str(exc)  # is "a failed page produces a placeholder," not a narrower contract

    written = update_memory(llm, store, embedder, plan, page_index=page_index, image_path=image_path)
    scene = next(record for record in written if record.kind == "scene")

    result = PageResult(
        page=Page(index=page_index, text=""),
        plan=plan,
        image_prompt=prompt,
        negative_prompt=negative,
        seed=seed,
        retrieved=context,
        image_path=image_path,
        error=error,
        duration_s=time.monotonic() - started,
    )
    return result, scene


def _collect_references(
    context: list[MemoryRecord], previous_scene: MemoryRecord | None
) -> list[Path]:
    """Primary character's reference image, then the previous panel, then a setting's — the
    same order an illustrator would reach for them. Deduplicated, capped at 3 (what
    Qwen-Image-Edit-2511 accepts)."""
    ordered = [record for record in context if record.kind == "character"]
    if previous_scene is not None:
        ordered.append(previous_scene)
    ordered += [record for record in context if record.kind == "setting"]

    seen: set[str] = set()
    references: list[Path] = []
    for record in ordered:
        if record.reference_image and record.reference_image not in seen:
            seen.add(record.reference_image)
            references.append(Path(record.reference_image))
    return references[:3]
