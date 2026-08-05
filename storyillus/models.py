"""Core data types shared by every stage of the pipeline.

These live here rather than beside the code that first creates them: `Book` is produced by
the ingest layer but consumed by the CLI, the renderer, and eventually the web API, so no
single stage owns it.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RecordKind = Literal["character", "setting", "style", "scene"]


@dataclass(frozen=True)
class Book:
    """A catalog entry, before any text has been downloaded."""

    id: int
    title: str
    author: str
    language: str

    @property
    def slug(self) -> str:
        text = re.sub(r"[^a-z0-9]+", "-", self.title.lower())
        return text.strip("-")[:60] or f"pg{self.id}"

    def __str__(self) -> str:
        return f"{self.title} - {self.author} (#{self.id}, {self.language})"


@dataclass(frozen=True)
class Chapter:
    """One `## ` section of a fetched book — the unit a run illustrates.

    `index` is 1-based so it matches what the user types in `--chapter 3`.
    """

    index: int
    heading: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __str__(self) -> str:
        return f"{self.index:3}. {self.heading}  ({self.word_count:,} words)"


@dataclass
class Page:
    """One unit of story text, sized for a single illustration."""

    index: int
    text: str


@dataclass
class ScenePlan:
    """The LLM's structured read of a page."""

    summary: str
    characters: list[str]
    setting: str
    mood: str
    key_visual: str


@dataclass
class MemoryRecord:
    """What lives in the vector store."""

    kind: RecordKind
    name: str
    description: str
    first_seen_page: int
    embedding_text: str


@dataclass
class PageResult:
    """Everything a run knows about one finished page, enough to reproduce it."""

    page: Page
    plan: ScenePlan
    image_prompt: str
    negative_prompt: str
    seed: int
    retrieved: list[MemoryRecord] = field(default_factory=list)
    image_path: Path | None = None
    error: str | None = None
