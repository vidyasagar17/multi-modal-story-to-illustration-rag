"""The retrieval seam: a protocol plus the one implementation this project needs today.

A book's memory is tens of records — a handful of characters, settings, and per-page scene
notes — not the millions of vectors an ANN index earns its keep on. A linear scan ranked by
cosine similarity is exactly as correct as a real vector database at this scale, for a
fraction of the machinery.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from storyillus.models import MemoryRecord, RecordKind


class VectorStore(Protocol):
    """Implementations are one process's worth of memory for one book."""

    def add(self, record: MemoryRecord, embedding: list[float]) -> None: ...
    def get_by_name(self, name: str) -> MemoryRecord | None: ...
    def query(
        self, embedding: list[float], *, kind: RecordKind | None = None, top_k: int = 3
    ) -> list[MemoryRecord]: ...
    def persist(self, path: Path) -> None: ...


class LocalStore:
    """Every record kept in memory as `(record, embedding)`, persisted as one JSON file."""

    def __init__(self) -> None:
        self._records: list[tuple[MemoryRecord, list[float]]] = []

    def add(self, record: MemoryRecord, embedding: list[float]) -> None:
        self._records.append((record, embedding))

    def get_by_name(self, name: str) -> MemoryRecord | None:
        for record, _ in self._records:
            if record.name.casefold() == name.casefold():
                return record
        return None

    def query(
        self, embedding: list[float], *, kind: RecordKind | None = None, top_k: int = 3
    ) -> list[MemoryRecord]:
        candidates = [pair for pair in self._records if kind is None or pair[0].kind == kind]
        ranked = sorted(candidates, key=lambda pair: _cosine(embedding, pair[1]), reverse=True)
        return [record for record, _ in ranked[:top_k]]

    def persist(self, path: Path) -> None:
        data = [{"record": asdict(record), "embedding": embedding} for record, embedding in self._records]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalStore":
        """An empty store if `path` doesn't exist yet — the first run for a book."""
        store = cls()
        if path.exists():
            for item in json.loads(path.read_text(encoding="utf-8")):
                store.add(MemoryRecord(**item["record"]), item["embedding"])
        return store


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
