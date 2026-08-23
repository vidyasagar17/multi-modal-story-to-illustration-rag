"""A scripted embedder, so the test suite needs no weights, no token, and no network."""

import hashlib


class FakeEmbedder:
    """Derives a short vector from a hash of the text — same text, same vector, forever."""

    dim = 8

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()[: self.dim]
        return [byte / 255 for byte in digest]
