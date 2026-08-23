"""Local embeddings via sentence-transformers.

Weights download once from the Hub on first use — the model is public and ungated, so no
token is needed — then every call runs fully offline on this machine.
"""

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Wraps one local sentence-transformers model."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_id)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()
