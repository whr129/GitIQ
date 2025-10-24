from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import List, Protocol, Sequence


class EmbeddingBackend(Protocol):
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


@dataclass
class SentenceTransformerBackend:
    """Embedding backend powered by sentence-transformers."""

    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL

    def __post_init__(self) -> None:
        sentence_transformers = importlib.import_module("sentence_transformers")
        self._model = sentence_transformers.SentenceTransformer(self.model_name, device="cpu")

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(list(texts), convert_to_numpy=False, normalize_embeddings=True)
        return [list(vec) for vec in vectors]


@dataclass
class OpenAIBackend:
    """Embedding backend using OpenAI's API."""

    model_name: str = DEFAULT_OPENAI_MODEL

    def __post_init__(self) -> None:
        openai = importlib.import_module("openai")
        self._client = openai.OpenAI()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=self.model_name, input=list(texts))
        return [item.embedding for item in response.data]


def ensure_backend(name: str, *, model_name: str | None = None) -> EmbeddingBackend:
    """Factory helper that resolves a backend from a string identifier."""

    if name == "sentence-transformer":
        return SentenceTransformerBackend(model_name=model_name or DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    if name == "openai":
        return OpenAIBackend(model_name=model_name or DEFAULT_OPENAI_MODEL)
    raise ValueError(f"Unknown embedding backend '{name}'.")
