"""RAG starter toolkit for source repositories."""

from .config import RepoRagConfig, load_config
from .documents import Document
from .generator import GenerationConfig, generate_answer
from .embeddings import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    EmbeddingBackend,
    ensure_backend,
)
from .pipeline import build_index, load_index, query_index

__all__ = [
    "Document",
    "EmbeddingBackend",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
    "ensure_backend",
    "GenerationConfig",
    "generate_answer",
    "build_index",
    "load_index",
    "query_index",
    "RepoRagConfig",
    "load_config",
]
