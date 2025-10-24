from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    import tomli as tomllib  # type: ignore[assignment]


@dataclass
class EmbeddingConfig:
    backend: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class IndexingConfig:
    repo: Optional[Path] = None
    output: Optional[Path] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    extensions: Optional[List[str]] = None
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class QueryConfig:
    index: Optional[Path] = None
    top_k: Optional[int] = None
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class RepoRagConfig:
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    query: QueryConfig = field(default_factory=QueryConfig)


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(path: Path) -> RepoRagConfig:
    """Parse a TOML configuration file into structured settings."""

    data = _load_toml(path)

    indexing_data = data.get("indexing", {})
    query_data = data.get("query", {})

    indexing_embedding = indexing_data.get("embedding", data.get("embedding", {}))
    query_embedding = query_data.get("embedding", data.get("embedding", {}))

    indexing_config = IndexingConfig(
        repo=_coerce_path(indexing_data.get("repo")),
        output=_coerce_path(indexing_data.get("output")),
        chunk_size=_coerce_int(indexing_data.get("chunk_size")),
        chunk_overlap=_coerce_int(indexing_data.get("chunk_overlap")),
        extensions=_coerce_list(indexing_data.get("extensions")),
        embedding=_coerce_embedding(indexing_embedding),
    )

    query_config = QueryConfig(
        index=_coerce_path(query_data.get("index")),
        top_k=_coerce_int(query_data.get("top_k")),
        embedding=_coerce_embedding(query_embedding),
    )

    return RepoRagConfig(indexing=indexing_config, query=query_config)


def _coerce_path(value: Any) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(str(value))


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:  # pragma: no cover - defensive guard
        raise ValueError(f"Expected an integer value, got {value!r}") from error


def _coerce_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(f"Expected a list value, got {value!r}")


def _coerce_embedding(value: Any) -> EmbeddingConfig:
    if not isinstance(value, dict):
        return EmbeddingConfig()
    backend = value.get("backend")
    model_name = value.get("model_name")
    return EmbeddingConfig(
        backend=str(backend) if backend is not None else None,
        model_name=str(model_name) if model_name is not None else None,
    )
