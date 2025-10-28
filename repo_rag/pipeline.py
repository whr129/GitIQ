from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from .chunker import chunk_documents
from .documents import Document
from .embeddings import EmbeddingBackend
from .loader import DEFAULT_EXTENSIONS, iter_repository_documents
from .vectorstore import VectorStore, create_persistent_vector_store, load_persistent_vector_store


def _safe_relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _collect_hierarchy(root: Path, include_extensions: Sequence[str]) -> dict:
    include_set = {ext.lower() for ext in include_extensions}

    def walk(current: Path) -> dict:
        node = {
            "type": "directory",
            "name": current.name if current != root else current.name or str(current),
            "path": "." if current == root else _safe_relative_path(root, current),
            "children": [],
        }
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return node

        for entry in entries:
            if entry.is_dir():
                child = walk(entry)
                if child["children"]:
                    node["children"].append(child)
            elif entry.is_file():
                suffix = entry.suffix.lower()
                if suffix and suffix in include_set:
                    node["children"].append(
                        {
                            "type": "file",
                            "name": entry.name,
                            "path": _safe_relative_path(root, entry),
                            "suffix": suffix,
                        }
                    )
        return node

    return walk(root)


def _documents_to_json(documents: Sequence[Document]) -> list[dict]:
    records: list[dict] = []
    for doc in documents:
        metadata = doc.metadata or {}
        records.append(
            {
                "path": doc.path,
                "size_bytes": int(metadata["size"]) if "size" in metadata else None,
                "line_count": int(metadata["line_count"]) if "line_count" in metadata else None,
            }
        )
    return records


def _chunks_to_json(chunks: Sequence[Document]) -> list[dict]:
    records: list[dict] = []
    for chunk in chunks:
        metadata = chunk.metadata or {}
        records.append(
            {
                "path": chunk.path,
                "chunk_range": metadata.get("chunk_range"),
                "line_range": metadata.get("line_range"),
            }
        )
    return records


def build_index(
    repo_root: Path,
    *,
    embedder: EmbeddingBackend,
    output_path: Path,
    include_extensions: Sequence[str] | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> None:
    """Create a vector index for the repository at `repo_root`."""

    selected_extensions = (
        tuple(ext.lower() for ext in include_extensions) if include_extensions else DEFAULT_EXTENSIONS
    )

    documents = list(
        iter_repository_documents(repo_root, include_extensions=selected_extensions)
    )
    if not documents:
        raise ValueError("No documents found to index.")

    output_path.mkdir(parents=True, exist_ok=True)

    hierarchy = _collect_hierarchy(repo_root, selected_extensions)
    (output_path / "hierarchy.json").write_text(json.dumps(hierarchy, indent=2))

    documents_payload = _documents_to_json(documents)
    (output_path / "documents.json").write_text(json.dumps(documents_payload, indent=2))

    chunks = list(
        chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )

    if not chunks:
        raise ValueError("No documents found to index.")

    chunks_payload = _chunks_to_json(chunks)
    (output_path / "chunks.json").write_text(json.dumps(chunks_payload, indent=2))

    texts = [chunk.content for chunk in chunks]
    metadatas = [chunk.as_metadata() for chunk in chunks]

    embeddings = embedder.embed(texts)

    store = create_persistent_vector_store(output_path)
    store.add(embeddings, texts, metadatas)
    store.save(output_path)


def load_index(path: Path) -> VectorStore:
    return load_persistent_vector_store(path)


def query_index(
    store: VectorStore,
    *,
    embedder: EmbeddingBackend,
    question: str,
    top_k: int = 5,
) -> List[tuple[Document, float]]:
    """Run similarity search against the index and return top chunks."""

    query_embedding = embedder.embed([question])[0]
    results = store.search(query_embedding, top_k=top_k)
    documents: List[tuple[Document, float]] = []
    for stored_doc, score in results:
        doc = Document(
            path=stored_doc.metadata.get("path", "unknown"),
            content=stored_doc.text,
            metadata=stored_doc.metadata,
        )
        documents.append((doc, score))
    return documents
