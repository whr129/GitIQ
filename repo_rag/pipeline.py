from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from .chunker import chunk_documents
from .documents import Document
from .embeddings import EmbeddingBackend
from .loader import iter_repository_documents
from .vectorstore import VectorStore, create_persistent_vector_store, load_persistent_vector_store


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

    documents = iter_repository_documents(repo_root, include_extensions=include_extensions)
    chunks = list(
        chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )

    if not chunks:
        raise ValueError("No documents found to index.")

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
