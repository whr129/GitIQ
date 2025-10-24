from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence, Tuple


@dataclass
class StoredDocument:
    text: str
    metadata: dict


class VectorStore(Protocol):
    def add(
        self,
        embeddings: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        ...

    def search(self, embedding: Sequence[float], top_k: int = 5) -> List[Tuple[StoredDocument, float]]:
        ...

    def save(self, path: Path) -> None:
        ...


@dataclass
class ChromaVectorStore:
    """Vector store powered by Chroma with persistent storage."""

    persist_path: Path
    collection_name: str

    def __post_init__(self) -> None:
        chromadb = importlib.import_module("chromadb")
        self._chromadb = chromadb
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Track next id to avoid collisions when appending.
        try:
            self._next_id = int(self._collection.count())
        except Exception:  # pragma: no cover - defensive fallback
            self._next_id = 0

    @classmethod
    def create(cls, path: Path) -> "ChromaVectorStore":
        store = cls(path, _collection_name_from_path(path))
        # Drop existing collection data to start fresh.
        try:
            store._client.delete_collection(name=store.collection_name)
        except Exception:
            pass
        store._collection = store._client.create_collection(
            name=store.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        store._next_id = 0
        return store

    @classmethod
    def load(cls, path: Path) -> "ChromaVectorStore":
        return cls(path, _collection_name_from_path(path))

    def add(
        self,
        embeddings: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if len(texts) != len(metadatas) or len(texts) != len(embeddings):
            raise ValueError("Embeddings, texts, and metadatas must be same length.")

        ids = []
        for _ in texts:
            ids.append(f"{self.collection_name}-{self._next_id}")
            self._next_id += 1

        self._collection.add(
            ids=ids,
            embeddings=[list(map(float, emb)) for emb in embeddings],
            documents=list(texts),
            metadatas=[dict(meta) for meta in metadatas],
        )

    def search(self, embedding: Sequence[float], top_k: int = 5) -> List[Tuple[StoredDocument, float]]:
        if not embedding:
            return []

        results = self._collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        distances = results.get("distances") or []

        if not documents:
            return []

        docs_list = documents[0]
        metas_list = metadatas[0] if metadatas else [{}] * len(docs_list)
        dist_list = distances[0] if distances else [0.0] * len(docs_list)

        output: List[Tuple[StoredDocument, float]] = []
        for doc_text, meta, dist in zip(docs_list, metas_list, dist_list, strict=False):
            stored = StoredDocument(text=doc_text, metadata=meta or {})
            # Convert cosine distance back into similarity score.
            score = 1.0 - float(dist)
            output.append((stored, score))
        return output

    def save(self, path: Path) -> None:  # pragma: no cover - persistence handled by Chroma
        # PersistentClient writes to disk automatically; nothing to do.
        self.persist_path.mkdir(parents=True, exist_ok=True)


def _collection_name_from_path(path: Path) -> str:
    name = path.name or "repo_rag"
    return name.replace(".", "_")


def create_persistent_vector_store(path: Path) -> VectorStore:
    return ChromaVectorStore.create(path)


def load_persistent_vector_store(path: Path) -> VectorStore:
    return ChromaVectorStore.load(path)
