from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class StoredDocument:
    text: str
    metadata: dict


class NumpyVectorStore:
    """Lightweight cosine-similarity vector store backed by numpy arrays."""

    def __init__(self) -> None:
        self._vectors: NDArray[np.float32] | None = None
        self._documents: List[StoredDocument] = []

    def add(
        self,
        embeddings: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if len(texts) != len(metadatas) or len(texts) != len(vectors):
            raise ValueError("Embeddings, texts, and metadatas must be same length.")

        for text, metadata in zip(texts, metadatas, strict=False):
            self._documents.append(StoredDocument(text=text, metadata=metadata))

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.concatenate([self._vectors, vectors], axis=0)

    def search(self, embedding: Sequence[float], top_k: int = 5) -> List[Tuple[StoredDocument, float]]:
        if self._vectors is None:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector.")

        # Compute cosine similarity
        query_norm = np.linalg.norm(query)
        base_norms = np.linalg.norm(self._vectors, axis=1)
        similarities = np.dot(self._vectors, query) / (base_norms * query_norm + 1e-10)

        top_indices = np.argsort(-similarities)[:top_k]
        results: List[Tuple[StoredDocument, float]] = []
        for idx in top_indices:
            doc = self._documents[int(idx)]
            score = float(similarities[int(idx)])
            results.append((doc, score))
        return results

    def save(self, path: Path) -> None:
        if self._vectors is None:
            raise ValueError("Nothing to save; the store has no vectors yet.")

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self._vectors)
        metadata_path = path.with_suffix(".jsonl")
        with metadata_path.open("w", encoding="utf-8") as handle:
            for doc in self._documents:
                payload = {"text": doc.text, "metadata": doc.metadata}
                handle.write(json.dumps(payload))
                handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "NumpyVectorStore":
        store = cls()
        vectors_path = path.with_suffix(".npy")
        metadata_path = path.with_suffix(".jsonl")

        store._vectors = np.load(vectors_path)

        documents: List[StoredDocument] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                documents.append(StoredDocument(text=payload["text"], metadata=payload["metadata"]))
        store._documents = documents
        return store

