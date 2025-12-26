from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .documents import Document


class SessionMemoryError(RuntimeError):
    """Raised when Redis-backed memory cannot be used."""


@dataclass
class RedisSessionMemory:
    """Small Redis wrapper to persist chat/Q&A history across runs."""

    url: str
    ttl_seconds: int = 3600
    namespace: str = "repo-rag"
    max_messages: int = 8

    def __post_init__(self) -> None:
        try:
            redis_module = importlib.import_module("redis")
        except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
            raise SessionMemoryError("redis-py is required for Redis session memory.") from exc
        self._client = redis_module.Redis.from_url(self.url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"{self.namespace}:{session_id}"

    def load(self, session_id: str) -> List[Dict]:
        data = self._client.lrange(self._key(session_id), 0, -1)
        history: List[Dict] = []
        for entry in data[-self.max_messages :]:
            try:
                history.append(json.loads(entry))
            except json.JSONDecodeError:
                continue
        return history

    def append(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        record = {
            "question": question,
            "answer": answer,
            "metadata": metadata or {},
        }
        self._client.rpush(self._key(session_id), json.dumps(record, ensure_ascii=True))
        self._client.ltrim(self._key(session_id), -self.max_messages, -1)
        if self.ttl_seconds > 0:
            self._client.expire(self._key(session_id), self.ttl_seconds)

    def as_documents(self, session_id: str) -> List[Document]:
        docs: List[Document] = []
        for idx, entry in enumerate(self.load(session_id), start=1):
            text = f"Q: {entry.get('question', '')}\nA: {entry.get('answer', '')}"
            docs.append(
                Document(
                    path=f"session/{session_id}/{idx}",
                    content=text,
                    metadata={"source": "redis_session", "index": str(idx)},
                )
            )
        return docs
