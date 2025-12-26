from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class MetadataError(RuntimeError):
    """Raised when metadata persistence fails."""


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


@dataclass
class PostgresMetadataStore:
    """Lightweight PostgreSQL recorder for index and generation runs."""

    dsn: str

    def __post_init__(self) -> None:
        try:
            self._driver = importlib.import_module("psycopg")
            self._connect = self._driver.connect
        except ModuleNotFoundError:
            try:
                self._driver = importlib.import_module("psycopg2")
                self._connect = self._driver.connect
            except ModuleNotFoundError as exc:
                raise MetadataError(
                    "psycopg or psycopg2 is required to record metadata, but neither is installed."
                ) from exc
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS repo_index_runs (
                        id BIGSERIAL PRIMARY KEY,
                        repo_root TEXT NOT NULL,
                        index_path TEXT NOT NULL,
                        commit_hash TEXT NOT NULL,
                        branch TEXT NOT NULL,
                        changed_files JSONB,
                        notes JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS repo_generation_runs (
                        id BIGSERIAL PRIMARY KEY,
                        repo_root TEXT NOT NULL,
                        index_path TEXT,
                        commit_hash TEXT,
                        branch TEXT,
                        question TEXT NOT NULL,
                        answer TEXT,
                        response_model TEXT,
                        metadata JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_repo_index_runs_repo_created_at ON repo_index_runs (repo_root, created_at DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_repo_generation_runs_repo_created_at ON repo_generation_runs (repo_root, created_at DESC)"
                )
            conn.commit()

    def record_index_run(
        self,
        *,
        repo_root: Path,
        index_path: Path,
        commit_hash: str,
        branch: str,
        changed_files: list[str],
        notes: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = notes or {}
        with self._connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO repo_index_runs (repo_root, index_path, commit_hash, branch, changed_files, notes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        str(repo_root),
                        str(index_path),
                        commit_hash,
                        branch,
                        _safe_json(changed_files),
                        _safe_json(payload),
                    ],
                )
            conn.commit()

    def record_generation(
        self,
        *,
        repo_root: Path,
        index_path: Optional[Path],
        commit_hash: Optional[str],
        branch: Optional[str],
        question: str,
        answer: str,
        response_model: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = metadata or {}
        with self._connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO repo_generation_runs (
                        repo_root, index_path, commit_hash, branch, question, answer, response_model, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        str(repo_root),
                        str(index_path) if index_path else None,
                        commit_hash,
                        branch,
                        question,
                        answer,
                        response_model,
                        _safe_json(payload),
                    ],
                )
            conn.commit()
