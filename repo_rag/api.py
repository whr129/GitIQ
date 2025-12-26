from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .embeddings import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    ensure_backend,
)
from .generator import (
    DEFAULT_RESPONSE_MODEL,
    GenerationConfig,
    _write_log,
    generate_answer,
    _index_exists,
    _build_change_documents,
)
from .git_utils import GitError, capture_snapshot
from .metadata_store import MetadataError, PostgresMetadataStore
from .pipeline import build_index, load_index, query_index
from .session_memory import RedisSessionMemory, SessionMemoryError


class IndexRequest(BaseModel):
    repo: str
    output: Optional[str] = None
    backend: str = "sentence-transformer"
    model_name: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 100
    max_workers: int = 4
    embedding_batch_size: int = 32
    force_reindex: bool = False
    git_changes_since: Optional[str] = None
    metadata_dsn: Optional[str] = None
    metadata_notes: Optional[dict] = None


class GenerateRequest(BaseModel):
    repo: str
    question: str
    output: Optional[str] = None
    index: Optional[str] = None
    backend: str = "sentence-transformer"
    model_name: Optional[str] = None
    top_k: int = 5
    response_model: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    prompt_template: Optional[str] = None
    openai_api_key: Optional[str] = None
    force_reindex: bool = False
    document_changes: bool = False
    git_changes_since: Optional[str] = None
    change_question_template: Optional[str] = None
    metadata_dsn: Optional[str] = None
    metadata_notes: Optional[dict] = None
    redis_url: Optional[str] = None
    session_id: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 100
    max_workers: int = 4
    embedding_batch_size: int = 32


app = FastAPI(title="Repo RAG API", version="1.0.0")


def _resolve_embedder(backend: str, model_name: Optional[str]) -> object:
    selected_model = model_name
    if selected_model is None:
        selected_model = (
            DEFAULT_SENTENCE_TRANSFORMER_MODEL
            if backend == "sentence-transformer"
            else DEFAULT_OPENAI_MODEL
        )
    return ensure_backend(backend, model_name=selected_model)


def _maybe_metadata_store(dsn: Optional[str]) -> Optional[PostgresMetadataStore]:
    if not dsn:
        return None
    try:
        return PostgresMetadataStore(dsn)
    except MetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _maybe_session_memory(url: Optional[str]) -> Optional[RedisSessionMemory]:
    if not url:
        return None
    try:
        return RedisSessionMemory(url)
    except SessionMemoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/index")
def index_repository(payload: IndexRequest) -> dict:
    repo_root = Path(payload.repo).expanduser().resolve()
    output_path = (
        Path(payload.output) if payload.output else Path("repo_index") / repo_root.name
    ).expanduser().resolve()

    embedder = _resolve_embedder(payload.backend, payload.model_name)
    metadata_store = _maybe_metadata_store(payload.metadata_dsn)

    snapshot = None
    if payload.git_changes_since or metadata_store:
        try:
            snapshot = capture_snapshot(repo_root, since_ref=payload.git_changes_since)
        except GitError:
            snapshot = None

    if payload.force_reindex or not _index_exists(output_path):
        build_index(
            repo_root,
            embedder=embedder,
            output_path=output_path,
            include_extensions=None,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            max_workers=payload.max_workers,
            embedding_batch_size=payload.embedding_batch_size,
        )

        if metadata_store and snapshot:
            metadata_store.record_index_run(
                repo_root=repo_root,
                index_path=output_path,
                commit_hash=snapshot.commit,
                branch=snapshot.branch,
                changed_files=snapshot.changed_files,
                notes=payload.metadata_notes,
            )

    return {
        "index_path": str(output_path),
        "repo": str(repo_root),
        "reindexed": payload.force_reindex,
    }


@app.post("/generate")
def generate(payload: GenerateRequest) -> dict:
    repo_root = Path(payload.repo).expanduser().resolve()
    index_path = (
        Path(payload.index) if payload.index else Path("repo_index") / repo_root.name
    ).expanduser().resolve()
    output_path = (
        Path(payload.output)
        if payload.output
        else Path("output") / f"{repo_root.name}_run.txt"
    ).expanduser().resolve()

    embedder = _resolve_embedder(payload.backend, payload.model_name)
    metadata_store = _maybe_metadata_store(payload.metadata_dsn)
    session_memory = _maybe_session_memory(payload.redis_url) if payload.session_id else None

    snapshot = None
    if payload.document_changes or payload.git_changes_since or metadata_store:
        try:
            snapshot = capture_snapshot(repo_root, since_ref=payload.git_changes_since)
        except GitError:
            snapshot = None

    if payload.force_reindex or not _index_exists(index_path):
        build_index(
            repo_root,
            embedder=embedder,
            output_path=index_path,
            include_extensions=None,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            max_workers=payload.max_workers,
            embedding_batch_size=payload.embedding_batch_size,
        )

    store = load_index(index_path)
    results = query_index(
        store,
        embedder=embedder,
        question=payload.question,
        top_k=payload.top_k,
    )

    documents = [doc for doc, _ in results]
    history_docs = session_memory.as_documents(payload.session_id) if session_memory and payload.session_id else []
    context_docs = history_docs + documents

    gen_config = GenerationConfig(
        model=payload.response_model or DEFAULT_RESPONSE_MODEL,
        temperature=payload.temperature,
        max_output_tokens=payload.max_output_tokens,
        prompt_template=payload.prompt_template or GenerationConfig().prompt_template,
    )

    api_key = payload.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if payload.backend == "openai" and not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required.")

    openai_module = __import__("openai")
    client = openai_module.OpenAI(api_key=api_key) if api_key else openai_module.OpenAI()

    answer = generate_answer(payload.question, context_docs, config=gen_config, client=client)

    change_answer = None
    change_question = None
    if payload.document_changes:
        changed_files = snapshot.changed_files if snapshot else []
        change_question = (payload.change_question_template or "Document the recent code changes for {files}.").format(
            files=", ".join(changed_files) if changed_files else "no tracked files"
        )
        change_docs = _build_change_documents(
            repo_root,
            changed_files,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
        if change_docs:
            change_answer = generate_answer(change_question, change_docs, config=gen_config, client=client)
        else:
            change_answer = "No changed files detected; nothing to document."

    if payload.session_id and session_memory:
        session_memory.append(
            payload.session_id,
            question=payload.question,
            answer=answer,
            metadata={"response_model": gen_config.model, "changed_files": snapshot.changed_files if snapshot else []},
        )

    if metadata_store:
        metadata_store.record_generation(
            repo_root=repo_root,
            index_path=index_path,
            commit_hash=snapshot.commit if snapshot else None,
            branch=snapshot.branch if snapshot else None,
            question=payload.question,
            answer=answer,
            response_model=gen_config.model,
            metadata=payload.metadata_notes or {},
        )
        if change_answer and change_question:
            metadata_store.record_generation(
                repo_root=repo_root,
                index_path=index_path,
                commit_hash=snapshot.commit if snapshot else None,
                branch=snapshot.branch if snapshot else None,
                question=change_question,
                answer=change_answer,
                response_model=gen_config.model,
                metadata={"change_doc": True, **(payload.metadata_notes or {})},
            )

    _write_log(
        output_path,
        payload.question,
        results,
        answer,
        snapshot=snapshot,
        change_question=change_question,
        change_answer=change_answer,
    )

    return {
        "answer": answer,
        "change_answer": change_answer,
        "index_path": str(index_path),
        "output_path": str(output_path),
        "matches": [
            {"path": doc.path, "score": score}
            for doc, score in results
        ],
    }


try:  # pragma: no cover - optional Lambda adapter
    from mangum import Mangum

    handler = Mangum(app)
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    handler = None
