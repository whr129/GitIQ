from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import importlib
import os
from typing import Optional, Sequence

from .chunker import chunk_document
from .documents import Document
from .embeddings import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    ensure_backend,
)
from .git_utils import GitError, capture_snapshot
from .metadata_store import MetadataError, PostgresMetadataStore
from .pipeline import build_index, load_index, query_index
from .session_memory import RedisSessionMemory, SessionMemoryError

DEFAULT_RESPONSE_MODEL = "gpt-5-nano"

@dataclass
class GenerationConfig:
    """Settings for answer generation."""

    model: str = DEFAULT_RESPONSE_MODEL
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    prompt_template: str = (
        "Use the following pieces of context to answer the question at the end.\n"
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
        "Use three sentences maximum and keep the answer as concise as possible.\n"
        'Always say "thanks for asking!" at the end of the answer.\n\n'
        "{context}\n\n"
        "Question: {question}\n\n"
        "Helpful Answer:"
    )


def _build_context(chunks: Sequence[Document]) -> str:
    if not chunks:
        return "No relevant context provided."

    formatted_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        path = metadata.get("path", "unknown")
        header = f"[{idx}] path={path}"
        body = chunk.content
        formatted_chunks.append(f"{header}\n{body}")
    return "\n\n".join(formatted_chunks)


def _build_change_documents(
    repo_root: Path,
    changed_files: Sequence[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Load and chunk changed files so we can feed them directly to the LLM."""

    change_docs: list[Document] = []
    for rel_path in changed_files:
        file_path = (repo_root / rel_path).resolve()
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
        metadata = {"path": rel_path, "source": "git_change"}
        doc = Document(path=rel_path, content=text, metadata=metadata)
        change_docs.extend(
            chunk_document(
                doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return change_docs


def generate_answer(
    question: str,
    context_documents: Sequence[Document],
    *,
    config: GenerationConfig | None = None,
    client: Optional[object] = None,
) -> str:
    """Generate a concise answer using the given question and supporting documents."""

    params = config or GenerationConfig()

    prompt = params.prompt_template.format(
        context=_build_context(context_documents),
        question=question,
    )

    if client is None:
        openai = importlib.import_module("openai")
        client = openai.OpenAI()

    request_kwargs = {
        "model": params.model,
        "instructions": "You are a helpful assistant that answers concisely.",
        "input": prompt,
    }
    if params.temperature is not None:
        request_kwargs["temperature"] = params.temperature
    if params.max_output_tokens is not None:
        request_kwargs["max_output_tokens"] = params.max_output_tokens

    response = client.responses.create(**request_kwargs)

    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return output_text.strip()

    # Fallback: concatenate textual content if output_text is unavailable.
    chunks = []
    for item in getattr(response, "output", []):
        if isinstance(item, dict):
            for content_part in item.get("content", []):
                if isinstance(content_part, dict) and content_part.get("type") == "text":
                    text = content_part.get("text")
                    if text:
                        chunks.append(text)
    return "\n".join(chunks).strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repo-rag-generate",
        description="Run retrieval-augmented generation over a repository.",
    )
    parser.add_argument("--repo", type=Path, required=True, help="Path to the repository to index.")
    parser.add_argument("--question", type=str, required=True, help="Question to answer.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="File path where retrieved context and answer will be written.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Directory for the persistent vector store (defaults to repo_index/<repo-name>).",
    )
    parser.add_argument(
        "--backend",
        choices=("sentence-transformer", "openai"),
        default="sentence-transformer",
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Embedding model identifier passed to the backend. Defaults to "
            f"'{DEFAULT_SENTENCE_TRANSFORMER_MODEL}' or '{DEFAULT_OPENAI_MODEL}'."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk when indexing.")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlapping characters between chunks when indexing.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of threads to use for concurrent embedding.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding calls (per worker).",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Optional list of file extensions to include during indexing.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to include in context.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild the index even if a persistent store already exists.",
    )
    parser.add_argument(
        "--response-model",
        type=str,
        default=None,
        help="Override the LLM model used for answer generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the response temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Limit the number of tokens produced by the response model.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Custom prompt template containing {context} and {question} placeholders.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Explicit OpenAI API key (falls back to OPENAI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis URL used to persist chat memory between Q&A runs.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session identifier for Redis-backed memory (requires --redis-url).",
    )
    parser.add_argument(
        "--git-changes-since",
        type=str,
        default=None,
        help="Git reference to diff against for change-aware documentation (e.g., HEAD~1).",
    )
    parser.add_argument(
        "--document-changes",
        action="store_true",
        help="Generate documentation for detected git changes and append it to the log output.",
    )
    parser.add_argument(
        "--change-question-template",
        type=str,
        default="Document the recent code changes for {files}. Highlight intent, affected modules, and usage notes.",
        help="Template used when summarizing code changes; {files} placeholder is replaced with a comma-separated list.",
    )
    parser.add_argument(
        "--metadata-dsn",
        type=str,
        default=None,
        help="PostgreSQL DSN for recording index and generation runs (optional).",
    )
    parser.add_argument(
        "--metadata-notes",
        type=str,
        default=None,
        help="Optional JSON payload stored alongside metadata records for auditing.",
    )
    return parser


def _index_exists(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    except OSError:
        return False
    return True


def _write_log(
    output_path: Path,
    question: str,
    results: Sequence[tuple[Document, float]],
    answer: str,
    *,
    snapshot: object | None = None,
    change_question: str | None = None,
    change_answer: str | None = None,
) -> None:
    lines = [f"Question: {question}"]

    if snapshot:
        commit = getattr(snapshot, "commit", None)
        branch = getattr(snapshot, "branch", None)
        changed_files = getattr(snapshot, "changed_files", None)
        lines.append("Git snapshot:")
        if commit:
            lines.append(f"- commit={commit}")
        if branch:
            lines.append(f"- branch={branch}")
        if changed_files:
            lines.append("- changed_files=" + ", ".join(changed_files))

    lines.append(f"Top {len(results)} matches:")

    for rank, (doc, score) in enumerate(results, start=1):
        snippet = doc.content.replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        metadata = doc.metadata or {}
        path = metadata.get("path", doc.path)
        lines.append(f"[{rank}] score={score:.3f} path={path}")
        lines.append(snippet)

    lines.append("")
    lines.append("Answer:")
    lines.append(answer)

    if change_question and change_answer:
        lines.append("")
        lines.append("Change documentation question:")
        lines.append(change_question)
        lines.append("Change documentation answer:")
        lines.append(change_answer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = args.repo.expanduser().resolve()
    index_path = (
        args.index or Path("repo_index") / repo_root.name
    ).expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    metadata_store = None
    metadata_notes: dict = {}
    if args.metadata_notes:
        try:
            metadata_notes = json.loads(args.metadata_notes)
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid JSON for --metadata-notes: {exc}")

    if args.metadata_dsn:
        try:
            metadata_store = PostgresMetadataStore(args.metadata_dsn)
        except MetadataError as exc:
            parser.error(str(exc))

    session_memory = None
    if args.session_id and not args.redis_url:
        parser.error("--session-id requires --redis-url.")
    if args.redis_url:
        try:
            session_memory = RedisSessionMemory(args.redis_url)
        except SessionMemoryError as exc:
            parser.error(str(exc))

    snapshot = None
    if args.document_changes or args.git_changes_since or metadata_store:
        try:
            snapshot = capture_snapshot(repo_root, since_ref=args.git_changes_since)
        except GitError as exc:
            print(f"Warning: unable to read git metadata: {exc}")

    backend_model = args.model_name
    if backend_model is None:
        backend_model = (
            DEFAULT_SENTENCE_TRANSFORMER_MODEL
            if args.backend == "sentence-transformer"
            else DEFAULT_OPENAI_MODEL
        )

    embedder = ensure_backend(args.backend, model_name=backend_model)

    if args.force_reindex or not _index_exists(index_path):
        build_index(
            repo_root,
            embedder=embedder,
            output_path=index_path,
            include_extensions=args.extensions,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_workers=args.max_workers,
            embedding_batch_size=args.embedding_batch_size,
        )

        if metadata_store and snapshot:
            try:
                metadata_store.record_index_run(
                    repo_root=repo_root,
                    index_path=index_path,
                    commit_hash=snapshot.commit,
                    branch=snapshot.branch,
                    changed_files=snapshot.changed_files,
                    notes=metadata_notes,
                )
            except Exception as exc:  # pragma: no cover - metadata persistence best-effort
                print(f"Warning: failed to record index metadata: {exc}")

    store = load_index(index_path)
    results = query_index(
        store,
        embedder=embedder,
        question=args.question,
        top_k=args.top_k,
    )

    documents = [doc for doc, _ in results]
    history_docs: list[Document] = []
    if session_memory and args.session_id:
        history_docs = session_memory.as_documents(args.session_id)
    context_documents = history_docs + documents

    gen_kwargs = {}
    if args.response_model is not None:
        gen_kwargs["model"] = args.response_model
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.max_output_tokens is not None:
        gen_kwargs["max_output_tokens"] = args.max_output_tokens
    if args.prompt_template is not None:
        gen_kwargs["prompt_template"] = args.prompt_template

    generation_config = GenerationConfig(**gen_kwargs)

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OpenAI API key is required. Provide --openai-api-key or set OPENAI_API_KEY in the environment.")

    openai_module = importlib.import_module("openai")
    response_client = openai_module.OpenAI(api_key=api_key)
    answer = generate_answer(
        args.question,
        context_documents,
        config=generation_config,
        client=response_client,
    )

    if session_memory and args.session_id:
        try:
            session_memory.append(
                args.session_id,
                question=args.question,
                answer=answer,
                metadata={
                    "retrieved_paths": [doc.path for doc, _ in results],
                    "response_model": generation_config.model,
                },
            )
        except Exception as exc:  # pragma: no cover - best-effort persistence
            print(f"Warning: failed to persist session memory: {exc}")

    change_answer: str | None = None
    change_question: str | None = None
    if args.document_changes:
        changed_files = snapshot.changed_files if snapshot else []
        change_question = args.change_question_template.format(
            files=", ".join(changed_files) if changed_files else "no tracked files"
        )
        change_docs = _build_change_documents(
            repo_root,
            changed_files,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if change_docs:
            change_answer = generate_answer(
                change_question,
                change_docs,
                config=generation_config,
                client=response_client,
            )
        else:
            change_answer = "No changed files detected; nothing to document."

    generation_metadata = {
        "backend": args.backend,
        "model_name": backend_model,
        "top_k": args.top_k,
        "changed_files": snapshot.changed_files if snapshot else [],
        "session_id": args.session_id,
        "history_context": len(history_docs),
    }
    if metadata_notes:
        generation_metadata["notes"] = metadata_notes

    if metadata_store:
        try:
            metadata_store.record_generation(
                repo_root=repo_root,
                index_path=index_path,
                commit_hash=snapshot.commit if snapshot else None,
                branch=snapshot.branch if snapshot else None,
                question=args.question,
                answer=answer,
                response_model=generation_config.model,
                metadata=generation_metadata,
            )
            if change_answer and change_question:
                metadata_store.record_generation(
                    repo_root=repo_root,
                    index_path=index_path,
                    commit_hash=snapshot.commit if snapshot else None,
                    branch=snapshot.branch if snapshot else None,
                    question=change_question,
                    answer=change_answer,
                    response_model=generation_config.model,
                    metadata={**generation_metadata, "change_doc": True},
                )
        except Exception as exc:  # pragma: no cover - metadata persistence best-effort
            print(f"Warning: failed to record generation metadata: {exc}")

    _write_log(
        output_path,
        args.question,
        results,
        answer,
        snapshot=snapshot,
        change_question=change_question,
        change_answer=change_answer,
    )
    print(answer)
    if change_answer:
        print("\nChange documentation:\n")
        print(change_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
