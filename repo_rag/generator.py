from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import importlib
import os
from typing import Optional, Sequence

from .documents import Document
from .embeddings import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    ensure_backend,
)
from .pipeline import build_index, load_index, query_index

DEFAULT_RESPONSE_MODEL = "gpt-5"

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
) -> None:
    lines = [f"Question: {question}", f"Top {len(results)} matches:"]

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
        )

    store = load_index(index_path)
    results = query_index(
        store,
        embedder=embedder,
        question=args.question,
        top_k=args.top_k,
    )

    documents = [doc for doc, _ in results]

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
        documents,
        config=generation_config,
        client=response_client,
    )

    _write_log(output_path, args.question, results, answer)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
