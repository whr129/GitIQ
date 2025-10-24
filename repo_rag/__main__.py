from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

from .embeddings import DEFAULT_OPENAI_MODEL, DEFAULT_SENTENCE_TRANSFORMER_MODEL, ensure_backend
from .pipeline import build_index, load_index, query_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repo-rag",
        description="Minimal RAG starter kit for local repositories.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a repository into a vector store.")
    index_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path to the repository root (defaults to current directory).",
    )
    index_parser.add_argument(
        "--output",
        type=Path,
        default=Path("repo_index"),
        help="Output path prefix for the vector store artifacts.",
    )
    index_parser.add_argument(
        "--backend",
        choices=("sentence-transformer", "openai"),
        default="sentence-transformer",
        help="Embedding backend to use.",
    )
    index_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Model identifier passed to the embedding backend. "
            f"Defaults to '{DEFAULT_SENTENCE_TRANSFORMER_MODEL}' for sentence-transformers "
            f"and '{DEFAULT_OPENAI_MODEL}' for OpenAI."
        ),
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of characters per chunk.",
    )
    index_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Number of overlapping characters between chunks.",
    )
    index_parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Optional list of file extensions to include (e.g. --extensions .py .md).",
    )

    query_parser = subparsers.add_parser("query", help="Query a previously built repository index.")
    query_parser.add_argument(
        "--index",
        type=Path,
        default=Path("repo_index"),
        help="Path prefix to the vector store artifacts (same value used during indexing).",
    )
    query_parser.add_argument(
        "--backend",
        choices=("sentence-transformer", "openai"),
        default="sentence-transformer",
        help="Embedding backend to use (should match the one used during indexing).",
    )
    query_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model identifier passed to the embedding backend.",
    )
    query_parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Natural language question to search for.",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of matches to return.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "index":
            embedder = ensure_backend(args.backend, model_name=args.model_name)
            repo_root = args.repo.expanduser().resolve()
            output_path = args.output.expanduser().resolve()
            build_index(
                repo_root,
                embedder=embedder,
                output_path=output_path,
                include_extensions=args.extensions,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            print(f"Index written to '{output_path}'.")
            return 0

        if args.command == "query":
            embedder = ensure_backend(args.backend, model_name=args.model_name)
            index_path = args.index.expanduser().resolve()
            store = load_index(index_path)
            results = query_index(
                store,
                embedder=embedder,
                question=args.question,
                top_k=args.top_k,
            )
            if not results:
                print("No matches found.")
                return 0

            print(f"Top {len(results)} matches:")
            for rank, (doc, score) in enumerate(results, start=1):
                snippet = doc.content.replace("\n", " ")
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                print(dedent(
                    f"""
                    [{rank}] score={score:.3f} path={doc.path}
                    {snippet}
                    """
                ).strip())
            return 0

    except ModuleNotFoundError as exc:
        missing_module = exc.name or "unknown module"
        parser.error(
            f"Dependency '{missing_module}' is required for the requested embedding backend. "
            "Install the necessary package and try again."
        )
    except Exception as exc:  # pragma: no cover - defensive logging only
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
