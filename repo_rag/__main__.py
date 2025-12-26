from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

from .config import RepoRagConfig, load_config
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
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML configuration file (see repo_rag/config.toml for an example).",
    )
    index_parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Path to the repository root (defaults to current directory).",
    )
    index_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory path where the persistent vector store will be written.",
    )
    index_parser.add_argument(
        "--backend",
        choices=("sentence-transformer", "openai"),
        default=None,
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
        default=None,
        help="Number of characters per chunk.",
    )
    index_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Number of overlapping characters between chunks.",
    )
    index_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of threads to use when embedding chunks (default: 4).",
    )
    index_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Number of chunks per embedding batch submitted to each worker (default: 32).",
    )
    index_parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Optional list of file extensions to include (e.g. --extensions .py .md).",
    )

    query_parser = subparsers.add_parser("query", help="Query a previously built repository index.")
    query_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML configuration file (see repo_rag/config.toml for an example).",
    )
    query_parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Directory path of the previously persisted vector store.",
    )
    query_parser.add_argument(
        "--backend",
        choices=("sentence-transformer", "openai"),
        default=None,
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
        default=None,
        help="Number of matches to return.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config: RepoRagConfig | None = None
        if getattr(args, "config", None) is not None:
            config_path = args.config.expanduser().resolve()
            config = load_config(config_path)

        if args.command == "index":
            config_index = config.indexing if config else None

            backend = (
                args.backend
                or (config_index.embedding.backend if config_index else None)
                or "sentence-transformer"
            )

            model_name = (
                args.model_name
                or (config_index.embedding.model_name if config_index else None)
                or (DEFAULT_SENTENCE_TRANSFORMER_MODEL if backend == "sentence-transformer" else DEFAULT_OPENAI_MODEL)
            )

            repo_root = (
                args.repo
                or (config_index.repo if config_index else None)
                or Path.cwd()
            ).expanduser().resolve()

            output_path = (
                args.output
                or (config_index.output if config_index else None)
                or Path("repo_index")
            ).expanduser().resolve()

            chunk_size = args.chunk_size
            if chunk_size is None:
                chunk_size = config_index.chunk_size if config_index and config_index.chunk_size else 500

            chunk_overlap = args.chunk_overlap
            if chunk_overlap is None:
                chunk_overlap = config_index.chunk_overlap if config_index and config_index.chunk_overlap else 100

            max_workers = args.max_workers if args.max_workers is not None else 4
            embedding_batch_size = (
                args.embedding_batch_size if args.embedding_batch_size is not None else 32
            )

            extensions = (
                args.extensions
                if args.extensions is not None
                else (config_index.extensions if config_index else None)
            )

            embedder = ensure_backend(backend, model_name=model_name)
            build_index(
                repo_root,
                embedder=embedder,
                output_path=output_path,
                include_extensions=extensions,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_workers=max_workers,
                embedding_batch_size=embedding_batch_size,
            )
            print(f"Index written to '{output_path}'.")
            return 0

        if args.command == "query":
            config_query = config.query if config else None

            backend = (
                args.backend
                or (config_query.embedding.backend if config_query else None)
                or "sentence-transformer"
            )

            model_name = (
                args.model_name
                or (config_query.embedding.model_name if config_query else None)
                or (DEFAULT_SENTENCE_TRANSFORMER_MODEL if backend == "sentence-transformer" else DEFAULT_OPENAI_MODEL)
            )

            index_path = (
                args.index
                or (config_query.index if config_query else None)
                or Path("repo_index")
            ).expanduser().resolve()

            top_k = args.top_k
            if top_k is None:
                top_k = config_query.top_k if config_query and config_query.top_k else 5

            embedder = ensure_backend(backend, model_name=model_name)
            store = load_index(index_path)
            results = query_index(
                store,
                embedder=embedder,
                question=args.question,
                top_k=top_k,
            )
            if not results:
                print("No matches found.")
                return 0

            print(f"Question: {args.question}")
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
            print("")
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
