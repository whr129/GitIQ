from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .documents import Document

DEFAULT_EXTENSIONS: Sequence[str] = (
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".md",
    ".txt",
    ".rst",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
)

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


def iter_repository_documents(
    root: Path,
    *,
    include_extensions: Sequence[str] | None = None,
    exclude_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
    max_file_size_bytes: int = 750_000,
) -> Iterator[Document]:
    """Yield `Document` instances for each text file in a repository-like tree."""

    include_extensions = include_extensions or DEFAULT_EXTENSIONS
    exclude_dirs = set(exclude_dirs)

    for dirpath, dirnames, filenames in os.walk(root):
        path_obj = Path(dirpath)

        # Drop excluded directories before deeper traversal.
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for filename in filenames:
            file_path = path_obj / filename
            if not file_path.suffix:
                continue
            if file_path.suffix.lower() not in include_extensions:
                continue

            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size > max_file_size_bytes:
                continue

            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            yield Document(
                path=str(file_path.relative_to(root)),
                content=text,
                metadata={"size": str(size)},
            )

