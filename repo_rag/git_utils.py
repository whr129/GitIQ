from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


class GitError(RuntimeError):
    """Raised when git data cannot be retrieved."""


def _run_git(repo_root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise GitError(stderr or f"git {' '.join(args)} failed with code {result.returncode}")
    return (result.stdout or "").strip()


def get_current_commit(repo_root: Path) -> str:
    """Return the current HEAD commit hash."""
    return _run_git(repo_root, ["rev-parse", "HEAD"])


def get_branch(repo_root: Path) -> str:
    """Return the active branch name (falls back to HEAD)."""
    try:
        return _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    except GitError:
        return "HEAD"


def list_changed_files(repo_root: Path, *, since_ref: Optional[str] = None) -> List[str]:
    """
    List files changed since `since_ref` (defaults to HEAD^).

    Falls back to `git status --porcelain` if no reference is provided
    or the repository has uncommitted changes.
    """

    if since_ref:
        try:
            output = _run_git(repo_root, ["diff", "--name-only", since_ref, "HEAD"])
            return [line for line in output.splitlines() if line.strip()]
        except GitError:
            pass

    status = _run_git(repo_root, ["status", "--porcelain"])
    files = []
    for line in status.splitlines():
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            files.append(parts[1])
    return files


@dataclass
class RepoSnapshot:
    commit: str
    branch: str
    changed_files: List[str]


def capture_snapshot(repo_root: Path, *, since_ref: Optional[str] = None) -> RepoSnapshot:
    """Capture a lightweight snapshot of repository state for logging/metadata."""

    commit = get_current_commit(repo_root)
    branch = get_branch(repo_root)
    changed_files = list_changed_files(repo_root, since_ref=since_ref)
    return RepoSnapshot(commit=commit, branch=branch, changed_files=changed_files)
