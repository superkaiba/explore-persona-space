"""Git provenance for result-JSON reproducibility metadata.

Consolidates the three duplicate `_git_commit_hash()` helpers previously
scattered across analysis/artifacts modules. Adds dirty-tree flagging so a
committed result JSON never claims provenance from a commit that does not
contain the code that produced it (task #2065; incident #1482).

Public entry point: `git_provenance()` returns a `GitProvenance` dataclass
carrying `commit_sha`, `dirty` (bool | None), and `dirty_paths` (bounded list).
`commit_string(prov)` renders the human-legible `<sha>` or `<sha>+dirty` form
for non-JSON channels (PDF metadata, PNG chunks, WandB run names). Result
JSONs merge `as_metadata_dict(prov)` into their `metadata` block, exposing the
structured fields `git_commit` / `git_dirty` / `git_dirty_paths`.

Contract:
- Never fails loud: a non-git tree, missing git binary, or subprocess timeout
  degrades to `commit_sha="unknown", dirty=None` (record it, don't crash the
  run). The rule this closes is "the git_commit field must not silently claim
  clean provenance while the working tree is dirty" — a `None` sentinel
  explicitly says "we could not check" (the code-style.md caveat).
- Working-tree-wide scope: `git status --porcelain=v1 --untracked-files=no`
  captures every modified tracked file, not just files matching some plan-time
  allow-list. `dirty_paths` is capped at `_MAX_DIRTY_PATHS` (default 50) with
  a trailing "... N more" marker so a large sweep never blows up the JSON.
- Bounded: 5s subprocess timeout per git call.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_GIT_TIMEOUT_SEC = 5
_MAX_DIRTY_PATHS = 50
_UNKNOWN = "unknown"


@dataclass(frozen=True)
class GitProvenance:
    """Structured git-provenance record for a run.

    Attributes:
        commit_sha: 8-hex short SHA of HEAD, or `"unknown"` if unresolved.
        dirty: True if the working tree has uncommitted tracked-file
            modifications; False if clean; None if the check could not
            run (non-git tree / missing binary / timeout).
        dirty_paths: List of modified tracked paths (porcelain v1 format),
            capped at `_MAX_DIRTY_PATHS` entries with a `... N more` tail.
    """

    commit_sha: str
    dirty: bool | None
    dirty_paths: list[str] = field(default_factory=list)


def _run_git(args: list[str], cwd: Path | None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git_short_sha(cwd: Path | None = None) -> str:
    """Return the 8-hex short SHA of HEAD, or `"unknown"` on any failure."""
    out = _run_git(["rev-parse", "--short=8", "HEAD"], cwd=cwd)
    if out is None:
        return _UNKNOWN
    sha = out.strip()
    return sha or _UNKNOWN


def _git_dirty_status(cwd: Path | None = None) -> tuple[bool | None, list[str]]:
    """Return (dirty?, capped modified-paths list).

    Uses `git status --porcelain=v1 --untracked-files=no` for a tracked-file
    read; untracked files are excluded because a run cannot produce different
    numbers from untracked files it never imported.
    """
    out = _run_git(
        ["status", "--porcelain=v1", "--untracked-files=no"],
        cwd=cwd,
    )
    if out is None:
        return None, []
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return False, []
    # Porcelain-v1 format: two status chars + space + path (rename arrow retained
    # verbatim; the audit trail wants the raw entry).
    paths = [ln[3:] if len(ln) > 3 else ln for ln in lines]
    if len(paths) > _MAX_DIRTY_PATHS:
        overflow = len(paths) - _MAX_DIRTY_PATHS
        paths = paths[:_MAX_DIRTY_PATHS] + [f"... {overflow} more"]
    return True, paths


def git_provenance(cwd: Path | None = None) -> GitProvenance:
    """Capture the current git provenance for reproducibility metadata."""
    sha = _git_short_sha(cwd=cwd)
    dirty, paths = _git_dirty_status(cwd=cwd)
    return GitProvenance(commit_sha=sha, dirty=dirty, dirty_paths=paths)


def commit_string(prov: GitProvenance) -> str:
    """Human-legible `<sha>` or `<sha>+dirty` for non-JSON channels.

    Used in PDF Keywords, PNG pnginfo `Commit` chunks, and any other flat-string
    context where the JSON `git_dirty` field cannot ride along. A `dirty=None`
    provenance (git-unavailable lane) renders as the bare sha — the JSON
    metadata carries the explicit `null` signal separately.
    """
    if prov.dirty is True:
        return f"{prov.commit_sha}+dirty"
    return prov.commit_sha


def as_metadata_dict(prov: GitProvenance) -> dict[str, object]:
    """Render the provenance as reproducibility-metadata dict fields.

    Consumers `metadata.update(as_metadata_dict(git_provenance()))` into their
    result JSON's `metadata` block. Fields:

    - `git_commit`: str, short SHA (or `"unknown"`).
    - `git_dirty`: bool | None. True/False when checked; None when the check
      could not run (record the explicit sentinel — don't infer clean).
    - `git_dirty_paths`: list[str], present ONLY when `dirty is True`.
    """
    out: dict[str, object] = {
        "git_commit": prov.commit_sha,
        "git_dirty": prov.dirty,
    }
    if prov.dirty is True:
        out["git_dirty_paths"] = list(prov.dirty_paths)
    return out
