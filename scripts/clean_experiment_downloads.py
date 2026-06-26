#!/usr/bin/env python3
"""Delete a finished experiment's HF-download caches under ``data/issue_<N>/``.

Each experiment downloads its source data from HF into per-issue cache
directories (``data/issue_<N>/hf_dl/`` and ``data/issue_<N>/g*_dl/`` — the
``g1_dl`` / ``g2_dl`` group-download buckets). NOTHING ever cleans them, so a
single finished experiment can pin ~100 GB of re-downloadable cache on the VM
root disk (incident 2026-06-25: ``/`` hit 100% full, one finished experiment
held 97 GB). These directories are CACHES — the data is on HF and re-downloads
on the next run — so deletion is safe and needs NO on-HF presence check.

What is and is NOT a cache (the safety contract):
  * ``data/issue_<N>/hf_dl/``  — DELETE (re-downloadable HF cache)
  * ``data/issue_<N>/g*_dl/``  — DELETE (re-downloadable group-download cache)
  * ``data/issue_<N>/store/``  — KEEP (generated, not re-downloadable)
  * ``eval_results/``          — KEEP (the durable result artifacts)
  * anything else under ``data/issue_<N>/`` — KEEP (only the two cache globs
    are ever touched).

The ``data/`` tree uses two naming conventions for the same N — ``issue_<N>``
(underscore) AND ``issue<N>`` (no underscore, sometimes with a ``_<slug>``
suffix, e.g. ``issue295_marker_only_loss``). Both forms are matched so a cache
is never silently missed for being on the other side of the underscore.

Idempotent (a missing cache is a no-op) and DRY-RUN BY DEFAULT — ``--apply``
gates all deletion. The library functions are importable by
``scripts/vm_disk_guard.py`` (its tier-(b) cleanup) and wired into the
``/issue`` Step 8 post-experiment teardown.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.task_workflow import repo_root

# The two cache-dir glob patterns under data/issue_<N>/ that are
# re-downloadable and therefore safe to delete. Everything else under the
# per-issue data dir (notably ``store/``) is KEPT. ``hf_dl`` is an exact name;
# ``g*_dl`` matches ``g1_dl`` / ``g2_dl`` / ... (the group-download buckets).
CACHE_DIR_GLOBS = ("hf_dl", "g*_dl")


def _data_root() -> Path:
    """Absolute path of the repo's ``data/`` directory."""
    return repo_root() / "data"


def _worktree_data_roots(issue_n: int) -> list[Path]:
    """``data/`` directories inside this issue's worktree(s).

    The live experiment's download/store data often lives in the WORKTREE,
    not the repo root — e.g. ``.claude/worktrees/issue-658/data/issue_658/``
    (the worktrees tree was 139 GB on 2026-06-26, dominated by per-issue
    worktree data). A `/issue` run can have ``issue-<N>`` AND
    ``issue-<N>-<suffix>`` (same-issue follow-up round) worktrees, so every
    ``issue-<N>*`` worktree whose name maps to exactly ``issue_n`` is
    scanned. Returns only existing ``<worktree>/data`` dirs."""
    wt_root = repo_root() / ".claude" / "worktrees"
    if not wt_root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(wt_root.iterdir()):
        if not child.is_dir():
            continue
        # issue-<N> or issue-<N>-<suffix> (the N boundary pinned by the
        # exact name / trailing-dash prefix, so issue-65 never matches
        # issue-658).
        name = child.name
        if name == f"issue-{issue_n}" or name.startswith(f"issue-{issue_n}-"):
            data_dir = child / "data"
            if data_dir.is_dir():
                out.append(data_dir)
    return out


def _resolve_data_roots(issue_n: int, data_root: Path | None) -> list[Path]:
    """Every ``data/`` root to search for ``issue_n``'s caches.

    When ``data_root`` is given (tests / explicit scoping) it is the SOLE
    root. Otherwise the search spans the repo-root ``data/`` AND every
    worktree ``data/`` for the issue — the worktree copies are where the
    live experiment actually writes (coordinator evidence, #658)."""
    if data_root is not None:
        return [data_root]
    return [_data_root(), *_worktree_data_roots(issue_n)]


def issue_data_dirs(issue_n: int, data_root: Path | None = None) -> list[Path]:
    """Per-issue data directories for ``issue_n`` across the resolved root(s).

    Returns every existing directory whose name is ``issue_<N>`` or
    ``issue<N>`` or ``issue<N>_<slug>`` (the two real naming conventions in
    ``data/``), under the repo-root ``data/`` AND every worktree ``data/``
    for the issue. The N boundary is matched exactly so ``issue_65`` never
    picks up ``issue_658``. ``data_root`` (when given) scopes the search to
    that single root — used by tests pointing at a temp filesystem.
    """
    n = str(issue_n)
    out: list[Path] = []
    for root in _resolve_data_roots(issue_n, data_root):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if name in (f"issue_{n}", f"issue{n}"):
                out.append(child)
            elif name.startswith(f"issue_{n}_") or name.startswith(f"issue{n}_"):
                # issue<N>_<slug> — but NOT issue<M>_... where M just starts
                # with N (the trailing underscore pins the N boundary, so
                # "issue65_" never matches "issue658..." — no underscore there).
                out.append(child)
    return out


def download_cache_dirs(issue_n: int, data_root: Path | None = None) -> list[Path]:
    """Re-downloadable cache directories to delete for ``issue_n``.

    The union of ``CACHE_DIR_GLOBS`` matches across every per-issue data dir
    (both naming conventions, repo-root AND worktree copies). Only existing
    directories are returned; ``store/`` and any non-cache content are never
    included.
    """
    out: list[Path] = []
    for issue_dir in issue_data_dirs(issue_n, data_root):
        for pattern in CACHE_DIR_GLOBS:
            for match in sorted(issue_dir.glob(pattern)):
                if match.is_dir():
                    out.append(match)
    return out


def _dir_size_bytes(path: Path) -> int:
    """Recursive on-disk size of ``path`` in bytes (best-effort; a stat error
    on a single entry is skipped, never raised — sizing is reporting only)."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


@dataclass
class CleanResult:
    issue_n: int
    apply: bool
    removed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    sizes_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def bytes_freed(self) -> int:
        """Total bytes of the directories removed (or that would be removed)."""
        return sum(self.sizes_bytes.get(name, 0) for name in self.removed)


def clean_issue_downloads(
    issue_n: int,
    *,
    apply: bool = False,
    data_root: Path | None = None,
) -> CleanResult:
    """Delete (``apply=True``) or report (default) ``issue_n``'s download caches.

    Idempotent: an absent cache contributes nothing. ``store/`` and
    non-cache content are never touched — only the ``CACHE_DIR_GLOBS`` matches
    under the per-issue data dir(s) are removed. A removal that raises is
    recorded in ``failed`` and never aborts the rest (fail-soft per directory,
    fail-loud in the report).
    """
    res = CleanResult(issue_n=issue_n, apply=apply)
    for cache_dir in download_cache_dirs(issue_n, data_root):
        rel = _rel_name(cache_dir)
        res.sizes_bytes[rel] = _dir_size_bytes(cache_dir)
        if not apply:
            res.removed.append(rel)  # would-remove (dry-run)
            continue
        try:
            shutil.rmtree(cache_dir)
        except OSError as exc:
            print(f"  ! FAILED to remove {rel}: {exc}", file=sys.stderr)
            res.failed.append(rel)
            continue
        res.removed.append(rel)
    return res


def _rel_name(path: Path) -> str:
    """Path relative to the repo root for display (falls back to absolute)."""
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def _fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f}G"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Delete a finished experiment's HF-download caches "
            "(data/issue_<N>/hf_dl + g*_dl). Re-downloadable; store/ + "
            "eval_results/ are never touched. Dry-run by default."
        )
    )
    ap.add_argument("issue", type=int, help="Issue / task number N.")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (default: dry-run, report what would be removed).",
    )
    args = ap.parse_args(argv)

    res = clean_issue_downloads(args.issue, apply=args.apply)
    verb = "removed" if args.apply else "would remove"
    print(
        f"clean_experiment_downloads issue {args.issue}: {verb} "
        f"{len(res.removed)} cache dir(s), {_fmt_gb(res.bytes_freed)} | "
        f"failed {len(res.failed)}"
    )
    for name in res.removed:
        print(f"  - {verb}: {name} [{_fmt_gb(res.sizes_bytes.get(name, 0))}]")
    for name in res.failed:
        print(f"  ! FAILED: {name}")
    if not res.removed and not res.failed:
        print("  (no download caches found — nothing to do)")
    return 2 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
