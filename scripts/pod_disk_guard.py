#!/usr/bin/env python3
"""Pod-side disk guard: quota probe, stale git-lock clear, intermediate-checkpoint reclaim.

This helper runs ON the pod against the pod's own local filesystem (it never SSHes
out and never touches the task workflow — see CLAUDE.md "Pod-side code NEVER shells
out to scripts/task.py"). It exists to recover a pod that has wedged on the RunPod
MooseFS per-pod disk quota (~130 GB, separate from share-level free space; writes
past it fail with ``OSError errno=122 EDQUOT``). Three jobs:

1. ``report`` — true per-pod writable-bytes headroom via a ``posix_fallocate`` canary
   probe (``shutil.disk_usage`` only sees share-level free, which lies about the
   per-pod quota — see CLAUDE.md gotcha + ``run_experiment_389._assert_disk_headroom``).
2. ``clear-git-lock`` — remove a stale ``.git/index.lock`` left behind when a git op
   died mid-write because the quota was hit. Safe + idempotent.
3. ``reclaim`` — propose (DRY-RUN BY DEFAULT) which intermediate ``*_merged`` checkpoint
   directories can be deleted to free space, with sizes. Actual deletion is opt-in
   behind ``--apply`` and is restricted to INTERMEDIATE merged dirs only (names
   containing ``coupling_merged`` or matching ``*phase1_merged*``). It NEVER deletes a
   final/only merged dir, and it FAILS LOUD on ambiguity rather than guessing.

Pure stdlib on purpose: a quota-wedged pod may not have the project venv usable, so
this must run under a bare ``python3``.

Usage::

    python3 scripts/pod_disk_guard.py report [--min-gb 50] [--root /workspace/...]
    python3 scripts/pod_disk_guard.py clear-git-lock [--repo /workspace/explore-persona-space]
    python3 scripts/pod_disk_guard.py reclaim [--root /workspace] [--apply]
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fnmatch
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Intermediate merged-checkpoint dirs that are safe to delete once the matching
# pre_em_checkpoint / adapter exists (CLAUDE.md: "delete coupling_merged/ after each
# phase if you have the matching pre_em_checkpoint/"). A name is reclaimable iff it
# matches one of these. Anything else (notably a final ``em_merged`` or a lone
# ``*_merged`` that is the only checkpoint) is left untouched.
INTERMEDIATE_SUBSTR = "coupling_merged"
INTERMEDIATE_GLOBS = ("*phase1_merged*",)

GB = 1024**3


def is_intermediate_merged(name: str) -> bool:
    """True iff a directory name denotes an INTERMEDIATE merged checkpoint.

    Intermediate = name contains ``coupling_merged`` OR matches ``*phase1_merged*``.
    A final/only merged dir (e.g. ``em_merged``) returns False so it is never
    proposed for deletion. Returns a plain bool; never raises.
    """
    if INTERMEDIATE_SUBSTR in name:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in INTERMEDIATE_GLOBS)


# ── report ───────────────────────────────────────────────────────────────────


def probe_quota_headroom(root: Path, min_gb: int) -> tuple[bool, float, str]:
    """Probe true per-pod writable headroom at ``root`` via ``posix_fallocate``.

    Reserves ``min_gb`` GB in a temp file under ``root`` (then deletes it) to detect
    the MooseFS per-pod EDQUOT ceiling that ``shutil.disk_usage`` misses. Returns
    ``(ok, share_free_gb, detail)`` where ``ok`` is whether ``min_gb`` GB could be
    reserved. Does NOT raise on EDQUOT/ENOSPC — it reports ``ok=False`` so the CLI can
    print a structured failure and exit non-zero. Re-raises unexpected OSErrors loud.
    """
    assert min_gb > 0, f"min_gb must be positive, got {min_gb}"
    min_bytes = min_gb * GB
    share_free_gb = shutil.disk_usage(str(root)).free / GB

    root.mkdir(parents=True, exist_ok=True)
    probe_path = root / ".pod_disk_guard_probe.tmp"
    fd = None
    try:
        fd = os.open(str(probe_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.posix_fallocate(fd, 0, min_bytes)
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                detail = (
                    f"QUOTA EXHAUSTED: cannot reserve {min_gb} GB under {root} "
                    f"(errno={e.errno} {errno.errorcode.get(e.errno, '?')}). "
                    f"Share-level free reports {share_free_gb:.1f} GB but this pod has "
                    f"hit its per-pod writable-bytes budget. Run "
                    f"`reclaim --apply` or delete intermediate checkpoints."
                )
                return False, share_free_gb, detail
            if e.errno == errno.EOPNOTSUPP:
                # Filesystem doesn't support fallocate (e.g. local tmpfs in tests);
                # fall back to share-level free, which CANNOT see a per-pod quota.
                ok = share_free_gb >= min_gb
                detail = (
                    f"posix_fallocate unsupported on {root}; fell back to "
                    f"shutil.disk_usage (share-level free {share_free_gb:.1f} GB). "
                    f"EDQUOT cannot be detected by the fallback."
                )
                return ok, share_free_gb, detail
            raise
        detail = (
            f"OK: reserved {min_gb} GB probe under {root}; share-level free {share_free_gb:.1f} GB."
        )
        return True, share_free_gb, detail
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            probe_path.unlink()


def cmd_report(args: argparse.Namespace) -> int:
    """Print per-pod disk headroom; exit 0 if >= min_gb reservable, 1 otherwise."""
    root = Path(args.root)
    ok, share_free_gb, detail = probe_quota_headroom(root, args.min_gb)
    print(f"root: {root}")
    print(f"share-level free (shutil): {share_free_gb:.1f} GB")
    print(f"per-pod quota probe ({args.min_gb} GB): {'PASS' if ok else 'FAIL'}")
    print(detail)
    return 0 if ok else 1


# ── clear-git-lock ────────────────────────────────────────────────────────────


def clear_git_lock(repo: Path) -> tuple[bool, str]:
    """Remove a stale ``.git/index.lock`` if present. Safe + idempotent.

    Returns ``(removed, detail)``. ``removed`` is True only when a lock file actually
    existed and was deleted; absence is not an error (idempotent). Raises loudly if the
    repo has no ``.git`` directory (caller passed a non-repo path) — the crash is the
    signal, per FAIL-LOUD.
    """
    git_dir = repo / ".git"
    if not git_dir.exists():
        raise FileNotFoundError(
            f"{repo} is not a git repository (no .git directory); refusing to guess."
        )
    lock_path = git_dir / "index.lock"
    if not lock_path.exists():
        return False, f"no stale lock at {lock_path} (nothing to do)."
    lock_path.unlink()
    return True, f"removed stale git index lock at {lock_path}."


def cmd_clear_git_lock(args: argparse.Namespace) -> int:
    """Clear a stale .git/index.lock; always exit 0 (idempotent, never wedges)."""
    _removed, detail = clear_git_lock(Path(args.repo))
    print(detail)
    return 0


# ── reclaim ───────────────────────────────────────────────────────────────────


@dataclass
class MergedDir:
    """One ``*_merged`` checkpoint directory found under the scan root."""

    path: Path
    size_bytes: int
    intermediate: bool

    @property
    def size_gb(self) -> float:
        """Directory size in GB (float)."""
        return self.size_bytes / GB


def _dir_size_bytes(path: Path) -> int:
    """Sum of regular-file sizes under ``path`` (follows no symlinks). Stdlib only."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for fn in filenames:
            fp = Path(dirpath) / fn
            with contextlib.suppress(OSError):
                if fp.is_symlink():
                    continue
                total += fp.stat().st_size
    return total


def find_merged_dirs(root: Path) -> list[MergedDir]:
    """Find every dir whose name contains ``_merged`` under ``root``, tagged intermediate.

    Discovery is by ``_merged`` substring (not a bare ``endswith``) so suffixed
    intermediate dirs like ``fold_phase1_merged_seed1`` are not silently skipped.
    Sorted by descending size so the biggest reclaim candidate prints first.
    """
    found: list[MergedDir] = []
    for dirpath, dirnames, _filenames in os.walk(root, followlinks=False):
        for d in list(dirnames):
            # Discover any dir that LOOKS like a merged checkpoint: name contains
            # ``_merged`` (catches ``coupling_merged`` / ``em_merged`` / ``phase1_merged``
            # AND suffixed variants like ``fold_phase1_merged_seed1`` that a bare
            # ``endswith("_merged")`` would miss). Intermediacy is decided separately by
            # is_intermediate_merged so the deletion rule never silently skips a dir.
            if "_merged" in d:
                full = Path(dirpath) / d
                found.append(
                    MergedDir(
                        path=full,
                        size_bytes=_dir_size_bytes(full),
                        intermediate=is_intermediate_merged(d),
                    )
                )
    found.sort(key=lambda m: m.size_bytes, reverse=True)
    return found


def cmd_reclaim(args: argparse.Namespace) -> int:
    """Propose (default) or delete (--apply) INTERMEDIATE merged checkpoint dirs.

    Dry-run by default. ``--apply`` deletes ONLY dirs whose name marks them as
    intermediate (``coupling_merged`` substring or ``*phase1_merged*`` glob). Final /
    only merged dirs (e.g. ``em_merged``) are listed as KEEP and never touched.
    """
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"scan root {root} does not exist")

    merged = find_merged_dirs(root)
    reclaimable = [m for m in merged if m.intermediate]
    keep = [m for m in merged if not m.intermediate]

    print(f"scan root: {root}")
    print(
        f"found {len(merged)} *_merged dirs "
        f"({len(reclaimable)} intermediate / {len(keep)} final-or-only)"
    )
    print()
    for m in reclaimable:
        print(f"  RECLAIM  {m.size_gb:7.2f} GB  {m.path}")
    for m in keep:
        print(f"  KEEP     {m.size_gb:7.2f} GB  {m.path}  (final/only merged — never deleted)")

    total_gb = sum(m.size_gb for m in reclaimable)
    print()
    if not reclaimable:
        print("nothing reclaimable (no intermediate merged dirs).")
        return 0

    if not args.apply:
        print(
            f"DRY RUN: would reclaim {total_gb:.2f} GB across {len(reclaimable)} dir(s). "
            f"Re-run with --apply to delete."
        )
        return 0

    print(f"--apply: deleting {len(reclaimable)} intermediate dir(s) ({total_gb:.2f} GB)...")
    for m in reclaimable:
        # Defense in depth: re-check intermediacy at deletion time so a refactor of
        # find_merged_dirs can never widen what --apply removes. Fail loud on drift.
        if not is_intermediate_merged(m.path.name):
            raise RuntimeError(
                f"refusing to delete non-intermediate dir {m.path} — "
                f"name does not match intermediate rule (this is a bug)"
            )
        shutil.rmtree(m.path)
        print(f"  deleted {m.size_gb:7.2f} GB  {m.path}")
    print(f"reclaimed {total_gb:.2f} GB.")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the pod_disk_guard argument parser (report / clear-git-lock / reclaim)."""
    parser = argparse.ArgumentParser(
        prog="pod_disk_guard",
        description="Pod-side disk guard: quota probe, stale git-lock clear, "
        "intermediate-checkpoint reclaim.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="report true per-pod disk headroom")
    p_report.add_argument("--root", default="/workspace", help="path to probe (default /workspace)")
    p_report.add_argument(
        "--min-gb", type=int, default=50, help="GB of headroom to probe for (default 50)"
    )
    p_report.set_defaults(func=cmd_report)

    p_lock = sub.add_parser("clear-git-lock", help="remove a stale .git/index.lock")
    p_lock.add_argument(
        "--repo",
        default="/workspace/explore-persona-space",
        help="repo whose .git/index.lock to clear",
    )
    p_lock.set_defaults(func=cmd_clear_git_lock)

    p_reclaim = sub.add_parser(
        "reclaim", help="propose (dry-run default) intermediate merged-checkpoint deletions"
    )
    p_reclaim.add_argument("--root", default="/workspace", help="scan root (default /workspace)")
    p_reclaim.add_argument(
        "--apply",
        action="store_true",
        help="actually delete intermediate merged dirs (default: dry-run only)",
    )
    p_reclaim.set_defaults(func=cmd_reclaim)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args and dispatch to the chosen subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
