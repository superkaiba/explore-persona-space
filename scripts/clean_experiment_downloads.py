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

**Incremental (within-run, between-phase) cleanup.** Step-8 cleanup only fires
at experiment END, so a multi-phase experiment whose phases each materialize a
fresh download cache (phase-1 downloads ``g1_dl``, phase-2 ``g2_dl``, ...) holds
the PEAK of all phases' caches at once. When an experiment's footprint is too
big for the VM disk (incident 2026-06-26: #658's Phase-1 analysis put a 139 GB
store on the VM worktree on a 188 GB fleet-shared disk), that peak can fill the
root disk mid-run. ``clean_issue_downloads`` is deliberately phase-agnostic —
it reaps the SAME ``hf_dl`` / ``g*_dl`` re-downloadable caches under the SAME
keep/delete contract whether called once at the end or after each phase. The
``--incremental`` CLI flag (and the ``clean_issue_downloads_incremental`` thin
wrapper) document the between-phase use: call it after a phase's judge /
extraction step has CONSUMED its download inputs, BEFORE the next phase
downloads more, to bound peak footprint rather than only cleaning at the end.
The safety contract is identical (``store/`` + ``eval_results/`` NEVER touched,
re-downloadable caches only) — the cache is rebuilt on demand if a later phase
needs it again, so reaping a consumed phase's cache mid-run is safe. The
``vm_disk_guard`` fleet backstop's terminal-status gate (``--apply`` only on
``completed`` / ``archived`` / ``awaiting_promotion`` issues) does NOT cover
the active-issue case; the incremental entry point is the experiment's OWN
deliberate self-cleanup while it runs, so it intentionally has no
terminal-status check — the experiment knows the phase is done.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from explore_persona_space.task_workflow import repo_root

# The two cache-dir glob patterns under data/issue_<N>/ that are
# re-downloadable and therefore safe to delete. Everything else under the
# per-issue data dir (notably ``store/``) is KEPT. ``hf_dl`` is an exact name;
# ``g*_dl`` matches ``g1_dl`` / ``g2_dl`` / ... (the group-download buckets).
CACHE_DIR_GLOBS = ("hf_dl", "g*_dl")

# The HF dataset repo a per-issue ``store/`` would have been mirrored to. Used
# ONLY by the defensive nested-``store/`` parity guard below to verify a
# generated (NOT re-downloadable) store tree is present on HF before a wholesale
# ``rmtree(hf_dl)`` would destroy it. Env-overridable for tests / repo moves.
HF_DATA_REPO_DEFAULT = "superkaiba1/explore-persona-space-data"


def hf_data_repo() -> str:
    """The data repo the nested-``store/`` parity guard checks against
    (env ``EPM_HF_DATA_REPO``; defaults to :data:`HF_DATA_REPO_DEFAULT`)."""
    return os.environ.get("EPM_HF_DATA_REPO", "").strip() or HF_DATA_REPO_DEFAULT


# Shared sidecar stream for ALL VM-disk escalations (this guard's SKIP events,
# vm_disk_guard's active-task escalations, the watcher's sub-floor sentinel) —
# one queryable trace beyond the rotating cron logs. Relative to the repo root.
DISK_GUARD_SIDECAR_REL = Path(".claude") / "cache" / "disk-guard-events.jsonl"


def disk_guard_sidecar_path() -> Path:
    """Absolute path of the shared disk-guard escalation sidecar JSONL."""
    return repo_root() / DISK_GUARD_SIDECAR_REL


def append_disk_guard_event(event: dict, *, apply: bool = True) -> None:
    """Append one JSON line to the shared disk-guard sidecar (fail-soft).

    Used by every VM-disk escalation path so all disk events share one stream.
    A ``ts`` is stamped if the caller did not supply one. The parent dir is
    created idempotently. A write failure is logged loudly but NEVER raises —
    the sidecar is observability, and losing one escalation row must not crash
    the cleanup / guard pass that emits it. ``apply=False`` reports only."""
    row = {"ts": datetime.now().astimezone().isoformat(), **event}
    line = json.dumps(row)
    if not apply:
        print(f"  [report-only] would append disk-guard event: {line[:160]}", file=sys.stderr)
        return
    dest = disk_guard_sidecar_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  WARNING: appending disk-guard event failed: {exc}", file=sys.stderr)


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


def _nested_store_dirs(cache_dir: Path) -> list[Path]:
    """Any ``store/`` subtree NESTED under a re-downloadable ``hf_dl`` /
    ``g*_dl`` cache dir about to be wholesale-deleted.

    A ``store/`` directory holds GENERATED (NOT re-downloadable) artifacts and
    normally lives as a SIBLING of the cache dirs (the cleaner's keep/delete
    contract keeps ``store/`` and only touches the cache globs). But a
    mis-rooted run can write a ``store/`` UNDER the download cache dir, where a
    wholesale ``shutil.rmtree(cache_dir)`` would silently destroy it. This
    finds every such nested ``store/`` so the parity guard can refuse the reap
    unless the generated data is verifiably preserved on HF."""
    out: list[Path] = []
    for p in cache_dir.rglob("store"):
        if p.is_dir() and not p.is_symlink():
            out.append(p)
    return out


def _store_files_with_sizes(store_dir: Path) -> dict[str, int]:
    """``{relative_posix_path: size_bytes}`` for every file under ``store_dir``.

    Keyed by path relative to ``store_dir`` so the on-HF comparison is by the
    store-internal layout, not the absolute VM path. A stat error on one file
    is recorded as size -1 (an impossible match) so the file fails the parity
    check rather than being silently skipped — fail-toward-keep."""
    out: dict[str, int] = {}
    for p in sorted(store_dir.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        out[p.relative_to(store_dir).as_posix()] = size
    return out


def _hf_file_sizes(repo_id: str, revision: str = "main") -> dict[str, int] | None:
    """``{path_in_repo: size_bytes}`` for the data repo, or ``None`` on ANY
    failure (missing token, network error, unknown revision, import error).

    Revision-pinned (defaults to ``main``) so the parity check reads a stable
    snapshot. ``None`` is the fail-toward-keep signal: the caller must NOT
    delete generated data it could not positively confirm is mirrored."""
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFile

        api = HfApi(token=token)
        sizes: dict[str, int] = {}
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            recursive=True,
        ):
            if isinstance(entry, RepoFile):
                size = getattr(entry, "size", None)
                if isinstance(size, int):
                    sizes[entry.path] = size
        return sizes
    except Exception as exc:
        print(
            f"  ! nested-store parity: HF listing for {repo_id}@{revision} failed "
            f"({type(exc).__name__}: {exc}); fail-toward-keep",
            file=sys.stderr,
        )
        return None


def _local_file_is_mirrored(rel: str, size: int, hf_sizes: dict[str, int]) -> bool:
    """PATH-FAITHFUL per-file mirror check (fail-toward-keep on any ambiguity).

    ``rel`` is a local store file's POSIX path relative to its ``store/`` dir
    (e.g. ``runA/result.pt``). The data repo mirrors a store as
    ``issue<N>_<slug>/store/<rel>`` (verified against the live repo layout) —
    ALWAYS rooted at a real ``store/`` directory component. The IDENTITY-
    preserving anchor is therefore a ``store/<rel>`` match where the ``store``
    segment is itself a complete path component, i.e. the HF path is EXACTLY
    ``store/<rel>`` (store at repo root) OR ends in ``/store/<rel>`` (store
    under a parent dir such as ``issue<N>_<slug>/``). It must hold at the SAME
    size.

    Two narrower matches that an earlier revision used are deliberately GONE
    (the #679 component-boundary BLOCKER): (1) a bare ``/<rel>`` suffix —
    an unrelated HF ``unrelated/runA/result.pt`` (or, worse, ANY HF
    ``*/result.pt`` for a single-segment ``rel``) at the same size would
    falsely license ``rmtree(hf_dl)`` to delete non-re-downloadable data;
    (2) an unbounded ``store/<rel>`` ``endswith`` — ``issue/notstore/runA/...``
    would match ``store/runA/...`` because ``notstore`` ends in ``store``.
    Requiring ``store`` to be a full component (start-of-path or after a ``/``)
    closes both holes. Because every legitimate mirror is rooted at a real
    ``store/`` component, the component-anchored match succeeds for every true
    mirror, so dropping the looser matches loses no true positives.

    ``size < 0`` (a local stat error) can never match a real HF size => keep."""
    if size < 0:
        return False
    store_root = f"store/{rel}"  # store at the repo root: store/<rel>
    store_anchored = f"/store/{rel}"  # store under a parent dir: .../store/<rel>
    for hf_path, hf_size in hf_sizes.items():
        if hf_size != size:
            continue
        if hf_path == store_root or hf_path.endswith(store_anchored):
            return True
    return False


def nested_store_is_mirrored(
    store_dir: Path,
    hf_sizes: dict[str, int] | None,
) -> bool:
    """True only if EVERY file under ``store_dir`` is verifiably present on HF
    at a MATCHING size via a PATH-FAITHFUL match (a per-file match, NOT a
    size-SUM — a sum can coincide while individual files differ).

    ``hf_sizes`` of ``None`` (any HF-listing failure) is fail-toward-keep =>
    returns False. A local file whose size is -1 (stat error) can never match a
    real HF size, so it also fails the check. Matching is by the IDENTITY-
    preserving ``store/``-COMPONENT-anchored path match (see
    ``_local_file_is_mirrored``) — NOT by basename and NOT by an unanchored
    suffix, so neither an unrelated same-name-same-size HF file nor a
    ``notstore/``-prefixed path can license deleting generated data (#679
    BLOCKER #2 + the component-boundary residual)."""
    if hf_sizes is None:
        return False
    local = _store_files_with_sizes(store_dir)
    if not local:
        # An empty nested store has nothing to lose — safe to reap.
        return True
    return all(_local_file_is_mirrored(rel, size, hf_sizes) for rel, size in local.items())


def _cache_dir_reap_blocked(
    cache_dir: Path,
    *,
    issue_n: int,
    apply: bool,
    hf_sizes_cache: dict[str, dict[str, int] | None],
) -> str | None:
    """Return a SKIP reason if a wholesale ``rmtree(cache_dir)`` would destroy a
    nested ``store/`` not verifiably mirrored on HF; ``None`` to allow the reap.

    The HF listing is fetched at most once per process (cached in
    ``hf_sizes_cache``) so a multi-cache-dir issue makes a single Hub call. On
    a SKIP, an escalation row is appended to the shared disk-guard sidecar."""
    nested = _nested_store_dirs(cache_dir)
    if not nested:
        return None  # no generated data at risk — normal re-downloadable reap
    repo = hf_data_repo()
    if repo not in hf_sizes_cache:
        hf_sizes_cache[repo] = _hf_file_sizes(repo)
    hf_sizes = hf_sizes_cache[repo]
    unmirrored = [s for s in nested if not nested_store_is_mirrored(s, hf_sizes)]
    if not unmirrored:
        return None  # every nested store is verifiably on HF — safe to reap
    rel = _rel_name(cache_dir)
    paths = ", ".join(_rel_name(s) for s in unmirrored)
    reason = (
        f"nested store/ not verifiably mirrored on HF ({repo}): {paths} — "
        f"wholesale rmtree({rel}) would destroy generated data; KEPT"
    )
    append_disk_guard_event(
        {
            "kind": "nested-store-reap-skipped",
            "task": issue_n,
            "path": rel,
            "nested_stores": [_rel_name(s) for s in unmirrored],
            "hf_repo": repo,
            "reason": reason,
        },
        apply=apply,
    )
    return reason


@dataclass
class CleanResult:
    issue_n: int
    apply: bool
    removed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    # Cache dirs deliberately KEPT by the nested-``store/`` parity guard (a
    # wholesale reap would have destroyed generated data not verifiably on HF).
    # Each entry is ``(rel_name, reason)``; an escalation row is also sidecar-
    # logged. These are NOT failures — they are a safe fail-toward-keep.
    skipped: list[tuple[str, str]] = field(default_factory=list)
    sizes_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def bytes_freed(self) -> int:
        """Total bytes of the directories removed (or that would be removed).

        Excludes parity-SKIPPED caches (they are kept, so they free nothing) —
        size an *escalation* via :pyattr:`total_discovered_bytes` instead."""
        return sum(self.sizes_bytes.get(name, 0) for name in self.removed)

    @property
    def total_discovered_bytes(self) -> int:
        """Total bytes of EVERY cache dir traversed, regardless of reap fate
        (removed AND parity-skipped AND failed).

        ``sizes_bytes`` is populated the moment each cache dir is discovered —
        before the reap-vs-skip decision — so this is the footprint of all the
        re-downloadable cache an issue holds. The active-task escalation MUST
        size from this, not :pyattr:`bytes_freed`: a large active
        ``hf_dl/.../store/`` correctly KEPT by the nested-store parity guard
        contributes 0 to ``bytes_freed`` (it is in ``skipped``, not
        ``removed``), which would silently suppress the escalation for the
        exact large-unmirrored-active-cache shape #679 targets (BLOCKER #1)."""
        return sum(self.sizes_bytes.values())


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
    # Cache the HF listing across cache dirs so a multi-cache-dir issue makes at
    # most one Hub call regardless of how many nested store/ checks run.
    hf_sizes_cache: dict[str, dict[str, int] | None] = {}
    for cache_dir in download_cache_dirs(issue_n, data_root):
        rel = _rel_name(cache_dir)
        res.sizes_bytes[rel] = _dir_size_bytes(cache_dir)
        # Defensive parity guard: a wholesale rmtree(cache_dir) would destroy a
        # nested store/ (generated, NOT re-downloadable). Refuse unless every
        # nested store file is verifiably mirrored on HF (fail-toward-keep). The
        # check runs in BOTH dry-run (reports the would-skip) and apply mode.
        skip_reason = _cache_dir_reap_blocked(
            cache_dir, issue_n=issue_n, apply=apply, hf_sizes_cache=hf_sizes_cache
        )
        if skip_reason is not None:
            print(f"  ~ SKIP {rel}: {skip_reason}", file=sys.stderr)
            res.skipped.append((rel, skip_reason))
            continue
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


def clean_issue_downloads_incremental(
    issue_n: int,
    *,
    apply: bool = False,
    data_root: Path | None = None,
) -> CleanResult:
    """Between-phase cleanup of ``issue_n``'s consumed download caches (within-run).

    Identical behavior + safety contract to ``clean_issue_downloads`` — this is a
    thin, explicitly-named alias for the INCREMENTAL use case: an experiment
    calls it after a phase's judge / extraction step has consumed its
    ``hf_dl`` / ``g*_dl`` download inputs and BEFORE the next phase downloads
    more, to bound peak VM-disk footprint rather than only cleaning at
    experiment end (Step 8). Unlike the ``vm_disk_guard`` fleet backstop, there
    is NO terminal-status gate: the calling experiment is itself the authority
    that the phase is done, so an ACTIVE issue self-reaping its own consumed
    cache mid-run is the intended path. ``store/`` + ``eval_results/`` are never
    touched; the re-downloadable cache is rebuilt on demand if a later phase
    needs it again."""
    return clean_issue_downloads(issue_n, apply=apply, data_root=data_root)


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
    ap.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Label this as a within-run between-phase cleanup (after a phase "
            "consumed its download inputs, before the next phase downloads "
            "more). Behavior + safety contract are identical to the default "
            "(end-of-run) cleanup; the flag only documents intent in the "
            "report line. No terminal-status gate — the experiment knows the "
            "phase is done."
        ),
    )
    args = ap.parse_args(argv)

    cleaner = clean_issue_downloads_incremental if args.incremental else clean_issue_downloads
    res = cleaner(args.issue, apply=args.apply)
    mode = "incremental " if args.incremental else ""
    verb = "removed" if args.apply else "would remove"
    print(
        f"clean_experiment_downloads {mode}issue {args.issue}: {verb} "
        f"{len(res.removed)} cache dir(s), {_fmt_gb(res.bytes_freed)} | "
        f"skipped {len(res.skipped)} | failed {len(res.failed)}"
    )
    for name in res.removed:
        print(f"  - {verb}: {name} [{_fmt_gb(res.sizes_bytes.get(name, 0))}]")
    for name, reason in res.skipped:
        print(f"  ~ SKIP (kept): {name} — {reason}")
    for name in res.failed:
        print(f"  ! FAILED: {name}")
    if not res.removed and not res.failed and not res.skipped:
        print("  (no download caches found — nothing to do)")
    return 2 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
