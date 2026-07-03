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
  * EXCEPTION (the active-consumer gate, #773): a ``hf_dl`` / ``g*_dl`` cache
    that would otherwise be deleted is KEPT (never deleted) while a DIFFERENT,
    currently-ACTIVE task declares ``data/issue_<N>/`` as a planned input in
    its ``plans/plan.md`` or ``body.md``. The reap is skipped + sidecar-logged
    (``kind: "active-consumer-reap-skipped"``), fail-toward-keep. This guards
    against the cross-issue strand-the-consumer failure mode (#742 died on a
    ``FileNotFoundError`` after ``#658``'s caches were panic-reaped while it
    was reading ``data/issue_658/store/v0_summaries.pt``, a symlink whose
    target lived under ``data/issue_658/hf_dl/``). See
    ``_active_consumer_protected_issues`` /
    ``_cache_dir_reap_blocked_by_active_consumer``.
  * SYMLINKED caches (#915, the #681 data-disk relocation): a ``hf_dl`` /
    ``g*_dl`` that is a symlink — or whose ``issue_<N>`` PARENT dir is a
    symlink — is disposition-checked instead of plain-``rmtree``'d
    (``shutil.rmtree`` refuses a symlink by design). The RESOLVED target is
    reaped ONLY when it lives strictly inside the managed data-disk root
    (env ``EPS_VM_DATA_DISK_PATH``, default ``/mnt/eps-data``), a path
    component names the OWNING issue, AND it is a directory whose basename
    equals the cache name (the relocation is name-preserving). Anything else
    — external, foreign-issue, renamed, a ``store/``, a file — is KEPT
    (fail-toward-keep, sidecar-escalated as
    ``kind: "symlink-external-target-kept"``): a direct link is unlinked
    (target kept); with a symlinked PARENT nothing is EVER unlinked — not the
    shared parent link, and not a child entry inside its target tree (even
    when that child is itself a link — the double-link case; the only
    permitted mutation under a linked parent is the managed-target reap). A
    DIRECT dangling cache symlink is discovered and unlinked; a dangling
    entry reached through a linked parent is kept + sidecar-escalated.
    Managed reaps delete the TARGET FIRST, then the link — a mid-reap crash
    leaves the link, so the next run re-discovers and retries.

  * NON-CANONICAL issue-keyed caches (#911) — ALSO swept, under a stricter
    contract: top-level ``/tmp`` dirs named ``i<N>*`` / ``issue<N>*`` /
    ``issue-<N>*`` / ``issue_<N>*`` or ``*_<N>`` (P1/P2 — dirs/symlinks only,
    uid-owned, never recursive; P2 requires the underscore so ``tmux-1000``
    never matches), and whole-dir ``data/`` caches named
    ``issue…<N>…{_dl,_hfstage,_cache}`` (P3). The /tmp part is STRICTLY
    OPT-IN: only the two CLI ``main()`` entry points pass
    ``tmp_root=production_tmp_root()``; every library call with
    ``tmp_root=None`` is hermetic by construction. A whole-dir non-canonical
    candidate is deleted ONLY when it is >48h quiet (max of mtime/atime),
    holds NO nested ``store/``/``eval_results/``, AND carries positive
    re-downloadability evidence (hub-layout markers, or every top-level name
    verified as a prefix on the HF data repo); anything else is kept +
    sidecar-escalated. ``EPM_SKIP_NONCANONICAL_CACHE_SWEEP=1`` is the
    emergency kill switch (the sweep returns no candidates).

There are now SIX reap gates, all composing additively (any one puts a cache
dir in ``CleanResult.skipped`` instead of deleting it):
  1. The TERMINAL-status gate in ``vm_disk_guard.py`` (the OWNING issue must be
     at a terminal-for-reap status before its caches are reaped at all).
  2. The active-CONSUMER gate (#773, this module): never reap while a DIFFERENT
     active task declares ``data/issue_<N>/`` as a planned input. Checked FIRST
     in the per-cache-dir loop (a cheap in-memory set lookup; short-circuiting
     on a consumer hit also avoids the parity guard's potential HF call).
  3. The nested-``store/`` parity gate (#679, below): never wholesale-rmtree a
     cache dir holding a mis-rooted ``store/`` not verifiably mirrored on HF
     (canonical candidates; gate 5 subsumes it for non-canonical ones).
  4. The RECENCY gate (#911, non-canonical only): a candidate touched within
     the 48h window (env ``EPS_NONCANONICAL_CACHE_MIN_AGE_HOURS``) is kept —
     a live reader/writer may hold it.
  5. The NESTED-DURABLE gate (#911, non-canonical only): a nested ``store/``
     OR ``eval_results/`` blocks the whole-dir reap outright (no mirror check
     attempted — durable results never ride a whole-dir rmtree).
  6. The POSITIVE-EVIDENCE gate (#911, non-canonical only): the reap license —
     hub-layout markers or data-repo-prefix mirror verification; a predicate
     failure escalates (sidecar), never deletes.

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

**Off-main graceful degrade (#924).** On a NON-``main`` remote checkout (the
GCE/pod ``issue-<N>`` clone lanes), ``task_workflow.repo_root()`` either raises
its branch-guard ``RuntimeError`` (a ``--depth 1 --branch`` clone with no local
``main`` — the #841 att-6 crash) or silently routes reads to a pin worktree
whose ``data/`` is empty — both defeat the between-phase cleanup exactly where
disk pressure bites. Path resolution therefore goes through
``_resolution_root()``: the probe ``_off_main_checkout_root()`` classifies the
checkout once per process, and on an off-main NON-shared-VM checkout the
``data/`` root, the worktree scan root, the sidecar, and display names resolve
against the checkout itself. The cross-task active-consumer gate (#773) is
still ATTEMPTED there — only a ``RuntimeError`` from the read (the genuinely
unserviceable no-local-``main`` shape) is caught, with a loud WARN + sidecar
row (``kind: "off-main-consumer-gate-skipped"``); a succeeding read (full clone
with a local ``main``) keeps the gate ON. The shared VM never degrades, and VM
main-checkout behavior is byte-identical (the probe returns ``None`` there).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import is_shared_vm_env
from explore_persona_space.task_workflow import (
    STATUSES,
    list_by_status,
    primary_checkout_root,
    repo_root,
    tasks_dir,
)

# The two cache-dir glob patterns under data/issue_<N>/ that are
# re-downloadable and therefore safe to delete. Everything else under the
# per-issue data dir (notably ``store/``) is KEPT. ``hf_dl`` is an exact name;
# ``g*_dl`` matches ``g1_dl`` / ``g2_dl`` / ... (the group-download buckets).
CACHE_DIR_GLOBS = ("hf_dl", "g*_dl")

# Statuses whose work is DONE or paused — a task at one of these does NOT
# actively consume its declared inputs, so it never protects another issue's
# cache (the active-CONSUMER gate, #773). The "active" set this guard cares
# about is ``set(STATUSES) - _CONSUMER_INACTIVE_STATUSES``.
#   - completed / archived : terminal, work finished.
#   - on_hold              : explicitly parked, excluded from auto-dispatch.
#   - blocked              : halted awaiting user; not actively reading inputs.
# Everything ELSE (proposed / planning / plan_pending / approved / running /
# verifying / interpreting / reviewing / awaiting_promotion /
# followups_running) is "active": currently doing work or imminently planned
# to, so its declared inputs may be read at any moment.
_CONSUMER_INACTIVE_STATUSES = frozenset({"completed", "archived", "on_hold", "blocked"})

# Match ``data/issue_<M>/`` (underscore) OR ``data/issue<M>/`` (no underscore) —
# the two real naming conventions in ``data/``. The trailing lookahead pins the
# M boundary to a ``/`` or ``_`` so ``data/issue65/`` never matches a
# ``data/issue658/`` substring and ``data/issue658_slug/`` still matches 658.
_DATA_ISSUE_REF = re.compile(r"\bdata/issue_?(\d+)(?=[/_])")

# ─── non-canonical issue-keyed caches (#911) ─────────────────────────────────
# Runs also stage re-downloadable HF-mirror caches OUTSIDE the canonical
# ``data/issue_<N>/{hf_dl,g*_dl}`` layout: top-level ``/tmp`` dirs named
# ``i<N>_*`` / ``issue<N>*`` / ``issue-<N>*`` / ``issue_<N>*`` (P1) or
# ``*_<N>`` (P2, underscore-only + name-final so ``tmux-1000`` never matches),
# and whole-dir ``data/`` caches named ``issue…<N>…{_dl,_hfstage,_cache}``
# (P3). These escaped both janitors until the boot disk hit 95% with ~90 GB
# of invisible caches (incident 2026-07-02). P1 is tried before P2
# (``i653_766_fixed`` extracts 653, not 766).
_TMP_ISSUE_PREFIX_RE = re.compile(r"^(?:i|issue[-_]?)(\d+)(?:[._-]|$)")
_TMP_ISSUE_SUFFIX_RE = re.compile(r"_(\d+)$")  # underscore ONLY (never `tmux-1000`)
_DATA_NONCANONICAL_CACHE_RE = re.compile(r"^issue_?(\d+)\w*(?:_dl\d*|_hfstage|_cache)$")
TMP_CACHE_ROOT_DEFAULT = "/tmp"  # env EPM_TMP_CACHE_ROOT (main()-only opt-in)
NONCANONICAL_MIN_AGE_HOURS_DEFAULT = 48.0  # env EPS_NONCANONICAL_CACHE_MIN_AGE_HOURS
NONCANONICAL_SWEEP_KILL_ENV = "EPM_SKIP_NONCANONICAL_CACHE_SWEEP"

# The HF dataset repo a per-issue ``store/`` would have been mirrored to. Used
# ONLY by the defensive nested-``store/`` parity guard below to verify a
# generated (NOT re-downloadable) store tree is present on HF before a wholesale
# ``rmtree(hf_dl)`` would destroy it. Env-overridable for tests / repo moves.
HF_DATA_REPO_DEFAULT = "superkaiba1/explore-persona-space-data"


def hf_data_repo() -> str:
    """The data repo the nested-``store/`` parity guard checks against
    (env ``EPM_HF_DATA_REPO``; defaults to :data:`HF_DATA_REPO_DEFAULT`)."""
    return os.environ.get("EPM_HF_DATA_REPO", "").strip() or HF_DATA_REPO_DEFAULT


# The #681 managed data-disk mount that holds relocated per-issue caches.
# Mirrors vm_disk_guard.DEFAULT_DATA_DISK_PATH / data_disk_path() (that module
# imports FROM this one, so the constant cannot live there without a cycle).
DATA_DISK_ROOT_DEFAULT = "/mnt/eps-data"


def data_disk_root() -> Path:
    """Managed data-disk root (env ``EPS_VM_DATA_DISK_PATH``; blank -> default)."""
    raw = os.environ.get("EPS_VM_DATA_DISK_PATH", "").strip()
    return Path(raw or DATA_DISK_ROOT_DEFAULT)


def _path_component_names_issue(rel: Path, issue_n: int) -> bool:
    """True iff some component of ``rel`` names issue ``issue_n`` — exactly
    ``issue_<n>`` / ``issue<n>``, or an ``issue_<n>_<slug>`` / ``issue<n>_<slug>``
    prefix form — the same exact-N-boundary rules as :func:`issue_data_dirs`
    (``issue_65`` never matches ``issue_658``).

    NOTE: the post-#681-cutover worktree layout may name issue dirs with a
    HYPHEN (``issue-<n>``); deliberately NOT matched yet — a hyphen-named
    target takes the fail-toward-keep disposition, the safe direction. Widen
    when the cutover lands, with a test."""
    n = str(issue_n)
    for comp in rel.parts:
        if comp in (f"issue_{n}", f"issue{n}"):
            return True
        if comp.startswith((f"issue_{n}_", f"issue{n}_")):
            return True
    return False


def _managed_symlink_target(cache_dir: Path, issue_n: int) -> Path | None:
    """Fully-resolved symlink target iff it is verifiably ``issue_n``'s
    RELOCATED cache on the managed data disk; ``None`` otherwise
    (fail-toward-keep). ALL of the following must hold, else ``None``:

      * the resolved target lies STRICTLY inside :func:`data_disk_root`
        (``os.path.realpath`` on BOTH sides defeats nested-link /
        relative-link escapes; the root itself is never a valid target);
      * some path component below the root names ``issue_n``
        (:func:`_path_component_names_issue`) — defends against a hand-made
        alias into ANOTHER issue's (or shared) data-disk state;
      * the target IS a directory whose basename EXACTLY matches the cache
        dir's name (the relocation pattern is name-preserving —
        ``hf_dl -> hf_dl``, ``g1_dl -> g1_dl``). This stops a same-issue
        alias (``hf_dl -> .../issue_<n>/store``, or -> any FILE) from being
        reaped: gate 2's ``rglob`` matches descendants only, never the
        resolved root itself, so this predicate is the only defense."""
    target = Path(os.path.realpath(cache_dir))
    root = Path(os.path.realpath(data_disk_root()))
    if target == root or not target.is_relative_to(root):
        return None
    if not _path_component_names_issue(target.relative_to(root), issue_n):
        return None
    if not target.is_dir() or target.name != cache_dir.name:
        return None
    return target


# Shared sidecar stream for ALL VM-disk escalations (this guard's SKIP events,
# vm_disk_guard's active-task escalations, the watcher's sub-floor sentinel) —
# one queryable trace beyond the rotating cron logs. Relative to the repo root.
DISK_GUARD_SIDECAR_REL = Path(".claude") / "cache" / "disk-guard-events.jsonl"


def _running_pod_side() -> bool:
    """True when running on a RunPod pod (NOT the dev VM).

    On a pod, importing/using ``task_workflow.repo_root()`` on a non-``main``
    HEAD auto-routes to a managed worktree via ``git worktree add`` /
    ``git reset --hard`` on MooseFS-backed ``/workspace``, which hangs
    indefinitely (#803, pod-778). Detected by two conditions that both hold on
    every RunPod pod (the container has ``/.dockerenv``; ``bootstrap_pod.sh``
    creates ``/workspace/logs``). ``/workspace/logs`` alone is used as a pod
    signal elsewhere (e.g. ``scripts/issue634_extract_behavior_vectors.py``),
    but on the dev VM ``/workspace/logs`` is a real populated dir — so the
    ``/.dockerenv`` conjunct is load-bearing: it is what makes the AND False
    on the VM and prevents a dev-VM false-positive (which would silently
    no-op a VM-side reap)."""
    return Path("/.dockerenv").exists() and Path("/workspace/logs").is_dir()


def _checkout_branch(checkout: Path) -> str | None:
    """Attached branch name of ``checkout``'s HEAD, or ``None`` (detached /
    unreadable).

    Runs under a sanitized git env (``GIT_DIR`` / ``GIT_WORK_TREE`` cleared) so
    the probe reads the SAME repo ``primary_checkout_root()`` resolved — a
    leaked ``GIT_DIR`` would silently point the two at different repos (#924
    v2). ``rc != 0`` / empty output / OSError / timeout all map to ``None``."""
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")}
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), "symbolic-ref", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


@functools.lru_cache(maxsize=1)
def _off_main_checkout_root() -> Path | None:
    """Root of a NON-``main`` remote checkout (GCE/pod issue-branch clone),
    else ``None`` (#924).

    ``None``  => normal path: every resolution goes through
    ``task_workflow.repo_root()``.
    ``Path``  => degrade gracefully: ``data/``, the worktree scan root, the
    sidecar, and display names resolve against THIS checkout; the cross-task
    active-consumer read is ATTEMPTED and its branch-guard ``RuntimeError``
    (the no-local-``main`` clone shape) is caught at the gate-1 call site
    only — a succeeding read keeps the gate ON (#924 v2).

    Detection reads the PRIMARY checkout (git common-dir parent, NOT the
    script's own dir — a VM worktree invocation must still resolve the repo
    root). The SHARED VM never degrades (``is_shared_vm_env()`` conjunct): a
    pathologically off-main VM keeps today's loud ``repo_root()`` behavior
    rather than silently skipping the consumer gate on a shared ``tasks/``
    tree. Detached HEAD or an unresolvable layout also returns ``None``
    (``repo_root()`` then raises its own loud, unchanged error)."""
    try:
        primary = primary_checkout_root()
    except RuntimeError:
        return None  # unresolvable layout: let repo_root() raise its own loud error
    branch = _checkout_branch(primary)
    if branch is None or branch == "main":
        return None
    if is_shared_vm_env():
        print(
            f"  WARNING: primary checkout {primary} is on {branch!r} on the SHARED VM — "
            f"refusing the #924 off-main degrade (the active-consumer gate must not be "
            f"skipped on a shared tasks/ tree); falling through to repo_root().",
            file=sys.stderr,
        )
        return None
    return primary


def _resolution_root() -> Path:
    """``repo_root()`` on the primary-on-main VM; the checkout root on an
    off-main remote clone (#924). Future path-resolution sites in this module
    should call THIS helper, not ``repo_root()`` directly."""
    off = _off_main_checkout_root()
    return off if off is not None else repo_root()


def disk_guard_sidecar_path() -> Path:
    """Absolute path of the shared disk-guard escalation sidecar JSONL."""
    return _resolution_root() / DISK_GUARD_SIDECAR_REL


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
    """Absolute path of the checkout's ``data/`` directory (repo root on the
    VM; the clone root on an off-main remote checkout, #924)."""
    return _resolution_root() / "data"


def _worktree_data_roots(issue_n: int) -> list[Path]:
    """``data/`` directories inside this issue's worktree(s).

    The live experiment's download/store data often lives in the WORKTREE,
    not the repo root — e.g. ``.claude/worktrees/issue-658/data/issue_658/``
    (the worktrees tree was 139 GB on 2026-06-26, dominated by per-issue
    worktree data). A `/issue` run can have ``issue-<N>`` AND
    ``issue-<N>-<suffix>`` (same-issue follow-up round) worktrees, so every
    ``issue-<N>*`` worktree whose name maps to exactly ``issue_n`` is
    scanned. Returns only existing ``<worktree>/data`` dirs."""
    wt_root = _resolution_root() / ".claude" / "worktrees"
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
    (both naming conventions, repo-root AND worktree copies). Directories AND
    symlinks are returned — a symlink is admitted even when DANGLING or
    pointing at a file (``is_dir()`` follows the link and returns False for
    both, which used to leave dangling relocation links invisible forever;
    #915). ``store/`` and plain non-dir files named like a cache are never
    included.
    """
    out: list[Path] = []
    for issue_dir in issue_data_dirs(issue_n, data_root):
        for pattern in CACHE_DIR_GLOBS:
            if "*" not in pattern:
                # A literal name (hf_dl): Path.glob's precise selector calls
                # path.exists(), which FOLLOWS a symlink — a DANGLING
                # literal-name link would stay invisible forever. Probe the
                # candidate directly with non-following checks instead.
                # (The wildcard selector below scandir-lists entries, so it
                # DOES yield dangling g*_dl links.)
                cand = issue_dir / pattern
                if cand.is_dir() or cand.is_symlink():
                    out.append(cand)
                continue
            for match in sorted(issue_dir.glob(pattern)):
                if match.is_dir() or match.is_symlink():
                    out.append(match)
    return out


def extract_issue_number(name: str) -> int | None:
    """Issue number a ``/tmp`` entry NAME is keyed to, or ``None``.

    P1 (prefix — ``i<N>``, ``issue<N>``, ``issue-<N>``, ``issue_<N>``, with the
    N terminated by ``.``/``_``/``-`` or end-of-name) is tried FIRST, then P2
    (suffix — ``_<N>`` name-final, underscore only). Pure; unit-tested
    directly. ``i18n_cache`` / ``in2_foo`` / ``tmux-1000`` / ``foo_823_bar``
    all return ``None``; ``i653_766_fixed`` extracts 653 (P1 precedence)."""
    m = _TMP_ISSUE_PREFIX_RE.match(name)
    if m is not None:
        return int(m.group(1))
    m = _TMP_ISSUE_SUFFIX_RE.search(name)
    if m is not None:
        return int(m.group(1))
    return None


def production_tmp_root() -> Path:
    """The ``/tmp`` root PRODUCTION ENTRY POINTS pass explicitly
    (env ``EPM_TMP_CACHE_ROOT``; blank -> :data:`TMP_CACHE_ROOT_DEFAULT`).

    Called ONLY from ``main()`` here and from ``vm_disk_guard.main()`` — a
    source-scan test pins that invariant (#911 I7). Hermeticity contract: the
    /tmp sweep is STRICTLY OPT-IN — library-level calls with ``tmp_root=None``
    NEVER touch any /tmp (no both-None fallback). The existing suite has >=7
    ``data_root=None, apply=True`` library call sites (two under
    constant-terminal status monkeypatches); a both-None -> real-/tmp fallback
    would have them destructively rmtree live /tmp caches during pytest."""
    return Path(os.environ.get("EPM_TMP_CACHE_ROOT", "").strip() or TMP_CACHE_ROOT_DEFAULT)


def _tmp_entry_owned(path: Path) -> bool:
    """True iff the entry (lstat — the link itself, never the target) is owned
    by the current uid. Another user's /tmp dir is skipped (``rmtree`` would
    fail on it anyway); a stat error fails toward not-ours (skip)."""
    try:
        return path.lstat().st_uid == os.getuid()
    except OSError:
        return False


def _noncanonical_min_age_hours() -> float:
    """Recency-keep window (hours) for non-canonical candidates
    (env ``EPS_NONCANONICAL_CACHE_MIN_AGE_HOURS``; invalid/negative -> default)."""
    raw = os.environ.get("EPS_NONCANONICAL_CACHE_MIN_AGE_HOURS", "").strip()
    if not raw:
        return NONCANONICAL_MIN_AGE_HOURS_DEFAULT
    try:
        val = float(raw)
    except ValueError:
        return NONCANONICAL_MIN_AGE_HOURS_DEFAULT
    return val if val >= 0.0 else NONCANONICAL_MIN_AGE_HOURS_DEFAULT


def _dir_max_recency(path: Path) -> float | None:
    """Newest touch time over ``path`` itself + every entry under it
    (lstat-based — never follows symlinks; a per-entry stat error is skipped).

    Regular FILES contribute ``max(st_mtime, st_atime)`` — a live READER
    (e.g. an inline free-analysis on a terminal task) keeps file atimes fresh
    under relatime (<=24h stale, inside the 48h window) while never touching
    mtime, and the janitor's own scans never refresh them (lstat/stat are not
    content reads). DIRECTORIES and symlinks contribute st_mtime ONLY:
    dir/symlink atimes refresh on readdir/traversal — including THIS module's
    own sizing rglob — so folding them in would let each guard pass re-warm
    every candidate and keep it forever (a self-keeping loop). Dir mtime
    changes only on entry create/delete/rename, a true write signal. Returns
    ``None`` when even the top-level lstat fails (the caller keeps —
    fail-toward-keep)."""

    def _touch_time(st: os.stat_result) -> float:
        if stat.S_ISREG(st.st_mode):
            return max(st.st_mtime, st.st_atime)
        return st.st_mtime

    try:
        newest = _touch_time(path.lstat())
    except OSError:
        return None
    try:
        for p in path.rglob("*"):
            try:
                newest = max(newest, _touch_time(p.lstat()))
            except OSError:
                continue
    except OSError:
        pass
    return newest


def noncanonical_cache_dirs(
    issue_n: int,
    *,
    data_root: Path | None = None,
    tmp_root: Path | None = None,
    sweep_tmp: bool = True,
) -> list[Path]:
    """Non-canonical issue-keyed cache candidates for ``issue_n`` (#911).

    P1+P2: TOP-LEVEL entries of ``tmp_root`` whose name extracts ``issue_n``
    (:func:`extract_issue_number`) — dirs or symlinks, uid-owned, never files,
    never recursive (mirrors tier (c)'s deliberate non-recursive /tmp policy);
    P2 (suffix) matches plain dirs ONLY. The /tmp part runs ONLY when
    ``tmp_root`` is EXPLICITLY non-None AND ``sweep_tmp`` is True (strict
    opt-in — no both-None fallback; see :func:`production_tmp_root`).

    P3: whole-dir ``data/`` caches across ``_resolve_data_roots(issue_n,
    data_root)`` whose name matches ``issue…<N>…{_dl,_hfstage,_cache}`` with
    the regex group equal to ``issue_n`` (dirs or symlinks).

    Empty when :data:`NONCANONICAL_SWEEP_KILL_ENV` is set (emergency rollback
    without a revert). Every candidate returned here must STILL pass the reap
    gates in ``clean_issue_downloads`` (recency, nested-durable, positive
    re-downloadability evidence) before anything is deleted."""
    if os.environ.get(NONCANONICAL_SWEEP_KILL_ENV, "").strip():
        return []
    out: list[Path] = []
    if sweep_tmp and tmp_root is not None and tmp_root.is_dir():
        for child in sorted(tmp_root.iterdir()):
            try:
                is_dir_or_link = child.is_dir() or child.is_symlink()
            except OSError:
                continue
            if not is_dir_or_link or not _tmp_entry_owned(child):
                continue
            m = _TMP_ISSUE_PREFIX_RE.match(child.name)
            if m is not None:
                if int(m.group(1)) == issue_n:
                    out.append(child)
                continue
            m = _TMP_ISSUE_SUFFIX_RE.search(child.name)
            if (
                m is not None
                and int(m.group(1)) == issue_n
                and child.is_dir()
                and not child.is_symlink()
            ):
                out.append(child)
    for root in _resolve_data_roots(issue_n, data_root):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not (child.is_dir() or child.is_symlink()):
                continue
            m = _DATA_NONCANONICAL_CACHE_RE.match(child.name)
            if m is not None and int(m.group(1)) == issue_n:
                out.append(child)
    return out


def _dedup_nested(paths: list[Path]) -> list[Path]:
    """Topmost-only candidate set: drop any candidate that is a DESCENDANT of
    another (sort shallow-first; keep ``p`` iff no kept ``k`` is in
    ``p.parents``). Prevents the rmtree-parent-then-fail-child double-handling
    when a P3 whole-dir cache also contains a canonical ``hf_dl`` (#911 I10)."""
    kept: list[Path] = []
    for p in sorted(set(paths), key=lambda q: (len(q.parts), str(q))):
        if any(k in p.parents for k in kept):
            continue
        kept.append(p)
    return kept


def _noncanonical_recency_blocked(
    cache_dir: Path,
    *,
    issue_n: int,
    apply: bool,
    min_age_hours: float,
    now: float,
) -> str | None:
    """Gate 1.5 (NEW non-canonical candidates only): SKIP reason when the
    tree's newest touch time (:func:`_dir_max_recency` — file
    ``max(st_mtime, st_atime)``, dir/symlink mtime) is within
    ``min_age_hours`` — a live reader OR writer (e.g. an inline free-analysis
    on a terminal task) may hold it; /tmp paths are never declared in plans,
    so the #773 consumer gate cannot see those readers and recency is the
    only signal. Sidecar kind ``noncanonical-cache-recent-kept``.
    Fail-toward-keep on stat errors."""
    newest = _dir_max_recency(cache_dir)
    if newest is None:
        reason = "recency unreadable (stat failed) — fail-toward-keep; KEPT"
    else:
        age_hours = (now - newest) / 3600.0
        if age_hours >= min_age_hours:
            return None
        reason = (
            f"touched {age_hours:.1f}h ago (< {min_age_hours:g}h recency window) — "
            f"a live reader/writer may hold it; KEPT"
        )
    append_disk_guard_event(
        {
            "kind": "noncanonical-cache-recent-kept",
            "task": issue_n,
            "path": _rel_name(cache_dir),
            "reason": reason,
        },
        apply=apply,
    )
    return reason


def _p2_suffix_only(name: str) -> bool:
    """True iff ``name``'s ONLY extraction route is the P2 ``_(\\d+)$`` suffix
    (no P1 prefix match, no P3 cache-suffix match).

    P2 stays in discovery for ATTRIBUTION visibility, but its match class
    includes foreign mkdtemp leftovers (``tmpdu2m4w_7`` extracts issue 7), so
    a P2-suffix-only candidate never gets the empty-dir evidence license — it
    must show REAL non-empty positive evidence (hub-layout markers or a
    data-repo-prefix mirror) to be reap-eligible; an empty P2-only dir is
    kept + escalated (``unverified-kept``), never deleted. P1/P3 candidates
    keep the empty-dir license unchanged. (r2 fix, review concern
    ``p2-empty-tempdir-false-reap``.)"""
    if _TMP_ISSUE_PREFIX_RE.match(name) is not None:
        return False
    if _DATA_NONCANONICAL_CACHE_RE.match(name) is not None:
        return False
    return _TMP_ISSUE_SUFFIX_RE.search(name) is not None


def _nested_durable_dirs(cache_dir: Path) -> list[Path]:
    """Nested dirs named ``store`` OR ``eval_results`` under the candidate —
    the v4 generalization of :func:`_nested_store_dirs` (#911 I12).

    For NON-CANONICAL whole-dir candidates a non-empty result BLOCKS the reap
    outright (skip + sidecar kind ``noncanonical-cache-durable-content-kept``,
    fail-toward-keep — NO mirror check attempted; durable results must never
    ride a whole-dir rmtree). Canonical ``hf_dl``/``g*_dl`` keep the existing
    #679 store-parity behavior unchanged."""
    out: list[Path] = []
    for name in ("store", "eval_results"):
        for p in cache_dir.rglob(name):
            if p.is_dir() and not p.is_symlink():
                out.append(p)
    return sorted(out)


def _data_repo_toplevel_names() -> frozenset[str] | None:
    """Top-level entry names of the HF data repo (ONE
    ``list_repo_tree(recursive=False)`` listing, cached per run by the
    caller), or ``None`` on ANY failure (missing token, network error, import
    error) — the fail-toward-keep signal for evidence branch (b)."""
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        names: set[str] = set()
        for entry in api.list_repo_tree(
            repo_id=hf_data_repo(),
            repo_type="dataset",
            revision="main",
            recursive=False,
        ):
            names.add(str(entry.path).split("/", 1)[0])
        return frozenset(names)
    except Exception as exc:
        print(
            f"  ! noncanonical evidence: HF top-level listing for {hf_data_repo()} failed "
            f"({type(exc).__name__}: {exc}); fail-toward-keep",
            file=sys.stderr,
        )
        return None


def _noncanonical_reap_gates(
    cache_dir: Path,
    *,
    issue_n: int,
    apply: bool,
    min_age_hours: float,
    now: float,
    data_repo_toplevel_cache: dict[str, frozenset[str] | None],
) -> str | tuple[str, str]:
    """Run gates 1.5 -> 1.6 -> 1.7 on a NON-CANONICAL candidate (#911),
    ordered cheap-first (recency needs only stats, the durable scan is a name
    rglob, the evidence gate may make one HF call — memoized in
    ``data_repo_toplevel_cache``).

    Returns the positive-evidence STRING when the reap is licensed, or a
    ``(disposition, skip_reason)`` tuple when blocked (fail-toward-keep; the
    per-gate sidecar row is already appended by the time this returns)."""
    rel = _rel_name(cache_dir)
    # Gate 1.5 — recency (sidecar row appended inside on a block).
    recency_reason = _noncanonical_recency_blocked(
        cache_dir, issue_n=issue_n, apply=apply, min_age_hours=min_age_hours, now=now
    )
    if recency_reason is not None:
        return ("recency-kept", recency_reason)
    # Gate 1.6 — nested durable content blocks the whole-dir reap outright.
    durable = _nested_durable_dirs(cache_dir)
    if durable:
        paths = ", ".join(_rel_name(d) for d in durable)
        reason = (
            f"nested durable dir(s) under a non-canonical candidate ({paths}) — "
            f"a whole-dir rmtree would destroy results; KEPT "
            f"(fail-toward-keep, no mirror check attempted)"
        )
        append_disk_guard_event(
            {
                "kind": "noncanonical-cache-durable-content-kept",
                "task": issue_n,
                "path": rel,
                "nested_durable": [_rel_name(d) for d in durable],
                "reason": reason,
            },
            apply=apply,
        )
        return ("durable-content-kept", reason)
    # Gate 1.7 — positive re-downloadability evidence. Branch (a) + the
    # trivial empty case need no network; fetch the data-repo listing only
    # when branch (b) is actually needed. A P2-suffix-only candidate (a
    # ``*_<N>`` name with no P1/P3 route — the foreign-mkdtemp shape,
    # ``tmpdu2m4w_7``) never gets the empty-dir license: it must show REAL
    # non-empty evidence (r2 fix, concern ``p2-empty-tempdir-false-reap``).
    p2_only = _p2_suffix_only(cache_dir.name)
    evidence = _positive_redownloadability_evidence(
        cache_dir, data_repo_toplevel=None, allow_empty_license=not p2_only
    )
    if evidence is None:
        repo = hf_data_repo()
        if repo not in data_repo_toplevel_cache:
            data_repo_toplevel_cache[repo] = _data_repo_toplevel_names()
        evidence = _positive_redownloadability_evidence(
            cache_dir,
            data_repo_toplevel=data_repo_toplevel_cache[repo],
            allow_empty_license=not p2_only,
        )
    if evidence is None:
        reason = (
            "no positive re-downloadability evidence (no hub-layout markers; "
            "top-level names not verified as data-repo prefixes"
            + ("; P2 suffix-only route — requires non-empty positive evidence" if p2_only else "")
            + ") — KEPT (escalate-only, never deleted)"
        )
        append_disk_guard_event(
            {
                "kind": "noncanonical-cache-unverified-kept",
                "task": issue_n,
                "path": rel,
                "reason": reason,
            },
            apply=apply,
        )
        return ("unverified-kept", reason)
    return evidence


def _positive_redownloadability_evidence(
    cache_dir: Path,
    *,
    data_repo_toplevel: frozenset[str] | None,
    allow_empty_license: bool = True,
) -> str | None:
    """Gate 1.7 — the v4 REAP LICENSE for whole-dir non-canonical candidates
    (#911 I11). Returns an evidence string, or ``None`` (=> escalate-only,
    NEVER delete).

    Branch (a): HF hub-layout markers at depth <= 2 (``models--*`` /
    ``datasets--*`` / ``blobs`` / ``snapshots`` / ``refs``) — the dir is a hub
    cache, re-downloadable by construction. HIDDEN entries (any name starting
    with ``.``) are EXCLUDED from this scan at both depths: a dot-dir is tool
    state — a git checkout's ``.git/refs`` is present in EVERY checkout and
    ``refs``/``snapshots`` are generic enough to collide — so scanning it
    would spoof hub evidence and license a whole-dir rmtree of a non-hub dir
    (r2 fix, review concern ``evidence-scan-hidden-dir-collision``). Branch
    (b): every top-level entry name is a top-level prefix that exists on the
    HF data repo (``data_repo_toplevel``; ``None`` = the listing fetch failed
    => fail-toward-keep) — the dir is a partial mirror of on-repo folders. A
    top-level ``.cache`` DIRECTORY is ignored in branch (b): it is the
    huggingface_hub ``snapshot_download(local_dir=...)`` bookkeeping dir
    (``.cache/huggingface/`` metadata + locks — hub-client tooling state,
    never data; the live ``data/issue_744_dl`` mirror carries exactly this
    shape), and at least one NON-bookkeeping entry must remain and match.
    Other hidden top-level entries (``.git``, ...) STAY in the branch-(b)
    name set, where an unmatched name can only BLOCK the license
    (fail-toward-keep). An empty candidate dir passes trivially (nothing to
    lose) ONLY when ``allow_empty_license`` is True — the caller disables it
    for P2-suffix-only candidates (see :func:`_p2_suffix_only`). A flat pile
    of generated files (the ``fact_check_823`` judge-cache shape) matches
    NEITHER branch and correctly escalates."""
    try:
        depth1 = sorted(cache_dir.iterdir())
    except OSError:
        return None  # unreadable / dangling — nothing verifiable, keep
    if not depth1:
        if allow_empty_license:
            return "empty dir (nothing to lose)"
        return None  # P2 suffix-only candidate — empty-dir license disallowed

    def _is_hub_marker(name: str) -> bool:
        return (
            name.startswith("models--")
            or name.startswith("datasets--")
            or name in ("blobs", "snapshots", "refs")
        )

    visible1 = [c for c in depth1 if not c.name.startswith(".")]
    entries = list(visible1)
    for child in visible1:
        if child.is_dir() and not child.is_symlink():
            try:
                entries.extend(e for e in child.iterdir() if not e.name.startswith("."))
            except OSError:
                continue
    for entry in entries:
        if _is_hub_marker(entry.name):
            return f"hub-layout marker: {entry.name}"
    if data_repo_toplevel is None:
        return None  # branch (b) unavailable — fail-toward-keep
    names = [
        c.name
        for c in depth1
        # Ignore the hub-client local_dir bookkeeping dir (see docstring).
        if not (c.name == ".cache" and c.is_dir() and not c.is_symlink())
    ]
    if names and all(n in data_repo_toplevel for n in names):
        shown = ", ".join(sorted(names)[:5])
        return f"data-repo-prefix mirror: {shown}"
    return None


def _dir_size_bytes(path: Path) -> int:
    """Recursive on-disk size of ``path`` in bytes (best-effort; a stat error
    on a single entry is skipped, never raised — sizing is reporting only).

    Follows ``path`` itself when it is a symlink-to-dir (``rglob`` traverses
    through a top-level link), so a relocated cache's size reflects the
    data-disk bytes a managed reap frees (#915); symlinks INSIDE the tree
    are still excluded. A dangling link or a link-to-file sizes to 0."""
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


def _active_consumer_protected_issues(self_issue_n: int) -> dict[int, list[int]]:
    """Map ``{protected_issue_M: [consumer_task_ids...]}`` for every ``M`` that
    some ACTIVE task declares as a ``data/issue_<M>/`` input in its
    ``plans/plan.md`` OR ``body.md`` (the active-CONSUMER reap gate, #773).

    An "active" task is one whose status is NOT in
    :data:`_CONSUMER_INACTIVE_STATUSES` — i.e. it is currently doing work or
    imminently planned to, so its declared inputs may be read at any moment.
    A protected ``M`` blocks the reap of ``data/issue_<M>/``'s caches: deleting
    them could strand the active consumer mid-run (#742 died on a
    ``FileNotFoundError`` after ``#658``'s caches were reaped out from under it).

    ``self_issue_n`` is excluded from the CONSUMER set — a task never blocks
    its OWN cache reap (Step-8 self-cleanup + the incremental within-run path
    both reap the task's own cache; the guard must not protect a task against
    itself). NOTE the exclusion is on the consumer, NOT the referenced issue: a
    DIFFERENT active task referencing ``data/issue_<self_issue_n>/`` is exactly
    the cross-issue protection this gate exists for. READ-ONLY: walks ``tasks/``
    text via ``list_by_status``, never
    mutates task state. Missing ``body.md`` / ``plans/plan.md`` is skipped
    fail-soft. Returns ``{}`` when no active task references any
    ``data/issue_<M>/`` (the common case, and the clean no-op for an
    absent / empty ``tasks/`` tree).

    KNOWN LIMITATION (false negative): a consumer that assembles the input
    path in code / a Hydra YAML / an env-var override, with no literal
    ``data/issue_<M>/`` substring in its plan.md or body.md, is NOT seen by
    this regex. The project's reuse-provenance convention puts input paths in
    the plan as literals (so the realistic case is caught), and the consumer's
    OWN cache reap is independently protected by the owning-issue terminal-
    status gate; this residual is a cross-issue config-indirection gap."""
    base = tasks_dir()
    active_statuses = [s for s in STATUSES if s not in _CONSUMER_INACTIVE_STATUSES]
    protected: dict[int, list[int]] = {}
    for status in active_statuses:
        # list_by_status defaults to limit=200 and SILENTLY truncates — a 201st+
        # active task is dropped, so its declared input cache would be reaped out
        # from under it (the #742 strand class this gate exists to prevent,
        # reintroduced through a cap inside the guard). Pass an explicit large
        # limit so the active-consumer scan is complete: a missed consumer is a
        # silent fail-OPEN, the opposite of this guard's fail-toward-keep contract.
        for row in list_by_status(status, limit=10_000):
            consumer_id = row["id"]
            # self_issue_n never protects its OWN reap: exclude it from the
            # CONSUMER set entirely (a task referencing data/issue_<self>/ in its
            # own plan/body must not block self-cleanup). The exclusion is on the
            # CONSUMER, NOT on the referenced issue — a DIFFERENT task (742)
            # referencing data/issue_<self>/ (658) is exactly the protection we
            # want, so we must NOT drop referenced == self_issue_n.
            if consumer_id == self_issue_n:
                continue
            task_dir = base / status / str(consumer_id)
            text_parts: list[str] = []
            for rel in ("body.md", "plans/plan.md"):
                try:
                    text_parts.append((task_dir / rel).read_text())
                except (FileNotFoundError, OSError):
                    continue  # fail-soft: a missing/unreadable file just adds no text
            if not text_parts:
                continue
            blob = "\n".join(text_parts)
            for match in _DATA_ISSUE_REF.finditer(blob):
                referenced = int(match.group(1))
                if referenced == consumer_id:
                    continue  # a task referencing its OWN data/issue_<self>/ does not self-protect
                protected.setdefault(referenced, [])
                if consumer_id not in protected[referenced]:
                    protected[referenced].append(consumer_id)
    # Sort each consumer list for deterministic sidecar telemetry.
    return {m: sorted(consumers) for m, consumers in protected.items()}


def _cache_dir_reap_blocked_by_active_consumer(
    cache_dir: Path,
    *,
    issue_n: int,
    apply: bool,
    protected: dict[int, list[int]],
) -> str | None:
    """Return a SKIP reason if ``issue_n`` is consumed as a planned input by an
    ACTIVE task; ``None`` to allow the reap. Composes with
    :func:`_cache_dir_reap_blocked` (the nested-``store/`` parity gate) — both
    can independently put a cache dir in ``CleanResult.skipped``.

    On a hit, an escalation row is appended to the shared disk-guard sidecar
    (``kind="active-consumer-reap-skipped"``), mirroring the nested-store
    guard's pattern (``append_disk_guard_event``, fail-soft, ``apply=False``
    reports only). ``protected`` is the once-per-call map from
    :func:`_active_consumer_protected_issues`."""
    consumers = protected.get(issue_n)
    if not consumers:
        return None
    rel = _rel_name(cache_dir)
    consumer_str = ", ".join(f"#{c}" for c in consumers)
    reason = (
        f"active task(s) {consumer_str} declare data/issue_{issue_n}/ as a "
        f"planned input — reaping {rel} could strand their run; KEPT"
    )
    append_disk_guard_event(
        {
            "kind": "active-consumer-reap-skipped",
            "task": issue_n,
            "path": rel,
            "consumers": consumers,
            "reason": reason,
        },
        apply=apply,
    )
    return reason


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


def _protected_issues_for_reap(
    issue_n: int,
    *,
    skip_guard: bool,
    apply: bool,
) -> dict[int, list[int]]:
    """Gate-1 protected set for :func:`clean_issue_downloads`, with the #924
    off-main attempt-and-catch.

    ``skip_guard`` (the private test seam) returns ``{}`` with no ``tasks/``
    walk. On the normal (VM / on-main) path — probe ``None`` — the read is a
    plain call and any raise propagates loudly exactly as today. Off-main
    (probe non-``None``, #924 v2, round-1 alternatives reconcile): ATTEMPT the
    cross-task read; only a ``RuntimeError`` from it is caught. On the
    no-local-``main`` clone shape (GCE ``--depth 1 --branch``) ``tasks_dir()``
    -> ``repo_root()`` raises the branch-guard ``RuntimeError`` -> the read is
    genuinely unserviceable and a single-task remote instance has no sibling
    consumers, so an empty protected set is safe: loud WARN + sidecar row,
    never silent. On the full-clone shape the read SUCCEEDS via the fresh-main
    pin worktree and the gate stays fully ON. A non-``RuntimeError`` always
    propagates (the catch is narrow, not a blanket swallow)."""
    if skip_guard:
        return {}
    off_root = _off_main_checkout_root()
    if off_root is None:
        # Normal (VM / on-main) path — plain call, any raise propagates
        # loudly exactly as today.
        return _active_consumer_protected_issues(issue_n)
    try:
        return _active_consumer_protected_issues(issue_n)
    except RuntimeError as exc:
        print(
            f"  WARNING: non-main checkout {off_root} — cross-task "
            f"active-consumer protected-issues read unserviceable "
            f"({exc}); #924: skipping the gate (single-task remote "
            f"instance assumed). Reap proceeds scoped to issue "
            f"{issue_n}'s own checkout-local caches.",
            file=sys.stderr,
        )
        append_disk_guard_event(
            {
                "kind": "off-main-consumer-gate-skipped",
                "task": issue_n,
                "checkout": str(off_root),
                "reason": f"gate read raised on non-main checkout: {exc}",
            },
            apply=apply,
        )
        return {}


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
    # rel -> fully-resolved symlink target ("" for a dangling link); populated
    # in BOTH modes (dry-run + apply) whenever the reap loop takes the symlink
    # branch — direct-link AND symlinked-parent cases (#915). A cache SKIPped
    # by an earlier gate never reaches the branch, so it is not recorded here.
    symlink_targets: dict[str, str] = field(default_factory=dict)
    # (rel, resolved_target): the target is NOT ours to reap — KEPT (external /
    # foreign-issue / non-cache-shaped). Direct-link case: the link itself is
    # (would be) unlinked. Symlinked-parent case: NOTHING is unlinked (the
    # parent link is shared); the sidecar row distinguishes the two via its
    # "via" field ("link" | "parent"). Deliberately NOT in ``removed`` (so
    # ``bytes_freed`` never overstates); still in ``sizes_bytes`` (so
    # ``total_discovered_bytes`` — the escalation sizing — sees the footprint).
    symlink_external_kept: list[tuple[str, str]] = field(default_factory=list)
    # v4 structured reporting (#911): rel -> disposition for every discovered
    # NON-CANONICAL candidate ("removed" | "would-remove" | "recency-kept" |
    # "durable-content-kept" | "unverified-kept" | "consumer-kept" |
    # "external-target-kept" | "failed"). Canonical hf_dl/g*_dl caches are
    # deliberately ABSENT (their reap license is the canonical name convention
    # + the #679 parity gate). vm_disk_guard's tier (b) surfaces these in its
    # --json output — report-only escalation persists NOTHING to the sidecar,
    # so the dry-run acceptance reads these fields instead.
    noncanonical_dispositions: dict[str, str] = field(default_factory=dict)
    # rel -> the gate-1.7 positive re-downloadability evidence string for
    # non-canonical candidates that were (or would be) reaped. A candidate
    # with NO entry here is never in the non-canonical would-remove set.
    noncanonical_evidence: dict[str, str] = field(default_factory=dict)

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
    tmp_root: Path | None = None,
    sweep_tmp: bool = True,
    _skip_active_consumer_guard: bool = False,
) -> CleanResult:
    """Delete (``apply=True``) or report (default) ``issue_n``'s download caches.

    Idempotent: an absent cache contributes nothing. ``store/`` and
    non-cache content are never touched — only the ``CACHE_DIR_GLOBS`` matches
    under the per-issue data dir(s) are removed. A removal that raises is
    recorded in ``failed`` and never aborts the rest (fail-soft per directory,
    fail-loud in the report).

    Two reap gates run per cache dir, both fail-toward-keep:
      1. The active-CONSUMER gate (#773, FIRST): never reap while a DIFFERENT
         active task declares ``data/issue_<N>/`` as a planned input. It is a
         cheap in-memory set lookup, so running it first short-circuits before
         the parity gate's potential HF call. On an OFF-MAIN checkout (#924)
         this gate's ``tasks/`` read is ATTEMPTED and only a ``RuntimeError``
         from it (the branch-guard raise) is caught — loud WARN + sidecar row
         + empty protected set; a succeeding read keeps the gate ON. See
         :func:`_off_main_checkout_root`.
      2. The nested-``store/`` parity gate (#679): never wholesale-rmtree a
         cache dir holding a mis-rooted ``store/`` not verifiably mirrored on HF.

    Symlinked caches (#915): a cache dir that is itself a symlink, or whose
    ``issue_<N>`` PARENT dir is a symlink, routes through symlink-disposition
    logic instead of a plain ``rmtree`` (which refuses symlinks). The resolved
    target is reaped only when :func:`_managed_symlink_target` positively
    identifies it as the issue's relocated cache on the managed data disk
    (strictly inside ``data_disk_root()``, names the owning issue, a directory
    whose basename equals the cache name); otherwise the target is KEPT and
    recorded in ``CleanResult.symlink_external_kept`` + the sidecar (a direct
    link is unlinked; with a symlinked parent NOTHING is unlinked — not the
    shared parent link and not a child entry inside its target tree, even
    when that child is itself a link). DIRECT dangling links are unlinked;
    a dangling entry through a linked parent is kept + sidecar-escalated.
    Managed reaps delete the target BEFORE the link (crash safety: a
    surviving link is re-discovered and retried on the next run). Both gates
    above run FIRST, unchanged —
    gate 1 is issue-number-keyed and path-independent; gate 2's ``rglob``
    traverses THROUGH the symlink into the target.

    NON-CANONICAL candidates (#911): with an EXPLICIT ``tmp_root`` (strict
    opt-in — the two CLI ``main()`` bodies pass ``production_tmp_root()``;
    library callers with ``tmp_root=None`` never touch any /tmp), the sweep
    additionally covers :func:`noncanonical_cache_dirs` — top-level /tmp
    issue-keyed dirs (P1/P2) + whole-dir ``data/`` cache-named dirs (P3).
    These pass gate 1 unchanged, then THREE additional gates in order (each
    fail-toward-keep, each sidecar-escalated):

      1.5 RECENCY — a tree touched (max of mtime/atime) within the 48h window
          is kept (``noncanonical-cache-recent-kept``);
      1.6 NESTED-DURABLE — any nested ``store/`` OR ``eval_results/`` blocks
          the whole-dir reap outright, no mirror check attempted
          (``noncanonical-cache-durable-content-kept``);
      1.7 POSITIVE RE-DOWNLOADABILITY EVIDENCE — the reap license: hub-layout
          markers OR every top-level name verified as a data-repo prefix (one
          cached ``list_repo_tree(recursive=False)`` per run); a predicate
          failure is kept + escalated (``noncanonical-cache-unverified-kept``),
          NEVER deleted.

    Gate 2 (#679 nested-store parity) keeps its existing contract on the
    CANONICAL candidates (gate 1.6 subsumes it — more strictly — on the
    non-canonical ones). ``EPM_SKIP_NONCANONICAL_CACHE_SWEEP=1`` disables the
    non-canonical sweep entirely (emergency kill switch).

    ``_skip_active_consumer_guard`` (private, keyword-only, defaulted False) opts
    out of gate 1, skipping the ``tasks/`` walk entirely. It is a TEST SEAM only
    — no production caller passes it. In particular
    :func:`clean_issue_downloads_incremental` does NOT set it: the within-run
    path keeps the active-CONSUMER gate ON so a DIFFERENT active task's declared
    input is still protected (the gate self-excludes ``self_issue_n`` via the
    protected-set helper, so a task never blocks its own reap regardless of the
    flag; the cross-issue protection is what the flag would have removed).
    """
    res = CleanResult(issue_n=issue_n, apply=apply)
    # Cache the HF listing across cache dirs so a multi-cache-dir issue makes at
    # most one Hub call regardless of how many nested store/ checks run.
    hf_sizes_cache: dict[str, dict[str, int] | None] = {}
    # Gate-1.7 evidence: the data-repo top-level listing is fetched at most once
    # per call (lazily, only when a non-canonical candidate actually reaches the
    # gate and branch (a) did not already license it) — same cost class as the
    # #679 parity gate's HF call.
    data_repo_toplevel_cache: dict[str, frozenset[str] | None] = {}
    # The active-consumer protected set is the same for every cache dir of this
    # issue, so compute it ONCE before the loop (a single tasks/ walk).
    # Merge of #911 (non-canonical discovery + v4 gates) with #924 (off-main
    # graceful degrade): the protected set comes through #924's
    # attempt-and-catch wrapper; discovery + the dedup'd loop are #911's.
    protected = _protected_issues_for_reap(
        issue_n, skip_guard=_skip_active_consumer_guard, apply=apply
    )
    canonical = download_cache_dirs(issue_n, data_root)
    noncanon = noncanonical_cache_dirs(
        issue_n, data_root=data_root, tmp_root=tmp_root, sweep_tmp=sweep_tmp
    )
    noncanon_keys = {os.path.normpath(str(p)) for p in noncanon}
    now = time.time()
    min_age = _noncanonical_min_age_hours()
    for cache_dir in _dedup_nested([*canonical, *noncanon]):
        rel = _rel_name(cache_dir)
        res.sizes_bytes[rel] = _dir_size_bytes(cache_dir)
        is_noncanonical = os.path.normpath(str(cache_dir)) in noncanon_keys
        # Gate 1 (#773): a DIFFERENT active task consumes data/issue_<N>/ as a
        # planned input. Cheap set lookup, checked FIRST so a consumer hit
        # short-circuits before the parity gate's potential HF call.
        consumer_reason = _cache_dir_reap_blocked_by_active_consumer(
            cache_dir, issue_n=issue_n, apply=apply, protected=protected
        )
        if consumer_reason is not None:
            print(f"  ~ SKIP {rel}: {consumer_reason}", file=sys.stderr)
            res.skipped.append((rel, consumer_reason))
            continue
        # Gate dispatch: gates 1.5/1.6/1.7 for non-canonical candidates,
        # gate 2 (#679 nested-store parity) for canonical ones — see
        # _apply_reap_gates. A True return = BLOCKED (fail-toward-keep).
        if _apply_reap_gates(
            res,
            cache_dir,
            is_noncanonical=is_noncanonical,
            issue_n=issue_n,
            apply=apply,
            min_age_hours=min_age,
            now=now,
            hf_sizes_cache=hf_sizes_cache,
            data_repo_toplevel_cache=data_repo_toplevel_cache,
        ):
            continue
        # Symlink disposition (#915): a cache relocated onto the managed data
        # disk is reachable through an in-tree link at either of the two
        # components below the data root — the cache dir itself
        # (data/issue_<N>/hf_dl -> /mnt/eps-data/.../issue_<N>/hf_dl) or a
        # symlinked PARENT issue dir. shutil.rmtree refuses a symlink by
        # design, so both route through disposition logic instead. A data
        # root that is ITSELF a symlink is a deliberate wholesale relocation
        # and stays on the plain path below.
        #
        # parent_linked is computed INDEPENDENTLY of the child's own linkness
        # (the round-2 MF2 fix): when the parent issue dir is a link AND the
        # cache entry inside its target tree is ITSELF a link (double-link),
        # cache_dir.is_symlink() is True THROUGH the parent link, so the old
        # `(not is_symlink()) and parent.is_symlink()` formula misclassified
        # the case as a DIRECT link — apply-mode unlink() then resolved
        # through the shared parent link and removed the child entry inside
        # the not-ours target tree. Parent-link ownership DOMINATES: with a
        # linked parent, the only permitted mutation is rmtree() of a
        # fully-validated managed target; nothing is ever unlink()ed.
        parent_linked = cache_dir.parent.is_symlink()
        direct_linked = cache_dir.is_symlink()
        if direct_linked or parent_linked:
            target = _managed_symlink_target(cache_dir, issue_n)
            resolved = Path(os.path.realpath(cache_dir))
            dangling = not cache_dir.exists()  # exists() follows links
            res.symlink_targets[rel] = "" if dangling else str(resolved)
            if target is None and (not dangling or parent_linked):
                # External / foreign-issue / non-cache-shaped target — or a
                # DANGLING entry reached through a linked parent (nothing to
                # reap, and the entry inside the parent's target tree is not
                # ours to unlink): KEPT (fail-toward-keep), sidecar-escalated
                # for a human read.
                res.symlink_external_kept.append((rel, str(resolved)))
                kept_what = "dangling child link" if dangling else "external target"
                note = (
                    f"symlink parent kept (nothing unlinked); {kept_what} kept: {resolved}"
                    if parent_linked
                    else f"symlink unlinked; external target kept: {resolved}"
                )
                print(
                    f"  ~ {'' if apply else '[report-only] would: '}{note}",
                    file=sys.stderr,
                )
                append_disk_guard_event(
                    {
                        "kind": "symlink-external-target-kept",
                        "task": issue_n,
                        "path": rel,
                        "target": str(resolved),
                        "via": "parent" if parent_linked else "link",
                    },
                    apply=apply,
                )
                if apply and not parent_linked:
                    # Direct link ONLY: the pointer goes; the target stays.
                    # With a linked PARENT nothing is unlinked — the parent
                    # link is shared (sibling caches + non-cache entries
                    # resolve through it), and in the double-link case the
                    # child entry lives INSIDE the parent's target tree, so an
                    # unlink() here would mutate that not-ours tree (the
                    # round-1 MF2 violation).
                    try:
                        cache_dir.unlink()
                    except OSError as exc:
                        print(f"  ! FAILED to remove {rel}: {exc}", file=sys.stderr)
                        res.failed.append(rel)
                continue
            if not apply:
                res.removed.append(rel)  # would-remove (managed or dangling)
                continue
            try:
                if target is not None:
                    # Managed data-disk reap (the tier-b "reap on EITHER
                    # disk" contract). Target FIRST, link second: if this
                    # rmtree fails midway the link survives, so the next run
                    # re-discovers and retries; unlinking first would orphan
                    # a half-deleted target no janitor could ever find again.
                    # _managed_symlink_target guarantees target is a dir.
                    shutil.rmtree(target)
                if not parent_linked:
                    # The boot-disk link itself (direct case; also the
                    # direct dangling-link unlink). NEVER when the parent is
                    # a link: not the shared parent link itself, and not a
                    # child link entry inside the parent's target tree
                    # (unlink() would resolve through the shared parent link
                    # and mutate the not-ours tree — the round-1 MF2
                    # violation; a managed double-link reap leaves the child
                    # link dangling in the managed tree instead).
                    cache_dir.unlink()
            except OSError as exc:
                print(f"  ! FAILED to remove {rel}: {exc}", file=sys.stderr)
                res.failed.append(rel)
                continue
            res.removed.append(rel)
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
    _fill_noncanonical_dispositions(res, noncanon, apply=apply)
    return res


def _apply_reap_gates(
    res: CleanResult,
    cache_dir: Path,
    *,
    is_noncanonical: bool,
    issue_n: int,
    apply: bool,
    min_age_hours: float,
    now: float,
    hf_sizes_cache: dict[str, dict[str, int] | None],
    data_repo_toplevel_cache: dict[str, frozenset[str] | None],
) -> bool:
    """Run the per-candidate reap gates AFTER gate 1 (#773): gates 1.5/1.6/1.7
    (#911) on a NON-CANONICAL candidate, or gate 2 (#679 nested-store parity)
    on a canonical one. Records the skip + disposition/evidence on ``res``.
    Returns True when the candidate is BLOCKED (caller skips it) — every
    block is fail-toward-keep and sidecar-escalated by the gate that fired."""
    rel = _rel_name(cache_dir)
    if is_noncanonical:
        gate = _noncanonical_reap_gates(
            cache_dir,
            issue_n=issue_n,
            apply=apply,
            min_age_hours=min_age_hours,
            now=now,
            data_repo_toplevel_cache=data_repo_toplevel_cache,
        )
        if isinstance(gate, tuple):
            disposition, reason = gate
            print(f"  ~ SKIP {rel}: {reason}", file=sys.stderr)
            res.skipped.append((rel, reason))
            res.noncanonical_dispositions[rel] = disposition
            return True
        res.noncanonical_evidence[rel] = gate
        return False
    # Gate 2 (#679, CANONICAL candidates): a wholesale rmtree(cache_dir) would
    # destroy a nested store/ (generated, NOT re-downloadable). Refuse unless
    # every nested store file is verifiably mirrored on HF (fail-toward-keep).
    # Runs in BOTH dry-run (reports the would-skip) and apply mode. Gate 1.6
    # subsumes this — more strictly: eval_results/ included, no mirror escape
    # — for the non-canonical candidates above.
    skip_reason = _cache_dir_reap_blocked(
        cache_dir, issue_n=issue_n, apply=apply, hf_sizes_cache=hf_sizes_cache
    )
    if skip_reason is not None:
        print(f"  ~ SKIP {rel}: {skip_reason}", file=sys.stderr)
        res.skipped.append((rel, skip_reason))
        return True
    return False


def _fill_noncanonical_dispositions(res: CleanResult, noncanon: list[Path], *, apply: bool) -> None:
    """Tag every discovered non-canonical candidate's reap OUTCOME on
    ``res.noncanonical_dispositions`` (#911 structured reporting). The gate
    helpers already tagged the gate-kept ones; this fills the rest from the
    reap result. A candidate dropped by ``_dedup_nested`` rides its topmost
    ancestor and deliberately gets no row of its own."""
    removed_set = set(res.removed)
    failed_set = set(res.failed)
    external_set = {name for name, _ in res.symlink_external_kept}
    skipped_set = {name for name, _ in res.skipped}
    for p in noncanon:
        rel = _rel_name(p)
        if rel in res.noncanonical_dispositions:
            continue
        if rel in removed_set:
            res.noncanonical_dispositions[rel] = "removed" if apply else "would-remove"
        elif rel in failed_set:
            res.noncanonical_dispositions[rel] = "failed"
        elif rel in external_set:
            res.noncanonical_dispositions[rel] = "external-target-kept"
        elif rel in skipped_set:
            # The only untagged skip path for a non-canonical candidate is
            # gate 1 (#773 active-consumer).
            res.noncanonical_dispositions[rel] = "consumer-kept"


def clean_issue_downloads_incremental(
    issue_n: int,
    *,
    apply: bool = False,
    data_root: Path | None = None,
    tmp_root: Path | None = None,
    sweep_tmp: bool = True,
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
    needs it again.

    The active-CONSUMER gate (#773) RUNS on this path too. The within-run reaper
    self-excludes ``self_issue_n`` in :func:`_active_consumer_protected_issues`,
    so the calling experiment never blocks its OWN reap. But the gate's purpose
    is CROSS-ISSUE protection — a DIFFERENT active task referencing
    ``data/issue_<self>/`` must still block the reap, exactly as on the
    end-of-run path: the self-exclusion is on the CONSUMER, NOT on the
    referenced issue, so skipping the guard here would have removed the
    cross-issue protection on the incremental path entirely (a fail-OPEN strand
    of the #742 class). It defaults to ON; ``clean_issue_downloads`` retains the
    private ``_skip_active_consumer_guard`` kwarg only as a test seam.
    ``tmp_root`` / ``sweep_tmp`` forward verbatim (#911 — same strict opt-in)."""
    return clean_issue_downloads(
        issue_n, apply=apply, data_root=data_root, tmp_root=tmp_root, sweep_tmp=sweep_tmp
    )


def _rel_name(path: Path) -> str:
    """Path relative to the resolution root for display (falls back to
    absolute). Off-main (#924) the root is the checkout itself, so a raising
    ``repo_root()`` can never crash this display helper."""
    try:
        return str(path.relative_to(_resolution_root()))
    except ValueError:
        return str(path)


def _fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f}G"


def main(argv: list[str] | None = None) -> int:
    # Pod-side short-circuit (#803): on a RunPod pod any use of
    # task_workflow.repo_root() on a non-main HEAD hangs on MooseFS
    # (git worktree add / git reset --hard never completes). The between-phase
    # disk-hygiene call is contracted as pod-safe; make it a fast no-op here.
    # There are no hf_dl/g*_dl caches to reap that repo_root() would find on a
    # pod anyway, so returning 0 is behaviorally correct, not just a bail-out.
    if _running_pod_side():
        print(
            "clean_experiment_downloads.py: pod-side no-op "
            "(repo_root() worktree auto-routing hangs on MooseFS; #803)"
        )
        return 0
    ap = argparse.ArgumentParser(
        description=(
            "Delete a finished experiment's HF-download caches "
            "(data/issue_<N>/hf_dl + g*_dl, plus non-canonical issue-keyed "
            "staging caches: top-level /tmp i<N>*/issue<N>*/*_<N> dirs and "
            "data/ issue…<N>…{_dl,_hfstage,_cache} dirs — the latter set "
            "gated on a 48h recency window, a nested store/+eval_results/ "
            "block, and positive re-downloadability evidence). "
            "Re-downloadable; store/ + eval_results/ are never touched. "
            "Dry-run by default."
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
    # The /tmp sweep opt-in lives HERE (and in vm_disk_guard.main()) ONLY —
    # CLI users get the widened non-canonical sweep; library callers stay
    # hermetic unless they pass an explicit tmp_root (#911 I7). The kwarg is
    # passed signature-adaptively: an existing test seam monkeypatches the
    # cleaner with a stub that predates tmp_root (tests/test_clean_experiment_
    # downloads_pod_side_short_circuit.py::test_vm_side_runs_normal_dispatch),
    # and those tests must keep passing unmodified; both production cleaners
    # accept it, so the production CLI always opts in.
    kwargs: dict = {"apply": args.apply}
    with contextlib.suppress(TypeError, ValueError):
        if "tmp_root" in inspect.signature(cleaner).parameters:
            kwargs["tmp_root"] = production_tmp_root()
    res = cleaner(args.issue, **kwargs)
    mode = "incremental " if args.incremental else ""
    verb = "removed" if args.apply else "would remove"
    print(
        f"clean_experiment_downloads {mode}issue {args.issue}: {verb} "
        f"{len(res.removed)} cache dir(s), {_fmt_gb(res.bytes_freed)} | "
        f"skipped {len(res.skipped)} | "
        f"external-kept {len(res.symlink_external_kept)} | failed {len(res.failed)}"
    )
    for name in res.removed:
        print(f"  - {verb}: {name} [{_fmt_gb(res.sizes_bytes.get(name, 0))}]")
    for name, reason in res.skipped:
        print(f"  ~ SKIP (kept): {name} — {reason}")
    for name, tgt in res.symlink_external_kept:
        kept_verb = "kept" if args.apply else "would keep"
        print(f"  ~ {kept_verb} external symlink target: {name} -> {tgt}")
    for name in res.failed:
        print(f"  ! FAILED: {name}")
    if not res.removed and not res.failed and not res.skipped and not res.symlink_external_kept:
        print("  (no download caches found — nothing to do)")
    return 2 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
