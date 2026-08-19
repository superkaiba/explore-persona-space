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

  * STAGING-ROOT caches (#2095) — the same non-canonical contract extends,
    STRICTLY OPT-IN (``staging_roots`` — only ``main()`` passes
    ``production_staging_roots()``: default ``/mnt/eps-data/$USER`` iff the
    data disk is a live mount; env ``EPM_STAGING_CACHE_ROOTS`` overrides), to
    TOP-LEVEL issue-keyed dirs at the CLAUDE.md staging convention. Extraction
    is PREFIX-ONLY (P1 plus ``tmp_issue<N>_*`` / ``tmp-<N>-*``; NO P2 suffix
    route — the foreign-mkdtemp false-attribution shape), recency reads
    TOP-LEVEL-ONLY (no rglob over multi-100-GB trees), a NEW gate 1.55
    HARD-ESCALATES cross-issue CONTENT (a dir named for issue A holding issue
    B's files — never auto-reaped in v1), ``unverified-kept`` escalations
    carry a class label (``:derived-partial-mirror`` / ``:orphan-no-mirror``),
    and a branch-(b) reap license above ~1 GB must pass a SAMPLED MIRROR
    PROBE (largest file byte-equal on the data repo) or is refused as
    ``unverified-kept:probe-failed``. ``EPM_SKIP_STAGING_CACHE_SWEEP=1``
    kills this leg alone (the non-canonical kill switch kills it
    transitively).

  * TOP-LEVEL /tmp GATE/SMOKE SCRATCH trees (#2127) — a SEPARATE,
    owner-status-INDEPENDENT leg (``sweep_tmp_scratch``, driven from the
    ``vm_disk_guard`` boot-disk pass): top-level ``/tmp`` dirs matching the
    scratch shape globs (``*-gate*`` / ``*smoke*`` / ``eps-*-scratch-*`` /
    ``scratch-*`` / ``mkstest-*``), never a denylisted name (``claude-*`` /
    ``pytest-of-*`` / session+system dirs — checked FIRST). User-facing bulk
    deletions are gated on POSITIVE VERIFIED evidence — HF-backedness
    (mirror/hub-layout) or git-reproducibility (blob-existence proof;
    recoverability-now, see the gc residual note) — never on age alone; age
    (recency) is only ever a KEEP signal (#2127, user directive 2026-08-06;
    #1092 is the standing counter-example an age gate would have destroyed).
    The same git-blob evidence also arms the issue-keyed /tmp legs' gate 1.7
    as branch (c) (``git_evidence_repo``, main()-only opt-in). Kill switch
    ``EPM_SKIP_TMP_SCRATCH_SWEEP=1`` (this leg alone; the non-canonical
    family switch kills it transitively). Named residuals: a blob proof is
    recoverability at VERIFY time (a later ``git gc`` can drop unreachable
    blobs); other-uid/root processes are invisible to the live-process
    probe; and there is deliberately NO ``git worktree prune`` fallback —
    a global prune would unregister PEER worktrees whose trees are
    momentarily missing (e.g. a registered ``/mnt/eps-data``-rooted scratch
    during a mount hiccup) — registration loss is repairable via
    ``git worktree repair`` and destroys no data, but is not this janitor's
    to inflict (a failed ``git worktree remove`` simply KEEPS the tree).

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
     hub-layout markers, data-repo-prefix mirror verification, or (branch (c),
     #2127 — /tmp P1/P2 only, ``git_evidence_repo`` opt-in, never a registered
     worktree) per-file git-blob reproducibility against the main repo; a
     predicate failure escalates (sidecar), never deletes.

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
re-downloadable caches only) — hub-download code paths re-fetch a reaped cache
on a miss, so a reap placed strictly AFTER the cache's LAST consumer in the run
is safe; a direct-path open() reader implements no re-download and crashes
FileNotFoundError if any later phase still reads the cache (#1489;
.claude/rules/gotchas.md). The
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
import fnmatch
import functools
import getpass
import hashlib
import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from explore_persona_space.backends.slurm import WORKING_TREE_OVERLAY_PATHS
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

# ─── staging-root sweep (#2095) ──────────────────────────────────────────────
# Inline rounds also stage multi-GB inputs at the CLAUDE.md staging convention
# ``/mnt/eps-data/$USER/issue<N>_<slug>/``. Extraction there is PREFIX-ONLY:
# the shared P1 regex first, then the two tmp-variants P1 cannot match
# (``tmp_issue<N>_*`` / ``tmp-<N>-*`` — leading ``t``); deliberately NO P2
# suffix route (the foreign-mkdtemp false-attribution shape — staging dirs
# are huge, so a false attribution is unacceptable there).
_STAGING_TMP_PREFIX_RE = re.compile(r"^tmp[-_](?:issue[-_]?)?(\d+)(?:[._-]|$)")
STAGING_SWEEP_KILL_ENV = "EPM_SKIP_STAGING_CACHE_SWEEP"
STAGING_ROOTS_ENV = "EPM_STAGING_CACHE_ROOTS"
STAGING_EVIDENCE_PROBE_FLOOR_GB = 1.0  # env EPM_STAGING_EVIDENCE_PROBE_FLOOR_GB

# ─── top-level /tmp gate/smoke scratch sweep (#2127) ─────────────────────────
# Gate/smoke scratch trees (repo checkouts, registered linked worktrees, gate
# output dirs) pile up at the /tmp TOP LEVEL under names the issue-keyed
# sweep never matches (~70 GiB found untouched >= 48 h at 98% root-disk
# usage, 2026-08-06). They are swept OWNER-STATUS-INDEPENDENTLY under a
# stricter, EVIDENCE-gated contract: deletion requires a per-file
# git-reproducibility PROOF — every non-exempt, non-tolerated regular file's
# git blob must EXIST in the MAIN repo's object database — never age alone;
# age (recency) is only ever a KEEP signal (#2127, user directive 2026-08-06;
# #1092 is the standing counter-example an age gate would have destroyed).
# See ``sweep_tmp_scratch``. NOTE the gc residual: a blob-existence proof is
# recoverability-NOW — an unreachable blob can be gc'd later, so the proof
# guarantees the bytes are recoverable at sweep time, not forever.
_SCRATCH_SHAPE_GLOBS = ("*-gate*", "*smoke*", "eps-*-scratch-*", "scratch-*", "mkstest-*")
# Hard denylist, checked FIRST (belt-and-braces on top of the structural
# shape non-match): the live Claude task-output tree (``/tmp/claude-1001``),
# live pytest basetemps (``pytest-of-*`` — a Step-9c gate may be running),
# and session/system dirs must stay unreachable even if a future shape glob
# widens. A denylist match KEEPS regardless of any evidence.
_SCRATCH_DENYLIST_GLOBS = (
    "claude-*",
    "pytest-of-*",
    "tmux-*",
    "systemd-private-*",
    "snap-private-tmp*",
    "ssh-*",
)
# Rebuildable tool state by construction — pruned from the VERIFICATION walk
# (their contents never need a blob proof) and from the READER-atime signal
# (a janitor's own git probes refresh clone ``.git`` atimes; uv-hardlinked
# ``.venv`` site-packages share atimes with the global uv cache — neither is
# evidence of a live reader of THIS tree). Their MTIMES still count toward
# the write-recency keep signal (a mid-build ``.venv`` reads as a live
# writer).
_SCRATCH_EXEMPT_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        ".ruff_cache",
        ".mypy_cache",
    }
)
# Small-text tolerance: the ONE lever that accepts (and therefore deletes)
# bytes WITHOUT a blob proof. Deliberately narrow — process-telemetry
# extensions only. ``.diff`` is deliberately EXCLUDED: a .diff is by
# construction a serialization of at-the-time-UNCOMMITTED work (the
# redirect-a-diff-before-a-risky-operation pattern puts exactly the
# only-copy class into scratch trees). ``.md`` / ``.json`` / ``.jsonl`` are
# NOT tolerated (notes + judge outputs live there). Widening this allowlist
# or the byte caps widens unproven deletion.
_SCRATCH_TOLERATED_EXTS = frozenset({".txt", ".log", ".out", ".err"})
SCRATCH_TOLERATED_FILE_MAX_BYTES = 5 * 2**20  # 5 MiB per tolerated file
SCRATCH_TOLERATED_TOTAL_MAX_BYTES = 64 * 2**20  # 64 MiB tolerated per candidate
SCRATCH_SWEEP_KILL_ENV = "EPM_SKIP_TMP_SCRATCH_SWEEP"  # this leg only (family switch also binds)
SCRATCH_VERIFY_MAX_GB_DEFAULT = 64.0  # env EPS_SCRATCH_VERIFY_MAX_GB (hash-byte cap per candidate)
SCRATCH_ESCALATE_FLOOR_GB_DEFAULT = (
    1.0  # env EPS_SCRATCH_ESCALATE_FLOOR_GB (keep-row sidecar floor)
)
SCRATCH_VERDICT_CACHE_REL = Path(".claude") / "cache" / "scratch-verify-cache.json"

# ── stray top-level /tmp uv PROJECT FILES (#2377) ────────────────────────────
# uv project discovery walks UP from the cwd, so a stray
# /tmp/{pyproject.toml,uv.toml,uv.lock} pair makes EVERY /tmp-cwd `uv run` on
# the shared VM resolve the stray project (2026-08-18: fleet-wide rc=2 on an
# unresolvable stray lock). File-granular sibling of the #2127 scratch leg:
# exactly these three fixed names, at the top level of tmp_root only.
UV_PROJECT_POISON_NAMES = ("pyproject.toml", "uv.toml", "uv.lock")  # top-level /tmp only
# Freshness grace (seconds) before a VERIFIED poison file may be quarantined —
# defends against racing a file mid-write. Deliberately MINUTES, not the
# #2127 leg's 48 h window: the poison is a live fleet-correctness hazard, and
# recency is only ever a KEEP signal (the next guard pass acts once quiescent).
UVPROJ_RECENT_GRACE_SECONDS = 600.0
# The uv blast radius every sidecar row names (acceptance criterion 2).
UVPROJ_BLAST_RADIUS = "poisons uv project discovery for every /tmp-cwd `uv run` on the shared VM"

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


def append_disk_guard_event(event: dict, *, apply: bool = True) -> bool:
    """Append one JSON line to the shared disk-guard sidecar (fail-soft).

    Used by every VM-disk escalation path so all disk events share one stream.
    A ``ts`` is stamped if the caller did not supply one. The parent dir is
    created idempotently. A write failure is logged loudly but NEVER raises —
    the sidecar is observability, and losing one escalation row must not crash
    the cleanup / guard pass that emits it. ``apply=False`` reports only.

    Returns whether the caller's durable-emission obligation is DISCHARGED:
    ``True`` on a successful append (or in report-only mode, where no durable
    row is owed), ``False`` when the append FAILED — so a dedup/suppression
    layer (#2147 D6 / review round 2 M2) can decline to record the event as
    "alerted" and re-alert on the next pass instead of silently suppressing
    an escalation that never landed."""
    row = {"ts": datetime.now().astimezone().isoformat(), **event}
    line = json.dumps(row)
    if not apply:
        print(f"  [report-only] would append disk-guard event: {line[:160]}", file=sys.stderr)
        return True
    dest = disk_guard_sidecar_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a") as fh:
            fh.write(line + "\n")
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  WARNING: appending disk-guard event failed: {exc}", file=sys.stderr)
        return False
    return True


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


def extract_staging_issue_number(name: str) -> int | None:
    """Issue number a STAGING-root entry NAME is keyed to, or ``None`` (#2095).

    P1 (the shared :func:`extract_issue_number` prefix regex) first, then the
    staging tmp-variant (``tmp-<N>-*``, ``tmp_issue<N>_*``). Deliberately NO
    P2 suffix route (the foreign-mkdtemp shape; staging dirs are huge). A
    leading ``.`` (dot-dir, e.g. ``.hf_i1092_operator``) can match neither
    anchored regex — NEVER a candidate by construction. Pure; unit-tested."""
    m = _TMP_ISSUE_PREFIX_RE.match(name)
    if m is not None:
        return int(m.group(1))
    m = _STAGING_TMP_PREFIX_RE.match(name)
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


def production_staging_roots() -> list[Path]:
    """Staging roots PRODUCTION ENTRY POINTS pass explicitly (#2095 —
    main()-only, the sibling of :func:`production_tmp_root`; the I7-style AST
    pin extends to this symbol).

    Default: ``[data_disk_root()/<getpass.getuser()>]`` iff
    ``os.path.ismount(data_disk_root())`` (an unmounted data disk is a clean
    no-op — the #681 round-2 Major class) AND the per-user dir ``is_dir()``.
    Env :data:`STAGING_ROOTS_ENV` (colon-separated absolute roots) OVERRIDES
    the default — each entry kept iff ``is_dir()``, no mount guard on
    explicit roots. :data:`STAGING_SWEEP_KILL_ENV` set -> ``[]``. Scope note:
    that kill switch disables the SWEEP leg only — the watcher's read-only
    ``_staging_top_caches`` attribution deliberately ignores it (#2095 §4.3).
    Hermeticity contract mirrors the /tmp leg: library calls with
    ``staging_roots=None`` NEVER touch any staging root (no fallback)."""
    if os.environ.get(STAGING_SWEEP_KILL_ENV, "").strip():
        return []
    raw = os.environ.get(STAGING_ROOTS_ENV, "").strip()
    if raw:
        return [Path(p) for p in raw.split(":") if p.strip() and Path(p).is_dir()]
    root = data_disk_root()
    if not os.path.ismount(root):
        return []
    user_dir = root / getpass.getuser()
    if not user_dir.is_dir():
        return []
    return [user_dir]


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


def _dir_max_recency(path: Path, *, top_level_only: bool = False) -> float | None:
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
    fail-toward-keep).

    ``top_level_only`` (#2095, staging candidates): lstat ``path`` + each
    IMMEDIATE ``iterdir`` entry only — same ``_touch_time`` rules, no rglob
    (a full rglob over a multi-100-GB staging tree is a real walk). Accepted
    residual: a deep writer in a TERMINAL-status dir can look old — the
    active-status gate protects live issues and gates 1.6/1.7 still bind."""

    def _touch_time(st: os.stat_result) -> float:
        if stat.S_ISREG(st.st_mode):
            return max(st.st_mtime, st.st_atime)
        return st.st_mtime

    try:
        newest = _touch_time(path.lstat())
    except OSError:
        return None
    try:
        for p in path.iterdir() if top_level_only else path.rglob("*"):
            try:
                newest = max(newest, _touch_time(p.lstat()))
            except OSError:
                continue
    except OSError:
        pass
    return newest


def _staging_cache_dirs(issue_n: int, staging_roots: list[Path] | None) -> list[Path]:
    """TOP-LEVEL staging-root candidates keyed to ``issue_n`` (#2095).

    Per root: one ``iterdir`` — dirs or symlinks only (top-level FILES like
    ``issue<N>_p3_judge.log`` are never candidates), uid-owned
    (:func:`_tmp_entry_owned` — root-owned strays skipped), name keyed via
    :func:`extract_staging_issue_number` (PREFIX-ONLY — no P2 suffix route).
    Empty when ``staging_roots`` is ``None`` (hermeticity: library callers
    never touch any staging root — only ``main()`` passes
    :func:`production_staging_roots`) or when either kill env is set
    (:data:`STAGING_SWEEP_KILL_ENV` kills the staging leg alone;
    :data:`NONCANONICAL_SWEEP_KILL_ENV` kills the whole non-canonical sweep,
    this leg transitively)."""
    if staging_roots is None:
        return []
    if os.environ.get(STAGING_SWEEP_KILL_ENV, "").strip():
        return []
    if os.environ.get(NONCANONICAL_SWEEP_KILL_ENV, "").strip():
        return []
    out: list[Path] = []
    for root in staging_roots:
        try:
            if not root.is_dir():
                continue
            children = sorted(root.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                is_dir_or_link = child.is_dir() or child.is_symlink()
            except OSError:
                continue
            if not is_dir_or_link or not _tmp_entry_owned(child):
                continue
            if extract_staging_issue_number(child.name) == issue_n:
                out.append(child)
    return out


def noncanonical_cache_dirs(
    issue_n: int,
    *,
    data_root: Path | None = None,
    tmp_root: Path | None = None,
    sweep_tmp: bool = True,
    staging_roots: list[Path] | None = None,
    exclude_scratch_shapes: bool = False,
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

    STAGING (#2095): TOP-LEVEL entries of each EXPLICIT ``staging_roots``
    root keyed via :func:`extract_staging_issue_number` — same strict opt-in
    as the /tmp leg (``None`` = hermetic), plus its own kill switch
    (:data:`STAGING_SWEEP_KILL_ENV`); see :func:`_staging_cache_dirs`.

    ``exclude_scratch_shapes`` (#2127): True ONLY from a guard invocation
    that ALSO runs the scratch leg (:func:`sweep_tmp_scratch`) in the same
    pass — an issue-keyed /tmp entry that is ALSO scratch-shaped
    (:func:`is_tmp_scratch_name`) is then skipped here so one dir is never
    double-attributed (once as an issue cache, once as a scratch row). The
    per-issue CLI keeps the old routing bit-identically (default False).

    Empty when :data:`NONCANONICAL_SWEEP_KILL_ENV` is set (emergency rollback
    without a revert). Every candidate returned here must STILL pass the reap
    gates in ``clean_issue_downloads`` (recency, nested-durable, positive
    re-downloadability evidence — staging candidates additionally the
    cross-issue content gate 1.55) before anything is deleted."""
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
            if exclude_scratch_shapes and is_tmp_scratch_name(child.name):
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
    out.extend(_staging_cache_dirs(issue_n, staging_roots))
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
    staging: bool = False,
) -> str | None:
    """Gate 1.5 (NEW non-canonical candidates only): SKIP reason when the
    tree's newest touch time (:func:`_dir_max_recency` — file
    ``max(st_mtime, st_atime)``, dir/symlink mtime) is within
    ``min_age_hours`` — a live reader OR writer (e.g. an inline free-analysis
    on a terminal task) may hold it; /tmp paths are never declared in plans,
    so the #773 consumer gate cannot see those readers and recency is the
    only signal. Sidecar kind ``noncanonical-cache-recent-kept``.
    Fail-toward-keep on stat errors. STAGING candidates (#2095) read
    TOP-LEVEL-ONLY recency (no rglob over a multi-100-GB staging tree)."""
    newest = _dir_max_recency(cache_dir, top_level_only=staging)
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


def _staging_cross_issue_content(cache_dir: Path, issue_n: int) -> list[str]:
    """Gate 1.55 scan (#2095): relative paths (depth <= 2, ``iterdir`` only —
    never a full rglob) of entries whose NAME keys a DIFFERENT issue than
    ``issue_n`` (P1 regex, M != issue_n — the dir-named-for-A-holding-B's-
    tensors shape). Fail-toward-keep on OSError: an unreadable level
    contributes an ``<unreadable>`` sentinel entry so the caller BLOCKS.
    Symlinked subdirs are not descended (a link can point into another
    issue's real tree; the link's own depth-1 NAME is still scanned);
    same-issue names contribute nothing."""

    def _foreign(name: str) -> bool:
        m = _TMP_ISSUE_PREFIX_RE.match(name)
        return m is not None and int(m.group(1)) != issue_n

    try:
        depth1 = sorted(cache_dir.iterdir())
    except OSError:
        return [f"{cache_dir.name}/<unreadable>"]
    hits: list[str] = []
    for child in depth1:
        if _foreign(child.name):
            hits.append(child.name)
            continue
        try:
            if child.is_dir() and not child.is_symlink():
                hits.extend(
                    f"{child.name}/{sub.name}"
                    for sub in sorted(child.iterdir())
                    if _foreign(sub.name)
                )
        except OSError:
            hits.append(f"{child.name}/<unreadable>")
    return hits


def _staging_unverified_class(
    cache_dir: Path, data_repo_toplevel: frozenset[str] | None
) -> str | None:
    """Escalation class label for a STAGING candidate gate 1.7 refused (#2095
    delta 5): ``derived-partial-mirror`` when >=1 top-level name matches a
    data-repo prefix (the audit's dominant ESCALATE class — derived from
    verified-mirrored HF inputs, regenerable, not itself mirrored) vs
    ``orphan-no-mirror`` (0 matched). ``None`` when the listing is
    unavailable (fetch failed — no branch-(b) match counts exist to
    classify) or the dir is unreadable; the caller then keeps the plain
    ``unverified-kept`` disposition. Name rules mirror branch (b)'s (the
    ``.cache`` hub-client bookkeeping dir is ignored)."""
    if data_repo_toplevel is None:
        return None
    try:
        names = [
            c.name
            for c in cache_dir.iterdir()
            if not (c.name == ".cache" and c.is_dir() and not c.is_symlink())
        ]
    except OSError:
        return None
    matched = sum(1 for n in names if n in data_repo_toplevel)
    return "derived-partial-mirror" if matched >= 1 else "orphan-no-mirror"


def _staging_probe_floor_bytes() -> int:
    """Sampled-mirror-probe floor in bytes (#2095 delta 9; env
    ``EPM_STAGING_EVIDENCE_PROBE_FLOOR_GB``, invalid/negative -> the
    :data:`STAGING_EVIDENCE_PROBE_FLOOR_GB` default)."""
    raw = os.environ.get("EPM_STAGING_EVIDENCE_PROBE_FLOOR_GB", "").strip()
    gb = STAGING_EVIDENCE_PROBE_FLOOR_GB
    if raw:
        try:
            val = float(raw)
        except ValueError:
            val = -1.0
        if val >= 0.0:
            gb = val
    return int(gb * 1e9)


def _hf_path_size(rel: str) -> int | None:
    """Byte size of ``rel`` on the HF data repo (one ``get_paths_info`` call
    — per-path POST, safe on the ~1M-file repo), or ``None`` when the path is
    absent. Raises on transport/auth errors — the probe caller treats ANY
    exception as refusal (fail-toward-keep)."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for info in api.get_paths_info(hf_data_repo(), [rel], repo_type="dataset"):
        if str(getattr(info, "path", "")) == rel:
            size = getattr(info, "size", None)
            return int(size) if size is not None else None
    return None


def _staging_mirror_probe(cache_dir: Path) -> str | None:
    """Delta-9 sampled mirror probe (#2095) on a branch-(b)-LICENSED staging
    candidate above :func:`_staging_probe_floor_bytes`. Branch (b) is
    NAME-level prefix matching only, so a SAME-issue partial mirror (a dir
    whose top-level names match repo prefixes while holding local-only
    files) would be licensed by name coincidence — unacceptable when the
    license triggers rmtree of a multi-GB tree. The probe locates the
    LARGEST regular file under the candidate (one rglob — only on
    already-licensed candidates; the rmtree that would follow walks the same
    tree anyway), maps its cache-relative path onto the data repo, and
    verifies existence + byte-equal size. Returns ``None`` when the license
    STANDS (byte-equal mirror hit, or no regular file exists — nothing at
    risk beyond empty dirs), else a refusal reason string; ANY exception is
    a refusal (fail-toward-keep)."""
    try:
        largest: Path | None = None
        largest_size = -1
        for p in cache_dir.rglob("*"):
            try:
                st = p.lstat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_size > largest_size:
                largest, largest_size = p, st.st_size
        if largest is None:
            return None  # no regular file — nothing at risk beyond empty dirs
        rel = largest.relative_to(cache_dir).as_posix()
        size = _hf_path_size(rel)
        if size == largest_size:
            return None
        detail = "absent from" if size is None else f"{size} B on"
        return f"largest local file {rel} ({largest_size} B) is {detail} {hf_data_repo()}"
    except Exception as exc:
        return f"probe error ({type(exc).__name__}: {exc})"


# ─── #2127 top-level /tmp gate/smoke scratch sweep — helpers ─────────────────


def is_tmp_scratch_name(name: str) -> bool:
    """True iff a top-level ``/tmp`` entry NAME is a gate/smoke scratch
    candidate (#2127): it matches one of :data:`_SCRATCH_SHAPE_GLOBS` AND
    none of :data:`_SCRATCH_DENYLIST_GLOBS`. The denylist is checked FIRST
    so the live Claude task-output tree (``claude-*``) and live pytest
    basetemps (``pytest-of-*``) stay unreachable even if a future shape
    glob widens."""
    if any(fnmatch.fnmatchcase(name, deny) for deny in _SCRATCH_DENYLIST_GLOBS):
        return False
    return any(fnmatch.fnmatchcase(name, g) for g in _SCRATCH_SHAPE_GLOBS)


def tmp_scratch_sweep_enabled() -> bool:
    """Two-layer kill switch for the scratch leg (#2127, mirrors the #2095
    pattern): runs only when BOTH the family switch
    (:data:`NONCANONICAL_SWEEP_KILL_ENV`) and the leg's own switch
    (:data:`SCRATCH_SWEEP_KILL_ENV`) are unset."""
    if os.environ.get(NONCANONICAL_SWEEP_KILL_ENV, "").strip():
        return False
    return not os.environ.get(SCRATCH_SWEEP_KILL_ENV, "").strip()


def scratch_verdict_cache_path() -> Path:
    """Production location of the scratch-sweep verdict cache
    (``<main checkout>/.claude/cache/scratch-verify-cache.json``).

    Env ``EPS_SCRATCH_VERDICT_CACHE`` overrides the location (#2147: the
    report-mode acceptance run must never WRITE the production cache —
    report mode legitimately caches verify work, so the override redirects
    the cache instead of disabling it).

    main()-ONLY opt-in, the same hermeticity contract as
    :func:`production_tmp_root`: library callers default to ``None`` (no
    cache reads/writes), so tests never touch — nor are influenced by —
    the real cache. AST-pinned by
    ``tests/test_janitor_noncanonical_caches.py::test_production_tmp_root_only_in_mains``."""
    raw = os.environ.get("EPS_SCRATCH_VERDICT_CACHE", "").strip()
    if raw:
        return Path(raw)
    return _resolution_root() / SCRATCH_VERDICT_CACHE_REL


def _scratch_verify_cap_bytes() -> int:
    """Per-candidate cap on bytes HASHED for blob verification (#2127); a
    candidate whose non-exempt regular bytes exceed it is KEPT unverified
    (``over-verify-cap``). Env ``EPS_SCRATCH_VERIFY_MAX_GB``."""
    raw = os.environ.get("EPS_SCRATCH_VERIFY_MAX_GB", "").strip()
    try:
        gb = float(raw) if raw else SCRATCH_VERIFY_MAX_GB_DEFAULT
    except ValueError:
        gb = SCRATCH_VERIFY_MAX_GB_DEFAULT
    return int(gb * 1e9)


def _scratch_escalate_floor_bytes() -> int:
    """Size floor for KEEP-row sidecar escalation (#2127) — every candidate
    still appears in the report/JSON rows; only the durable sidecar rows are
    floor-gated. Env ``EPS_SCRATCH_ESCALATE_FLOOR_GB``."""
    raw = os.environ.get("EPS_SCRATCH_ESCALATE_FLOOR_GB", "").strip()
    try:
        gb = float(raw) if raw else SCRATCH_ESCALATE_FLOOR_GB_DEFAULT
    except ValueError:
        gb = SCRATCH_ESCALATE_FLOOR_GB_DEFAULT
    return int(gb * 1e9)


def _scratch_is_exempt_rel(rel_parts: tuple[str, ...]) -> bool:
    """True when a candidate-relative path sits under (or is) an exempt
    tool-state dir (:data:`_SCRATCH_EXEMPT_DIR_NAMES`), or is the root-level
    ``.git`` worktree-pointer FILE (pure git admin metadata — rebuildable
    via ``git worktree repair``; its content is deliberately not in any
    odb)."""
    if any(part in _SCRATCH_EXEMPT_DIR_NAMES for part in rel_parts[:-1]):
        return True
    if rel_parts and rel_parts[-1] in _SCRATCH_EXEMPT_DIR_NAMES:
        # a DIR named e.g. ``.git`` — callers pass dir paths too; and the
        # root-level ``.git`` regular FILE (worktree pointer) lands here.
        return True
    return False


def _scratch_walk_stats(cand: Path) -> dict | None:
    """One defensive lstat-classified walk of a scratch candidate (#2127).

    Returns a dict with:

    - ``newest_mtime`` — max mtime over the root dir, EVERY subdir (exempt
      dirs included: a mid-build ``.venv`` reads as a live writer) and
      every regular file;
    - ``newest_nonexempt_mtime`` — the same max EXCLUDING exempt-dir
      content (the reap re-check key: the sweep's own git probes rewrite a
      clone's in-tree ``.git/index`` mid-verification, which must not
      self-abort the reap — a real mid-verify writer still bumps non-exempt
      file/dir mtimes, and a live exempt-dir writer holds a process the
      live-process probe sees);
    - ``newest_reader_atime`` — max atime over regular files with
      ``st_nlink == 1`` that are NOT exempt (uv-hardlinked ``.venv`` files
      share atimes with the global uv cache; the janitor's own git probes
      refresh ``.git`` atimes — neither is tree-READER evidence), or
      ``None`` when no such file exists;
    - ``nonregular`` — str path of the first FIFO/socket/device found
      anywhere, else ``None`` (symlinks are NOT non-regular: skipped —
      never followed, never opened — and contribute no recency);
    - ``total_bytes`` / ``nonexempt_bytes`` / ``n_regular`` /
      ``n_symlinks`` — tree totals (``n_regular`` counts EVERY regular
      file, exempt included, so the empty-tree carve-out stays strict).

    Returns ``None`` on ANY walk/stat error (fail-toward-keep: callers
    treat an unreadable tree as KEEP)."""
    walk_errors: list[OSError] = []
    try:
        root_st = cand.lstat()
        if not stat.S_ISDIR(root_st.st_mode):
            return None
        newest_mtime = root_st.st_mtime
        newest_nonexempt_mtime = root_st.st_mtime
        newest_reader_atime: float | None = None
        nonregular: str | None = None
        total_bytes = 0
        nonexempt_bytes = 0
        n_regular = 0
        n_symlinks = 0
        for dirpath, dirnames, filenames in os.walk(
            cand, topdown=True, onerror=walk_errors.append, followlinks=False
        ):
            if walk_errors:
                return None
            dp = Path(dirpath)
            rel_dir_parts = dp.relative_to(cand).parts
            for name in list(dirnames):
                try:
                    st = (dp / name).lstat()
                except OSError:
                    return None
                if stat.S_ISLNK(st.st_mode):
                    dirnames.remove(name)  # never follow/descend a dir symlink
                    n_symlinks += 1
                    continue
                newest_mtime = max(newest_mtime, st.st_mtime)
                if not _scratch_is_exempt_rel((*rel_dir_parts, name)):
                    newest_nonexempt_mtime = max(newest_nonexempt_mtime, st.st_mtime)
            for name in filenames:
                p = dp / name
                try:
                    st = p.lstat()
                except OSError:
                    return None
                mode = st.st_mode
                if stat.S_ISLNK(mode):
                    n_symlinks += 1
                    continue
                if not stat.S_ISREG(mode):
                    if nonregular is None:
                        nonregular = str(p)
                    continue
                n_regular += 1
                total_bytes += st.st_size
                newest_mtime = max(newest_mtime, st.st_mtime)
                if _scratch_is_exempt_rel((*rel_dir_parts, name)):
                    continue
                newest_nonexempt_mtime = max(newest_nonexempt_mtime, st.st_mtime)
                nonexempt_bytes += st.st_size
                if st.st_nlink == 1 and (
                    newest_reader_atime is None or st.st_atime > newest_reader_atime
                ):
                    newest_reader_atime = st.st_atime
        if walk_errors:
            return None
        return {
            "newest_mtime": newest_mtime,
            "newest_nonexempt_mtime": newest_nonexempt_mtime,
            "newest_reader_atime": newest_reader_atime,
            "nonregular": nonregular,
            "total_bytes": total_bytes,
            "nonexempt_bytes": nonexempt_bytes,
            "n_regular": n_regular,
            "n_symlinks": n_symlinks,
        }
    except Exception:
        return None


def _open_scratch_regular(path: Path) -> tuple[int, os.stat_result] | None:
    """Open a regular file for hashing without following symlinks and
    without blocking (``O_RDONLY | O_NOFOLLOW | O_NONBLOCK``, plus
    ``O_NOATIME`` where permitted — retried without it on ``EPERM``), then
    re-verify via ``fstat`` that the OPENED object is still a regular file
    (an lstat->open race swap to a FIFO would otherwise wedge or misread —
    ``O_NONBLOCK`` makes even that open non-blocking). Returns
    ``(fd, fstat)`` or ``None`` (callers fail toward keep). The caller owns
    closing ``fd``."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd: int | None = None
    try:
        try:
            fd = os.open(path, flags | getattr(os, "O_NOATIME", 0))
        except PermissionError:
            fd = os.open(path, flags)  # O_NOATIME requires file ownership
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            return None
        return fd, st
    except OSError:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        return None


def _blob_sha1_from_fd(fd: int, size: int) -> str | None:
    """git blob sha1 of exactly ``size`` bytes read from ``fd``
    (``sha1(b"blob %d\\0" + content)`` — git's own object id, sha1 object
    format only). Returns ``None`` when the file yields FEWER bytes than
    ``size`` or still has bytes PAST it (a concurrent truncate/grow race:
    the hash would name a different object than the one stat'd —
    fail-toward-keep)."""
    h = hashlib.sha1(b"blob %d\x00" % size)
    remaining = size
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 1 << 20))
        if not chunk:
            return None  # shrank under us
        h.update(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        return None  # grew under us
    return h.hexdigest()


def _git(
    args: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess | None:
    """Run ``git <args>`` in ``cwd``. Returns the completed process, or
    ``None`` on timeout / OSError (callers fail toward keep)."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _git_bytes(
    args: list[str],
    *,
    cwd: Path,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess | None:
    """Run ``git <args>`` in ``cwd`` with BINARY (bytes) output — the
    PATH-PRODUCING sibling of :func:`_git` (#2147 round 8). ``text=True``
    applies universal newlines, which silently rewrites a CR / CRLF inside
    an emitted PATH to LF — a ghost path in the registration layer.
    Callers decode via :func:`_decode_git_path`. Returns the completed
    process, or ``None`` on timeout / OSError (callers fail toward keep).
    A SIBLING by design: shared :func:`_git`'s signature and its tier-(f)
    call shapes stay byte-identical (plan invariant K3)."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _decode_git_path(stdout: bytes) -> str | None:
    """Decode ONE path-producing git stdout captured by :func:`_git_bytes`:
    UTF-8 decode, then strip exactly ONE trailing LF (git terminates the
    value with a single ``\\n``; every OTHER byte — CR and edge whitespace
    included — is path content that ``.strip()`` would destroy, #2147
    round 8). Returns ``None`` on undecodable or empty output (ambiguity —
    callers fail toward keep)."""
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if text.endswith("\n"):
        text = text[:-1]
    return text or None


def _git_first_missing_blob(main_repo: Path, shas: list[str]) -> int | None:
    """Batched blob-existence probe against ``main_repo``'s object database
    (``git cat-file --batch-check``, ~200 shas per call, short-circuiting).
    Returns ``-1`` when EVERY sha exists as a blob, the index of the first
    missing/non-blob sha otherwise, or ``None`` on any probe failure
    (fail-toward-keep)."""
    for start in range(0, len(shas), 200):
        chunk = shas[start : start + 200]
        proc = _git(
            ["cat-file", "--batch-check=%(objectname) %(objecttype)"],
            cwd=main_repo,
            input_text="".join(s + "\n" for s in chunk),
        )
        if proc is None or proc.returncode != 0:
            return None
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if len(lines) != len(chunk):
            return None
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 2 or parts[1] != "blob":
                return start + i
    return -1


def _git_dir_kind(cand: Path) -> tuple[str, Path | None]:
    """Classify a candidate's git nature from its top-level ``.git`` entry:

    - ``("none", None)`` — no ``.git`` at the candidate root;
    - ``("worktree", admin_dir)`` — ``.git`` is a regular FILE with a
      parseable ``gitdir:`` pointer to an existing admin dir;
    - ``("clone", git_dir)`` — ``.git`` is a real directory;
    - ``("unknown", None)`` — anything else (symlink ``.git``, unreadable,
      unparseable pointer, dangling admin dir). Callers KEEP on unknown."""
    dotgit = cand / ".git"
    try:
        st = dotgit.lstat()
    except FileNotFoundError:
        return "none", None
    except OSError:
        return "unknown", None
    if stat.S_ISDIR(st.st_mode):
        return "clone", dotgit
    if not stat.S_ISREG(st.st_mode):
        return "unknown", None
    try:
        text = dotgit.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "unknown", None
    if not text.startswith("gitdir:"):
        return "unknown", None
    admin = Path(text[len("gitdir:") :].strip())
    if not admin.is_absolute():
        admin = cand / admin
    try:
        admin = admin.resolve()
        if not admin.is_dir():
            return "unknown", None
    except OSError:
        return "unknown", None
    return "worktree", admin


def _worktree_admin_of_main(admin: Path, main_repo: Path) -> bool:
    """True iff ``admin`` is a REGISTERED linked-worktree admin dir of
    ``main_repo`` (realpath under ``<git-common-dir>/worktrees/``). The
    common-dir path is read BYTE-EXACTLY (#2147 round 8 audit sibling: the
    text-mode ``.strip()`` read mangled a CR / edge-whitespace repo path,
    misclassifying every genuinely-ours admin dir as foreign — a
    KEEP-direction failure, but a WRONG answer from an authoritative
    layer-1 probe). Any probe/decode failure reads as not-ours (callers
    treat that as foreign ⇒ KEEP)."""
    proc = _git_bytes(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=main_repo)
    if proc is None or proc.returncode != 0:
        return False
    common = _decode_git_path(proc.stdout)
    if common is None:
        return False
    try:
        admin_real = admin.resolve()
        wt_root = (Path(common) / "worktrees").resolve()
    except OSError:
        return False
    return wt_root in admin_real.parents


def _candidate_worktree_registration(cand: Path, main_repo: Path) -> tuple[str, Path | None]:
    """PARSE-FREE per-candidate registration probe (#2147 round 6): reads
    the CANDIDATE's own ``.git`` entry — never the newline-delimited
    porcelain listing — so a worktree at ANY byte-exact path (embedded
    newline included) is classified from its gitfile alone. The candidate
    path is already known as a real path; nothing is ever recovered from a
    listing. Classes:

    - ``("ours", admin)`` — ``.git`` is a regular file whose ``gitdir:``
      pointer (relative pointers resolved against the candidate dir;
      realpath — never string-prefix matching on unresolved paths) names an
      existing dir whose parent is THIS repo's
      ``<git-common-dir>/worktrees`` (:func:`_worktree_admin_of_main`);
    - ``("foreign", None)`` — a ``…/worktrees/<id>`` admin dir of some
      OTHER repo (registered elsewhere — never ours to delete);
    - ``("submodule", None)`` — resolves under a ``…/modules/…`` component:
      a submodule gitfile is NOT a worktree registration;
    - ``("clone", None)`` — ``.git`` is a real directory;
    - ``("none", None)`` — no ``.git`` entry at all (candidate-side
      evidence structurally absent — the R3-C1 pointer-deleted downgrade;
      registration must then be proven ADMIN-side,
      :func:`_admin_registered_worktree_paths`);
    - ``("unreadable", None)`` — everything else (I/O error, symlink
      ``.git``, unparseable pointer, dangling admin path, a gitdir target
      that is neither a worktree admin nor a module). Callers KEEP
      (fail-closed)."""
    dotgit = cand / ".git"
    try:
        st = dotgit.lstat()
    except FileNotFoundError:
        return "none", None
    except OSError:
        return "unreadable", None
    if stat.S_ISDIR(st.st_mode):
        return "clone", None
    if not stat.S_ISREG(st.st_mode):
        return "unreadable", None
    try:
        # BINARY read + explicit decode (#2147 round 7 discipline): never let
        # universal newlines rewrite path bytes inside the pointer value. A
        # CR-corrupted value here fails toward "unreadable" (KEEP) rather
        # than open, but the read discipline is uniform across every
        # registration-feeding file read this task added.
        text = dotgit.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return "unreadable", None
    if not text.startswith("gitdir:"):
        return "unreadable", None
    value = text[len("gitdir:") :].strip()
    if not value:
        return "unreadable", None
    admin = Path(value)
    if not admin.is_absolute():
        admin = cand / admin
    try:
        admin = Path(os.path.realpath(admin))
        if not admin.is_dir():
            return "unreadable", None
    except OSError:
        return "unreadable", None
    parts = admin.parts
    if len(parts) >= 2 and parts[-2] == "worktrees":
        if _worktree_admin_of_main(admin, main_repo):
            return "ours", admin
        return "foreign", None
    if "modules" in parts:
        return "submodule", None
    return "unreadable", None


def _admin_registered_worktree_paths(main_repo: Path) -> frozenset[str] | None:
    """PARSE-FREE admin-side registration enumeration (#2147 round 6): the
    byte-exact registered-path set the porcelain listing structurally
    cannot deliver. git stores ONE ``gitdir`` FILE per linked worktree at
    ``<git-common-dir>/worktrees/<id>/gitdir`` whose content is
    ``<worktree>/.git`` plus exactly one trailing newline — a per-record
    FILE, not a newline-delimited list — so a worktree path containing ANY
    byte (embedded newline included) is recovered exactly: read the file,
    strip ONE trailing newline, take the dirname. The main working tree
    (``rev-parse --show-toplevel``) is included. Returns realpaths, or
    ``None`` on ANY probe failure (fail-toward-keep): failed rev-parse,
    undecodable/empty rev-parse output, MISSING git common dir (an
    unresolvable scan root — round 8), unreadable ``worktrees/`` dir,
    unreadable/empty ``gitdir`` file.

    Round 8 (round-5 cap residual, reproduced live): BOTH rev-parse
    outputs are PATHS, so they are read through :func:`_git_bytes` — the
    text-mode pipe translated a CR/CRLF inside the REPOSITORY'S OWN path
    to LF, and ``.strip()`` ate genuine edge-whitespace path bytes; the
    derived ``worktrees/`` root then named a directory that does not
    exist, the ``FileNotFoundError`` was swallowed into ``entries = []``,
    and the function returned a SUCCESSFUL-LOOKING set missing every real
    registration. An INCOMPLETE authoritative set must be DISTINGUISHABLE
    from "no linked worktrees": a missing COMMON DIR (the scan root's
    parent) is ambiguity ⇒ ``None``; only a PRESENT common dir with an
    absent ``worktrees/`` subdirectory is the legitimate empty shape.

    This is the AUTHORITATIVE registration source for a candidate with NO
    usable ``.git`` entry (deleted/replaced pointer — the R3-C1 downgrade
    class), where :func:`_candidate_worktree_registration` is structurally
    blind and the porcelain listing is poisoned by any newline-bearing
    registered path (round-5 residual, coordinator-reproduced: a decoy dir
    at the TRUNCATION of a newline path makes the hardened listing parse
    "successfully" while the real registered path is absent)."""
    common_proc = _git_bytes(
        ["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=main_repo
    )
    if common_proc is None or common_proc.returncode != 0:
        return None
    top_proc = _git_bytes(["rev-parse", "--path-format=absolute", "--show-toplevel"], cwd=main_repo)
    if top_proc is None or top_proc.returncode != 0:
        return None
    common_out = _decode_git_path(common_proc.stdout)
    top_out = _decode_git_path(top_proc.stdout)
    if common_out is None or top_out is None:
        return None  # undecodable/empty path output — ambiguity keeps
    paths: set[str] = set()
    try:
        common_dir = Path(common_out)
        if not common_dir.is_dir():
            # Round 8: the SCAN ROOT'S PARENT is missing — an unresolvable
            # (or externally mutated) common dir is AMBIGUITY, never "no
            # worktrees". Pre-fix this state was swallowed into
            # ``entries = []``, returning a successful-looking INCOMPLETE
            # set in a code path that licenses deletion.
            return None
        paths.add(os.path.realpath(top_out))
        wt_root = common_dir / "worktrees"
        try:
            entries = list(os.scandir(wt_root))
        except FileNotFoundError:
            # Common dir PRESENT, ``worktrees/`` absent — the one
            # legitimate "no linked worktrees registered" shape.
            entries = []
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue  # stray non-dir in worktrees/ — not a registration
            # BINARY read + explicit decode (#2147 round 7, Codex R4-1):
            # ``read_text`` uses universal newlines, silently translating a
            # CR / CRLF inside the registered PATH into LF — a GHOST path
            # enters the set and the REAL registration goes missing, failing
            # this AUTHORITATIVE layer open. Decode failure ⇒ ``None``.
            content = (Path(entry.path) / "gitdir").read_bytes().decode("utf-8")
            if content.endswith("\n"):
                content = content[:-1]  # exactly ONE trailing LF; embedded CR/LF are path bytes
            if not content:
                return None  # malformed registration — ambiguity keeps
            pointer = Path(content)
            if not pointer.is_absolute():
                pointer = Path(entry.path) / pointer
            paths.add(os.path.realpath(pointer.parent))
    except (OSError, UnicodeDecodeError):
        return None  # any unreadable registration poisons the proof — KEEP
    return frozenset(paths)


# `git worktree list --porcelain` record keys (#2147 round 4). Value keys
# carry ` <payload>`; flag keys are bare, with `locked`/`prunable` optionally
# carrying a ` <reason>` suffix. `branch` and `detached` are mutually
# exclusive within a record, tracked under one slot ("headstate").
_WORKTREE_PORCELAIN_VALUE_KEYS: dict[str, str] = {"HEAD ": "HEAD", "branch ": "headstate"}
_WORKTREE_PORCELAIN_FLAG_KEYS: dict[str, str] = {
    "detached": "headstate",
    "bare": "bare",
    "locked": "locked",
    "prunable": "prunable",
}


def _registered_worktree_paths(main_repo: Path) -> frozenset[str] | None:
    """Realpaths of EVERY working tree registered to ``main_repo`` — the main
    working tree included — from ONE ``git worktree list --porcelain`` query
    (#2147 review round 3 C1: the POSITIVE non-registration proof consulted
    before any ``shutil.rmtree``). The listing is read from the admin-dir
    metadata, so a registered worktree whose in-tree ``.git`` pointer was
    deleted or replaced STILL appears (it merely shows as prunable) — exactly
    the state the candidate-side class probes cannot see.

    Round 4 (R3-C1/SIB-1): the porcelain is parsed in RECORD form and FAILS
    CLOSED on ambiguity. git 2.34.1 (verified by live reproduction) emits
    worktree paths RAW — space, tab, backslash, and double-quote all survive
    unescaped on one line, and there is NO C-quoting — but a path containing
    a NEWLINE necessarily SPLITS its record: the ``worktree`` line carries a
    TRUNCATED path and the remainder lands as an orphan continuation line.
    The previous line-wise parse recorded the truncated path, so the REAL
    registered path compared unequal downstream and a REGISTERED worktree
    could reach ``shutil.rmtree``. Record-form contract: a record OPENS with
    ``worktree <path>`` (path taken VERBATIM after the prefix — no strip;
    leading/trailing whitespace is part of the path) and CLOSES at a blank
    line; inside a record the only recognized lines are the porcelain keys
    ``HEAD <oid>``, ``branch <ref>`` XOR ``detached``, ``bare``,
    ``locked[ <reason>]``, ``prunable[ <reason>]`` — each slot at most once.
    ANY other non-blank line — an orphan continuation from a newline-bearing
    path, an attribute outside a record, a duplicated slot (a truncated
    record absorbing the real record's attributes) — makes the WHOLE listing
    AMBIGUOUS: return ``None`` (every caller treats ``None`` as
    refuse-to-reap).

    Round 5 (coordinator repro — REAL git, no adversary): a continuation
    line that exactly spells a recognized flag the record does not otherwise
    carry (a path embedding ``\\nbare``) passes the slot rules with a
    TRUNCATED path — ``bare`` + ``detached`` coexist because a genuine
    detached record simply lacks ``bare``. Closed by a positive EXISTENCE
    cross-check at record close: every parsed ``worktree`` path must exist
    on disk as a directory; a missing directory is tolerated ONLY when its
    own record carries ``prunable`` (git 2.34.1 verified: a deleted worktree
    dir lists with ``prunable gitdir file points to non-existent
    location``) — any other missing path makes the listing AMBIGUOUS:
    return ``None``. Remaining named residual (three-way collision only): a
    newline-bearing registered path whose TRUNCATION names a directory that
    ALSO exists on disk, with the continuation spelling an absent flag —
    that truncated record passes both the slot rules and the existence
    check. Round 6 (coordinator-reproduced as exploitable): that residual
    no longer licenses anything — this function is DEFENCE-IN-DEPTH ONLY
    (its output can only ADD keeps) behind the authoritative parse-free
    per-candidate probe (:func:`_candidate_worktree_registration`) and the
    byte-exact admin-side enumeration
    (:func:`_admin_registered_worktree_paths`), both consulted FIRST at
    every deletion-licensing site.

    Returns ``None`` on probe failure OR parse ambiguity: a successful
    listing always contains at least the main working tree, so an empty
    parse is ambiguity, never evidence of non-registration (fail-toward-keep).
    Residual (named): registration in some OTHER repo is undetectable once
    the candidate's ``.git`` pointer is gone — there is no back-pointer left
    to follow; this proof covers ``main_repo``'s registrations, the set this
    janitor can strand."""
    # KNOWN CR/CRLF BLINDNESS, deliberately unfixed (#2147 round 7): ``_git``
    # runs with ``text=True`` (universal newlines), so a CR inside a
    # registered path is translated to LF before this parse — typically
    # splitting the record (=> None) or, in the CR-flag-spoof shape, yielding
    # a truncated-path record. Acceptable ONLY because this layer is
    # KEEP-ONLY: both consumers consult it AFTER the authoritative parse-free
    # sources (candidate gitfile probe + binary-read admin enumeration), and
    # its two live outcomes are KEEP (None / membership) or defer — a ghost
    # path here can fail to ADD a keep but can never LICENSE a reap.
    # ``_git`` itself is shared with tier (f) and stays untouched (K3).
    proc = _git(["worktree", "list", "--porcelain"], cwd=main_repo)
    if proc is None or proc.returncode != 0:
        return None
    paths: set[str] = set()
    pending: str | None = None  # raw path of the OPEN record, pre-validation
    seen_slots: set[str] = set()

    def _close_record() -> bool:
        """Validate + commit the open record; ``False`` = ambiguous listing.

        Round 5 existence cross-check: the raw path must exist on disk as a
        directory (a spoof-truncated path does not), tolerating a missing
        directory ONLY for a ``prunable`` record (a genuinely pruned/deleted
        worktree — the one legitimate missing-dir shape)."""
        nonlocal pending
        if pending is None:
            return True  # nothing open (leading/consecutive blank lines)
        if not os.path.isdir(pending) and "prunable" not in seen_slots:
            return False
        try:
            paths.add(os.path.realpath(pending))
        except OSError:
            return False
        pending = None
        return True

    # split("\n"), never splitlines(): splitlines() also splits on \x0b/\x0c/
    # U+2028/... which git emits RAW inside a path — splitting there would
    # only widen the fail-closed surface, but split("\n") parses them exactly.
    for line in proc.stdout.split("\n"):
        if not line:  # blank separator closes the current record
            if not _close_record():
                return None
            seen_slots = set()
            continue
        if line.startswith("worktree "):
            if pending is not None:
                return None  # record re-opened without a separator — ambiguous
            raw = line[len("worktree ") :]
            if not raw:
                return None  # malformed entry — ambiguity keeps
            pending = raw
            seen_slots = set()
            continue
        slot = next(
            (s for k, s in _WORKTREE_PORCELAIN_VALUE_KEYS.items() if line.startswith(k)),
            None,
        )
        if slot is None:
            slot = next(
                (
                    s
                    for k, s in _WORKTREE_PORCELAIN_FLAG_KEYS.items()
                    if line == k or line.startswith(k + " ")
                ),
                None,
            )
        if slot is None or pending is None or slot in seen_slots:
            # Unrecognized continuation (a newline-split path), an attribute
            # outside any record, or a duplicated slot: AMBIGUOUS — refuse
            # rather than trust a possibly-truncated set.
            return None
        seen_slots.add(slot)
    if not _close_record():
        return None  # final record failed the existence cross-check
    if not paths:
        return None  # no worktree lines at all — ambiguity, not proof
    return frozenset(paths)


def _reachable_in_main(main_repo: Path, sha: str) -> bool:
    """True iff ``sha`` is reachable from a SURVIVING ref of ``main_repo``:
    the fast path ``merge-base --is-ancestor <sha> origin/main`` (rc==0),
    else a non-empty ``for-each-ref --contains`` over heads/tags/remotes.
    Any probe failure reads as unreachable (fail-toward-keep)."""
    proc = _git(["merge-base", "--is-ancestor", sha, "origin/main"], cwd=main_repo)
    if proc is not None and proc.returncode == 0:
        return True
    proc = _git(
        [
            "for-each-ref",
            "--format=%(refname)",
            f"--contains={sha}",
            "refs/heads",
            "refs/tags",
            "refs/remotes",
        ],
        cwd=main_repo,
    )
    return proc is not None and proc.returncode == 0 and bool(proc.stdout.strip())


def _scratch_git_class_probes(
    cand: Path, kind: str, admin: Path | None, *, main_repo: Path
) -> str | None:
    """Class-discriminated git-state safety probes (#2127). Returns ``None``
    when the candidate's git metadata carries NOTHING the main repo would
    lose on rmtree, else a keep-reason slug (fail-toward-keep on every
    probe failure).

    WORKTREE class (``.git`` file -> registered admin dir; object database
    lives in the MAIN repo, so only worktree-LOCAL state matters): must be
    a main-repo registration, unlocked, ``status --porcelain`` empty, HEAD
    reachable from a surviving main-repo ref. The SHARED stash is
    deliberately NOT probed — stash entries live in the main odb and
    survive the worktree (#2127 plan §12).

    CLONE class (``.git`` DIR — its whole odb dies with the tree): status
    empty, OWN stash empty, every ref tip + HEAD reachable from the main
    repo's surviving refs (reachability, not mere object presence — a
    present commit object does not imply its tree/parents are)."""
    if kind == "none":
        return None
    if kind != "worktree" and kind != "clone":
        return "git-entry-unrecognized"
    if kind == "worktree":
        if admin is None or not _worktree_admin_of_main(admin, main_repo):
            return "foreign-worktree"
        try:
            if (admin / "locked").exists():
                return "worktree-locked"
        except OSError:
            return "git-probe-failed"
    # NOTE (review round 2): `git status` reads .gitignore files WITHOUT
    # O_NOATIME, so a cache-miss report-only run can atime-pin a would-reap
    # tree for +48h. Bounded: the verdict cache suppresses repeat probes,
    # relatime suppresses <24h refreshes, and the apply path's atime gate
    # decides on PRE-probe stats, so a run can never self-block.
    status = _git(["status", "--porcelain"], cwd=cand)
    if status is None or status.returncode != 0:
        return "git-probe-failed"
    if status.stdout.strip():
        return "worktree-dirty" if kind == "worktree" else "clone-dirty"
    head = _git(["rev-parse", "HEAD"], cwd=cand)
    if head is None or head.returncode != 0:
        return "git-probe-failed"
    head_sha = head.stdout.strip()
    if kind == "worktree":
        if not _reachable_in_main(main_repo, head_sha):
            return "head-unreachable"
        return None
    stash = _git(["stash", "list"], cwd=cand)
    if stash is None or stash.returncode != 0:
        return "git-probe-failed"
    if stash.stdout.strip():
        return "clone-stash"
    tips_proc = _git(["for-each-ref", "--format=%(objectname)"], cwd=cand)
    if tips_proc is None or tips_proc.returncode != 0:
        return "git-probe-failed"
    tips = sorted({*tips_proc.stdout.split(), head_sha})
    if len(tips) > 32:
        return "clone-ref-fanout"
    for sha in tips:
        if not _reachable_in_main(main_repo, sha):
            return "head-unreachable" if sha == head_sha else "clone-unpushed-ref"
    return None


class _ScratchVerdictCache:
    """Fail-soft JSON verdict cache for scratch blob verification (#2127),
    keyed ``<realpath>|<newest_mtime>|<total_bytes>`` so any tree change
    invalidates. Only DEFINITIVE verdicts are stored (PASS /
    ``unverified-file`` / ``tolerance-only`` / ``no-verifiable-content`` /
    ``over-verify-cap``); transient probe errors and git CLASS-probe slugs
    are recomputed every run (they can flip without any tree mtime change —
    e.g. a push makes a HEAD reachable). A cached PASS still EMBEDS the
    class-probe AND overlay conclusions from verify time — reap-ward and
    equally flippable — which is why :func:`_reap_scratch_tree` re-runs the
    class probes + the overlay probe + the round 3 C1 registration proof on
    the destructive path (review rounds 2-3), and why a cached PASS on an
    overlay-bearing leg is overlay-re-probed BEFORE it is honored at all
    (round 3 C2); the cache alone never licenses a deletion. ``prune()``
    drops a reaped path's entries. All IO
    fail-soft: corrupt/unwritable degrades to no-cache.
    Known residuals (plan §12 + review round 2): (a) a cached PASS can
    outlive a later ``git gc`` of the proving blobs — the reap-time
    re-checks cover recency and git class state, not blob existence; the
    proof is recoverability at VERIFY time. (b) the cache key AND the reap
    re-walk are both blind to mtime-preserving SAME-SIZE content overwrites
    (``tar -x``, ``cp -p``, ``rsync --checksum`` over an existing file):
    stale PASS + unchanged key + no fresh mtime => unverified bytes reaped.
    A narrow, deliberate-backdating class — disclosed, not chased."""

    _CACHEABLE = frozenset(
        {"pass", "unverified-file", "tolerance-only", "no-verifiable-content", "over-verify-cap"}
    )

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._data: dict[str, dict] | None = None
        self._dirty = False

    def _load(self) -> dict[str, dict]:
        if self._data is None:
            self._data = {}
            if self.path is not None:
                try:
                    raw = json.loads(self.path.read_text())
                    if isinstance(raw, dict):
                        self._data = {
                            k: v for k, v in raw.items() if isinstance(v, dict) and "detail" in v
                        }
                except (OSError, ValueError):
                    self._data = {}
        return self._data

    @staticmethod
    def _key(cand: Path, stats: dict) -> str:
        return f"{os.path.realpath(cand)}|{stats['newest_mtime']}|{stats['total_bytes']}"

    def lookup(self, cand: Path, stats: dict) -> tuple[str | None, dict] | None:
        """Cached ``(evidence, detail)`` for an unchanged tree, else None."""
        if self.path is None:
            return None
        row = self._load().get(self._key(cand, stats))
        if not isinstance(row, dict) or not isinstance(row.get("detail"), dict):
            return None
        ev = row.get("evidence")
        if ev is not None and not isinstance(ev, str):
            return None
        return ev, dict(row["detail"])

    def store(self, cand: Path, stats: dict, evidence: str | None, detail: dict) -> None:
        """Record a verdict iff its reason class is cacheable."""
        if self.path is None:
            return
        reason = "pass" if evidence is not None else str(detail.get("reason"))
        if reason not in self._CACHEABLE:
            return
        self._load()[self._key(cand, stats)] = {"evidence": evidence, "detail": dict(detail)}
        self._dirty = True

    def prune(self, cand: Path) -> None:
        """Drop every entry keyed on a just-reaped path."""
        if self.path is None:
            return
        prefix = os.path.realpath(cand) + "|"
        data = self._load()
        stale = [k for k in data if k.startswith(prefix)]
        for k in stale:
            del data[k]
        self._dirty = self._dirty or bool(stale)

    def save(self) -> None:
        """Atomic write-back (tmp + ``os.replace``); fail-soft."""
        if self.path is None or not self._dirty or self._data is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self._data, sort_keys=True))
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError:
            pass


def _overlay_member(rel: str, overlay_paths: tuple[str, ...]) -> str | None:
    """The overlay path owning candidate-relative ``rel`` (#2147 D9), or
    None. Membership is "/"-joined path-prefix; a FILE sitting AT the
    overlay path itself is a member too (the nested-repo probe then KEEPs
    the tree as ``overlay-not-a-repo``)."""
    if not overlay_paths:
        return None
    for op in overlay_paths:
        if rel == op or rel.startswith(op + "/"):
            return op
    return None


def _overlay_nested_repo_probe(nested: Path, *, surviving_repo: Path) -> str | None:
    """#2147 D9 nested-repo checks for a ``WORKING_TREE_OVERLAY_PATHS`` copy
    inside a slurm-src candidate (review round 2 C2: the FULL clone-class
    evidence standard, anchored in the SURVIVING repo). The overlay must be
    a real nested git CLONE (its own ``.git`` DIRECTORY) with a clean tree,
    an EMPTY own stash, and EVERY ref tip + HEAD reachable from the refs of
    ``surviving_repo`` — the main working tree's own copy of the overlay
    (``<main_repo>/<overlay path>``), the one repo that SURVIVES the reap.
    Anchoring reachability in the nested copy's own odb would be CIRCULAR:
    that odb is a ``.git`` DIR inside the tree being deleted, so it dies
    with the tree and proves nothing about recoverability (the same rule
    the outer clone class already follows — see
    :func:`_scratch_git_class_probes`).

    Returns ``None`` when all hold, else a keep-reason slug
    (fail-toward-keep on every probe failure). An overlay that is NOT a git
    repo KEEPs the whole tree — deliberately NO silent fallback to the
    outer-odb proof (plan #2147 D9). Reachability of the (possibly many —
    the production overlay carries ~900 refs) nested tips is checked in ONE
    batched ``rev-list`` against the surviving repo's branches/tags/remotes,
    so there is no ref-fanout cap here."""
    kind, _admin = _git_dir_kind(nested)
    if kind != "clone":
        return "overlay-not-a-repo"
    if _git_dir_kind(surviving_repo)[0] != "clone":
        return "overlay-surviving-repo-missing"
    status = _git(["status", "--porcelain"], cwd=nested)
    if status is None or status.returncode != 0:
        return "overlay-probe-failed"
    if status.stdout.strip():
        return "overlay-dirty"
    stash = _git(["stash", "list"], cwd=nested)
    if stash is None or stash.returncode != 0:
        return "overlay-probe-failed"
    if stash.stdout.strip():
        return "overlay-stash"
    head = _git(["rev-parse", "HEAD"], cwd=nested)
    if head is None or head.returncode != 0:
        return "overlay-probe-failed"
    tips_proc = _git(["for-each-ref", "--format=%(objectname)"], cwd=nested)
    if tips_proc is None or tips_proc.returncode != 0:
        return "overlay-probe-failed"
    tips = sorted({*tips_proc.stdout.split(), head.stdout.strip()})
    # One batched reachability probe in the SURVIVING repo: rev-list
    # enumerates ancestors of the nested tips minus ancestors of every
    # surviving ref — empty output (rc 0) iff ALL tips are reachable. A
    # nonzero rc (an object absent from the surviving odb is a "bad
    # revision") or any output line means at least one nested-only ref.
    reach = _git(
        ["rev-list", "--max-count=1", *tips, "--not", "--branches", "--tags", "--remotes"],
        cwd=surviving_repo,
    )
    if reach is None:
        return "overlay-probe-failed"
    if reach.returncode != 0 or reach.stdout.strip():
        return "overlay-unpushed-ref"
    return None


def _overlay_keep_slug(
    cand: Path, *, main_repo: Path, overlay_paths: tuple[str, ...]
) -> tuple[str | None, str | None]:
    """The NON-CACHEABLE overlay presence/state probe (#2147 D9, review
    round 3 C2): every DECLARED overlay path PRESENT on disk under ``cand``
    is validated as a nested repo against its SURVIVING anchor
    (``main_repo / <overlay path>``) via :func:`_overlay_nested_repo_probe`.
    Returns ``(keep_slug, overlay_path)``, or ``(None, None)`` when every
    present overlay positively validates.

    Presence is keyed on the directory entry itself (``lstat``), NOT on
    whether any walk hashed files under it, and the entry MUST be a real
    directory (round 3 C3: ``lstat`` alone followed the parent symlink into
    ``nested/.git`` — a declared overlay that is a symlink, a file, or any
    other non-directory shape is ``overlay-not-a-repo``, never probed
    through). Fail-toward-keep: an ``lstat`` OSError keeps as a walk error.

    Called from THREE sites, deliberately (round 3 C2): the fresh evidence
    walk, the cached-PASS path in :func:`_git_blob_reproducibility_evidence`
    (a cached PASS embeds overlay conclusions that can flip with ZERO
    candidate-tree change — e.g. the surviving anchor's refs), and the
    destructive path in :func:`_reap_scratch_tree` immediately before
    removal."""
    for op in overlay_paths:
        nested = cand / Path(op)
        try:
            st = nested.lstat()
        except FileNotFoundError:
            continue  # overlay not materialized in this tree — nothing to prove
        except OSError:
            return "walk-error", op
        if not stat.S_ISDIR(st.st_mode):
            return "overlay-not-a-repo", op  # symlink/file at the declared path (C3)
        slug = _overlay_nested_repo_probe(nested, surviving_repo=main_repo / Path(op))
        if slug is not None:
            return slug, op
    return None, None


def _git_blob_reproducibility_evidence(
    cand: Path,
    *,
    main_repo: Path,
    full_stats: dict,
    verdict_cache: _ScratchVerdictCache | None = None,
    overlay_paths: tuple[str, ...] = (),
) -> tuple[str | None, dict]:
    """Per-file git-reproducibility proof for a scratch candidate (#2127).

    Returns ``(evidence, detail)``: ``evidence`` is a positive-evidence
    string when EVERY non-exempt, non-tolerated regular file's git blob
    exists in ``main_repo``'s object database (with ``n_verified >= 1``, or
    the strict empty-tree carve-out), else ``None`` with
    ``detail["reason"]`` naming the first blocker (fail-toward-keep on
    every ambiguity). ``detail`` carries ``reason`` / ``first_unverified``
    / ``n_verified`` / ``n_tolerated`` / ``git_class``.

    Gate order: cache hit -> sha1 object-format guard -> git class probes
    (:func:`_scratch_git_class_probes`) -> empty-tree carve-out ->
    verify-cap -> per-file blob walk (exempt dirs pruned; symlinks skipped;
    any non-regular file aborts; small-text tolerance per
    :data:`_SCRATCH_TOLERATED_EXTS`, never under a ``store``/
    ``eval_results`` component) -> batched odb existence probe.

    ``overlay_paths`` (#2147 D9, default empty = pre-#2147 behavior
    byte-identical): files whose candidate-relative path lies under a named
    overlay path (``backends.slurm.WORKING_TREE_OVERLAY_PATHS`` copies —
    working-tree-only nested repos rsync'd into slurm-src staging trees,
    absent from the OUTER committed tree) are proven against the SURVIVING
    overlay repo — the main working tree's own copy at
    ``main_repo / <overlay path>`` — under the FULL clone-class standard
    (clean tree + empty own stash + every ref tip and HEAD reachable in the
    surviving repo; :func:`_overlay_nested_repo_probe`, review round 2 C2:
    the nested copy's own odb dies with the tree, so anchoring there was
    circular). Overlay MEMBERSHIP is classified BEFORE the small-text
    tolerance and BEFORE exemption/symlink skips are allowed to hide the
    overlay (round 2 C3): no file under a claimed overlay is ever tolerated
    into the outer proof, and every DECLARED overlay path present on disk
    is validated as a nested repo even when the walk hashed nothing under
    it (only tolerated logs / exempt content / symlinks / an empty dir) —
    an overlay that is not a positively-established nested git repo KEEPs
    the whole tree; never a fallback to the outer proof. The
    ``under_durable`` no-tolerance rule and the outer-odb proof for
    non-overlay paths are UNCHANGED (strictly additive). Nested-state
    hygiene (round 3 C2): overlay keep slugs are non-cacheable, and a
    cached PASS embeds overlay conclusions that can flip with ZERO
    candidate-tree change (a surviving-anchor ref deleted, the anchor repo
    removed) — so the overlay probe (:func:`_overlay_keep_slug`) re-runs
    BEFORE a cached PASS is honored here, and AGAIN on the destructive
    path in :func:`_reap_scratch_tree` immediately before removal."""
    detail: dict = {
        "reason": None,
        "first_unverified": None,
        "n_verified": 0,
        "n_tolerated": 0,
        "git_class": None,
    }

    def _keep(reason: str, first: str | None = None) -> tuple[None, dict]:
        detail["reason"] = reason
        detail["first_unverified"] = first
        if verdict_cache is not None:
            verdict_cache.store(cand, full_stats, None, detail)
        return None, detail

    if verdict_cache is not None:
        cached = verdict_cache.lookup(cand, full_stats)
        if cached is not None:
            ev, det = cached
            # #2147 review round 3 C2: a cached PASS embeds OVERLAY
            # conclusions that can flip with ZERO candidate-tree change —
            # the surviving anchor repo can lose refs or vanish entirely
            # without touching the cache key (which captures candidate tree
            # content only). Re-run the non-cacheable overlay presence/state
            # probe before honoring a cached PASS; a cached KEEP needs no
            # re-probe (it cannot license a deletion).
            if ev is not None and overlay_paths:
                slug, op = _overlay_keep_slug(
                    cand, main_repo=main_repo, overlay_paths=overlay_paths
                )
                if slug is not None:
                    return _keep(slug, op)
            det["cache_hit"] = True
            return ev, det

    fmt = _git(["rev-parse", "--show-object-format"], cwd=main_repo)
    if fmt is None or fmt.returncode != 0 or fmt.stdout.strip() != "sha1":
        return _keep("object-format")
    kind, admin = _git_dir_kind(cand)
    detail["git_class"] = kind
    probe = _scratch_git_class_probes(cand, kind, admin, main_repo=main_repo)
    if probe is not None:
        return _keep(probe)
    # #2147 review round 2 C3 (round 3: extracted to _overlay_keep_slug, which
    # also rejects a SYMLINKED declared-overlay path and re-runs on the
    # cached-PASS + destructive paths): every DECLARED overlay path PRESENT on
    # disk is validated as a nested repo BEFORE any other disposition —
    # including the empty-tree carve-out, the tolerance-only /
    # no-verifiable-content keeps, and the outer blob proof. Presence is keyed
    # on the directory entry itself (lstat + S_ISDIR), NOT on whether the walk
    # hashed any file under it, so a non-git overlay holding only tolerated
    # logs, exempt content, symlinks, or nothing at all still KEEPs the tree
    # (fail-toward-keep; a probe OSError keeps as a walk error).
    if overlay_paths:
        slug, op = _overlay_keep_slug(cand, main_repo=main_repo, overlay_paths=overlay_paths)
        if slug is not None:
            return _keep(slug, op)
    if (
        full_stats["n_regular"] == 0
        and full_stats["nonregular"] is None
        and full_stats["n_symlinks"] == 0
    ):
        evidence = "git-scratch-empty-tree: no files at all — nothing to lose"
        if verdict_cache is not None:
            detail["reason"] = "pass"
            verdict_cache.store(cand, full_stats, evidence, detail)
        return evidence, detail
    if full_stats["nonexempt_bytes"] > _scratch_verify_cap_bytes():
        return _keep("over-verify-cap")
    walk_errors: list[OSError] = []
    entries: list[tuple[str, str]] = []  # (sha, cand-relative path) — outer-odb proofs
    overlay_entries: dict[str, list[tuple[str, str]]] = {}  # #2147 D9 nested-odb proofs
    tolerated_bytes = 0
    try:
        for dirpath, dirnames, filenames in os.walk(
            cand, topdown=True, onerror=walk_errors.append, followlinks=False
        ):
            if walk_errors:
                return _keep("walk-error")
            dp = Path(dirpath)
            rel_dir_parts = dp.relative_to(cand).parts
            dirnames[:] = [d for d in dirnames if d not in _SCRATCH_EXEMPT_DIR_NAMES]
            under_durable = any(p in ("store", "eval_results") for p in rel_dir_parts)
            for name in sorted(filenames):
                p = dp / name
                rel = "/".join((*rel_dir_parts, name))
                try:
                    lst = p.lstat()
                except OSError:
                    return _keep("walk-error", rel)
                mode = lst.st_mode
                if stat.S_ISLNK(mode):
                    continue  # skipped: rmtree removes the link, not the target
                if not stat.S_ISREG(mode):
                    return _keep("nonregular", rel)
                if _scratch_is_exempt_rel((*rel_dir_parts, name)):
                    continue  # rebuildable tool state (root ``.git`` pointer file)
                # #2147 review round 2 C3: overlay membership is classified
                # BEFORE the small-text tolerance — a file under a claimed
                # overlay is NEVER tolerated into the outer proof; it is
                # hashed and proven against the surviving overlay repo.
                op = _overlay_member(rel, overlay_paths)
                ext = os.path.splitext(name)[1].lower()
                if (
                    op is None
                    and ext in _SCRATCH_TOLERATED_EXTS
                    and not under_durable
                    and lst.st_size <= SCRATCH_TOLERATED_FILE_MAX_BYTES
                    and tolerated_bytes + lst.st_size <= SCRATCH_TOLERATED_TOTAL_MAX_BYTES
                ):
                    tolerated_bytes += lst.st_size
                    detail["n_tolerated"] += 1
                    continue
                opened = _open_scratch_regular(p)
                if opened is None:
                    return _keep("walk-error", rel)
                fd, fst = opened
                try:
                    sha = _blob_sha1_from_fd(fd, fst.st_size)
                finally:
                    with contextlib.suppress(OSError):
                        os.close(fd)
                if sha is None:
                    return _keep("concurrent-write", rel)
                if op is not None:
                    overlay_entries.setdefault(op, []).append((sha, rel))
                else:
                    entries.append((sha, rel))
        if walk_errors:
            return _keep("walk-error")
    except Exception:
        return _keep("walk-error")
    n_overlay = sum(len(v) for v in overlay_entries.values())
    if not entries and not n_overlay:
        if detail["n_tolerated"] > 0:
            return _keep("tolerance-only")
        return _keep("no-verifiable-content")
    if entries:
        missing = _git_first_missing_blob(main_repo, [sha for sha, _ in entries])
        if missing is None:
            return _keep("git-probe-failed")
        if missing >= 0:
            return _keep("unverified-file", entries[missing][1])
    # #2147 D9 (round 2 C2): overlay files are proven against the SURVIVING
    # overlay repo's odb (``main_repo / <overlay path>`` — the copy that
    # outlives the reap; the nested copy's own odb dies with the tree). The
    # nested repos themselves were already validated by the presence loop
    # above (clone class + clean + stash-empty + all tips/HEAD reachable in
    # the surviving repo); deliberately NO fallback to the outer-odb proof.
    for op in sorted(overlay_entries):
        surviving = main_repo / Path(op)
        op_entries = overlay_entries[op]
        missing = _git_first_missing_blob(surviving, [sha for sha, _ in op_entries])
        if missing is None:
            return _keep("git-probe-failed", op)
        if missing >= 0:
            return _keep("unverified-file", op_entries[missing][1])
    detail["n_verified"] = len(entries) + n_overlay
    evidence = (
        f"git-blob-reproducible: {len(entries)} files verified in the main odb, "
        f"{detail['n_tolerated']} small-text tolerated"
    )
    if n_overlay:
        evidence += f", {n_overlay} overlay files verified in the surviving overlay repo(s)"
    if verdict_cache is not None:
        det = dict(detail)
        det["reason"] = "pass"
        verdict_cache.store(cand, full_stats, evidence, det)
    return evidence, detail


def _tmp_git_evidence_branch_c(cache_dir: Path, *, main_repo: Path) -> tuple[str | None, dict]:
    """Gate-1.7 evidence branch (c) (#2127) for the ISSUE-KEYED /tmp legs
    (P1/P2): per-file git-blob reproducibility, reusing the scratch-leg
    primitive. REFUSES a candidate that is a REGISTERED main-repo worktree
    (reason ``registered-worktree``): gate 1.7 licenses a plain ``rmtree``,
    which would strand the registration — worktree-aware removal belongs to
    :func:`sweep_tmp_scratch`'s reap step, never here.

    Round 3 C1 sibling fix, restructured in round 6: registration is proven
    per-candidate and parse-free. Layer 1 — the AUTHORITATIVE gitfile probe
    (:func:`_candidate_worktree_registration`): ``ours`` and ``foreign``
    worktrees refuse outright, ``unreadable`` refuses fail-closed. Layer
    2 — for candidates with no usable pointer (``none``/``clone``/
    ``submodule``: the pointer-deleted/replaced downgrade class), the
    ADMIN-side per-record ``gitdir``-file enumeration
    (:func:`_admin_registered_worktree_paths`) proves non-registration
    byte-exactly (newline-bearing paths included). Layer 3 — the hardened
    porcelain listing (:func:`_registered_worktree_paths`, rounds 4/5) is
    KEPT as defence-in-depth: an independent second implementation whose
    failure or membership can only add KEEPs, never license. Probe failure
    or ambiguity anywhere refuses (fail-toward-keep)."""
    stats = _scratch_walk_stats(cache_dir)
    if stats is None:
        return None, {"reason": "walk-error", "first_unverified": None}
    if stats["nonregular"] is not None:
        return None, {"reason": "nonregular", "first_unverified": stats["nonregular"]}
    reg, _admin = _candidate_worktree_registration(cache_dir, main_repo)
    if reg == "ours":
        return None, {"reason": "registered-worktree", "first_unverified": None}
    if reg == "foreign":
        return None, {"reason": "foreign-worktree", "first_unverified": None}
    if reg == "unreadable":
        return None, {"reason": "registration-probe-failed", "first_unverified": None}
    admin_set = _admin_registered_worktree_paths(main_repo)
    if admin_set is None:
        return None, {"reason": "registration-probe-failed", "first_unverified": None}
    if os.path.realpath(cache_dir) in admin_set:
        return None, {"reason": "registered-worktree", "first_unverified": None}
    registered = _registered_worktree_paths(main_repo)
    if registered is None:
        return None, {"reason": "registration-probe-failed", "first_unverified": None}
    if os.path.realpath(cache_dir) in registered:
        return None, {"reason": "registered-worktree", "first_unverified": None}
    return _git_blob_reproducibility_evidence(cache_dir, main_repo=main_repo, full_stats=stats)


def _scratch_live_process_hit(cand: Path, *, exact: bool = False) -> str | None:
    """Live-process probe (#2127 amendment 1): scan every ``/proc/<pid>``'s
    ``cwd``/``exe``/``fd/*`` readlink for a realpath-prefix hit on the
    candidate. Returns a short description of the first hit, the sentinel
    ``"probe-unavailable"`` when ``/proc`` cannot be listed
    (fail-toward-keep), or ``None`` when no visible process holds the tree.
    Unreadable per-pid entries are SKIPPED — other-uid/root processes are
    invisible to this probe, a NAMED residual (plan v2 §12 item 6).

    ``exact=True`` is the file-granular mode (#2377): match on realpath
    EQUALITY only — no prefix matching for a regular file (a file has no
    children, so a prefix hit could only be a different path)."""
    try:
        real = os.path.realpath(cand)
        prefix = real.rstrip("/") + "/"
        try:
            pids = [n for n in os.listdir("/proc") if n.isdigit()]
        except OSError:
            return "probe-unavailable"
        self_pid = str(os.getpid())
        for pid in pids:
            if pid == self_pid:
                continue
            base = f"/proc/{pid}"
            links = [f"{base}/cwd", f"{base}/exe"]
            try:
                links.extend(f"{base}/fd/{fd}" for fd in os.listdir(f"{base}/fd"))
            except OSError:
                pass  # exited race / other-uid fds — named residual above
            for link in links:
                try:
                    target = os.readlink(link)
                except OSError:
                    continue
                if target == real or (not exact and target.startswith(prefix)):
                    return f"pid={pid} via {link[len(base) + 1 :]} -> {target}"
        return None
    except Exception:
        return "probe-unavailable"


def _reap_scratch_tree(
    cand: Path,
    *,
    main_repo: Path,
    verify_started: float,
    overlay_paths: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """Reap step for a fully-licensed scratch candidate (#2127): one FRESH
    re-walk first (any NON-EXEMPT mtime >= ``verify_started``, or any walk
    error, aborts the reap — the tree changed under verification; exempt-dir
    mtimes are excluded from THIS check only, because the sweep's own git
    probes rewrite a clone's in-tree ``.git/index`` and must not self-abort
    every clean-clone reap — a real mid-verify writer bumps non-exempt
    file/dir mtimes, and a still-live exempt-dir writer is the live-process
    probe's catch), then a fresh git CLASS re-probe, then a fresh OVERLAY
    re-probe (when the leg declares overlays), then worktree-aware removal
    behind a POSITIVE non-registration proof.

    The class re-probe (:func:`_scratch_git_class_probes`, review round 2)
    is what makes a CACHED PASS safe: the verdict-cache key captures tree
    content only, but a PASS embeds conclusions about EXTERNAL git state
    (HEAD/ref reachability, lock, status) that can flip with ZERO tree
    change — ``git branch -D``, a pruning fetch, an upstream rewrite — so
    a cache-hit PASS would otherwise skip those probes forever. Re-probing
    HERE (the only destructive path) keeps the cache's re-hash skip intact
    while guaranteeing no deletion ever acts on stale external-state
    conclusions; a hit KEEPS with ``reap-reprobe-<slug>``. Round 3 C2
    extends the same law to nested-overlay state: stash/ref mutations in a
    nested overlay live under the exempt ``.git`` dir (invisible to the
    non-exempt recency re-walk) and the surviving anchor can change with
    zero candidate-tree change, so :func:`_overlay_keep_slug` re-runs here
    immediately before removal — any slug KEEPS as ``reap-reprobe-<slug>``.
    The ``git status`` atime side effect is mooted by deletion on success
    and harmless on abort; blobs are NOT re-hashed, so the documented gc
    residual stands unchanged.

    A REGISTERED main-repo worktree goes through
    ``git worktree remove --force`` with ONE ``--force``: a lock acquired
    between gate check and reap makes the remove FAIL (git needs ``--force``
    twice for a locked removal), and that failure KEEPS the tree
    (amendment 2 — no ``shutil.rmtree`` fallback). There is deliberately NO
    ``git worktree prune`` fallback either: a GLOBAL prune would unregister
    PEER worktrees whose trees are momentarily missing (e.g. a registered
    ``/mnt/eps-data``-rooted scratch during a mount hiccup) — repairable
    via ``git worktree repair`` and data-preserving, but not this
    janitor's to inflict (amendment 3, the named global-prune residual).

    Everything else reaches ``shutil.rmtree`` ONLY behind the round 3 C1
    POSITIVE non-registration proof (:func:`_registered_worktree_paths`):
    the candidate-side class probes read the tree's OWN ``.git`` entry, so
    a REGISTERED worktree whose pointer file was deleted (class ``none``)
    or replaced by a real clone (class ``clone``) would otherwise classify
    into the rmtree branch and be deleted WITHOUT unregistering — defeating
    the round 2 C1 structural-unreachability contract by a different route.
    Probe failure or parse ambiguity KEEPS; a registered candidate KEEPS as
    ``reap-reprobe-registered-path``. Returns ``(reaped, reason)``."""
    fresh = _scratch_walk_stats(cand)
    if fresh is None or fresh["newest_nonexempt_mtime"] >= verify_started:
        return False, "reap-recheck-recency"
    kind, admin = _git_dir_kind(cand)
    probe = _scratch_git_class_probes(cand, kind, admin, main_repo=main_repo)
    if probe is not None:
        return False, f"reap-reprobe-{probe}"
    if overlay_paths:
        slug, _op = _overlay_keep_slug(cand, main_repo=main_repo, overlay_paths=overlay_paths)
        if slug is not None:
            return False, f"reap-reprobe-{slug}"
    if kind == "worktree":
        # #2147 review round 2 C1: dispatch SOLELY on the freshly proven
        # ``kind``. The class re-probe above already required a non-None
        # admin dir REGISTERED to ``main_repo`` (else it kept with
        # ``foreign-worktree``), so re-running the registration lookup here
        # would add nothing — except a transient failure mode that would
        # fall through to ``shutil.rmtree`` on a REGISTERED worktree. With
        # this dispatch the only reachable outcomes for the worktree class
        # are a checked ``git worktree remove --force`` or a KEEP;
        # ``rmtree`` is structurally unreachable.
        proc = _git(["worktree", "remove", "--force", str(cand)], cwd=main_repo, timeout=600.0)
        if proc is None or proc.returncode != 0:
            locked = False
            if admin is not None:
                try:
                    locked = (admin / "locked").exists()
                except OSError:
                    locked = False  # message-detail only; the KEEP stands regardless
            err = "timeout" if proc is None else (proc.stderr.strip() or "unknown error")
            tag = "locked" if locked else "remove-failed"
            return False, f"worktree-remove-failed ({tag}: {err.splitlines()[-1] if err else ''})"
        return True, "worktree-removed"
    # #2147 review round 3 C1, restructured round 6: POSITIVE
    # non-registration proof before ANY rmtree. The class probes above read
    # only the candidate's own ``.git`` entry (the per-candidate gitfile
    # authority — an intact-pointer worktree never reaches this branch), so
    # a REGISTERED worktree with a deleted/replaced pointer classifies as
    # ``none``/``clone`` and would otherwise be rmtree'd without
    # unregistering. Non-registration is proven ADMIN-side from the
    # per-record ``gitdir`` files (byte-exact, newline-safe —
    # :func:`_admin_registered_worktree_paths`), with the hardened porcelain
    # listing (rounds 4/5) KEPT as defence-in-depth; failure, ambiguity, or
    # membership in EITHER source KEEPS (fail-toward-keep).
    admin_set = _admin_registered_worktree_paths(main_repo)
    if admin_set is None:
        return False, "reap-reprobe-registration-probe-failed"
    if os.path.realpath(cand) in admin_set:
        return False, "reap-reprobe-registered-path"
    registered = _registered_worktree_paths(main_repo)
    if registered is None:
        return False, "reap-reprobe-registration-probe-failed"
    if os.path.realpath(cand) in registered:
        return False, "reap-reprobe-registered-path"
    try:
        shutil.rmtree(cand)
    except OSError as exc:
        return False, f"rmtree-failed ({exc})"
    return True, "rmtree"


def _tmp_scratch_candidates(tmp_root: Path) -> list[Path]:
    """Top-level scratch-shaped candidates under ``tmp_root`` (#2127): the
    entry must be a REAL directory (lstat ``S_ISDIR`` — a scratch-named
    symlink is never followed), uid-owned, shape-matched and not denylisted
    (:func:`is_tmp_scratch_name`). Kill switches are the caller's
    (:func:`sweep_tmp_scratch`) concern."""
    try:
        names = sorted(os.listdir(tmp_root))
    except OSError:
        return []
    out: list[Path] = []
    for name in names:
        if not is_tmp_scratch_name(name):
            continue
        p = tmp_root / name
        try:
            st = p.lstat()
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode) or not _tmp_entry_owned(p):
            continue
        out.append(p)
    return out


@dataclass
class ScratchSweepResult:
    """Outcome of one :func:`sweep_tmp_scratch` run (#2127).

    ``skip_reason`` (#2147 review round 2 M1) is set when the sweep did NOT
    enumerate its root (e.g. the slurm-src staging root does not exist) —
    an explicit signal distinct from "enumerated and found zero
    candidates", so the tier adapter can surface a skipped tier instead of
    silently reporting an empty sweep."""

    rows: list[dict] = field(default_factory=list)
    bytes_freed: int = 0
    total_discovered_bytes: int = 0
    skip_reason: str | None = None


def _scratch_row_finish(
    row: dict,
    disposition: str,
    reason: str,
    *,
    leg: str,
    apply: bool,
    floor: int,
    escalate: bool | None = None,
    escalation_gate: Callable[[dict, str, str], Callable[[], None] | None] | None = None,
) -> None:
    """Finish one scratch-sweep row (#2127, parameterized for #2147): record
    disposition + reason on ``row``, print the per-candidate report line, and
    append the floor-gated sidecar escalation row. ``leg`` prefixes the
    printed tag and the sidecar ``kind`` — byte-identical to the pre-#2147
    inline ``_finish`` closure for ``leg="tmp-scratch"``. ``escalation_gate``
    (#2147 D6) is consulted ONLY when an escalation would fire; it returns
    ``None`` to SUPPRESS the sidecar append (row + print are unaffected) or
    a COMMIT thunk invoked ONLY AFTER :func:`append_disk_guard_event`
    reports the durable emission landed (review round 2 M2 — recording the
    dedup timestamp before the append would let a FAILED append suppress
    the alert for the whole re-alert window). ``None`` for the gate keeps
    the tmp-scratch behavior of appending every escalation."""
    row["disposition"] = disposition
    row["reason"] = reason
    size = row.get("bytes", 0)
    if escalate is None:
        escalate = size >= floor
    print(
        f"  [{leg}] {disposition}: {row['path']} ({size / 1e9:.2f} GB) — {reason}",
        file=sys.stderr,
    )
    if not escalate:
        return

    def _no_commit() -> None:
        return None

    commit: Callable[[], None] | None = _no_commit
    if escalation_gate is not None:
        commit = escalation_gate(row, disposition, reason)
    if commit is None:
        return  # deduped within the re-alert window — nothing appended
    appended = append_disk_guard_event(
        {
            "kind": (disposition if disposition.startswith(f"{leg}-") else f"{leg}-{disposition}"),
            "path": row["path"],
            "bytes": size,
            "reason": reason,
            "evidence": row.get("evidence"),
        },
        apply=apply,
    )
    if appended:
        commit()


def _sweep_scratch_candidate(
    cand: Path,
    row: dict,
    *,
    leg: str,
    apply: bool,
    main_repo: Path,
    window_start: float,
    min_age_hours: float,
    now: float,
    cache: _ScratchVerdictCache,
    result: ScratchSweepResult,
    floor: int,
    overlay_paths: tuple[str, ...] = (),
    escalation_gate: Callable[[dict, str, str], Callable[[], None] | None] | None = None,
) -> None:
    """The SHARED per-candidate #2127 verified-scratch pipeline, extracted for
    #2147 so the slurm-src leg reuses it UNCHANGED: defensive walk -> write
    recency -> git-blob evidence (verdict-cached; ``overlay_paths`` adds the
    #2147 D9 nested-repo evidence class) -> reader-atime pin -> live-process
    probe -> (report) would-reap / (apply) worktree-aware reap.

    The caller creates ``row`` (``path``/``name``/``leg`` keys) and appends
    it to ``result.rows`` BEFORE calling; keep dispositions are tagged
    ``{leg}-...`` (``would-reap`` stays unprefixed). Behavior for
    ``leg="tmp-scratch"``, ``overlay_paths=()``, ``escalation_gate=None`` is
    byte-identical to the pre-extraction ``sweep_tmp_scratch`` loop body
    (pinned by ``tests/test_vm_disk_guard_slurm_src.py`` T15)."""

    def _finish(disposition: str, reason: str, *, escalate: bool | None = None) -> None:
        _scratch_row_finish(
            row,
            disposition,
            reason,
            leg=leg,
            apply=apply,
            floor=floor,
            escalate=escalate,
            escalation_gate=escalation_gate,
        )

    stats = _scratch_walk_stats(cand)
    if stats is None:
        _finish(f"{leg}-unverified-kept", "walk error — unreadable tree; KEPT")
        return
    row["bytes"] = stats["total_bytes"]
    row["age_hours"] = round((now - stats["newest_mtime"]) / 3600.0, 2)
    result.total_discovered_bytes += stats["total_bytes"]
    if stats["nonregular"] is not None:
        _finish(
            f"{leg}-nonregular-kept",
            f"non-regular file in tree ({stats['nonregular']}); KEPT",
        )
        return
    if stats["newest_mtime"] > window_start:
        _finish(
            f"{leg}-recent-kept",
            f"written within the last {min_age_hours:.0f}h; KEPT (age is only a keep signal)",
            escalate=False,
        )
        return
    # The overlay kwarg is threaded ONLY when the leg declares overlays
    # (#2147 D9): the tmp-scratch leg's call shape stays byte-identical to
    # the pre-extraction loop (T15), incl. for test stubs of the evidence fn.
    overlay_kwargs = {"overlay_paths": overlay_paths} if overlay_paths else {}
    evidence, detail = _git_blob_reproducibility_evidence(
        cand,
        main_repo=main_repo,
        full_stats=stats,
        verdict_cache=cache,
        **overlay_kwargs,
    )
    row["git_class"] = detail.get("git_class")
    row["n_verified"] = detail.get("n_verified")
    row["n_tolerated"] = detail.get("n_tolerated")
    if evidence is None:
        reason_slug = str(detail.get("reason"))
        row["reason_slug"] = reason_slug  # #2147 M2: stable slug for the D6 dedup key
        row["first_unverified"] = detail.get("first_unverified")
        disposition = {
            "worktree-locked": f"{leg}-worktree-locked-kept",
            "tolerance-only": f"{leg}-tolerance-only-kept",
            "nonregular": f"{leg}-nonregular-kept",
        }.get(reason_slug, f"{leg}-unverified-kept")
        first = detail.get("first_unverified")
        _finish(
            disposition,
            f"no git-reproducibility proof ({reason_slug}"
            + (f"; first unverified: {first}" if first else "")
            + "); KEPT",
        )
        return
    row["evidence"] = evidence
    atime = stats["newest_reader_atime"]
    if atime is not None and atime > window_start:
        row["reader_atime_age_hours"] = round((now - atime) / 3600.0, 2)
        _finish(
            f"{leg}-verified-atime-pinned",
            "verified reproducible, but a non-hardlinked file was READ within the "
            f"window (atime {row['reader_atime_age_hours']}h ago); KEPT",
            escalate=True,
        )
        return
    hit = _scratch_live_process_hit(cand)
    if hit is not None:
        _finish(f"{leg}-live-process-kept", f"live process holds the tree ({hit}); KEPT")
        return
    if not apply:
        _finish("would-reap", f"evidence: {evidence}", escalate=True)
        return
    # Same thread-only-when-declared rule as the evidence call above: the
    # tmp-scratch leg's reap call shape stays byte-identical (T15), and the
    # slurm-src leg's destructive path re-probes overlay state (round 3 C2).
    reaped, reap_reason = _reap_scratch_tree(
        cand, main_repo=main_repo, verify_started=now, **overlay_kwargs
    )
    if reaped:
        result.bytes_freed += stats["total_bytes"]
        cache.prune(cand)
        _finish(f"{leg}-reaped", f"{reap_reason}; evidence: {evidence}", escalate=True)
    elif reap_reason == "reap-recheck-recency":
        _finish(
            f"{leg}-reap-aborted-recency",
            "tree changed between verification and reap; KEPT",
            escalate=True,
        )
    elif reap_reason.startswith("reap-reprobe-"):
        row["reason_slug"] = reap_reason  # #2147 M2: stable slug for the D6 dedup key
        _finish(
            f"{leg}-reap-reprobe-kept",
            "git state flipped since (possibly cached) verification "
            f"({reap_reason.removeprefix('reap-reprobe-')}); KEPT",
            escalate=True,
        )
    elif reap_reason.startswith("worktree-remove-failed"):
        _finish(f"{leg}-worktree-remove-failed", f"{reap_reason}; KEPT", escalate=True)
    else:
        _finish(f"{leg}-reap-failed", f"{reap_reason}; KEPT", escalate=True)


def sweep_tmp_scratch(
    tmp_root: Path | None,
    *,
    apply: bool,
    main_repo: Path | None,
    min_age_hours: float | None = None,
    now: float | None = None,
    verdict_cache_path: Path | None = None,
) -> ScratchSweepResult:
    """Owner-status-INDEPENDENT sweep of top-level ``/tmp`` gate/smoke
    scratch trees (#2127), gated on VERIFIED git-reproducibility — never on
    age (age is only ever a KEEP signal). Report-only unless ``apply``.

    STRICT opt-in, same hermeticity contract as the #911 leg: runs only
    when ``tmp_root`` AND ``main_repo`` are explicitly non-None (production
    ``main()`` bodies pass :func:`production_tmp_root` /
    :func:`_resolution_root`; library/test callers default to no-/tmp) and
    both kill-switch layers are unset (:func:`tmp_scratch_sweep_enabled`).

    Per-candidate gate order (every early exit is a KEEP):

    1. shape + denylist + uid + real-dir (:func:`_tmp_scratch_candidates`);
    2. one defensive walk (:func:`_scratch_walk_stats`) — walk error =>
       ``tmp-scratch-unverified-kept``; any FIFO/socket/device =>
       ``tmp-scratch-nonregular-kept``;
    3. WRITE recency (newest mtime over everything, exempt dirs included)
       younger than ``min_age_hours`` => ``tmp-scratch-recent-kept``;
    4. evidence (:func:`_git_blob_reproducibility_evidence`, verdict-cached)
       — no proof => ``tmp-scratch-{unverified,tolerance-only,
       worktree-locked,nonregular}-kept`` by reason;
    5. READER recency (nlink==1 non-exempt atimes) younger than the window
       on a VERIFIED tree => ``tmp-scratch-verified-atime-pinned`` (kept +
       escalated: reproducible, but someone read it recently);
    6. live-process probe (:func:`_scratch_live_process_hit`, amendment 1)
       => ``tmp-scratch-live-process-kept``;
    7. reap (:func:`_reap_scratch_tree`) — fresh re-walk abort =>
       ``tmp-scratch-reap-aborted-recency``; reap-time git class RE-probe
       hit (external git state flipped since the — possibly cached —
       verification: ref deleted, lock taken, tree dirtied) =>
       ``tmp-scratch-reap-reprobe-kept``; worktree-remove failure =>
       ``tmp-scratch-worktree-remove-failed`` (kept); else
       ``tmp-scratch-reaped`` (or ``would-reap`` in report mode).

    Sidecar escalation rows are floor-gated (``EPS_SCRATCH_ESCALATE_FLOOR_GB``)
    for KEEP dispositions; reap/would-reap/atime-pinned rows always land.
    Every candidate appears in ``rows`` regardless."""
    result = ScratchSweepResult()
    if tmp_root is None or main_repo is None or not tmp_scratch_sweep_enabled():
        return result
    now = time.time() if now is None else now
    if min_age_hours is None:
        min_age_hours = _noncanonical_min_age_hours()
    window_start = now - min_age_hours * 3600.0
    cache = _ScratchVerdictCache(verdict_cache_path)
    floor = _scratch_escalate_floor_bytes()
    for cand in _tmp_scratch_candidates(tmp_root):
        row: dict = {"path": str(cand), "name": cand.name, "leg": "tmp-scratch"}
        result.rows.append(row)
        _sweep_scratch_candidate(
            cand,
            row,
            leg="tmp-scratch",
            apply=apply,
            main_repo=main_repo,
            window_start=window_start,
            min_age_hours=min_age_hours,
            now=now,
            cache=cache,
            result=result,
            floor=floor,
        )
    cache.save()
    return result


# ─── slurm-src staging-tree sweep (#2147, vm_disk_guard tier (g)) ─────────────

# ~/.eps-slurm-src/issue-<N> trees are full repo checkouts materialized by
# backends/slurm.py::materialize_branch_src for the SLURM lanes; nothing
# reaped TERMINAL issues' copies before tier (g) (13 dirs / 112 GB measured
# 2026-08-16). The sweep below runs the D4 pre-gates, then the SHARED #2127
# verified-scratch per-candidate core unchanged.
_SLURM_SRC_NAME_RE = re.compile(r"^issue-(\d+)$")
SLURM_SRC_SWEEP_KILL_ENV = "EPM_SKIP_SLURM_SRC_SWEEP"
SLURM_SRC_ESCALATION_STATE_REL = Path(".claude") / "cache" / "slurm-src-escalation-state.json"
# D6: leg-scoped escalation dedup — re-alert cadence (days) for STANDING
# slurm-src keeps (an active issue's tree, a head-unreachable worktree),
# which would otherwise append a sidecar row on every guard pass. Weekly.
SLURM_SRC_ESCALATION_REALERT_DAYS_DEFAULT = 7.0
_SLURM_SRC_ESCALATION_BANDS_GB = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0)


def slurm_src_sweep_enabled() -> bool:
    """Two-layer kill switch for the slurm-src leg (#2147, mirrors the #2127
    scratch pattern): runs only when BOTH the family switch
    (:data:`NONCANONICAL_SWEEP_KILL_ENV`) and the leg's own switch
    (:data:`SLURM_SRC_SWEEP_KILL_ENV`) are unset."""
    if os.environ.get(NONCANONICAL_SWEEP_KILL_ENV, "").strip():
        return False
    return not os.environ.get(SLURM_SRC_SWEEP_KILL_ENV, "").strip()


def slurm_src_escalation_state_path() -> Path:
    """Production location of the tier-(g) leg-scoped escalation-dedup state
    (#2147 D6): ``<main checkout>/.claude/cache/slurm-src-escalation-state.json``.

    main()-ONLY opt-in, the same hermeticity contract as
    :func:`scratch_verdict_cache_path`: library callers default to ``None``
    (no dedup-state reads/writes). AST-pinned by
    ``tests/test_janitor_noncanonical_caches.py::test_production_tmp_root_only_in_mains``."""
    return _resolution_root() / SLURM_SRC_ESCALATION_STATE_REL


def _slurm_src_escalation_band_gb(bytes_: int) -> float:
    """Coarse size band (GB) for the D6 dedup key — the largest crossed
    boundary, integer-GB above the top band (mirrors the tier-(b) band shape
    so a growing standing keep re-alerts when it crosses into a new band)."""
    gb = bytes_ / 1e9
    band = 0.0
    for boundary in _SLURM_SRC_ESCALATION_BANDS_GB:
        if gb >= boundary:
            band = boundary
    if gb >= _SLURM_SRC_ESCALATION_BANDS_GB[-1]:
        band = max(band, float(int(gb)))
    return band


def _slurm_src_escalation_realert_secs() -> float:
    """D6 re-alert window in seconds (env
    ``EPS_SLURM_SRC_ESCALATION_REALERT_DAYS``; invalid -> the weekly default)."""
    raw = os.environ.get("EPS_SLURM_SRC_ESCALATION_REALERT_DAYS", "").strip()
    try:
        days = float(raw) if raw else SLURM_SRC_ESCALATION_REALERT_DAYS_DEFAULT
    except ValueError:
        days = SLURM_SRC_ESCALATION_REALERT_DAYS_DEFAULT
    return days * 86400.0


def _load_slurm_src_escalation_state(path: Path) -> dict:
    """Fail-soft read of the D6 dedup state (missing/corrupt -> ``{}`` —
    dedup-state loss re-alerts, it never blocks the sweep)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_slurm_src_escalation_state(path: Path, state: dict) -> None:
    """Atomic tmp+rename write of the D6 dedup state; fail-soft (a write
    failure re-alerts next pass — it never corrupts or blocks the sweep) but
    NEVER silent (review round 2 M1): the failure is logged with the path +
    error so a persistently unwritable state file is diagnosable."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True))
        os.replace(tmp, path)
    except OSError as exc:
        print(
            f"  WARNING: writing slurm-src escalation dedup state failed "
            f"({path}: {exc.__class__.__name__}: {exc}) — the alert will re-fire next pass",
            file=sys.stderr,
        )


def _slurm_src_escalation_gate(
    state_path: Path | None, *, apply: bool, now: float
) -> Callable[[dict, str, str], Callable[[], None] | None]:
    """Leg-scoped escalation dedup gate for tier (g) (#2147 D6).

    The #2127 scratch core appends a sidecar escalation row on EVERY pass —
    correct for the rare tmp-scratch keeps, but slurm-src STANDING keeps (an
    ACTIVE issue's tree, a head-unreachable worktree) would re-alert every
    guard run. Dedup key: (path, disposition, stable reason slug, size
    band) — the reason slug (``row["reason_slug"]``, falling back to the
    disposition) is IN the key per plan D6 and review round 2 M2: several
    materially different keep reasons share one disposition
    (``slurm-src-unverified-kept`` covers head-unreachable / dirty-stash /
    unverified-file / git-probe-failed), so a disposition-only key would
    hide a changed reason for the whole re-alert window. Re-alert after
    ``EPS_SLURM_SRC_ESCALATION_REALERT_DAYS`` (default 7 d).

    Returns a DECIDE function: ``decide(row, disposition, reason)`` yields
    ``None`` to suppress (deduped within the window) or a COMMIT thunk the
    caller invokes ONLY AFTER the sidecar append durably landed (round 2
    M2 — committing the dedup timestamp before the append would let a
    failed append suppress the alert for 7 days). Report-only runs and a
    ``None`` state path never dedup and never write (the printed report
    stays complete); state IO is fail-soft-but-logged."""

    def _noop() -> None:
        return None

    def decide(row: dict, disposition: str, reason: str) -> Callable[[], None] | None:
        if not apply or state_path is None:
            return _noop
        band = _slurm_src_escalation_band_gb(int(row.get("bytes", 0) or 0))
        slug = str(row.get("reason_slug") or disposition)
        key = f"{row.get('path')}|{disposition}|{slug}|{band:g}"
        state = _load_slurm_src_escalation_state(state_path)
        prev = state.get(key)
        prev_ts = prev.get("ts") if isinstance(prev, dict) else None
        if isinstance(prev_ts, int | float):
            if now - prev_ts < _slurm_src_escalation_realert_secs():
                return None

        def commit() -> None:
            fresh = _load_slurm_src_escalation_state(state_path)
            fresh[key] = {"ts": now, "bytes": int(row.get("bytes", 0) or 0)}
            _save_slurm_src_escalation_state(state_path, fresh)

        return commit

    return decide


def _assert_safe_slurm_src_root(staging_root: Path) -> Path:
    """#2147 review round 2 C4: fail LOUD (ValueError) unless the
    canonicalized staging root satisfies the strict staging-root contract —
    BEFORE any enumeration, status probe, or evidence probe runs — and
    RETURN that canonical root (round 3 C4): the caller MUST enumerate,
    contain, and construct candidates from the RETURNED path, never the raw
    argument. Validating one path and enumerating another leaves a
    symlink-target-swap window in which an armed sweep is redirected into a
    root that was never checked for broadness / mount-point / ``.git``.
    (Named residual: a component of the CANONICAL path re-swapped after
    validation is a directory-rename race outside a path-based API's reach;
    the returned-path discipline closes the demonstrated symlink-root class.)

    ``EPS_SLURM_SRC_ROOT`` is operator input, and tier (g)'s g3 containment
    is defined RELATIVE to the supplied root: a broad root would turn every
    top-level ``issue-<N>`` dir under it into a deletion candidate. Rejected
    unconditionally:

    - a non-absolute root (would resolve against an accidental cwd);
    - ``/`` or any DIRECT child of ``/`` (``/home``, ``/tmp``, ``/mnt``,
      ``/root``, ...) — comparably broad anchors;
    - the current user's home directory, or any ancestor of it;
    - a filesystem mount point (a whole-disk anchor like ``/mnt/eps-data``);
    - any git repository / worktree root (a ``.git`` entry exists — a repo
      checkout is never a staging root).

    The default ``~/.eps-slurm-src`` passes every check. Validation runs on
    the REALPATH (symlinked misconfigurations cannot dodge it); an OSError
    while validating is itself a ValueError (fail-closed, never enumerate an
    unvalidatable root)."""
    if not staging_root.is_absolute():
        raise ValueError(
            f"sweep_slurm_src: staging root {staging_root} is not absolute — refusing to "
            "sweep a cwd-relative root (check EPS_SLURM_SRC_ROOT)"
        )
    real = Path(os.path.realpath(staging_root))
    if len(real.parts) <= 2:
        raise ValueError(
            f"sweep_slurm_src: staging root {staging_root} resolves to {real} — '/' and "
            "direct children of '/' are never staging roots (check EPS_SLURM_SRC_ROOT)"
        )
    home_real = Path(os.path.realpath(Path.home()))
    if real == home_real or real in home_real.parents:
        raise ValueError(
            f"sweep_slurm_src: staging root {staging_root} resolves to {real}, the current "
            "user's home directory (or an ancestor of it) — refusing (check EPS_SLURM_SRC_ROOT)"
        )
    try:
        if os.path.ismount(real):
            raise ValueError(
                f"sweep_slurm_src: staging root {staging_root} resolves to the mount point "
                f"{real} — a whole-filesystem anchor is never a staging root "
                "(check EPS_SLURM_SRC_ROOT)"
            )
        git_entry = real / ".git"
        if git_entry.is_symlink() or git_entry.exists():
            raise ValueError(
                f"sweep_slurm_src: staging root {staging_root} resolves to {real}, which "
                "carries a .git entry — a repository root is never a staging root "
                "(check EPS_SLURM_SRC_ROOT)"
            )
    except OSError as exc:
        raise ValueError(
            f"sweep_slurm_src: cannot validate staging root {staging_root} "
            f"({exc.__class__.__name__}: {exc}) — refusing to sweep an unvalidatable root"
        ) from exc
    return real


def sweep_slurm_src(
    staging_root: Path | None,
    *,
    apply: bool,
    main_repo: Path | None,
    status_resolver: Callable[[int], str | None] | None = None,
    terminal_statuses: frozenset[str] | set[str] | None = None,
    min_age_hours: float | None = None,
    now: float | None = None,
    verdict_cache_path: Path | None = None,
    escalation_state_path: Path | None = None,
    overlay_paths: tuple[str, ...] | None = None,
) -> ScratchSweepResult:
    """Evidence-gated sweep of ``~/.eps-slurm-src/issue-<N>`` SLURM staging
    trees (#2147, ``vm_disk_guard`` tier (g)): the D4 pre-gates below, then
    the SHARED #2127 verified-scratch per-candidate core
    (:func:`_sweep_scratch_candidate`) UNCHANGED, keep reasons re-tagged
    ``slurm-src-*``. Report-only unless ``apply``.

    Round 2 hardening: the ARMED sweep first validates the staging root
    against the strict staging-root contract
    (:func:`_assert_safe_slurm_src_root` — ValueError on ``/``, ``$HOME``,
    repo roots, mount points, relative paths, BEFORE any probe; C4); an
    ABSENT root is an explicit ``skip_reason`` and any other enumeration
    failure raises (M1 — never an indistinguishable empty sweep). Round 3
    C4: the validator RETURNS the canonical root and enumeration /
    containment / candidate construction all use that single returned path
    (a post-validation symlink-target swap of the raw root is inert).

    Pre-gates (every early exit is a KEEP row; g4/g4b escalate through the
    D6 leg-scoped dedup gate):

    - g1  name not ``issue-<N>`` shaped -> ``slurm-src-unrecognized-kept``;
    - g2  not uid-owned -> ``slurm-src-not-owned-kept``;
    - g3  symlink / non-directory entry, or resolved path escaping the
      staging root -> ``slurm-src-containment-kept`` (never followed);
    - g4  owning issue's status not in ``terminal_statuses`` (incl.
      unresolvable) -> ``slurm-src-active-kept`` + escalate;
    - g4b the status probe RAISED -> ``slurm-src-status-probe-failed-kept``
      + escalate.

    There is deliberately NO durable-path presence gate (plan #2147 §0,
    binding reconcile): these trees are full repo checkouts, so committed
    ``store/`` / ``eval_results/`` content is EXPECTED — the shared core's
    native ``under_durable`` rule already PROOF-gates every file under those
    components per-file against the odb (denying the small-text tolerance
    there), which is strictly STRONGER than a presence block for
    verified-reproducible checkouts.

    STRICT opt-in, hermetic by construction: runs only when ``staging_root``
    AND ``main_repo`` are non-None and both kill-switch layers are unset
    (:func:`slurm_src_sweep_enabled`); production ``main()`` bodies pass
    ``vm_disk_guard.slurm_src_root()`` / :func:`_resolution_root` /
    :func:`scratch_verdict_cache_path` /
    :func:`slurm_src_escalation_state_path`. ``status_resolver`` +
    ``terminal_statuses`` are REQUIRED once armed (fail fast — a
    status-blind sweep could reap an ACTIVE issue's staging tree); the
    production adapter passes ``vm_disk_guard._resolve_issue_status`` +
    ``TERMINAL_CACHE_REAP_STATUSES`` verbatim (plan D3, read-only).
    ``overlay_paths`` defaults to
    ``backends.slurm.WORKING_TREE_OVERLAY_PATHS`` (D9 — the writer's own
    constant, imported, never a re-typed literal)."""
    result = ScratchSweepResult()
    if staging_root is None or main_repo is None or not slurm_src_sweep_enabled():
        return result
    if status_resolver is None or terminal_statuses is None:
        raise ValueError(
            "sweep_slurm_src: status_resolver + terminal_statuses are REQUIRED when the "
            "sweep is armed — a status-blind slurm-src sweep could reap an ACTIVE issue's "
            "staging tree (fail fast, never a silent default)"
        )
    # Round 2 C4: the staging-root contract is validated BEFORE any
    # enumeration / status probe / evidence probe — a misconfigured
    # EPS_SLURM_SRC_ROOT (``/``, ``$HOME``, a repo root, a mount point)
    # aborts the armed sweep loudly instead of turning ``issue-<N>`` dirs
    # under a broad anchor into deletion candidates. Round 3 C4: the
    # VALIDATED CANONICAL root is the single path used for enumeration,
    # containment, and candidate construction below — a symlink-target swap
    # after validation can no longer redirect the armed sweep into an
    # unvalidated root.
    root = _assert_safe_slurm_src_root(staging_root)
    if overlay_paths is None:
        overlay_paths = WORKING_TREE_OVERLAY_PATHS
    now = time.time() if now is None else now
    if min_age_hours is None:
        min_age_hours = _noncanonical_min_age_hours()
    window_start = now - min_age_hours * 3600.0
    cache = _ScratchVerdictCache(verdict_cache_path)
    floor = _scratch_escalate_floor_bytes()
    gate = _slurm_src_escalation_gate(escalation_state_path, apply=apply, now=now)
    # Round 2 M1: an absent staging root is an EXPLICIT skip (surfaced via
    # ``skip_reason``, distinct from "enumerated, zero candidates"); any
    # OTHER enumeration failure RAISES — a permission/IO error silently
    # reported as an empty sweep would hide a broken tier indefinitely.
    try:
        names = sorted(os.listdir(root))
    except FileNotFoundError:
        result.skip_reason = (
            f"staging root absent: {staging_root} (resolves to {root}; "
            "no SLURM staging trees on this machine)"
        )
        return result
    except OSError as exc:
        raise RuntimeError(
            f"sweep_slurm_src: cannot enumerate staging root {staging_root} "
            f"(resolves to {root}; {exc.__class__.__name__}: {exc}) — "
            "refusing to report an empty sweep"
        ) from exc
    root_real = str(root)  # already canonical (round 3 C4) — never re-resolved
    for name in names:
        cand = root / name
        row: dict = {"path": str(cand), "name": name, "leg": "slurm-src"}
        result.rows.append(row)

        def _finish(
            disposition: str, reason: str, *, row: dict = row, escalate: bool | None = None
        ) -> None:
            _scratch_row_finish(
                row,
                disposition,
                reason,
                leg="slurm-src",
                apply=apply,
                floor=floor,
                escalate=escalate,
                escalation_gate=gate,
            )

        m = _SLURM_SRC_NAME_RE.match(name)
        if m is None:  # g1 — name shape
            _finish(
                "slurm-src-unrecognized-kept", "name not issue-<N> shaped; KEPT", escalate=False
            )
            continue
        issue_n = int(m.group(1))
        row["issue"] = issue_n
        if not _tmp_entry_owned(cand):  # g2 — uid ownership (lstat, link itself)
            _finish(
                "slurm-src-not-owned-kept",
                "entry not owned by the current uid; KEPT",
                escalate=False,
            )
            continue
        # g3 — containment: a REAL directory whose resolved path stays
        # strictly inside the staging root (a symlinked entry is never
        # followed; rmtree/worktree-remove must never chase an escape).
        try:
            st = cand.lstat()
        except OSError:
            _finish("slurm-src-containment-kept", "lstat failed; KEPT", escalate=False)
            continue
        if not stat.S_ISDIR(st.st_mode):
            _finish(
                "slurm-src-containment-kept",
                "entry is a symlink or non-directory — never followed; KEPT",
                escalate=False,
            )
            continue
        real = os.path.realpath(cand)
        if real == root_real or os.path.commonpath([root_real, real]) != root_real:
            _finish(
                "slurm-src-containment-kept",
                f"resolved path escapes the staging root ({real}); KEPT",
                escalate=False,
            )
            continue
        try:  # g4/g4b — terminal-status gate (read-only, plan D3)
            status = status_resolver(issue_n)
        except Exception as exc:  # g4b: the probe ITSELF failed — keep + escalate
            row["bytes"] = _dir_size_bytes(cand)
            result.total_discovered_bytes += row["bytes"]
            row["reason_slug"] = f"status-probe-{exc.__class__.__name__}"  # M2 dedup slug
            _finish(
                "slurm-src-status-probe-failed-kept",
                f"status probe failed for issue {issue_n} ({exc.__class__.__name__}: {exc}); KEPT",
                escalate=True,
            )
            continue
        if status is None or status not in terminal_statuses:  # g4 — active/unresolved
            row["status"] = status
            row["bytes"] = _dir_size_bytes(cand)
            result.total_discovered_bytes += row["bytes"]
            row["reason_slug"] = f"status-{status or 'unresolved'}"  # M2 dedup slug
            _finish(
                "slurm-src-active-kept",
                f"issue {issue_n} status {status or 'unresolved'} not terminal-for-reap; KEPT",
                escalate=True,
            )
            continue
        row["status"] = status
        _sweep_scratch_candidate(
            cand,
            row,
            leg="slurm-src",
            apply=apply,
            main_repo=main_repo,
            window_start=window_start,
            min_age_hours=min_age_hours,
            now=now,
            cache=cache,
            result=result,
            floor=floor,
            overlay_paths=overlay_paths,
            escalation_gate=gate,
        )
    cache.save()
    return result


def sweep_tmp_uv_project_files(
    tmp_root: Path | None,
    *,
    apply: bool,
    main_repo: Path | None,
    now: float | None = None,
) -> ScratchSweepResult:
    """Detect (and, evidence-licensed, QUARANTINE — never delete) stray
    top-level uv PROJECT FILES ``tmp_root/{pyproject.toml,uv.toml,uv.lock}``
    (#2377). A stray pair poisons uv project discovery for every /tmp-cwd
    ``uv run`` fleet-wide, so the hazard is CORRECTNESS, not bytes: every row
    lands a sidecar event via :func:`append_disk_guard_event` regardless of
    size (no byte floor) and in BOTH modes — ``apply=False`` rows included
    (plan v3 §1: EVERY row, ``would-quarantine`` too, persists durably; see
    :func:`_uvproj_finish`) — with a reason naming the uv blast radius.

    STRICT opt-in, same hermeticity contract as :func:`sweep_tmp_scratch`:
    no-ops unless ``tmp_root`` AND ``main_repo`` are explicitly non-None
    (production ``main()`` bodies pass :func:`production_tmp_root` /
    :func:`_resolution_root`; library/test callers default to no-/tmp) and
    both kill-switch layers are unset (:func:`tmp_scratch_sweep_enabled` —
    acceptance criterion 3).

    Per-candidate gate order (every early exit is a KEEP; rows use
    ``leg: "tmp-uvproj"``):

    1. ``lstat``: symlink / non-regular => ``tmp-uvproj-nonregular-escalated``;
    2. foreign uid (:func:`_tmp_entry_owned` False) =>
       ``tmp-uvproj-foreign-owner-escalated`` (sticky-bit /tmp forbids
       renaming another uid's file anyway);
    3. open-handle probe (:func:`_scratch_live_process_hit` in ``exact``
       file-granular mode; the ``probe-unavailable`` sentinel fails toward
       keep) => ``tmp-uvproj-live-process-kept``;
    4. evidence: the file is NON-EMPTY and its git blob sha1
       (:func:`_blob_sha1_from_fd`) exists in ``main_repo``'s odb
       (:func:`_git_first_missing_blob`). The non-empty requirement pins the
       known hermeticity trap — an empty file's blob exists in EVERY repo.
       No proof => ``tmp-uvproj-unverified-escalated`` (KEPT + escalated;
       NEVER an age-gated deletion — age never licenses anything here);
    5. freshness grace: mtime younger than
       :data:`UVPROJ_RECENT_GRACE_SECONDS` =>
       ``tmp-uvproj-recent-escalated`` (KEPT this pass; a recent row is
       already VERIFIED by gate order, so the next pass quarantines once the
       file is quiescent);
    6. ``apply=False`` => ``tmp-uvproj-would-quarantine``;
    7. apply: pre-rename re-``lstat`` BOUND TO THE VERIFIED INODE — the
       hashing fd stays open through the rename so the hashed
       ``(st_dev, st_ino)`` cannot be recycled, and the fresh lstat must
       match it AND the evidence-read size + mtime (re-bound / changed =>
       ``tmp-uvproj-reap-aborted-recency``, the :func:`_reap_scratch_tree`
       fresh-recheck idiom hardened per the round-2 ``uvproj-evidence-toctou``
       concern) — then a same-filesystem ``os.rename`` into a FRESH private
       quarantine dir ``tmp_root/eps-quarantine-uvproj-<UTC-ts>-<rand>.q/<name>``
       (:func:`_uvproj_quarantine_dir`: ``tempfile.mkdtemp`` — atomically
       fresh + unpredictable, verified real / owned / 0700, so a pre-created
       symlink or dir at a predictable path can never be adopted — and the
       destination is asserted non-existent before the rename, so the move
       can never replace an existing entry) =>
       ``tmp-uvproj-quarantined`` (the row carries the quarantine path —
       the reversible restore point — plus the evidence string). Dir-setup
       or rename failure => ``tmp-uvproj-quarantine-failed`` (KEPT +
       escalated; never a fallback delete). Accepted residual (reconciler,
       round 1): a swap inside the lstat->rename window is bounded to a
       REVERSIBLE move into the private 0700 dir — never deletion, never an
       escape from ``tmp_root``.

    NO deletion path exists in this arm at all — the only mutation is the
    same-fs rename into the quarantine dir. Returns a
    :class:`ScratchSweepResult` (``bytes_freed`` stays 0: a rename frees
    nothing; ``total_discovered_bytes`` sums candidate sizes)."""
    result = ScratchSweepResult()
    if tmp_root is None or main_repo is None or not tmp_scratch_sweep_enabled():
        return result
    now = time.time() if now is None else now
    quarantine_dir: Path | None = None
    for name in UV_PROJECT_POISON_NAMES:
        cand = tmp_root / name
        try:
            st = cand.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            row = {"path": str(cand), "name": name, "leg": "tmp-uvproj"}
            result.rows.append(row)
            _uvproj_finish(
                row,
                "tmp-uvproj-unverified-escalated",
                f"lstat failed ({exc}); KEPT — {UVPROJ_BLAST_RADIUS}",
                apply=apply,
            )
            continue
        row = {"path": str(cand), "name": name, "leg": "tmp-uvproj"}
        result.rows.append(row)
        if not stat.S_ISREG(st.st_mode):
            _uvproj_finish(
                row,
                "tmp-uvproj-nonregular-escalated",
                f"symlink / non-regular entry; KEPT — {UVPROJ_BLAST_RADIUS}",
                apply=apply,
            )
            continue
        row["bytes"] = st.st_size
        result.total_discovered_bytes += st.st_size
        if not _tmp_entry_owned(cand):
            _uvproj_finish(
                row,
                "tmp-uvproj-foreign-owner-escalated",
                f"owned by another uid (sticky-bit /tmp forbids renaming it); "
                f"KEPT — {UVPROJ_BLAST_RADIUS}",
                apply=apply,
            )
            continue
        hit = _scratch_live_process_hit(cand, exact=True)
        if hit is not None:
            _uvproj_finish(
                row,
                "tmp-uvproj-live-process-kept",
                f"live process holds the file ({hit}); KEPT — {UVPROJ_BLAST_RADIUS}",
                apply=apply,
            )
            continue
        opened = _open_scratch_regular(cand)
        if opened is None:
            _uvproj_finish(
                row,
                "tmp-uvproj-unverified-escalated",
                f"open/fstat failed (race or non-regular swap); KEPT — {UVPROJ_BLAST_RADIUS}",
                apply=apply,
            )
            continue
        fd, fst = opened
        try:
            empty = fst.st_size == 0
            sha = None if empty else _blob_sha1_from_fd(fd, fst.st_size)
            if empty:
                _uvproj_finish(
                    row,
                    "tmp-uvproj-unverified-escalated",
                    "empty file — never evidence-licensed (an empty blob exists in "
                    f"every repo); KEPT — {UVPROJ_BLAST_RADIUS}",
                    apply=apply,
                )
                continue
            if sha is None:
                _uvproj_finish(
                    row,
                    "tmp-uvproj-unverified-escalated",
                    f"content changed under hashing (truncate/grow race); "
                    f"KEPT — {UVPROJ_BLAST_RADIUS}",
                    apply=apply,
                )
                continue
            if _git_first_missing_blob(main_repo, [sha]) != -1:
                _uvproj_finish(
                    row,
                    "tmp-uvproj-unverified-escalated",
                    f"no git-blob identity proof in the main repo odb "
                    f"(sha {sha[:12]}); KEPT — {UVPROJ_BLAST_RADIUS}",
                    apply=apply,
                )
                continue
            row["evidence"] = f"git-blob:{sha}"
            age_s = now - fst.st_mtime
            row["mtime_age_seconds"] = round(age_s, 1)
            if age_s < UVPROJ_RECENT_GRACE_SECONDS:
                _uvproj_finish(
                    row,
                    "tmp-uvproj-recent-escalated",
                    f"verified, but written {age_s:.0f}s ago "
                    f"(< {UVPROJ_RECENT_GRACE_SECONDS:.0f}s grace — recency is only a "
                    f"KEEP signal; the next pass acts once quiescent); "
                    f"KEPT — {UVPROJ_BLAST_RADIUS}",
                    apply=apply,
                )
                continue
            if not apply:
                _uvproj_finish(
                    row,
                    "tmp-uvproj-would-quarantine",
                    f"would quarantine (evidence: {row['evidence']}) — {UVPROJ_BLAST_RADIUS}",
                    apply=apply,
                )
                continue
            # The mutation is BOUND to the verified inode (round-2
            # uvproj-evidence-toctou fix): ``fd`` — still open here, closed
            # only by the enclosing ``finally`` — pins the hashed object, so
            # its (st_dev, st_ino) cannot be recycled, and a fresh lstat
            # immediately before the rename must resolve the pathname to that
            # SAME inode with the SAME size + mtime. Any mismatch =>
            # abort-to-KEEP (never quarantine unverified bytes under stale
            # evidence).
            try:
                re_st = cand.lstat()
            except OSError:
                _uvproj_finish(
                    row,
                    "tmp-uvproj-reap-aborted-recency",
                    f"file vanished between evidence read and rename; KEPT — {UVPROJ_BLAST_RADIUS}",
                    apply=apply,
                )
                continue
            if (re_st.st_dev, re_st.st_ino) != (fst.st_dev, fst.st_ino):
                _uvproj_finish(
                    row,
                    "tmp-uvproj-reap-aborted-recency",
                    f"path re-bound to a different inode between evidence read and "
                    f"rename (swap race); KEPT — {UVPROJ_BLAST_RADIUS}",
                    apply=apply,
                )
                continue
            if re_st.st_size != fst.st_size or re_st.st_mtime != fst.st_mtime:
                _uvproj_finish(
                    row,
                    "tmp-uvproj-reap-aborted-recency",
                    f"file changed between evidence read and rename; KEPT — {UVPROJ_BLAST_RADIUS}",
                    apply=apply,
                )
                continue
            try:
                if quarantine_dir is None:
                    quarantine_dir = _uvproj_quarantine_dir(tmp_root, now)
                dest = quarantine_dir / name
                # Inside a directory mkdtemp just created private + empty,
                # non-existence of ``dest`` is guaranteed (mode 0700: no other
                # uid can create entries; this sweep writes each fixed name at
                # most once). Assert it anyway so the move can NEVER replace
                # an existing entry (round-2 destination-unsafe fix).
                if os.path.lexists(dest):
                    raise OSError(f"quarantine destination {dest} unexpectedly exists")
                os.rename(cand, dest)
            except OSError as exc:
                _uvproj_finish(
                    row,
                    "tmp-uvproj-quarantine-failed",
                    f"quarantine dir setup / rename failed ({exc}); KEPT — {UVPROJ_BLAST_RADIUS}",
                    apply=apply,
                )
                continue
            row["quarantine_path"] = str(dest)
            _uvproj_finish(
                row,
                "tmp-uvproj-quarantined",
                f"quarantined to {dest} (same-fs rename — reversible restore point, "
                f"never deleted; evidence: {row['evidence']}) — {UVPROJ_BLAST_RADIUS}",
                apply=apply,
            )
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)
    return result


def _uvproj_quarantine_dir(tmp_root: Path, now: float) -> Path:
    """Create the #2377 quarantine dir ATOMICALLY FRESH and verify it before
    any file is moved in (round-2 fix for the BLOCKER
    ``uvproj-quarantine-destination-unsafe``: the round-1 predictable
    timestamped path with ``mkdir(exist_ok=True)`` silently ADOPTED a
    pre-created symlink-to-directory, letting the subsequent rename escape
    ``tmp_root`` and replace an entry at the redirect target).

    ``tempfile.mkdtemp`` creates a brand-new directory (mode 0700, masked by
    umask) at an UNPREDICTABLE name and fails rather than reusing anything
    already at the chosen name — ``os.mkdir`` underneath refuses an existing
    entry, symlink included — so a pre-created entry can never be adopted.
    The literal ``.q`` suffix keeps the name-final character non-numeric so
    the #911 P2 issue-suffix route (name-final ``_<N>``) can never key the
    dir to an issue even if the random middle happens to end ``_<digits>``.
    Post-creation the dir is verified via ``lstat``: a REAL directory (never
    a symlink — ``lstat`` does not follow), owned by our uid; then chmod'd to
    exactly 0700 (normalizing any umask residue) and re-verified. Any
    surprise raises ``OSError`` — the caller's ``tmp-uvproj-quarantine-failed``
    KEEP + escalate path; never a fallback delete. Returns the verified dir.
    """
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    qdir = Path(
        tempfile.mkdtemp(prefix=f"eps-quarantine-uvproj-{ts}-", suffix=".q", dir=str(tmp_root))
    )
    st = os.lstat(qdir)
    if not stat.S_ISDIR(st.st_mode):
        raise OSError(f"quarantine dir {qdir} is not a real directory (mode {st.st_mode:o})")
    if st.st_uid != os.getuid():
        raise OSError(f"quarantine dir {qdir} owned by uid {st.st_uid}, not {os.getuid()}")
    os.chmod(qdir, 0o700)  # verified a real non-symlink dir above; normalize umask residue
    st = os.lstat(qdir)
    if stat.S_IMODE(st.st_mode) != 0o700:
        raise OSError(f"quarantine dir {qdir} mode {stat.S_IMODE(st.st_mode):o} != 0700")
    return qdir


def _uvproj_finish(row: dict, disposition: str, reason: str, *, apply: bool) -> None:
    """Record one #2377 uv-project row: stamp disposition + reason, print the
    stderr line, and append the sidecar event DURABLY IN BOTH MODES (no byte
    floor — the hazard is fleet correctness, not bytes; acceptance criterion
    2; plan v3 §1 requires EVERY row, ``would-quarantine`` included, to land
    in the sidecar — round-2 fix for ``uvproj-report-sidecar-missing``).

    ``append_disk_guard_event`` is therefore called with ``apply=True``
    unconditionally — the least-invasive shape: the helper's global
    report-only contract (``apply=False`` prints and returns) stays unchanged
    for every existing caller, and only this arm's rows opt in to durable
    report-mode persistence. The event carries the row's own ``apply`` flag
    so the sidecar records which mode observed it."""
    row["disposition"] = disposition
    row["reason"] = reason
    print(f"  [tmp-uvproj] {disposition}: {row['path']} — {reason}", file=sys.stderr)
    event = {
        "kind": disposition,
        "path": row["path"],
        "bytes": row.get("bytes", 0),
        "reason": reason,
        "evidence": row.get("evidence"),
        "apply": apply,
    }
    if row.get("quarantine_path"):
        event["quarantine_path"] = row["quarantine_path"]
    append_disk_guard_event(event, apply=True)


def _noncanonical_reap_gates(
    cache_dir: Path,
    *,
    issue_n: int,
    apply: bool,
    min_age_hours: float,
    now: float,
    data_repo_toplevel_cache: dict[str, frozenset[str] | None],
    staging: bool = False,
    size_bytes: int = 0,
    git_evidence_repo: Path | None = None,
) -> str | tuple[str, str]:
    """Run gates 1.5 -> 1.6 -> 1.7 on a NON-CANONICAL candidate (#911),
    ordered cheap-first (recency needs only stats, the durable scan is a name
    rglob, the evidence gate may make one HF call — memoized in
    ``data_repo_toplevel_cache``).

    STAGING candidates (#2095, ``staging=True``): gate 1.5 reads
    TOP-LEVEL-ONLY recency; a NEW gate 1.55 (cross-issue CONTENT
    hard-escalate) runs between 1.5 and 1.6; gate 1.7's ``unverified-kept``
    disposition carries a class label (``derived-partial-mirror`` /
    ``orphan-no-mirror``), the empty-dir license is allowed (prefix-only
    extraction removes the P2 foreign-mkdtemp concern), and a branch-(b)
    license on a candidate above the probe floor (``size_bytes`` vs
    :func:`_staging_probe_floor_bytes`) must additionally pass the sampled
    mirror probe (:func:`_staging_mirror_probe`) or it is refused as
    ``unverified-kept:probe-failed``.

    Gate 1.7 evidence branch (c) (#2127, ``git_evidence_repo`` non-None):
    an ISSUE-KEYED /tmp candidate (P1/P2 — never staging, never a P3
    ``data/`` name) with no HF evidence may still be licensed by per-file
    git-blob reproducibility against the main repo
    (:func:`_tmp_git_evidence_branch_c`). STRICT opt-in: only production
    ``main()`` bodies pass a repo (the hermetic default keeps library/test
    callers' fixtures from being licensed by the REAL repo's odb — an empty
    file's blob exists in every repo).

    Returns the positive-evidence STRING when the reap is licensed, or a
    ``(disposition, skip_reason)`` tuple when blocked (fail-toward-keep; the
    per-gate sidecar row is already appended by the time this returns)."""
    rel = _rel_name(cache_dir)
    # Gate 1.5 — recency (sidecar row appended inside on a block).
    recency_reason = _noncanonical_recency_blocked(
        cache_dir,
        issue_n=issue_n,
        apply=apply,
        min_age_hours=min_age_hours,
        now=now,
        staging=staging,
    )
    if recency_reason is not None:
        return ("recency-kept", recency_reason)
    # Gate 1.55 (STAGING only, #2095) — cross-issue CONTENT hard-escalates:
    # a dir named for issue A holding issue B's tensors is NEVER auto-reaped
    # in v1, even with full mirror evidence (a name-keyed terminal reap would
    # destroy a parked task's store; the human acts off the escalation row).
    if staging:
        cross = _staging_cross_issue_content(cache_dir, issue_n)
        if cross:
            content_issues = sorted(
                {
                    int(m.group(1))
                    for h in cross
                    if (m := _TMP_ISSUE_PREFIX_RE.match(Path(h).name)) is not None
                }
            )
            shown = ", ".join(cross[:5])
            reason = (
                f"staging content names other issue(s) {content_issues or ['<unreadable>']} "
                f"({shown}) — cross-issue content hard-escalates; KEPT "
                f"(human decision off the escalation row; never auto-reaped in v1)"
            )
            append_disk_guard_event(
                {
                    "kind": "staging-cache-cross-issue-kept",
                    "task": issue_n,
                    "path": rel,
                    "content_issues": content_issues,
                    "reason": reason,
                },
                apply=apply,
            )
            return ("cross-issue-kept", reason)
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
    # STAGING candidates keep the empty-dir license unconditionally (#2095 —
    # prefix-only extraction removes the P2 foreign-mkdtemp concern that
    # motivated the /tmp restriction).
    p2_only = False if staging else _p2_suffix_only(cache_dir.name)
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
    branch_c_note = ""
    if (
        evidence is None
        and git_evidence_repo is not None
        and not staging
        and _DATA_NONCANONICAL_CACHE_RE.match(cache_dir.name) is None
    ):
        # Gate-1.7 evidence branch (c) (#2127) — /tmp P1/P2 legs only.
        git_ev, git_detail = _tmp_git_evidence_branch_c(cache_dir, main_repo=git_evidence_repo)
        if git_ev is not None:
            evidence = git_ev
        else:
            first = git_detail.get("first_unverified")
            branch_c_note = f"; git-blob branch (c): {git_detail.get('reason')}" + (
                f" (first unverified: {first})" if first else ""
            )
    if evidence is None:
        disposition = "unverified-kept"
        if staging:
            label = _staging_unverified_class(
                cache_dir, data_repo_toplevel_cache.get(hf_data_repo())
            )
            if label is not None:
                disposition = f"unverified-kept:{label}"
        reason = (
            "no positive re-downloadability evidence (no hub-layout markers; "
            "top-level names not verified as data-repo prefixes"
            + ("; P2 suffix-only route — requires non-empty positive evidence" if p2_only else "")
            + branch_c_note
            + ") — KEPT (escalate-only, never deleted)"
        )
        row = {
            "kind": "noncanonical-cache-unverified-kept",
            "task": issue_n,
            "path": rel,
            "reason": reason,
        }
        if disposition != "unverified-kept":
            row["disposition"] = disposition  # staging class label (#2095)
        append_disk_guard_event(row, apply=apply)
        return (disposition, reason)
    # Delta-9 sampled mirror probe (#2095, STAGING only): a branch-(b)
    # name-match license on a candidate above the probe floor must survive a
    # byte-equal existence probe of the candidate's largest file on the data
    # repo. Branch (a) hub-layout and the empty-dir license are unchanged.
    if (
        staging
        and evidence.startswith("data-repo-prefix mirror")
        and size_bytes > _staging_probe_floor_bytes()
    ):
        probe_fail = _staging_mirror_probe(cache_dir)
        if probe_fail is not None:
            reason = (
                f"branch-(b) name-match license refused for a "
                f"{size_bytes / 1e9:.2f} GB staging candidate — sampled mirror probe: "
                f"{probe_fail}; KEPT (escalate-only, never deleted)"
            )
            append_disk_guard_event(
                {
                    "kind": "noncanonical-cache-unverified-kept",
                    "task": issue_n,
                    "path": rel,
                    "disposition": "unverified-kept:probe-failed",
                    "reason": reason,
                },
                apply=apply,
            )
            return ("unverified-kept:probe-failed", reason)
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
    # "external-target-kept" | "failed"; STAGING candidates (#2095) may add
    # "cross-issue-kept" and the labeled "unverified-kept:derived-partial-
    # mirror" / "unverified-kept:orphan-no-mirror" /
    # "unverified-kept:probe-failed" variants). Canonical hf_dl/g*_dl caches are
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
    staging_roots: list[Path] | None = None,
    exclude_scratch_shapes: bool = False,
    git_evidence_repo: Path | None = None,
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

    STAGING candidates (#2095): with an EXPLICIT ``staging_roots`` (same
    strict opt-in as ``tmp_root`` — only the CLI ``main()`` bodies pass
    :func:`production_staging_roots`; the library default ``None`` never
    touches any staging root), the sweep additionally covers TOP-LEVEL
    issue-keyed dirs under each root (:func:`_staging_cache_dirs`,
    PREFIX-ONLY extraction — no P2 suffix route). They ride the SAME gate
    chain with three staging-specific tightenings: gate 1.5 reads
    TOP-LEVEL-ONLY recency (no rglob over a multi-100-GB tree); a NEW gate
    1.55 between 1.5 and 1.6 HARD-ESCALATES cross-issue CONTENT (sidecar
    kind ``staging-cache-cross-issue-kept`` with content-issue attribution —
    never auto-reaped in v1, even mirror-verified); gate 1.7's
    ``unverified-kept`` disposition carries a class label
    (``:derived-partial-mirror`` / ``:orphan-no-mirror``) and a branch-(b)
    license above ~1 GB (:func:`_staging_probe_floor_bytes`) must pass the
    sampled mirror probe or is refused as ``unverified-kept:probe-failed``.
    ``EPM_SKIP_STAGING_CACHE_SWEEP=1`` kills the staging leg alone.

    #2127 knobs (both hermetic-default, main()-only opt-ins like
    ``tmp_root``): ``git_evidence_repo`` arms gate 1.7's evidence branch (c)
    — git-blob reproducibility against that repo for the /tmp P1/P2 legs;
    ``exclude_scratch_shapes`` skips issue-keyed /tmp entries that are ALSO
    scratch-shaped, for the ONE guard invocation that runs
    :func:`sweep_tmp_scratch` in the same pass (no double-attribution).

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
        issue_n,
        data_root=data_root,
        tmp_root=tmp_root,
        sweep_tmp=sweep_tmp,
        staging_roots=staging_roots,
        exclude_scratch_shapes=exclude_scratch_shapes,
    )
    noncanon_keys = {os.path.normpath(str(p)) for p in noncanon}
    # Staging-origin tagging (#2095): the staging discovery is deterministic,
    # so re-running it yields the same set — a second normalized-path set
    # mirrors ``noncanon_keys`` and routes those candidates through the
    # staging-specific gate variants (top-level recency, gate 1.55, class
    # labels, the sampled mirror probe).
    staging_keys = {os.path.normpath(str(p)) for p in _staging_cache_dirs(issue_n, staging_roots)}
    now = time.time()
    min_age = _noncanonical_min_age_hours()
    for cache_dir in _dedup_nested([*canonical, *noncanon]):
        rel = _rel_name(cache_dir)
        res.sizes_bytes[rel] = _dir_size_bytes(cache_dir)
        is_noncanonical = os.path.normpath(str(cache_dir)) in noncanon_keys
        is_staging = os.path.normpath(str(cache_dir)) in staging_keys
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
            staging=is_staging,
            issue_n=issue_n,
            apply=apply,
            min_age_hours=min_age,
            now=now,
            hf_sizes_cache=hf_sizes_cache,
            data_repo_toplevel_cache=data_repo_toplevel_cache,
            git_evidence_repo=git_evidence_repo,
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
    staging: bool = False,
    git_evidence_repo: Path | None = None,
) -> bool:
    """Run the per-candidate reap gates AFTER gate 1 (#773): gates 1.5/1.6/1.7
    (#911) on a NON-CANONICAL candidate — with the staging variants (1.55,
    top-level recency, class labels, mirror probe; #2095) when ``staging`` —
    or gate 2 (#679 nested-store parity) on a canonical one. Records the
    skip + disposition/evidence on ``res``.
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
            staging=staging,
            size_bytes=res.sizes_bytes.get(rel, 0),
            git_evidence_repo=git_evidence_repo,
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
    staging_roots: list[Path] | None = None,
    git_evidence_repo: Path | None = None,
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
    touched; the cache re-downloads on demand ONLY via hub-download paths —
    place the reap strictly after the cache's LAST consumer in the whole run
    (a direct open() reader does not re-download; #1489, see
    .claude/rules/gotchas.md).

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
    ``tmp_root`` / ``sweep_tmp`` forward verbatim (#911 — same strict opt-in);
    ``staging_roots`` forwards verbatim too (#2095 — same strict opt-in), as
    does ``git_evidence_repo`` (#2127 gate-1.7 branch (c) opt-in)."""
    return clean_issue_downloads(
        issue_n,
        apply=apply,
        data_root=data_root,
        tmp_root=tmp_root,
        sweep_tmp=sweep_tmp,
        staging_roots=staging_roots,
        git_evidence_repo=git_evidence_repo,
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
            "block, and positive re-downloadability evidence; plus the "
            "issue's own /mnt/eps-data/$USER staging dirs (#2095 — prefix-"
            "keyed, cross-issue content hard-escalates, kill switch "
            "EPM_SKIP_STAGING_CACHE_SWEEP=1). "
            "Re-downloadable; store/ + eval_results/ are never touched. "
            "Dry-run by default."
        )
    )
    ap.add_argument("issue", type=int, help="Issue / task number N.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (default: dry-run, report what would be removed).",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Explicit no-op alias of the default preview mode (report what would "
            "be removed, delete nothing). Accepted so the sibling janitor's "
            "`pod.py cleanup --dry-run` spelling does not exit 2; mutually "
            "exclusive with --apply."
        ),
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
        params = inspect.signature(cleaner).parameters
        if "tmp_root" in params:
            kwargs["tmp_root"] = production_tmp_root()
        if "staging_roots" in params:
            kwargs["staging_roots"] = production_staging_roots()
        if "git_evidence_repo" in params:
            # #2127 gate-1.7 branch (c): git-blob evidence against the main
            # checkout — a main()-only opt-in like tmp_root above.
            kwargs["git_evidence_repo"] = _resolution_root()
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
