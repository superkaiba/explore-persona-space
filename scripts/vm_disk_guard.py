#!/usr/bin/env python3
"""VM root-disk guard — tiered safe cleanup when ``/`` crosses a threshold.

The VM root disk fills because each experiment downloads its source data into
``data/issue_<N>/hf_dl/`` + ``g*_dl/`` caches that nothing reclaims (incident
2026-06-25: ``/`` hit 100% full, one finished experiment held 97 GB), plus the
``uv`` package cache and accumulating logs. This guard reads ``df`` for ``/``
and, when usage exceeds a threshold (default 85%, env ``EPS_VM_DISK_THRESHOLD``),
runs five TIERS of strictly-safe cleanup, reporting bytes freed per tier:

  (a) ``uv cache prune`` (skipped gracefully if the uv lock is held — never
      ``--force``).
  (b) ``data/issue_*/hf_dl`` + ``g*_dl`` download caches for issues whose task
      status is TERMINAL (``completed`` / ``archived`` / ``awaiting_promotion``).
      Status is resolved READ-ONLY via the task workflow — task state is NEVER
      mutated. An issue at any ACTIVE status (its caches may be in use) is
      skipped. (#911) The tier ALSO covers NON-CANONICAL issue-keyed caches —
      top-level ``/tmp`` dirs named ``i<N>*`` / ``issue<N>*`` / ``issue-<N>*``
      / ``issue_<N>*`` / ``*_<N>`` and whole-dir ``data/``
      ``issue…<N>…{_dl,_hfstage,_cache}`` dirs — under the same
      terminal-reap / active-escalate contract PLUS three extra gates in
      ``clean_issue_downloads`` (48h recency, nested ``store/`` +
      ``eval_results/`` block, positive re-downloadability evidence). The
      /tmp part is a ``main()``-only opt-in (``tmp_root=production_tmp_
      root()``); library calls stay hermetic. Structured outcome rows
      (``active_cache_attributions`` / ``noncanonical_candidates`` /
      ``total_discovered_bytes``) ride the ``--json`` output — report-only
      escalation persists nothing to the sidecar, so dry-run acceptance reads
      the JSON.
  (d) The VM's pod-style ``/workspace/.cache/huggingface`` hub cache (#911):
      age-gated ``delete_revisions`` of repos unused >= 14 days (env
      ``EPS_VM_WORKSPACE_HF_CACHE_MAX_AGE_DAYS``), pod-guarded twice
      (``os.path.ismount('/workspace')`` OR pod-side detection refuses) so it
      can never run where /workspace is a real volume; every failure degrades
      to a skipped tier. Boot-disk pass only, same ``main()``-only opt-in.
  (e) The HOME HF hub cache ``~/.cache/huggingface/hub`` (#1376 + #1377, two
      independently-landed tiers reconciled into ONE) — the fleet's dominant
      root-disk consumer (the data repo accumulates one revision per pinned
      read / upload commit; 12 revisions / 76.2 GB observed at the 2026-07-16
      episode). ALWAYS attributes per-repo size / revision count /
      ``last_accessed`` age (detail lines + the structured
      ``hf_repo_attributions`` ``--json`` field), escalates any single repo
      > 40 GB (``EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB``) with a per-revision
      breakdown (sidecar + Telegram, deduped per (repo, band)), and reaps
      safely on ``--apply``: (arm 2) within fresh multi-revision repos,
      unref'd non-newest revisions with ``last_modified`` older than 7 d
      (``EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS``) AND no fresh EXCLUSIVE-blob
      atime — the newest + every ref'd revision per repo is ALWAYS kept;
      (arm 1) whole repos whose repo-level ``last_accessed`` is older than
      the same window (ref'd revisions included — this covers stale models).
      Deletion goes exclusively through
      ``HFCacheInfo.delete_revisions().execute()`` (blob-refcount safe);
      every failure degrades toward KEEP. A later FileNotFoundError on a
      trimmed snapshot path means tier (e) trimmed it — re-download on demand
      (the data lives on HF), not data loss. Boot-disk pass only, same
      ``main()``-only opt-in as tier (d).
  (c) Stale logs: ``logs/**/*.log`` older than N days (default 14, env
      ``EPS_VM_DISK_LOG_MAX_AGE_DAYS``) plus ``/tmp/*.log`` of the same age.

Report-only by default; ``--apply`` acts. After cleanup, if usage is STILL over
the threshold, a loud WARNING line is printed and (when present) the my-goat
``telegram_push.sh`` is invoked fail-soft so the disk-pressure situation is
surfaced for manual triage.

Mirrors the style of ``scripts/worktree_audit.py`` + ``scripts/gcp_audit.py``:
pure decision helpers are unit-testable, side effects are gated on ``--apply``,
and the cron wrapper (``scripts/cron_vm_disk_guard.sh``) runs ``--apply``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.task_workflow import find_task_path, repo_root

# Re-use the single source of truth for what a download cache is + how to
# delete it (so the two cleanup entrypoints can never drift on the safety
# contract — store/ + eval_results/ are KEPT in both). The sidecar-event
# helper is shared so every disk escalation lands on ONE stream.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_experiment_downloads import (
    _running_pod_side,
    _tmp_entry_owned,
    append_disk_guard_event,
    clean_issue_downloads,
    extract_issue_number,
    production_tmp_root,
)

# Default usage threshold (% of /) above which cleanup runs. Env-overridable.
DEFAULT_THRESHOLD_PCT = 85.0

# The dedicated data disk that holds the relocated `.claude/worktrees/` tree
# (task #681). It is a SECOND watched mount, distinct from `/` (the boot disk).
# The guard watches it ESCALATE-ONLY: the `/`-rooted reclaim tiers (uv cache
# prune, stale-log sweep) operate on boot-disk caches and must NOT run keyed off
# the data disk; only the active-cache escalation + the terminal-cache reap
# (the one safe data-disk arm — it reaps a TERMINAL issue's re-downloadable
# cache on EITHER disk) fire there. Default mount + env override.
DEFAULT_DATA_DISK_PATH = "/mnt/eps-data"


def data_disk_path() -> str:
    """The watched data-disk mount, env-overridable (``EPS_VM_DATA_DISK_PATH``).

    A blank / unset value falls back to the default mount. The data disk is the
    second filesystem the guard watches (escalate-only) — see
    :func:`run_guard`'s ``reclaim_tiers`` param and the #681 plan §4 Phase 4."""
    raw = os.environ.get("EPS_VM_DATA_DISK_PATH", "").strip()
    return raw or DEFAULT_DATA_DISK_PATH


def _is_mounted(path: str) -> bool:
    """True iff ``path`` is itself a mount point (a distinct filesystem).

    A real mount-presence check — NOT ``Path(path).is_dir()`` (#681 round-2
    Major). ``is_dir()`` is True for ANY directory, mounted or not: after
    Phase-1's ``sudo mkdir -p /mnt/eps-data`` (which runs BEFORE the mount) or
    after a ``nofail`` boot where the disk failed to mount, ``/mnt/eps-data``
    exists as a plain root-fs directory, and the data-disk pass would then
    misread ``/``'s statvfs (the boot disk) as data-disk usage. Comparing
    ``st_dev`` against the parent's catches that: a real mount sits on a
    different device than its parent; a plain subdirectory shares its parent's
    device. Pure ``os.stat`` — no subprocess, fast, self-evidently correct.

    Fail-soft: a missing path / stat error returns False (treated as "not
    mounted" → the pass cleanly no-ops, never reports ``/``'s usage as the data
    disk's). The filesystem root ``/`` is its own parent (``st_dev`` equal), so
    this never claims an unmounted path IS the data disk."""
    try:
        st = os.stat(path)
        parent = os.stat(os.path.join(path, os.pardir))
    except OSError:
        return False
    return st.st_dev != parent.st_dev


# Threshold-band boundaries (bytes) for the ACTIVE-task escalation dedup key. An
# active issue holding a re-downloadable cache the terminal-gate cannot reap is
# escalated (NEVER deleted); the band coarsens its footprint so a row re-fires
# only when it crosses into a bigger bucket — paired with the >25% growth
# re-alert below. ~20/50/100 GB buckets match the real incident scale.
_ACTIVE_ESCALATION_BANDS_GB = (20.0, 50.0, 100.0)

# Re-alert an already-escalated (task, band) only when the cache GREW by more
# than this fraction since the last alert — bounds churn while still surfacing a
# steadily-growing active cache.
_ACTIVE_ESCALATION_GROWTH_REALERT = 0.25

# Footprint floor (bytes) below which an active-task cache is too small to be
# worth escalating — avoids noise on trivial caches.
_ACTIVE_ESCALATION_MIN_BYTES = 5 * 10**9  # 5 GB

# Per-(task, band) escalation state (last-alerted bytes), so an already-alerted
# active cache re-fires only on a band crossing OR >25% growth. Relative to the
# repo root; one shared JSON keyed "<task>:<band_gb>".
_ACTIVE_ESCALATION_STATE_REL = Path(".claude") / "cache" / "disk-guard-active-state.json"


def _active_escalation_band_gb(bytes_: int) -> float:
    """The largest band boundary (GB) the footprint has crossed, or the cache's
    GB rounded down for anything above the top band — the coarse dedup key."""
    gb = bytes_ / 10**9
    band = 0.0
    for boundary in _ACTIVE_ESCALATION_BANDS_GB:
        if gb >= boundary:
            band = boundary
    # Above the top configured band, key on the integer-GB bucket so a runaway
    # cache still re-alerts as it climbs past 100 GB.
    return max(band, float(int(gb)) if gb >= _ACTIVE_ESCALATION_BANDS_GB[-1] else band)


def _active_escalation_state_path() -> Path:
    return repo_root() / _ACTIVE_ESCALATION_STATE_REL


def _load_active_escalation_state() -> dict:
    path = _active_escalation_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_active_escalation_state(state: dict) -> None:
    """Atomic temp+rename write of the active-escalation dedup state
    (fail-soft — a write failure is logged, never raised)."""
    dest = _active_escalation_state_path()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(dest)
    except OSError as exc:  # pragma: no cover - fail-soft I/O guard
        print(f"  WARNING: saving active-escalation state failed: {exc}", file=sys.stderr)


def _active_ack_sentinel_path(issue_n: int, band_gb: float) -> Path:
    """Per-(task, band) ack sentinel: ``touch`` it to silence this escalation
    until the cache crosses into a bigger band. Relative to the repo root."""
    return repo_root() / ".claude" / "cache" / f"disk-guard-ack-{issue_n}-{band_gb:g}"


def _should_escalate_active(
    issue_n: int,
    band_gb: float,
    bytes_: int,
    state: dict,
) -> tuple[bool, float]:
    """Decide whether to (re-)escalate an active-task cache, and the growth %
    vs the last alert. Dedups on (task, band) and re-alerts only on >25%
    growth; an ack sentinel for this (task, band) suppresses entirely."""
    if _active_ack_sentinel_path(issue_n, band_gb).exists():
        return (False, 0.0)
    key = f"{issue_n}:{band_gb:g}"
    prev = state.get(key)
    if not isinstance(prev, int | float) or prev <= 0:
        return (True, 0.0)  # first alert for this (task, band)
    growth = (bytes_ - prev) / prev
    return (growth > _ACTIVE_ESCALATION_GROWTH_REALERT, growth * 100.0)


def escalate_active_cache(
    issue_n: int,
    status: str | None,
    apply: bool,
    *,
    data_root: Path | None = None,
    state: dict | None = None,
    tmp_root: Path | None = None,
    sweep_tmp: bool = True,
    report_to: TierResult | None = None,
) -> TierResult | None:
    """ESCALATE (never delete) an ACTIVE task's re-downloadable cache that the
    terminal-status gate cannot reap.

    Sizes the cache via a dry-run of ``clean_issue_downloads`` (no deletion),
    and — when the footprint is band-worthy and not already alerted (dedup on
    (task, band) + >25%-growth re-alert + ack-sentinel suppression) — emits a
    Telegram push AND a shared-sidecar row naming the largest cache path, the
    footprint, and the SAFE reclaim command. NEVER calls rmtree. Returns a
    TierResult describing the escalation, or None when nothing was escalated.
    ``state`` (the dedup map) is read+updated in place by the caller.

    ``tmp_root``/``sweep_tmp`` forward VERBATIM into the sizing call (#911 —
    with ``tmp_root=None`` the sizing sees ZERO /tmp bytes; the tier-b caller
    threads its own opt-in through so an active owner's /tmp + P3 footprint
    is attributed). ``report_to`` (the calling tier's TierResult, #911) —
    when given — receives the DEDUP-INDEPENDENT structured rows BEFORE any
    floor/ack/band suppression: the owner's ``active_cache_attributions`` row
    + one ``noncanonical_candidates`` row per discovered non-canonical dir
    (disposition ``escalated`` / ``unresolved-kept``) + the discovered-bytes
    total. Report-only escalation persists nothing, so these fields are the
    dry-run acceptance surface."""
    sub = clean_issue_downloads(
        issue_n, apply=False, data_root=data_root, tmp_root=tmp_root, sweep_tmp=sweep_tmp
    )
    # Size from EVERY discovered cache dir, NOT bytes_freed: a large active
    # hf_dl/.../store/ correctly KEPT by the nested-store parity guard lands in
    # `skipped` (contributing 0 to bytes_freed), so bytes_freed would suppress
    # the escalation for the exact large-unmirrored-active-cache shape #679
    # targets (BLOCKER #1). total_discovered_bytes counts removed + skipped.
    bytes_ = sub.total_discovered_bytes
    if report_to is not None:
        report_to.total_discovered_bytes += bytes_
        if bytes_ > 0:
            largest_path = max(sub.sizes_bytes, key=lambda n: sub.sizes_bytes.get(n, 0), default="")
            report_to.active_cache_attributions.append(
                {"task": issue_n, "status": status, "path": largest_path, "bytes": bytes_}
            )
        owner_disposition = "escalated" if status is not None else "unresolved-kept"
        for rel in sub.noncanonical_dispositions:
            report_to.noncanonical_candidates.append(
                {
                    "path": rel,
                    "issue": issue_n,
                    "bytes": sub.sizes_bytes.get(rel, 0),
                    "disposition": owner_disposition,
                    "evidence": sub.noncanonical_evidence.get(rel, ""),
                }
            )
    if bytes_ < _ACTIVE_ESCALATION_MIN_BYTES:
        return None
    band_gb = _active_escalation_band_gb(bytes_)
    st = state if state is not None else {}
    do_alert, growth_pct = _should_escalate_active(issue_n, band_gb, bytes_, st)
    if not do_alert:
        return None
    # Largest cache path drives the human-facing pointer + reclaim command —
    # over ALL discovered dirs (removed AND skipped), so a parity-skipped store
    # still surfaces its path even though it freed nothing.
    largest = max(sub.sizes_bytes, key=lambda n: sub.sizes_bytes.get(n, 0), default="")
    reclaim_cmd = f"uv run python scripts/clean_experiment_downloads.py {issue_n} --apply"
    res = TierResult(name="active-cache-escalation")
    res.detail.append(
        f"issue {issue_n} ({status or 'active'}): ACTIVE cache {_fmt_gb(bytes_)} "
        f"at {largest} — ESCALATED (never deleted while active); "
        f"reclaim AFTER the run with `{reclaim_cmd}`"
    )
    append_disk_guard_event(
        {
            "kind": "active-cache-escalation",
            "task": issue_n,
            "status": status,
            "path": largest,
            "bytes": bytes_,
            "reclaim_cmd": reclaim_cmd,
            "band": band_gb,
            "growth_pct": round(growth_pct, 1),
        },
        apply=apply,
    )
    _telegram_push(
        f"VM disk: active task #{issue_n} ({status or 'active'}) holds "
        f"{_fmt_gb(bytes_)} of re-downloadable cache the terminal-gate can't reap. "
        f"NOT auto-deleted (active). Reclaim AFTER the run: {reclaim_cmd}",
        apply,
    )
    if apply and state is not None:
        state[f"{issue_n}:{band_gb:g}"] = bytes_
    return res


# Default age (days) above which a log file is stale and reclaimable.
DEFAULT_LOG_MAX_AGE_DAYS = 14.0

# Issue task statuses whose download caches are safe to reclaim — the work is
# done (or parked awaiting promotion), so the re-downloadable source cache is
# no longer needed. Mirrors worktree_audit.REAPABLE_ISSUE_STATUSES and the
# brief: completed / archived / awaiting_promotion. NOTE this deliberately
# differs from task_workflow.TERMINAL_STATUSES (completed/blocked/archived):
# `blocked` is excluded (a blocked task may resume and need its cache), and
# `awaiting_promotion` is included (the experiment is done, the park is just
# the user's promotion call).
TERMINAL_CACHE_REAP_STATUSES = frozenset({"completed", "archived", "awaiting_promotion"})

_TELEGRAM_PUSH_SCRIPT_DEFAULT = Path.home() / "my-goat" / "scripts" / "telegram_push.sh"


# ─── pure helpers (unit-tested) ──────────────────────────────────────────────


def over_threshold(used_pct: float, threshold_pct: float) -> bool:
    """True when ``/`` usage is strictly above the cleanup threshold."""
    return used_pct > threshold_pct


def threshold_pct() -> float:
    """Cleanup threshold (% used of /), env-overridable, clamped to (0, 100]."""
    raw = os.environ.get("EPS_VM_DISK_THRESHOLD", str(DEFAULT_THRESHOLD_PCT))
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_THRESHOLD_PCT
    if not (0.0 < val <= 100.0):
        return DEFAULT_THRESHOLD_PCT
    return val


def log_max_age_days() -> float:
    """Stale-log age cutoff in days, env-overridable, clamped to >= 0."""
    raw = os.environ.get("EPS_VM_DISK_LOG_MAX_AGE_DAYS", str(DEFAULT_LOG_MAX_AGE_DAYS))
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_LOG_MAX_AGE_DAYS
    return val if val >= 0.0 else DEFAULT_LOG_MAX_AGE_DAYS


def disk_used_pct(path: str = "/") -> float:
    """Percent used of the filesystem holding ``path`` (via ``shutil.disk_usage``)."""
    usage = shutil.disk_usage(path)
    return 100.0 * usage.used / usage.total


def disk_free_gb(path: str = "/") -> float:
    """Free GB on the filesystem holding ``path``."""
    return shutil.disk_usage(path).free / (1024**3)


def _resolve_issue_status(issue_n: int) -> str | None:
    """Task status for ``issue_n`` (the parent folder name under tasks/), or
    None when the task cannot be resolved. READ-ONLY — never mutates state."""
    try:
        return find_task_path(issue_n).parent.name
    except (FileNotFoundError, OSError, ValueError):
        return None


def _discover_data_issue_numbers(data_roots: list[Path]) -> list[int]:
    """Issue numbers that have an ``issue*<N>`` directory under ANY of the
    given ``data/`` roots (both naming conventions). Returns sorted unique
    ints; non-issue dirs are ignored."""
    found: set[int] = set()
    for data_root in data_roots:
        if not data_root.is_dir():
            continue
        for child in sorted(data_root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            for prefix in ("issue_", "issue"):
                if name.startswith(prefix):
                    rest = name[len(prefix) :]
                    # rest is "<N>" or "<N>_<slug>"
                    num = rest.split("_", 1)[0]
                    if num.isdigit():
                        found.add(int(num))
                    break
    return sorted(found)


def _discover_tmp_issue_numbers(tmp_root: Path) -> list[int]:
    """Issue numbers extracted from TOP-LEVEL dirs/symlinks of ``tmp_root``
    (uid-owned; ``ced.extract_issue_number`` over the entry NAME — the P1/P2
    non-canonical patterns, #911). Lets tier (b) visit a /tmp-ONLY issue (the
    #823 shape: a ``/tmp/fact_check_823`` staging dir with no ``data/issue*``
    dir anywhere). Never recursive; files are never considered. Returns
    sorted unique ints."""
    found: set[int] = set()
    if not tmp_root.is_dir():
        return []
    for child in sorted(tmp_root.iterdir()):
        try:
            if not (child.is_dir() or child.is_symlink()):
                continue
        except OSError:
            continue
        if not _tmp_entry_owned(child):
            continue
        n = extract_issue_number(child.name)
        if n is not None:
            found.add(n)
    return sorted(found)


# ─── tier results ────────────────────────────────────────────────────────────


@dataclass
class TierResult:
    name: str
    bytes_freed: int = 0
    detail: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    # ── v4 structured dry-run reporting (#911) — surfaced in --json ──
    # Report-only escalation persists NOTHING (sidecar / Telegram / state are
    # all apply-gated, pinned by test_dry_run_escalation_reports_no_sidecar_
    # write), so a dry-run acceptance MUST read these fields, never the
    # sidecar. One attribution row per ACTIVE/unresolved owner from tier (b)'s
    # sizing pass ({task, status, path, bytes}) — dedup-independent: computed
    # even when the alert row is ack/band-suppressed or below the alert floor.
    active_cache_attributions: list[dict] = field(default_factory=list)
    # One row per discovered NON-CANONICAL candidate:
    # {path, issue, bytes, disposition, evidence} with disposition in
    # would-remove | removed | escalated | recency-kept | unverified-kept |
    # durable-content-kept | unresolved-kept | consumer-kept |
    # external-target-kept | failed.
    noncanonical_candidates: list[dict] = field(default_factory=list)
    # Total bytes of every cache dir this tier DISCOVERED (canonical +
    # non-canonical, any disposition); tier (d) reports its
    # expected_freed_size here as the would-free upper bound.
    total_discovered_bytes: int = 0
    # ── tier (e) structured attribution (#1376) — surfaced in --json ──
    # One row per repo in the HOME HF hub cache, populated in report-only AND
    # apply alike (dedup-independent — the #911 dry-run acceptance surface):
    # {repo, repo_type, bytes, revisions, last_accessed_age_days,
    #  reap_candidate_bytes, over_escalate_threshold}.
    hf_repo_attributions: list[dict] = field(default_factory=list)


@dataclass
class GuardResult:
    used_pct_before: float
    used_pct_after: float
    free_gb_before: float
    free_gb_after: float
    threshold_pct: float
    triggered: bool
    apply: bool
    tiers: list[TierResult] = field(default_factory=list)
    still_over_after: bool = False

    @property
    def bytes_freed(self) -> int:
        return sum(t.bytes_freed for t in self.tiers)

    @property
    def total_discovered_bytes(self) -> int:
        """Sum of every tier's discovered-cache footprint (any disposition —
        would-remove, escalated, recency-/unverified-/durable-content-kept;
        #911 acceptance surface)."""
        return sum(t.total_discovered_bytes for t in self.tiers)


# ─── tier (a): uv / pip caches ───────────────────────────────────────────────


def _uv_cache_dir_size() -> int:
    """Best-effort size of the uv cache dir (for before/after reporting)."""
    rc, out, _ = _run(["uv", "cache", "dir"])
    if rc != 0 or not out.strip():
        return 0
    cache_dir = Path(out.strip())
    if not cache_dir.is_dir():
        return 0
    return _du_bytes(cache_dir)


def clean_uv_cache(apply: bool) -> TierResult:
    """Tier (a): ``uv cache prune`` — never ``--force``. A held uv lock makes
    prune skip gracefully (it is non-destructive of in-use entries); a missing
    uv binary or a prune error degrades to a logged skip, never a crash."""
    res = TierResult(name="uv-cache")
    before = _uv_cache_dir_size()
    if not apply:
        # Dry-run: report the cache size as the upper bound that prune could
        # reclaim (prune keeps in-use entries, so this is an over-estimate).
        res.bytes_freed = before
        res.detail.append(f"uv cache ~{_fmt_gb(before)} (prune would reclaim unused entries)")
        return res
    rc, out, err = _run(["uv", "cache", "prune"], timeout=300)
    if rc != 0:
        res.skipped = True
        res.skip_reason = (err or out or "uv cache prune failed").strip()[:200]
        res.detail.append(f"uv cache prune skipped: {res.skip_reason}")
        return res
    after = _uv_cache_dir_size()
    res.bytes_freed = max(0, before - after)
    res.detail.append(f"uv cache prune freed {_fmt_gb(res.bytes_freed)}")
    return res


# ─── tier (b): terminal-issue download caches ────────────────────────────────


def _all_worktree_data_roots() -> list[Path]:
    """Every ``<worktree>/data`` dir under ``.claude/worktrees/`` (across all
    worktrees, used to DISCOVER which issues have worktree data). The big
    backlog lives here — the worktrees tree was 139 GB on 2026-06-26,
    dominated by per-issue download/store data."""
    wt_root = repo_root() / ".claude" / "worktrees"
    if not wt_root.is_dir():
        return []
    return [c / "data" for c in sorted(wt_root.iterdir()) if (c / "data").is_dir()]


def clean_terminal_download_caches(
    apply: bool,
    data_root: Path | None = None,
    *,
    tmp_root: Path | None = None,
    sweep_tmp: bool = True,
) -> TierResult:
    """Tier (b): delete ``hf_dl`` / ``g*_dl`` caches for issues at a terminal
    status (completed / archived / awaiting_promotion) — across BOTH the
    repo-root ``data/`` AND every worktree ``data/`` (the live experiment's
    data often lives in ``.claude/worktrees/issue-<N>*/data/``, #658 evidence;
    139 GB of worktree data on 2026-06-26). Status is resolved READ-ONLY; an
    active or unresolvable issue is skipped (its cache may be in use — #658 is
    mid-analysis writing tensors into its worktree). Reuses
    ``clean_issue_downloads`` so the keep/delete contract (``store/`` +
    ``eval_results/`` always kept, in worktrees too) is identical to the Step-8
    helper.

    NON-CANONICAL caches (#911): with an EXPLICIT ``tmp_root`` (strict opt-in
    — ``main()`` passes ``production_tmp_root()`` on the boot-disk pass;
    library calls with ``tmp_root=None`` never touch any /tmp), discovery is
    widened with ``_discover_tmp_issue_numbers`` so a /tmp-ONLY issue (the
    #823 shape) is visited, and ``tmp_root``/``sweep_tmp`` forward into every
    per-issue cleanup + escalation-sizing call. The per-candidate reap gates
    (recency / nested-durable / positive re-downloadability evidence) live in
    ``clean_issue_downloads``; the terminal/active/unresolved branching here
    is UNCHANGED. Structured outcome rows (``active_cache_attributions`` +
    ``noncanonical_candidates`` + ``total_discovered_bytes``) land on the
    returned TierResult for the ``--json`` dry-run acceptance surface.

    With an explicit ``data_root`` (tests) the search is scoped to that single
    root; in production (``data_root is None``) it spans repo-root + all
    worktree data roots."""
    res = TierResult(name="terminal-download-caches")
    if data_root is not None:
        discover_roots = [data_root]
    else:
        discover_roots = [repo_root() / "data", *_all_worktree_data_roots()]
    issue_numbers = set(_discover_data_issue_numbers(discover_roots))
    if sweep_tmp and tmp_root is not None:
        issue_numbers.update(_discover_tmp_issue_numbers(tmp_root))
    escalation_state = _load_active_escalation_state()
    escalated_any = False
    for issue_n in sorted(issue_numbers):
        status = _resolve_issue_status(issue_n)
        if status not in TERMINAL_CACHE_REAP_STATUSES:
            res.detail.append(
                f"issue {issue_n}: kept (status={status or 'unresolved'} not terminal)"
            )
            # An ACTIVE task's cache is NEVER deleted — but a large one the
            # terminal-gate can't reap is ESCALATED (Telegram + sidecar) so a
            # human can reclaim it AFTER the run (#679). Fail-soft. The
            # structured attribution rows land on `res` via report_to
            # regardless of the alert's floor/ack/band suppression (#911).
            esc = escalate_active_cache(
                issue_n,
                status,
                apply,
                data_root=data_root,
                state=escalation_state,
                tmp_root=tmp_root,
                sweep_tmp=sweep_tmp,
                report_to=res,
            )
            if esc is not None:
                escalated_any = True
                res.detail.extend(esc.detail)
            continue
        # data_root=data_root forwards the test-scoping; None lets
        # clean_issue_downloads resolve repo-root + this issue's worktree(s).
        sub = clean_issue_downloads(
            issue_n, apply=apply, data_root=data_root, tmp_root=tmp_root, sweep_tmp=sweep_tmp
        )
        res.bytes_freed += sub.bytes_freed
        res.total_discovered_bytes += sub.total_discovered_bytes
        for rel, disposition in sub.noncanonical_dispositions.items():
            res.noncanonical_candidates.append(
                {
                    "path": rel,
                    "issue": issue_n,
                    "bytes": sub.sizes_bytes.get(rel, 0),
                    "disposition": disposition,
                    "evidence": sub.noncanonical_evidence.get(rel, ""),
                }
            )
        for name in sub.removed:
            verb = "removed" if apply else "would remove"
            line = (
                f"issue {issue_n} ({status}): {verb} {name} "
                f"[{_fmt_gb(sub.sizes_bytes.get(name, 0))}]"
            )
            evidence = sub.noncanonical_evidence.get(name)
            if evidence:
                line += f" (evidence: {evidence})"
            res.detail.append(line)
        for name, reason in sub.skipped:
            res.detail.append(f"issue {issue_n} ({status}): kept {name} — {reason}")
        for name in sub.failed:
            res.detail.append(f"issue {issue_n}: FAILED to remove {name}")
        for name, tgt in sub.symlink_external_kept:
            res.detail.append(f"issue {issue_n}: external symlink target kept: {name} -> {tgt}")
    if apply and escalated_any:
        _save_active_escalation_state(escalation_state)
    return res


# ─── tier (d): the VM's pod-style /workspace HF hub cache (#911) ─────────────

# The VM carries a pod-CONVENTION stray HF hub cache at /workspace/.cache
# (plain directory on the boot disk — NOT a mounted pod volume; 21 GB on
# 2026-07-03). Age-gated reap: only repos whose repo-level last_accessed is
# older than the cutoff lose their revisions (relatime keeps last_accessed
# >= daily-fresh for actively read repos; the 14 d margin dwarfs the 24 h
# relatime coarseness). datasets/ + xet/ + stray top-level dirs untouched.
DEFAULT_WORKSPACE_HF_CACHE = "/workspace/.cache/huggingface"  # env EPS_VM_WORKSPACE_HF_CACHE
DEFAULT_WORKSPACE_HF_CACHE_MAX_AGE_DAYS = 14.0  # env EPS_VM_WORKSPACE_HF_CACHE_MAX_AGE_DAYS


def workspace_hf_cache_root() -> Path:
    """The watched pod-style HF cache root on the VM
    (env ``EPS_VM_WORKSPACE_HF_CACHE``; blank -> default)."""
    raw = os.environ.get("EPS_VM_WORKSPACE_HF_CACHE", "").strip()
    return Path(raw or DEFAULT_WORKSPACE_HF_CACHE)


def workspace_hf_cache_max_age_days() -> float:
    """Tier-(d) age cutoff in days (env ``EPS_VM_WORKSPACE_HF_CACHE_MAX_AGE_DAYS``;
    invalid/negative -> default)."""
    raw = os.environ.get("EPS_VM_WORKSPACE_HF_CACHE_MAX_AGE_DAYS", "").strip()
    if not raw:
        return DEFAULT_WORKSPACE_HF_CACHE_MAX_AGE_DAYS
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_WORKSPACE_HF_CACHE_MAX_AGE_DAYS
    return val if val >= 0.0 else DEFAULT_WORKSPACE_HF_CACHE_MAX_AGE_DAYS


def _scan_hf_cache(hub: Path):
    """Import + scan seam (monkeypatched by tests): ``scan_cache_dir`` over the
    hub dir. Raises on any import/scan failure — the caller degrades to a
    skipped tier with the reason."""
    from huggingface_hub import scan_cache_dir

    return scan_cache_dir(str(hub))


def clean_vm_workspace_hf_cache(
    apply: bool,
    *,
    max_age_days: float | None = None,
    cache_root: Path | None = None,
    now: float | None = None,
) -> TierResult:
    """Tier (d): age-gated reap of the VM's pod-style HF hub cache (#911).

    Pod guards FIRST (both must clear): skip when ``os.path.ismount(
    '/workspace')`` is True (a real pod volume — mirrors ``orchestrate.env``'s
    plain-dir-vs-mount discriminator; on a pod this cache is the live
    ``HF_HOME``) OR ``ced._running_pod_side()`` is True (defense in depth,
    #803). Then ``scan_cache_dir(root/'hub')``, collect ALL revisions of
    repos whose repo-level ``last_accessed`` is older than
    ``max_age_days``, build one ``delete_revisions`` strategy, report its
    ``expected_freed_size``; ``apply`` executes it + writes one sidecar row
    (``kind='workspace-hf-cache-reaped'``). EVERY failure (import, scan,
    execute) degrades to ``TierResult.skipped`` + reason — the guard never
    crashes on a corrupt cache. ``datasets/`` + ``xet/`` + stray top-level
    dirs are untouched (~220 MB, below the noise floor by design)."""
    res = TierResult(name="workspace-hf-cache")
    try:
        if os.path.ismount("/workspace"):
            res.skipped = True
            res.skip_reason = "/workspace is a real mount (pod volume) — tier (d) refuses"
            return res
        if _running_pod_side():
            res.skipped = True
            res.skip_reason = "pod-side detected (#803) — tier (d) refuses"
            return res
    except OSError as exc:
        res.skipped = True
        res.skip_reason = f"pod-guard probe failed: {exc}"
        return res
    root = cache_root if cache_root is not None else workspace_hf_cache_root()
    hub = root / "hub"
    if not hub.is_dir():
        res.skipped = True
        res.skip_reason = f"no hub cache at {hub}"
        return res
    age_days = max_age_days if max_age_days is not None else workspace_hf_cache_max_age_days()
    cutoff = (time.time() if now is None else now) - age_days * 86400.0
    try:
        info = _scan_hf_cache(hub)
        stale_repos = [r for r in info.repos if r.last_accessed < cutoff]
        hashes = [rev.commit_hash for r in stale_repos for rev in r.revisions]
        if not hashes:
            res.detail.append(f"no hub repos unused >= {age_days:g}d (of {len(list(info.repos))})")
            return res
        strategy = info.delete_revisions(*hashes)
        freed = int(strategy.expected_freed_size)
        repo_ids = sorted(r.repo_id for r in stale_repos)
        res.total_discovered_bytes = freed  # the would-free upper bound
        shown = ", ".join(repo_ids[:5])
        if not apply:
            res.bytes_freed = freed
            res.detail.append(
                f"would delete {len(hashes)} revision(s) of {len(stale_repos)} stale "
                f"repo(s) [{_fmt_gb(freed)}]: {shown}"
            )
            return res
        strategy.execute()
        res.bytes_freed = freed
        res.detail.append(
            f"deleted {len(hashes)} revision(s) of {len(stale_repos)} stale repo(s) "
            f"[{_fmt_gb(freed)}]: {shown}"
        )
        append_disk_guard_event(
            {
                "kind": "workspace-hf-cache-reaped",
                "repos": repo_ids,
                "bytes": freed,
                "max_age_days": age_days,
            },
            apply=apply,
        )
    except Exception as exc:  # deliberate degrade — never crash the guard pass
        res.skipped = True
        res.skip_reason = f"{type(exc).__name__}: {exc}"[:200]
        res.detail.append(f"workspace hf-cache tier skipped: {res.skip_reason}")
    return res


# ─── tier (e): the HOME HF hub cache (~/.cache/huggingface, #1376 + #1377) ───

# The home cache is the DEFAULT-HF_HOME consumer on this VM and the fleet's
# dominant root-disk consumer: every pinned data-repo read / upload commit
# mints one revision that nothing reclaims (12 revisions / 76.2 GB observed at
# the 2026-07-16 episode while the guard freed 0.53 GB, blind to the cache;
# regrown to 88 unref'd revisions / ~20G within 5 days per #1377). Whole-repo
# age-gating alone (tier (d)'s predicate) cannot help — the repo is touched
# daily — so arm 2 trims REVISIONS. Deliberately NOT derived from
# HF_HOME/HF_HUB_CACHE: the tier watches a FIXED filesystem location so the
# cron's coverage is deterministic; env-derived resolution varies per process
# and is the watcher arm's job (plan #1376 §11). #1376 and #1377 landed this
# tier independently; the reconciled tier keeps #1377's incumbent names/knobs
# (function, tier name, age env) + the UNION of both tasks' KEEP protections.
DEFAULT_HOME_HF_CACHE = "~/.cache/huggingface"  # env EPS_VM_HOME_HF_CACHE
DEFAULT_HOME_HF_REVISION_MAX_AGE_DAYS = 7.0  # env EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS
DEFAULT_HOME_HF_REPO_ESCALATE_GB = 40.0  # env EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB
HOME_HF_ATTRIBUTION_TOP_N = 5  # top consumers named in detail lines
_HOME_HF_ESCALATION_BREAKDOWN_TOP_N = 8  # revisions named in the escalation row


def home_hf_cache_root() -> Path:
    """The watched home HF cache root (env ``EPS_VM_HOME_HF_CACHE``; blank ->
    default ``~/.cache/huggingface``; the tier appends ``hub/`` itself)."""
    raw = os.environ.get("EPS_VM_HOME_HF_CACHE", "").strip()
    return Path(raw).expanduser() if raw else Path(DEFAULT_HOME_HF_CACHE).expanduser()


def home_hf_revision_max_age_days() -> float:
    """Tier-(e) age window in days, shared by both arms (repo ``last_accessed``;
    revision ``last_modified`` + exclusive-blob atime). Env
    ``EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS``; blank/invalid/negative -> default."""
    raw = os.environ.get("EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS", "").strip()
    if not raw:
        return DEFAULT_HOME_HF_REVISION_MAX_AGE_DAYS
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_HOME_HF_REVISION_MAX_AGE_DAYS
    return val if val >= 0.0 else DEFAULT_HOME_HF_REVISION_MAX_AGE_DAYS


def home_hf_repo_escalate_bytes() -> int:
    """Single-repo footprint (bytes) above which tier (e) always escalates
    with a per-revision breakdown. Env ``EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB``
    (float GB); invalid/negative -> default."""
    raw = os.environ.get("EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB", "").strip()
    gb = DEFAULT_HOME_HF_REPO_ESCALATE_GB
    if raw:
        try:
            val = float(raw)
        except ValueError:
            val = -1.0
        if val >= 0.0:
            gb = val
    return int(gb * 1e9)


def _hf_ack_sentinel_path(repo_key: str, band_gb: float) -> Path:
    """Per-(repo, band) ack sentinel for the tier-(e) home-hub repo escalation:
    ``touch`` it to silence this repo's escalation until it crosses into a
    bigger band. ``repo_key`` is ``<repo_type>/<repo_id>``; '/' maps to '--'
    so the sentinel is a flat filename (mirrors ``_active_ack_sentinel_path``)."""
    safe = repo_key.replace("/", "--")
    return repo_root() / ".claude" / "cache" / f"disk-guard-ack-hf-{safe}-{band_gb:g}"


def _should_escalate_hf_repo(
    repo_key: str,
    band_gb: float,
    bytes_: int,
    state: dict,
) -> tuple[bool, float]:
    """Decide whether to (re-)escalate a home-hub repo, and the growth % vs the
    last alert. Dedups on (repo, band) under the ``hf:``-NAMESPACED state key
    ``hf:<repo_type>/<repo_id>:<band>`` (no collision with the integer-keyed
    ``<issue>:<band>`` active-cache entries in the same JSON); re-alerts only
    on >25% growth; an ack sentinel for this (repo, band) suppresses entirely."""
    if _hf_ack_sentinel_path(repo_key, band_gb).exists():
        return (False, 0.0)
    key = f"hf:{repo_key}:{band_gb:g}"
    prev = state.get(key)
    if not isinstance(prev, int | float) or prev <= 0:
        return (True, 0.0)  # first alert for this (repo, band)
    growth = (bytes_ - prev) / prev
    return (growth > _ACTIVE_ESCALATION_GROWTH_REALERT, growth * 100.0)


def _home_hf_reap_selection(info, now: float, age_seconds: float):
    """Pure reap selector over a scanned home-hub cache (tier (e), #1376).

    Returns ``(whole_repo_stale, rev_candidates, kept_reason_counts)``:

    * ``whole_repo_stale`` — repos whose repo-level ``last_accessed`` is older
      than the window (ARM 1): the WHOLE repo is reaped, ref'd (incl.
      ``main``) + newest revisions included — the repo-level age gate is the
      sole arm-1 predicate BY DESIGN (this is what covers stale models; the
      arm-2 protections below do not bind on arm 1).
    * ``rev_candidates`` — ``(repo, [revisions])`` pairs within FRESH
      multi-revision repos (ARM 2): a revision is a candidate only when it is
      NOT the repo's newest (deterministic ``(last_modified, commit_hash)``
      tie-break — the newest is ALWAYS kept, #1377; this subsumes the older
      never-empty-a-fresh-repo clamp) AND carries NO ref at all (any ref
      protects — the UNION of #1376's main-ref and #1377's all-refs
      protections; live non-``main`` refs are truncated-commit-hash refs
      minted by pinned reads, and #1377 keeps them) AND its ``last_modified``
      is older than the window AND none of its EXCLUSIVE blobs (blobs
      referenced by no newest/ref'd/fresh surviving revision) has a fresh
      atime — hourly reads of the fresh revision keep SHARED blobs' atimes
      fresh, so a whole-revision atime gate selects 0 candidates on the
      motivating repo (measured, plan #1376 §2); the exclusive-blob guard
      protects exactly the data deletion would destroy while
      ``delete_revisions`` refcounting protects the rest.
    * ``kept_reason_counts`` — e.g. ``{"degenerate-repo-kept": N}`` for repos
      whose per-repo selection raised (None timestamps, malformed fields):
      that repo is skipped entirely (kept, fail toward KEEP) while every
      other repo keeps its selection + attribution.

    atime dependence: the exclusive-blob keep-guard reads
    ``blob_last_accessed`` (atime). ``/`` is mounted ``rw,relatime`` (<=24 h
    coarseness, dwarfed by the 7 d window); a future ``noatime`` remount
    would FREEZE atimes and invert this keep signal toward REAP for
    read-only-but-active revisions — re-verify mount options before trusting
    the guard under a different mount regime.
    """
    whole, revlevel = [], []
    kept_reason_counts: dict[str, int] = {}
    for repo in info.repos:
        try:
            if now - repo.last_accessed > age_seconds:
                whole.append(repo)  # arm 1: covers stale models/datasets wholesale
                continue
            revs = list(repo.revisions)
            if len(revs) <= 1:
                continue  # single revision == newest: always kept, any age
            # Deterministic newest pick (#1377's tie-break): the newest
            # revision is ALWAYS kept, which also guarantees a fresh repo is
            # never wholly emptied (subsumes the older never-empty clamp).
            newest = max(revs, key=lambda r: (r.last_modified, r.commit_hash))
            kept_blobs = {
                f.blob_path
                for rev in revs
                for f in rev.files
                if rev is newest or rev.refs or now - rev.last_modified <= age_seconds
            }
            cands = []
            for rev in revs:
                if rev is newest:  # newest per repo: always kept (arm 2, #1377)
                    continue
                if rev.refs:  # ANY ref protects (union of #1376 + #1377)
                    continue
                if now - rev.last_modified <= age_seconds:  # recently written: kept
                    continue
                excl = [f for f in rev.files if f.blob_path not in kept_blobs]
                newest_excl_atime = max((f.blob_last_accessed for f in excl), default=0.0)
                if excl and now - newest_excl_atime <= age_seconds:
                    continue  # exclusive data recently READ: kept (fail-keep)
                cands.append(rev)
            if cands:
                revlevel.append((repo, cands))
        except Exception:
            # Degenerate repo (None timestamps, malformed fields): keep THAT
            # repo, count it, keep selecting the rest (fail toward KEEP
            # without degrading the whole tier).
            kept_reason_counts["degenerate-repo-kept"] = (
                kept_reason_counts.get("degenerate-repo-kept", 0) + 1
            )
            continue
    return whole, revlevel, kept_reason_counts


def _attribute_home_hf_repos(
    repos: list,
    cand_by_repo: dict[str, list],
    escalate_bytes: int,
    ts: float,
    res: TierResult,
) -> None:
    """Attribution arm of tier (e): one ``hf_repo_attributions`` row per repo
    (populated in report-only AND apply alike — the dry-run acceptance
    surface), top ``HOME_HF_ATTRIBUTION_TOP_N`` also named in detail lines.
    Per-repo try/except: a degenerate repo (None timestamps) loses only ITS
    row — attribution for every other repo survives (#1376 review concern;
    the selector applies the same per-repo fail-keep)."""
    n_degenerate = 0
    for repo in repos:
        try:
            cands = cand_by_repo.get(repo.repo_id, [])
            row = {
                "repo": repo.repo_id,
                "repo_type": repo.repo_type,
                "bytes": int(repo.size_on_disk),
                "revisions": len(list(repo.revisions)),
                "last_accessed_age_days": round((ts - repo.last_accessed) / 86400.0, 2),
                "reap_candidate_bytes": int(sum(rev.size_on_disk for rev in cands)),
                "over_escalate_threshold": repo.size_on_disk > escalate_bytes,
            }
        except Exception:
            n_degenerate += 1
            continue
        res.hf_repo_attributions.append(row)
    if n_degenerate:
        res.detail.append(f"attribution skipped for {n_degenerate} degenerate repo(s)")
    for row in res.hf_repo_attributions[:HOME_HF_ATTRIBUTION_TOP_N]:
        res.detail.append(
            f"{row['repo_type']}/{row['repo']}: {_fmt_gb(row['bytes'])}, "
            f"{row['revisions']} revision(s), last accessed "
            f"{row['last_accessed_age_days'] * 24.0:.1f}h ago (hub/ only)"
        )


def _escalate_home_hf_repos(
    repos: list,
    cand_by_repo: dict[str, list],
    escalate_bytes: int,
    ts: float,
    apply: bool,
    st: dict,
    res: TierResult,
) -> bool:
    """Escalation arm of tier (e): for every repo over the always-escalate
    footprint, a detail line ALWAYS, plus one sidecar row (per-revision
    breakdown + ``reap_cmd``) + one Telegram push deduped per (repo, band)
    with the 25%-growth re-alert + ack-sentinel suppression. State updates
    (``st[key] = bytes``) are apply-gated; persistence of ``st`` itself is
    the caller's. Returns True when any alert fired (state dirty). A
    degenerate repo (None timestamps in the breakdown) loses only ITS
    escalation — the loop continues (per-repo fail-keep, #1376)."""
    escalated_any = False
    for repo in repos:
        try:
            escalated_any = (
                _escalate_one_home_hf_repo(repo, cand_by_repo, escalate_bytes, ts, apply, st, res)
                or escalated_any
            )
        except Exception:
            continue  # degenerate repo: skip its escalation, keep the rest
    return escalated_any


def _escalate_one_home_hf_repo(
    repo,
    cand_by_repo: dict[str, list],
    escalate_bytes: int,
    ts: float,
    apply: bool,
    st: dict,
    res: TierResult,
) -> bool:
    """One repo's escalation decision + emission (see ``_escalate_home_hf_repos``).
    Returns True when the alert fired for this repo."""
    if repo.size_on_disk <= escalate_bytes:
        return False
    repo_key = f"{repo.repo_type}/{repo.repo_id}"
    band_gb = _active_escalation_band_gb(int(repo.size_on_disk))
    n_cands = len(cand_by_repo.get(repo.repo_id, []))
    res.detail.append(
        f"ESCALATION: {repo_key} holds {_fmt_gb(int(repo.size_on_disk))} across "
        f"{len(list(repo.revisions))} revision(s) "
        f"(> {_fmt_gb(escalate_bytes)} always-escalate threshold; "
        f"{n_cands} reapable)"
    )
    do_alert, growth_pct = _should_escalate_hf_repo(repo_key, band_gb, int(repo.size_on_disk), st)
    if not do_alert:
        return False
    breakdown = [
        {
            "commit": rev.commit_hash[:8],
            "bytes": int(rev.size_on_disk),
            "age_days": round((ts - rev.last_modified) / 86400.0, 1),
            "refs": sorted(rev.refs),
        }
        for rev in sorted(repo.revisions, key=lambda v: v.size_on_disk, reverse=True)[
            :_HOME_HF_ESCALATION_BREAKDOWN_TOP_N
        ]
    ]
    append_disk_guard_event(
        {
            "kind": "home-hf-cache-repo-escalation",
            "repo": repo_key,
            "bytes": int(repo.size_on_disk),
            "revisions": len(list(repo.revisions)),
            "band": band_gb,
            "growth_pct": round(growth_pct, 1),
            "revision_breakdown": breakdown,
            "reap_cmd": "uv run python scripts/vm_disk_guard.py --apply",
        },
        apply=apply,
    )
    _telegram_push(
        f"VM disk: home HF hub cache repo {repo_key} holds "
        f"{_fmt_gb(int(repo.size_on_disk))} across {len(list(repo.revisions))} "
        f"revisions ({n_cands} unreferenced+stale). Guard reaps stale unreferenced "
        f"revisions on --apply; ack: touch {_hf_ack_sentinel_path(repo_key, band_gb)}",
        apply,
    )
    if apply:
        st[f"hf:{repo_key}:{band_gb:g}"] = int(repo.size_on_disk)
    return True


def clean_home_hf_stale_revisions(
    apply: bool,
    *,
    max_age_days: float | None = None,
    repo_escalate_gb: float | None = None,
    cache_root: Path | None = None,
    now: float | None = None,
    state: dict | None = None,
) -> TierResult:
    """Tier (e): attribution + escalation + safe reap of the HOME HF hub cache
    (``~/.cache/huggingface/hub``; #1376 + #1377 reconciled into ONE tier —
    #1377's incumbent names/knobs, the UNION of both tasks' KEEP protections,
    plus #1376's attribution/escalation + whole-stale-repo arm).

    On every triggered boot-disk pass (report-only AND apply):

    1. ATTRIBUTES — per-repo size / revision count / ``last_accessed`` age for
       every repo (structured ``hf_repo_attributions`` rows; top
       ``HOME_HF_ATTRIBUTION_TOP_N`` named in detail lines). ``hub/`` only —
       ``datasets/`` / ``xet/`` / ``stored_tokens/`` are out of scope.
    2. ESCALATES — any single repo > ``EPS_VM_HOME_HF_CACHE_REPO_ESCALATE_GB``
       gets a detail line always, plus one sidecar row
       (``kind='home-hf-cache-repo-escalation'`` with a per-revision
       breakdown + ``reap_cmd``) + one Telegram push, deduped per (repo,
       band) with the 25%-growth re-alert and ack-sentinel suppression
       (``_hf_ack_sentinel_path``). Persistence is apply-gated throughout
       (``append_disk_guard_event(..., apply=apply)`` /
       ``_telegram_push(msg, apply)`` / the state write below) — report-only
       persists NOTHING; dry-run acceptance reads the ``--json`` fields.
    3. REAPS (apply only) — the ``_home_hf_reap_selection`` candidate set:
       arm 1 whole stale repos (repo-level ``last_accessed`` > window; ref'd
       + newest revisions included by design) + arm 2 unref'd non-newest
       stale cold revisions of fresh multi-revision repos (the newest + every
       ref'd revision per repo is ALWAYS kept). Deletion goes
       EXCLUSIVELY through ``HFCacheInfo.delete_revisions().execute()``
       (blob-refcount safe — never an rmtree of blobs/snapshots).
       ``bytes_freed`` books the strategy's ``expected_freed_size`` (blobs
       shared with surviving revisions are excluded by the hub's
       refcounting) and is set only AFTER a successful ``execute()`` on the
       apply path (an execute-raise leaves 0); the realized read is the
       guard-level free-GB delta. One sidecar row
       (``kind='home-hf-revisions-trimmed'``) records the apply-path reap.
       A later FileNotFoundError on a trimmed snapshot path means tier (e)
       trimmed it — re-download on demand (the data lives on HF), not data
       loss.

    atime dependence: arm 1 (repo ``last_accessed``) and the arm-2
    exclusive-blob keep-guard read atimes — meaningful under the VM's
    ``rw,relatime`` mount; see ``_home_hf_reap_selection``'s docstring for
    the ``noatime`` caveat.

    Safety posture: EVERY failure degrades toward KEEP — the pod guard,
    missing hub dir, the tier-(d) double-cover guard, a scan/execute
    exception all land in ``TierResult.skipped`` + reason (per-repo
    degeneracy is narrower: that repo alone is kept, see the selector). A
    concurrent race with the watcher's independent 14 d evictor
    (``EPM_VM_DISK_HF_TTL_DAYS``) surfaces as an ``execute()`` exception ->
    skipped, nothing else deleted (``delete_revisions`` is idempotent).

    ``state`` is the escalation dedup map: tests pass ``{}`` (updated in
    place, apply-gated, never persisted here); ``state=None`` (the
    production ``run_guard`` path, reached only via the ``main()``-only
    opt-in) loads + saves the shared ``_load_active_escalation_state`` JSON
    under ``hf:``-namespaced keys."""
    res = TierResult(name="home-hf-revisions")
    try:
        if _running_pod_side():
            res.skipped = True
            res.skip_reason = "pod-side detected — tier (e) refuses"
            return res
    except OSError as exc:
        res.skipped = True
        res.skip_reason = f"pod-guard probe failed: {exc}"
        return res
    root = cache_root if cache_root is not None else home_hf_cache_root()
    hub = root / "hub"
    if not hub.is_dir():
        res.skipped = True
        res.skip_reason = f"no hub cache at {hub}"
        return res
    try:
        if hub.resolve() == (workspace_hf_cache_root() / "hub").resolve():
            res.skipped = True
            res.skip_reason = (
                "home cache root == workspace cache root — tier (d) owns it (no double-reap)"
            )
            return res
    except OSError as exc:
        res.skipped = True
        res.skip_reason = f"root-resolution probe failed: {exc}"  # fail toward keep
        return res
    age_days = max_age_days if max_age_days is not None else home_hf_revision_max_age_days()
    escalate_bytes = (
        int(repo_escalate_gb * 1e9)
        if repo_escalate_gb is not None
        else home_hf_repo_escalate_bytes()
    )
    ts = time.time() if now is None else now
    try:
        info = _scan_hf_cache(hub)
        warnings = list(getattr(info, "warnings", None) or [])
        if warnings:
            res.detail.append(f"scan warnings: {len(warnings)} (repos kept)")
        repos = sorted(info.repos, key=lambda r: r.size_on_disk, reverse=True)
        whole, revlevel, kept_counts = _home_hf_reap_selection(info, ts, age_days * 86400.0)
        for reason, count in sorted(kept_counts.items()):
            res.detail.append(f"{reason}: {count} repo(s)")
        cand_by_repo: dict[str, list] = {r.repo_id: list(r.revisions) for r in whole}
        cand_by_repo.update({r.repo_id: list(cands) for r, cands in revlevel})
        # 1) ATTRIBUTION — always, before any reap decision (hub/ only).
        _attribute_home_hf_repos(repos, cand_by_repo, escalate_bytes, ts, res)
        # 2) ESCALATION — per repo over the always-escalate footprint.
        manage_state = state is None
        st = _load_active_escalation_state() if manage_state else state
        escalated_any = _escalate_home_hf_repos(
            repos, cand_by_repo, escalate_bytes, ts, apply, st, res
        )
        if apply and manage_state and escalated_any:
            _save_active_escalation_state(st)
        # 3) REAP — arm 1 (whole stale repos) + arm 2 (stale unreferenced revs).
        arm1_hashes = [rev.commit_hash for r in whole for rev in r.revisions]
        arm2_hashes = [rev.commit_hash for _, cands in revlevel for rev in cands]
        hashes = arm1_hashes + arm2_hashes
        n_repos = len(repos)
        n_revs = sum(len(list(r.revisions)) for r in repos)
        if not hashes:
            res.detail.append(
                f"no unref'd revision older than {age_days:g}d and no wholly-stale repo "
                f"(of {n_revs} revision(s) in {n_repos} repo(s))"
            )
            return res
        strategy = info.delete_revisions(*hashes)
        freed = int(strategy.expected_freed_size)
        res.total_discovered_bytes = freed  # the would-free upper bound
        # Per-repo counts for detail lines + the sidecar row (#1377's
        # ``repos`` dict shape: {repo_id: n revisions deleted}).
        trimmed_by_repo: dict[str, int] = {r.repo_id: len(list(r.revisions)) for r in whole}
        for r, cands in revlevel:
            trimmed_by_repo[r.repo_id] = trimmed_by_repo.get(r.repo_id, 0) + len(cands)
        for r, cands in revlevel:
            approx = sum(rev.size_on_disk for rev in cands)
            res.detail.append(f"{r.repo_id}: {len(cands)} stale revision(s), ~{_fmt_gb(approx)}")
        for r in whole:
            # Name every wholesale arm-1 reap explicitly — ref'd + newest
            # revisions ARE included, so a later "why did my model
            # re-download" trace lands here / on the sidecar row.
            res.detail.append(
                f"arm 1 wholesale: {r.repo_type}/{r.repo_id} — repo-level last_accessed "
                f"{(ts - r.last_accessed) / 86400.0:.1f}d > {age_days:g}d window; "
                f"ALL {len(list(r.revisions))} revision(s) reaped, ref'd + newest included"
            )
        verb = "trimmed" if apply else "would trim"
        res.detail.append(
            f"{verb} {len(hashes)} revision(s) across {len(trimmed_by_repo)} repo(s) "
            f"[{_fmt_gb(freed)} expected, blob-refcount; realized = guard free-GB delta] "
            f"(arm1 whole-repo: {len(whole)} repo(s) / {len(arm1_hashes)} rev(s); "
            f"arm2 revision-level: {len(arm2_hashes)} rev(s) across {len(revlevel)} repo(s))"
        )
        if not apply:
            res.bytes_freed = freed  # dry-run: report, execute NOTHING
            return res
        strategy.execute()
        res.bytes_freed = freed  # set only AFTER execute() — an execute-raise leaves 0
        append_disk_guard_event(
            {
                "kind": "home-hf-revisions-trimmed",
                "repos": dict(sorted(trimmed_by_repo.items())),
                "n_revisions": len(hashes),
                "bytes": freed,
                "max_age_days": age_days,
                "arms": {"whole_repo": len(arm1_hashes), "revision_level": len(arm2_hashes)},
            },
            apply=apply,
        )
    except Exception as exc:  # deliberate degrade — never crash the guard pass
        res.skipped = True
        res.skip_reason = f"{type(exc).__name__}: {exc}"[:200]
        res.detail.append(f"home hf-revisions tier skipped: {res.skip_reason}")
    return res


# ─── tier (c): stale logs ────────────────────────────────────────────────────


def _stale_log_files(roots: list[Path], max_age_seconds: float, now: float) -> list[Path]:
    """``*.log`` files under ``roots`` older than ``max_age_seconds``.

    For ``logs/`` the search is recursive (``rglob``); for a flat dir like
    ``/tmp`` only the top level is scanned (``glob``) — a recursive /tmp walk
    is both slow and risks unrelated trees. The distinction is encoded by the
    caller passing the right roots. A stat error on one file skips it."""
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        # logs/ is recursive; /tmp is top-level only (see docstring).
        pattern_iter = root.rglob("*.log") if root.name == "logs" else root.glob("*.log")
        for p in pattern_iter:
            try:
                if not p.is_file() or p.is_symlink():
                    continue
                age = now - p.stat().st_mtime
            except OSError:
                continue
            if age > max_age_seconds:
                out.append(p)
    return out


def clean_stale_logs(
    apply: bool,
    max_age_days: float,
    now: float | None = None,
    extra_roots: list[Path] | None = None,
) -> TierResult:
    """Tier (c): delete ``logs/**/*.log`` + ``/tmp/*.log`` older than
    ``max_age_days``. ``extra_roots`` overrides the default root set (tests
    point it at a temp filesystem)."""
    res = TierResult(name="stale-logs")
    now = time.time() if now is None else now
    max_age_seconds = max_age_days * 86400.0
    roots = extra_roots if extra_roots is not None else [repo_root() / "logs", Path("/tmp")]
    for f in _stale_log_files(roots, max_age_seconds, now):
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        if not apply:
            res.bytes_freed += size
            res.detail.append(f"would remove {f} [{_fmt_gb(size)}]")
            continue
        try:
            f.unlink()
        except OSError as exc:
            res.detail.append(f"FAILED to remove {f}: {exc}")
            continue
        res.bytes_freed += size
        res.detail.append(f"removed {f} [{_fmt_gb(size)}]")
    return res


# ─── orchestration ───────────────────────────────────────────────────────────


def run_guard(
    apply: bool,
    *,
    threshold: float | None = None,
    log_max_age: float | None = None,
    data_root: Path | None = None,
    disk_path: str = "/",
    reclaim_tiers: bool = True,
    now: float | None = None,
    tmp_root: Path | None = None,
) -> GuardResult:
    """Read disk usage, and if over threshold run the cleanup tiers.

    Pure-ish orchestration: all side effects are gated on ``apply`` inside the
    tier helpers. When usage is under the threshold the tiers are NOT run and
    ``triggered`` is False (a no-op pass).

    ``reclaim_tiers`` (default True for the boot-disk ``/`` watch) gates the
    ``/``-rooted reclaim arms — tier (a) ``uv cache prune`` and tier (c) the
    stale-``logs/**/*.log`` + ``/tmp/*.log`` sweep — which operate on boot-disk
    caches and MUST NOT run when the guard is watching the data disk
    (``disk_path="/mnt/eps-data"``). The DATA-DISK pass passes
    ``reclaim_tiers=False`` so only tier (b) runs there — and tier (b) is the
    ONE data-disk-appropriate arm: it reaps a TERMINAL issue's re-downloadable
    ``hf_dl``/``g*_dl`` cache on EITHER disk and ESCALATES (never deletes) an
    ACTIVE task's cache. So the data-disk pass is escalate-only + reap-terminal,
    never the `/`-rooted uv/log reclaims (#681 plan §4 Phase 4, §11).

    ``tmp_root`` (#911) is forwarded VERBATIM to tier (b) — ``run_guard``
    itself NEVER calls ``production_tmp_root()`` (the /tmp opt-in lives ONLY
    in the two CLI ``main()`` bodies; a source-scan test pins the invariant):
    the existing suite calls ``run_guard(apply=True, data_root=temp)`` as a
    LIBRARY under constant-terminal status monkeypatches, and a run_guard-side
    production fallback would sweep the real /tmp during pytest. Tiers (d)
    AND (e) (the /workspace and HOME hub-cache arms) ride the SAME production
    opt-in — they run only when ``reclaim_tiers`` AND an explicit ``tmp_root``
    are set, so every library call stays hermetic by construction (neither
    the data-disk pass nor a pytest library call can ever scan or reap the
    real ``~/.cache/huggingface``)."""
    thr = threshold if threshold is not None else threshold_pct()
    age = log_max_age if log_max_age is not None else log_max_age_days()
    used_before = disk_used_pct(disk_path)
    free_before = disk_free_gb(disk_path)

    res = GuardResult(
        used_pct_before=used_before,
        used_pct_after=used_before,
        free_gb_before=free_before,
        free_gb_after=free_before,
        threshold_pct=thr,
        triggered=over_threshold(used_before, thr),
        apply=apply,
    )
    if not res.triggered:
        return res

    if reclaim_tiers:
        res.tiers.append(clean_uv_cache(apply))
    res.tiers.append(clean_terminal_download_caches(apply, data_root=data_root, tmp_root=tmp_root))
    if reclaim_tiers and tmp_root is not None:
        # Tiers (d) + (e) ride the production opt-in (an explicit tmp_root) so
        # library callers can never touch the real /workspace OR home HF
        # caches (#911, #1376, #1377).
        res.tiers.append(clean_vm_workspace_hf_cache(apply, now=now))
        res.tiers.append(clean_home_hf_stale_revisions(apply, now=now))  # tier (e), #1376+#1377
    if reclaim_tiers:
        res.tiers.append(clean_stale_logs(apply, age, now=now))

    res.used_pct_after = disk_used_pct(disk_path)
    res.free_gb_after = disk_free_gb(disk_path)
    res.still_over_after = over_threshold(res.used_pct_after, thr)
    return res


# ─── shell helpers ───────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -1, "", f"command not found: {cmd[0]}"
    except OSError as e:
        return -1, "", str(e)


def _du_bytes(path: Path) -> int:
    rc, out, _ = _run(["du", "-sx", "--block-size=1", str(path)], timeout=120)
    if rc != 0 or not out.strip():
        return 0
    with contextlib.suppress(ValueError, IndexError):
        return int(out.split()[0])
    return 0


def _fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f}G"


def _telegram_push(msg: str, apply: bool) -> bool:
    """Fail-soft phone push when the disk stays over threshold after cleanup.

    Mirrors autonomous_session_watch._telegram_push: a missing script or a
    failed call is logged loudly but NEVER raises (the push is observability).
    Skipped entirely in report-only mode."""
    override = os.environ.get("EPM_TELEGRAM_PUSH_SCRIPT", "").strip()
    script = Path(override) if override else _TELEGRAM_PUSH_SCRIPT_DEFAULT
    if not apply:
        # stderr, not stdout — keeps the --json stdout exactly one JSON object.
        print(f"  [report-only] would telegram-push: {msg[:120]}", file=sys.stderr)
        return False
    if not script.is_file():
        print(f"  WARNING: telegram push script missing at {script}; push dropped", file=sys.stderr)
        return False
    try:
        r = subprocess.run(
            ["bash", str(script), msg],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "NOTIF_CAT": "research"},
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  WARNING: telegram push failed: {e}", file=sys.stderr)
        return False
    if r.returncode != 0:
        print(
            f"  WARNING: telegram push failed: {(r.stderr or r.stdout).strip()[:200]}",
            file=sys.stderr,
        )
        return False
    return True


def _result_json(res: GuardResult) -> dict:
    """The JSON-serializable summary for one GuardResult (one watched disk)."""
    return {
        "apply": res.apply,
        "threshold_pct": res.threshold_pct,
        "used_pct_before": round(res.used_pct_before, 2),
        "used_pct_after": round(res.used_pct_after, 2),
        "free_gb_before": round(res.free_gb_before, 2),
        "free_gb_after": round(res.free_gb_after, 2),
        "triggered": res.triggered,
        "bytes_freed": res.bytes_freed,
        "still_over_after": res.still_over_after,
        # v4 (#911): the discovered-cache footprint across all tiers, any
        # disposition — the dry-run acceptance surface (report-only persists
        # no sidecar rows; read THESE fields, never the sidecar).
        "total_discovered_bytes": res.total_discovered_bytes,
        "tiers": [
            {
                "name": t.name,
                "bytes_freed": t.bytes_freed,
                "skipped": t.skipped,
                "skip_reason": t.skip_reason,
                "detail": t.detail,
                "active_cache_attributions": t.active_cache_attributions,
                "noncanonical_candidates": t.noncanonical_candidates,
                "total_discovered_bytes": t.total_discovered_bytes,
                "hf_repo_attributions": t.hf_repo_attributions,
            }
            for t in res.tiers
        ],
    }


def _print_report(res: GuardResult, disk_label: str = "/") -> None:
    verb = "apply" if res.apply else "report-only"
    print(
        f"vm_disk_guard ({verb}): {disk_label} at {res.used_pct_before:.1f}% used "
        f"({res.free_gb_before:.1f}G free), threshold {res.threshold_pct:.0f}%"
    )
    if not res.triggered:
        print("  under threshold — no cleanup needed")
        return
    for tier in res.tiers:
        head = f"  [{tier.name}] freed {_fmt_gb(tier.bytes_freed)}"
        if tier.skipped:
            head += f" (skipped: {tier.skip_reason})"
        print(head)
        for line in tier.detail:
            print(f"      {line}")
    print(
        f"  total freed {_fmt_gb(res.bytes_freed)} | "
        f"{disk_label} now {res.used_pct_after:.1f}% used ({res.free_gb_after:.1f}G free)"
    )
    if res.still_over_after:
        print(
            f"  !! WARNING: {disk_label} STILL at {res.used_pct_after:.1f}% used after cleanup "
            f"(threshold {res.threshold_pct:.0f}%) — manual triage needed",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "VM root-disk guard: tiered safe cleanup when / crosses a "
            "usage threshold. Report-only by default; --apply to act."
        )
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually clean (default: report what would be freed).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Usage %% of / above which cleanup runs (default {DEFAULT_THRESHOLD_PCT}; "
        "env EPS_VM_DISK_THRESHOLD).",
    )
    ap.add_argument(
        "--log-max-age-days",
        type=float,
        default=None,
        help=f"Logs older than this many days are reclaimed (default {DEFAULT_LOG_MAX_AGE_DAYS}; "
        "env EPS_VM_DISK_LOG_MAX_AGE_DAYS).",
    )
    ap.add_argument(
        "--data-disk-path",
        type=str,
        default=None,
        help="Dedicated data-disk mount to ALSO watch escalate-only (the relocated "
        f".claude/worktrees/ tree; default {DEFAULT_DATA_DISK_PATH}; env EPS_VM_DATA_DISK_PATH). "
        "Watched only when the mount exists; the /-rooted reclaim tiers never run there.",
    )
    ap.add_argument(
        "--no-data-disk",
        action="store_true",
        help="Skip the data-disk pass entirely (watch only /).",
    )
    ap.add_argument("--json", action="store_true", help="Emit a JSON summary.")
    args = ap.parse_args(argv)

    # Boot disk (/) — the full tiered cleanup. The /tmp + /workspace-cache
    # opt-in lives HERE (and in clean_experiment_downloads.main()) ONLY: the
    # CLI passes production_tmp_root(); library callers stay hermetic (#911).
    res = run_guard(
        args.apply,
        threshold=args.threshold,
        log_max_age=args.log_max_age_days,
        tmp_root=production_tmp_root(),
    )

    # Data disk (/mnt/eps-data) — a SECOND, ESCALATE-ONLY pass: reclaim_tiers=False
    # so the /-rooted uv/log reclaims never run there, only tier (b)
    # (terminal-cache reap + active-cache escalation). Watched only when the mount
    # is actually LIVE (a missing data disk before the #681 cutover, OR an
    # existing-but-unmounted /mnt/eps-data — a plain dir left by Phase-1's
    # `mkdir -p` or a `nofail` boot that failed to mount — must be a clean no-op,
    # NEVER misread /'s statvfs as the data disk's, #681 round-2 Major). The
    # is_dir() check is insufficient: a plain directory passes it. Require a real
    # mount (_is_mounted, st_dev != parent).
    dd_path = args.data_disk_path or data_disk_path()
    data_res: GuardResult | None = None
    if not args.no_data_disk and _is_mounted(dd_path):
        data_res = run_guard(
            args.apply,
            threshold=args.threshold,
            data_root=None,
            disk_path=dd_path,
            reclaim_tiers=False,
            # The /tmp tree lives on `/` — the data-disk pass never sweeps it
            # (no double-sweep per run; #911 test pins this).
            tmp_root=None,
        )

    if args.json:
        payload = _result_json(res)
        if data_res is not None:
            payload["data_disk"] = {"path": dd_path, **_result_json(data_res)}
        print(json.dumps(payload))
    else:
        _print_report(res, disk_label="/")
        if data_res is not None:
            _print_report(data_res, disk_label=dd_path)

    if res.still_over_after:
        _telegram_push(
            f"VM disk guard: / still {res.used_pct_after:.0f}% full after cleanup "
            f"(freed {_fmt_gb(res.bytes_freed)}); manual triage needed",
            res.apply,
        )
    if data_res is not None and data_res.still_over_after:
        _telegram_push(
            f"VM disk guard: data disk {dd_path} still {data_res.used_pct_after:.0f}% full "
            f"after escalate-only pass; manual triage needed (reclaim a TERMINAL issue's "
            f"cache or raise its setquota -P cap — never delete active data)",
            args.apply,
        )

    # Exit 2 when EITHER disk is still over threshold after cleanup (signals the
    # cron wrapper to keep the alarm channel hot); 0 otherwise.
    still_over = res.still_over_after or (data_res is not None and data_res.still_over_after)
    return 2 if still_over else 0


if __name__ == "__main__":
    sys.exit(main())
