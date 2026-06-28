#!/usr/bin/env python3
"""VM root-disk guard — tiered safe cleanup when ``/`` crosses a threshold.

The VM root disk fills because each experiment downloads its source data into
``data/issue_<N>/hf_dl/`` + ``g*_dl/`` caches that nothing reclaims (incident
2026-06-25: ``/`` hit 100% full, one finished experiment held 97 GB), plus the
``uv`` package cache and accumulating logs. This guard reads ``df`` for ``/``
and, when usage exceeds a threshold (default 85%, env ``EPS_VM_DISK_THRESHOLD``),
runs three TIERS of strictly-safe cleanup, reporting bytes freed per tier:

  (a) ``uv cache prune`` (skipped gracefully if the uv lock is held — never
      ``--force``).
  (b) ``data/issue_*/hf_dl`` + ``g*_dl`` download caches for issues whose task
      status is TERMINAL (``completed`` / ``archived`` / ``awaiting_promotion``).
      Status is resolved READ-ONLY via the task workflow — task state is NEVER
      mutated. An issue at any ACTIVE status (its caches may be in use) is
      skipped.
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
    append_disk_guard_event,
    clean_issue_downloads,
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
) -> TierResult | None:
    """ESCALATE (never delete) an ACTIVE task's re-downloadable cache that the
    terminal-status gate cannot reap.

    Sizes the cache via a dry-run of ``clean_issue_downloads`` (no deletion),
    and — when the footprint is band-worthy and not already alerted (dedup on
    (task, band) + >25%-growth re-alert + ack-sentinel suppression) — emits a
    Telegram push AND a shared-sidecar row naming the largest cache path, the
    footprint, and the SAFE reclaim command. NEVER calls rmtree. Returns a
    TierResult describing the escalation, or None when nothing was escalated.
    ``state`` (the dedup map) is read+updated in place by the caller."""
    sub = clean_issue_downloads(issue_n, apply=False, data_root=data_root)
    # Size from EVERY discovered cache dir, NOT bytes_freed: a large active
    # hf_dl/.../store/ correctly KEPT by the nested-store parity guard lands in
    # `skipped` (contributing 0 to bytes_freed), so bytes_freed would suppress
    # the escalation for the exact large-unmirrored-active-cache shape #679
    # targets (BLOCKER #1). total_discovered_bytes counts removed + skipped.
    bytes_ = sub.total_discovered_bytes
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


# ─── tier results ────────────────────────────────────────────────────────────


@dataclass
class TierResult:
    name: str
    bytes_freed: int = 0
    detail: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


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


def clean_terminal_download_caches(apply: bool, data_root: Path | None = None) -> TierResult:
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

    With an explicit ``data_root`` (tests) the search is scoped to that single
    root; in production (``data_root is None``) it spans repo-root + all
    worktree data roots."""
    res = TierResult(name="terminal-download-caches")
    if data_root is not None:
        discover_roots = [data_root]
    else:
        discover_roots = [repo_root() / "data", *_all_worktree_data_roots()]
    escalation_state = _load_active_escalation_state()
    escalated_any = False
    for issue_n in _discover_data_issue_numbers(discover_roots):
        status = _resolve_issue_status(issue_n)
        if status not in TERMINAL_CACHE_REAP_STATUSES:
            res.detail.append(
                f"issue {issue_n}: kept (status={status or 'unresolved'} not terminal)"
            )
            # An ACTIVE task's cache is NEVER deleted — but a large one the
            # terminal-gate can't reap is ESCALATED (Telegram + sidecar) so a
            # human can reclaim it AFTER the run (#679). Fail-soft.
            esc = escalate_active_cache(
                issue_n, status, apply, data_root=data_root, state=escalation_state
            )
            if esc is not None:
                escalated_any = True
                res.detail.extend(esc.detail)
            continue
        # data_root=data_root forwards the test-scoping; None lets
        # clean_issue_downloads resolve repo-root + this issue's worktree(s).
        sub = clean_issue_downloads(issue_n, apply=apply, data_root=data_root)
        res.bytes_freed += sub.bytes_freed
        for name in sub.removed:
            verb = "removed" if apply else "would remove"
            res.detail.append(
                f"issue {issue_n} ({status}): {verb} {name} "
                f"[{_fmt_gb(sub.sizes_bytes.get(name, 0))}]"
            )
        for name in sub.failed:
            res.detail.append(f"issue {issue_n}: FAILED to remove {name}")
    if apply and escalated_any:
        _save_active_escalation_state(escalation_state)
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
    never the `/`-rooted uv/log reclaims (#681 plan §4 Phase 4, §11)."""
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
    res.tiers.append(clean_terminal_download_caches(apply, data_root=data_root))
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
        "tiers": [
            {
                "name": t.name,
                "bytes_freed": t.bytes_freed,
                "skipped": t.skipped,
                "skip_reason": t.skip_reason,
                "detail": t.detail,
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

    # Boot disk (/) — the full tiered cleanup, unchanged.
    res = run_guard(args.apply, threshold=args.threshold, log_max_age=args.log_max_age_days)

    # Data disk (/mnt/eps-data) — a SECOND, ESCALATE-ONLY pass: reclaim_tiers=False
    # so the /-rooted uv/log reclaims never run there, only tier (b)
    # (terminal-cache reap + active-cache escalation). Watched only when the mount
    # actually exists (a missing data disk before the #681 cutover, or a failed
    # mount, must be a clean no-op — the boot-disk pass is unaffected).
    dd_path = args.data_disk_path or data_disk_path()
    data_res: GuardResult | None = None
    if not args.no_data_disk and Path(dd_path).is_dir():
        data_res = run_guard(
            args.apply,
            threshold=args.threshold,
            data_root=None,
            disk_path=dd_path,
            reclaim_tiers=False,
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
