"""Pipeline progress + ETA estimator for in-flight tasks (task #587).

Single source of truth for ALL progress/ETA math. Two consumers:

1. The Happy session title suffix (``scripts/session_progress_report.py``
   calls :func:`load_stats_readonly` + :func:`estimate_task_progress` +
   :func:`format_title_suffix`).
2. The EPS dashboard (``dashboard/lib/progress.ts`` reads the snapshot
   file this module writes and applies the ONE pinned interpolation
   formula mirrored from :func:`interpolate`; the shared test-vector
   fixture ``tests/fixtures/task_progress_vectors.json`` pins both).

Data flow (single-writer contract)
----------------------------------

The 5-minute summarize cron (``scripts/cron_session_summarize.sh``) is
the ONLY writer: it calls ``scripts/task_progress.py snapshot`` which
runs :func:`write_snapshot` → ``~/.eps-autonomous/task_progress.json``
(atomic temp+rename). Stage statistics are rebuilt at most once per
``STATS_TTL_S`` (24 h) inside the snapshot writer; every other tick
reuses the prior stats section. :func:`load_stats_readonly` NEVER
rebuilds — a dead cron degrades the title suffix to ``None`` instead of
turning every title tick into a 33 MB events.jsonl scan.

Estimand + censoring direction (IMPORTANT, read before trusting numbers)
------------------------------------------------------------------------

Stage-duration samples are *clean forward spans*: an interval that
enters machine stage S (``epm:status-changed`` with ``to == S``) and
exits FORWARD (next transition maps to a later canonical stage or
``awaiting_promotion``/``completed``). Spans exiting to ``blocked``, a
backward stage, or ``archived`` are EXCLUDED — they measure stuckness /
abandonment, not stage cost — and re-entries restart the interval. This
makes every statistic optimistic relative to lived history: the
displayed estimand is a "typical clean forward pass", NEVER a
guarantee. The ``overdue`` state (band suppressed once the current
stage's elapsed time exceeds its p75) is what covers the non-clean
passes the stats deliberately exclude.

Two further honesty caveats baked into the surfaces:

- Summing per-stage quantiles is a heuristic, not the quantile of the
  sum; the band is presented as "typical range (clean pass)".
- The GPU-hours refinement of the running stage is SOFT (measured
  median error ~2x across 29 historical token-carrying tasks); it is
  basis-tagged (``eta_basis``) and rendered distinguishably ("≈"
  prefix) by both surfaces.

Phone-title staleness bound: the title is push-only, so a dead session
freezes its last suffix exactly as it already freezes its step text.
Bounds: the zombie-wrapper watcher auto-stops dead sessions (~2 h),
and the ``overdue`` rule removes the hour band in the alive-session /
stalled-task case. No per-title staleness guard is possible by
construction.

Read-only contract: NOTHING in this module writes under ``tasks/``.
The only write is the snapshot file under ``~/.eps-autonomous/``.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.task_workflow import (
    CODE_KINDS,
    _iter_jsonl,
    find_task_path,
    get_task,
    list_events,
    tasks_dir,
)

# ── Stage model ─────────────────────────────────────────────────────────────

MACHINE_STAGES = [
    "planning",
    "plan_pending",
    "approved",
    "running",
    "verifying",
    "interpreting",
    "reviewing",
]
_STAGE_INDEX = {s: i for i, s in enumerate(MACHINE_STAGES)}

# Optional post-pipeline stage: a same-issue follow-up round executing while
# the task is HELD at status `followups_running`. It is deliberately NOT part
# of the 7-stage floor/total model (most tasks never enter it, so its median
# must not distort every task's floors) — it renders as its own 0→1 track.
# Stats cells ARE built for it (clean spans exit forward to
# awaiting_promotion/completed) and every bucket's pct_floor_by_stage carries
# ``followups_running: 0.0`` so the dashboard's floor-clamp renders work.
FOLLOWUP_STAGE = "followups_running"

# Statuses whose Happy-title gets the suffix (machine-active; plan_pending is
# a human wait — the dashboard shows the parked bar, the title stays clean).
ACTIVE_TITLE_STATUSES = frozenset(
    {"planning", "approved", "running", "verifying", "interpreting", "reviewing"}
)

# Forward terminal exits that close a clean stage span.
FORWARD_EXITS = frozenset({"awaiting_promotion", "completed"})

# Defensive normalization for any surprise legacy status value. Legacy
# statuses were verified to occur ONLY in legacy `state_changed` rows (older
# schema, never scanned); this map exists so a surprise value degrades to the
# right canonical stage instead of being mis-skipped.
LEGACY_STAGE_MAP = {
    "awaiting_approval": "plan_pending",
    "queued": "approved",
    "implementing": "running",
    "code_reviewing": "running",
    "testing": "running",
    "uploading": "verifying",  # defensive only — zero observed occurrences
    "under_review": "reviewing",  # defensive only — zero observed occurrences
    "clean_result_drafting": "reviewing",
}

# Statuses we recognize in `to` fields. Anything else (null, unknown strings)
# is skipped as noise (98 null→null rows observed historically).
_KNOWN_STATUSES = (
    set(MACHINE_STAGES) | FORWARD_EXITS | {"proposed", "followups_running", "blocked", "archived"}
)

# ── Tunables ────────────────────────────────────────────────────────────────

SNAPSHOT_PATH = Path.home() / ".eps-autonomous" / "task_progress.json"
STATS_TTL_S = 24 * 3600
# load_stats_readonly tolerates up to 2x the TTL before declaring the stats
# stale (one missed daily rebuild must not blank every title suffix).
STATS_READ_MAX_AGE_S = 2 * STATS_TTL_S
WINDOW_K, MIN_N = 60, 10
EPS_H = 0.01  # epsilon floor for zero-median stages (36 s)
FRAC_CAP = 0.95  # within-stage frac cap — keeps pct below the next floor

# §7 kill criterion — ETA hour band (measured at implementation, 2026-06-11):
# the one-time calibration backtest (`scripts/task_progress.py backtest`)
# replayed 625 historical stage entries and the [p25, p75] quantile-sum band
# covered only 0.368 of realized clean-forward remaining times (0.404 with
# the §7 denominator guard applied; misses balanced 150 low / 164 high —
# within-task stage-duration dependence narrows quantile-sum bands). The
# pinned keep threshold is 0.50, so the hour band ships DISABLED: position
# bar + state labels (overdue / blocked / waiting-on-you) render; the
# countdown chip does not. The ratio prong passed (experiment
# 3.07/3.32/2.32, code 4.68/4.87/2.72 — all < 8). The full band machinery +
# shared test vectors stay intact (tested with an explicit override) so the
# chip can be re-enabled when calibration improves (e.g. the WandB-progress
# mid-run refinement follow-up). `dashboard/lib/progress.ts` mirrors this
# switch — flip BOTH together.
ETA_BAND_ENABLED = False

TERMINAL_SCAN_STATUSES = ("completed", "archived", "awaiting_promotion")
# CODE_KINDS is imported from task_workflow (the canonical single source) so it
# can never drift from the lifecycle enum (incident #672).

GPU_INTENT_COUNTS = {"eval": 1, "lora-7b": 1, "ft-7b": 4, "inf-70b": 8, "ft-70b": 8, "debug": 1}

_GPU_HOURS_RE = re.compile(r"gpu_hours_total=([0-9.]+)")
# Anchored to a GPU-type token so a prose "2x consideration" never matches.
_GPU_COUNT_RE = re.compile(r"(\d+)\s*[×x]\s*(?=H100|H200|A100|B200|GPU)")  # noqa: RUF001
_GPU_COUNT_MARKER_KINDS = ("epm:pod-provisioned", "epm:progress", "epm:cluster-launched")


# ── Small helpers ───────────────────────────────────────────────────────────


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp (trailing-Z or offset). None on garbage."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _halfup(x: float) -> int:
    """Half-up integer rounding, identical to JS ``Math.floor(x + 0.5)``.

    Python's built-in round() is banker's rounding and would drift from the
    TS mirror on .5 boundaries — the shared test vectors pin this.
    """
    return math.floor(x + 0.5)


def _normalize_status(value: Any) -> str | None:
    """Map a raw `to`/`from` value to a known canonical status, else None."""
    if not isinstance(value, str) or not value:
        return None
    status = LEGACY_STAGE_MAP.get(value, value)
    return status if status in _KNOWN_STATUSES else None


# ── Stage stats (recency-windowed, kind-bucketed) ───────────────────────────


def _kind_bucket(kind: str | None) -> str | None:
    """experiment | code | None (pooled-only) bucket for a task kind."""
    if kind == "experiment":
        return "experiment"
    if kind in CODE_KINDS:
        return "code"
    return None


def _read_registry_kinds() -> dict[str, str]:
    """task-id -> kind from REGISTRY.json (denormalized). Empty on miss."""
    reg_path = tasks_dir() / "REGISTRY.json"
    try:
        reg = json.loads(reg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, str] = {}
    tasks = reg.get("tasks") if isinstance(reg, dict) else None
    if isinstance(tasks, dict):
        for tid, entry in tasks.items():
            if isinstance(entry, dict) and isinstance(entry.get("kind"), str):
                out[str(tid)] = entry["kind"]
    return out


def _frontmatter_kind(task_dir: Path) -> str | None:
    """Cheap frontmatter `kind:` read (fallback when REGISTRY lacks the id)."""
    body = task_dir / "body.md"
    try:
        with body.open() as f:
            first = f.readline()
            if first.strip() != "---":
                return None
            for line in f:
                if line.strip() == "---":
                    return None
                if line.startswith("kind:"):
                    return line.split(":", 1)[1].strip() or None
    except OSError:
        return None
    return None


def _status_transitions(events: list[dict[str, Any]]) -> list[tuple[datetime, str]]:
    """Ordered (ts, normalized-to) pairs from ``epm:status-changed`` rows.

    Reads ``epm:status-changed`` rows ONLY — legacy ``state_changed`` history
    (older schema) is deliberately excluded. Rows with unparseable ts or a
    non-status ``to`` are skipped.
    """
    out: list[tuple[datetime, str]] = []
    for e in events:
        if e.get("kind") != "epm:status-changed":
            continue
        to = _normalize_status(e.get("to"))
        ts = _parse_iso(e.get("ts"))
        if to is None or ts is None:
            continue
        out.append((ts, to))
    out.sort(key=lambda p: p[0])
    return out


def collect_stage_spans() -> list[dict[str, Any]]:
    """Clean forward stage spans across terminal-ish tasks.

    Returns rows ``{task_id, bucket, stage, dur_h, end_ts}`` where ``bucket``
    is "experiment" | "code" | None (pooled-only). See the module docstring
    for the estimand / censoring direction.
    """
    td = tasks_dir()
    reg_kinds = _read_registry_kinds()
    spans: list[dict[str, Any]] = []
    for status in TERMINAL_SCAN_STATUSES:
        status_dir = td / status
        if not status_dir.is_dir():
            continue
        for d in sorted(status_dir.iterdir()):
            if not d.is_dir() or not d.name.isdigit():
                continue
            ev_path = d / "events.jsonl"
            if not ev_path.is_file():
                continue
            try:
                # canonical tolerant reader: no splitlines() Unicode-boundary
                # shred, and per-line skip — one bad line no longer drops the
                # whole task (#950). OSError guard retained for the
                # is_file()-to-read race (_iter_jsonl pre-checks exists() but
                # does not catch a read-time OSError).
                events = _iter_jsonl(ev_path)
            except OSError:
                continue
            kind = reg_kinds.get(d.name) or _frontmatter_kind(d)
            bucket = _kind_bucket(kind)
            transitions = _status_transitions(events)
            for (t0, s0), (t1, s1) in itertools.pairwise(transitions):
                if s0 not in _STAGE_INDEX and s0 != FOLLOWUP_STAGE:
                    continue
                if s0 == FOLLOWUP_STAGE:
                    # A follow-up round's only clean forward exit is re-parking
                    # (awaiting_promotion) or completing.
                    forward = s1 in FORWARD_EXITS
                else:
                    forward = s1 in FORWARD_EXITS or (
                        s1 in _STAGE_INDEX and _STAGE_INDEX[s1] > _STAGE_INDEX[s0]
                    )
                if not forward:
                    continue  # blocked / backward / archived exits excluded
                dur_h = max((t1 - t0).total_seconds() / 3600.0, 0.0)
                spans.append(
                    {
                        "task_id": int(d.name),
                        "bucket": bucket,
                        "stage": s0,
                        "dur_h": dur_h,
                        "end_ts": t1,
                    }
                )
    return spans


def _quantile_cell(durations: list[float], n: int, basis: str) -> dict[str, Any]:
    """{n, p25_h, median_h, p75_h, basis} with every quantile EPS_H-floored."""
    if not durations:
        return {"n": 0, "p25_h": EPS_H, "median_h": EPS_H, "p75_h": EPS_H, "basis": basis}
    if len(durations) == 1:
        v = max(durations[0], EPS_H)
        return {"n": 1, "p25_h": v, "median_h": v, "p75_h": v, "basis": basis}
    q1, q2, q3 = statistics.quantiles(durations, n=4, method="inclusive")
    return {
        "n": n,
        "p25_h": max(q1, EPS_H),
        "median_h": max(q2, EPS_H),
        "p75_h": max(q3, EPS_H),
        "basis": basis,
    }


def build_stage_stats(now: datetime | None = None) -> dict[str, Any]:
    """Build the recency-windowed, kind-bucketed stage statistics.

    Window rule: per (bucket, stage) keep the last ``WINDOW_K`` spans by
    span-end timestamp; if n < ``MIN_N`` fall back to the pooled same-window
    cell; if pooled n < ``MIN_N`` fall back to all-history pooled. Each cell
    records the basis actually used. NEVER writes anything.
    """
    now = now or _utcnow()
    spans = collect_stage_spans()

    by_bucket_stage: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sp in spans:
        # Pooled gets everything; the named bucket (if any) gets its own copy.
        by_bucket_stage.setdefault(("pooled", sp["stage"]), []).append(sp)
        if sp["bucket"] in ("experiment", "code"):
            by_bucket_stage.setdefault((sp["bucket"], sp["stage"]), []).append(sp)

    def _windowed(samples: list[dict[str, Any]]) -> list[float]:
        recent = sorted(samples, key=lambda s: s["end_ts"])[-WINDOW_K:]
        return [s["dur_h"] for s in recent]

    buckets: dict[str, dict[str, dict[str, Any]]] = {}
    for bucket in ("experiment", "code", "pooled"):
        cells: dict[str, dict[str, Any]] = {}
        for stage in [*MACHINE_STAGES, FOLLOWUP_STAGE]:
            own = by_bucket_stage.get((bucket, stage), [])
            pooled = by_bucket_stage.get(("pooled", stage), [])
            if bucket != "pooled":
                durs = _windowed(own)
                if len(durs) >= MIN_N:
                    cells[stage] = _quantile_cell(durs, len(durs), "bucket")
                    continue
            durs = _windowed(pooled)
            if len(durs) >= MIN_N:
                cells[stage] = _quantile_cell(durs, len(durs), "pooled")
            else:
                all_durs = [s["dur_h"] for s in pooled]
                cells[stage] = _quantile_cell(all_durs, len(all_durs), "all-history")
        buckets[bucket] = cells

    # Floors cover the 7-stage main pass ONLY — the optional follow-up round
    # gets a flat 0.0 floor (its own 0→1 track restarts the bar).
    pct_floor_by_stage: dict[str, dict[str, float]] = {}
    for bucket, cells in buckets.items():
        medians = [cells[s]["median_h"] for s in MACHINE_STAGES]  # already EPS-floored
        total = sum(medians)
        floors: dict[str, float] = {}
        acc = 0.0
        for stage, m in zip(MACHINE_STAGES, medians, strict=True):
            floors[stage] = acc / total
            acc += m
        floors[FOLLOWUP_STAGE] = 0.0
        pct_floor_by_stage[bucket] = floors

    return {
        "window_rule": (
            f"last-{WINDOW_K}-spans-per-stage, min_n={MIN_N}, "
            f"fallback bucket->pooled->all-history, eps_h={EPS_H}"
        ),
        "stats_generated_at": _iso(now),
        "buckets": buckets,
        "pct_floor_by_stage": pct_floor_by_stage,
    }


# ── GPU-hours refinement (running stage only; SOFT, basis-tagged) ───────────


def _pods_ephemeral_path() -> Path:
    """Path of the pods_ephemeral.json sidecar (separate hook for test injection).

    Prefers the LIVE relocated copy at ``<git-common-dir>/eps/`` (task #1183);
    falls back to the tracked seed. Read-only — never migrates, never locks.
    NOTE: ``repo_root()`` can return the managed ``_task-main-pin`` worktree
    (primary checkout off-``main``), where ``.git`` is a gitfile (a FILE) —
    the ``live.exists()`` check then reads False and this degrades to the
    seed read (today's behavior). Never assert ``.git`` is a directory here.
    """
    from explore_persona_space.task_workflow import repo_root

    root = repo_root()
    live = root / ".git" / "eps" / "pods_ephemeral.json"
    return live if live.exists() else root / "scripts" / "pods_ephemeral.json"


def extract_gpu_hours(events: list[dict[str, Any]]) -> float | None:
    """``gpu_hours_total=<X>`` from the latest ``epm:plan`` note (fallback
    ``epm:plan-approved``). None when absent or X <= 0 — every ``kind: infra``
    plan emits ``gpu_hours_total=0``, which must SKIP refinement entirely."""
    for kind in ("epm:plan", "epm:plan-approved"):
        for e in reversed(events):
            if e.get("kind") != kind:
                continue
            note = e.get("note")
            if not isinstance(note, str):
                continue
            m = _GPU_HOURS_RE.search(note)
            if m:
                try:
                    value = float(m.group(1))
                except ValueError:
                    continue
                return value if value > 0 else None
    return None


def recover_gpu_count(issue: int, events: list[dict[str, Any]]) -> tuple[int, str]:
    """3-level GPU-count recovery chain: anchored note regex → intent map →
    assume 1 GPU. Returns (count, conversion-tag)."""
    for e in reversed(events):
        if e.get("kind") not in _GPU_COUNT_MARKER_KINDS:
            continue
        note = e.get("note")
        if not isinstance(note, str):
            continue
        m = _GPU_COUNT_RE.search(note)
        if m:
            count = int(m.group(1))
            if count >= 1:
                return count, "note-regex"
    # pods_ephemeral.json → gpu_intent → count map. Unknown intents (observed
    # "custom") fall through to assume-1gpu — never a KeyError.
    try:
        pods = json.loads(_pods_ephemeral_path().read_text()).get("pods", {})
        for entry in pods.values():
            if isinstance(entry, dict) and entry.get("issue") == issue:
                count = GPU_INTENT_COUNTS.get(entry.get("gpu_intent"))
                if count:
                    return count, "intent-map"
    except (OSError, json.JSONDecodeError, RuntimeError):
        pass
    return 1, "assumed-1gpu"


# ── Per-task estimate ───────────────────────────────────────────────────────


def _stage_entered_at(events: list[dict[str, Any]], stage: str, now: datetime) -> datetime:
    """ts of the LAST epm:status-changed entering ``stage``; falls back to the
    earliest event ts, then ``now``."""
    transitions = _status_transitions(events)
    for ts, to in reversed(transitions):
        if to == stage:
            return ts
    earliest: datetime | None = None
    for e in events:
        ts = _parse_iso(e.get("ts"))
        if ts is not None and (earliest is None or ts < earliest):
            earliest = ts
    return earliest or now


def _last_machine_stage_before_block(events: list[dict[str, Any]]) -> str:
    """Newest forward `to` that maps to a machine stage; "planning" (floor 0)
    when a task blocked with no prior machine stage (pinned crash guard)."""
    for _ts, to in reversed(_status_transitions(events)):
        if to in _STAGE_INDEX:
            return to
    return "planning"


def estimate_task_progress(
    issue: int, stats: dict[str, Any], now: datetime | None = None
) -> dict[str, Any] | None:
    """One task's snapshot row (the §3.7 pinned contract), or None for any
    status outside the 7 machine stages + ``followups_running`` + ``blocked``
    (explicit allowlist — every other status gets NO bar, NO suffix, NO
    snapshot row). NEVER writes.
    """
    now = now or _utcnow()
    task = get_task(issue)
    status = task["status"]
    fm = task.get("frontmatter") or {}
    kind = fm.get("kind") if isinstance(fm.get("kind"), str) else None

    blocked = status == "blocked"
    stage = _normalize_status(status) if not blocked else None
    followup = (not blocked) and stage == FOLLOWUP_STAGE
    if not blocked and not followup and stage not in _STAGE_INDEX:
        return None

    events = list_events(issue)
    if blocked:
        stage = _last_machine_stage_before_block(events)
    assert stage is not None  # narrowed above

    bucket = _kind_bucket(kind) or "pooled"
    cells = stats["buckets"].get(bucket) or stats["buckets"]["pooled"]
    floors = stats["pct_floor_by_stage"].get(bucket) or stats["pct_floor_by_stage"]["pooled"]

    if followup:
        # Own 0→1 track: the round executes while the status HOLDS at
        # followups_running (no inner transitions to interpolate over), so the
        # bar restarts and paces over the round's own historical clean spans.
        cell = cells[FOLLOWUP_STAGE]
        floor, span = 0.0, 1.0
        stats_basis = cell["basis"]
        remaining = {"p25_h": 0.0, "median_h": 0.0, "p75_h": 0.0}
    else:
        idx = _STAGE_INDEX[stage]
        floor = floors[stage]
        next_floor = floors[MACHINE_STAGES[idx + 1]] if idx + 1 < len(MACHINE_STAGES) else 1.0
        span = next_floor - floor

        bases = {cells[s]["basis"] for s in MACHINE_STAGES}
        stats_basis = bases.pop() if len(bases) == 1 else "mixed"

        remaining = {"p25_h": 0.0, "median_h": 0.0, "p75_h": 0.0}
        for s in MACHINE_STAGES[idx + 1 :]:
            if s == "plan_pending":
                continue  # human wait — excluded from every machine-ETA term
            for q in remaining:
                remaining[q] += cells[s][q]
        cell = cells[stage]

    frac_median_h = max(cell["median_h"], EPS_H)
    stage_q = {q: cell[q] for q in ("p25_h", "median_h", "p75_h")}
    eta_basis = "historical"
    gpu_hours_total: float | None = None
    gpu_count: int | None = None
    gpu_conversion: str | None = None

    if not blocked and not followup and stage == "running":
        gpu_hours_total = extract_gpu_hours(events)
        if gpu_hours_total is not None:
            gpu_count, gpu_conversion = recover_gpu_count(issue, events)
            hist_median = max(cell["median_h"], EPS_H)
            # Clamp ≥ historical p25 — guards the known optimism of assuming
            # perfect N-GPU parallelism. The band is ratio-scaled.
            refined_median = max(gpu_hours_total / max(gpu_count, 1), cell["p25_h"])
            stage_q = {
                "p25_h": refined_median * (cell["p25_h"] / hist_median),
                "median_h": refined_median,
                "p75_h": refined_median * (cell["p75_h"] / hist_median),
            }
            eta_basis = "gpu-assumed" if gpu_conversion == "assumed-1gpu" else "gpu-refined"

    # Expected TOTAL machine time for a typical clean pass: the 7-stage main
    # pipeline minus plan_pending (human wait), with the current stage's
    # EFFECTIVE quantiles substituted (GPU-refined when running). A follow-up
    # row's total is the round's own expected duration — the main pass is
    # already behind it.
    if followup:
        total = dict(stage_q)
    else:
        total = {"p25_h": 0.0, "median_h": 0.0, "p75_h": 0.0}
        for s in MACHINE_STAGES:
            if s == "plan_pending":
                continue
            src = stage_q if s == stage else cells[s]
            for q in total:
                total[q] += src[q]

    entered = _stage_entered_at(events, stage, now)
    row = {
        "issue": int(issue),
        "status": status,
        "stage": stage,
        "kind_bucket": bucket,
        "stats_basis": stats_basis,
        "stage_entered_at": _iso(entered),
        "pct_floor": round(floor, 6),
        "pct_span": round(span, 6),
        # frac pace stays HISTORICAL (§3.3); stage_*_h are the EFFECTIVE band
        # quantiles (GPU-refined when eta_basis says so; p75 drives overdue).
        "frac_median_h": round(frac_median_h, 6),
        "stage_p25_h": round(stage_q["p25_h"], 6),
        "stage_median_h": round(stage_q["median_h"], 6),
        "stage_p75_h": round(stage_q["p75_h"], 6),
        "remaining_after_p25_h": round(remaining["p25_h"], 6),
        "remaining_after_median_h": round(remaining["median_h"], 6),
        "remaining_after_p75_h": round(remaining["p75_h"], 6),
        "total_p25_h": round(total["p25_h"], 6),
        "total_median_h": round(total["median_h"], 6),
        "total_p75_h": round(total["p75_h"], 6),
        "human_wait": (not blocked) and stage == "plan_pending",
        "blocked": blocked,
        "plan_review_ahead": (
            (not blocked) and (not followup) and _STAGE_INDEX[stage] < _STAGE_INDEX["plan_pending"]
        ),
        "gpu_hours_total": gpu_hours_total,
        "gpu_count": gpu_count,
        "gpu_conversion": gpu_conversion,
        "eta_basis": eta_basis,
    }
    return row


# ── Interpolation (THE formula mirrored in dashboard/lib/progress.ts) ───────


def interpolate(
    row: dict[str, Any], now: datetime | None = None
) -> tuple[float, dict[str, float] | None, bool]:
    """(pct, eta_band | None, overdue) from a snapshot row at time ``now``.

    Pinned by ``tests/fixtures/task_progress_vectors.json`` — consumed by BOTH
    pytest and the tsx mirror test. ``overdue`` is computed here (read time)
    so a long-lived snapshot row degrades to overdue without waiting for the
    next cron tick. The band is None when blocked or overdue (suppressed).
    Boundary: ``elapsed == p75`` is NOT overdue (strict >).
    """
    now = now or _utcnow()
    if row.get("blocked"):
        return float(row["pct_floor"]), None, False
    entered = _parse_iso(row.get("stage_entered_at"))
    elapsed_h = max((now - entered).total_seconds() / 3600.0, 0.0) if entered else 0.0
    if row.get("human_wait"):
        eta = {
            "p25_h": float(row["remaining_after_p25_h"]),
            "median_h": float(row["remaining_after_median_h"]),
            "p75_h": float(row["remaining_after_p75_h"]),
        }
        return float(row["pct_floor"]), eta, False
    frac = min(elapsed_h / max(float(row["frac_median_h"]), EPS_H), FRAC_CAP)
    pct = float(row["pct_floor"]) + frac * float(row["pct_span"])
    if elapsed_h > float(row["stage_p75_h"]):
        return pct, None, True
    eta = {
        q: max(float(row[f"stage_{q}"]) - elapsed_h, 0.0) + float(row[f"remaining_after_{q}"])
        for q in ("p25_h", "median_h", "p75_h")
    }
    return pct, eta, False


# ── Display formatting (shared spec — TS mirrors byte-for-byte) ─────────────


def _fmt_hours(v: float) -> str:
    """<10 h → one decimal (trailing .0 stripped); else half-up integer."""
    if v < 10:
        d = math.floor(v * 10 + 0.5) / 10
        return str(int(d)) if d == int(d) else f"{d:.1f}"
    return str(_halfup(v))


def _fmt_days(v: float) -> str:
    """Hours → days with one decimal (trailing .0 stripped), half-up."""
    d = math.floor(v / 24 * 10 + 0.5) / 10
    return str(int(d)) if d == int(d) else f"{d:.1f}"


def format_eta_band(p25_h: float, p75_h: float, eta_basis: str) -> str:
    """Compact band: "~25-50m" | "~4-9h" | "~1.3-2.5d" (en-dash separator);
    "≈" when the basis is GPU-refined/assumed (soft estimate, rendered
    distinguishably). The unicode dash/almost-equal are part of the pinned
    display format shared with the TS mirror."""
    prefix = "~" if eta_basis == "historical" else "≈"
    if p75_h < 1:
        a = max(1, _halfup(p25_h * 60))
        b = max(1, _halfup(p75_h * 60))
        return f"{prefix}{a}–{b}m"  # noqa: RUF001
    if p75_h < 24:
        return f"{prefix}{_fmt_hours(p25_h)}–{_fmt_hours(p75_h)}h"  # noqa: RUF001
    return f"{prefix}{_fmt_days(p25_h)}–{_fmt_days(p75_h)}d"  # noqa: RUF001


def format_duration(hours: float, eta_basis: str = "historical") -> str:
    """Compact single duration: "~25m" | "~2.1h" | "~1.3d" — same prefix and
    unit thresholds as :func:`format_eta_band`. Used for the dashboard's
    median remaining/total labels (point estimates — deliberately NOT gated
    by the §7 band kill switch, which is about [p25, p75] coverage claims).
    ``dashboard/lib/progress.ts`` ``formatDuration`` mirrors this; the shared
    fixture pins both."""
    prefix = "~" if eta_basis == "historical" else "≈"
    if hours < 1:
        return f"{prefix}{max(1, _halfup(hours * 60))}m"
    if hours < 24:
        return f"{prefix}{_fmt_hours(hours)}h"
    return f"{prefix}{_fmt_days(hours)}d"


def format_title_suffix(
    row: dict[str, Any] | None,
    now: datetime | None = None,
    include_band: bool | None = None,
) -> str | None:
    """``"▓▓░░░ 43% ~4-9h"`` (en-dash band) for machine-active rows; overdue
    drops the band (``"▓▓▓▓░ 87% overdue"``); plan_pending / blocked /
    out-of-scope → None.

    ``include_band`` defaults to :data:`ETA_BAND_ENABLED` (the §7 kill-switch
    — currently OFF, so production titles carry ``"▓▓░░░ 43%"`` with no hour
    band). Tests pass ``include_band=True`` to pin the full format for
    re-enablement.
    """
    if include_band is None:
        include_band = ETA_BAND_ENABLED
    if not row or row.get("blocked") or row.get("human_wait"):
        return None
    if row.get("stage") not in _STAGE_INDEX or row.get("stage") == "plan_pending":
        return None
    pct, eta, overdue = interpolate(row, now)
    filled = min(5, max(0, _halfup(pct * 5)))
    bar = "▓" * filled + "░" * (5 - filled)
    pct_i = _halfup(pct * 100)
    if overdue:
        return f"{bar} {pct_i}% overdue"
    if eta is None or not include_band:
        return f"{bar} {pct_i}%"
    band = format_eta_band(eta["p25_h"], eta["p75_h"], row.get("eta_basis", "historical"))
    return f"{bar} {pct_i}% {band}"


# ── Snapshot I/O ────────────────────────────────────────────────────────────


def _read_snapshot_file() -> dict[str, Any] | None:
    try:
        data = json.loads(SNAPSHOT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _stats_age_s(stats: Any, now: datetime) -> float | None:
    if not isinstance(stats, dict):
        return None
    generated = _parse_iso(stats.get("stats_generated_at"))
    if generated is None:
        return None
    return (now - generated).total_seconds()


def load_stats_readonly() -> dict[str, Any] | None:
    """Stats section from the snapshot file; None when missing or stale.

    NEVER rebuilds (single-writer contract: the title path degrades to
    ``suffix=None`` instead of scanning 33 MB of events.jsonl inline).
    """
    snap = _read_snapshot_file()
    if not snap:
        return None
    stats = snap.get("stats")
    age = _stats_age_s(stats, _utcnow())
    if age is None or age > STATS_READ_MAX_AGE_S:
        return None
    if not isinstance(stats, dict) or "buckets" not in stats or "pct_floor_by_stage" not in stats:
        return None
    return stats


def write_snapshot(force_stats: bool = False, now: datetime | None = None) -> Path:
    """Materialize ``~/.eps-autonomous/task_progress.json`` (atomic
    temp+rename). Reuses the prior stats section when fresher than
    ``STATS_TTL_S``; estimates every in-flight task (7 machine stages +
    followups_running + blocked). THE ONLY WRITER — called by the cron + CLI
    only, never from the title path. Strictly read-only over ``tasks/``."""
    now = now or _utcnow()
    stats: dict[str, Any] | None = None
    if not force_stats:
        prior = _read_snapshot_file()
        if prior:
            prior_stats = prior.get("stats")
            age = _stats_age_s(prior_stats, now)
            if (
                age is not None
                and age <= STATS_TTL_S
                and isinstance(prior_stats, dict)
                and "buckets" in prior_stats
                and "pct_floor_by_stage" in prior_stats
            ):
                stats = prior_stats
    if stats is None:
        stats = build_stage_stats(now)

    tasks: dict[str, dict[str, Any]] = {}
    td = tasks_dir()
    for status in [*MACHINE_STAGES, FOLLOWUP_STAGE, "blocked"]:
        status_dir = td / status
        if not status_dir.is_dir():
            continue
        for d in sorted(status_dir.iterdir()):
            if not d.is_dir() or not d.name.isdigit():
                continue
            issue = int(d.name)
            try:
                row = estimate_task_progress(issue, stats, now=now)
            except Exception as exc:  # loud-but-resilient: one bad task must
                # not blank the whole snapshot; the cron log carries the line.
                print(f"task_progress: estimate failed for #{issue}: {exc}", file=sys.stderr)
                continue
            if row is not None:
                tasks[str(issue)] = row

    payload = {
        "version": 1,
        "generated_at": _iso(now),
        "stats": stats,
        "tasks": tasks,
    }
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_PATH.with_suffix(SNAPSHOT_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(SNAPSHOT_PATH)
    return SNAPSHOT_PATH


__all__ = [
    "ACTIVE_TITLE_STATUSES",
    "EPS_H",
    "ETA_BAND_ENABLED",
    "FOLLOWUP_STAGE",
    "FRAC_CAP",
    "LEGACY_STAGE_MAP",
    "MACHINE_STAGES",
    "MIN_N",
    "SNAPSHOT_PATH",
    "STATS_TTL_S",
    "WINDOW_K",
    "build_stage_stats",
    "collect_stage_spans",
    "estimate_task_progress",
    "extract_gpu_hours",
    "find_task_path",
    "format_duration",
    "format_eta_band",
    "format_title_suffix",
    "interpolate",
    "load_stats_readonly",
    "recover_gpu_count",
    "write_snapshot",
]
