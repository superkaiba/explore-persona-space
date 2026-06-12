"""One-call tick triage for the /issue-tick and /campaign-tick skills.

The lightweight tick skills used to spend ~5 LLM tool-call turns per fire
(state reads, title refresh, branch logic, snapshot write) even when nothing
needed doing. This script collapses the whole healthy-path decision into ONE
Bash call: it reads the task's status + latest marker through the
task-workflow library (no subprocess shellouts, branch-guard-safe from any
cwd), compares against the previous tick's snapshot, and prints exactly one
verdict line::

    HEALTHY <reason>          # nothing to do — the tick skill ENDS THE TURN
    TERMINAL <reason>         # done/parked — the tick skill tears down its cron
    GATE-TRANSITION <reason>  # just crossed into a user gate — push + teardown
    STALE-REDRIVE <reason>    # chain likely dead — the tick skill loads /issue

Exit code 0 on ANY successful triage (the verdict word carries the decision).
ANY failure (missing task, unreadable registry, unknown status) exits
non-zero with a loud stderr line — the tick skill treats a non-zero exit as
STALE-REDRIVE (fail toward coverage, never toward silence).

Side effects (both under ``~/.eps-autonomous``, overridable for tests via
``EPM_TICK_STATE_DIR``):

* ``issue-tick-last-status/<N>.json`` — the per-issue snapshot (same file the
  tick skills wrote before; this script now owns the write). Adds a
  ``terminal_streak`` counter to the legacy ``{issue, status, ts}`` shape.
* ``tick-runaway-<N>.flag`` — written on the ``EPM_TICK_RUNAWAY_STREAK``-th
  (default 3rd) consecutive TEARDOWN-verdict triage (TERMINAL or
  GATE-TRANSITION — covers terminal statuses, over-cap plan_pending, and
  stranded campaign crons); cleared on any streak reset. A cron that keeps
  firing at a teardown site means CRON-TEARDOWN keeps whiffing (the #501
  runaway class: 1,951 wasted ticks over ~40h); the flag is the watcher's
  signal to force-stop the session (``autonomous_session_watch`` gate-push
  pass), which kills the session-scoped cron with it.

CLI::

    uv run python scripts/tick_triage.py <N>                  # /issue-tick
    uv run python scripts/tick_triage.py <N> --kind campaign  # /campaign-tick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

# ── status sets (issue mode) ────────────────────────────────────────────────
# Mirror the /issue-tick skill's branch sets. Members must stay inside the
# runtime enum `task_workflow.STATUSES`.
ISSUE_TERMINAL = frozenset({"completed", "archived", "awaiting_promotion", "blocked"})
ISSUE_GATE = frozenset({"awaiting_promotion", "blocked"})
ISSUE_PARK = frozenset({"proposed", "planning", "plan_pending", "followups_running"})
ISSUE_ACTIVE = frozenset({"approved", "running", "verifying", "interpreting", "reviewing"})

# ── status sets (campaign mode) ─────────────────────────────────────────────
CAMPAIGN_TERMINAL = frozenset({"completed", "archived", "blocked"})
# A tick should never be armed before the brief-approval gate — a cron seen
# at these statuses is stranded and gets torn down (TERMINAL verdict).
CAMPAIGN_STRANDED = frozenset({"proposed", "planning", "plan_pending"})
CAMPAIGN_ACTIVE = frozenset({"approved", "running"})

# Campaign-state experiment rows in these statuses need no further decision.
CAMPAIGN_ROW_FINISHED = frozenset({"ingested", "abandoned"})
# A child at one of these statuses has LANDED a result (or died) — the
# campaign owes a reconcile/ingest decision round.
CAMPAIGN_CHILD_LANDED = frozenset({"awaiting_promotion", "completed", "blocked"})
# A child at any status outside LANDED + archived is genuinely in flight; its
# own /issue session + the watcher passes cover it, so the campaign can idle.
CAMPAIGN_CHILD_DONEISH = CAMPAIGN_CHILD_LANDED | {"archived"}

STALE_S_DEFAULT = 25 * 60  # the tick skills' long-standing ~25-min staleness window
RUNAWAY_STREAK_DEFAULT = 3

# Watcher-posted campaign markers carry this sentinel in their note; they are
# alerts, not campaign progress, so they never count as freshness.
_WATCHER_NOTE_SENTINEL = "[autonomous_session_watch"


# ── state files ─────────────────────────────────────────────────────────────


def state_dir() -> Path:
    """Root for snapshot + runaway-flag files (``EPM_TICK_STATE_DIR`` for
    tests; defaults to the shared ``~/.eps-autonomous``)."""
    override = os.environ.get("EPM_TICK_STATE_DIR", "").strip()
    return Path(override) if override else (Path.home() / ".eps-autonomous")


def snapshot_path(issue: int) -> Path:
    return state_dir() / "issue-tick-last-status" / f"{issue}.json"


def runaway_flag_path(issue: int) -> Path:
    return state_dir() / f"tick-runaway-{issue}.flag"


def read_snapshot(issue: int) -> dict:
    """Previous tick's snapshot (``{}`` when absent/garbled — a missing
    snapshot means 'previous status unknown')."""
    path = snapshot_path(issue)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def write_snapshot(issue: int, status: str, terminal_streak: int) -> None:
    """Atomic temp+rename write of the per-issue snapshot (legacy shape plus
    the ``terminal_streak`` runaway counter)."""
    path = snapshot_path(issue)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": issue,
        "status": status,
        "ts": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "terminal_streak": terminal_streak,
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{issue}-")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_runaway_flag(issue: int, status: str, streak: int) -> None:
    """Drop the runaway flag for the watcher's force-stop check. Idempotent
    (overwrites); content is diagnostic only."""
    path = runaway_flag_path(issue)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": issue,
        "status": status,
        "terminal_streak": streak,
        "ts": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".runaway-{issue}-")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


# ── task-state readers (lazy task_workflow imports; monkeypatchable) ────────


def load_task_state(issue: int) -> tuple[str, list[dict]]:
    """Return ``(status, events)`` via the task-workflow library.

    Raises on ANY read failure — main() converts that to a loud non-zero
    exit so the tick skill falls back to the full re-drive path."""
    from explore_persona_space.task_workflow import get_task, list_events

    task = get_task(issue)
    status = task.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"task #{issue}: unreadable status")
    return status, list_events(issue)


def load_children(issue: int) -> list[dict]:
    """Campaign mode: the child-task rows (id/status) via the library."""
    from explore_persona_space.task_workflow import list_children

    return list_children(issue)


def load_campaign_state(issue: int) -> dict:
    """Campaign mode: ``artifacts/campaign-state.json`` (``{}`` if absent —
    a campaign with no state file yet owes a decision round, which the
    verdict logic surfaces as STALE-REDRIVE via the unreconciled check)."""
    from explore_persona_space.task_workflow import find_task_path

    path = find_task_path(issue) / "artifacts" / "campaign-state.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def parse_event_ts(ts: str | None) -> float | None:
    """ISO-8601 ``Z`` timestamp -> epoch seconds (``None`` on garbage)."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def latest_event_ts(events: list[dict], *, prefix: str | None = None) -> float | None:
    """Epoch ts of the newest event (optionally restricted to a kind prefix;
    watcher-sentinel notes never count as freshness)."""
    best: float | None = None
    for row in events:
        if not isinstance(row, dict):
            continue
        kind = row.get("kind", "")
        if prefix is not None and not str(kind).startswith(prefix):
            continue
        note = row.get("note")
        if isinstance(note, str) and _WATCHER_NOTE_SENTINEL in note:
            continue
        ts = parse_event_ts(row.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def plan_pending_over_cap(events: list[dict]) -> bool:
    """True iff the newest ``epm:awaiting-spend-approval`` marker is newer
    than the newest ``epm:status-changed`` — the over-cap plan_pending park
    (a user gate), vs the under-cap in-skill park."""
    spend = latest_event_ts(events, prefix="epm:awaiting-spend-approval")
    if spend is None:
        return False
    changed = latest_event_ts(events, prefix="epm:status-changed")
    return changed is None or spend >= changed


# ── pure verdict logic ──────────────────────────────────────────────────────


def stale_s() -> float:
    raw = os.environ.get("EPM_TICK_STALE_S", "")
    try:
        val = float(raw)
    except ValueError:
        return STALE_S_DEFAULT
    return val if val > 0 else STALE_S_DEFAULT


def runaway_streak_threshold() -> int:
    raw = os.environ.get("EPM_TICK_RUNAWAY_STREAK", "")
    try:
        val = int(raw)
    except ValueError:
        return RUNAWAY_STREAK_DEFAULT
    return val if val > 0 else RUNAWAY_STREAK_DEFAULT


def compute_issue_verdict(
    status: str,
    prev_status: str | None,
    marker_age_s: float | None,
    over_cap: bool,
    *,
    stale_after_s: float,
) -> tuple[str, str]:
    """Pure verdict for /issue-tick. Returns ``(verdict, reason)``.

    Raises ValueError on a status outside the known enum sets — main()
    converts that to a non-zero exit (fail toward coverage)."""
    gate_now = status in ISSUE_GATE or (status == "plan_pending" and over_cap)
    if status in ISSUE_TERMINAL or (status == "plan_pending" and over_cap):
        if gate_now and prev_status != status:
            return (
                "GATE-TRANSITION",
                f"status={status} (prev={prev_status or 'unknown'}) — user gate just "
                "reached; push + teardown",
            )
        return ("TERMINAL", f"status={status} — teardown")
    if status not in ISSUE_PARK and status not in ISSUE_ACTIVE:
        raise ValueError(f"unknown status {status!r}")
    age_desc = "no markers" if marker_age_s is None else f"marker age {marker_age_s / 60:.0f}m"
    if marker_age_s is not None and marker_age_s <= stale_after_s:
        return ("HEALTHY", f"status={status}, {age_desc} — chain alive")
    kind = "in-skill chain" if status in ISSUE_PARK else "bg poll chain"
    return ("STALE-REDRIVE", f"status={status}, {age_desc} — {kind} likely dead")


def compute_campaign_verdict(
    status: str,
    prev_status: str | None,
    campaign_marker_age_s: float | None,
    *,
    landed_unreconciled: list[int],
    open_rows_all_in_flight: bool,
    stale_after_s: float,
) -> tuple[str, str]:
    """Pure verdict for /campaign-tick. Returns ``(verdict, reason)``."""
    if status in CAMPAIGN_TERMINAL:
        if status == "blocked" and prev_status != status:
            return (
                "GATE-TRANSITION",
                f"status=blocked (prev={prev_status or 'unknown'}) — push + teardown",
            )
        return ("TERMINAL", f"status={status} — teardown")
    if status in CAMPAIGN_STRANDED:
        return ("TERMINAL", f"status={status} — stranded cron (campaign not approved); teardown")
    if status not in CAMPAIGN_ACTIVE:
        raise ValueError(f"unknown campaign status {status!r}")
    if landed_unreconciled:
        ids = ", ".join(f"#{c}" for c in landed_unreconciled[:6])
        return ("STALE-REDRIVE", f"results landed unreconciled ({ids}) — run a decision round")
    age_desc = (
        "no campaign markers"
        if campaign_marker_age_s is None
        else f"campaign marker age {campaign_marker_age_s / 60:.0f}m"
    )
    if campaign_marker_age_s is not None and campaign_marker_age_s <= stale_after_s:
        return ("HEALTHY", f"status={status}, {age_desc} — decision loop alive")
    if open_rows_all_in_flight:
        return ("HEALTHY", f"status={status}, {age_desc} — all open arms in flight in children")
    return ("STALE-REDRIVE", f"status={status}, {age_desc} — decision round owed")


def campaign_open_rows(state: dict, children: list[dict]) -> tuple[list[int], bool]:
    """Derive ``(landed_unreconciled_child_ids, open_rows_all_in_flight)``
    from the campaign-state experiment rows + the live child statuses.

    ``open_rows_all_in_flight`` is True ONLY when at least one open
    (non-finished) row exists AND every open row maps to a child at a
    genuinely in-flight status. ZERO open rows — missing/garbled state
    file, or every row ingested/abandoned — returns False: such a campaign
    owes a decision round (propose the next arm or conclude), so a
    stale-marker tick must STALE-REDRIVE it, never idle as HEALTHY
    (review blocker, 2026-06-12)."""
    child_status = {row.get("id"): row.get("status") for row in children}
    rows = state.get("experiments")
    rows = rows if isinstance(rows, list) else []
    landed: list[int] = []
    open_rows = 0
    in_flight = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("status") in CAMPAIGN_ROW_FINISHED:
            continue
        open_rows += 1
        child = row.get("child_task")
        if not isinstance(child, int):
            # A planned row with no child filed yet — a decision is owed
            # (not in flight).
            continue
        cstat = child_status.get(child)
        if cstat in CAMPAIGN_CHILD_LANDED:
            landed.append(child)
        elif cstat is not None and cstat not in CAMPAIGN_CHILD_DONEISH:
            in_flight += 1
    return landed, open_rows > 0 and in_flight == open_rows


# ── streak + main ───────────────────────────────────────────────────────────


def update_terminal_streak(issue: int, status: str, prev: dict, *, count_streak: bool) -> int:
    """Advance (or reset) the consecutive-teardown-tick counter and drop the
    runaway flag at the threshold. Returns the new streak value.

    A reset ALSO unlinks any existing runaway flag: a flag written during an
    earlier teardown-whiff episode must not survive a recovery (e.g.
    blocked -> running in the same live session) — a stale flag would
    force-stop the session on weeks-old corroboration the next time the
    task parks (review major, 2026-06-12)."""
    prev_streak = prev.get("terminal_streak")
    prev_streak = prev_streak if isinstance(prev_streak, int) and prev_streak >= 0 else 0
    if not count_streak:
        runaway_flag_path(issue).unlink(missing_ok=True)
        return 0
    streak = prev_streak + 1
    if streak >= runaway_streak_threshold():
        write_runaway_flag(issue, status, streak)
        print(
            f"tick_triage: #{issue} hit {streak} consecutive teardown-verdict ticks "
            f"(status={status}) — runaway flag written for the watcher "
            f"({runaway_flag_path(issue)})",
            file=sys.stderr,
        )
    return streak


def triage(issue: int, kind: str, now: float | None = None) -> tuple[str, str]:
    """Full triage for one tick. Returns ``(verdict, reason)``; raises on any
    state-read failure (the CLI converts that to a non-zero exit)."""
    now = now if now is not None else time.time()
    status, events = load_task_state(issue)
    prev = read_snapshot(issue)
    prev_status = prev.get("status") if isinstance(prev.get("status"), str) else None

    if kind == "campaign":
        marker_ts = latest_event_ts(events, prefix="epm:campaign")
        landed, all_in_flight = campaign_open_rows(load_campaign_state(issue), load_children(issue))
        verdict, reason = compute_campaign_verdict(
            status,
            prev_status,
            (now - marker_ts) if marker_ts is not None else None,
            landed_unreconciled=landed,
            open_rows_all_in_flight=all_in_flight,
            stale_after_s=stale_s(),
        )
    else:
        marker_ts = latest_event_ts(events)
        verdict, reason = compute_issue_verdict(
            status,
            prev_status,
            (now - marker_ts) if marker_ts is not None else None,
            plan_pending_over_cap(events),
            stale_after_s=stale_s(),
        )

    # Runaway streak counts every TEARDOWN verdict, not just the terminal
    # STATUS sets — a teardown that whiffs forever at over-cap plan_pending
    # or at a stranded campaign cron deserves the same parachute (review
    # minor, 2026-06-12). The watcher's force-stop still acts only on the
    # DONE set; other flagged statuses get its loud alert-only arm.
    count_streak = verdict in ("TERMINAL", "GATE-TRANSITION")
    streak = update_terminal_streak(issue, status, prev, count_streak=count_streak)
    write_snapshot(issue, status, streak)
    return verdict, reason


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "issue", type=int, help="task number (the integer naming tasks/<status>/<N>/)"
    )
    parser.add_argument(
        "--kind",
        choices=("issue", "campaign"),
        default="issue",
        help="which tick skill is asking (default: issue)",
    )
    args = parser.parse_args(argv)
    try:
        verdict, reason = triage(args.issue, args.kind)
    except Exception as e:
        print(f"tick_triage: FAILED for #{args.issue}: {e}", file=sys.stderr)
        return 2
    print(f"{verdict} {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
