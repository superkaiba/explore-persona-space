#!/usr/bin/env python3
"""One-pass, read-only queue report for the PM session's STATUS pass.

Gathers every task in the workflow (all statuses except ``completed`` /
``archived`` by default) in a single run so the PM persona
(`.claude/agents/research-pm.md` Mode 1) can build its structured
per-status report without N x ``task.py view`` calls. Consumed by the
`/pm` boot scan (`.claude/skills/pm/SKILL.md` step 2) and any "status"
re-run.

Per task: ``id``, ``status``, ``kind``, ``title``, ``goal`` (frontmatter,
may be null), ``tags``, ``has_clean_result``, ``created_ts`` (first
events.jsonl event ts; falls back to frontmatter ``created_at``),
``status_arrival_ts`` (ts of the last ``epm:status-changed`` event into
the current status; falls back to the last event ts), and — for active
statuses — ``latest_marker_kind`` + ``latest_marker_ts``.

Output: JSON (default) grouped by status, or ``--markdown`` for a
pre-sorted report skeleton (Active work / Awaiting promotion "Most
recent" / Proposed "Recently filed"). Research-theme grouping stays with
the PM persona (derived at read time), never this script.

Read-only: zero mutations, no git. Malformed / empty events.jsonl LINES
are skipped; file-level errors (unreadable file, missing body.md for a
registered task) still raise — fail loud per project rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Make the package importable without `uv run` plumbing. Any path containing
# `tasks/` MUST go via the canonical resolver — see
# `tests/test_no_direct_task_path_construction.py`.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.task_workflow import (  # noqa: E402
    STATUSES,
    find_task_path,
    get_task,
    list_by_status,
)

# Statuses whose entries carry latest-marker fields and feed the "Active
# work" section of research-pm.md Mode 1.
ACTIVE_STATUSES: tuple[str, ...] = (
    "planning",
    "plan_pending",
    "approved",
    "running",
    "verifying",
    "interpreting",
    "reviewing",
    "followups_running",
    "blocked",
)

# Default report scope: everything except the terminal/historical statuses
# (`blocked` is terminal for set-status purposes but is live PM work, so it
# stays in the default scope).
DEFAULT_STATUSES: tuple[str, ...] = tuple(s for s in STATUSES if s not in ("completed", "archived"))


def _read_events(task_path: Path) -> list[dict]:
    """Parse events.jsonl, skipping malformed/empty LINES (file errors raise)."""
    events_path = task_path / "events.jsonl"
    if not events_path.exists():
        return []
    events: list[dict] = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # one corrupt line must not hide the rest of the history
        if isinstance(ev, dict):
            events.append(ev)
    return events


def _task_record(task_id: int) -> dict:
    """Build one report row from the task body + events.jsonl."""
    task = get_task(task_id)
    fm = task["frontmatter"]
    status = task["status"]
    events = _read_events(find_task_path(task_id))

    created_ts = events[0].get("ts") if events else None
    if not created_ts:
        created_ts = fm.get("created_at")

    status_arrival_ts = None
    for ev in reversed(events):
        if ev.get("kind") == "epm:status-changed" and ev.get("to") == status:
            status_arrival_ts = ev.get("ts")
            break
    if status_arrival_ts is None and events:
        status_arrival_ts = events[-1].get("ts")

    record: dict = {
        "id": task_id,
        "status": status,
        "kind": fm.get("kind", "experiment"),
        "title": fm.get("title", ""),
        "goal": fm.get("goal") or None,
        "parent_id": fm.get("parent_id"),
        "tags": fm.get("tags") or [],
        "has_clean_result": bool(fm.get("has_clean_result", False)),
        "created_ts": created_ts,
        "status_arrival_ts": status_arrival_ts,
    }
    if status in ACTIVE_STATUSES:
        last = events[-1] if events else {}
        record["latest_marker_kind"] = last.get("kind")
        record["latest_marker_ts"] = last.get("ts")
    return record


def build_report(statuses: tuple[str, ...]) -> dict:
    """Collect report rows for every task in the requested statuses.

    Rows are grouped by the AUTHORITATIVE status ``get_task`` resolves at
    read time, not the folder the listing pass found them in — concurrent
    sessions ``git mv`` task folders between the two reads, and a stale
    registry can disagree with the filesystem. A row whose authoritative
    status moved outside the requested scope mid-scan is dropped (snapshot
    semantics); duplicates across folder scans are de-duped by id.
    """
    requested = set(statuses)
    tasks: dict[str, list[dict]] = {status: [] for status in statuses}
    seen: set[int] = set()
    for status in statuses:
        for row in list_by_status(status, limit=10_000):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            rec = _task_record(row["id"])
            if rec["status"] in requested:
                tasks[rec["status"]].append(rec)
    return {
        "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {status: len(rows) for status, rows in tasks.items()},
        "tasks": tasks,
    }


def _date(ts: str | None) -> str:
    return ts[:10] if ts else "unknown"


def _age(ts: str | None, now: datetime) -> str:
    if not ts:
        return "no events"
    try:
        then = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return "unparseable ts"
    delta = now - then
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() // 60)}m ago"
    if hours < 48:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.0f}d ago"


def render_markdown(report: dict) -> str:
    """Pre-sorted skeleton of the Mode 1 structured report.

    Theme grouping ("By theme") and pod cross-references are the PM
    persona's job at read time; this skeleton supplies the sorted lists
    and recency subsections.
    """
    now = datetime.now(tz=UTC)
    tasks = report["tasks"]
    out: list[str] = [f"# PM queue report — {report['generated_at']}", ""]

    out.append("## Active work")
    for status in ACTIVE_STATUSES:
        rows = tasks.get(status, [])
        if not rows:
            continue
        out.append(f"\n### {status} ({len(rows)})")
        for r in rows:
            marker = r.get("latest_marker_kind") or "no events"
            out.append(
                f"- #{r['id']} — {r['title']} | {marker}, {_age(r.get('latest_marker_ts'), now)}"
            )

    ap = sorted(
        tasks.get("awaiting_promotion", []),
        key=lambda r: r.get("status_arrival_ts") or "",
        reverse=True,
    )
    out.append(f"\n## Awaiting promotion ({len(ap)})")
    out.append("\n### Most recent")
    for r in ap[:5]:
        out.append(f"- #{r['id']} — {r['title']} — arrived {_date(r.get('status_arrival_ts'))}")
    out.append("\n### By theme (PM groups these at read time)")
    for r in ap:
        out.append(f"- #{r['id']} — {r['title']}")

    proposed = sorted(
        tasks.get("proposed", []),
        key=lambda r: r.get("created_ts") or "",
        reverse=True,
    )
    out.append(f"\n## Proposed queue ({len(proposed)})")
    out.append("\n### Recently filed")
    for r in proposed[:10]:
        out.append(f"- #{r['id']} — {r['title']} — filed {_date(r.get('created_ts'))}")
    out.append("\n### By theme (PM groups these at read time)")
    for r in proposed:
        out.append(f"- #{r['id']} — {r['title']} — filed {_date(r.get('created_ts'))}")

    # `on_hold` is a parking lot — deliberately kept OUT of the queue report
    # (the whole point is to remove these from the active queue's sightline).
    extras = [
        s for s in tasks if s not in (*ACTIVE_STATUSES, "awaiting_promotion", "proposed", "on_hold")
    ]
    for status in extras:
        rows = tasks[status]
        if not rows:
            continue
        out.append(f"\n## {status} ({len(rows)})")
        for r in rows:
            out.append(f"- #{r['id']} — {r['title']}")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--status",
        action="append",
        choices=list(STATUSES),
        default=None,
        help="restrict to this status (repeatable; overrides the default non-terminal scope)",
    )
    parser.add_argument(
        "--include-terminal",
        action="store_true",
        help="add completed + archived to the default scope",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="emit a pre-sorted markdown skeleton instead of JSON",
    )
    args = parser.parse_args(argv)

    if args.status and args.include_terminal:
        parser.error(
            "--include-terminal cannot be combined with --status; "
            "pass --status completed / --status archived explicitly"
        )

    if args.status:
        statuses = tuple(dict.fromkeys(args.status))  # de-dupe, keep order
    elif args.include_terminal:
        statuses = STATUSES
    else:
        statuses = DEFAULT_STATUSES

    report = build_report(statuses)
    if args.markdown:
        sys.stdout.write(render_markdown(report))
    else:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
