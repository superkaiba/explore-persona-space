#!/usr/bin/env python3
"""task.py — CLI for the repo-native task workflow.

Drop-in API-compatible replacement for scripts/sagan_state.py. Same
subcommand surface, same flags. All state lives in `tasks/` instead of
Sagan's Postgres.

Subcommands (see `task.py --help`):

    view <N>
    new --kind <k> --title "..." [--body|--body-file ...] [--parent N] [--status proposed]
    set-status <N> <status> [--note ...]
    post-marker <N> <marker> [--note ...]              # alias: post-event
    list-by-status [--status ...] [--limit N]
    list-markers <N> [--prefix epm:] [--json]
    latest-marker <N>                                  # alias: latest-event
    set-body <N> --body "..." | --file path           # snapshots old → original-body.md
    set-title <N> "..."
    set-clean-result <N>
    add-tag <N> <tag>
    remove-tag <N> <tag>
    promote <N> useful|not-useful
    new-plan-version <N> --file path
    find <N>
    audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable without `uv run` plumbing.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.task_workflow import (  # noqa: E402
    STATUSES,
    NewTaskRequest,
    add_tag,
    audit,
    create_task,
    find_task_path,
    get_task,
    latest_event,
    list_by_status,
    list_events,
    new_plan_version,
    post_event,
    promote,
    remove_tag,
    set_body,
    set_clean_result,
    set_status,
    set_title,
)

# ─── Subcommand handlers ──────────────────────────────────────────────────


def cmd_view(args: argparse.Namespace) -> None:
    task = get_task(args.number)
    print(f"# task #{task['id']} — {task['frontmatter'].get('title', '')}")
    print(f"  path:    {task['path']}")
    print(f"  status:  {task['status']}")
    print(f"  kind:    {task['frontmatter'].get('kind', '')}")
    print(f"  tags:    {task['frontmatter'].get('tags') or []}")
    parent = task["frontmatter"].get("parent_id")
    if parent:
        print(f"  parent:  #{parent}")
    print(f"  clean-result: {bool(task['frontmatter'].get('has_clean_result'))}")
    print()
    events = list_events(task["id"])
    print(f"## Last {min(10, len(events))} events of {len(events)}")
    for ev in events[-10:]:
        note = ev.get("note", "")
        note = (note[:80] + "…") if len(note) > 80 else note
        print(f"  {ev['ts']}  {ev['kind']:30s}  {note}")


def cmd_create(args: argparse.Namespace) -> None:
    body = ""
    if args.body:
        body = args.body
    elif args.body_file:
        body = Path(args.body_file).read_text()
    req = NewTaskRequest(
        kind=args.kind,
        title=args.title,
        body=body,
        parent_id=args.parent,
        tags=list(args.tag) if args.tag else None,
        status=args.status,
    )
    new_id = create_task(req)
    print(f"#{new_id}")


def cmd_set_status(args: argparse.Namespace) -> None:
    path = set_status(args.number, args.status, note=args.note)
    print(str(path.relative_to(path.parents[2])))  # tasks/<status>/<id>


def cmd_post_event(args: argparse.Namespace) -> None:
    payload = post_event(
        args.number,
        args.marker,
        version=args.version,
        by=args.by,
        note=args.note,
    )
    print(json.dumps(payload, indent=2))


def cmd_list_by_status(args: argparse.Namespace) -> None:
    if args.status:
        rows = list_by_status(args.status, limit=args.limit)
    else:
        rows = []
        for status in STATUSES:
            rows.extend(list_by_status(status, limit=args.limit))
            if len(rows) >= args.limit:
                rows = rows[: args.limit]
                break
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"{'ID':>5}  {'STATUS':<22}  {'KIND':<12}  TITLE")
    for row in rows:
        print(f"{row['id']:>5}  {row['status']:<22}  {row['kind']:<12}  {row['title']}")


def cmd_list_markers(args: argparse.Namespace) -> None:
    events = list_events(args.number)
    if args.prefix:
        events = [e for e in events if e["kind"].startswith(args.prefix)]
    if args.json:
        print(json.dumps(events, indent=2))
        return
    for ev in events:
        note = ev.get("note", "")
        note = (note[:80] + "…") if len(note) > 80 else note
        print(f"{ev['ts']}  {ev['kind']:30s}  {note}")


def cmd_latest_marker(args: argparse.Namespace) -> None:
    ev = latest_event(args.number, prefix=args.prefix)
    if ev is None:
        print("(no events)")
        return
    print(json.dumps(ev, indent=2))


def cmd_set_body(args: argparse.Namespace) -> None:
    if args.body is not None:
        new_body = args.body
    elif args.file:
        new_body = Path(args.file).read_text()
    else:
        new_body = sys.stdin.read()
    set_body(args.number, new_body, snapshot_original=args.snapshot)
    print("ok")


def cmd_set_title(args: argparse.Namespace) -> None:
    set_title(args.number, args.title)
    print("ok")


def cmd_set_clean_result(args: argparse.Namespace) -> None:
    set_clean_result(args.number, value=not args.unset)
    print("ok")


def cmd_add_tag(args: argparse.Namespace) -> None:
    add_tag(args.number, args.tag)
    print("ok")


def cmd_remove_tag(args: argparse.Namespace) -> None:
    remove_tag(args.number, args.tag)
    print("ok")


def cmd_promote(args: argparse.Namespace) -> None:
    new_path = promote(args.number, args.verdict)
    print(str(new_path))


def cmd_new_plan_version(args: argparse.Namespace) -> None:
    if args.file:
        plan_md = Path(args.file).read_text()
    else:
        plan_md = sys.stdin.read()
    v = new_plan_version(args.number, plan_md)
    rel = (
        find_task_path(args.number).relative_to(Path.cwd())
        if False
        else f"tasks/<status>/{args.number}/plans/v{v}.md"
    )
    print(f"Plan v{v} written → https://eps.superkaiba.com/tasks/{args.number}/plan")
    print(f"  ({rel})", file=sys.stderr)


def cmd_find(args: argparse.Namespace) -> None:
    path = find_task_path(args.number)
    print(str(path))


def cmd_audit(args: argparse.Namespace) -> None:
    problems = audit()
    if not problems:
        print("AUDIT PASS — registry and filesystem agree")
        return
    print(f"AUDIT FAIL — {len(problems)} problem(s):")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)


# ─── Argparse wiring ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("view", help="show task summary + recent events")
    p.add_argument("number", type=int)
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("new", help="create a new task")
    p.add_argument("--kind", required=True, choices=["experiment", "infra", "analysis", "survey"])
    p.add_argument("--title", required=True)
    body_group = p.add_mutually_exclusive_group()
    body_group.add_argument("--body", help="body text directly")
    body_group.add_argument("--body-file", help="path to body file")
    p.add_argument("--parent", type=int, default=None, help="parent task id (optional)")
    p.add_argument("--tag", action="append", default=[], help="tag (repeatable)")
    p.add_argument("--status", default="proposed", choices=STATUSES)
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("set-status", help="move task to a new status (git mv + commit)")
    p.add_argument("number", type=int)
    p.add_argument("status", choices=STATUSES)
    p.add_argument("--note", default=None)
    p.set_defaults(func=cmd_set_status)

    for name in ("post-marker", "post-event"):
        p = sub.add_parser(name, help="append an event to events.jsonl")
        p.add_argument("number", type=int)
        p.add_argument("marker", help="marker kind, e.g. epm:plan, epm:reviewer-verdict")
        p.add_argument("--note", default=None)
        p.add_argument("--version", type=int, default=1)
        p.add_argument("--by", default="unknown")
        p.set_defaults(func=cmd_post_event)

    p = sub.add_parser("list-by-status", help="list tasks in a status (or all)")
    p.add_argument("--status", default=None, choices=list(STATUSES))
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list_by_status)

    p = sub.add_parser("list-markers", help="list events on a task")
    p.add_argument("number", type=int)
    p.add_argument("--prefix", default="epm:")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list_markers)

    for name in ("latest-marker", "latest-event"):
        p = sub.add_parser(name, help="show most recent event on a task")
        p.add_argument("number", type=int)
        p.add_argument("--prefix", default=None, help="restrict to events with this prefix")
        p.set_defaults(func=cmd_latest_marker)

    p = sub.add_parser("set-body", help="replace body.md content (preserves frontmatter)")
    p.add_argument("number", type=int)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--body", default=None)
    g.add_argument("--file", default=None)
    p.add_argument(
        "--snapshot", action="store_true", help="save current body to original-body.md first"
    )
    p.set_defaults(func=cmd_set_body)

    p = sub.add_parser("set-title", help="update task title (frontmatter)")
    p.add_argument("number", type=int)
    p.add_argument("title")
    p.set_defaults(func=cmd_set_title)

    p = sub.add_parser(
        "set-clean-result", help="flip has_clean_result=true (or false with --unset)"
    )
    p.add_argument("number", type=int)
    p.add_argument("--unset", action="store_true")
    p.set_defaults(func=cmd_set_clean_result)

    p = sub.add_parser("add-tag", help="add a tag to frontmatter")
    p.add_argument("number", type=int)
    p.add_argument("tag")
    p.set_defaults(func=cmd_add_tag)

    p = sub.add_parser("remove-tag", help="remove a tag from frontmatter")
    p.add_argument("number", type=int)
    p.add_argument("tag")
    p.set_defaults(func=cmd_remove_tag)

    p = sub.add_parser("promote", help="USER-ONLY: awaiting_promotion → completed")
    p.add_argument("number", type=int)
    p.add_argument("verdict", choices=["useful", "not-useful"])
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("new-plan-version", help="append plans/v{next}.md")
    p.add_argument("number", type=int)
    p.add_argument("--file", default=None, help="path to plan markdown (else stdin)")
    p.set_defaults(func=cmd_new_plan_version)

    p = sub.add_parser("find", help="print absolute path of task N's folder")
    p.add_argument("number", type=int)
    p.set_defaults(func=cmd_find)

    p = sub.add_parser("audit", help="validate REGISTRY.json against filesystem")
    p.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
