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
    comment_add,
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
from explore_persona_space.task_workflow_why_gate import (  # noqa: E402
    MIN_WHY_LINE_CHARS,
    WHY_GATED_KINDS,
    WHY_LINE_LABELS,
    find_why_section,
)

# ─── Subcommand handlers ──────────────────────────────────────────────────


def cmd_view(args: argparse.Namespace) -> None:
    task = get_task(args.number)
    events = list_events(task["id"])
    if args.json:
        payload = {
            "id": task["id"],
            "path": task["path"],
            "status": task["status"],
            "frontmatter": task["frontmatter"],
            "body": task["body"],
            "events": events,
            "n_events": len(events),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if args.rich:
        _print_rich_view(task, events)
        return
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
    print(f"## Last {min(10, len(events))} events of {len(events)}")
    for ev in events[-10:]:
        note = ev.get("note", "")
        note = (note[:80] + "…") if len(note) > 80 else note
        print(f"  {ev['ts']}  {ev['kind']:30s}  {note}")


def _print_rich_view(task: dict, events: list[dict]) -> None:
    """Terminal-friendly one-page summary (≤60 lines) for `view --rich <N>`.

    Sections (in order):
      1. Status         — canonical status from the parent folder.
      2. Frontmatter    — key/value block from body.md frontmatter.
      3. Body excerpt   — first 30 lines of the body.
      4. Last 5 events  — most recent rows from events.jsonl.
      5. Latest reviewer verdict (optional) — most recent
         `epm:clean-result-critique` marker's verdict line.

    Designed to fit in one terminal screen without scrolling. Truncates
    notes to keep within ~60 lines total.
    """
    fm = task["frontmatter"]
    print(f"# task #{task['id']} — {fm.get('title', '')}")
    print()
    # 1. Status
    print(f"Status: {task['status']}")
    print(f"  path: {task['path']}")
    print()
    # 2. Frontmatter (key fields only — exclude the bulky `title` we already
    # printed and any nested structures that would overflow).
    print("Frontmatter:")
    for key in ("kind", "parent_id", "tags", "has_clean_result", "classification", "created_at"):
        if key in fm and fm[key] not in (None, [], ""):
            print(f"  {key}: {fm[key]}")
    print()
    # 3. Body excerpt — first 30 lines.
    body_lines = task["body"].splitlines()
    excerpt = body_lines[:30]
    print(f"Body excerpt ({len(excerpt)} of {len(body_lines)} lines):")
    for line in excerpt:
        # Truncate any one body line to ~110 chars so we don't blow up.
        print(f"  {line[:110]}")
    print()
    # 4. Last 5 events.
    last_n = min(5, len(events))
    print(f"Last {last_n} events (of {len(events)}):")
    for ev in events[-5:]:
        note = ev.get("note", "")
        # First line of note, truncated.
        first_line = note.splitlines()[0] if note else ""
        first_line = (first_line[:80] + "…") if len(first_line) > 80 else first_line
        print(f"  {ev['ts']}  {ev['kind']:30s}  {first_line}")
    # 5. Latest reviewer verdict (optional).
    critique_events = [e for e in events if e["kind"] == "epm:clean-result-critique"]
    if critique_events:
        latest = critique_events[-1]
        note = latest.get("note", "")
        # Find the first line that contains a verdict marker.
        verdict_line = ""
        for line in note.splitlines():
            stripped = line.strip()
            if stripped and (
                "verdict" in stripped.lower()
                or stripped.startswith("Round ")
                or "PASS" in stripped
                or "FAIL" in stripped
            ):
                verdict_line = stripped
                break
        if not verdict_line:
            verdict_line = note.splitlines()[0] if note else "(no note)"
        verdict_line = (verdict_line[:90] + "…") if len(verdict_line) > 90 else verdict_line
        print()
        print(f"Latest reviewer verdict: {verdict_line}")


def cmd_create(args: argparse.Namespace) -> None:
    body = ""
    if args.body:
        body = args.body
    elif args.body_file:
        body = Path(args.body_file).read_text()
    # `## Why this experiment` gate (CLAUDE.md / workflow.yaml § gates).
    # The PM session and /ideation/proposer paths drive the four-line
    # interrogation in chat; this CLI guard catches manual `task.py new`
    # invocations that try to bypass it.
    _enforce_why_this_experiment_gate(kind=args.kind, body=body)
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


def _enforce_why_this_experiment_gate(*, kind: str, body: str) -> None:
    """Reject `task.py new` when an experiment/survey/infra body lacks a
    complete `## Why this experiment` section.

    The check is intentionally mechanical and conservative — the
    substantive check (does the answer actually name a concrete
    decision?) is the job of the `/why-experiment-gate` skill prompt,
    which runs in chat with an LLM. Here we only ensure the section
    exists and each labeled line carries non-trivial substance, so the
    "I'll just type it manually" bypass doesn't escape with empty
    placeholders.

    Constants + section walker live in
    ``explore_persona_space.task_workflow_why_gate`` — same source of
    truth ``scripts/verify_task_body.py`` check #12 reads from, so the
    gate's mechanical surface cannot drift between the two call sites.
    """
    if kind not in WHY_GATED_KINDS:
        return

    section = find_why_section(body)
    seen_labels: dict[str, str] = {}
    if section is not None:
        seen_labels = {label: val for label, val in section.line_values.items() if val is not None}

    missing = [label for label in WHY_LINE_LABELS if label not in seen_labels]
    stubby = [label for label, value in seen_labels.items() if len(value) < MIN_WHY_LINE_CHARS]
    if not missing and not stubby:
        return

    msg_lines = [
        f"ERROR: {kind} tasks require `## Why this experiment` (4 lines).",
        "This section must be filled by an interrogating agent, not written from scratch.",
        "Use the PM session, `/ideation`, or `/experiment-proposer` to draft it.",
        "",
    ]
    if missing:
        msg_lines.append(f"  Missing labeled lines: {', '.join(missing)}")
    if stubby:
        lengths = ", ".join(
            f"`{label}` ({len(seen_labels[label])} chars, need ≥{MIN_WHY_LINE_CHARS})"
            for label in stubby
        )
        msg_lines.append(f"  Stubby labeled lines: {lengths}")
    print("\n".join(msg_lines), file=sys.stderr)
    sys.exit(2)


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
    plan_md = Path(args.file).read_text() if args.file else sys.stdin.read()
    v = new_plan_version(args.number, plan_md)
    rel = f"tasks/<status>/{args.number}/plans/v{v}.md"
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


def cmd_migrate_body(args: argparse.Namespace) -> None:
    """`task.py migrate-body` — patch awaiting_promotion bodies to verify_task_body PASS.

    Three modes:
      --report                  classification table over all awaiting_promotion bodies
      --dry-run <N> | --all     show proposed patches (no writes)
      --apply <N>  | --all      write patches (snapshots original-body.md on v4-legacy)
    """
    # Lazy import to keep `task.py --help` fast and to avoid the migrate
    # module loading verify_task_body on every CLI invocation.
    from explore_persona_space.task_workflow_migrate import (
        BodyClass,
        list_awaiting_promotion_ids,
        migrate_one,
    )

    # Determine target ids
    if args.all or args.report:
        target_ids = list_awaiting_promotion_ids()
    else:
        if args.number is None:
            print(
                "task.py migrate-body: must pass <N> or --all (or --report)",
                file=sys.stderr,
            )
            sys.exit(2)
        target_ids = [args.number]

    if args.report:
        print(f"{'ID':<7}  {'CLASS':<22}  before -> after")
        print("─" * 64)
        for tid in target_ids:
            # Classification-only — no apply, no patch.
            try:
                result = migrate_one(tid, apply=False, shape=args.shape, verbose=False)
            except FileNotFoundError as e:
                print(f"#{tid:<5}  (error: {e})")
                continue
            print(result.report_line())
        return

    # dry-run / apply path
    n_changed = 0
    n_needs_user = 0
    n_skip = 0
    for tid in target_ids:
        try:
            result = migrate_one(
                tid,
                apply=args.apply,
                shape=args.shape,
                verbose=args.verbose,
            )
        except FileNotFoundError as e:
            print(f"#{tid}: ERROR — {e}", file=sys.stderr)
            continue
        # Render
        if result.classification in (BodyClass.PASS, BodyClass.LEGACY_HTML):
            n_skip += 1
            if args.verbose:
                print(f"#{tid}: skip ({result.classification.value})")
            continue
        print(result.report_line())
        for action in result.actions:
            print(f"    - {action}")
        if result.needs_user:
            n_needs_user += 1
            print(f"    [needs-user] {result.needs_user_reason}")
        else:
            n_changed += 1
        if args.verbose and result.diff_preview:
            print("    ─── diff preview ───")
            for line in result.diff_preview.splitlines()[:30]:
                print(f"    {line}")
            print()

    print()
    verb = "applied" if args.apply else "dry-run"
    print(
        f"task.py migrate-body — {verb}: {n_changed} changed, "
        f"{n_needs_user} needs-user, {n_skip} skipped"
    )


def cmd_comment_add(args: argparse.Namespace) -> None:
    comment = comment_add(
        task_n=args.number,
        author=args.author,
        body_md=args.body_md,
        thread_id=args.thread_id,
        reply_to=args.reply_to,
        source=args.source,
    )
    print(json.dumps(comment, ensure_ascii=False))


# ─── Argparse wiring ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("view", help="show task summary + recent events")
    p.add_argument("number", type=int)
    p.add_argument(
        "--json",
        action="store_true",
        help="emit full frontmatter + body + all events as JSON (for pipelines)",
    )
    p.add_argument(
        "--rich",
        action="store_true",
        help=(
            "terminal-friendly one-page summary (≤60 lines): status, "
            "frontmatter, body excerpt, last 5 events, latest reviewer verdict"
        ),
    )
    p.set_defaults(func=cmd_view)

    # `new` is the preferred name; `create-experiment` is a sagan_state.py
    # compatibility alias so agent specs that still spell it that way work.
    for name in ("new", "create-experiment"):
        p = sub.add_parser(name, help="create a new task")
        p.add_argument(
            "--kind",
            required=False,
            default="experiment",
            choices=["experiment", "infra", "analysis", "survey"],
        )
        p.add_argument("--title", required=True)
        body_group = p.add_mutually_exclusive_group()
        body_group.add_argument("--body", help="body text directly")
        body_group.add_argument("--body-file", help="path to body file")
        p.add_argument("--parent", type=int, default=None, help="parent task id (optional)")
        p.add_argument("--tag", action="append", default=[], help="tag (repeatable)")
        p.add_argument("--status", default="proposed", choices=STATUSES)
        # Sagan-compatibility: accept --runpod-account but ignore it.
        p.add_argument("--runpod-account", default=None, help="(ignored; Sagan compat)")
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

    p = sub.add_parser(
        "set-body",
        help="replace body.md body content (frontmatter is preserved, NOT replaced)",
        description=(
            "Replace the body portion of body.md while preserving the existing "
            "YAML frontmatter verbatim. The new content passed via --body, --file, "
            "or stdin is written AS-IS into the body region (after the closing "
            "`---` line). If the new content itself begins with `---\\n...\\n---\\n`, "
            "those lines become literal body text — set-body does NOT parse them as "
            "frontmatter and does NOT update any frontmatter field. To change a "
            "frontmatter field, use the dedicated mutators (`set-title`, "
            "`set-clean-result`, `add-tag`, `remove-tag`) or edit body.md directly."
        ),
    )
    p.add_argument("number", type=int)
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--body", default=None, help="new body content as a string (excludes frontmatter)"
    )
    g.add_argument("--file", default=None, help="path to a file containing the new body content")
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

    p = sub.add_parser(
        "comment-add",
        help="append a comment to tasks/<status>/<N>/comments.jsonl",
    )
    p.add_argument("number", type=int)
    p.add_argument("--author", required=True, choices=["user", "claude", "codex"])
    p.add_argument("--body-md", dest="body_md", required=True)
    p.add_argument("--thread-id", dest="thread_id", default=None)
    p.add_argument("--reply-to", dest="reply_to", default=None)
    p.add_argument(
        "--source",
        default=None,
        help="audit-log source, e.g. 'sagan-user:<session-id>' or 'cli'",
    )
    p.set_defaults(func=cmd_comment_add)

    p = sub.add_parser(
        "migrate-body",
        help="patch awaiting_promotion bodies into verify_task_body compliance",
        description=(
            "Migrate awaiting_promotion task bodies to the markdown clean-result spec "
            "(verify_task_body.py 11-check). Conformant-but-failing bodies are patched "
            "in place (Repro subgroups, cherry-picked label, qualitative-data link); "
            "v4-legacy bodies (## TL;DR / ## Summary / ## Details / ## Source issues) "
            "are converted to the four-H2 target shape (TL;DR / Figure / Details / "
            "Reproducibility). HTML bodies carrying <!-- legacy-sagan-card --> are "
            "grandfathered and skipped."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--report",
        action="store_true",
        help="print a classification table for every awaiting_promotion body",
    )
    mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="(default) show what would change without writing",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="write the patched body via task_workflow.set_body (commits per body)",
    )
    p.add_argument(
        "number",
        nargs="?",
        type=int,
        default=None,
        help="task number to migrate (omit when using --all or --report)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="operate on every body in tasks/awaiting_promotion/",
    )
    p.add_argument(
        "--shape",
        choices=["v4-to-new", "conformant-failing"],
        default=None,
        help="force a specific patch chain (overrides auto-classification)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print a unified-diff preview after each body",
    )
    p.set_defaults(func=cmd_migrate_body)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
