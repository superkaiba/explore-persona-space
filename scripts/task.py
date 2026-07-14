#!/usr/bin/env python3
"""task.py — CLI for the repo-native task workflow.

Drop-in API-compatible replacement for scripts/sagan_state.py. Same
subcommand surface, same flags. All state lives in `tasks/` instead of
Sagan's Postgres.

Subcommands (see `task.py --help`):

    view <N>
    new --kind <k> --title "..." [--body|--body-file ...] [--goal "..."] [--parent N]
        [--origin-prompt "..."] [--status proposed]
    set-status <N> <status> [--note ...]
    set-kind <N> <kind>                                # reclassify a misfiled task kind
    post-marker <N> <marker> [--note ... | --file path]   # alias: post-event
    list-by-status [--status ...] [--limit N]
    list-children <N> [--json]                         # tasks with parent_id == N
    list-markers <N> [--prefix epm:] [--json]
    latest-marker <N>                                  # alias: latest-event
    set-body <N> --body "..." | --file path           # snapshots old → original-body.md
    set-title <N> "..."
    set-goal <N> "..." [--by user|clarifier|planner] [--reason ...]
    set-clean-result <N>
    add-tag <N> <tag>
    remove-tag <N> <tag>
    promote <N> useful|not-useful
    new-plan-version <N> --file path
    raise-concern <N> --concern-id <id> --severity BLOCKER|CONCERN|NIT
                     --summary "..." --by <reviewer> --round <int> [--evidence ...]
    address-concern <N> --concern-id <id> --by <implementer> --round <int> [--summary ...]
    defer-concern <N> --concern-id <id> --by user|reconciler --rationale "..."
    list-concerns <N> [--open-only] [--json]
    find <N>
    tasks-dir
    audit
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import os
import re
import sys
from pathlib import Path

# Make the package importable without `uv run` plumbing.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.task_workflow import (  # noqa: E402
    CONCERN_SEVERITIES,
    KINDS,
    STATUSES,
    WORKFLOW_VERSIONS,
    GoalH2DropError,
    NewTaskRequest,
    ReconcileReport,
    add_tag,
    address_concern,
    audit,
    create_task,
    defer_concern,
    find_task_path,
    get_task,
    is_paper_task,
    latest_event,
    list_by_status,
    list_children,
    list_concerns,
    list_events,
    new_plan_version,
    post_event,
    promote,
    raise_concern,
    reconcile_registry,
    remove_tag,
    set_body,
    set_clean_result,
    set_goal,
    set_kind,
    set_status,
    set_title,
    set_track,
    tasks_dir,
)

# ─── Subcommand handlers ──────────────────────────────────────────────────


def _safe_echo(text: str, *, context: str) -> None:
    """Echo a post-commit confirmation without letting the echo flip the rc.

    Every mutating subcommand appends + commits BEFORE echoing its
    confirmation to stdout; the echo is cosmetic. A BrokenPipeError on the
    echo (caller tore the pipe down early — Bash-tool teardown, `| head`,
    dead SSH) must NOT flip the exit code to nonzero: callers treat rc!=0
    as "mutation failed" and retry, duplicating the mutation (incident
    #537, 2026-06-10 — codex_task._post_marker re-posted
    epm:codex-task-spawned after a post-commit echo failure). Pre-commit
    failures raise out of the mutating API call before the echo and stay
    fatal. ``context`` names the subcommand for the stderr notice.
    """
    try:
        print(text)
        sys.stdout.flush()
    except BrokenPipeError:
        print(
            f"{context}: committed; stdout echo failed (BrokenPipeError) — "
            "suppressed so the exit code reflects the commit, not the echo.",
            file=sys.stderr,
        )
        # Point stdout at devnull so the interpreter-shutdown flush of the
        # broken pipe can't raise again and flip the exit status after the
        # commit landed. Best-effort only: when stdout has no real fileno
        # (pytest capture), the echo is already abandoned either way.
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull_fd, sys.stdout.fileno())
            finally:
                os.close(devnull_fd)
        except Exception:
            pass


_FIELD_LED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*[:=]")


def _looks_field_led(note: str) -> bool:
    """True when the note HEAD is a `field:` / `field=` line-core after
    stripping the same leading whitespace/bullet/bold mix
    task_workflow.parse_followup_note_field strips per segment. Head-only
    by design (false-positive-averse; see #1178 plan §4 D2)."""
    core = re.sub(r"^[\s\-*]+", "", note)
    return bool(_FIELD_LED_RE.match(core))


def _safe_print(*args: object, context: str = "task.py", **kwargs: object) -> None:
    """``print()`` that swallows BrokenPipeError on a torn stdout pipe.

    Read-only commands emit multi-line output a downstream ``| head`` /
    ``| grep -q .`` routinely closes early; the resulting BrokenPipeError must
    NOT surface as a traceback / nonzero exit — a torn pipe is not a failure
    for a read-only command. Prints a one-line suppression notice to stderr and
    redirects stdout to devnull so the interpreter-shutdown flush can't re-raise
    and flip the exit code. Generalizes ``_safe_echo`` for arbitrary args.

    Includes ``sys.stdout.flush()`` INSIDE the ``try`` (mirroring
    ``_safe_echo``): a buffered-pipe BrokenPipeError often only surfaces at
    flush/close, not at the ``print`` call, so the flush is what makes the
    guard actually catch it. ``context`` names the subcommand for the notice.
    """
    try:
        print(*args, **kwargs)
        sys.stdout.flush()
    except BrokenPipeError:
        print(
            f"{context}: stdout closed early (BrokenPipeError) — suppressed.",
            file=sys.stderr,
        )
        # Point stdout at devnull so the interpreter-shutdown flush of the
        # broken pipe can't raise again and flip the exit status. Best-effort:
        # when stdout has no real fileno (pytest capture) the print is already
        # abandoned either way.
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull_fd, sys.stdout.fileno())
            finally:
                os.close(devnull_fd)
        except Exception:
            pass


def cmd_view(args: argparse.Namespace) -> None:
    task = get_task(args.number)
    events = list_events(task["id"])
    goal_value = task["frontmatter"].get("goal")
    goal_value = goal_value.strip() if isinstance(goal_value, str) and goal_value.strip() else None
    if args.json:
        payload = {
            "id": task["id"],
            "path": task["path"],
            "status": task["status"],
            "goal": goal_value,
            "frontmatter": task["frontmatter"],
            "body": task["body"],
            "events": events,
            "n_events": len(events),
        }
        _safe_print(json.dumps(payload, indent=2, ensure_ascii=False), context="task.py view")
        return
    if args.rich:
        _print_rich_view(task, events)
        return
    _safe_print(
        f"# task #{task['id']} — {task['frontmatter'].get('title', '')}", context="task.py view"
    )
    _safe_print(f"  path:    {task['path']}", context="task.py view")
    _safe_print(f"  status:  {task['status']}", context="task.py view")
    _safe_print(f"  kind:    {task['frontmatter'].get('kind', '')}", context="task.py view")
    _safe_print(f"  tags:    {task['frontmatter'].get('tags') or []}", context="task.py view")
    parent = task["frontmatter"].get("parent_id")
    if parent:
        _safe_print(f"  parent:  #{parent}", context="task.py view")
    _safe_print(
        f"  clean-result: {bool(task['frontmatter'].get('has_clean_result'))}",
        context="task.py view",
    )
    if goal_value:
        _safe_print(f"  goal:    {goal_value}", context="task.py view")
    _safe_print(context="task.py view")
    _safe_print(f"## Last {min(10, len(events))} events of {len(events)}", context="task.py view")
    for ev in events[-10:]:
        note = ev.get("note", "")
        note = (note[:80] + "…") if len(note) > 80 else note
        _safe_print(f"  {ev['ts']}  {ev['kind']:30s}  {note}", context="task.py view")


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
    _safe_print(f"# task #{task['id']} — {fm.get('title', '')}", context="task.py view")
    _safe_print(context="task.py view")
    # 1. Status
    _safe_print(f"Status: {task['status']}", context="task.py view")
    _safe_print(f"  path: {task['path']}", context="task.py view")
    _safe_print(context="task.py view")
    # 2. Frontmatter (key fields only — exclude the bulky `title` we already
    # printed and any nested structures that would overflow).
    _safe_print("Frontmatter:", context="task.py view")
    for key in ("kind", "parent_id", "tags", "has_clean_result", "classification", "created_at"):
        if key in fm and fm[key] not in (None, [], ""):
            _safe_print(f"  {key}: {fm[key]}", context="task.py view")
    _safe_print(context="task.py view")
    # 3. Body excerpt — first 30 lines.
    body_lines = task["body"].splitlines()
    excerpt = body_lines[:30]
    _safe_print(
        f"Body excerpt ({len(excerpt)} of {len(body_lines)} lines):", context="task.py view"
    )
    for line in excerpt:
        # Truncate any one body line to ~110 chars so we don't blow up.
        _safe_print(f"  {line[:110]}", context="task.py view")
    _safe_print(context="task.py view")
    # 4. Last 5 events.
    last_n = min(5, len(events))
    _safe_print(f"Last {last_n} events (of {len(events)}):", context="task.py view")
    for ev in events[-5:]:
        note = ev.get("note", "")
        # First line of note, truncated.
        first_line = note.splitlines()[0] if note else ""
        first_line = (first_line[:80] + "…") if len(first_line) > 80 else first_line
        _safe_print(f"  {ev['ts']}  {ev['kind']:30s}  {first_line}", context="task.py view")
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
        _safe_print(context="task.py view")
        _safe_print(f"Latest reviewer verdict: {verdict_line}", context="task.py view")


def cmd_create(args: argparse.Namespace) -> None:
    body = ""
    if args.body:
        body = args.body
    elif args.body_file:
        body = Path(args.body_file).read_text()
    goal_value: str | None = (args.goal or "").strip() or None
    if goal_value and args.kind != "experiment":
        print(
            f"warning: --goal is only honored for kind=experiment "
            f"(got kind={args.kind!r}); ignoring.",
            file=sys.stderr,
        )
        goal_value = None
    req = NewTaskRequest(
        kind=args.kind,
        title=args.title,
        body=body,
        parent_id=args.parent,
        tags=list(args.tag) if args.tag else None,
        status=args.status,
        goal=goal_value,
        origin_prompt=(args.origin_prompt or "").strip() or None,
        workflow=getattr(args, "workflow", None),
    )
    new_id = create_task(req)
    # Track: explicit --track wins; otherwise derive a human track from the
    # human kinds so CLI-created think/read tasks land in the Human board.
    human_kinds = {"note", "reading", "idea", "question", "decision"}
    track = getattr(args, "track", None) or ("human" if args.kind in human_kinds else None)
    if track:
        set_track(new_id, track)
    _safe_echo(f"#{new_id}", context="task.py new")


def cmd_set_goal(args: argparse.Namespace) -> None:
    changed = set_goal(args.number, args.goal, by=args.by, reason=args.reason)
    if changed:
        _safe_echo(
            f"ok — goal updated for #{args.number} (by={args.by})", context="task.py set-goal"
        )
    else:
        _safe_echo(
            f"ok — goal unchanged for #{args.number} (idempotent no-op)",
            context="task.py set-goal",
        )


def _status_error_message(bad_status: str) -> str:
    """Build a helpful 'did you mean ...?' error for an invalid status value.

    argparse `choices=` rejects with a bare 'invalid choice' dump that
    callers have repeatedly misread (inventing statuses like 'uploading' /
    'api'). This guides them to the closest valid enum member and lists the
    full enum. Returns the message string; the caller raises SystemExit.
    """
    suggestions = difflib.get_close_matches(bad_status, STATUSES, n=1, cutoff=0.4)
    lines = [f"task.py set-status: invalid status {bad_status!r}."]
    if suggestions:
        lines.append(f"  did you mean {suggestions[0]!r}?")
    lines.append("  valid statuses: " + ", ".join(STATUSES))
    return "\n".join(lines)


def _resolve_autonomous_plan_gate(gpu_hours: float | None) -> tuple[str, float, bool]:
    """Decide the autonomous plan-approval gate outcome from env + gpu_hours.

    Returns ``(decision, cap, autonomous)`` where ``decision`` is one of
    ``"auto_approved" | "parked_over_cap" | "interactive_pending"``.

    Deterministic and code-enforced — reads ``EPM_AUTONOMOUS_SESSION`` +
    ``EPM_PLAN_AUTOAPPROVE_GPU_HOURS`` from the process env (the Bash tool
    inherits the claude-process env, so a spawned ``--auto`` session's vars
    are visible here). Putting the decision in code means the plan-approval
    gate no longer depends on the LLM reading a deeply-nested skill step and
    choosing to obey it over the global "ask before spending money" prior.

    FAIL SAFE: a missing/None ``gpu_hours`` parks (never auto-approves on a
    blank estimate), matching the SKILL.md Step 2c contract.
    """
    # Case-insensitive truthiness. The falsy set {"", "0", "false", "no"}
    # MUST stay identical to the AskUserQuestion PreToolUse hook in
    # .claude/settings.json (which lowercases via `tr` before comparing) so
    # the two layers never disagree on a value like "no" / "FALSE".
    _auto_raw = os.environ.get("EPM_AUTONOMOUS_SESSION", "").strip().lower()
    autonomous = _auto_raw not in ("", "0", "false", "no")
    cap_raw = os.environ.get("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "24")
    try:
        cap = float(cap_raw)
    except (TypeError, ValueError):
        cap = 24.0
    if not autonomous:
        return ("interactive_pending", cap, False)
    if gpu_hours is None or gpu_hours > cap:
        return ("parked_over_cap", cap, True)
    return ("auto_approved", cap, True)


def cmd_set_status(args: argparse.Namespace) -> None:
    if args.status not in STATUSES:
        raise SystemExit(_status_error_message(args.status))

    force_followup_exit = getattr(args, "force_followup_exit", False)

    # Autonomous plan-approval gate (code-enforced, not LLM discretion).
    # When the caller opts in via --auto-approve-if-autonomous on a
    # plan_pending transition, the decision is made HERE in the script so a
    # spawned `--auto` session deterministically auto-approves an under-cap
    # plan (or parks an over-cap / blank-estimate one) instead of relying on
    # the orchestrator to follow SKILL.md Step 2c. Interactive sessions
    # (EPM_AUTONOMOUS_SESSION unset) fall through to the normal plan_pending
    # transition unchanged.
    if getattr(args, "auto_approve_if_autonomous", False) and args.status == "plan_pending":
        # Same-issue follow-up status-hold rule (code-enforced; SKILL.md
        # Step 9b § Same-issue follow-up loop, step 3): a `followups_running`
        # task HOLDS that status for the WHOLE round. A plan-gate call
        # mid-round still FIRES the gate decision + markers but the status
        # stays in place ("the Step 2c plan-approval gate still fires, it
        # just no longer moves the status to plan_pending"). Any other
        # pipeline re-entry is refused by task_workflow.set_status below
        # unless --force-followup-exit is passed.
        followup_hold = (
            not force_followup_exit and get_task(args.number)["status"] == "followups_running"
        )
        if followup_hold:
            gpu_hours = getattr(args, "gpu_hours", None)
            decision, cap, _autonomous = _resolve_autonomous_plan_gate(gpu_hours)
            if decision == "auto_approved":
                post_event(
                    args.number,
                    "epm:plan-approved",
                    version=1,
                    by="autonomous-gate",
                    note=(
                        "Auto-approved by the code-enforced autonomous plan-gate "
                        f"(task.py --auto-approve-if-autonomous): gpu_hours_total={gpu_hours} "
                        f"<= cap {cap}. Same-issue follow-up round: status HELD at "
                        "followups_running (status-hold rule, SKILL.md Step 9b)."
                    ),
                )
            elif decision == "parked_over_cap":
                reason = (
                    "estimate missing/unparseable"
                    if gpu_hours is None
                    else f"est {gpu_hours} GPU-h exceeds {cap}h auto-approve cap"
                )
                post_event(
                    args.number,
                    "epm:awaiting-spend-approval",
                    version=1,
                    by="autonomous-gate",
                    note=(
                        f"Autonomous plan-gate parked IN PLACE at followups_running: {reason}; "
                        "awaiting user approval (status-hold rule, SKILL.md Step 9b — "
                        "the plan gate fires but the status does not move)."
                    ),
                )
            _safe_echo(
                f"PLAN_GATE_DECISION: {decision} gpu_hours={gpu_hours} cap={cap} "
                "(followups_running hold: status unchanged)",
                context="task.py set-status",
            )
            return
        gpu_hours = getattr(args, "gpu_hours", None)
        decision, cap, _autonomous = _resolve_autonomous_plan_gate(gpu_hours)
        if decision == "auto_approved":
            note = (args.note or "").strip()
            gate_note = (
                f"[autonomous plan-gate: auto-approved, est {gpu_hours} GPU-h <= {cap}h cap]"
            )
            path = set_status(
                args.number,
                "approved",
                note=(f"{note} {gate_note}" if note else gate_note),
                force_followup_exit=force_followup_exit,
            )
            post_event(
                args.number,
                "epm:plan-approved",
                version=1,
                by="autonomous-gate",
                note=(
                    "Auto-approved by the code-enforced autonomous plan-gate "
                    f"(task.py --auto-approve-if-autonomous): gpu_hours_total={gpu_hours} "
                    f"<= cap {cap}. EPM_AUTONOMOUS_SESSION set; no human asked."
                ),
            )
            _safe_echo(
                str(path.relative_to(path.parents[2])),  # tasks/<status>/<id>
                context="task.py set-status",
            )
            _safe_echo(
                f"PLAN_GATE_DECISION: auto_approved gpu_hours={gpu_hours} cap={cap}",
                context="task.py set-status",
            )
            return
        if decision == "parked_over_cap":
            path = set_status(
                args.number,
                "plan_pending",
                note=args.note,
                force_followup_exit=force_followup_exit,
            )
            reason = (
                "estimate missing/unparseable"
                if gpu_hours is None
                else f"est {gpu_hours} GPU-h exceeds {cap}h auto-approve cap"
            )
            post_event(
                args.number,
                "epm:awaiting-spend-approval",
                version=1,
                by="autonomous-gate",
                note=(
                    f"Autonomous plan-gate parked at plan_pending: {reason}; "
                    "awaiting user approval (set-status <N> approved)."
                ),
            )
            _safe_echo(
                str(path.relative_to(path.parents[2])),  # tasks/<status>/<id>
                context="task.py set-status",
            )
            _safe_echo(
                f"PLAN_GATE_DECISION: parked_over_cap gpu_hours={gpu_hours} cap={cap}",
                context="task.py set-status",
            )
            return
        # interactive_pending: fall through to the normal plan_pending move,
        # then signal the orchestrator to run the interactive approval ask.
        path = set_status(
            args.number,
            args.status,
            note=args.note,
            force_followup_exit=force_followup_exit,
        )
        _safe_echo(
            str(path.relative_to(path.parents[2])),  # tasks/<status>/<id>
            context="task.py set-status",
        )
        _safe_echo("PLAN_GATE_DECISION: interactive_pending", context="task.py set-status")
        return

    try:
        path = set_status(
            args.number,
            args.status,
            note=args.note,
            force_followup_exit=force_followup_exit,
        )
    except ValueError as exc:
        # Followup status-hold refusal (or another library-level rejection):
        # surface the message cleanly instead of a traceback.
        raise SystemExit(f"task.py set-status: {exc}") from exc
    _safe_echo(
        str(path.relative_to(path.parents[2])),  # tasks/<status>/<id>
        context="task.py set-status",
    )
    if args.status == "followups_running":
        tags = get_task(args.number)["frontmatter"].get("tags") or []
        if not {"followup-auto", "followup-manual"} & set(tags):
            _safe_echo(
                "WARNING: transitioned to followups_running without a "
                "followup-auto/followup-manual tag. Same-issue follow-up rounds "
                "MUST record the initiation mode in the same step "
                "(`task.py add-tag <N> followup-auto` for proposer-initiated, "
                "`followup-manual` for user-initiated; a bare `followup` tag does "
                "not count) — see SKILL.md Step 9b § Same-issue follow-up loop, "
                "step 2. Legacy children-in-flight transitions can ignore this.",
                context="task.py set-status",
            )


def cmd_post_event(args: argparse.Namespace) -> None:
    # Note body comes from either --note (inline string) or --file (path
    # to a file containing the body). Mutually exclusive at the argparse
    # layer; either may be omitted (post a marker with no body). File
    # input avoids the shell-quoting traps that bite multi-line / special-
    # char bodies passed via `--note "$(cat ...)"`. The 50_000-char cap
    # is enforced by `post_event` itself (raises ValueError on oversize),
    # so file-read bodies inherit it automatically.
    note = args.note
    if args.file is not None:
        # `--file -` reads the body from stdin (mirrors new-plan-version at
        # the line-705/748 pattern); without this a heredoc'd
        # `post-marker ... --file - <<EOF` crashes on `Path("-").read_text()`
        # with FileNotFoundError (daily 2026-06-24 incident, #658).
        note = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
    payload = post_event(
        args.number,
        args.marker,
        version=args.version,
        by=args.by,
        note=note,
    )
    # The marker is appended once post_event returns; on the primary checkout
    # the bookkeeping commit may be DEFERRED (#1030) — post_event then logs an
    # ERROR to stderr + a forensic row in ~/.task-workflow/
    # deferred-commits.jsonl and still returns, so rc stays 0: do NOT re-post
    # on that warning (a re-post duplicates the marker). The JSON echo below
    # is cosmetic (see _safe_echo for the rc contract + incident #537).
    # Append failures (oversize note, flock timeout, missing task), routed-
    # mode commit failures, and genuine bugs raise out of post_event above
    # and stay fatal.
    if args.note is not None and "\\n" in note and "\n" not in note and _looks_field_led(note):
        # Poster-side twin of the #1120 parse-side normalization
        # (task_workflow.parse_followup_note_field): a single physical
        # line whose separators arrived as literal backslash-n escapes is
        # almost always a shell-quoting mistake ("...\n..." instead of
        # $'...\n...'). WARN only — the marker already posted; guarded so
        # a torn-down stderr can never flip the exit code post-commit
        # (the #537 rc-contract class, see _safe_echo; a closed stderr
        # stream raises ValueError, not only OSError).
        with contextlib.suppress(OSError, ValueError):
            print(
                f"WARNING: task.py post-marker {args.marker}: --note is a single "
                "physical line containing literal \\n escape sequences with "
                "field-led content; field parsers treat REAL newlines as "
                "separators and normalize these escapes where they apply "
                "(#1120). Did you mean $'...' shell quoting, or --file for a "
                "multi-line body? Do NOT re-post this marker — it was posted "
                "successfully; fix the quoting on your NEXT marker instead.",
                file=sys.stderr,
            )
    _safe_echo(
        json.dumps(payload, indent=2),
        context=f"task.py post-marker: marker {args.marker}",
    )


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
        _safe_print(json.dumps(rows, indent=2), context="task.py list-by-status")
        return
    _safe_print(f"{'ID':>5}  {'STATUS':<22}  {'KIND':<12}  TITLE", context="task.py list-by-status")
    for row in rows:
        _safe_print(
            f"{row['id']:>5}  {row['status']:<22}  {row['kind']:<12}  {row['title']}",
            context="task.py list-by-status",
        )


def cmd_list_children(args: argparse.Namespace) -> None:
    """List tasks whose frontmatter `parent_id` == N (campaign children, child
    follow-up tasks). `--json` emits the row list verbatim (`[]` when none)."""
    rows = list_children(args.number)
    if args.json:
        _safe_print(json.dumps(rows, indent=2), context="task.py list-children")
        return
    if not rows:
        _safe_print(f"(no tasks with parent_id == {args.number})", context="task.py list-children")
        return
    _safe_print(
        f"{'ID':>5}  {'STATUS':<22}  {'KIND':<12}  {'CLEAN':<5}  TITLE",
        context="task.py list-children",
    )
    for row in rows:
        clean = "yes" if row["has_clean_result"] else "no"
        _safe_print(
            f"{row['id']:>5}  {row['status']:<22}  {row['kind']:<12}  {clean:<5}  {row['title']}",
            context="task.py list-children",
        )


def cmd_list_clean_results(args: argparse.Namespace) -> None:
    """List clean results — tasks the user has promoted from awaiting_promotion.

    Default filter: `classification ∈ {useful, not-useful}`. Tasks with
    `has_clean_result=true` but no classification are still pending
    promotion and don't count as clean results. Pass `--include-pending`
    to surface those too. `--useful-only` drops `not-useful` rows.
    Substring `--search` matches title + body, case-insensitive.
    """
    rows: list[dict[str, object]] = []
    needle = (args.search or "").strip().lower() or None
    for status in STATUSES:
        for row in list_by_status(status, limit=10_000):
            if not row.get("has_clean_result"):
                continue
            task_dir = find_task_path(row["id"])
            body_path = task_dir / "body.md"
            try:
                body_path.read_text()
            except FileNotFoundError:
                continue
            # Parse classification + promoted_at from frontmatter.
            from explore_persona_space.task_workflow import _read_body

            fm, body_only = _read_body(body_path)
            classification = fm.get("classification") or "pending"
            if classification not in ("useful", "not-useful", "pending"):
                classification = "pending"
            if classification == "pending" and not args.include_pending:
                continue
            if args.useful_only and classification != "useful":
                continue
            promoted_at = fm.get("promoted_at") or ""
            date = (promoted_at[:10] if promoted_at else "") or body_path.stat().st_mtime
            if needle:
                hay = f"{row['title']}\n{body_only}".lower()
                if needle not in hay:
                    continue
            rows.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "kind": row["kind"],
                    "status": row["status"],
                    "classification": classification,
                    "promoted_at": promoted_at,
                    "date": date if isinstance(date, str) else "",
                }
            )
    # Newest first by promoted_at when present.
    rows.sort(key=lambda r: r.get("promoted_at") or "", reverse=True)
    if args.limit:
        rows = rows[: args.limit]
    if args.json:
        _safe_print(json.dumps(rows, indent=2), context="task.py list-clean-results")
        return
    _safe_print(
        f"{'ID':>5}  {'CLASS':<11}  {'DATE':<10}  TITLE", context="task.py list-clean-results"
    )
    for r in rows:
        cls = str(r.get("classification") or "")
        date = str(r.get("promoted_at") or "")[:10]
        _safe_print(
            f"{r['id']:>5}  {cls:<11}  {date:<10}  {r['title']}",
            context="task.py list-clean-results",
        )


def cmd_list_markers(args: argparse.Namespace) -> None:
    events = list_events(args.number)
    if args.prefix:
        events = [e for e in events if e["kind"].startswith(args.prefix)]
    if args.json:
        _safe_print(json.dumps(events, indent=2), context="task.py list-markers")
        return
    for ev in events:
        note = ev.get("note", "")
        note = (note[:80] + "…") if len(note) > 80 else note
        _safe_print(f"{ev['ts']}  {ev['kind']:30s}  {note}", context="task.py list-markers")


def cmd_latest_marker(args: argparse.Namespace) -> None:
    ev = latest_event(args.number, prefix=args.prefix)
    if ev is None:
        _safe_print("(no events)", context="task.py latest-marker")
        return
    _safe_print(json.dumps(ev, indent=2), context="task.py latest-marker")


_SET_BODY_MIN_CHARS = 500
_SET_BODY_STUB_TOKENS = {"placeholder", "tbd", "todo", "stub"}


def _assert_body_nontrivial(text: str, *, source: str) -> None:
    """Refuse to write an obviously broken body to body.md via the CLI.

    Defense-in-depth against the cache → body.md silent-handoff failure
    (incident: task #385, 2026-05-25 — analyzer exited between cache-write
    and set-body, leaving body.md reading literally `placeholder` for ~26h
    while `has_clean_result=true`). Two checks:
      - length ≥ MIN_BODY_CHARS (500) chars
      - body is not a literal stub token (placeholder / tbd / todo / stub)

    The H1 check that the round-1 version imposed has been DROPPED — many
    legitimate non-clean-result bodies (proposed-task auto-drafts, idea
    captures, clarifier "fold answers into body" output) start with `##`
    rather than `# <title>` and would have been spuriously rejected.

    The check is on the CLI path only — the library `set_body()` function
    is unchanged, so internal callers (creation-time stubs, tests, the
    snapshot path) keep working. CLI users can bypass with `--allow-stub`
    when they intentionally need to write a short body. The shape-
    specific guarantees (four required H2s, H1 with confidence tag) for
    promoted clean-result bodies live in `scripts/verify_task_body.py`,
    which the analyzer's Step 6 pre-flight + post-flight grep gates run
    on either side of the cache → body.md handoff.
    """
    if len(text) < _SET_BODY_MIN_CHARS:
        raise SystemExit(
            f"set-body: source ({source}) is suspiciously short "
            f"({len(text)} chars; floor is {_SET_BODY_MIN_CHARS}). "
            "Real bodies are ≥ 500 chars. "
            "If you really mean to write a stub, pass --allow-stub."
        )
    if text.strip().casefold() in _SET_BODY_STUB_TOKENS:
        raise SystemExit(
            f"set-body: source ({source}) is a literal stub token "
            f"({text.strip()!r}). Defense-in-depth against the cache → "
            "body.md silent-handoff failure (incident: task #385, "
            "2026-05-25). If you really mean to write a stub, pass "
            "--allow-stub."
        )


def cmd_set_body(args: argparse.Namespace) -> None:
    """CLI handler for `task.py set-body <N> [--body|--file|stdin] [--snapshot]
    [--allow-stub] [--allow-goal-drop]`.

    Reads the new body from one of three sources (--body string, --file
    path, or stdin), runs the non-trivial-body assertion via
    `_assert_body_nontrivial` unless `--allow-stub` is passed, then
    delegates to the library `set_body()` for the actual write +
    flock + commit. The library-side Goal-H2 drop guard (incident #1112)
    raises `GoalH2DropError` when a `kind: experiment` body update would
    remove the `## Goal` H2 present in the prior body — caught here and
    re-raised as a clean `SystemExit` (the `--allow-stub` style); pass
    `--allow-goal-drop` for a deliberate drop (e.g. the v2 report write).
    """
    if args.body is not None:
        new_body = args.body
        source = "<--body string>"
    elif args.file:
        new_body = Path(args.file).read_text()
        source = args.file
    else:
        new_body = sys.stdin.read()
        source = "<stdin>"
    # A `paper: true` task's body.md is a thin paper-stub (H1 + abstract +
    # paper link) by design — a short stub is NOT the #385 cache→body.md
    # silent-handoff defect for a paper-task, so the non-trivial-body guard
    # is skipped for it (the paper itself is the clean-result, verified by
    # verify_paper.py). `--allow-stub` still bypasses the guard for any task.
    is_paper = False
    try:
        is_paper = is_paper_task(get_task(args.number)["frontmatter"])
    except (FileNotFoundError, KeyError):
        is_paper = False
    if not args.allow_stub and not is_paper:
        _assert_body_nontrivial(new_body, source=source)
    try:
        set_body(
            args.number,
            new_body,
            snapshot_original=args.snapshot,
            allow_goal_drop=args.allow_goal_drop,
        )
    except GoalH2DropError as exc:
        # Clean one-line refusal (no raw traceback) — matches the
        # `--allow-stub` guard's SystemExit style.
        raise SystemExit(str(exc)) from exc
    _safe_echo("ok", context="task.py set-body")


def cmd_set_title(args: argparse.Namespace) -> None:
    set_title(args.number, args.title)
    _safe_echo("ok", context="task.py set-title")


def cmd_set_kind(args: argparse.Namespace) -> None:
    set_kind(args.number, args.kind)
    _safe_echo("ok", context="task.py set-kind")


def cmd_set_clean_result(args: argparse.Namespace) -> None:
    set_clean_result(
        args.number,
        value=not args.unset,
        allow_paper_warn=not getattr(args, "require_pdf_url", False),
    )
    _safe_echo("ok", context="task.py set-clean-result")


def cmd_add_tag(args: argparse.Namespace) -> None:
    add_tag(args.number, args.tag)
    _safe_echo("ok", context="task.py add-tag")


def cmd_remove_tag(args: argparse.Namespace) -> None:
    remove_tag(args.number, args.tag)
    _safe_echo("ok", context="task.py remove-tag")


def cmd_set_track(args: argparse.Namespace) -> None:
    set_track(args.number, args.track)
    _safe_echo("ok", context="task.py set-track")


def cmd_promote(args: argparse.Namespace) -> None:
    new_path = promote(args.number, args.verdict)
    _safe_echo(str(new_path), context="task.py promote")


def cmd_new_plan_version(args: argparse.Namespace) -> None:
    plan_md = Path(args.file).read_text() if args.file else sys.stdin.read()
    v = new_plan_version(args.number, plan_md)
    rel = f"tasks/<status>/{args.number}/plans/v{v}.md"
    _safe_echo(
        f"Plan v{v} written → https://eps.superkaiba.com/tasks/{args.number}/plan",
        context="task.py new-plan-version",
    )
    print(f"  ({rel})", file=sys.stderr)


def cmd_find(args: argparse.Namespace) -> None:
    path = find_task_path(args.number)
    _safe_print(str(path), context="task.py find")


def cmd_tasks_dir(_args: argparse.Namespace) -> None:
    """Print the absolute path of the canonical ``tasks/`` directory.

    Resolves via ``task_workflow.tasks_dir()``, which goes through the
    main-repo branch-guard. Exits non-zero with a one-line error message
    on stderr if the resolver refuses (main worktree off `main`, detached
    HEAD, etc.) — never leaks a raw traceback to the user.
    """
    try:
        _safe_print(str(tasks_dir()), context="task.py tasks-dir")
    except RuntimeError as e:
        print(f"task.py tasks-dir: {e}", file=sys.stderr)
        sys.exit(1)


def _print_reconcile_report(rep: ReconcileReport) -> None:
    """Render a ReconcileReport: per-class counts, then the per-task change
    lines, then the empty-stub orientation line + skips."""
    verb = "applied" if rep.applied else "would apply"
    _safe_print(
        f"reconcile: {len(rep.stale_real)} stale path(s) {verb}, "
        f"{len(rep.missing_real)} missing entry(ies) {verb}, "
        f"{len(rep.empty_stubs)} empty stub(s), {len(rep.skipped)} skipped",
        context="task.py audit",
    )
    for c in rep.stale_real:
        _safe_print(f"  [stale_real] #{c.task_id}: {c.detail}", context="task.py audit")
    for c in rep.missing_real:
        _safe_print(f"  [missing_real] #{c.task_id}: {c.detail}", context="task.py audit")
    if rep.highest_id_bumped is not None:
        _safe_print(f"  [highest_id] {rep.highest_id_bumped.detail}", context="task.py audit")
    if rep.empty_stubs:
        _safe_print(
            "empty stubs (NOT reconciled — surfaced for manual triage):", context="task.py audit"
        )
        for c in rep.empty_stubs:
            _safe_print(f"  [empty_stub] #{c.task_id}: {c.detail}", context="task.py audit")
        _safe_print(
            "  These are empty task stubs on `main` (no body.md / events.jsonl). "
            "Their bodies likely live on `issue-<N>` branches that haven't merged. "
            "Manual triage required.",
            context="task.py audit",
        )
    if rep.skipped:
        _safe_print(
            "skipped (NOT reconciled — registry entry left untouched):", context="task.py audit"
        )
        for c in rep.skipped:
            _safe_print(f"  [skipped] #{c.task_id}: {c.detail}", context="task.py audit")


def cmd_audit(args: argparse.Namespace) -> None:
    repair = getattr(args, "repair", False)
    do_apply = getattr(args, "apply", False)
    if do_apply and not repair:
        print("use --apply only with --repair", file=sys.stderr)
        sys.exit(2)

    if not repair:
        # Report-only mode (unchanged): list drift, exit 1 on any problem.
        problems = audit()
        if not problems:
            _safe_print("AUDIT PASS — registry and filesystem agree", context="task.py audit")
            return
        _safe_print(f"AUDIT FAIL — {len(problems)} problem(s):", context="task.py audit")
        for p in problems:
            _safe_print(f"  - {p}", context="task.py audit")
        sys.exit(1)

    rep = reconcile_registry(apply=do_apply)
    if rep.is_clean:
        _safe_print("registry already reconciled — nothing to repair", context="task.py audit")
        return
    _print_reconcile_report(rep)
    if not do_apply:
        # Dry-run is informational — always exit 0.
        return
    # Apply: exit 0 iff the registry now agrees with the filesystem (no
    # unresolved empty stubs / skips), else exit 1 so the operator triages.
    if rep.unresolved_count > 0:
        sys.exit(1)


# ─── Binding-concerns handlers ────────────────────────────────────────────


def cmd_raise_concern(args: argparse.Namespace) -> None:
    """Append a `raised` (or `verified-open` on re-raise) event to
    concerns.jsonl. Re-raising the SAME concern_id at the SAME round with
    the SAME severity is a no-op (returns the existing event)."""
    payload = raise_concern(
        args.number,
        args.concern_id,
        severity=args.severity,
        summary=args.summary,
        raised_by=args.by,
        raised_at_round=args.round,
        evidence=args.evidence,
    )
    _safe_echo(
        json.dumps(payload, indent=2, ensure_ascii=False),
        context="task.py raise-concern",
    )


def cmd_address_concern(args: argparse.Namespace) -> None:
    """Append an `addressed` event recording that the implementer (or
    analyzer / planner) believes the concern has been fixed. The next
    reviewer round verifies; a re-raise after `addressed` becomes a
    `verified-open` event rather than a fresh `raised`."""
    payload = address_concern(
        args.number,
        args.concern_id,
        addressed_by=args.by,
        addressed_at_round=args.round,
        summary=args.summary,
    )
    _safe_echo(
        json.dumps(payload, indent=2, ensure_ascii=False),
        context="task.py address-concern",
    )


def cmd_defer_concern(args: argparse.Namespace) -> None:
    """USER-ONLY: append a `deferred` event with a substantive rationale.

    CLI layer enforces `--by user` (or `--by reconciler` for ensemble
    severity downgrades, per design spec); the library layer enforces
    the same as defense-in-depth. BLOCKERs cannot be user-deferred —
    the sole exception is the reconciler's binding severity-downgrade
    via `--by reconciler` (workflow.yaml § concerns_protocol.
    reconciler_special_case). Rationale must be ≥ 40 chars AND not
    match a known boilerplate phrase ("user accepted", "ok", "lgtm",
    "wontfix", etc.).
    """
    if args.by not in ("user", "reconciler"):
        raise SystemExit(
            "task.py defer-concern: --by must be 'user' (or 'reconciler' "
            f"for ensemble-tie-break severity downgrade); got {args.by!r}. "
            "This command is user-only — automation must NOT defer concerns."
        )
    payload = defer_concern(
        args.number,
        args.concern_id,
        by=args.by,
        rationale=args.rationale,
    )
    _safe_echo(
        json.dumps(payload, indent=2, ensure_ascii=False),
        context="task.py defer-concern",
    )


def cmd_list_concerns(args: argparse.Namespace) -> None:
    """List the concerns ledger for a task. Default: full event stream;
    with `--open-only`, only concerns whose latest event is `raised` or
    `verified-open` (excludes `addressed` / `deferred`)."""
    rows = list_concerns(args.number, open_only=args.open_only)
    if args.json:
        _safe_print(json.dumps(rows, indent=2, ensure_ascii=False), context="task.py list-concerns")
        return
    if not rows:
        _safe_print("(no concerns)", context="task.py list-concerns")
        return
    _safe_print(
        f"{'TS':<20}  {'EVENT':<15}  {'SEV':<8}  CONCERN_ID  SUMMARY",
        context="task.py list-concerns",
    )
    for row in rows:
        summary = (row.get("summary") or "")[:60]
        _safe_print(
            f"{row['ts']:<20}  {row['event']:<15}  "
            f"{(row.get('severity') or '?'):<8}  "
            f"{row['concern_id']:<30}  {summary}",
            context="task.py list-concerns",
        )


def cmd_migrate_body(args: argparse.Namespace) -> None:
    """`task.py migrate-body` — patch awaiting_promotion bodies to verify_task_body PASS.

    Three modes:
      --report                  classification table over all awaiting_promotion bodies
      --dry-run <N> | --all     show proposed patches (no writes)
      --apply <N>  | --all      write patches (conformant-but-failing bodies only;
                                v4-legacy bodies report needs-user — converter
                                retired 2026-06-09, migrate manually per SPEC.md)
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
        _safe_print(f"{'ID':<7}  {'CLASS':<22}  before -> after", context="task.py migrate-body")
        _safe_print("─" * 64, context="task.py migrate-body")
        for tid in target_ids:
            # Classification-only — no apply, no patch.
            try:
                result = migrate_one(tid, apply=False, shape=args.shape, verbose=False)
            except FileNotFoundError as e:
                _safe_print(f"#{tid:<5}  (error: {e})", context="task.py migrate-body")
                continue
            _safe_print(result.report_line(), context="task.py migrate-body")
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
                _safe_print(
                    f"#{tid}: skip ({result.classification.value})", context="task.py migrate-body"
                )
            continue
        _safe_print(result.report_line(), context="task.py migrate-body")
        for action in result.actions:
            _safe_print(f"    - {action}", context="task.py migrate-body")
        if result.needs_user:
            n_needs_user += 1
            _safe_print(
                f"    [needs-user] {result.needs_user_reason}", context="task.py migrate-body"
            )
        else:
            n_changed += 1
        if args.verbose and result.diff_preview:
            _safe_print("    ─── diff preview ───", context="task.py migrate-body")
            for line in result.diff_preview.splitlines()[:30]:
                _safe_print(f"    {line}", context="task.py migrate-body")
            _safe_print(context="task.py migrate-body")

    _safe_print(context="task.py migrate-body")
    verb = "applied" if args.apply else "dry-run"
    _safe_print(
        f"task.py migrate-body — {verb}: {n_changed} changed, "
        f"{n_needs_user} needs-user, {n_skip} skipped",
        context="task.py migrate-body",
    )


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
            # Single source of truth: task_workflow.KINDS (mirrors the routing
            # law in CLAUDE.md § "Routing experiment intent"). `campaign` =
            # the question-level runner (/campaign <N>, task #586); the
            # note/reading/idea/question/decision tail are the Human-board kinds.
            choices=list(KINDS),
        )
        p.add_argument(
            "--track",
            default=None,
            choices=["experiment", "human"],
            help=(
                "task track: 'experiment' (agent-runnable end-to-end) or "
                "'human' (think/read/decide). Default: derived from --kind "
                "(note/reading/idea/question/decision -> human, else experiment)."
            ),
        )
        p.add_argument("--title", required=True)
        body_group = p.add_mutually_exclusive_group()
        body_group.add_argument("--body", help="body text directly")
        body_group.add_argument("--body-file", help="path to body file")
        p.add_argument("--parent", type=int, default=None, help="parent task id (optional)")
        p.add_argument("--tag", action="append", default=[], help="tag (repeatable)")
        p.add_argument("--status", default="proposed", choices=STATUSES)
        p.add_argument(
            "--goal",
            default=None,
            help=(
                "one-sentence canonical Goal of the experiment. Honored only "
                "when --kind=experiment (warning + ignore otherwise). When "
                "set, writes frontmatter `goal:` AND injects a `## Goal` H2 "
                "between H1 and any other H2. Optional at creation time; "
                "enforced at /issue Step 0c for kind=experiment tasks."
            ),
        )
        p.add_argument(
            "--origin-prompt",
            default=None,
            help=(
                "verbatim user prompt(s) that originated this task. Written "
                "to frontmatter `origin_prompt:` (any kind); the clean-result "
                "`## Reproducibility` `**Context:**` row carries it forward "
                "(SPEC.md; verify_task_body.py check 17). Optional — when the "
                "prompt is long or there are several, a `## Provenance` body "
                "section (see task #611) works too."
            ),
        )
        p.add_argument(
            "--workflow",
            default=None,
            choices=list(WORKFLOW_VERSIONS),
            help=(
                "workflow pipeline: v1 (default) or v2 (report-only); "
                "resolved as this flag > env EPM_DEFAULT_WORKFLOW > v1."
            ),
        )
        # Sagan-compatibility: accept --runpod-account but ignore it.
        p.add_argument("--runpod-account", default=None, help="(ignored; Sagan compat)")
        p.set_defaults(func=cmd_create)

    p = sub.add_parser("set-status", help="move task to a new status (git mv + commit)")
    p.add_argument("number", type=int)
    # No argparse `choices=` here on purpose: cmd_set_status validates the
    # value itself so it can emit a 'did you mean <closest>?' hint instead
    # of argparse's bare 'invalid choice' dump (see _status_error_message).
    p.add_argument("status", help="target status; one of: " + ", ".join(STATUSES))
    p.add_argument("--note", default=None)
    p.add_argument(
        "--auto-approve-if-autonomous",
        action="store_true",
        help=(
            "On a plan_pending transition, apply the code-enforced autonomous "
            "plan-approval gate: if EPM_AUTONOMOUS_SESSION is set and --gpu-hours "
            "<= EPM_PLAN_AUTOAPPROVE_GPU_HOURS (default 24), auto-flip to approved "
            "and post epm:plan-approved; if over-cap or --gpu-hours is omitted, "
            "stay at plan_pending and post epm:awaiting-spend-approval; if not "
            "autonomous, stay at plan_pending (interactive). Prints a "
            "'PLAN_GATE_DECISION: <decision> ...' line."
        ),
    )
    p.add_argument(
        "--gpu-hours",
        type=float,
        default=None,
        help="Plan's estimated total GPU-hours; used by --auto-approve-if-autonomous.",
    )
    p.add_argument(
        "--force-followup-exit",
        action="store_true",
        help=(
            "Override the same-issue follow-up status-hold rule: allow a "
            "followups_running task to move to an intermediate pipeline status "
            "(planning/plan_pending/approved/running/verifying/interpreting/"
            "reviewing). Without this flag the transition is refused — the round "
            "HOLDS followups_running end-to-end and exits only at "
            "awaiting_promotion/blocked (SKILL.md Step 9b § Same-issue follow-up "
            "loop, step 3). Pass only to deliberately abandon the round."
        ),
    )
    p.set_defaults(func=cmd_set_status)

    for name in ("post-marker", "post-event"):
        p = sub.add_parser(name, help="append an event to events.jsonl")
        p.add_argument("number", type=int)
        p.add_argument("marker", help="marker kind, e.g. epm:plan, epm:reviewer-verdict")
        # --note (inline string) and --file (path to a body file) are
        # mutually exclusive; both may be omitted (marker with no note).
        # File input is the documented idiom for multi-line / shell-
        # special bodies (see .claude/skills/issue/markers.md), matching
        # set-body and new-plan-version. Size cap (50,000 UTF-8 chars,
        # task_workflow.EVENT_NOTE_MAX) is enforced by post_event itself.
        note_group = p.add_mutually_exclusive_group()
        note_group.add_argument("--note", default=None, help="note body as an inline string")
        note_group.add_argument(
            "--file", default=None, help="path to a file containing the note body"
        )
        p.add_argument(
            "--version",
            type=int,
            default=None,
            help=(
                "explicit marker version; omitted -> max(existing versions "
                "for this marker kind) + 1, so re-posts never shadow a "
                "higher version under highest-version-wins resume"
            ),
        )
        p.add_argument("--by", default="unknown")
        p.set_defaults(func=cmd_post_event)

    p = sub.add_parser("list-by-status", help="list tasks in a status (or all)")
    p.add_argument("--status", default=None, choices=list(STATUSES))
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list_by_status)

    p = sub.add_parser(
        "list-children",
        help="list tasks whose frontmatter parent_id == N (campaign / follow-up children)",
    )
    p.add_argument("number", type=int)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list_children)

    p = sub.add_parser(
        "list-clean-results",
        help="list every task with has_clean_result=true, newest first",
    )
    p.add_argument("--search", default=None, help="case-insensitive substring on title + body")
    p.add_argument(
        "--include-pending",
        action="store_true",
        help="also include has_clean_result=true tasks not yet promoted (pending classification)",
    )
    p.add_argument("--useful-only", action="store_true", help="drop not-useful rows")
    p.add_argument("--limit", type=int, default=0, help="cap rows (0 = no cap)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list_clean_results)

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
            "or stdin is written into the body region (after the closing `---` "
            "line). Any leading YAML frontmatter block(s) in the new content are "
            "STRIPPED before write — this prevents the duplicate-frontmatter trap "
            "where callers pass a complete markdown document (frontmatter + body) "
            "and end up with two `---...---` blocks in body.md (incident: task "
            "#389, 2026-05-26). The strip is idempotent. If you need to change a "
            "frontmatter field, use the dedicated mutators (`set-title`, "
            "`set-clean-result`, `add-tag`, `remove-tag`, `set-goal`) — the "
            "frontmatter inside the new content is discarded, not merged."
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
    p.add_argument(
        "--allow-stub",
        action="store_true",
        help=(
            "bypass the <500-char and literal-stub-token (placeholder / tbd / "
            "todo / stub) checks. Defense-in-depth against the cache→body.md "
            "silent-handoff failure (incident: task #385, 2026-05-25). Use only "
            "when you genuinely intend to write a stub (e.g. creation-time "
            "placeholder); the analyzer's clean-result handoff must NOT use this "
            "flag. NOTE: a `paper: true` task auto-allows a short paper-stub "
            "(H1 + abstract + paper link) WITHOUT this flag — the paper itself, "
            "not the body, is the clean-result (verified by verify_paper.py)."
        ),
    )
    p.add_argument(
        "--allow-goal-drop",
        action="store_true",
        help=(
            "allow the new body to REMOVE the `## Goal` H2 present in the prior "
            "kind:experiment body. Without this flag the write refuses "
            "(GoalH2DropError) — the Goal is the canonical target every "
            "downstream agent reads (incident #1112). Deliberate droppers: the "
            "workflow-v2 report write (report-v1 carries `## Motivation:`, no "
            "`## Goal`; `goal:` frontmatter survives). Paper-stub writes are "
            "auto-exempt via `paper: true`, no flag needed."
        ),
    )
    p.set_defaults(func=cmd_set_body)

    p = sub.add_parser("set-title", help="update task title (frontmatter)")
    p.add_argument("number", type=int)
    p.add_argument("title")
    p.set_defaults(func=cmd_set_title)

    p = sub.add_parser(
        "set-kind",
        help="reclassify a misfiled task kind (frontmatter + REGISTRY snapshot)",
        description=(
            "Correct a task's `kind` through the canonical flock-protected, "
            "registry-consistent path. Use when a task was filed under the "
            "wrong kind — e.g. a fix-validation / 'test that X works' task "
            "created as `kind: experiment` that should be `kind: infra` (so "
            "it completes on the test-verdict path instead of being dragged "
            "through the clean-result/promotion machinery; incident #672). "
            "`kind` IS denormalized into REGISTRY.json, so this updates the "
            "registry snapshot too. Does NOT touch status, body, or "
            "has_clean_result — flip those separately if the misfile also "
            "left the task in a wrong state."
        ),
    )
    p.add_argument("number", type=int)
    p.add_argument("kind", choices=list(KINDS), help="target kind; one of: " + ", ".join(KINDS))
    p.set_defaults(func=cmd_set_kind)

    p = sub.add_parser(
        "set-goal",
        help="set / refine the canonical Goal of the experiment (frontmatter + `## Goal` H2)",
        description=(
            "Update the task's Goal-of-the-experiment field. Writes "
            "frontmatter `goal:` AND ensures a `## Goal` H2 block is "
            "present in body.md between H1 and any other H2. Emits an "
            "`epm:goal-updated v1` marker with from/to/by fields. "
            "Idempotent: identical re-application is a no-op. The "
            "`--by` flag identifies which agent fired the update: "
            "`user` (default; Step 0c or manual), `clarifier` (Step 1 "
            "refinement gate), `planner` (/adversarial-planner Phase 1 "
            "refinement gate). Critic / experiment-implementer / "
            "analyzer / clean-result-critic / interpretation-critic / "
            "follow-up-proposer MUST NOT call this command."
        ),
    )
    p.add_argument("number", type=int)
    p.add_argument("goal", help="one-sentence Goal of the experiment")
    p.add_argument(
        "--by",
        default="user",
        choices=["user", "clarifier", "planner"],
        help="which agent is making the change (default: user)",
    )
    p.add_argument(
        "--reason",
        default=None,
        help="optional free-form rationale; included verbatim in the marker note",
    )
    p.set_defaults(func=cmd_set_goal)

    p = sub.add_parser(
        "set-clean-result",
        help="flip has_clean_result=true (or false with --unset)",
        description=(
            "Flip has_clean_result. For a `paper: true` task, flipping it to "
            "True first VALIDATES docs/papers/issue_<N>/paper_manifest.json "
            "(required tex/pdf/paper_html artifacts present + sha256 match); a "
            "hard problem blocks the flip. A null pdf_hf_url (local-only build, "
            "not yet uploaded) is a WARN tolerated by default — pass "
            "--require-pdf-url to make it blocking."
        ),
    )
    p.add_argument("number", type=int)
    p.add_argument("--unset", action="store_true")
    p.add_argument(
        "--require-pdf-url",
        action="store_true",
        help="(paper tasks) treat a null pdf_hf_url as a hard FAIL, not a WARN",
    )
    p.set_defaults(func=cmd_set_clean_result)

    p = sub.add_parser("add-tag", help="add a tag to frontmatter")
    p.add_argument("number", type=int)
    p.add_argument("tag")
    p.set_defaults(func=cmd_add_tag)

    p = sub.add_parser("remove-tag", help="remove a tag from frontmatter")
    p.add_argument("number", type=int)
    p.add_argument("tag")
    p.set_defaults(func=cmd_remove_tag)

    p = sub.add_parser("set-track", help="set task track (experiment|human) in frontmatter")
    p.add_argument("number", type=int)
    p.add_argument("track", choices=["experiment", "human"])
    p.set_defaults(func=cmd_set_track)

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

    p = sub.add_parser(
        "tasks-dir",
        help="print absolute path of the canonical tasks/ directory in main repo",
    )
    p.set_defaults(func=cmd_tasks_dir)

    p = sub.add_parser("audit", help="validate REGISTRY.json against filesystem")
    p.add_argument(
        "--repair",
        action="store_true",
        help=(
            "reconcile REGISTRY.json against the on-disk task tree (re-point "
            "stale paths + add missing entries, re-snapshotting from each "
            "task's body.md). Dry-run unless --apply is also passed."
        ),
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="with --repair: write the reconcile to REGISTRY.json + one git commit.",
    )
    p.set_defaults(func=cmd_audit)

    # ─── Binding-concerns subcommands ────────────────────────────────────

    p = sub.add_parser(
        "raise-concern",
        help="append a `raised` event to concerns.jsonl (binding-review surface)",
        description=(
            "Reviewer subcommand. Records a concern raised against a task "
            "during the review loop. ``--concern-id`` MUST be stable "
            "kebab-case (lowercase letters/digits/hyphens, 2-80 chars, "
            "starts alphanum). Re-raising the SAME concern_id at the SAME "
            "round with the SAME severity is a no-op; re-raising after the "
            "concern was `addressed` records a `verified-open` event (the "
            "reviewer is saying 'you said you fixed this but the issue is "
            "still visible'). BLOCKER concerns cannot be user-deferred — "
            "they signal a strict gate the orchestrator must address or "
            "pivot. Mirror event posts to events.jsonl as "
            "`epm:concern-raised v1`."
        ),
    )
    p.add_argument("number", type=int)
    p.add_argument(
        "--concern-id",
        dest="concern_id",
        required=True,
        help="stable kebab-case id (e.g. probe-position-undefined)",
    )
    p.add_argument(
        "--severity",
        required=True,
        choices=sorted(CONCERN_SEVERITIES),
        help="BLOCKER (no deferral), CONCERN (binding), NIT (optional)",
    )
    p.add_argument("--summary", required=True, help="one-line ≤200-char description")
    p.add_argument("--by", required=True, help="reviewer name (e.g. code-reviewer, critic)")
    p.add_argument(
        "--round",
        required=True,
        type=int,
        help="current review round (≥1) for the raising reviewer",
    )
    p.add_argument("--evidence", default=None, help="optional path / quote / pointer")
    p.set_defaults(func=cmd_raise_concern)

    p = sub.add_parser(
        "address-concern",
        help="append an `addressed` event recording implementer believes concern is fixed",
        description=(
            "Implementer / analyzer subcommand. Records that this round's "
            "implementer believes the concern has been fixed. The next "
            "reviewer round verifies — if the issue is still visible, that "
            "reviewer calls raise-concern again, which transitions the "
            "record to `verified-open` (NOT a fresh `raised`). The "
            "concern_id MUST refer to a concern that has been raised at "
            "least once on this task. Mirror event posts to events.jsonl "
            "as `epm:concern-addressed v1`."
        ),
    )
    p.add_argument("number", type=int)
    p.add_argument(
        "--concern-id",
        dest="concern_id",
        required=True,
        help="the kebab-case id raised by the prior reviewer round",
    )
    p.add_argument("--by", required=True, help="implementer name (e.g. implementer, analyzer)")
    p.add_argument(
        "--round",
        required=True,
        type=int,
        help="current implementer round (≥1) recording the address",
    )
    p.add_argument(
        "--summary",
        "--rationale",
        dest="summary",
        default=None,
        help="optional updated summary; defaults to the original raised summary "
        "(--rationale is an accepted alias, matching defer-concern's flag name)",
    )
    p.set_defaults(func=cmd_address_concern)

    p = sub.add_parser(
        "defer-concern",
        help="USER-ONLY: append a `deferred` event with substantive rationale",
        description=(
            "USER-ONLY subcommand for explicit concern deferral. CLI "
            "rejects without `--by user` (or `--by reconciler` for "
            "ensemble-tie-break severity downgrades, per the design "
            "spec); the library function ALSO rejects defense-in-depth. "
            "BLOCKER concerns CANNOT be user-deferred — address them or "
            "pivot strategy. Rationale must be ≥40 chars AND not match a "
            "known boilerplate phrase ('user accepted', 'ok', 'lgtm', "
            "'wontfix', etc.) — rubber-stamp deferrals defeat the "
            "purpose. Mirror event posts to events.jsonl as "
            "`epm:concern-deferred v1`."
        ),
    )
    p.add_argument("number", type=int)
    p.add_argument(
        "--concern-id",
        dest="concern_id",
        required=True,
        help="the kebab-case id of the open concern being deferred",
    )
    p.add_argument(
        "--by",
        required=True,
        help="must be 'user' (or 'reconciler' for severity-downgrade tie-break)",
    )
    p.add_argument(
        "--rationale",
        required=True,
        help="≥40-char non-boilerplate prose explaining why the concern survives",
    )
    p.set_defaults(func=cmd_defer_concern)

    p = sub.add_parser(
        "list-concerns",
        help="list the concerns ledger for a task",
        description=(
            "List all concerns raised against a task, or only currently OPEN "
            "ones (latest event is `raised` or `verified-open`). Reviewer "
            "agents read `task.py list-concerns <N> --open-only --json` at "
            "the start of every round to inherit cross-stage concern "
            "history."
        ),
    )
    p.add_argument("number", type=int)
    p.add_argument(
        "--open-only",
        dest="open_only",
        action="store_true",
        help="only concerns whose latest event is `raised` or `verified-open`",
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of table")
    p.set_defaults(func=cmd_list_concerns)

    p = sub.add_parser(
        "migrate-body",
        help="patch awaiting_promotion bodies into verify_task_body compliance",
        description=(
            "Migrate awaiting_promotion task bodies to the markdown clean-result spec "
            "(verify_task_body.py 13-check). Conformant-but-failing bodies are patched "
            "in place (Repro subgroups, cherry-picked label, qualitative-data link). "
            "v4-legacy bodies (## TL;DR / ## Summary / ## Details / ## Source issues) "
            "are classified but NOT converted — auto-conversion was retired 2026-06-09 "
            "(it targeted the retired four-H2 shape); they report needs-user, migrate "
            "manually per .claude/skills/clean-results/SPEC.md. HTML bodies carrying "
            "<!-- legacy-sagan-card --> are grandfathered and skipped."
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
        help=(
            "force a specific patch chain (overrides auto-classification); "
            "'v4-to-new' now always reports needs-user (converter retired)"
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print a unified-diff preview after each body",
    )
    p.set_defaults(func=cmd_migrate_body)

    args = parser.parse_args()
    # The canonical resolver (task_workflow.repo_root) raises a loud, distinct
    # RuntimeError for the non-routable states it still refuses: a genuinely
    # detached HEAD (after the bounded live-rebase wait,
    # `EPM_TASKPY_REBASE_WAIT_SECONDS` — #996; the wait-timeout refusal is
    # still a RuntimeError, so this catch covers it too), missing `tasks/`,
    # bare/submodule layouts, or a feature-branch primary with
    # no local `main` to route through. (The common "primary parked on a real
    # feature branch" case is auto-routed through a managed main-pinned worktree
    # and does NOT raise.) Catch RuntimeError at the top-level dispatch so EVERY
    # subcommand prints a clean one-line error to stderr and exits 1 instead of
    # leaking a raw traceback. `cmd_tasks_dir` keeps its own catch as defense in
    # depth; the message there is identical in shape.
    try:
        args.func(args)
    except RuntimeError as e:
        print(f"task.py {args.cmd}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
