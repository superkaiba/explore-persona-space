# task.py callable API (for Sagan dashboard)

This file documents the `scripts/task.py` subcommands that Sagan's
runner shells out to. All mutations go through `task.py` so the
flock + git-commit single-writer discipline is preserved.

## Convention

- Every subcommand that mutates state takes `--source=<scheme>:<id>`.
  Sagan passes `sagan-user:<sessions.id>`. CLI invocations from a tty
  default to `cli`. Agents pass `agent:<agent-name>`.
- `--source` is validated at the CLI layer: newlines/CRs are rejected
  (would corrupt `events.jsonl`); empty/whitespace is normalized to
  none.
- Read-only subcommands (`view`, `list-by-status`, `list-markers`,
  `latest-marker`, `find`) emit JSON when called with `--json`.
- Non-zero exit means the operation refused; stderr explains why.

## Subcommands Sagan calls

| Subcommand | Purpose | Sagan callers |
| --- | --- | --- |
| `view N --json` | Snapshot of task state (status, title, kind, has_clean_result, path) for the mirror cache. | `eps-mirror` job |
| `latest-marker N` | Last event on the task. | mirror reconciler |
| `list-markers N --json` | All events on the task. Used to populate the task detail timeline. | task detail page |
| `list-by-status --json` | Board population: rows per status folder. | initial mirror seed |
| `set-status N <status> --source=…` | Advance / regress status. Source flows into the `epm:status-changed` event payload. | approve, block, unblock |
| `post-event N <kind> --note=… --source=…` | Append an arbitrary marker (e.g. `epm:paused-via-sagan`). Source flows into the event payload. | Pause button |
| `comment-add N --author=… --body-md=… --source=… [--reply-to=… --thread-id=…]` | Append a comment line to `tasks/<status>/<N>/comments.jsonl`. Author is one of `user|claude|codex`. | comment composer, Claude reply |
| `promote N <verdict> --source=sagan-user:…` | Promote `awaiting_promotion → completed` with verdict `useful|not-useful`. **USER-ONLY:** refuses agent sources; accepts only `--source=cli` or `--source=sagan-user:*` (or no source from an interactive tty). | Promote button |

## NOT called from Sagan

These exist in `task.py` but Sagan does not shell out to them:

- `migrate-body` — one-off maintenance.
- `audit` — diagnostic only.
- `new` — task creation goes through `/issue` skill or direct CLI today.
- `set-body`, `set-title`, `set-clean-result`, `add-tag`, `remove-tag`,
  `new-plan-version` — agent-side writes only. Humans use the dashboard
  composer or terminal, not Sagan-mediated writes for these.

## Source-string conventions

| Source value pattern | Meaning | Set by |
| --- | --- | --- |
| `cli` | Interactive terminal invocation (default when stdin is a tty). | task.py itself when `--source` omitted from a tty. |
| `sagan-user:<sessions.id>` | Human action through Sagan dashboard. `<sessions.id>` is the row PK in `@sagan/db` `sessions` table. | Sagan runner when shelling out. |
| `agent:<agent-name>` | Agent-side automated write (experimenter, implementer, code-reviewer, …). | The agent script. |

`promote` accepts only `cli` and `sagan-user:*`. Other gates do not
currently gate on source — they record it for the audit log only.

## Exit codes

- `0` — success.
- `2` — refused (invalid source, USER-ONLY gate violated, etc.).
- Non-zero with traceback — unhandled exception (e.g. task does not
  exist).

## Single-writer guarantees

Every mutating subcommand:

1. Acquires the task-level flock (`tasks/<status>/<N>/` lockfile,
   distinct from the orchestrator lock at `.orchestrator.pid`).
2. Mutates files.
3. Records a `git commit` per operation. Non-event writers
   (`set-body`, `set-title`, `set-clean-result`, `add-tag`,
   `remove-tag`, `new-plan-version`) embed the `source` value in the
   commit message subject suffix (e.g. `set-title #192 [source=sagan-user:abc]`).
4. Releases the flock.

This is the contract Sagan relies on: any shell-out to `task.py`
either fully completes (events.jsonl line written, git commit landed)
or fully fails (no partial state). Sagan does not need to retry or
reconcile partial writes.
