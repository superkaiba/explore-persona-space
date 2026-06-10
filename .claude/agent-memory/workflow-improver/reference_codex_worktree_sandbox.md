---
name: codex-worktree-sandbox
description: Codex twin agents run with cwd = the issue-N worktree; cannot resolve paths to main's current-status task folder. task.py's branch-guard auto-routes through `_task-main-pin` worktree, so `task.py latest-marker <N>` works from anywhere — use that, then INLINE marker bodies into the Codex prompt.
metadata:
  type: reference
---

The four Codex twin agents (`codex-code-reviewer`, `codex-critic`,
`codex-interpretation-critic`, `codex-clean-result-critic`) are dispatched
from the orchestrator via `scripts/codex_task.py`, which calls Codex's
`codex-companion.mjs task --background`. **Codex's sandbox cwd defaults to
`process.cwd()` at spawn time** — when the orchestrator dispatches from the
issue-N worktree (the typical case), Codex's view of the filesystem is
rooted there.

The worktree's `tasks/` folder is FROZEN at branch-creation status:
- Issue-489 worktree only has `tasks/approved/489/` (branch cut at `approved`)
- After the task moves to `running`/`verifying`/etc on main, those folders
  exist only on main — never in the worktree.
- Markers posted via `task.py post-marker <N>` go to main's CURRENT-status
  folder, so they appear in main's `tasks/running/489/events.jsonl` (NOT in
  the worktree's `tasks/approved/489/events.jsonl`).

**Two consequences for prompt composition:**

1. **Never give Codex a `tasks/<status>/<N>/events.jsonl` path.** Codex
   cannot resolve it. The current-status folder is invisible from inside
   the worktree.

2. **`task.py` from inside a worktree IS safe** — see [[branch-guard-blocks-subprocess]]
   for the branch-guard details. `task.py latest-marker <N> --prefix
   epm:experiment-implementation` will return the marker that lives on
   main, because `repo_root()` resolves via `_MODULE_DIR` (the module's
   own filesystem location) and auto-routes through the managed
   `.claude/worktrees/_task-main-pin/` worktree when the caller is on a
   non-main branch.

**The fix pattern for Codex twin agents:**

- Fetch marker bodies (anything in `events.jsonl`) via `task.py
  latest-marker <N>` from the orchestrator/composer side.
- Write the body to a temp file (e.g. `/tmp/codex-<agent>-<N>-r<n>-body.md`).
- Substitute the body's CONTENTS into the Codex prompt template via
  Python `template.replace('{{marker_body}}', body)` — never shell variable
  interpolation (15KB+ markdown with `$`, backticks, etc. is hostile to
  shell quoting).
- Add a grep guard before returning: confirm the substituted prompt
  actually contains the begin/end envelope of the inlined body.

Plan files (committed at branch-cut time, e.g. `plans/v5.md`) are
resolvable inside the worktree ONLY when the branch was cut from main
after the task folder existed. A child task's worktree cut from a PARENT
issue branch predating the task (#550 cut from `origin/issue-538`,
2026-06-10) has NO `tasks/*/<N>/` folder at all — the plan is as
unreachable as the markers. Composers must existence-check
`<worktree>/<plan_marker_path>` and fall back to inlining the canonical
plan from main (`task.py find <N>` → `plans/plan.md`) with a begin/end
envelope (codex-code-reviewer.md Step 2-pre-b, fixed 2026-06-10).

**Where this rule lives in the workflow surface:**

- `codex-code-reviewer.md` Step 2-pre (fixed 2026-06-04, originating
  task #489 r1/r2).
- The other three Codex twins (`codex-critic`, `codex-interpretation-critic`,
  `codex-clean-result-critic`) MAY have analogous issues — `codex-critic`
  reads only the plan body (worktree-safe), but `codex-interpretation-critic`
  reads the latest `epm:interpretation` body (events.jsonl-resident, on
  main only), and `codex-clean-result-critic` reads the latest
  `epm:interpretation` body + the body.md. Audit those agents next time a
  Codex twin in those roles emits "could not read" / wrong-marker-shape
  failures. (Not fixed in this pass — one-candidate-per-invocation rule.)

**Incident:** #489 r1 + r2 (2026-06-04). Both rounds, Codex emitted
`marker-shape` + `smoke-run-missing` FAIL tags that the orchestrator had
to manually strip as false positives. Root cause was Codex's inability to
resolve `tasks/<current-status>/<N>/events.jsonl` from inside the
worktree, not a real marker-shape bug.
