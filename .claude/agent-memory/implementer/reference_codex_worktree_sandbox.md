---
name: codex-worktree-sandbox
description: Codex twins run cwd-rooted in the issue worktree and cannot resolve main's current-status task paths; fetch markers via task.py (branch-guard auto-routes) and INLINE bodies into the Codex prompt
metadata:
  type: reference
---

Codex twin dispatch (`scripts/codex_task.py` → `codex-companion.mjs`) roots
Codex's sandbox at the dispatch cwd — typically the issue-N worktree. The
worktree's `tasks/` tree is FROZEN at branch-cut status: markers posted later
live only in main's CURRENT-status folder, invisible from the worktree.

**Rules for prompt composition:**
1. Never give Codex a `tasks/<status>/<N>/events.jsonl` path — unresolvable.
2. `task.py` from a worktree IS safe (see [[branch-guard-blocks-subprocess]]):
   `task.py latest-marker <N> --prefix <p>` reads main via the managed
   `_task-main-pin` worktree. Fetch the marker body composer-side, write it to
   `/tmp/codex-<agent>-<N>-r<n>-body.md`, substitute via Python
   `template.replace(...)` (never shell interpolation — 15KB markdown is
   hostile to quoting), and grep-guard the begin/end envelope before returning.
3. Plan files resolve in the worktree ONLY if the branch was cut after the
   task folder existed — a child worktree cut from a parent issue branch has
   NO `tasks/*/<N>/` at all (#550). Existence-check, else inline the canonical
   plan from main (`task.py find <N>` → `plans/plan.md`) with an envelope.

**Per-twin state:** `codex-code-reviewer.md` Step 2-pre fixed 2026-06-04
(#489 r1/r2 — Codex's false `marker-shape`/`smoke-run-missing` FAILs were
unresolvable-path artifacts); `codex-clean-result-critic.md` audited + fixed
2026-06-10 (#550 follow-up — absolute canonical-main paths, temp file for the
interpretation note, `cd {{repo_root}} &&` pins). `codex-critic` reads only
the plan (worktree-safe); `codex-interpretation-critic` reads an
events.jsonl-resident marker — audit it on the next "could not read" failure.
