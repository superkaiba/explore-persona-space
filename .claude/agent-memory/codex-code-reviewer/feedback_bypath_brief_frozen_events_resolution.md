---
name: bypath-brief-frozen-events-resolution
description: When a brief orders by-path references (no inlining) and asserts Codex can read the main checkout, resolve each cited task-state path against frozen-worktree semantics before composing — events.jsonl citing post-branch markers goes to the MAIN-root absolute path
metadata:
  type: feedback
---

When the orchestrator's brief explicitly overrides the default inline-marker
pattern with "by path — never inline a large body" AND asserts Codex can read
the MAIN checkout via absolute paths (e.g. "resolve helper scripts from the
main checkout"), honor the override — but resolve EACH cited task-state path
against frozen-worktree semantics first (#2325 r1, 2026-08-16):

- `tasks/<status>/<N>/events.jsonl` citing POST-branch markers (the
  `epm:results` implementer report) must be given as the MAIN checkout's
  ABSOLUTE path. The worktree copy exists and looks plausible but is frozen
  at branch-cut (on #2325 its last row was the 14:22Z status change; the
  16:10Z `epm:results` row was main-only). Say so explicitly in the prompt
  ("do not read the worktree copy for the report") — Codex's default cwd is
  the worktree and the same relative path resolves there to the stale file
  with no error.
- The plan may be worktree-path'd ONLY after the standard Step 2-pre-b
  identity diff (worktree plan.md vs canonical) — unchanged from the inline
  era. On #2325 it was byte-identical (v4 both sides) so the worktree path
  was safe.
- Give the extraction command for the JSONL note
  (`jq -r 'select(.kind=="epm:results").note'`) plus a no-jq fallback, and
  keep the `data-access-blocked` routing: with NO inlined body, a failed
  events read is a GENUINE blocked lens (unlike the inline era, where a
  "could not read marker" FAIL was invalid by construction).

**Why:** the brief's path list is written from the MAIN-side view; the
composer is the only one who checks which side each path actually resolves
on. Passing the brief's relative path unresolved recreates the #489
unreachable-marker class in a quieter form (stale-read, not error).

**How to apply:** any compose whose brief says "by path" — run
`ls <wt>/tasks/*/<N>` + tail the worktree events.jsonl vs main before
choosing each reference. Related: [[worktree-status-folder-both-directions]],
[[concurrent-followups-wrong-plan-symlink]].
