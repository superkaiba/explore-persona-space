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

**Two r3 extensions (#2325 r3, 2026-08-16):**

- **Main-side events.jsonl by working-tree path needs the stash-race re-read
  rule IN the prompt** — under fleet concurrency an uncommitted row is
  transiently reverted by other sessions' pre-commit stash cycles
  (`git checkout -- .` for the hook window), and a working-tree read inside
  the window produced a false "marker destroyed" alarm on #2325. The prompt
  must say: a row that appears missing ⇒ re-read via
  `git -C <main-root> show HEAD:tasks/<status>/<N>/events.jsonl` before
  concluding anything; never file a missing-row finding from a working-tree
  read alone; `data-access-blocked` applies only AFTER the re-read.
- **Review-dispatched fix rounds may post NO new `epm:results` version**
  (the standing report stays v<k>; the round's record is an `epm:progress`
  dispatch note). Preempt the predictable spurious complaint with an explicit
  line: "the absence of an `epm:results` v<k+1> is by orchestrator design
  this round and is NOT a `marker-shape` finding" — and repeat the carve-out
  inside the Blocker-tags bracket.

**/tmp context-path extension (#2253 r5, 2026-08-21):** a brief-ordered
by-path context set can include a `/tmp/...` dispatch note. Reference it by
absolute path, but attach a NON-BLOCKING fallback line ("if unreadable, do
not mark BLOCKED/FAIL — the load-bearing facts are restated in this prompt
and the inlined report; note it in one line and proceed") AND restate its
load-bearing facts (scope decision, measured basis, out-of-scope list) in
the compose-time facts block — /tmp reachability from the Codex sandbox is
less proven than main-checkout reads, and a supporting-context miss must
never convert to a `data-access-blocked` FAIL.

**Dynamic-resolution extension (#2412 r1, 2026-08-20):** prefer
`$(uv run python scripts/task.py find <N>)/plans/plan.md` (and
`.../events.jsonl`) over a hardcoded main-root status path — status folders
MOVE mid-review (running→verifying happens DURING the round), so a hardcoded
`tasks/running/<N>/...` path can go stale while Codex is still reading.
Verified: `task.py find` works read-only from the WORKTREE cwd and returns
the main-root absolute path. Still state the frozen-worktree warning (do not
read the worktree events copy) and keep the stash-race re-read rule, with the
`git show HEAD:` fallback told to adjust the status folder to whatever `find`
returned.
