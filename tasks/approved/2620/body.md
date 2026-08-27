---
title: curate .claude/agent-memory/code-reviewer-lean/MEMORY.md under the 24KB agent-memory
  index FAIL cap (41KB on main; reds worktree spec-freshness sync commits + silently
  truncates the lean twin's loaded lessons)
kind: infra
tags: []
created_at: '2026-08-27T06:49:27Z'
has_clean_result: false
origin_prompt: 'Auto-filed by the #2610 /issue session: Step 9c pre-gate spec-freshness
  sync commit FAILed rc=1 on the workflow-lint-agent-memory-index-size hook for this
  main-side file.'
workflow: v1
---
# `.claude/agent-memory/code-reviewer-lean/MEMORY.md` on main is 41,371 bytes — past the 24,000-byte agent-memory index FAIL cap — and reds every worktree spec-freshness sync commit that includes it

## The defect

`workflow_lint.py`'s agent-memory index size check (#1891) FAILs any `MEMORY.md` over
24,000 bytes (the loader truncates the always-loaded index at ~25,000 bytes, silently
dropping the newest lessons). The copy of
`.claude/agent-memory/code-reviewer-lean/MEMORY.md` currently ON MAIN measures 41,371
bytes — 72% over the FAIL cap — so:

1. The `code-reviewer-lean` agent's always-loaded index is silently truncated at ~25 KB:
   roughly the newest 16 KB of its lessons never load. The lean twin exists precisely for
   context-constrained respawns, so an oversized index is self-defeating.
2. Every worktree spec-freshness sync (the `/issue` Step 5a family-atomic block and its
   Step 9c pre-gate re-run) that pulls the file from origin/main FAILs its sync commit on
   the `workflow-lint-agent-memory-index-size` pre-commit hook. Observed on #2610's
   Step 9c pre-gate re-sync (2026-08-27): the sync commit died rc=1 on exactly this file
   and the orchestrator had to exclude it by hand (revert to branch-era copy) to land the
   other ~20 synced files.

Two more indexes are in the WARN band (>20,000 bytes) and will cross the FAIL cap on
their current growth path: `.claude/agent-memory/codex-code-reviewer/MEMORY.md`
(21,545 B) and `.claude/agent-memory/experiment-implementer/MEMORY.md` (21,235 B).

## The fix (per the #1891 curation recipe the lint message names)

Curate `.claude/agent-memory/code-reviewer-lean/MEMORY.md` to under 20,000 bytes: trim
each index hook to ~1 line (≤~150 chars), move the detail into the pointed-to per-entry
file (creating per-entry files where a row has none), and merge duplicate/sibling rows.
No lesson content is deleted — detail moves to per-entry files; only the always-loaded
index shrinks. Opportunistically apply the same pass to the two WARN-band indexes above
if cheap.

Also worth checking (secondary): how a 41 KB index landed on main past the pre-commit
hook — likely appended by sessions whose commits bypassed the hook path or accreted
before the FAIL threshold existed. If a durable gap is found (e.g. a commit path that
skips the lint), name it in the clean-up commit message; no separate enforcement change
is in scope here.

## Acceptance

- `.claude/agent-memory/code-reviewer-lean/MEMORY.md` < 20,000 bytes on main, with the
  displaced detail present in per-entry files linked from the index.
- The no-flags `workflow_lint.py` run PASSes with zero agent-memory-index FAIL rows.
- A worktree spec-freshness sync including the file commits cleanly (reproduce: stage
  the file in any worktree and run the pre-commit hook).

## Provenance

Observed while driving task #2610 through /issue (Step 9c pre-gate spec-freshness
re-sync, 2026-08-27). Grounded by the failing hook output (rc=1,
`workflow-lint-agent-memory-index-size`) and byte counts from that run.
