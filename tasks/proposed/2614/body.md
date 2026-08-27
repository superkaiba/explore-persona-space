---
title: 'Curate oversized agent-memory indexes on main (code-reviewer-lean MEMORY.md
  41KB hard-FAILs #1891 lint; blocks Step-10d agent-memory re-sync commits)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-27T01:57:07Z'
has_clean_result: false
origin_prompt: 'Step 10d post-gate re-sync for issue-2607 aborted: pre-commit workflow-lint-agent-memory-index-size
  FAILed on origin/main''s own .claude/agent-memory/code-reviewer-lean/MEMORY.md (41,371
  bytes > 24,000)'
workflow: v1
---
# Curate oversized agent-memory indexes on main — `.claude/agent-memory/code-reviewer-lean/MEMORY.md` at 41,371 bytes hard-FAILs the #1891 lint and blocks every Step-10d agent-memory re-sync commit

## Goal

Bring every `.claude/agent-memory/<agent>/MEMORY.md` on `main` back under the #1891 size budget (WARN > 20,000 bytes, FAIL > 24,000 bytes; the loader truncates the always-loaded index at ~25,000 bytes, silently dropping the newest lessons), and keep them there.

## Incident (2026-08-27, issue-2607 Step 10d)

The Step-10d post-gate spec-freshness re-sync staged the drifted `.claude/agent-memory` set from origin/main; the sync commit's pre-commit hook `workflow-lint-agent-memory-index-size` FAILed on `.claude/agent-memory/code-reviewer-lean/MEMORY.md` (41,371 bytes > 24,000 FAIL threshold), aborting the merge attempt fail-closed. The offending content is main's OWN copy — no branch payload involved. Two additional files are in the WARN band: `.claude/agent-memory/codex-code-reviewer/MEMORY.md` (21,545 bytes) and `.claude/agent-memory/experiment-implementer/MEMORY.md` (21,235 bytes).

Fleet impact while unfixed:
1. The code-reviewer-lean index is silently truncated at load time (~25,000 bytes) — its newest lessons never reach the agent.
2. Any Step-10d post-gate re-sync (and any other explicit-path commit that stages that file) hard-FAILs, so agent-memory freshness syncs are structurally blocked fleet-wide; sessions must mark the family dirty and merge with stale memory copies (the #2607 remediation).
3. The oversize state landed on main through some path that bypassed the hook (the hook only fires when the file is in a staged set), so a recurrence guard is worth considering.

## Acceptance criteria

1. `.claude/agent-memory/code-reviewer-lean/MEMORY.md` on main is curated per #1891: each index hook trimmed to ~1 line (<= ~150 chars), detail moved into the pointed-to per-entry files, duplicate/sibling rows merged; final size <= 20,000 bytes (below the WARN band, headroom for growth). No lesson content deleted — detail relocates to per-entry files.
2. The two WARN-band indexes (`codex-code-reviewer`, `experiment-implementer`) receive the same curation pass (target <= 20,000 bytes) or a recorded decision to leave them in the WARN band.
3. `uv run python scripts/workflow_lint.py` (no-flags run) reports zero agent-memory index-size FAILs on the landed tree.
4. Brief root-cause note in the clean-result/test-verdict: which commit(s) grew code-reviewer-lean past 24 KB and why the hook did not fire there (e.g. a path that committed without staging the index, or a merge that unioned rows).
