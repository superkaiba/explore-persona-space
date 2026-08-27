---
title: Curate code-reviewer-lean agent-memory index under the 24KB lint cap (live
  no-flags FAIL redding Step 9c gates)
kind: infra
tags: []
created_at: '2026-08-27T10:20:42Z'
has_clean_result: false
origin_prompt: 'surfaced by #2587 9a-ter inline round lint gate 2026-08-27'
workflow: v1
---
## Goal

Curate `.claude/agent-memory/code-reviewer-lean/MEMORY.md` back under the #1891 agent-memory INDEX size cap (WARN >20 KB / FAIL >24 KB in `scripts/workflow_lint.py`). The file is over the 24 KB FAIL cap on `main`, so every no-flags `workflow_lint.py` run — the /issue Step 9c payload lint gate and every inline payload lint gate fleet-wide — exits 1 on a FAIL naming this file until it is curated. Surfaced by the #2587 9a-ter inline round (2026-08-27), whose push was only clean because the payload-attribution rule scopes the gate to round-committed files.

## What to do

1. Curate the index per the MEMORY.md convention (one line per memory, pointers only, no memory content in the index): move any inlined content into per-memory files, drop stale rows, keep the index a pure pointer list under 20 KB.
2. Confirm with a no-flags `uv run python scripts/workflow_lint.py` that the FAIL line is gone (remaining WARNs elsewhere are out of scope).
3. Commit by explicit path on a worktree branch and land on main per the standard route.

## Context

Prevention sibling: #2562 (proposed) adds the missing per-memory-FILE ratchet; this task is only the live-red curation. Do not implement the ratchet here.
