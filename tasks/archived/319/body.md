---
title: Harden 'no code on pods' + 'no dirty-tree pod-pull' rules at the hook layer
kind: infra
tags: []
created_at: '2026-05-07T20:30:20.000Z'
has_clean_result: false
sagan_id: b186185d-b85b-454b-bd1d-1045dc857bda
sagan_number: 319
priority: normal
legacy_why_unset: true
---
**Goal**

Convert two existing project rules — *"all code changes on local VM, never on pods"* and *"don't `git pull` on a pod over a dirty local tree"* — from memory-enforced norms into deterministic PreToolUse-hook warnings. Workshop integration plan, Phase 3.

**Context**

After the AI Coding Workshop integration audit, only Phase 3 survived the flow-fit check (Phase 1 sharpenings landed in `CLAUDE.md` directly; Phase 2's `/review` skill was dropped because the `/issue` workflow already routes every code change through `code-reviewer`). This issue is the only proposal that adds new enforcement surface, so it goes through the disciplined `/issue` path.

**Scope**

1. **Edit/Write hook on `/workspace/` paths** — extend the existing `PreToolUse(Edit|Write)` hook in `.claude/settings.json` to *warn* (not block) when `tool_input.file_path` resolves under `/workspace/...`. Rationale: Edit/Write tools edit local files; if a local edit lands in `/workspace/`, that's a stale symlink or misconfigured clone — flag it. Remote edits go through `mcp__ssh__ssh_execute` and are unaffected.
2. **Bash hook on dirty-tree pod-pull** — extend the existing `PreToolUse(Bash)` hook to detect `ssh epm-issue-* '... git pull ...'` patterns and warn if `git status --porcelain` on the local repo is non-empty.

**Non-goals**

- Hard blocking. Warnings only — false positives on hooks are expensive.
- Touching the existing experiment-script enforcement hook (already works, don't regress it).
- Adding `/review`, `/clarify`, or `SessionEnd` hooks (dropped after flow-fit audit).

**Verification**

- Edit hook fires on a synthetic Edit to `/workspace/foo.py`; stderr contains the expected warning string. Hook stays silent on Edit to `src/explore_persona_space/foo.py`.
- Bash hook fires on `ssh epm-issue-137 'cd /workspace/explore-persona-space && git pull'` only when local `git status --porcelain` is non-empty.
- Existing experiment-script enforcement still blocks `python scripts/train.py` without `.epm-authorized` (regression check).

**Resource estimate**

~30 min for diff; one round of `code-reviewer`. No pod time, no GPU.

**Open questions for the planner**

- Are the two warnings cohesive enough for one issue, or should they split into two `type:infra` issues with separate code-review rounds?
- For the dirty-tree check, is shelling out to `git status --porcelain` from inside the hook acceptable latency on every Bash call, or should it cache?
