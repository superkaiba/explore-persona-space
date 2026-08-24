---
title: 'workflow-fix: agent-memory per-file size ratchet + trim the 350KB codex-code-reviewer
  compose-recipe memory'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T22:35:21Z'
has_clean_result: false
parent_id: 2502
origin_prompt: 'codex-code-reviewer composer flag during #2502 r5 compose (2026-08-24):
  memory file feedback_revision_round_compose_recipe.md bloated to ~342-350 KB, unreadable;
  needs repair/trim + a lint ratchet so it cannot recur.'
workflow: v1
---
# Agent-memory file bloat: `codex-code-reviewer/feedback_revision_round_compose_recipe.md` is 350 KB and unreadable by its own agent; no per-file size ratchet exists

## Overview / Motivation

`.claude/agent-memory/codex-code-reviewer/feedback_revision_round_compose_recipe.md` has grown to **350,521 bytes** (measured 2026-08-24, repo content synced from origin/main). The `codex-code-reviewer` composer reported it as *unreadable* during the #2502 round-5 compose (2026-08-24): the file exceeds what the agent can page in, so the memory is dead weight — worse than absent, because it displaces the working recipe it was meant to preserve and inflates every compose round's fixed overhead (the same fixed-overhead class that drives Class-2 autocompact thrash, #2472).

The lint surface has an agent-memory **INDEX** size ratchet (`MEMORY.md` WARN >20KB / FAIL >24KB, #1891) but **no per-memory-FILE cap** — which is exactly how a single always-appended recipe file grew 15x past the index cap with no signal.

## Deliverables

1. **Repair/trim the file now**: distill `feedback_revision_round_compose_recipe.md` to a working-size recipe (target: within whatever cap item 2 sets; the durable content is the recipe's current form, not its append history — git history preserves the old bytes). Update the `MEMORY.md` index line if the hook changes. Run the gotchas.md no-lost-row discipline in spirit: anything dropped must be genuinely superseded-by-later-rows, not merely old.
2. **Add a per-file size ratchet** for `.claude/agent-memory/**/*.md` in `scripts/workflow_lint.py` (same shape as the #1891 index ratchet: WARN then FAIL; suggested WARN >24KB / FAIL >48KB, grandfathering audited existing files), wired into the same pre-commit surface so the next runaway append fails loud at commit time instead of at compose time months later.
3. **Persist the composer's round lesson** (from the same #2502 r5 compose, currently nowhere durable): *a re-posted smoke-arch marker can carry a verbatim prior-round narrative whose stale rationale sentence contradicts its own new verdict line — pre-adjudicate it as disclosed carry-forward, not a verdict/notes contradiction.* Land it as a small NEW memory file under `.claude/agent-memory/codex-code-reviewer/` (post-trim), indexed in that agent's `MEMORY.md`.

## Provenance

Reported by the `codex-code-reviewer` composer during #2502 code-review round 5 (2026-08-24): "my memory file ... has bloated to 342 KB (unreadable) and needs repair/trim at a safe commit point". Filed by the #2502 orchestrator session per `.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-up). Dedup checked: #2444 (sync-subject clobber) and #2341 (guard pathspec) touch the same path only incidentally — distinct bugs, distinct tasks.

## Provenance

workflow_fix_target: .claude/agent-memory/codex-code-reviewer/feedback_revision_round_compose_recipe.md, scripts/workflow_lint.py
Filed from the #2502 orchestrator session (code-review round 5, 2026-08-24) on a composer-surfaced prose follow-up.
