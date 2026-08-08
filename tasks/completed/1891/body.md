---
title: 'workflow-fix: experiment-implementer MEMORY.md exceeds the always-loaded read
  limit (41KB vs ~24.4KB) — curate + add size guard'
kind: infra
tags:
- wf-fix
created_at: '2026-07-30T19:04:26Z'
has_clean_result: false
origin_prompt: 'boundary-impl report 2026-07-30: MEMORY.md is 41 KB against a 24.4
  KB read limit, so its trailing ~40 percent of entries are silently dropped every
  load — worth a dedicated pass'
workflow: v1
---
## Overview / Motivation

Auto-filed from a surfaced prose follow-up (boundary-ablation implementer report, 2026-07-30): `.claude/agent-memory/experiment-implementer/MEMORY.md` is 41,137 bytes against the ~24.4 KB always-loaded read limit, so the trailing ~40% of its entries are SILENTLY DROPPED on every experiment-implementer spawn — the agent loads a truncated memory and never sees the newest lessons (which append at the end, i.e. exactly the dropped region).

- verified-at-filing: `wc -c .claude/agent-memory/experiment-implementer/MEMORY.md` -> 41,137 bytes; ~162 list/heading entries (2026-07-30).

## Goal

Bring the experiment-implementer agent memory back under the always-loaded read limit without losing lessons, and add a mechanical guard so agent-memory files can never silently exceed the limit again.

## Proposed change

1. Curation pass on `.claude/agent-memory/experiment-implementer/MEMORY.md`: merge duplicate/superseded entries, move long incident narratives into per-entry files under the same directory (pointer-indexed from MEMORY.md, the standard memory-dir shape), keep the index under ~20 KB with headroom.
2. Mechanical guard: extend `scripts/workflow_lint.py` (or the existing agent-spec size ratchet hook, which already WARNs >28KB / FAILs >40KB for `.claude/agents/*.md`) to cover `.claude/agent-memory/*/MEMORY.md` with a threshold tied to the actual loader read limit — an always-loaded file larger than what the loader reads is a silent-truncation bug, not a style issue.
3. Check the other `.claude/agent-memory/*/MEMORY.md` files for the same condition while in there.

## Constraints

- Curation must preserve every load-bearing lesson (merge, never drop silently); the owning agent remains the primary author — this pass is consolidation, not rewriting semantics.
- Workflow-surface only.
- est_gpu_hours: 0
