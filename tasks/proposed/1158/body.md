---
title: 'workflow-fix: Recovery rule for autocompact-thrash subagent '
kind: infra
tags:
- wf-fix
- wf-fix-fp:69cd3c45eda4
- daily-auto-filed
created_at: '2026-07-09T06:57:40Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): When a reviewer/implementer
  subagent dies to autocompact thrash and the transcript shows NO oversized tool result,
  orchestrators re-tighten read bounds (which does not help — the pressure is fixed
  overhead) instead of respawning with micro-scoped work + the DEFAULT model; #1090
  forensics (events.jsonl L247): read-bounded briefs did not help, default-model spawns
  compacted successfully, 3/6 sonnet'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1115 by a recursion-guarded workflow-fix session.

## Goal

Add an orchestrator-side recovery rule to the /issue SKILL.md subagent-dispatch guidance (and the CLAUDE.md subagent section, as the sibling of refusal rung (b2)): on an autocompact-thrash death with no oversized tool result in the transcript, respawn with MICRO-SCOPED work + the DEFAULT model instead of re-tightening read bounds.

## Workflow gap

- **Bug observed:** When a reviewer/implementer subagent dies to autocompact thrash and the transcript shows NO oversized tool result, orchestrators re-tighten read bounds (which does not help — the pressure is fixed overhead) instead of respawning with micro-scoped work + the DEFAULT model; #1090 forensics (events.jsonl L247): read-bounded briefs did not help, default-model spawns compacted successfully, 3/6 sonnet spawns thrashed.
- **Why it is a workflow gap:** The workflow has a refusal-death recovery ladder (CLAUDE.md rungs a-f) but no autocompact-thrash sibling; without it each orchestrator improvises the wrong fix (tighter read bounds) and burns rounds.
- **Confidence (emitter):** medium
- **Sweep verification (2026-07-08):** grep for 'micro-scop'/'MICRO-SCOP'/'default model' recovery guidance in SKILL.md + CLAUDE.md returns nothing (2026-07-08); SKILL.md's only autocompact rule (L2319, durable-verdict-first) covers verdict recovery after a thrash death, not respawn strategy. #1115 itself landed the read-hygiene context-budget sections (04e9c41911) — the complementary spec-side fix — leaving this orchestrator-side rule unimplemented.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; add a short 'autocompact-thrash respawn' paragraph next to the durable-verdict-first rule: check transcript for an oversized tool result; absent one, respawn micro-scoped + default model — per-subagent model pins are prompt-cache-safe)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, CLAUDE.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, CLAUDE.md
- origin: parked candidate on task #1115 at 2026-07-07T21:13:49Z

Verbatim parked note:

> parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug § Recursion guard). Surfaced by the Alternatives critic (prose, round 1): orchestrator-side recovery rule for autocompact-thrash subagent deaths — when the transcript shows NO oversized tool result, respawn with MICRO-SCOPED work + DEFAULT model instead of re-tightening read bounds (sibling to the refusal rung (b2)). Target: .claude/skills/issue/SKILL.md subagent-dispatch guidance / CLAUDE.md subagent section. Evidence: #1090 events.jsonl L247 forensics (fixed-overhead pressure; read-bounded brief did not help; default-model spawns compacted successfully, 3/6 sonnet spawns thrashed). NOT auto-routed by this session; for the next human/orchestrator pass.
