---
title: 'workflow-fix: 6d.0-bis smoke gate: name data-ingestion tiny-'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ca3eaa9e769d
- daily-auto-filed
created_at: '2026-07-09T06:57:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Step 6d.0-bis''s tiny-real
  smoke-evidence standard does not name real-corpus data-ingestion probes, and the
  implementer memory misstates WildChat moderation fields as ''per-turn openai_moderation''
  keys instead of per-turn-ALIGNED top-level list columns.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1113 (park_form: recursion-guard).

## Goal

Extend the Step 6d.0-bis smoke-evidence standard to name data-ingestion tiny-real probes (real-corpus streaming builders) the way it names the #906 GPU-driver class, and correct the implementer memory's WildChat moderation-shape line.

## Workflow gap

- **Bug observed:** Step 6d.0-bis's tiny-real smoke-evidence standard does not name real-corpus data-ingestion probes, and the implementer memory misstates WildChat moderation fields as 'per-turn openai_moderation' keys instead of per-turn-ALIGNED top-level list columns.
- **Why it is a workflow gap:** The #1092 P0 failure class (a streaming filter chain rejecting 100% of real rows while synthetic smokes stay green) is exactly what the smoke-evidence standard exists to force evidence for; the memory line seeds the wrong field-shape assumption.
- **Confidence (emitter):** medium (Alternatives critic + Methodology critic, #1113 Phase 2)

## Proposed change (candidate diff sketch — refine in planning)

(1) SKILL.md 6d.0-bis: add real-corpus streaming/data-ingestion builders to the named tiny-real probe classes (bounded tiny-real streaming probe, kept>0 per dataset, per-filter reject counters). (2) memory: 'per-turn openai_moderation' -> 'per-turn-ALIGNED top-level list columns (not keys inside each turn dict)'.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, .claude/agent-memory/experiment-implementer/feedback_real_corpus_streaming_filters_tiny_real_probe.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, .claude/agent-memory/experiment-implementer/feedback_real_corpus_streaming_filters_tiny_real_probe.md
- origin: parked candidate on task #1113 at 2026-07-07T18:29:07Z

parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard). TWO candidates surfaced by Phase 2 critics, logged not routed:
(1) target_file: .claude/skills/issue/SKILL.md — extend Step 6d.0-bis smoke-evidence standard to name data-ingestion tiny-real probes (real-corpus streaming builders) the way it names the #906 GPU-driver class. Source: Alternatives critic, confidence medium.
(2) target_file: .claude/agent-memory/experiment-implementer/feedback_real_corpus_streaming_filters_tiny_real_probe.md — one-line correction: 'per-turn openai_moderation' → 'per-turn-ALIGNED top-level list columns (not keys inside each turn dict)', matching the fact-checked shape. Source: Methodology critic, confidence medium.
routed: parked (recursion guard); surface on a future non-workflow-fix orchestrator pass.
