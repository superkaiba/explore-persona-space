---
title: 'workflow-fix: cap-headroom check in check_agent_spec_size'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:44eb14dc87be
created_at: '2026-07-04T07:12:30Z'
has_clean_result: false
origin_prompt: 'source: prose-followup (statistics critic, Phase 2; mechanizable:
  yes, conditional ''if recurs''). Proposed: fold a max-headroom hygiene check into
  check_agent_spec_size (cap must sit ≤~1-3KB above realized file size when raised)
  — target_file: scripts/workflow_lint.py. Prevents a loose cap-raise silently defeating
  the regrowth ratchet. routed: parked — workflow_fix_target recursion guard (log-and-notify
  only). Note: #948''s own code-reviewer can enforce the ≤1KB headroom this round
  via Rule 9 grep + wc -c.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

fold a max-headroom hygiene check into check_agent_spec_size (cap must sit <=~1-3KB above realized file size when raised)

## Workflow gap

- **Bug observed:** a loose cap-raise can silently defeat the agent-spec-size regrowth ratchet; nothing asserts the raised cap sits close to the realized file size
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

fold a max-headroom hygiene check into check_agent_spec_size (cap must sit <=~1-3KB above realized file size when raised)

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: 44eb14dc87be

source: prose-followup (statistics critic, Phase 2; mechanizable: yes, conditional 'if recurs'). Proposed: fold a max-headroom hygiene check into check_agent_spec_size (cap must sit ≤~1-3KB above realized file size when raised) — target_file: scripts/workflow_lint.py. Prevents a loose cap-raise silently defeating the regrowth ratchet. routed: parked — workflow_fix_target recursion guard (log-and-notify only). Note: #948's own code-reviewer can enforce the ≤1KB headroom this round via Rule 9 grep + wc -c.
