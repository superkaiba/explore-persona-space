---
title: 'workflow-fix: GCP crash-persist no-fire on #811 crashes: inv'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1dfb7a54aebd
- daily-auto-filed
created_at: '2026-07-09T06:56:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Both 2026-07-04 #811 GCE
  crashes left NO issue811_partial/att-20260703-172624/ prefix on the HF data repo
  despite the #854 300s-bounded, transcript-uploaded-last _eps_persist_diagnostics
  — the promised diagnosability was absent and the #811 diagnosis cycles were manual.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1055 (park_form: recursion-guard).

## Goal

Investigate why the #854-hardened _eps_persist_diagnostics left no issue811_partial/<attempt>/ prefix on either 2026-07-04 #811 GCE crash, and harden the crash-persist path so a persist no-fire is itself observable off-VM.

## Workflow gap

- **Bug observed:** Both 2026-07-04 #811 GCE crashes left NO issue811_partial/att-20260703-172624/ prefix on the HF data repo despite the #854 300s-bounded, transcript-uploaded-last _eps_persist_diagnostics — the promised diagnosability was absent and the #811 diagnosis cycles were manual.
- **Why it is a workflow gap:** The crash-persist trap is the ONLY diagnostics channel surviving the GCE --instance-termination-action=DELETE; a silent persist no-fire reproduces the exact blindness #854 was built to close.
- **Confidence (emitter):** medium (alternatives critic, Phase 2)

## Proposed change (candidate diff sketch — refine in planning)

Audit the EXIT-trap wiring for the #811 attempts' crash modes (trap not installed? persist guarded-out? pre-trap poweroff?); add an off-VM persist-attempted breadcrumb (e.g. an early tiny marker upload at trap entry) so absence distinguishes 'persist never ran' from 'persist ran and failed'.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- origin: parked candidate on task #1055 at 2026-07-05T18:48:16Z

parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target — see workflow-fix-on-bug § Recursion guard. source: prose-followup (alternatives critic, Phase 2). NOT auto-routed (this session is itself a workflow-fix session); logged for the next orchestrator/human pass.

Candidate (synthesized from critic prose): the #854-hardened GCE crash-persist (_eps_persist_diagnostics) left NO issue811_partial/att-20260703-172624/ prefix on either 2026-07-04 #811 crash despite the 300s bound + transcript-uploaded-last design — the very diagnosability #854 promised was absent, which is half of why the #811 diagnosis cycles were manual. Suggested target: src/explore_persona_space/backends/gcp.py (_eps_persist_diagnostics) — investigate why the persist died/skipped on those attempts (serial evidence unreadable post-DELETE; the #854 fix-engaged signal never fired). Confidence: medium.

