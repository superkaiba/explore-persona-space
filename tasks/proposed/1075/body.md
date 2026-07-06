---
title: 'workflow-fix: verify_plan heuristic — GPU basis vs routed machine'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2559c56f77b7
created_at: '2026-07-06T05:18:33Z'
has_clean_result: false
origin_prompt: 'Prose workflow-fix follow-up from the #1073 Methodology critic: add
  a verify_plan.py heuristic flagging §9 basis cells naming a GPU different from the
  intent-routed machine under auto with no scaling token (3rd recurrence: #599, #833,
  #1073).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1073 (emitting agent: critic, Methodology lens — prose follow-up).

## Goal

Add a `verify_plan.py` heuristic check that flags any §9 compute-table `basis` cell naming a GPU (e.g. `H100`) different from the machine the Spec line / intent routes under `auto` (GCP `INTENT_TO_MACHINE`), when the row contains no explicit scaling token (`×`, "scaled").

## Workflow gap

- **Bug observed:** plan §9 wall-time bases were H100-measured while the auto lane routes A100-80 with no stated cross-GPU scaling; third recurrence of the class (#599, #833, #1073).
- **Why it is a workflow gap:** `.claude/rules/plan-compute-sizing.md` § "Cost wall-time against the machine the router will ACTUALLY provision" is unconditional prose, but `verify_plan.py` has no mechanical check for it — the class keeps sailing through the mechanical pre-pass and is caught only by the (fallible) LM critics.
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)

+ In scripts/verify_plan.py, add check c26_gpu_basis_vs_routed_machine:
+   - detect the routed machine: --intent token in the plan (map via backends/gcp.py INTENT_TO_MACHINE import or a static mirror) when backend is auto/absent
+   - for each §9 table row, if the basis cell names a GPU model string (H100/A100/L4/H200) differing from the routed machine's GPU
+     AND the row lacks a scaling token (`×`, 'scaled', 'per-step rate'), emit FAIL (or WARN initially)
+   - N/A escape: 'N/A — basis measured on the routed machine' or an explicit backend pin matching the named GPU

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'INTENT_TO_MACHINE\|routed machine' .claude/ scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; keep consistent with `.claude/rules/plan-compute-sizing.md`.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 2559c56f77b7

Surfaced prose (verbatim, from the Methodology critic's report on #1073):
"`scripts/verify_plan.py`: a heuristic check flagging any §9 compute-table `basis` cell that names a GPU (e.g. `H100`) differing from the machine the Spec line / intent routes under `auto` (GCP `INTENT_TO_MACHINE`), when the row contains no explicit scaling token (`×`, "scaled"). This is the third recurrence of the class (#599, #833, this plan); the check is grep-shaped once the intent→machine map is imported."
