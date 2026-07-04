---
title: 'workflow-fix: gotchas.md cuDNN-TF32 parity-gate trap (float64-clone gate recipe)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:67ba134bff0d
created_at: '2026-07-04T04:25:03Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha candidate from #841 r21 (see body Provenance)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #841 round 21 (emitting agent: experiment-implementer).

## Goal

Add a `.claude/rules/gotchas.md` entry documenting the cuDNN-TF32 vs fp32-matmul parity-gate divergence trap and the float64-clone gate recipe.

## Workflow gap

- **Bug observed:** issue841 gru verify gate failed with max abs diff 1.86e-4 vs 1e-4 tol because cudnn TF32 defaults on while matmul TF32 defaults off
- **Why it is a workflow gap:** gotchas.md is the durable codebase-trap reference for implementers; this trap (empirically confirmed with a 3-probe differential on pod-841) plausibly recurs on any H100/A100 numerical-parity assert comparing cuDNN kernel paths to plain matmuls, and no existing entry covers it (the #667 L4/A100 entry is a different, cross-hardware trap).
- **Confidence (emitter):** high (root_cause_confirmed: yes; probes: TF32-on 1.86e-4 FAIL, cudnn.allow_tf32=False 1.88e-6 PASS, float64 1.11e-16)

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new bullet: **fp32 kernel-parity gates comparing a cuDNN RNN path (nn.GRU/nn.LSTM, even length-1 unroll) vs plain ATen matmuls (GRUCell / hand-rolled reference) diverge ~2e-4 on H100/A100 with CORRECT math** — `torch.backends.cudnn.allow_tf32` defaults TRUE while `torch.backends.cuda.matmul.allow_tf32` defaults FALSE. RULE: run the parity gate on a float64 deep-copy clone at a TIGHT ~1e-8 tol (measured fp64 noise 1.1e-16) + keep a loose (~1e-2) fp32-vs-fp64 cross-check for the production dtype; never loosen the fp32 tol to ~1e-3. Include the 3-probe differential diagnosis recipe. Long-form memory: `.claude/agent-memory/experiment-implementer/feedback_cudnn_tf32_fp32_parity_gate.md`. Incident #841 r21 (att-8 crash), fix commit 5ab0f95f7a.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'allow_tf32' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan. (Known at filing time: gotchas.md only — the #667 L4 entry mentions TF32 tangentially and stays distinct.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 67ba134bff0d

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: issue841 gru verify gate failed with max abs diff 1.86e-4 vs 1e-4 tol because cudnn TF32 defaults on while matmul TF32 defaults off
why_workflow_gap: gotchas.md is the durable trap reference; this cuDNN-TF32-vs-fp32-matmul parity-gate trap recurs on any H100/A100 parity assert and no entry covers it
proposed_change: Add gotchas.md entry: fp32 kernel-parity gates comparing cuDNN RNN paths vs ATen matmuls diverge ~2e-4 under default cudnn.allow_tf32=True on H100/A100 — run parity gates on a float64 clone with ~1e-8 tol
diff_sketch: |
  + new gotchas.md bullet per the failure-lesson: TF32 default asymmetry (cudnn TRUE, matmul FALSE),
  + the ~2e-4 divergence signature, the float64-clone-at-1e-8-tol gate recipe,
  + the loose fp32-vs-fp64 production-dtype cross-check, the 3-probe differential diagnosis,
  + pointer to the agent memory + incident #841 r21.
confidence: high
related_task: #841
<!-- /workflow-fix-candidate -->
