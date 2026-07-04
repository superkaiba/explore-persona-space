---
title: 'workflow-fix: gotchas.md — bf16 single-position equivalence-gate calibration'
kind: infra
tags:
- wf-fix
- wf-fix-fp:abf4c3f22139
created_at: '2026-07-03T15:11:43Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #779 r12 (epm:failure-lesson
  v6): bf16 padded-batch equivalence-gate calibration trap for single-position hidden-state
  captures'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes`
failure-lesson raised on task #779 (emitting agent: experiment-implementer,
crash-fix round 12).

## Goal

Add a `.claude/rules/gotchas.md` entry: bf16 padded-batch equivalence-gate
calibration for single-position hidden-state captures (early-layer per-layer
bar + flattened bar with measured headroom; fp32 re-probe before loosening any
tolerance).

## Workflow gap

- **Bug observed:** #779 pass-2 capture crashed its own batched-vs-serial
  equivalence gate at cos_min 0.99877 < 0.999 with a bug-free capture path —
  the bar was calibrated on span-mean summaries and had no headroom for
  single-position states' depth-amplified layer-27 bf16 padded-batch jitter.
- **Why it is a workflow gap:** gotchas.md documents the known codebase traps
  agents load when writing capture/eval code; batched-vs-serial equivalence
  gates over hidden-state captures are a recurring rig pattern (pass-1/pass-2
  here, `assert_batched_capture_equivalence` elsewhere) and the
  span-mean-calibrated-bar trap will recur on the next single-position capture
  rig.
- **Confidence (emitter):** high (fp32-confirmed root cause).

## Proposed change (candidate diff sketch — refine in planning)

```
+ ### bf16 padded-batch equivalence gates — calibrate on single-position states
+ A batched-vs-serial cosine bar calibrated on SPAN-MEAN summaries has no
+ headroom for SINGLE-POSITION hidden states: depth-amplified bf16
+ padded-batch kernel jitter concentrates in the LAST layer (#779 r12: worst
+ cell cos 0.99877, layer 27 alone 0.9969, layer 0 at 0.999999; fp32 probe =
+ 1.000000 everywhere — no batching bug). Split the gate: per-layer cos >=
+ 0.999 over layers 0-3 (real pad/mask/index bugs corrupt layer 0 to <=0.84)
+ + flattened all-layer cos with >=4x measured headroom (0.995). Attribute a
+ marginal miss with an fp32 re-probe BEFORE loosening anything. Worked
+ example: scripts/issue779_capture_answer_summaries_pass2.py::equivalence_gate_p2
+ (commit 7b35377edc).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'equivalence' .claude/ CLAUDE.md`) and update every hit that
  documents batched-vs-serial gate recipes; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: abf4c3f22139

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: pass2-capture equivalence gate (issue779_capture_answer_summaries_pass2.py::equivalence_gate_p2)
lesson: A batched-vs-serial cosine bar calibrated on SPAN-MEAN summaries (pass-1 realized 0.999748 vs bar 0.999) has no headroom for SINGLE-POSITION hidden states, where depth-amplified bf16 padded-batch kernel jitter concentrates in the last layer (worst cell 0.998770, layer 27 alone at 0.9969, layer 0 at 0.999999; fp32 = 1.000000 everywhere, proving no batching bug). Gate early layers tightly (per-layer cos >= 0.999 over layers 0-3 — real pad/mask/index bugs corrupt layer 0 to <=0.84) and bound the flattened all-layer cosine with measured headroom (>=0.995) instead of a one-size bar.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
