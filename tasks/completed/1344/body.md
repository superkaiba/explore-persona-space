---
title: 'workflow-fix: gotchas entry — smoke parity includes ZeRO-3 process width'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6889c19cc312
created_at: '2026-07-15T10:56:51Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate from #1315 failure-lesson: smoke narrowed ZeRO-3
  FT to 1 process, deterministic OOM at first optimizer step; width is a resource
  dimension of smoke/production parity'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes`
failure-lesson raised on task #1315 (emitting agent: experiment-implementer,
round 3 crash-fix; orchestrator routed).

## Goal

Add a `.claude/rules/gotchas.md` entry: smoke/production parity includes PROCESS WIDTH — a cloned dispatcher that narrows a ZeRO-3 full-FT smoke to `--num_processes 1` OOMs deterministically at the first optimizer step; audit every `if cfg.smoke:` branch that composes launch width / CUDA_VISIBLE_DEVICES when cloning a dispatcher.

## Workflow gap

- **Bug observed:** #1315's p1_train smoke OOMed deterministically (7B full-FT: bf16 weights + grads + fp32 Adam moments ≈ 86 GB unsharded on one 80 GB A100) because the dispatcher cloned from #1112 inherited a `smoke → num_processes 1` narrowing whose validity was HOST-scoped (#1112's smoke ran on a 1-GPU GCE instance; #1315's runs on the 4× A100-80 ft-7b pod).
- **Why it is a workflow gap:** gotchas.md carries the smoke/production-parity trap catalog (#397 class) but only for CODE-PATH divergence; the RESOURCE-dimension (process width / CVD) variant is undocumented, and it cost a full GCE provision + crash + diagnostic-boot cycle (~1.5 h) on 2026-07-15. The agent-memory lesson (feedback_smoke_ft_zero3_width_parity.md, commit 143e2f965d) covers the experiment-implementer; the gotchas entry makes it surface for every agent that touches launch composition.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn "num_processes" .claude/rules/gotchas.md` → 0 hits; `grep -n "smoke" .claude/rules/gotchas.md` → hits cover mock-seam smokes + tiny-real standard only, no width/CVD parity entry (2026-07-15). Absence-of-guard claim — 0-hit in-target result IS the evidence.

## Proposed change (candidate diff sketch — refine in planning)

```
+ ## Smoke/production parity includes PROCESS WIDTH (ZeRO-3 OOM class)
+ A smoke that narrows a ZeRO-3 full-FT launch to `--num_processes 1` OOMs
+ deterministically at the FIRST optimizer step (7B: ~86 GB unsharded weights+
+ grads+fp32 Adam moments vs 80 GB HBM) — single-process ZeRO-3 shards nothing.
+ Smoke keeps the production process width unless the smoke HOST genuinely
+ differs (a parent's narrow-smoke pin is host-scoped, never clone-portable).
+ Audit every `if cfg.smoke:` branch composing launch width / CUDA_VISIBLE_DEVICES
+ when cloning a dispatcher; pin with an arg-composition test asserting
+ `--num_processes <N>` + CVD in BOTH modes (worked example:
+ tests/test_issue1315_dispatch.py::test_ft_launch_width_smoke_invariant; #1315).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'num_processes' .claude/`) and consider whether
  `.claude/agents/experiment-implementer.md`'s clone-audit prose should
  cross-reference the new entry; list hits in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes
  (gotchas.md edits can trip the lessons-index check only if a new rule FILE
  is added — this is an in-file entry, no LESSONS.md row change expected).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 6889c19cc312

bug_observed: issue1315 p1_train smoke OOMed deterministically because the cloned dispatcher narrowed the ZeRO-3 FT launch to num_processes 1 on an 80GB GPU (86GB unsharded optimizer states)
why_workflow_gap: gotchas.md documents code-path smoke divergence (#397) but not the resource-dimension (process width) variant; cost a GCE provision+crash+diagnostic-boot cycle
proposed_change: add a gotchas.md entry: smoke/production parity includes process width - a cloned dispatcher narrowing a ZeRO-3 full-FT smoke to num_processes 1 OOMs deterministically at the first optimizer step; audit if cfg.smoke width branches when cloning
confidence: high
related_task: #1315
