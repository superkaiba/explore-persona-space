---
title: 'daily-fix: review names batched helper for fit loops'
kind: infra
tags:
- wf-fix
- wf-fix-fp:49321aa6268a
- daily-auto-filed
created_at: '2026-07-16T07:21:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): Serial inner loops shipped
  past plan AND review twice (#1332 killed mid-run and vectorized; #825''s reused
  MLP helper ran 120 serial CPU SGD fits)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

The code-review throughput check REQUIRES the diff (or review verdict) to name the batched helper implementing any per-layer/per-cell/per-fold fit loop, or an explicit not-batchable justification — absence is a FAIL, not a note.

## Workflow gap

- **Bug observed:** serial inner loops shipped past plan AND review twice in one day: #1332's serial per-layer loop had to be killed mid-run and vectorized (caught only by the mid-run ≥2× deviation check; 6d38a307 14:28-14:30Z), and #825's reused issue_779 MLP helper ran 120 serial CPU SGD fits after passing plan review (09f28ede 04:19-04:20Z).
- **Why it is a workflow gap:** the reviewer's existing throughput checks are conditional — Step 0.68 fires only when the plan/body NAMES a helper, and the compute-throughput anti-pattern list flags specific shapes (batch-1 forwards, generate loops) — so a per-cell fit loop with no plan-named twin and none of the listed shapes passes with at most a note.
- **Severity:** medium
- verified-at-filing: `grep -n 'batched' .claude/agents/code-reviewer.md` → Step 0.68 "Named-helper adherence check" (L804-825: fires only for helpers the body/plan names by `module::fn`; records N/A when none named) and "Compute-throughput anti-patterns" (L942-961: flags batch-1 forwards / GPU→CPU transfers / HF generate / per-row serialization — no per-cell FIT-loop batched-helper requirement) — the proposed unconditional requirement for fit loops is absent from both (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend `.claude/agents/code-reviewer.md`'s compute-throughput check (the anti-patterns block at L942, coordinating with Step 0.68 at L804): for ANY diff introducing a per-layer/per-cell/per-fold/per-draw fit or dense-factorization loop, the review verdict must name the batched helper implementing the inner loop (e.g. `vectorized_mlp_skill.py`, batched `perm_null_draws`, Gram/dual-space ridge) or record an explicit not-batchable justification — absence is a Major/FAIL finding, not a note, regardless of whether the plan named a helper. Mirror into the v2 `efficiency-critic` implementation-mode lens if trivially co-editable.

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md` (anchors: Step 0.68 L804; compute-throughput anti-patterns L942)
- Secondary: `.claude/agents/efficiency-critic.md` (v2 implementation mode)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 49321aa6268a

- workflow_fix_target: .claude/agents/code-reviewer.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 6d38a307 (#1332) 14:28-14:30Z (batch 04 P13); 09f28ede (#825) 04:19-04:20Z (batch 08 P8 leg a).
