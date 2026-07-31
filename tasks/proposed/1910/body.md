---
title: 'workflow-fix: fact-checker verifies realized grain of row-grain reuse rows'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2d5476067c0b
created_at: '2026-07-31T05:12:42Z'
has_clean_result: false
origin_prompt: '#1900 crash round 4: plan floor written vs assumed row range; fact-checker
  verified existence not grain; delta_tf pos.jsonl realized exactly 20 rows/mix; add
  grain-count duty to the fact-checker HF-reuse-row instructions'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson candidate
raised on task #1900 (emitting agent: experiment-implementer, crash-fix round 4;
`epm:new-bug-class v1 plan_row_count_floor_vs_realized_grain`).

## Goal

Extend the adversarial-planner Phase-1.5 fact-checker HF-reuse-row
instructions to verify the REALIZED GRAIN (row/line counts) of
row-grain-consuming reuse rows — not just existence at the pin.

## Workflow gap

- **Bug observed:** plan #1900 assumed "~50–300 positive rows/mix" for the reused `delta_tf/<mix>/pos.jsonl`; the fact-checker verified the files EXIST at the pinned revision but never counted rows. Realized grain was exactly 20 rows/mix, and a plan-derived 40-row hard floor crashed fellows job 16055 (one burned launch + a crash-fix round).
- **Why it is a workflow gap:** the fact-checker template (adversarial-planner SKILL.md § Phase 1.5, "For EACH HF reuse row: verify existence with the Python Hub API...") mandates existence/probe-at-pin only; a reuse row whose row/line count feeds a plan floor, sizing arithmetic, or subset draw needs its grain counted (a one-file `hf_hub_download` + `wc -l` is seconds). This is the grain sibling of the #1345 probe-at-pin rule the template already carries.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'realized grain\|count rows' .claude/skills/adversarial-planner/SKILL.md` → 0 hits (absence-of-guard claim — 0-hit in-target IS the evidence) (2026-07-31). Landed-fix history: recent SKILL.md commits carry no grain-count semantics.

## Proposed change (candidate diff sketch — refine in planning)

```
  For EACH HF reuse row: verify existence with the Python Hub API (...) at the
  pinned revision, per named stem/path ...
+ For a ROW-GRAIN-CONSUMING reuse row — a file whose row/line count feeds a
+ plan floor, sizing arithmetic, per-mix quota, or subset draw — ALSO verify
+ the realized grain: hf_hub_download the file (or one representative per
+ family) at the pin and COUNT its rows; a floor/sizing figure resting on an
+ assumed range with no counted basis is UNVERIFIED (#1900: assumed 50-300
+ rows/mix, realized exactly 20; a 40-row hard floor killed the launch).
```

## Scope / surfaces

- Primary target: `.claude/skills/adversarial-planner/SKILL.md`
- Consider the sibling surface `.claude/agents/planner.md` (reuse fitness self-attestation) — grep both for the pattern before editing; the planner's own §12 "how to verify" conventions may warrant a matching line.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/skills/adversarial-planner/SKILL.md
- fingerprint: 2d5476067c0b

Origin: #1900 crash-fix round 4 failure-lesson (root_cause_confirmed) + `epm:new-bug-class v1 plan_row_count_floor_vs_realized_grain`.
