---
title: 'workflow-fix: size boot disk vs end-of-run accumulation'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bf9368243cc3
- daily-auto-filed
created_at: '2026-07-19T07:07:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): Three #1481 GCP lanes died
  the same day to boot-disk headroom exhaustion after all 24 training cells'' retained
  outputs accumulated (disk-headroom-exhaustion:200gb-boot-disk); section-9 disk sizing
  covers dose-ladder retention + merge artifacts but not fan-out accumulated cell
  outputs (c3-P7).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c3-P7). Route-2 filing.

## Goal

Make plan §9 size the boot disk against the END-of-run ACCUMULATED footprint
of all fan-out phases' RETAINED outputs (not just the dose-ladder retained
set + transient merge high-water mark), OR wire between-phase reaps of
consumed per-cell training outputs into the driver.

## Workflow gap

- **Bug observed:** three #1481 GCP lanes died the same day to disk-headroom
  exhaustion — marker-a ~5.5h in (47.6 GB free < 60 GB required on the 200 GB
  boot disk, AFTER all 24 training cells completed), impolite + marker-b
  within 10 min (assert_tag `disk-headroom-exhaustion:200gb-boot-disk`).
- **Why it is a workflow gap:** `plan-compute-sizing.md` §9 sizes the
  merge-disk budget and the dose-ladder RETAINED set + transient high-water
  mark, but has no rule that a wide FAN-OUT's retained per-cell outputs
  ACCUMULATE across all cells and must be summed against the boot disk at
  end-of-run — nor a between-phase reap of consumed per-cell outputs
  (`clean_experiment_downloads --incremental` reaps hf_dl download caches,
  not per-cell training OUTPUTS).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -cin 'accumulated footprint\|end-of-run\|fan-out.*retained\|per-cell output' .claude/rules/plan-compute-sizing.md` → 0 hits (the retention language is scoped to the dose-ladder "size disk to the RETAINED set" + the transient merge high-water mark; no fan-out end-of-run accumulation rule) (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
# plan-compute-sizing.md §9 disk sizing: add a fan-out end-of-run block —
+ For a fan-out run (N cells each retaining outputs), size the boot disk to
+ the SUM of every cell's RETAINED output footprint at end-of-run (not one
+ cell's), plus the transient high-water mark — OR declare a between-phase
+ reap of each cell's CONSUMED outputs (safety contract: store/ + eval_results
+ never touched) wired into the driver so peak stays bounded. A §9 estimate
+ that assumes single-cell retention on a multi-cell fan-out is a REVISE.
```

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md`
- Add the fan-out accumulation block near the existing ladder-retention +
  merge-disk budget spans; cross-reference `critic.md` Methodology lens item
  10 so a silently-undersized fan-out disk is a REVISE.

## Constraints / invariants

- Workflow-surface only. The between-phase reap must inherit the existing
  `clean_experiment_downloads` safety contract (`store/` + `eval_results/`
  never deleted).
- `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/plan-compute-sizing.md
- fingerprint: 867212c8d4e2

Surfaced problem (c3-P7): three #1481 GCP lanes killed same-day by 200 GB
boot-disk exhaustion after all 24 training cells' retained outputs
accumulated (`disk-headroom-exhaustion:200gb-boot-disk`).
