---
title: 'daily-fix: null-statistic gates need measured calibration'
kind: infra
tags:
- wf-fix
- wf-fix-fp:46643e3b4037
- daily-auto-filed
created_at: '2026-08-06T07:23:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): #1491 Gate-1 abs(r2_null)<0.05
  unsatisfiable by construction killed the 0.5B rung; the asserted -3.0 floor killed
  1.5B; null depth is non-monotone'
workflow: v1
---
# daily-fix: pre-registered numeric gates on NULL statistics need a measured calibration basis — the #1491 Gate-1 predicate killed two rungs

## Workflow gap

#1491's approved plan pre-registered Gate-1 as `abs(r2_null) < 0.05`, but the
shuffle-REFIT null it thresholds has expected R² ≈ −1 — "unsatisfiable by construction."
All 8 shards of the 0.5B rung FAILed the mid-shard gate and aborted (GPUs at 0% on an
8×H200 pod at ~$44/h); the first fix replaced it with an ASSERTED fixed floor
(`null_floor=-3.0`) which then killed the 1.5B rung too (`r2_null=-3.40…-3.80`; "My
prediction was **wrong**"). Only the third change (advisory floor) held — implementer
measurement showed the null depth is non-monotone in n/h (double-descent peak at n≈h), so
ANY constant floor was wrong. The defect survived plan approval + multiple code-review
rounds because no review calibrated the gate against a measured null draw.

verified-at-filing: the gate FAIL values, both fix commits (ccc650f42e, 6d5c675a95), and
the double-descent measurement are the recovery miner's probed transcript reads (session
8d7f8b25 rows 2383–2638, incl. the epm:failure v3 diagnosis "unsatisfiable by
construction").

## Proposed change

Add to `.claude/rules/selection-symmetric-nulls.md` (and cross-reference from
`planner.md` §7): any pre-registered numeric gate that thresholds a NULL statistic
requires a MEASURED calibration basis — a 1-cell pilot of the null at production n/d
shape — before production; asserted constants on an unmeasured null distribution are the
#1491 Gate-1 class. Prefer advisory logging over hard-abort for null-side conditions
(the null is diagnostic, not a kill criterion, unless the plan argues otherwise).

## Provenance

- fingerprint: 46643e3b4037

- workflow_fix_target: .claude/rules/selection-symmetric-nulls.md
- origin: /daily 2026-08-04 recovery sweep — miner 2 P2 (probed rows).
