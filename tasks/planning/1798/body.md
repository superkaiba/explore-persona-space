---
title: 'daily-fix: plan-compute-sizing — draws, multipliers, RAM, po'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5b9e2c8a123d
- daily-auto-filed
created_at: '2026-07-29T07:10:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): five measured sizing failures
  on 2026-07-27/28: #1689''s 200-draw bootstrap battery drove ~240x phase cost and
  only a user order descoped it; #1689''s measured pilot extrapolated ~11x low because
  evals-per-pair was assumed rather than read off the fit code; #1739''s RAM was sized
  on an anchor behavior (2.2x table OOM twice, crash-persist blind); #1738''s ~4x-stale
  capture basis stood in the approved p'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-D P1, group-G P1, group-A P2(a), group-J P6, group-F P1 (5 miners, 3 issues).

## Goal

Fold five measured 2026-07-27/28 sizing failures into plan-compute-sizing.md (+ the vectorize rule's mid-run trigger).

## Workflow gap

- **Bug observed:** (1) #1689 fit_ladder: the 200-draw bootstrap-CI battery was ~99.6% of a 176 GPU-h phase (plan sized 4h; ~30+h spent before the user ordered 'drop bootstrap to 0', ETA 38h→5-6h). (2) The same phase's MEASURED pilot extrapolated ~11x low — the evals-per-pair multiplier was assumed arithmetic, not read off the fit script. (3) #1739: two kernel OOMs (163GiB/170GB) because RAM was sized on an anchor behavior while the hallucination lane's table ran 2.2x larger. (4) #1738: the plan's capture-cost basis was ~4x stale post-pilot with no recorded basis update. (5) #1738's near-dupe screen: pairwise similarity scales with pool^2; the pilot ran below production pool size (4h plan → 20-80h projection, ~5.75h burned), and CPU%-only health reads called the frozen serial screen healthy for hours. (Cost figures are transcript-mined/marker-quoted — verify at plan time against #1689/#1739/#1738 events.)
- **Why it is a workflow gap:** the sizing rule mandates a measured 1-cell pilot but not (a) that the per-cell multiplier be read off the code, (b) draw-count necessity, (c) largest-cell RAM keying, (d) a recorded basis update on pilot deviation, (e) pool-scale pilots for quadratic batteries; the mid-run >=2x trigger forces only the vectorize check (vectorize-many-cell-fits.md:265).
- **Confidence (emitter):** medium-high (each incident marker-backed; exact multipliers transcript-mined)
- verified-at-filing: `grep -n 'Per-cell fit phases' .claude/rules/plan-compute-sizing.md` → line 357 (measured-pilot clause present, no inner-loop-count derivation, no draw-necessity clause — 0 grep hits for 'descope'/'necessity'); `grep -n 'deviation' .claude/rules/vectorize-many-cell-fits.md` → mid-run trigger forces the vectorize signature check only (line 265) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Five short additive clauses at the named anchors (§ Per-cell fit phases; § Mid-run trigger; the RAM/RSS routing bullet); plus one monitoring line (output-growth over CPU%) in plan-compute-sizing or gotchas.

## Scope / surfaces

- Primary targets: `.claude/rules/plan-compute-sizing.md`, `.claude/rules/vectorize-many-cell-fits.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 5b9e2c8a123d

- workflow_fix_target: .claude/rules/plan-compute-sizing.md

