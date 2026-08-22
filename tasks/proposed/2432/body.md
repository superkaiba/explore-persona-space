---
title: 'Step 10d pre-push lint gate: 900s per-leg fence is under the measured 799-1085s
  wall, crashing gates fleet-wide'
kind: infra
tags: []
created_at: '2026-08-20T21:16:19Z'
has_clean_result: false
origin_prompt: 'Measured during #2217''s Step 10d on 2026-08-20: gate attempt 2 crashed
  BASE_RC=124 GATED_RC=124; the no-flags workflow_lint bundle measured 1085s in the
  gate tree / 799s at the repo root against a 900s fence. #2217 raised the fence to
  2700s in its own composed workload only, deliberately not editing the spec.'
workflow: v1
---
---
kind: infra
workflow: v1
---

# Step 10d pre-push lint gate: the 900s per-leg fence is smaller than the measured 799-1085s work, so gates crash intermittently fleet-wide

## Goal

Re-size (or restructure) the per-leg time fence on the Step 10d pre-push
workflow-lint gate so a healthy, passing lint run cannot be killed by its own
timeout. Today the fence is 900s and the work measures 799-1085s, so whether any
given Step 10d gate crashes is decided by machine load rather than by the branch
under test.

## Evidence (measured 2026-08-20 during #2217's Step 10d)

Both runs completed rc=0 / `workflow_lint: PASS` under an 1800s probe fence:

| configuration | wall | vs the 900s fence |
|---|---|---|
| GATE TREE (`git archive origin/main`, no `.git` — the exact tree the gate lints) | **1085s (18m05s)** | **20% OVER** |
| repo root (main checkout, has `.git`) | **799s (13m19s)** | 11% under |

- The slow leg is the **no-flags** bundle. The flags leg
  (`--check-references --check-tables --check-asks --check-autonomous-asks`) is
  9s from the repo root / 16s from the gate tree — not a contributor.
- The gate runs FOUR such legs per invocation (baseline x2, gated x2), each
  fenced at 900s, so a single invocation has multiple independent chances to
  cross.
- Observed consequences the same day: #2217's gate crashed with
  `BASE_RC=124 GATED_RC=124` (124 = `timeout`'s exit code) at load
  41.79/54.41/69.86 on 32 cores; #2212's gate crashed at 14:59Z; #2217's
  own EARLIER gate had passed at lower load, and sibling gates passed at
  17:05/17:09/17:12Z. Same code, opposite outcomes — the signature of a fence
  sitting inside the work's natural variance.
- Cost per loss: the whole ~40 min gate is discarded and re-run, plus the
  diagnosis time. #2217 spent roughly two hours on this one cause.

**Two hypotheses tested and DISCARDED** (recorded so they are not re-derived):

1. *A lint regression.* `1f22cfed7f` touched `scripts/workflow_lint.py` that day,
   between #2217's passing and crashing gates — the obvious suspect. Its entire
   change to that file is size-cap CONSTANT bumps; no new check, no new
   filesystem walk. Innocent.
2. *A gate-tree pathology from the missing `.git`.* Real but secondary: the gate
   tree is ~36% slower than the repo root, not 100x. Both configurations are
   inherently slow, so the fence is the defect, not the tree.

## Diagnostic note worth keeping (cost an inverted reading during #2217)

Leg 1 writes with `>` and leg 2 appends with `>>`, and Python block-buffers
stdout to a file. A fence-killed leg 1 therefore loses its buffered WARNs and
`PASS` entirely while its unbuffered stderr line survives — so the output file
reads as "leg 1 passed, leg 2 wrote nothing" when the truth is exactly the
reverse. Anyone diagnosing a `BASE_RC=124` from these files needs that fact.

## Suggested approach

1. Raise the per-leg fence to >=2x the measured gate-tree wall
   (2700s is the value #2217 used, per the p90-style x2 dispersion default in
   `.claude/rules/plan-compute-sizing.md`). Cheapest correct fix; note the fence
   is a CRASH backstop, not a throughput control, so a generous value costs
   nothing on healthy runs.
2. Consider re-measuring periodically, or deriving the fence from a measured
   basis rather than a literal: the wall grows with the workflow surface, so a
   fixed constant will drift back into the variance band as `.claude/` grows.
   `18-step-10d.md` alone is 287 KB today.
3. Optional, larger: investigate why the no-flags bundle needs 13-18 minutes at
   all, and whether the ~36% gate-tree penalty is a `.git`-absence fallback worth
   removing. Not required to close this task.

## Acceptance criteria

- The per-leg fence is >=2x a MEASURED wall, with the measurement and its date
  recorded next to the value.
- A gate run under representative fleet load (load avg comparable to 40-70 on 32
  cores) completes all four lint legs without a 124.
- The change is confined to the fence; verdict logic, check set, and subtraction
  arithmetic are untouched.

## Provenance

Measured during task #2217's Step 10d on 2026-08-20 (gate attempt 2 crashed on
this cause; attempts 3 and 5 passed with the fence raised to 2700s in that
issue's composed workload only — the spec's literal 900s was deliberately NOT
edited from that session, which is why this task exists). Full measurement and
the discarded hypotheses are in #2217's `epm:progress` notes for that date.
