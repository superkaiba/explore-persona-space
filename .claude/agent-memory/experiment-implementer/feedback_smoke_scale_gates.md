---
name: smoke-scale gates — production-n-calibrated verdicts bind at smoke n
description: Thread the smoke flag into every gate-bearing phase; demote production-n-calibrated verdicts (anchor-reproduction tolerances, absolute yield floors) to informational under smoke; smoke floors sized so any nonzero yield proceeds
type: feedback
---

A kill-criterion gate calibrated at production n (a ±tol anchor-reproduction
check, an absolute yield floor) is structurally unsatisfiable at smoke n and
deterministically kills the smoke leg — and a smoke-scale floor set above 1
can rc-halt the very phase the smoke exists to exercise (#1345 r3: parity
gate n=8 vs n≈5000 anchors → R²≈−1.5 vs ±0.02 tolerance; story floor 2 with
kept=1 skipped extract_stories in smoke).

**Why:** the smoke leg runs the IDENTICAL chain at tiny n (PASS_UNIFIED), so
any verdict whose calibration depends on production n fires spuriously there.

**How to apply:** when implementing any dispatcher with a --smoke mode, thread
the smoke flag into every gate-bearing phase: run the gate's COMPUTATION
identically (the code path must stay exercised) but demote
production-n-calibrated verdicts to informational log lines; size smoke
yield floors so any nonzero yield proceeds; keep production halt paths
byte-untouched and unit-pinned (fails-at-production-shape tests).
