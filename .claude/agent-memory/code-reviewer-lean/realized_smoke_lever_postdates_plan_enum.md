---
name: realized-smoke-lever-postdates-plan-enum
description: "Step 0.71 round audit: a runtime --smoke CLI lever added by the realized code is NOT covered by a plan enumeration that predates it (plan may even say 'downgrade exists only inside pytest'); the tell is the ONE marker phase-section missing its enumeration line (#2587 R1 g8)"
metadata:
  type: feedback
---

Rule: at the round-level Step 0.71 audit, grep EVERY round script for
smoke-conditional branches (`args.smoke`, `cfg.smoke`, `if not smoke`,
`--smoke`) and map each hit file to a named enumeration block — do not stop
at "the plan's §-smoke enumeration exists". #2587 R1 g8: unit 5b added a
runtime `--smoke` CLI flag to the analysis driver with three production-only
assert gates (expected-pairs/contexts coverage, axis-completeness,
n_car==12) plus a silent B=10,000→100 bootstrap narrowing; the plan's §4.7
enumeration PREDATED the lever and even asserted "no gate is DOWNGRADED in
the pod smoke … the --tiny mode exists only inside pytest" — true when
written, falsified by the realized code. smoke-blind-spots.md's implementer
duty ("mirror the block per phase when the realized code adds a branch the
plan did not anticipate") was skipped for exactly one phase.

**Why:** enumerations accumulate per unit in pre-split rounds; the cheap
structural tell is the marker's `## Smoke run` phase sections — the ONE
section without a "Smoke blind-spot enumeration" line is where the
unenumerated lever lives (here: `### analysis` had none while
figures/fits/battery/map_gen_capture/judge each carried one).

**How to apply:** CONTRACT-BEARING split-review groups / any round-level
0.71. Verdict: Critical `smoke-blind-spot-unenumerated` (substantive-class,
never stripped); fix is a marker re-post adding the enumeration block — no
code change when the downgrades are sanctioned gate-calibration. Sibling:
[[feedback_opportunistic_prod_assert_misses_blindspot_enum]] (fix-born
asserts, same family; this entry is the whole-lever variant). Also pairs
with a plan-vs-marker consistency read: a plan line "no gate downgraded"
scopes to the smoke the PLAN knew about, never to later-added levers.
