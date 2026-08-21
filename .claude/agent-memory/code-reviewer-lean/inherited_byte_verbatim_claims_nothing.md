---
name: inherited-byte-verbatim-claims-nothing
description: A plan registering an analysis output as "inherited byte-verbatim" from a parent that never had it lets the fork ship without it while the compliance table marks it IMPLEMENTED — grep the PARENT code AND its realized artifact keys for every inherited-by-name registered output.
metadata:
  type: feedback
---

When a plan registers a statistical/robustness output with the phrase "inherited
byte-verbatim" / "inherited from the parent stats" (#2329 q35_ladder_decay R1 g3:
"leave-one-carrier-out robustness folds", registered 4× in plan v8 incl. the
OOD-generalization-folds section), do NOT accept the fork's faithful inheritance as
implementation. Verify the premise both ways:

1. grep the PARENT script for the construct (loco/leave.one/jackknife/fold — try
   synonyms); 0 hits ⇒ the premise is false and the fork inherits NOTHING;
2. confirm against the parent's REALIZED artifact (`json.load(stats.json).keys()` +
   substring scan) — the code grep can miss a helper module, the artifact cannot.

The tell that makes this a blocker rather than a plan nit: the implementer's
compliance table marks the row IMPLEMENTED citing only a constants block ("LOCO
inherited | IMPLEMENTED | lines 88-92") — a false compliance claim over an empty
inheritance. Resolution is either implement the registered output (LOCO trend
re-runs are ~6 folds, trivially cheap) or a disclosed deviation naming the false
premise + a corrected compliance row; never ship the IMPLEMENTED claim as-is.

Sibling pattern in the same round (CONCERN grade): a fork-introduced second gate at
a DIFFERENT grain (direction-level tokgate vs rung-level anchor gate) wedges apart
"gate-surviving" from "data-bearing" unit counts; every inherited floor keyed on the
old single gate (MIN_TREND_RUNGS on gate-surviving rungs) silently under-counts in
the new contingency — sweep every registered floor/denominator for which gate set it
counts. Related: [[registered_gate_quantity_substituted]],
[[gate_mean_nanmean_denominator]].
