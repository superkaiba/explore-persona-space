---
name: inherited-analysis-code-semantics-trap
description: Strict-inheritance rerun plans can register statistical semantics (Holm family m, exclusion denominators) that contradict the reused analysis driver's own code — grep the driver's denominator computation before accepting plan-registered values
metadata:
  type: feedback
---

On a strict-inheritance / model-swap rerun, verify that every plan-REGISTERED
statistical semantic (Holm family size m, testability floors, exclusion
denominators) matches what the REUSED analysis driver actually computes —
e.g. `holm_family_m = len(pvals)` over post-exclusion testable cells means m
is analysis-time, and a plan clause like "a pair drop shrinks n, never m"
contradicts the code it inherits and will simply not be executed.

**Why:** the #2162→child rerun's v3 registered fixed m=31/15/28 with a
"never m" clause while the inherited `issue2162_analysis.py` computes
analysis-time m (parent realized 25/10/26 in `stats.json → families`). Caught
by the statistics critics at round 1 (S1); my lens's item-9 reuse backstop
also owns this shape (reused code whose semantics the plan text contradicts).
The fix that preserved comparability was registering the CODE's semantics
(Remedy A), not forking the estimator.

**How to apply:** when a plan reuses a parent's analysis/stats driver, grep
the driver for the family/denominator computation (`holm`, `len(pvals)`,
testability floors) AND read the parent's realized stats artifact, then check
the plan's §6 registration against both. A mismatch is a REVISE-shape defect
(the registered analysis will not be run); the low-confound remedy is usually
to register the inherited code's semantics and state the comparability effect.

**Resolution-shape refinement (validated at the same rerun's round 2,
2026-08-19):** a prose-vs-code contradiction legitimately resolves in EITHER
direction, and the low-confound choice depends on what the semantic IS: (a) a
cross-model/cross-run comparison DEFINITION (a verdict lattice, an exclusion
rule) → keep the CODE byte-verbatim and correct the registration (definition
identity is the comparability object); (b) a pre-registered STATISTICAL
GUARANTEE (Holm family m, FWER denominators) → change the code to hold the
registered value, ACCEPTED when the plan proves output-identity on the
parent's realized artifact (e.g. `trend_meta.n_testable == m` means the
placeholder form is byte-equivalent on parent data) and the divergence-shape
change is strictly conservative (p=1 placeholders only raise step-down
factors). Verify the output-identity claim against the artifact yourself —
do not accept it from plan prose.
