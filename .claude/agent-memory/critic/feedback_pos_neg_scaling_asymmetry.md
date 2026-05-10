---
name: Pos:neg gradient-ratio confound in length sweeps
description: When length-sweep plans hold negatives constant while scaling positives, the positive:negative gradient-mass ratio is the simplest alternative for any positive result
type: feedback
---

When a length-sweep training plan scales the POSITIVE side of an SFT
dataset while holding the NEGATIVE side fixed (e.g., issue #260 sub-exp (b):
positives ~50/316/1050 tokens, negatives constant at ~280), the
positive:negative gradient-mass ratio sweeps along with the nominal
length variable. Any monotone result in that sub-experiment is
observationally equivalent to "more positive gradient mass beats fixed
negative gradient mass, and the marker absorbs more weight" — which has
nothing to do with "more context tokens drive implantation."

**Why:** Issue #260 v2 shipped this confound after fact-check, marker as
"part of the experimental signal, not a confound." It is a confound: the
mechanism for source-rate growth could be entirely "positives outshout
negatives in the gradient" rather than the claimed context-size effect.

**How to apply:** When reviewing any LoRA/SFT length-sweep plan, check
whether the negatives scale in lockstep with positives. If not, demand
either (i) a matched-length-negatives control condition at the largest
scale, or (ii) explicit log-odds-ratio decomposition that separates
"positive-side gradient" from "negative-side gradient" in the
interpretation. This generalizes to any sweep where one side of an
asymmetric training dataset is the variable.
