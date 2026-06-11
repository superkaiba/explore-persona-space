---
name: Verification-row hidden degrees of freedom
description: Reproducing a published correlation from persisted artifacts hinges on two rarely-documented choices - the centering-bank composition and the partial-correlation residualization variant; enumerate both before declaring join-failed.
type: feedback
---

When writing a verification/re-grade adapter that must reproduce a published
Spearman/partial-correlation from persisted centroids, two implementation
degrees of freedom are almost never stated in the task body but decide whether
the gate passes:

1. **Centering-bank composition.** "Centered cosine, subset to K personas" is
   ambiguous between centering over the FULL persisted bank then subsetting,
   vs centering over the K-subset bank. Compute both; gate on whichever
   reproduces; record the discovery in the row (it matters for the
   bank-comparability caveat). Task #536: #142's published rho 0.567 reproduced
   ONLY under core-11-subset-bank centering (full-111-bank gave 0.76).
2. **Partial-correlation variant.** "Partial Spearman" implementations differ:
   pingouin partial_corr, rank-residualize-then-correlate, and
   VALUE-residualize-then-rank-correlate give materially different numbers at
   small N. Reproduce the body's stated Methodology wording first (verbatim
   mirror of the producing script when it exists). Task #536: #311's published
   -0.348 was value-residualized (ranks=False) -- the rank variant gave -0.083;
   #380's 0.1113 needed the script's exact rank_residualize, not pingouin
   (0.0875).

**Why:** in #536 round 2 all three new canonical-line verification rows would
have read join-failed under the "obvious" implementation; the published numbers
reproduced exactly once the right variant was found, turning would-be failures
into clean verifications.

**How to apply:** before any verification adapter raises its join gate, sweep
the small space {bank composition} x {residualization variant} deterministically
and gate on the published wording's variant; never declare join-failed from a
single variant.
