---
name: Matched-position û shared-offset tautology
description: u built as subset-mean of the same Δ-family shares the offset+position with Δ, so "matched > response" is near-tautological; mean-subtracted projection is the analyzer escape (#685)
type: feedback
---

When a matched-position direction û_match is constructed as the
diff-in-means (subset-mean) of the SAME Δ family the test Δs belong to,
read at the SAME token slot, a "matched > cross-position" cosine lift is
near-GUARANTEED by construction — û_match shares BOTH (a) the per-behavior
shared offset and (b) the token-slot lexical/format energy with Δ, while
the cross-position û_resp (built at a different read position) does not.

**Why:** The matched-norm RANDOM-direction null does NOT address this — it
only rules out random chance, not the shared-offset confound. So the lift
clearing the null is not evidence the matched alignment is genuine.

**Why it is a CONCERN, not a REVISE (#685 round-2, 2026-06-28):** The
tautology threatens only a MAGNITUDE overclaim, not the substantive
takeaway ("smaller context-specific residual / partial additivity" survives
at any projection value). Recoverable iff (1) the plan pre-commits the
analyzer to no re-framing (lists the falsifier), AND (2) the disambiguating
diagnostic is computable from committed tensors: a MEAN-SUBTRACTED
projection — subtract the per-(behavior,layer) mean of Δ across all contexts
AND the per-(behavior,layer) mean of û across the build subset, then cosine.
This is the SAME shared-offset control the body already reports for the
Δ-vs-Δ consistency read (`consistency_cosine_mean_subtracted`). If the
matched projection collapses under mean-subtraction the way the
Δ-consistency does, the matched read is "shared direction + small residual"
— same conclusion as the cross-position read, just larger.

**How to apply:** For any matched-position / construct-from-same-family
projection design, check (a) is û built from the same object family as the
DV at the same read position? (b) does the null only address random chance?
(c) is the mean-subtracted (shared-offset-removed) version registered OR
computable from committed tensors? If (a)+(b) yes and (c) computable →
APPROVE + Concern naming the mean-subtracted diagnostic; only REVISE if the
shared-offset control is NEITHER registered NOR computable AND the headline
re-frame rests on the un-disambiguated lift. The cross-û cosine
(cos(û_match,û_resp)) shows the two reads differ but does NOT stand in for
the shared-offset control.
