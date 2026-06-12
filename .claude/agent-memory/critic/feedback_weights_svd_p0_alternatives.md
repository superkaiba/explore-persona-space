---
name: weights-level SVD P0-test alternatives
description: Three endemic alternatives for adapter-SVD key/write/rotation designs — generic gradient-input alignment is P0's mechanism not a confound (scope caveat), attenuation/SNR growth mimics key rotation in dcos-vs-dose reads, and wrong-context write nulls degenerate under rank-1 shift matrices
type: feedback
---

From the #604 P0 adapter-SVD review (2026-06-11; alternatives lens, APPROVE).
Applies to any plan SVD-ing stored LoRA/ΔW weights and comparing singular
vectors to context vectors ("keys") / measured activation shifts ("writes"),
or reading key rotation vs dose.

**1. "SFT gradients align with their inputs by construction" is NOT an
alternative to a descriptive P0 goal — it is the candidate mechanism.** If the
Goal is "test prediction key=v_src at the weights level", generic
gradient-input alignment producing key=v_src CONFIRMS the prediction; it only
threatens the NARRATION (P0-confirmation ≠ discrimination of the leakage model
from anything sharing A2/A4). The discriminating control would be a benign
behavior trained UNDER a persona context (checked 2026-06-11: no such stored
adapter exists — #552's benign_turner arms are no-persona plain SFT matched to
#521 EM). Disposition: scope caveat + follow-up, never a Must-Fix, when the
Goal is the prediction itself.

**2. Attenuation-shaped rotation.** In Δcos = cos(key,u_contrast) −
cos(key,u_raw) vs dose designs, a noisy low-dose key attenuates BOTH cosines
toward 0, so if the asymptotic key is closer to u_contrast, SNR growth alone
produces a positive Δcos-dose trend with NO directional rotation (dose-constant
true key). Discriminators (demand they ship): the two component cosines
separately vs dose (true rotation = raw declines/holds while contrast rises;
attenuation = both rise proportionally), cross-seed key |cos| per dose tier as
the SNR proxy, σ₁ spectral gap. Recoverable iff key vectors persist per cell.

**3. Write-null degeneracy under rank-1 shifts.** "Write matches measured
shift above the p95 of wrong-CONTEXT shift rows" is unsatisfiable-by-
construction exactly when the shift matrix is near rank-1 (all context rows
parallel — which the theory itself predicts at deep dose, #538): null p95 ≈
match value, so "no better than null" fires mechanically. Condition the
write verdict on the null's dynamic range (report null p5–p95 spread per
cell); pair the selectivity null with an absolute bar (the #604 EM-control
≥0.5 pattern) so a kill conjunction can't be satisfied by a degenerate null.

Also recurring in this family: dose tiers assembled from separate producing
runs (issue identity ⊥̸ dose — use within-run landing spread + checkpoint
intermediates as the confound check), and last-prompt-token centroid as a
position-recipe proxy for "the training input" when the loss is slot-masked
(marker-only loss puts gradient at the post-response slot → a key-MISS is
ambiguous, a hit is informative; conservative kill criteria absorb this).
