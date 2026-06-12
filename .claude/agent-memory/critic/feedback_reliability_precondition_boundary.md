---
name: Reliability-precondition boundary arithmetic
description: Binary split-half >= 0.5 validity preconditions do NOT bound cosine attenuation; at the pass floor, attenuation alone spans typical confirmation-threshold gaps (true 0.96 reads ~0.78). Check the arithmetic; require persisted per-question tensors so the analyzer can disattenuate.
type: feedback
---

A binary reliability precondition ("median per-persona split-half cosine >= 0.5, else verdict downgrades to 'shift too small'") guards only the noise-DOMINATED extreme, not the moderate-reliability middle.

**Arithmetic (geometry cosine DVs, k-question mean shift vectors):** split-half cosine between two k/2-question half-means ~ R_half; Spearman-Brown to the full-k mean: R_k = 2*R_half/(1+R_half); observed cos-to-U1 ~ true_cos * sqrt(R_k). At the 0.5 pass floor: attenuation factor 0.816 -> a truly EM-shaped arm (true cos 0.96) reads 0.78; at split-half 0.7 it reads 0.87. So an arm passing the precondition can read BELOW a "confirmation: mean cos <= 0.85" threshold purely from attenuation — a false-confirmation pathway that passes BOTH registered checks (#552 plan v1, 2026-06-10).

**Compounding:** estimation noise pushes mean-cos AND top-share (sigma1/Sigma-sigma) in the SAME (confirming) direction — the two conjuncts of a "less concentrated" rule are not independent corroboration under low signal. The sign-flip null does not catch this regime (moderate real structure still clears it).

**Why:** the registered thresholds were calibrated on high-signal arms (parent EM sigma1 ~ 78-108) and silently assume comparable reliability in the new arm.

**How to apply:** for any plan comparing direction-concentration metrics (cos-to-U1, top singular share) across arms with plausibly different shift magnitudes: (a) verify per-question/per-item tensors are persisted so the analyzer can disattenuate (free analysis); (b) if persisted, this is a Concern bullet with the arithmetic spelled out, NOT a Must-Fix (the diagnostic is reported; per The Bar, don't demand new pass/fail rules on existing diagnostics); (c) if NOT persisted, it IS a Must-Fix (the analyzer cannot recover — phantom-input cousin of incident #509). Also flag the secondary read: cross-arm |cos(U1,U1')| attenuates the same way; benchmark against within-arm seed-pair cosines as the reliability ceiling.

**Variance-ratio variant (#480 f3, 2026-06-10):** a panel-informativeness gate of the form "between-cell SD >= 2x median per-cell SE" is the same boundary in variance-ratio form: pure noise gives E[SD] ~ SE, so 2x implies observed/noise variance >= 4 -> implied reliability ~0.75 at the pass floor (attenuation factor ~0.87). Sound as a noise-screen, but a NULL on a panel passing AT the floor is weaker kill-evidence than at 5x. Same triage as above: per-item rows persisted -> Concern (quote SD/SE next to each rho; split-half disattenuate before any informative-kill verdict); not persisted -> Must-Fix.
