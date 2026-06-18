---
name: Reliability-precondition boundary arithmetic
description: Binary split-half ≥ 0.5 (or SD ≥ 2×SE) validity gates don't bound attenuation — at the pass floor a true cos 0.96 reads ~0.78; Must-Fix only if per-item tensors aren't persisted (#552, #480 f3)
type: feedback
---

A binary reliability precondition ("median per-persona split-half cosine ≥ 0.5, else downgrade") guards only the noise-DOMINATED extreme, not the moderate-reliability middle.

**Arithmetic (geometry cosine DVs):** split-half R_half → Spearman-Brown R_k = 2R_half/(1+R_half); observed cos ≈ true_cos·√R_k. At the 0.5 floor: factor 0.816 → a truly EM-shaped arm (true cos 0.96) reads 0.78; at split-half 0.7 it reads 0.87. An arm passing the precondition can read below a "≤ 0.85 → less concentrated" threshold purely from attenuation — a false-confirmation path that passes BOTH registered checks (#552 v1). Compounding: estimation noise pushes mean-cos AND top-share in the SAME confirming direction, so the two conjuncts aren't independent corroboration; the sign-flip null doesn't catch this regime. Thresholds calibrated on high-signal parents silently assume comparable reliability in the new arm.

**Variance-ratio variant (#480 f3):** "between-cell SD ≥ 2× median per-cell SE" is the same boundary in variance form — pure noise gives E[SD] ~ SE, so 2× implies reliability ~0.75 at the floor (attenuation ~0.87). Sound as a noise screen, but a NULL on a panel passing AT the floor is weak kill-evidence.

**How to apply:** for any cross-arm comparison of concentration metrics (cos-to-U1, top singular share) where shift magnitudes plausibly differ: (a) verify per-question/per-item tensors persist so the analyzer can disattenuate (free); (b) if persisted → Concern with the arithmetic spelled out (per The Bar, don't demand new pass/fail rules on existing diagnostics); (c) if NOT persisted → Must-Fix (phantom-input cousin of incident #509). Also flag the secondary read: cross-arm |cos(U1,U1′)| attenuates the same way — benchmark against within-arm seed-pair cosines as the reliability ceiling.
