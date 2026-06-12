---
name: Paired-placement run-noise calibration
type: feedback
description: NEAR/CONTROL paired placement designs calibrated against same-mix seed gaps — checkpoint-match the noise, treat k-seed gap counts as dependent, keep magnitude bars descriptive (#600)
---

Rule: a paired NEAR−CONTROL placement design whose effect-size read is calibrated against an empirical same-mix across-seed gap distribution is statistically sound (exact target-level sign-flip permutation carries inference; paired difference = difference of two independent runs, same scale as a same-mix gap), but three calibration details belong in the analyzer bullets, not Must-Fix:

1. **Checkpoint-match the noise.** Marker-line run-noise gaps vary ~30x across checkpoints (#472: 0.002–0.069 normalized). If a pair's headline read lands at a non-terminal checkpoint via a band-entry fallback, its |d| must be overlaid on same-checkpoint gaps, never the pooled distribution.
2. **k seeds give C(k,2) DEPENDENT gaps per condition** (3 pairs from 3 draws); the pooled gap distribution's effective df is lower than its count. Fine descriptively; don't quote as independent N.
3. **Magnitude bars comparing a multi-target seed-mean statistic to single-pair gap medians mix noise scales** — conservative, keep descriptive; the permutation test is the inference. A triple-conjunction success label (permutation AND magnitude AND locality) is conservative labeling, not a REVISE, when the plan defines the partial outcomes and reports components separately.

**Why:** #600 (2026-06-11) had all of this right; the only real risks were post-hoc misuse of the calibration. Spot-check recipe that worked: recompute the cross-seed normalized gap directly from parent trajectory JSONs (per-question `g_logp − b_logp`, persona-mean, ÷ source `delta_g_mean`) — #472 anchor pair at terminal gave per-persona median gap ≈0.044, panel-mean gap ≈0.028, same order as the fact-checked 0.069 ceiling. Also: a parent sweep with 2 seeds per cell provides MANY same-mix gap draws (one per cell), so "single calibration draw" claims understate the available prior.

**How to apply:** any marker/behavior placement plan with paired conditions + seed-gap noise calibration + checkpoint-fallback reads. Verify the permutation min-p is attainable (1/2^n_targets ≤ 0.05 needs ≥5 targets... 6 targets → 1/64) and that descope rungs honestly demote the permutation when enumerations shrink below 20.
