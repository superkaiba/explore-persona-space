---
name: Paired-placement run-noise calibration
description: NEAR/CONTROL paired placement designs calibrated against same-mix seed gaps — checkpoint-match the noise (30× range across checkpoints), treat C(k,2) gaps as dependent, keep magnitude bars descriptive (#600)
type: feedback
---

A paired NEAR−CONTROL placement design whose effect-size read is calibrated against an empirical same-mix across-seed gap distribution is statistically sound (exact target-level sign-flip permutation carries inference; a paired difference of two independent runs is on the same scale as a same-mix gap), but three calibration details belong in analyzer bullets, not Must-Fix (#600, 2026-06-11):

1. **Checkpoint-match the noise.** Marker-line run-noise gaps vary ~30× across checkpoints (#472: 0.002–0.069 normalized); a band-entry-fallback read at a non-terminal checkpoint must be overlaid on SAME-checkpoint gaps, never the pooled distribution.
2. **k seeds give C(k,2) DEPENDENT gaps per condition** — the pooled gap distribution's effective df is lower than its count; fine descriptively, never quote as independent N.
3. **Magnitude bars comparing a multi-target seed-mean statistic to single-pair gap medians mix noise scales** — conservative, keep descriptive; the permutation test is the inference. A triple-conjunction success label (permutation AND magnitude AND locality) is conservative labeling, not a REVISE, when partial outcomes are defined and components reported separately.

Spot-check recipe that worked: recompute the cross-seed normalized gap directly from parent trajectory JSONs (per-question `g_logp − b_logp`, persona-mean, ÷ source `delta_g_mean`) — #472 terminal gave per-persona median ≈0.044, panel-mean ≈0.028, same order as the fact-checked 0.069 ceiling. A parent sweep with 2 seeds/cell provides MANY same-mix gap draws (one per cell) — "single calibration draw" claims understate the prior.

**How to apply:** any placement plan with paired conditions + seed-gap calibration + checkpoint-fallback reads. Verify the permutation min-p is attainable (1/2^n_targets ≤ 0.05 needs ≥5 targets) and that descope rungs honestly demote the permutation when enumerations shrink below ~20.
