---
title: 'Re-run #432 on-policy: marker emission rate + on-policy end-of-answer log
  p(※), drop pos0/fixed-stub'
kind: experiment
tags: []
created_at: '2026-05-31T23:19:53Z'
has_clean_result: false
parent_id: 432
goal: 'Measure whether the trained source persona actually emits the marker, and how
  marker leakage distributes across the panel, when scored on the model''s OWN generated
  answers (on-policy emission rate + on-policy end-of-answer log p), replacing #432''s
  off-distribution fixed-stub/pos0 probe.'
---
## Goal

Measure whether the trained source persona actually emits the marker, and how marker leakage distributes across the panel, when scored on the model's OWN generated answers (on-policy emission rate + on-policy end-of-answer log p), replacing #432's off-distribution fixed-stub/pos0 probe.

## Motivation

#432 ranked personas by teacher-forced log p(※) at pos0 (first token) and at endpos (after a FIXED canonical answer, "Sure, here's a brief answer."). Both are off the on-policy distribution: ※ was trained trailing the persona's OWN answer, so scoring it as the first token or after a generic stub measures it where the model never emits it. Result: every persona scored ~1e-8 (no dynamic range), and "source pinned at the bottom of the leaderboard" was a probe artifact. The earlier marker line measured emission via free generation and showed ~90-100% source implantation. This rerun restores the faithful measurement.

We do NOT care about pos0. The fixed-canonical-answer probe is dropped.

## Design sketch (for /adversarial-planner to refine)

- **Retrain #432's exact recipe** (seed 42; source = software_engineer; 9 contrastive negatives; 200 positive + 1800 negative rows; LoRA r=32 α=64 lr=1e-5 1600 steps) to regenerate the 22 checkpoints. The adapter was never uploaded but is deterministic from the committed training data + Hydra config + seed.
- **On-policy eval at each checkpoint** (vLLM batched generation): for each of the 28 panel personas × 20 fixed eval questions, GENERATE the answer with the trained model, then measure:
  - **(a) Marker emission rate** = fraction of generated completions that contain ※ (single-token ※ → substring match is the documented marker-leakage exception to the no-substring rule).
  - **(b) On-policy end-of-answer log p(※)** = teacher-forced log p of ※ at the end of the model's OWN generated answer (the trained position, on-distribution).
- **Drop pos0 and the fixed canonical answer entirely.**
- Headline DV is emission rate; on-policy endpos log p is the continuous companion. Same 28-persona panel as #432 so the leaderboard is directly comparable to the (artifactual) teacher-forced one.

## Measurement validity (§6 — dogfooding the new safeguard)

| DV | Construct | Metric | On-distribution? | Validation |
|---|---|---|---|---|
| Emission rate | Does the persona emit ※ when it generates an answer | fraction of free-generated completions containing ※ | **yes** — model's own answers, natural trailing position | direct measure of the construct (not a proxy) |
| On-policy endpos log p(※) | Strength of the marker habit at the trained position | teacher-forced log p(※) after the model's own generated answer | **yes** — own answer, trained position | companion to emission rate; report both |

## Open design questions for the planner

- Generation temperature + N samples per (persona, question) for a stable emission-rate estimate.
- All 22 checkpoints vs a subset (generation is pricier than one teacher-forced forward pass).
- Whether to also re-score the OLD fixed-stub endpos on the same checkpoints, side-by-side, to quantify the probe artifact directly.
- Compute budget / pod spec.
