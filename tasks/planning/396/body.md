---
title: Source-rate panel re-run on 48 personas with single-token marker ※ + log-prob
  eval
kind: experiment
tags: []
created_at: '2026-05-26T20:49:04Z'
has_clean_result: false
parent_id: 380
goal: 'Re-run the 48-persona panel (#274/#296/#380) with single-token marker ※ and
  extract the teacher-forced log p(※) trajectory across each response position, to
  characterize WHERE in the response personas inject marker bias and correlate trajectory-shape
  features (end-point, AUC, peak, slope, bare prior) with five candidate predictors
  (cosine-to-assistant, JS-to-baseline, pairwise output distance, marker-prior, first-step-gradient);
  bystander-leakage geometry replication of #207 becomes whether trajectories cluster
  by pairwise persona distance on the off-diagonal of the same 48×48 matrix.'
---
## Depends on

- [#401](https://eps.superkaiba.com/tasks/401) — finishes the marker abstraction (`cfg.marker_token`) and adds the `compute_marker_logprob` primitive this task needs. **Do not run `/issue 396` until #401 is in `completed` status.** Without it this task would have to ship its own marker-swap patch, duplicating work shared with #397, #398, #399, #400.

## Goal

Re-run the 48-persona panel (#274/#296/#380) with single-token marker ※ and extract the teacher-forced log p(※) trajectory across each response position, to characterize WHERE in the response personas inject marker bias and correlate trajectory-shape features (end-point, AUC, peak, slope, bare prior) with five candidate predictors (cosine-to-assistant, JS-to-baseline, pairwise output distance, marker-prior, first-step-gradient); bystander-leakage geometry replication of #207 becomes whether trajectories cluster by pairwise persona distance on the off-diagonal of the same 48×48 matrix.

## Background

The predictor lineage that asks "what makes some personas more vulnerable as marker source?" has hit three consecutive nulls ([#271](https://eps.superkaiba.com/tasks/271) → [#340](https://eps.superkaiba.com/tasks/340) → [#380](https://eps.superkaiba.com/tasks/380)). All three correlate per-persona predictors against the same 48-vector of substring-match source rates from the [#274](https://eps.superkaiba.com/tasks/274) / [#296](https://eps.superkaiba.com/tasks/296) panel — rates whose noise floor (~5 pp at N=100 completions per cell) is comparable to the per-cell variation between adjacent ranks. The hypothesis is that a tighter continuous measurement might surface signal that substring-match noise is masking.

This task does the methodology upgrade: switch from `[ZLT]` (4-token chain, joint base log-prob ~−33 nats) to `※` (single token, base log-prob ~−19 nats, validated in [#395](https://eps.superkaiba.com/tasks/395)) and replace per-cell substring-match rates with per-cell teacher-forced log-prob of the marker token at end-of-answer position.

The 48×48 evaluation matrix produced here also drops out the data for the bystander-leakage geometry analysis ([#207](https://eps.superkaiba.com/tasks/207) family) at higher resolution as a fall-out.

## What this tests

- Whether geometric-distance predictors (cosine-to-assistant, JS-to-baseline, pairwise output distance) that came up null against substring-match rates have signal against continuous log-prob scores.
- Whether the marker-prior predictor (base-model log p(`※` | persona) before training) predicts post-training log-prob source-rate analog.
- Whether the per-persona ranking changes when measured continuously vs binary firing.
- Bystander-leakage geometry replication ([#207](https://eps.superkaiba.com/tasks/207)) at higher resolution — same 48×48 matrix, off-diagonal cells.

## What this does NOT test

- Whether RL or SDF (queued separately as [#392](https://eps.superkaiba.com/tasks/392) / [#393](https://eps.superkaiba.com/tasks/393)) change the predictor story.
- Whether the per-token-loss vs whole-completion-loss factor result from [#383](https://eps.superkaiba.com/tasks/383) holds with `※` (queued as a separate task).
- Bystander-vs-source vulnerability comparisons across model families.

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. Train 48 LoRAs (one per source persona in the [#274](https://eps.superkaiba.com/tasks/274) / [#296](https://eps.superkaiba.com/tasks/296) panel) on Qwen-2.5-7B-Instruct with the same recipe as [#296](https://eps.superkaiba.com/tasks/296) (LoRA r=32, α=64, lr=1e-5, 3 epochs, 600 rows/source, seed=42) — but with `※` as the trained marker instead of `[ZLT]`. ~24 GPU-hours.
2. Evaluate each LoRA against all 48 personas: 20 questions per cell, teacher-forced log-prob of `※` at end-of-answer position. ~1-2 GPU-hours.
3. Optionally also sample 100 completions per cell for behavioral substring-match parity with the legacy `[ZLT]` panel. ~2 GPU-hours.
4. Re-run the four predictor analyses ([#271](https://eps.superkaiba.com/tasks/271) cosine, [#340](https://eps.superkaiba.com/tasks/340) length-partial, [#380](https://eps.superkaiba.com/tasks/380) JS, [#380](https://eps.superkaiba.com/tasks/380) pairwise) against continuous log-prob scores instead of substring rates.
5. Run the [#207](https://eps.superkaiba.com/tasks/207) bystander-geometry correlation analyses on the off-diagonal matrix (cosine and JS predicting per-bystander log-prob leakage).
6. Compare predictor strength: how much does correlation strength change when the dependent variable goes from rate to log-prob?

## Open questions for the planner

- Whether to keep the legacy `[ZLT]` source-rate panel as a control arm or trust [#395](https://eps.superkaiba.com/tasks/395)'s base-prior probe to characterize the marker swap effect.
- Multi-seed (seed=42 only as in [#296](https://eps.superkaiba.com/tasks/296), or seeds {42, 137, 256})? Multi-seed adds 2-3× cost but matches the pre-registration standard in [#296](https://eps.superkaiba.com/tasks/296).
- Position choice for log-prob extraction — end-of-answer EOS-adjacent vs the trained marker position. Test both?
- Whether to include the four base-prior predictors validated in [#395](https://eps.superkaiba.com/tasks/395) as additional candidate predictors against the log-prob source rates.
