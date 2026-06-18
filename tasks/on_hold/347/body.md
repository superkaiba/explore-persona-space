---
title: 'Probe what layer-20 direction actually elicits the [ZLT] marker — followup
  to #267''s random-as-good-as-centroid finding'
kind: experiment
tags: []
created_at: '2026-05-11T20:01:49.000Z'
has_clean_result: false
sagan_id: f681434c-67b9-4145-82cb-b0a375903b3d
sagan_number: 347
priority: normal
---
**Parent:** [#267](https://github.com/superkaiba/explore-persona-space/issues/267) (direction-only steering at L20 elicits the trained `[ZLT]` marker without the persona system prompt, BUT a norm-matched random vector at the same magnitude does at least as well by every metric — same-or-higher firing rate, more strongly anti-correlated with the prompted ranking, qualitatively coherent text in sampled outputs, and length stays in the no-steering coherence band for 9 of 10 personas).

## Goal

The random-as-good-as-centroid finding is the load-bearing surprise from #267 and it's currently unexplained. We need to figure out *what* layer-20 perturbation is actually doing the work. Two competing readings:

- **(A) Magnitude, not direction.** ANY layer-20 perturbation of magnitude ~30-60% of the baseline residual norm elicits the marker, because the LoRA learned a "respond to perturbation at L20" rule that's direction-agnostic.
- **(B) A specific non-centroid direction.** There's a load-bearing direction that lives in layer 20 — possibly a PCA component of the LoRA's weight delta, the difference between firing and non-firing residuals, or an artifact of LoRA's low-rank structure — and both the centroid AND random vectors happen to project onto it.

Pinning down (A) vs (B) tightens the mechanism story for `[ZLT]` marker training. It also bounds how alarming the finding is for defenses against persona-prompt-trained backdoors: under (A), any noise can re-elicit; under (B), the load-bearing direction can be probed for and blocked.

## Hypothesis

If (A): randomizing the direction across calls (fresh isotropic random per question, not the same vector across all 20 questions) should produce comparable firing rates to a fixed random direction.

If (B): the marker's firing rate should correlate with the projection of the steering vector onto a specific layer-20 direction. We can search for that direction by (1) running PCA on the LoRA's effective weight delta at layer 20, (2) computing the difference between firing and non-firing residuals at layer 20 across the training data, and (3) testing whether steering with the top-PCA-component or the firing-vs-non-firing direction outperforms both the centroid and a random vector.

## Design (single variable: which direction we steer with)

**4 conditions per persona** (n=10 personas, n=100 completions per cell, single seed, hooked-HF generation per #267 rig):

1. **Centroid** (reference from #267): each persona's layer-20 mean residual at the assistant-start token, mean-centered across the 10-persona set, at `c=+2.0`.
2. **PCA top-1 of LoRA weight delta:** compute the LoRA's effective layer-20 weight delta (`A @ B` for the LoRA's attention/MLP projections), take its top singular component, project onto residual-stream space. Steer with this direction at the same per-persona magnitude as the centroid.
3. **Firing-vs-non-firing residual difference:** on the training data, separate firing examples from non-firing examples, compute mean residual at layer 20 for each, take the difference. Steer with this direction at the same magnitude.
4. **Fresh random per question:** sample a NEW isotropic Gaussian random vector at layer 20 for every (persona, question) cell — not the same vector across all 20 questions of one persona. Tests whether the random-as-good-as-centroid finding survives randomness re-sampling.

**Eval:** mean firing rate per condition per persona, plus rank correlation with the prompted bridge ranking from #267.

**Pass / interpretation:**
- If condition (4) fires at ~13% mean rate (matching #267's fixed-random arm), reading (A) is strongly supported.
- If conditions (2) or (3) significantly outperform centroid AND random by Δρ ≥ +0.30 in the centroid-wins direction, we've found the load-bearing direction — reading (B) wins.
- If both (4) fires at ~13% AND (2)/(3) fire above centroid, both readings have partial support — the load-bearing direction exists but a chunk of the firing rate is direction-agnostic.

## Compute estimate

- All 4 conditions × 10 personas × n=100 = 4,000 generations
- Reuses #267's eval rig + LoRAs
- ~1 GPU-hour on 1× H100, hooked HF generation per #267 methodology
- PCA + difference-direction computation: ~10 minutes local CPU on the training data

## Why this can't be answered by re-analyzing existing data

#267 ran the same fixed random vector across all 20 questions per persona. We don't know if the marker fires equally well when the random vector is fresh per question (the "magnitude alone matters" reading vs the "we happened to sample a special direction" reading). And #267 didn't try PCA-of-LoRA-delta or firing-vs-non-firing residual differences as candidate load-bearing directions.

## Next step

Run `/issue <N>` to dispatch /adversarial-planner. No work starts inline.
