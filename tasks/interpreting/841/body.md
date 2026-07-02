---
title: Rolling an activation forward through a learned affine layer map beats a direct
  read of the source layer in a minority of read-out-layer cells (above chance in
  aggregate), on Qwen-2.5-7B-Instruct (MODERATE confidence)
kind: experiment
tags:
- activation-dynamics
- context-geometry
created_at: '2026-07-02T06:56:23Z'
has_clean_result: true
parent_id: 779
origin_prompt: '# Motivation - We''ve found that there is a map from context vector
  to answer profile - Potentially there could be a map from each activation to the
  next one (not sure if per position or a general map) - I want to see: how well can
  we actually predict this, and how well can we predict this for the purposes of predicting
  if a model will exhibit a specific behavior - Simply: take a bunch of contexts from
  LMSYS/WILDCHAT, train a linear map to predict next activation from previous activation,
  also try a small MLP, potentially position information in input/output, wait actually
  should we train a RNN/LSTM/SSM?, try at each layer and see what''s best - Also need
  a deep literature search to see what has been done before. Assess the novelty and
  likelihood that it will work. [after lit review + discussion:] okay I think it''s
  worth running, can we benchmark against 779?'
goal: 'On Qwen-2.5-7B-Instruct, characterize how well the residual-stream next-activation
  map can be learned and how much trait-relevant signal predicted activations retain:
  fit per-layer maps f_l: h_l -> h_{l+1} (identity / ridge / MLP / depth-sequence
  classes, target Delta_l, SmoothL1) at the last-prompt-token position on the #779
  LMSYS corpus, then on #779''s exact monitoring rig project rolled-forward predicted
  activations onto the persona directions at the trait read-out layers and benchmark
  the resulting trait monitor (within-condition Pearson r, #779 protocol verbatim)
  against #779''s measured rows — raw projection at the source layer (the matched-information
  comparison), raw projection at the target layer (the transport ceiling), a direct-hop
  ridge map, and the learned map h / direct predictor g reference rows — reporting
  the per-layer Delta-predictability atlas, the trait-signal retention curve vs prediction
  horizon, and the Delta-R2-vs-retention divergence per (layer, function class).'
relates_to:
- spec-context-as-vector
---
# Rolling an activation forward through a learned affine layer map beats a direct read of the source layer in a minority of read-out-layer cells (above chance in aggregate), on Qwen-2.5-7B-Instruct (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- An affine ridge map explains 70–91% of mean-centered layer-update variance on 500 held-out contexts and beats a small MLP at all 27 transitions; late-layer quality is data-limited (scale follow-up pending).
- At matched information the affine-transported read beats both the raw source read and identity transport (intervals excluding zero) in 20 of 136 primary-layer cells versus about 7 expected by chance.
- The win is cell-confined; reversals occur (sycophancy 0.10 below identity transport at source 20, system; evil 0.09 below the source read at source 14, many-shot). Winning cells: reconstruction cosine 0.81–0.99.
- No matched-information read CI-separates from the raw-target ceiling (evil 0.50, sycophancy 0.60, hallucination 0.46, system) or approaches the direct prompt-to-trait predictor (0.86–0.91); only the exploratory prefix-informed GRU CI-exceeds the ceiling.
- One-hop retention is high but trait-dependent: near or above 1.0 for sycophancy and hallucination, about 0.71 for evil in system mode (ridge); no trait or class met the stop criterion.
- Single-step fit quality does not predict rolled-out reliability: the affine map composes cleanly while the small MLP drifts off the activation manifold at long horizons.

## Goal

**This experiment in context:** [#779](https://eps.superkaiba.com/tasks/779) found that a learned map from a prompt's context vector to its answer-trait profile did not beat the raw persona-vector projection, while a direct context-to-trait predictor beat every pre-generation method. This experiment tests the next structural object on that same monitoring rig: the residual-stream next-activation map from one layer to the next, fit unsupervised on real chat contexts, asking how much trait-monitoring signal a predicted deeper activation retains. The data-processing inequality makes the only winnable comparison the matched-information one — does routing a layer-ℓ state through the learned dynamics to the read-out layer beat reading layer ℓ directly? The answer-profile map answered no; here it is a localized yes, above chance in aggregate but confined to a minority of source-layer cells.

**Broader narrative:** This serves the project question of whether behavior can be forecast from pre-generation internal state, and whether the residual stream's layer-to-layer dynamics are themselves a learnable, trait-informative object — a step toward monitoring traits earlier than the read-out layer where persona directions are usually taken.

## Methodology

**Design:** Two stages, both computed over #779's cached activation tensors for Qwen-2.5-7B-Instruct — no model fine-tuning, no new generation, no new judging. Stage 0 fits per-layer next-activation maps that predict the residual-stream update (the difference between consecutive layers' hidden states) at the last-prompt-token position, across four function classes: predict-zero (identity null), affine ridge, a small MLP, and a depth-GRU that consumes the whole depth trajectory. Stage 1 rolls a predicted state forward from a source layer ℓ to a per-trait read-out layer ℓ\*, projects it onto the reused persona direction there, and scores the resulting monitor with #779's within-condition Pearson-r protocol against #779's reused judged trait scores. The single manipulated variable versus #779 is what is projected onto the persona direction. The depth-GRU is prefix-informed (its state at ℓ has consumed layers 0 through ℓ, strictly more information than the source-layer-only baseline), so it is reported as exploratory and excluded from the matched-information verdict.

**Training:** No model is trained. The maps below are fit on the 5000-context LMSYS depth-trajectory corpus, split 4000 fit / 500 inner-validation / 500 test (seed 42); the 500-context test set is used only for the atlas and enters no fit, no lambda selection, and no early-stopping decision.

| Hyperparameter | Value | Source |
|---|---|---|
| Prediction target | update Δ = h(ℓ+1) − h(ℓ), in raw and per-block-RMS-normalized spaces | 2405.12250 (score the update, not the raw state); ReSAE 2605.27819 (RMS-norm) |
| Loss | SmoothL1 on the update | EAGLE 2401.15077 / HASS 2408.15766 / NextLat 2511.05963 |
| Affine ridge | closed-form with bias; lambda by generalized cross-validation over 1e-2 … 1e3 | ReSAE 2605.27819; #779/#658 lambda grid |
| Small MLP | hidden 512, GELU, AdamW lr 1e-3, weight-decay 1e-4, max-epochs 300, early-stop on inner-validation; init seed 658 | #779/#658 activation-regression defaults |
| Depth-GRU | hidden 1024, 1 layer, lr 1e-3, teacher-forced | ungrounded — smoke-tested at Stage 0 |
| Read-out layers ℓ\* (primary) | evil 20, sycophancy 26, hallucination 17 | #779 per-layer curves (step0_oracle) |
| Read-out layers ℓ\* (companion) | evil 14, sycophancy 19, hallucination 24 | #779 (the other layer-selection scheme) |
| Within-condition Pearson r | condition-std floor 1.0, minimum 3 conditions | #779 metrics (imported verbatim) |
| Bootstrap | resample conditions, ≥997 resamples, seed 0; retention ratios by joint (numerator+denominator per replicate) bootstrap | #779 |
| Judge (reused, not re-run) | claude-sonnet-4-5, graded 0–100, N=5, malformed/refusal dropped | #779 |

**Evaluation:** Two dependent variables. DV1 (atlas) is the held-out coordinate-averaged R² on the update, reported both identity-relative (predict-zero scores exactly 0) and mean-centered (the literature-comparable read that removes the carried mean). DV2 (transport monitor) is the within-condition Pearson r between the transported projection and the reused judged trait score, computed exactly as #779: per-(condition, question) scalar projection correlated against the per-(condition, question) mean judged score, condition-resampling bootstrap. Retention is DV2 as a fraction of the raw-target ceiling r, by joint bootstrap, reported unclipped. A rig-validation self-check confirmed my raw persona-vector projection reproduces #779's own numbers at #779's read-out layers to within 3e-7 for all three traits, so the internal comparisons rest on a faithfully reproduced rig (absolute anchoring stays #779's stated low-confidence regime; comparisons here are within the shared rig only). The reused judged scores' graded 0–100 trait rubric and judge prompt template are documented in the parent monitoring rig's methodology doc ([docs/methodology/issue_779.md](https://github.com/superkaiba/explore-persona-space/blob/968926c85cad1e188cdb346fd21f50a31cac3b23/docs/methodology/issue_779.md)) and its source `scripts/issue779_common.py`.

**Data extraction:** The maps are fit on the last-prompt-token residual-stream trajectories at all 28 layers for 5000 real LMSYS chat prompts. The persona directions and judged trait scores are reused verbatim from #779, which extracted each direction by the persona-vectors recipe (five contrastive positive/negative system-prompt pairs, a 20-question extraction set disjoint from evaluation, ten on-policy rollouts per arm, judge-filtered keeping positive-prompt responses above 50 and negative-prompt below 50 with refusals dropped, response-averaged difference of means per layer) and scored trait expression with the graded Sonnet judge above. Eval contexts are #779's persona-vectors conditions (8 system-prompt + 5 many-shot per trait × 40 questions), disjoint from the LMSYS map-fitting corpus by construction.

**Sample training/evaluation data + completions:** A Stage-0 fit row is one LMSYS context's depth trajectory — a `(28, 3584)` stack of last-prompt-token hidden states — with the map predicting each consecutive update from the current state; the map-fitting corpus is the full 5000-context set, not a subset. A Stage-1 evaluation unit is one (condition, question) pair carrying a scalar monitor projection and a mean judged trait score; five random system-mode units (seed 42) from hallucination's read-out-layer projection (of 160 system-mode units; the full per-unit set is embedded in each figure's data sidecar): projection −87.0 / judged 61.2; −152.5 / 28.6; −203.4 / 48.3; −186.8 / 30.2; −187.7 / 88.7 — an aggregate positive projection-to-score trend (Pearson 0.52, n=160) with visible outliers (the last two share a projection but differ 30 versus 89 in score). #779's raw rollout completions live in the data repo under `issue779_monitoring/raw_completions/` ([full listing, SHA-pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb210bc52c459959b0746cc268fe08bae96/issue779_monitoring)).

## Results

### The residual-stream layer update is highly predictable, and an affine map beats a small MLP at every depth

What is plotted: held-out identity-relative R² on the update versus layer transition (raw and RMS-normalized, 500 test contexts, predict-zero at 0), plus the per-context update-error distribution (median, p90, p99) as the low-level companion.

![Per-layer update-predictability atlas, R-squared versus layer transition per function class, raw and RMS-normalized panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/hero1_delta_atlas.png)

> **Figure.** *The layer update is highly predictable and the affine ridge leads at every depth.* Held-out identity-relative R² on the update per layer transition, per class, 500 test contexts; all learned classes sit far above the predict-zero null, predictability declines modestly with depth (ridge 0.99→0.81) and recovers at the final transition. Norm ratio on the right axis.

![Per-context update-error tails, median/p90/p99 versus layer transition](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/exploratory_delta_error_tails.png)

> **Figure.** *Per-context update error is small early and heavy-tailed at late transitions.* Median, p90, and p99 of per-context ‖Δ̂ − Δ‖ (ridge, raw) versus layer transition; the p99 tail reaches about 240 by transition 25 while the median stays near 65.

Every learned class predicts the update far above the predict-zero null. On the mean-centered R² (carried mean removed), the affine ridge explains 0.70–0.91 and beats the small MLP at all 27 transitions ([mean-centered atlas](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/exploratory_atlas_meancentered.png)). The MLP's 2000-to-4000-context scaling is flat, so ridge-beats-MLP is not a small-sample artifact. The ridge's late layers are data-limited — transitions 19 and 25 gain 21% and 24% from 2000 to 4000 contexts, above the plan's 20% data-scaling threshold — though the curve is non-monotonic near the n/d≈1.1 interpolation threshold, so a scale follow-up is pending.

### A rolled-forward affine-transported read beats the source-layer read in a minority of cells, above chance in aggregate

What is plotted: within-condition Pearson r versus source layer for the raw source read, identity transport, and affine-transported read (with ceiling and direct-predictor lines and 95% bootstrap intervals), plus the per-(condition, question) units behind hallucination's ceiling read.

![Matched-information transport: within-condition correlation versus source layer for source read, identity transport, and transported read, six trait-by-mode panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/hero_matched_information.png)

> **Figure.** *The transported read beats the source read only in a subset of cells.* Within-condition r versus source layer at the primary read-out layers, 95% intervals; near ℓ* all reads meet the ceiling, and from earlier layers the transported read sits above the raw source and identity transport in many-shot cells while evil system cells reverse. Conditions per cell 4–8.

![Per-unit hallucination scatter, raw target-layer projection versus mean judged score](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/exploratory_perunit_hallucination.png)

> **Figure.** *The per-unit data behind hallucination's ceiling read.* Each point is one (condition, question) pair: the raw target-layer projection versus its mean judged score, 260 units; the positive trend is the aggregate within-condition correlation, with wide per-unit scatter.

The transported read beats both the source and identity reads (intervals excluding zero) in 20 of 136 cells versus about 7 by chance (cleanest: evil source 8→20, 0.39 vs source 0.20, identity 0.29). Reversals occur (sycophancy 0.10 below identity transport at source 20); the largest raw delta (hallucination +0.51) is inflated by an anti-correlated source read, its transported read only 0.27, below the 0.30 ceiling. Attribution rests on transport fidelity (cosine 0.81–0.99, [fidelity](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/exploratory_transport_fidelity_cosine.png)), not the shuffle null, which exceeds the real read at one evil cell.

### Trait-signal retention stays near the ceiling for the first hops, then drops at a trait-specific mid-depth horizon

What is plotted: retained fraction of the raw-target ceiling r versus prediction horizon (read-out layer minus source layer), one line per class, system mode, per trait; the ceiling is 1.0.

![Trait-signal retention versus prediction horizon for each function class, three traits, system mode, ceiling at one](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/hero2_retention.png)

> **Figure.** *Retention holds for the first hops, then falls at a trait-specific horizon.* Retained fraction of the ceiling r versus horizon k, per class, system mode; near 1.0 for sycophancy and hallucination, about 0.71 for evil (ridge), falling sharply for sycophancy near k=14; the MLP goes negative at long horizons for hallucination. Ratios unclipped; evil companion omitted (small-ceiling ratio).

Retention is not front-loaded the way EAGLE (2401.15077) reports for token-axis rollouts: the first hop keeps most of the signal (all of it for sycophancy and hallucination, about 0.71 for evil in system mode), and the decline is a trait-specific mid-depth cliff (sharp for sycophancy near source layer 12–14, gradual for hallucination, noisy for evil). The signal surviving below the cliff is what the matched-information win draws on. Retention is unclipped and can exceed 1 or go negative; the small-ceiling companion layers are descriptive only.

### One-step fit quality does not predict rolled-out transport reliability

What is plotted: retention at each source layer versus that source transition's atlas update-R², one point per source layer, colored by class (affine ridge, small MLP), one panel per trait, system mode.

![Update-R-squared versus transport retention scatter, colored by class, three traits](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/hero3_r2_vs_retention.png)

> **Figure.** *Single-step fit quality does not determine transport reliability.* Retention at a source layer versus that layer's single-step update-R², per class, per trait; the two do not track — high-update-R² early transitions carry long horizons and low retention, and the affine ridge (blue) sits above the MLP (orange) at matched update-R².

The divergence anticipated by ReSAE (2605.27819) and EAGLE-3 (2503.01840) holds: a class's single-step update-R² does not determine how much trait signal it transports. The affine ridge has both the best update-R² and the most reliable transport, but the MLP — competitive on single-step fit — drifts off the activation manifold when composed over many hops (reconstruction R² goes large-negative at early source layers), so its long-horizon transport is unreliable. The affine map composes cleanly, and the one-long-hop direct ridge is no better than composing one-step maps.

---

**Repro:** No GPU model training or generation; all fits over the parent monitoring rig's cached tensors (~0.9 GPU-h on 1×H100, pod-841; the depth-GRU is the only GPU-worthy step). Code SHA `b2e61215f8` (Stage 0), branch `issue-841`; the result JSONs are on branch `issue-841` at `c80f143c2f` + `b64bdc93f3` (merge to main at Step 10d); HF data revision `037fcbb210bc52c459959b0746cc268fe08bae96`. Result JSONs: `eval_results/issue_841/{stage0_atlas,stage1_benchmark,retention_curve,transport_fidelity,norm_curve}.json`; per-context projections `stage1_projections.npz`. Figures pinned to `4824a567aa` on main. Vectorized split-fit MLP matched a serial reference to 2.4e-7 (parity check). Single seed (fit 42, bootstrap 0, MLP init 658). Reused metrics from #779 imported verbatim; reused judged scores not re-run.

Reused artifacts from the parent monitoring rig [#779](https://eps.superkaiba.com/tasks/779), all under `superkaiba1/explore-persona-space-data/issue779_monitoring/` at HF revision `037fcbb210bc52c459959b0746cc268fe08bae96` ([Hub tree, SHA-pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb210bc52c459959b0746cc268fe08bae96/issue779_monitoring), Hub-listed at write time):

- Reused fit corpus: `analysis_tensors/pass_b/train_context_vectors.pt` — fit: the 5000 LMSYS depth trajectories the maps are fit on; same model and last-prompt-token position as this atlas.
- Reused eval-context trajectories: 39× `analysis_tensors/pass_a/*_cx.pt` — fit: the held-out eval-context depth trajectories rolled forward in Stage 1; disjoint from the fit corpus by construction.
- Reused judged trait scores: 39× `analysis_tensors/pass_a/*.json` — fit: the DV2 regression target; graded Sonnet scores over the parent rig's exact eval conditions, imported unmodified.
- Reused persona directions: 3× `r_b/{evil,sycophancy,hallucination}.pt` — fit: the projection direction at each read-out layer; the single held-fixed variable versus the parent rig.
- Reused read-out-layer curves: `analysis_tensors/step0/step0_oracle.json` — fit: the per-layer oracle-predictivity curves from which each trait's read-out layer is selected (see the Training table).

**Context:** Originating prompt (2026-07-01 chat, excerpt): "We've found that there is a map from context vector to answer profile. Potentially there could be a map from each activation to the next one … I want to see: how well can we actually predict this, and how well can we predict this for the purposes of predicting if a model will exhibit a specific behavior … Take a bunch of contexts from LMSYS/WILDCHAT, train a linear map to predict next activation from previous activation, also try a small MLP … should we train a RNN/LSTM/SSM?, try at each layer and see what's best. Also need a deep literature search. Assess the novelty and likelihood that it will work." Follow-up scope approvals (2026-07-02 chat, excerpts): "okay I think it's worth running, can we benchmark against 779?"; "we're only interested in the next activation prediction part and benchmarking against 779." Parent #779; informed by #778, #658/#742/#761/#763, #493, #502. Created 2026-07-02; run 2026-07-02; analyzed 2026-07-02.

**Follow-up experiments:** (1) Data-scaling capture — the fired scaling gate warrants capturing last-prompt-token activations at all 28 layers for ~50–100k WildChat/LMSYS prompts and re-fitting, to test whether late-layer affine Δ-R² and transport improve with more data (cost_class: needs-gpu; headline_affecting: yes; est_gpu_hours: 5; question_relation: same). (2) Per-position sweep — capture all token positions and re-fit the atlas per position to validate the last-token magnitudes against the pooled-position literature (needs-gpu; headline_affecting: no; est_gpu_hours: 5; same). (3) Source-only-trained depth-GRU — refit a GRU consuming only the source-layer state so it enters the matched-information verdict instead of being prefix-informed (needs-gpu; headline_affecting: no; est_gpu_hours: 1; same).
