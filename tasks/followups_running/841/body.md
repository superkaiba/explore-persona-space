---
title: Rolling an activation forward through a learned affine layer map beats a direct
  read of the source layer in a minority of read-out-layer cells (above chance in
  aggregate), on Qwen-2.5-7B-Instruct (MODERATE confidence)
kind: experiment
tags:
- activation-dynamics
- context-geometry
- followup-auto
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

**Methodology:** [docs/methodology/issue_841.md](https://github.com/superkaiba/explore-persona-space/blob/863d2c05e20b3295ef970cadcda2331375504f1e/docs/methodology/issue_841.md) · [gist mirror](https://gist.github.com/superkaiba/b2b5f64629c513c99e797231863c24ce)

## Takeaways

- An affine ridge map explains 70–91% of mean-centered layer-update variance on 500 held-out contexts and beats a small MLP at all 27 transitions.
- At matched information the transported ridge read beats both the raw source read and identity transport (intervals excluding zero) in 12 of its 68 primary-layer cells versus about 3–4 by chance; the MLP separately wins 8 of its 68 (an earlier framing pooled the classes as 20 of 136).
- The late-layer data limitation is real but decelerating: 25× more fit data lifts late-band update R² 0.83 to 0.88 and grows ridge transport wins 37 to 56 of 130 pooled cells, while the mean paired transport gain stays near zero (+0.009).
- The win is cell-confined; reversals occur (sycophancy 0.10 below identity transport at source 20, system). Winning cells: reconstruction cosine 0.81–0.99.
- A source-only GRU loses to the ridge at all 27 stage-0 transitions and on the aggregate transport read (deficit 0.07, interval excluding zero; 6 versus 12 of 68 cells) — consistent with the extra prefix information, not recurrence, carrying the prefix-GRU's above-ceiling read, though that contrast is descriptive and its interval spans zero.
- No matched-information read CI-separates from the raw-target ceiling or approaches the direct prompt-to-trait predictor (0.86–0.91); one-hop retention is high but trait-dependent (about 0.71 for evil, system, ridge), and single-step fit quality does not predict rolled-out reliability.

## Goal

**This experiment in context:** [#779](https://eps.superkaiba.com/tasks/779) found that a learned map from a prompt's context vector to its answer-trait profile did not beat the raw persona-vector projection, while a direct context-to-trait predictor beat every pre-generation method. This experiment tests the next structural object on that same monitoring rig: the residual-stream next-activation map from one layer to the next, fit unsupervised on real chat contexts, asking how much trait-monitoring signal a predicted deeper activation retains. The data-processing inequality makes the only winnable comparison the matched-information one — does routing a layer-ℓ state through the learned dynamics to the read-out layer beat reading layer ℓ directly? The answer-profile map answered no; here it is a localized yes, above chance in aggregate but confined to a minority of source-layer cells.

**Broader narrative:** This serves the project question of whether behavior can be forecast from pre-generation internal state, and whether the residual stream's layer-to-layer dynamics are themselves a learnable, trait-informative object — a step toward monitoring traits earlier than the read-out layer where persona directions are usually taken.

## Methodology

**Design:** Two stages, both computed over the parent monitoring rig's cached activation tensors for Qwen-2.5-7B-Instruct — no model fine-tuning, no new generation, no new judging. Stage 0 fits per-layer next-activation maps that predict the residual-stream update (the difference between consecutive layers' hidden states) at the last-prompt-token position, across four function classes: predict-zero (identity null), affine ridge, a small MLP, and a depth-GRU that consumes the whole depth trajectory. Stage 1 rolls a predicted state forward from a source layer ℓ to a per-trait read-out layer ℓ\*, projects it onto the reused persona direction there, and scores the resulting monitor with the parent rig's within-condition Pearson-r protocol against its reused judged trait scores. The single manipulated variable versus the parent rig is what is projected onto the persona direction. The depth-GRU is prefix-informed (its state at ℓ has consumed layers 0 through ℓ, strictly more information than the source-layer-only baseline), so it is reported as exploratory and excluded from the matched-information verdict. Two same-issue follow-up rounds extend the rig with everything else held fixed: a capture-size scaling round (single manipulated variable: fit-corpus size — 96,000 newly captured LMSYS contexts, ridge refit at five sizes and the MLP at the two endpoints, the transport benchmark re-run at every size against the same 500-context test set and unchanged eval surface), and a source-only-GRU round (single manipulated variable: the GRU's input information set — from the whole depth trajectory to only the source-layer state, entering the recurrent class into the matched-information verdict it was excluded from).

**Training:** No model is trained. The maps below are fit on the 5000-context LMSYS depth-trajectory corpus, split 4000 fit / 500 inner-validation / 500 test (seed 42); the 500-context test set is used only for the atlas and enters no fit, no lambda selection, and no early-stopping decision. The scaling round grows only the fit pool; the validation and test sets stay fixed.

| Hyperparameter | Value | Source |
|---|---|---|
| Prediction target | update Δ = h(ℓ+1) − h(ℓ), in raw and per-block-RMS-normalized spaces | 2405.12250 (score the update, not the raw state); ReSAE 2605.27819 (RMS-norm) |
| Loss | SmoothL1 on the update | EAGLE 2401.15077 / HASS 2408.15766 / NextLat 2511.05963 |
| Affine ridge | closed-form with bias; lambda by generalized cross-validation over 1e-2 … 1e3 | ReSAE 2605.27819; #779/#658 lambda grid |
| Small MLP | hidden 512, GELU, AdamW lr 1e-3, weight-decay 1e-4, max-epochs 300, early-stop on inner-validation; init seed 658 | #779/#658 activation-regression defaults |
| Depth-GRU (prefix-informed, exploratory) | hidden 1024, 1 layer, lr 1e-3, teacher-forced | ungrounded — smoke-tested at Stage 0 |
| Scaling round: fit-corpus sizes | ridge n ∈ 4,000 / 10,000 / 25,000 / 50,000 / 100,000 (anchor 4,000); MLP at 4,000 and 100,000 only | round plan §4.2 (`scaling-capture`) |
| Scaling round: new capture | 96,000 new LMSYS last-prompt-token trajectories, 28 layers, bf16 (matches the parent capture path); 10,073 parent-colliding prompts dropped from a 111,073-prompt stream | capture manifest `issue841_scaling/cx_last_shards/manifest.json` |
| Source-only GRU round | GRU architecture, lr, loss, split, and RMS-norm target held at the parent depth-GRU's values (hidden 1024, embedding 32, 1 layer, lr 1e-3, SmoothL1); input reduced to the source-layer state alone; early-stopped at epoch 118 of 300 on inner-validation | round plan §4.1 (`gru-source-only`); `stage1_gru_source_only.json` convergence block |
| Read-out layers ℓ\* (primary) | evil 20, sycophancy 26, hallucination 17 | #779 per-layer curves (step0_oracle) |
| Read-out layers ℓ\* (companion) | evil 14, sycophancy 19, hallucination 24 | #779 (the other layer-selection scheme) |
| Within-condition Pearson r | condition-std floor 1.0, minimum 3 conditions | #779 metrics (imported verbatim) |
| Bootstrap | resample conditions, ≥997 resamples, seed 0; retention ratios by joint (numerator+denominator per replicate) bootstrap | #779 |
| Judge (reused, not re-run) | claude-sonnet-4-5, graded 0–100, N=5, malformed/refusal dropped | #779 |

**Evaluation:** Two dependent variables. DV1 (atlas) is the held-out coordinate-averaged R² on the update, reported both identity-relative (predict-zero scores exactly 0) and mean-centered (the literature-comparable read that removes the carried mean). DV2 (transport monitor) is the within-condition Pearson r between the transported projection and the reused judged trait score, computed exactly as the parent rig: per-(condition, question) scalar projection correlated against the per-(condition, question) mean judged score, condition-resampling bootstrap. Retention is DV2 as a fraction of the raw-target ceiling r, by joint bootstrap, reported unclipped. A transport "win" is the conjunction predicate — the transported read beats BOTH the raw source read AND identity transport with the per-context-paired 95% interval excluding zero; the chance line is 5% of cells (about 3–4 of 68 per class). The scaling round re-derives the win-count at every fit size over the 130 pooled primary+companion cells, with the newly-winning subcount Benjamini-Hochberg-corrected at FDR 0.05; the source-only-GRU round derives the ridge-alone baseline (12 of 68) in-run under the identical predicate and adds a shuffled-context chance line (0 cells pass). A rig-validation self-check confirmed my raw persona-vector projection reproduces the parent rig's own numbers at its read-out layers to within 3e-7 for all three traits, so the internal comparisons rest on a faithfully reproduced rig (absolute anchoring stays the parent rig's stated low-confidence regime; comparisons here are within the shared rig only). The reused judged scores' graded 0–100 trait rubric and judge prompt template are documented in the parent monitoring rig's methodology doc ([docs/methodology/issue_779.md](https://github.com/superkaiba/explore-persona-space/blob/968926c85cad1e188cdb346fd21f50a31cac3b23/docs/methodology/issue_779.md)) and its source `scripts/issue779_common.py`.

**Data extraction:** The maps are fit on the last-prompt-token residual-stream trajectories at all 28 layers for 5000 real LMSYS chat prompts; the scaling round extends this pool to 100,000 with a fresh capture gated by a 12-probe re-capture equivalence check (minimum stored-vs-recaptured cosine 0.99984, worst relative norm error 0.018, within tolerance) plus a logged adjacency audit reported under the scaling result below. The persona directions and judged trait scores are reused verbatim from the parent rig, which extracted each direction by the persona-vectors recipe (five contrastive positive/negative system-prompt pairs, a 20-question extraction set disjoint from evaluation, ten on-policy rollouts per arm, judge-filtered keeping positive-prompt responses above 50 and negative-prompt below 50 with refusals dropped, response-averaged difference of means per layer) and scored trait expression with the graded Sonnet judge above. Eval contexts are the parent rig's persona-vectors conditions (8 system-prompt + 5 many-shot per trait × 40 questions), disjoint from the LMSYS map-fitting corpus by construction; the scaling capture additionally deduplicates against the parent corpus by prompt text.

**Sample training/evaluation data + completions:** A Stage-0 fit row is one LMSYS context's depth trajectory — a `(28, 3584)` stack of last-prompt-token hidden states — with the map predicting each consecutive update from the current state; the map-fitting corpus is the full 5000-context set (100,000 in the scaling round), not a subset. A Stage-1 evaluation unit is one (condition, question) pair carrying a scalar monitor projection and a mean judged trait score; five random system-mode units (seed 42) from hallucination's read-out-layer projection (of 160 system-mode units; the full per-unit set is embedded in each figure's data sidecar): projection −87.0 / judged 61.2; −152.5 / 28.6; −203.4 / 48.3; −186.8 / 30.2; −187.7 / 88.7 — an aggregate positive projection-to-score trend (Pearson 0.52, n=160) with visible outliers (the last two share a projection but differ 30 versus 89 in score). From the source-only-GRU round's one winning system-mode cell (sycophancy, source layer 10, of 160 system-mode units; full per-unit projections in `gru_source_only_projections.npz` on the private overflow repo, pointer on the canonical data repo): projection −35.7 / judged 0.5; 53.5 / 66.9; −691.6 / 14.4; −765.2 / 70.6; −986.9 / 0.0 — within-condition r 0.38 with wide per-unit scatter, matching the benchmark JSON. The parent rig's raw rollout completions live in the data repo under `issue779_monitoring/raw_completions/` ([full listing, SHA-pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb210bc52c459959b0746cc268fe08bae96/issue779_monitoring)).

## Results

### The residual-stream layer update is highly predictable, and an affine map beats a small MLP at every depth

What is plotted: held-out identity-relative R² on the update versus layer transition (raw and RMS-normalized, 500 test contexts, predict-zero at 0), plus the per-context update-error distribution (median, p90, p99) as the low-level companion.

![Per-layer update-predictability atlas, R-squared versus layer transition per function class, raw and RMS-normalized panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/hero1_delta_atlas.png)

> **Figure.** *The layer update is highly predictable and the affine ridge leads at every depth.* Held-out identity-relative R² on the update per layer transition, per class, 500 test contexts; all learned classes sit far above the predict-zero null, predictability declines modestly with depth (ridge 0.99→0.81) and recovers at the final transition. Norm ratio on the right axis.

![Per-context update-error tails, median/p90/p99 versus layer transition](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/exploratory_delta_error_tails.png)

> **Figure.** *Per-context update error is small early and heavy-tailed at late transitions.* Median, p90, and p99 of per-context ‖Δ̂ − Δ‖ (ridge, raw) versus layer transition; the p99 tail reaches about 240 by transition 25 while the median stays near 65.

Every learned class predicts the update far above the predict-zero null. On the mean-centered R² (carried mean removed), the affine ridge explains 0.70–0.91 and beats the small MLP at all 27 transitions ([mean-centered atlas](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/exploratory_atlas_meancentered.png)). The MLP's 2000-to-4000-context scaling is flat, so ridge-beats-MLP is not a small-sample artifact. The ridge's late layers looked data-limited (transitions 19 and 25 gained 21% and 24% from 2000 to 4000 contexts, above the plan's 20% threshold); the scaling round below settles that read — the gain is real and decelerating, not subsample noise near the interpolation threshold.

### More fit data keeps improving the late-layer map, with strongly diminishing returns by 100,000 contexts

What is plotted: held-out identity-relative R² on the update versus fit-set size at the nine data-limited transitions 17–25, ridge and MLP, fixed 500-context test set; each line is one transition — the per-unit view behind the band mean.

![Update R-squared versus fit-set size at the data-limited transitions, ridge and MLP](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4e4485e464fac0452fa20415ac1631fc2ac274aa/figures/issue_841/scaling-capture/hero_r2_scaling_r2_id.png)

> **Figure.** *Late-layer update predictability keeps rising with fit data but decelerates.* Held-out identity-relative R² versus fit-set size per late transition; every ridge line (solid) rises monotonically, the 4,000-context anchor (dotted) reproduces the parent atlas exactly, and the MLP endpoints (dashed) stay below the ridge at matched size.

The late-band mean rises 0.829 to 0.876 across 25× more data — just under the plan's 0.05 threshold for a settled data-limitation call — and each doubling buys less; the final step adds about 0.004 ([mean-centered companion](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4e4485e464fac0452fa20415ac1631fc2ac274aa/figures/issue_841/scaling-capture/hero_r2_scaling_r2_meancentered.png): 0.752→0.822). The MLP gains similarly but stays below the ridge at matched size. The anchor refit reproduces the parent atlas and the dual-vs-primal cross-check agrees, both to numerical precision. A position-drift diagnostic finds the late capture window fits the test set 0.013 worse than the anchor, mixing mild distribution drift into "more data". The adjacency audit settles the previously unmeasured off-by-one concern: a second-to-last-token recapture is rejected for all 12 probes (adjacent cosine 0.61–0.74 versus the 0.9998 acceptance floor).

### Transport wins grow with fit-corpus size, but the aggregate paired gain stays small

What is plotted: per class, the conjunction win-count over the 130 pooled primary+companion cells and the mean paired within-condition-r delta versus the 4,000-context anchor (95% intervals) at each fit size, with the per-cell delta strip behind the aggregate.

![Transport win-count and mean paired delta versus fit-set size per class](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c9469af9e5d7a91dbe27cf3cf58f65226209302c/figures/issue_841/scaling-capture/hero_transport_scaling.png)

> **Figure.** *Wins rise with data; the mean gain stays near zero.* Left: conjunction wins of 130 cells versus fit size (ridge 37→56; chance line dashed red at 6.5). Right: mean paired r-delta versus the anchor, interval spanning zero for ridge and MLP at every size; the direct-hop ridge clears zero at 100,000.

![Per-cell within-condition r delta versus the 4,000-context anchor, all 130 ridge cells, per fit size](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4e4485e464fac0452fa20415ac1631fc2ac274aa/figures/issue_841/scaling-capture/exploratory_percell_delta_strip.png)

> **Figure.** *The per-unit data behind the aggregate.* Each point is one (trait, source, mode) cell's r change versus the anchor (ridge); per-cell spread (±0.2–0.4) dwarfs the +0.009 mean (black diamond, 95% interval), so the data benefit is cell-specific, not a uniform lift.

Ridge wins grow monotonically from 37 to 56 of 130 cells; 19 of the 22 newly-winning cells at the largest size survive multiple-comparison correction (6.5 by chance) — yet the mean paired delta is only +0.009 with an interval spanning zero at every size, and the per-cell change at 100,000 is two-sided (55 cells improve significantly, 20 degrade). This sits between the plan's data-limited and flat outcomes: more unsupervised data reliably flips specific cells without lifting the grid uniformly, and the win growth is trait-skewed (hallucination 16 to 28 cells, sycophancy 14 to 18, evil 7 to 10). Transport fidelity edges up (cosine 0.887 to 0.894) and retention is flat (0.57–0.59); no inverted-U appears.

### A rolled-forward affine-transported read beats the source-layer read in a minority of cells, above chance in aggregate

What is plotted: within-condition Pearson r versus source layer for the raw source read, identity transport, and affine-transported read (with ceiling and direct-predictor lines and 95% bootstrap intervals), plus the per-(condition, question) units behind hallucination's ceiling read.

![Matched-information transport: within-condition correlation versus source layer for source read, identity transport, and transported read, six trait-by-mode panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/hero_matched_information.png)

> **Figure.** *The transported read beats the source read only in a subset of cells.* Within-condition r versus source layer at the primary read-out layers, 95% intervals; near ℓ* all reads meet the ceiling, and from earlier layers the transported read sits above the raw source and identity transport in many-shot cells while evil system cells reverse. Conditions per cell 4–8.

![Per-unit hallucination scatter, raw target-layer projection versus mean judged score](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/exploratory_perunit_hallucination.png)

> **Figure.** *The per-unit data behind hallucination's ceiling read.* Each point is one (condition, question) pair: the raw target-layer projection versus its mean judged score, 260 units; the positive trend is the aggregate within-condition correlation, with wide per-unit scatter.

Per class, the transported ridge read beats both the source and identity reads (intervals excluding zero) in 12 of its 68 primary-layer cells and the small MLP in 8 of its 68, each versus about 3–4 by chance (cleanest: evil source 8→20, 0.39 vs source 0.20, identity 0.29); an earlier framing pooled the two classes into 20 of 136, crediting the affine map alone. Reversals occur; the largest raw delta (hallucination +0.51) is inflated by an anti-correlated source read (transported 0.27, ceiling 0.30). Attribution rests on transport fidelity (cosine 0.81–0.99, [fidelity](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/exploratory_transport_fidelity_cosine.png)), not the shuffle null, which exceeds the real read at one evil cell.

### A source-only GRU loses to the affine ridge at every stage-0 transition and on the aggregate transport read

What is plotted: the Stage-0 atlas with the source-only GRU overlaid on the parent's ridge, MLP, and prefix-GRU lines, plus the per-cell paired within-condition-r delta forest versus ridge and prefix-GRU.

![Stage-0 atlas with the source-only GRU overlaid on ridge, MLP, and prefix-GRU](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4e4485e464fac0452fa20415ac1631fc2ac274aa/figures/issue_841/gru_source_only/heroA_stage0_atlas.png)

> **Figure.** *The source-only GRU sits below ridge and MLP at every transition.* Held-out identity-relative R² on the update per transition; the source-only GRU (blue) sits below the MLP (green) and ridge (orange) in both target spaces, and above the prefix-GRU (red) at all 27 raw transitions but only 22 of 27 RMS-normalized (losing transitions 1–5).

![Per-cell paired delta forest, source-only GRU minus ridge and minus prefix-GRU](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c9469af9e5d7a91dbe27cf3cf58f65226209302c/figures/issue_841/gru_source_only/exp_delta_forest.png)

> **Figure.** *The per-cell transport deltas behind the aggregate deficit.* Paired within-condition-r delta per (source, mode) cell, source-only GRU minus affine ridge (blue) and minus prefix-GRU (orange), 95% intervals; most intervals sit at or below zero, with the scattered positive cells concentrated in hallucination.

The source-only GRU beats the ridge at 0 of 27 stage-0 transitions, wins 6 of 68 transport cells versus the ridge's 12, and trails the transported ridge by 0.07 in aggregate, interval excluding zero (shuffled chance: 0 cells). Four of its six wins are cells the ridge does not win — complementarity. It out-fits the prefix-GRU on all 27 raw transitions yet transports no better (−0.03, interval spanning zero) — consistent with the prefix information, not recurrence, carrying the parent's above-ceiling read; the contrast is cross-transport-regime (27× fewer prefix-arm optimizer steps; convergence diagnostics not stored), descriptive only. Scope caveat: unlike the per-transition ridge and MLP fits, this GRU is one model shared across the 27 transitions (transition embedding), partly conflating map class with parametrization.

### Trait-signal retention stays near the ceiling for the first hops, then drops at a trait-specific mid-depth horizon

What is plotted: retained fraction of the raw-target ceiling r versus prediction horizon (read-out layer minus source layer), one line per class, system mode, per trait (the ceiling is 1.0), alongside the raw transported within-condition r per source layer that the ratio is built from.

![Trait-signal retention versus prediction horizon for each function class, three traits, system mode, ceiling at one](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/hero2_retention.png)

> **Figure.** *Retention holds for the first hops, then falls at a trait-specific horizon.* Retained fraction of the ceiling r versus horizon k, per class, system mode; near 1.0 for sycophancy and hallucination, about 0.71 for evil (ridge), falling sharply for sycophancy near k=14; the MLP goes negative at long horizons for hallucination. Ratios unclipped; evil companion omitted (small-ceiling ratio).

![Raw transported within-condition r per source layer per function class, three traits, system mode, with the raw-target ceiling reference](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d840cebb1f48bc1ed8a9a5d590dab2094a970550/figures/issue_841/hero2_retention_raw.png)

> **Figure.** *The raw correlations behind the retention ratio.* Transported within-condition r per source layer, one line per class, system mode, against the constant raw-target ceiling r (dashed; evil 0.50, sycophancy 0.60, hallucination 0.46). Ratio above 1 is transported r above the ceiling (sycophancy, deep sources); negative ratio is r below zero (MLP, hallucination sources 2–4).

Retention is not front-loaded the way EAGLE (2401.15077) reports for token-axis rollouts: the first hop keeps most of the signal (all of it for sycophancy and hallucination, about 0.71 for evil in system mode), and the decline is a trait-specific mid-depth cliff (sharp for sycophancy near source layer 12–14, gradual for hallucination, noisy for evil). The signal surviving below the cliff is what the matched-information win draws on. Retention is unclipped and can exceed 1 or go negative; the small-ceiling companion layers are descriptive only.

### One-step fit quality does not predict rolled-out transport reliability

What is plotted: retention at each source layer versus that source transition's atlas update-R², one point per source layer, colored by class (affine ridge, small MLP), one panel per trait, system mode.

![Update-R-squared versus transport retention scatter, colored by class, three traits](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4824a567aae82fc212c29d8c37d4bbffa5f8b613/figures/issue_841/hero3_r2_vs_retention.png)

> **Figure.** *Single-step fit quality does not determine transport reliability.* Retention at a source layer versus that layer's single-step update-R², per class, per trait; the two do not track — high-update-R² early transitions carry long horizons and low retention, and the affine ridge (blue) sits above the MLP (orange) at matched update-R².

The divergence anticipated by ReSAE (2605.27819) and EAGLE-3 (2503.01840) holds: a class's single-step update-R² does not determine how much trait signal it transports. The affine ridge has both the best update-R² and the most reliable transport, but the MLP — competitive on single-step fit — drifts off the activation manifold when composed over many hops (reconstruction R² goes large-negative at early source layers), so its long-horizon transport is unreliable. The affine map composes cleanly, and the direct-hop ridge (one long hop) is no better than composing one-step maps at the anchor size; only at 100,000 contexts does its aggregate paired gain clear zero, and marginally (+0.013, interval lower bound +0.001), while its win count is non-monotone in fit size (42 to 58, 61, 57, 58).

---

**Repro:** No GPU model training or generation in the parent round; all fits over the parent monitoring rig's cached tensors (~0.9 GPU-h on 1×H100, pod-841; the depth-GRU is the only GPU-worthy step). Same-issue follow-up round 1 (label: `scaling-capture`, ~5 GPU-h: one A100 capture pass over 96,000 LMSYS contexts + refits) and same-issue follow-up round 2 (label: `gru-source-only`, ~1 GPU-h), both run at code SHA `566eb04d01`. Parent code SHA `b2e61215f8` (Stage 0), branch `issue-841`; result JSONs on branch `issue-841` — parent at `c80f143c2f` + `b64bdc93f3`, follow-up rounds at `d7662ff21b` (`eval_results/issue_841/scaling-capture/{stage0_scaling,stage1_scaling,transport_fidelity_scaling}.json`, `eval_results/issue_841/gru_source_only/{stage0,stage1,retention,transport_fidelity}_gru_source_only.json`); HF data revision `037fcbb210bc52c459959b0746cc268fe08bae96`. Parent result JSONs: `eval_results/issue_841/{stage0_atlas,stage1_benchmark,retention_curve,transport_fidelity,norm_curve}.json`; per-context projections `stage1_projections.npz`. Round HF artifacts (Hub-listed at write time): `issue841_scaling/{results,figures,cx_last_shards,ridge_maps_n4000..n100000,mlp_maps_n4000,mlp_maps_n100000}` and `issue841_gru_source_only/{results,figures}` on `superkaiba1/explore-persona-space-data`; LFS tensors (capture shards, fitted maps, `scaling_projections.npz`, `gru_source_only_projections.npz`, GRU state dicts) routed to the private overflow repo `superkaiba1/explore-persona-space-overflow` with `OVERFLOW_POINTER.json` alongside on the canonical repo (public LFS quota). Figures pinned to `4824a567aa` and `d840cebb1f` (parent), `4e4485e464` (follow-up rounds), and `c9469af9e5` (relabeled transport hero + delta forest) on branch `issue-841`. Vectorized split-fit MLP matched a serial reference to 2.4e-7; scaling anchor refit reproduced the parent atlas to 2e-16 and the 10,000-context dual-vs-primal cross-check to 3e-16; GRU forward parity to 1e-16 (fp64). Single seed (fit 42, bootstrap 0, MLP init 658). Reused metrics from #779 imported verbatim; reused judged scores not re-run.

Reused artifacts from the parent monitoring rig [#779](https://eps.superkaiba.com/tasks/779), all under `superkaiba1/explore-persona-space-data/issue779_monitoring/` at HF revision `037fcbb210bc52c459959b0746cc268fe08bae96` ([Hub tree, SHA-pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb210bc52c459959b0746cc268fe08bae96/issue779_monitoring), Hub-listed at write time):

- Reused fit corpus: `analysis_tensors/pass_b/train_context_vectors.pt` — fit: the 5000 LMSYS depth trajectories the maps are fit on; same model and last-prompt-token position as this atlas.
- Reused eval-context trajectories: 39× `analysis_tensors/pass_a/*_cx.pt` — fit: the held-out eval-context depth trajectories rolled forward in Stage 1; disjoint from the fit corpus by construction.
- Reused judged trait scores: 39× `analysis_tensors/pass_a/*.json` — fit: the DV2 regression target; graded Sonnet scores over the parent rig's exact eval conditions, imported unmodified.
- Reused persona directions: 3× `r_b/{evil,sycophancy,hallucination}.pt` — fit: the projection direction at each read-out layer; the single held-fixed variable versus the parent rig.
- Reused read-out-layer curves: `analysis_tensors/step0/step0_oracle.json` — fit: the per-layer oracle-predictivity curves from which each trait's read-out layer is selected (see the Training table).

**Context:** Originating prompt (2026-07-01 chat, excerpt): "We've found that there is a map from context vector to answer profile. Potentially there could be a map from each activation to the next one … I want to see: how well can we actually predict this, and how well can we predict this for the purposes of predicting if a model will exhibit a specific behavior … Take a bunch of contexts from LMSYS/WILDCHAT, train a linear map to predict next activation from previous activation, also try a small MLP … should we train a RNN/LSTM/SSM?, try at each layer and see what's best. Also need a deep literature search. Assess the novelty and likelihood that it will work." Follow-up scope approvals (2026-07-02 chat, excerpts): "okay I think it's worth running, can we benchmark against 779?"; "we're only interested in the next activation prediction part and benchmarking against 779." Proposer-initiated cheap-band rounds, run 2026-07-03/04 — same-issue follow-up round 1 (label: `scaling-capture`; scope excerpt: "capture last-prompt-token activations at all 28 layers for ~50-100k WildChat/LMSYS prompts … re-fit the affine ridge atlas on the larger corpus … Single manipulated variable: fit-corpus size") and same-issue follow-up round 2 (label: `gru-source-only`; scope excerpt: "refit the depth-GRU consuming ONLY the source-layer state … so the GRU class enters the matched-information verdict it was excluded from"). Parent #779; informed by #778, #658/#742/#761/#763, #493, #502. Created 2026-07-02; run 2026-07-02; follow-up rounds run 2026-07-03/04; analyzed 2026-07-04.
