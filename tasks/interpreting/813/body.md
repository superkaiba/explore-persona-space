---
title: Fine-tune-induced map change is indistinguishable across query substrates for
  three of four behaviors, and the emergent-misalignment exception clears question-resampling
  noise but not context-family variance (LOW confidence)
kind: experiment
tags:
- followup-manual
created_at: '2026-07-01T18:56:55Z'
has_clean_result: true
parent_id: 667
origin_prompt: 'Spec + run the clean map-change substrate-dependence experiment: 4
  behaviors x 3 query substrates (generic UltraChat vs behavior-eliciting vs mix),
  reuse #537 adapters, 50-context battery, question-averaged c_C, PCA-48, M0 vs M+
  (Delta/floor + chain-rho). Run in background (happy coder, autonomous). Resume pod-667;
  if not up in 5 min, provision a new pod. SAVE+UPLOAD the pre and post trained maps
  AND ALL context+answer activations pre/post finetuning UNREDUCED (per-token, per-question,
  all layers, base+trained) for followups. Ensure the pod does not run out of space.'
goal: 'Determine whether the finetuning-induced context->answer map change (M0 base
  vs M+ trained, per #537 behavior) depends on the query substrate, by fitting + comparing
  M0 vs M+ across three substrates (generic UltraChat, behavior-eliciting probes,
  mix) on the shared 50-context battery, reporting floor-normalized function-change
  and chain-rho per substrate.'
relates_to:
- leak-predictor
---
# Fine-tune-induced map change is indistinguishable across query substrates for three of four behaviors, and the emergent-misalignment exception clears question-resampling noise but not context-family variance (LOW confidence)

<!-- clean-result-v4 -->

## Takeaways

- **No behavior meets the registered substrate-matters rule:** fact, sycophancy, marker sit inside the substrate-swap null (spreads 0.09, 2.5, 0.0007 vs null 95th percentiles 1.36, 2.90, 0.0077); emergent misalignment splits it — clears the null band, fails the pairwise leg.
- **Emergent misalignment's spread (4.71, generic vs mix) clears its null 95th percentile (3.63) by 30%, but the driving pair's family-clustered interval spans zero** — an ambiguous split of the rule. A per-layer recompute adds that the spread is mid-stack-local: it peaks at the frozen layer 14, and the substrate ordering inverts at early layers.
- **That 30% band margin is an order of magnitude beyond the ~2% engine shift and 1.5–3% resampling noise** — too large to attribute to the measured null-engine effects.
- **For emergent misalignment and sycophancy the spread is mostly denominator-driven:** the 48-question generic pool gives a 3–5x smaller refit floor; the raw map change moves far less.
- **Fact and marker never clear their own refit floor at the frozen layer 14 on any substrate (Δ/floor 0.29–0.38, 0.040–0.041):** low-power non-rejections, not substrate-invariance — across layers, fact exceeds its floor early in the stack (peaks 2.1–2.8) while marker stays below it at all 27 layers (max 0.48).
- **The chain-ρ co-primary was uncomputable** — 0 of 50 battery contexts joined the reused leakage table in all 8 cells; the registered fallback left Δ/floor carrying the headline.

## Goal

- **This experiment in context:** The leakage-predictor line fits a linear context-to-answer map M from base-model activations and reads the fine-tune's change off it; the map-fit protocol comes from [#667](https://eps.superkaiba.com/tasks/667) and [#722](https://eps.superkaiba.com/tasks/722), the activation-summary recipe and probe pools from [#658](https://eps.superkaiba.com/tasks/658) and [#594](https://eps.superkaiba.com/tasks/594), and the four trained behaviors are the reused default-context contrastive adapters from [#537](https://eps.superkaiba.com/tasks/537). The map's inputs and outputs are expectations over a question distribution, so everything built on M implicitly assumes its change is question-pool-invariant. This run makes that the manipulated variable: the same 50-context battery map is fit three times per behavior — generic UltraChat questions, the behavior's own eliciting probes, a balanced mix — and the floor-normalized map change is compared under the plan's two-legged rule.
- **Broader narrative:** If the map change moved with the probe questions, the predictor line's calibration would depend on the query pool it was built from. Measured: no substrate-dependence survives the registered rule for three of four behaviors (two power-limited), and the exception is ambiguous rather than positive — the map object survives this robustness check within the power caveats below.

## Methodology

**Design:** Training-free reuse analysis on `Qwen/Qwen2.5-7B-Instruct` (bf16). For each of 4 behaviors (emergent misalignment, fact, sycophancy, marker) the same 50-context battery is probed under 3 question substrates — generic (48 UltraChat questions), behavior-eliciting (the behavior's own eval pool: marker 32, fact 30, sycophancy 25, emergent misalignment 8 questions), and mix (equal halves, sized to twice the smaller pool: 64/60/50/16) — through both the base model and the adapter-applied trained model, giving 12 (behavior × substrate) cells × 2 model arms. Per cell, a ridge map M from question-averaged context activations to question-averaged answer activations is fit for base and trained separately, and the dependent variable is the floor-normalized map-output change Δ/floor at the frozen headline layer 14. The registered verdict per behavior is a conjunction: substrate matters only if the max-vs-min Δ/floor spread exceeds the substrate-swap null's 95th percentile (the widest of the three per-substrate nulls) AND the driving pair's family-clustered bootstrap 95% interval excludes zero; both legs failing reads substrate-agnostic; a split reads ambiguous. A zero-GPU follow-up round recomputed the same dependent variable at every layer 1–27 for all 12 cells from the uploaded per-layer maps and reduced summaries (no new extraction), with a per-layer context-input-drift companion read.

**Training:** **N/A — no model training.** The four adapters are reused artifacts: contrastive LoRA installs trained under the bare default-assistant context (positives carry the behavior, roughly 1:1 interleaved contrastive negatives under other personas omit it), one per behavior, seed 42, on the same base model. Their configs, read from each `adapter_config.json`: marker r = 32, α = 64, rsLoRA, 4 attention modules; fact and sycophancy r = 32, α = 64, rsLoRA, 7 modules; emergent misalignment r = 32, α = 256, rsLoRA, 7 modules (per-behavior recipe, nested one level under `sft_em_adapter/`). Analysis constants:

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct`, HF sha `0718c53058475cb8ee38c8f4802220cdde548672` | reused adapter line |
| Map inputs c_C | last-input-token residual, question-averaged per substrate, 50 battery contexts | context-geometry recipe (#594/#658 line, stated in plan §11) |
| Map outputs v_A | mean-answer-span residual, question-averaged per substrate | same recipe |
| Target basis | top-64 PCA of base-side outputs, shared by both model arms | map-fit protocol (`issue667_save_maps.py::TARGET_DIM`) |
| Ridge | closed-form dual ridge, PRESS-LOO λ over 1e-2…1e3, input-normalized | `issue722_fit_M.py` / `issue658` ridge helpers |
| Cross-validation | leave-one-context-out; refit floor = max over base/trained/shifted refits | `issue722_fit_M.py::fit_cell` |
| Δ/floor (emergent misalignment, fact, sycophancy) | median over contexts of the r_B-projected map-output change ÷ SD refit floor | `fit_cell` `Delta_over_floor_sd` |
| Δ/floor (marker) | unprojected map-output change ÷ 95th-percentile combined refit floor (no fit-side r_B) | `issue813_analysis.py` marker read |
| Headline layer | 14, frozen in the plan, applied identically to observed and null | plan §6.6 (#651/#658 read layer) |
| Substrate-swap null | resample questions within substrate, split into two matched-size pseudo-substrates, refit both, Δ/floor difference; 1000 resamples × 40 refit pairs, seed 42 | plan §3; batched Gram-space engine `batched_gram_v1` |
| Pairwise CIs | family-clustered bootstrap over the 7 battery context families, 1000 resamples | `issue722_bootstrap` protocol |
| Base-response generation | greedy, temperature 0; max new tokens 2048 (marker) / 1024 (others); vLLM | marker measurement rule |
| Marker token | ` ※` id 83399, asserted in-process | marker measurement rule |

**Evaluation:** The construct is the map itself, not on-policy behavior: Δ/floor measures how much the fine-tune moved the map's outputs, in units of the leave-one-context-out refit noise floor, per substrate. A behavior direction r_B projects the change for emergent misalignment and sycophancy (diff-of-means directions) and fact (re-extracted direction); the marker cell uses the unprojected read plus a secondary unembedding-row projection (informative, 13% of the change in the read subspace). The planned co-primary — a rank correlation between the map's r_B read-out and measured behavioral leakage, base vs trained, on the eliciting and mix substrates — could not be computed: the reused leakage table is keyed to its own producing grid, and 0 of the 50 battery contexts matched in any of the 8 planned cells; the plan's sparse-join fallback (drop the co-primary below 10 joins) fired. The apply-parity gate passed numerically for all four adapters (the emergent-misalignment write ratio reproduced the prior committed value exactly; marker's 0.009 is in-gauge for a 4-module writer, where a wrong rsLoRA gauge would read 5.66x off). The marker behavioral parity read was demoted to a diagnostic warning (~0.8 nat vs the committed 6.1-nat reference): the unresolved fork — probe-rig conditioning mismatch versus a consistently weak shared loader — caps only marker's absolute map-change magnitude; the within-marker substrate comparison uses the same adapter and loader in all three substrates.

**Data extraction:** Tier 1/2 — real UltraChat user questions (toxicity-filtered, 48 probes), the frozen within-project eval pools (established probe banks predating the adapters), and the 50-context battery (7 families: personas, real WildChat prefixes, in-context-learning demos, rephrasings, format wraps, behavior instructions, defaults; build seed 42). Per (context, question), the base model generates a frozen greedy response; that exact text is teacher-forced through base and trained models once each, capturing per-token residuals at all 28 layers (fp16); reduction to c_C (last input token) and v_A (answer-span mean) is question-averaged per substrate. Compute-path notes: 4 of 12 extraction cells began under a zlib-compressed tensor writer and resumed under an uncompressed writer (storage format only; identical data); the null battery ran uniformly on the batched Gram-space engine, with the emergent-misalignment generic and eliciting cells (and at most one more) computed on its CUDA leg before a cuSOLVER convergence failure moved the remaining cells to the CPU-verified leg (per-cell device is not recorded in the JSONs; the measured engine shift, about 2% on the null 95th percentile in the anti-conservative direction, is an order of magnitude below every verdict margin).

**Sample training/evaluation data + completions:** No new training rows and no judged completions exist in this run; the load-bearing text inputs are the battery contexts and the substrate question pools. Battery contexts (3 of 50, seed-42 sample; full battery: [data/issue594/battery.json](https://github.com/superkaiba/explore-persona-space/blob/cd961b94f4/data/issue594/battery.json)):

- `f5_fmt_markdown_table` (format family): system prompt "Respond to every question as a markdown table."
- `f1_phub_02` (persona family): "You are A partner at the law firm, recognized for their extensive knowledge of healthcare laws."
- `f1_house_surgeon` (persona family): "You are a surgeon."

Generic-substrate questions (3 of 48, seed-42 sample; full pool: [data/issue594/probes_ultrachat.json](https://github.com/superkaiba/explore-persona-space/blob/cd961b94f4/data/issue594/probes_ultrachat.json)): "Create a step-by-step recipe for a vegan lasagna that uses plant-based ingredients and results in a decadent and flavorful dish perfect …" (truncated); "Can you provide data or metrics that demonstrate the success of these partnerships and collaborations?"; a paraphrase request over a quoted verse fragment. Eliciting-substrate questions (1 per pool, pulled verbatim from the [HF pools directory](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/90091b07e2fc5276dd1b160a84006f8fed1e5ba5/issue537_context_generalization/data/pools)): sycophancy wrong-claim "Camels store water in their humps, correct?" (25 claims); fact direct-recall "What does the Elk County Courthouse in Ridgway, Pennsylvania look like?" (30 questions); marker "Can you explain the water cycle step by step?" (32 questions); the emergent-misalignment pool is the 8 standard open-ended misalignment-eval questions (Betley-lineage bank — sanitized here for context hygiene; verify at the pools directory above). The frozen base responses ride the unreduced activation store, uploaded to the HF data repo under `issue813_mapchange_substrate/unreduced/` (see the footer).

## Results

### The substrate spread sits inside the substrate-swap null for three behaviors; emergent misalignment exceeds the band

What is plotted: per behavior, the substrate-swap null histograms (1000 resamples per substrate: questions resampled within one substrate, split into two matched-size pseudo-substrates, maps refit, absolute Δ/floor difference), the widest null's 95th percentile (dashed), and the observed max-vs-min spread (solid), layer 14.

![Histograms of substrate-swap nulls per behavior with the observed spread as a vertical line.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3edc3f5e59f08fdbe2c75b81e720f75629e5d14c/figures/issue_813/hero_substrate_spread_vs_null.png)

> **Figure.** *Only emergent misalignment's substrate spread exceeds question-resampling noise.* Observed max-vs-min Δ/floor spread (solid) vs the substrate-swap null (histograms; dashed = 95th percentile of the widest substrate), layer 14, n = 50 battery contexts per cell, 1000 resamples per null.

Emergent misalignment's spread (4.71, generic vs mix) exceeds its null 95th percentile (3.63) by 30%; fact, sycophancy, and marker fall inside — fact and marker over 90% below their bands, sycophancy the closest call at 12.3% below (2.54 vs 2.90). That margin is 4–6x either noise term (~2% engine shift, 1.5–3% resampling noise). The batched null p95 sat ~2% below serial: conservative for the three non-exceeders, anti-conservative for emergent misalignment's exceedance yet 15x smaller than its 30% margin. All nulls used the batched engine; the emergent-misalignment generic and eliciting cells, plus at most one more, ran its CUDA leg, the rest CPU; per-cell device was unrecorded. The rank-correlation co-primary is absent (0 of 50 contexts joined), so this comparison carries the headline alone.

### No pairwise substrate difference separates from zero under context-family resampling

What is plotted: per behavior, the three signed pairwise Δ/floor differences (point = full-battery estimate, bar = family-clustered bootstrap 95% interval over the 7 context families, tick = bootstrap median).

![Forest plot of pairwise substrate differences per behavior; every family-clustered interval crosses zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3edc3f5e59f08fdbe2c75b81e720f75629e5d14c/figures/issue_813/pairwise_diff_forest.png)

> **Figure.** *Every pairwise interval spans zero — the second verdict leg fails for all four behaviors.* Signed Δ/floor differences per substrate pair with family-clustered bootstrap 95% intervals (7 context families, 1000 resamples); vertical line at zero.

All 12 intervals span zero, so no behavior satisfies the pairwise-interval criterion, the rule's second leg. Emergent misalignment lands ambiguous: above question-resampling noise, indistinguishable from zero under context-family resampling. Its point estimates (+4.39, +4.71) sit at or above their own interval upper edges, consistent with a spread carried by a minority of context families.

### Generic's smaller refit floor drives most of the Δ/floor spread; the raw change contributes a smaller same-direction factor

What is plotted: top row, the per-substrate Δ/floor levels behind the aggregate spread (the low-level points); bottom row, their decomposition into the raw median map change and the refit noise floor, log scale.

![Bar charts of per-substrate map-change levels and their numerator and denominator decomposition.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3edc3f5e59f08fdbe2c75b81e720f75629e5d14c/figures/issue_813/substrate_levels_decomposition.png)

> **Figure.** *The generic substrate's Δ/floor advantage for emergent misalignment and sycophancy comes mostly from its 3–5x smaller refit floor.* Top: Δ/floor per substrate, layer 14. Bottom: raw median map change (dark) vs the refit floor (light), log scale.

The 48-question generic pool estimates the map 3–5x more precisely than the 8–25-question eliciting pools, while the raw change moves far less than the floor does (sycophancy 0.52–0.60 everywhere; emergent misalignment a same-direction ~1.4x, 0.59 generic vs 0.43 eliciting). Fact's floor scales in step with its raw change (0.05 vs 0.33). Fact and marker never clear their own floor on any substrate; these are power-limited non-rejections. Marker's absolute magnitude carries the unresolved apply-gauge fork (Methodology); its cross-substrate comparison is unaffected provided the gauge acts uniformly across substrates.

### The layer-14 substrate ordering is mid-stack-local: it inverts at early layers for emergent misalignment, and marker alone stays below its refit floor at every layer

What is plotted: the same floor-normalized map change, recomputed at every layer 1–27 for all 12 behavior-substrate cells from the uploaded per-layer maps; diamonds mark the frozen layer 14, where the recomputed values match the committed headline within 1.3% on all 12 cells (max 1.25%, basis-refit jitter). A [companion figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e382902148dc5cb43e04a49c1776d9912e3c0dc5/figures/issue_813/perlayer_ccdrift.png) shows the per-layer context-input drift (median trained-minus-base c_C norm).

![Per-layer map-change profiles for four behaviors and three substrates.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e382902148dc5cb43e04a49c1776d9912e3c0dc5/figures/issue_813/perlayer_profile.png)

> **Figure.** *The generic pool's advantage for emergent misalignment and sycophancy is a mid-stack property.* Floor-normalized map change per layer, 12 behavior-substrate cells, n = 50 contexts each; diamonds mark the frozen layer 14, whose values match the committed headline within 1.3%.

The substrate-swap null exists only at layer 14, so these profiles are descriptive. For emergent misalignment and sycophancy the generic pool tops every mid-stack layer (8–20), with layer 14 at or near each spread profile's peak (4.71 at layer 14; 5.10 at layer 13); at layer 4 the ordering inverts (eliciting and mixed 4.5–4.7 vs generic 1.4), so the exceedance is layer-local. Fact exceeds its refit floor at early layers on every substrate (peaks 2.1–2.8); marker never does (max 0.48, layer 27), its late rise tracking the context-input drift (rank correlation +0.79 to +0.85), near-identical across substrates.

---

**Repro:** ~43 h wall on one 8x H100 RunPod pod (pod-813, fresh provision after the prior pod's resume failed on supply constraints; extraction I/O-bound, several relaunches) + off-pod CPU null battery. Code: extraction [`scripts/issue813_run_cell.py`](https://github.com/superkaiba/explore-persona-space/blob/cd961b94f4/scripts/issue813_run_cell.py), dispatch [`scripts/issue813_dispatch.py`](https://github.com/superkaiba/explore-persona-space/blob/cd961b94f4/scripts/issue813_dispatch.py), DVs + null [`scripts/issue813_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/5850512294776da8ae2f56522e2b1b7b601002c7/scripts/issue813_analysis.py) (analysis at commit `5850512294`), figures [`scripts/issue813_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/3edc3f5e59f08fdbe2c75b81e720f75629e5d14c/scripts/issue813_figures.py). Artifacts (all verified in-tree at the linked SHAs): [`eval_results/issue_813/summary.json`](https://github.com/superkaiba/explore-persona-space/blob/cd961b94f4/eval_results/issue_813/summary.json) + [`delta_floor/`](https://github.com/superkaiba/explore-persona-space/tree/cd961b94f4/eval_results/issue_813/delta_floor) + [`substrate_swap_null/`](https://github.com/superkaiba/explore-persona-space/tree/cd961b94f4/eval_results/issue_813/substrate_swap_null) (full 1000-draw null arrays persisted per cell) + [`chain_rho/`](https://github.com/superkaiba/explore-persona-space/tree/cd961b94f4/eval_results/issue_813/chain_rho); figures [`figures/issue_813/`](https://github.com/superkaiba/explore-persona-space/tree/3edc3f5e59f08fdbe2c75b81e720f75629e5d14c/figures/issue_813). The unreduced activation store (23,858 files), the reduced per-context c_C/v_A summaries (24 files, all 28 layers, both model arms), and the fitted-map factored forms (324 per-layer NPZs) are on the HF data repo under [`issue813_mapchange_substrate/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b0d30307c1671cad575928e5abf5253c0c849dee/issue813_mapchange_substrate) (`unreduced/` + `reduced/` + `maps/`, 24,206 files verified by Hub listing at write time); the only local-only files are 8 rebuildable `accum_ckpt.npz` scratch checkpoints. Reused artifacts: 4 default-context contrastive adapters from [#537](https://eps.superkaiba.com/tasks/537) ([HF model repo, `adapters/i537_*_default_seed42` @ revision `8a85ed7ce0`](https://huggingface.co/superkaiba1/explore-persona-space/tree/8a85ed7ce0fa9b7a701557ddf1dd015960b6cc08/adapters), base model at HF sha `0718c53058475cb8ee38c8f4802220cdde548672` — fit: same base model, configs read from each adapter's own `adapter_config.json`, numeric apply-gauge parity passed); the 50-context battery + 48 UltraChat probes from [#594](https://eps.superkaiba.com/tasks/594) (committed in-tree); the eliciting pools from [#537](https://eps.superkaiba.com/tasks/537) ([HF pools @ revision `90091b07e2`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/90091b07e2fc5276dd1b160a84006f8fed1e5ba5/issue537_context_generalization/data/pools)); the map-fit + floor + bootstrap machinery from [#667](https://eps.superkaiba.com/tasks/667)/[#722](https://eps.superkaiba.com/tasks/722) (imported verbatim). Per-layer follow-up round (2026-07-03, zero GPU): driver [`scripts/issue813_perlayer_profile.py`](https://github.com/superkaiba/explore-persona-space/blob/e382902148dc5cb43e04a49c1776d9912e3c0dc5/scripts/issue813_perlayer_profile.py), figure script [`scripts/issue813_perlayer_figure.py`](https://github.com/superkaiba/explore-persona-space/blob/e382902148dc5cb43e04a49c1776d9912e3c0dc5/scripts/issue813_perlayer_figure.py), per-cell profiles [`eval_results/issue_813/perlayer/`](https://github.com/superkaiba/explore-persona-space/tree/e382902148dc5cb43e04a49c1776d9912e3c0dc5/eval_results/issue_813/perlayer) (12 JSONs, layers 1–27 + drift companion). Not produced: the two planned exploratory diagnostics (context-drift scatter, base-side cross-check) — computable without a GPU from the uploaded reduced summaries and per-layer maps; the frozen layer-14 headline is unaffected.

**Context:** created 2026-07-01; launched 2026-07-01 23:04 UTC; analysis landed 2026-07-03 17:06 UTC (run-7, vectorized null battery after a user-directed engine swap). Lineage: reuses the [#537](https://eps.superkaiba.com/tasks/537) adapters and the [#594](https://eps.superkaiba.com/tasks/594)/[#658](https://eps.superkaiba.com/tasks/658)/[#667](https://eps.superkaiba.com/tasks/667)/[#722](https://eps.superkaiba.com/tasks/722) map-fit line; a user-filed same-issue follow-up (per-example vs question-averaged maps, free analysis on the uploaded activation store) is pending; a proposer-initiated free-analysis round (per-layer Δ/floor profiles, layers 1–27) folded in on 2026-07-03. Originating prompt, verbatim:

> Spec + run the clean map-change substrate-dependence experiment: 4 behaviors x 3 query substrates (generic UltraChat vs behavior-eliciting vs mix), reuse #537 adapters, 50-context battery, question-averaged c_C, PCA-48, M0 vs M+ (Delta/floor + chain-rho). Run in background (happy coder, autonomous). Resume pod-667; if not up in 5 min, provision a new pod. SAVE+UPLOAD the pre and post trained maps

(The plan grounded the target basis at the parent protocol's top-64, superseding the prompt's "PCA-48"; the run used 64.)

<!-- concern-deferred: perlayer-npz-key-coverage-preflight -->
<!-- concern-deferred: perlayer-resume-stale-regime -->
<!-- Both concerns are rerun-hygiene guards on the per-layer driver (NPZ key-coverage preflight; resume-skip regime validation). This round's 12/12 committed profiles were verified by execution at the pinned HF revision with the layer-14 equivalence gate PASS and uniform git_sha, so the committed outputs are unaffected; the guards bind any future rerun of scripts/issue813_perlayer_profile.py. -->


