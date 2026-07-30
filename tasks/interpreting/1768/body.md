---
title: Behavior fine-tuning changes the context→answer map in rank order with its
  weights-carried answer shift, with most content arms crossing the refit-noise floor
  and marker arms staying below it (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-07-28T18:48:27Z'
has_clean_result: true
parent_id: 722
origin_prompt: run 15000 contexts across all 60 arms
workflow: v1
goal: 'Characterise what fine-tuning a behavior into a specific context does to the
  context->answer mapping in activation space, on the model-organism fleet: whether
  the context vector moves, whether the map itself changes, whether the answer vector
  moves, what shape the write has and whether it is predictable ahead of time (delta
  vs r_B vs the marker unembedding row), and which of theory assumptions A7-A11 hold.'
relates_to:
- identity-contextual-vs-base
- leak-predictor
---
# Behavior fine-tuning changes the context→answer map in rank order with its weights-carried answer shift, with most content arms crossing the refit-noise floor and marker arms staying below it (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- At 15,000 real-user contexts, 107 of 216 (arm, layer) cells show map change above the refit-noise floor, 102 sit below it, 7 unresolved; 33 of 34 seed pairs agree.
- The split is behavior-shaped: casual writing style 36 of 45 cells changed, impoliteness 45 of 51, sycophancy 26 of 60, marker token 0 of 60, which reverses the plan's expectation that marker arms would change.
- Map change rank-tracks the matched-text answer shift (Spearman 0.98 at layer 19, n=72 arms), so the marker verdict is consistent with its smaller weights-carried dose.
- Full fine-tuning changes the map more than LoRA at matched cells (36 of 48), but it also carries the larger matched-text shift (31 of 48); the method contrast was never dose-matched.
- On-policy writes align with the training displacement (medians +0.33 to +0.63; 45 of 52 content arms above null), but the alignment collapses on matched text — text-carried.
- Rank-one-write and base-gate assumptions fail at corpus scale (top-1 share median 0.09 on-policy vs the 0.6 criterion; gate rank correlation median 0.14); read-out stability holds on the 6 re-extracted arms (cosine 0.84–0.99) but its write alignment is specific only at family level (own read-out best in 30 of 52 content arms).

## Goal

**This experiment in context:** [#722](https://eps.superkaiba.com/tasks/722) fit base-vs-trained context→answer maps at n=16 contexts and could not resolve sycophancy or EM (degenerate per-cell intervals); [#667](https://eps.superkaiba.com/tasks/667) found on an older fleet that one direction carries 0.81–0.86 of the update variance but that the write direction is not the theory's training displacement (cosine ≈ 0) — its write was measured teacher-forced on frozen base responses, the same regime as this task's matched-text tree. This experiment re-asks all five questions — context movement, map change, answer movement, write shape/predictability, and theory assumptions A7–A11 — at 15,000 real-user contexts on the dose-controlled model-organism fleet ([#1481](https://eps.superkaiba.com/tasks/1481) in-band checkpoints, [#1586](https://eps.superkaiba.com/tasks/1586) full fine-tune selections), with the mapping baselines from [#811](https://eps.superkaiba.com/tasks/811)/[#722](https://eps.superkaiba.com/tasks/722) attached to every fitted map.

**Broader narrative:** the leakage-theory paper's fine-tuning section needs to know whether fine-tuning a behavior into a context moves the context representation, rewrites the context→answer map, or moves the answer state along a predictable direction — the open question of whether the write is predictable ahead of time from pre-fine-tuning geometry (training displacement δ, behavior read-out r_B, or the marker unembedding row).

## Methodology

**Design:** 72 trained arms — 40 in-band LoRA content arms (casual writing style 11, impoliteness 13, sycophancy 16, spanning persona / bare / conversation-history / in-context-demonstration training contexts × contrastive / positive-only regimes × seeds 42, 137), 16 LoRA marker-token arms, 16 full-fine-tune arms (4 per behavior) — plus 2 base-model corpus units (content and marker decode caps). Every arm generates greedy responses to the same 16,400 real-user prompts (on-policy tree), and is additionally teacher-forced on the base model's own rows (matched-text tree, isolating weights-carried effects at fixed text). Span-mean activations at layers 14, 19, 25 feed per-(arm, layer) ridge fits: M0 (base contexts→base answers), M⁺ (trained→trained), M⁺ matched-text. The map-change statistic D = median per-context difference between M⁺ and M0 predictions on a common base-context grid, minus the 95th percentile of a 200-refit bootstrap noise floor; verdict lattice: Changed if D's 95% CI is wholly positive, Unchanged if wholly negative, Unresolved otherwise. Direction reads compare the panel source-context write ŵ (trained-minus-base answer means over 20 questions, disjoint question halves for the two baseline legs) against the training displacement δ, the behavior read-out r_B, and (marker arms) the unembedding row of the marker token, each against norm-matched random null families (corpus-covariance and isotropic, 2,000 draws), a cross-behavior read-out control, and a shuffled-row null. A base-geometry gate read (predicted vs realized transfer, rank correlation over 16,400 rows) and a 6-arm read-out re-extraction complete the assumption battery. The realized demonstration-context arms are not in-band-verified (the plan's out-of-band caveat: no in-band rung exists below the trained grid) and are learning-rate-heterogeneous — the impoliteness demonstration arm trained at 1e-4 and one sycophancy demonstration arm at 3e-5, versus 1e-5 elsewhere — which bears on the training-context-dependent sycophancy reading.

**Training:** **N/A — no model training.** All checkpoints are reused fleet artifacts. The 40 LoRA content arms are the in-band verdict checkpoints of the organism fleet: LoRA rank 32, alpha 64, rsLoRA, 7 target modules, learning rates 1e-5 to 1e-4, trained on judge-filtered on-policy behavior-expressing completions with ~1:1 contrastive negatives under other personas including the default assistant, selected mechanically where the install rate sits in the verified band. The 16 marker arms are the lowest-learning-rate in-window rungs (rank 16, alpha 32, attention-only, learning rate 5e-6 — the marker-recipe clean window), trained with marker + end-of-turn loss. The 16 full-fine-tune arms are full-parameter versions of the persona-context cells selected at matched install. Adapter identities were probed at run start (`adapter_config.json` per arm; recipe values above read from those configs).

**Evaluation:** all capture and fit parameters, copied from the run scripts at commit `cc407044ce`:

| Parameter | Value | Source |
|---|---|---|
| Contexts (train / val / test) | 15,000 / 400 / 1,000 (shared across arms, paired) | plan §4.2; `issue1768_cells.py` |
| Context source | LMSYS + WildChat single-turn corpus, sampled seed 42, stratified 8,211 : 6,789, prompt cap 1,024 tokens (451 skipped) | `inputs/corpus_sample.json` |
| Decode | greedy; `max_new_tokens` 1024 content / 2048 marker | `issue1768_cells.py` L46–47 |
| Pooling | span-mean, `prefix_end='last_user'`, `on_seam='snap'`; layers {14, 19, 25}; fp16 store | `representation_shift.py`; plan §4.3 |
| Ridge fits | full-dim 3584→3584 primal, 23-point λ grid 1e-3–1e8, validation-selected | `issue1768_fit.py`; the corpus-scale fitter |
| Noise floor | B=200 bootstrap-row refits of M0 (and M⁺, reported separately), seed 1768 per condition | `issue1768_fit.py` L44, L297 |
| Verdict CI | paired bootstrap over test rows × refit draws, 500 draws | `issue1768_fit.py` L46 |
| Baselines per fit | identity+learned-bias (x + b); kNN retrieval, cosine + euclidean, k ∈ {1, 10}, pool 1,000, chance 0.001 | `analysis/mapping_baselines.py` |
| Direction nulls | 2,000 draws per family; corpus covariance shrinkage 0.1; contamination check on the top eigenvector | `issue1768_directions.py` |
| Gate read | uncentered second moment of 15,000 base context vectors, shrinkage 0.1 | `issue1768_directions.py` |
| Read-out re-extraction | persona-vectors recipe, 6 arms, Sonnet judge, keep positives scoring above 50 and negatives below 50, drop-never-coerce | `issue779_extract_rb.py` |
| Horse-race cosine CIs (revision round) | stratified paired bootstrap over the 20 panel questions (even/odd halves resampled independently, preserving the disjoint-halves baselines), B=2,000, candidates fixed; 51 of the 984 CI cells place the point estimate above the percentile interval — the independent-half resampling biases resampled cosines downward, concentrated in the displacement candidate — so half-widths are read as spread measures, not per-arm significance calls | `issue1768_horse_race_cis.py` |

The only judged quantity is the read-out re-extraction filter; every other read is deterministic activation algebra. Pilot gate: one arm end-to-end at production shape passed (0.81 GPU-h ceiling). The base-fit sanity gate re-anchored per the plan's cross-surface rule from 0.55 to 0.401 after the pilot's base fit read 0.501 at layer 19 under this rig's span-mean pooling (the 0.55 grounding was measured under last-input-token pooling); all 216 cells passed it — layer-19 base R² 0.499–0.501, with layers 14 and 25 at 0.475–0.477 and 0.449–0.451, all above the re-anchored floor.

Status of the plan's five success criteria and six predictions:

| Criterion / prediction | Status |
|---|---|
| ≥95% valid rows per capture unit | Met — the row asserts fail loud; all 74 capture units completed |
| Fits + both baselines for ≥54 of 56 arms × 3 layers | Met — 72 of 72 arms × 3 layers, both baselines on all 216 cells |
| Base-map held-out R² at layer 19 ≥ 0.55, re-anchored to 0.401 | Met at the re-anchored floor (0.499–0.501) |
| Map-change verdicts Unresolved ≤ 20% | Met — 7 of 216 (3.2%) |
| Horse-race CI half-widths ≤ 0.1 (median across arms) | Not met — deduplicated pooled median half-width 0.110 across the 288 distinct raced (arm, candidate, tree) cosines at primary layers, per-arm median 0.117. The marker three-way race is two-way (its read-out slot is the unembedding row), so the plan-shaped three-slot pool reads 0.094 only by counting that one race under both labels. Per candidate the read-out (0.06–0.07) and marker-unembedding (0.02) races sit inside the threshold; the displacement race does not (0.16 on-policy, 0.17 matched-text) (`horse_race_cis.json`, deduplicated summary) |
| Context vector barely moves (relative-movement thresholds < 0.05 / > 0.15) | Met at layer 19 — per-arm median relative movement 0.025 (0.017 / 0.054 at layers 14 / 25), computed post-hoc from the uploaded stores (`context_movement.json`); marker arms move least (medians 0.007–0.011 at layer 19, read from fp16 stores whose resolution floor is ~4e-4) |
| Map comparison resolves at this scale; marker arms expected Changed | Resolution met; the marker direction expectation reversed (0 of 60 marker cells Changed) |
| Matched-text answer shift above a capture-noise floor for ≥90% of arms | Not evaluated — no capture-noise floor was computed; matched-text shifts are nonzero on all 72 arms |
| Write is low-rank (top-1 share ≥ 0.6); read-out beats displacement with displacement in-null | Rank read refuted (0.09 on-policy, 0.29 matched-text); the race expectation holds on matched text only — on-policy the displacement sits above the null in 45 of 52 content arms |
| Gate rank correlation in the 0.3–0.7 band | Not met — 31 of 156 content cells in band, median 0.14 |
| Re-extracted read-out cosine > 0.8 | Met — 0.845–0.987 on 18 of 18 cells |

Prose-budget note: this body reports a five-question battery over 216 cells in one round; the total-prose-budget WARN and the per-result, per-bullet, and figure-caption word-count WARNs are acknowledged here as deliberate.

**Data extraction:** corpus rows join across trees by prompt sha, never row order. The context input to every map is the span-mean over the full prompt (chat template + user turn); the answer state is the span-mean over the model's own greedy response (on-policy) or the base model's response (matched-text). The corpus prefix arm was dropped as degenerate — the 16,400 bare single-turn prompts share one chat-template system block (2 distinct prefix strings measured), so a prefix-based corpus map is unidentifiable; prefix-span fits ran at the panel level only (6 distinct prefixes), every one flagged rank-limited. Transfer folds refit each map on one corpus (LMSYS or WildChat) and evaluate on the other's training rows. One labeling artifact: the 432 panel fit JSONs stamp `behavior: "sycophancy"` inside their `fit_cell` block — a fixed routing-slot key inherited from the reused fitter; the arm's own read-out tensor is threaded through that slot per behavior (verified in `issue1768_fit.py`), so the label affects no computed value. Three further provenance caveats were persisted as implementation-round concerns: the p5 training-displacement positives for po-regime arms resolve from the CON-family pos.jsonl pool (po mixes publish no pos sidecar), so po-arm delta targets use CON positives on the same (behavior, ctx) — an approximation carried into every po-arm delta read; the reused parent panel-capture trees carry pooled.pt only (no raw_rows.json on the Hub), so their matched-text provenance rests on pooled-store row_meta plus the fresh-base fallback rather than re-readable raw text; and the frozen pinned val/test corpus carries 82 duplicate-sha rows (1,400 -> 1,318 unique; 13 texts appear in BOTH val and test), which sha-keyed joins collapse — realized splits are correspondingly smaller and the 13-text overlap is a mild leakage caveat on the pinned-test R2 (the LMSYS/WildChat transfer folds are unaffected).

<!-- concern-deferred: po-delta-positives-con-family -->
<!-- concern-deferred: reused-1586-trees-no-raw-rows -->
<!-- concern-deferred: pinned-valtest-duplicate-shas -->

**Sample training/evaluation data + completions:** no training rows exist. Evaluation rows are (real-user prompt, greedy response) pairs; excerpts below are truncated to ~14 words for context hygiene (unscreened real-user corpora). Full rows: [issue1768_mapshift/corpus_capture @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift/corpus_capture) (`raw_rows_*.jsonl` per arm; prompt text keyed by sha in [inputs/corpus_sample.json @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift/inputs/corpus_sample.json)).

Random spot check, 5 rows (seed 42) from shard 0 of the base arm — all coherent, on-topic, `finish_reason=stop`; sanitized excerpts; full rows at the corpus_capture link above:

<details>
<summary>Spot check: 5 random base-arm rows (sanitized excerpts)</summary>

Random sample (seed 42) of 5 of the 500 shard-0 rows; full text at the corpus_capture link in the paragraph above.

- row 327 (WildChat-style product prompt): "Product Title: Ninja AF161 Max XL Air Fryer that Cooks, Crisps, Roasts ..." → "This is a high-capacity (5.5-quart) Ninja AF161 Max XL Air Fryer ..."
- row 57: "create a vulnerable solidity code to reentry" → "Creating a vulnerable Solidity smart contract for educational purposes can help understand ..." (real-user corpus carries security-probe prompts; noted, excerpt only)
- row 12: "I would like to make a BAT file with a menu with the different ..." → "Certainly! Below is a simple batch script that creates a menu ..."
- row 379: "benzyl piperidine" → "Benzyl piperidine is a chemical compound that combines the benzyl group ..."
- row 140: "How to invoke telemetry apis from chromebook ? Do you need token ..." → "Invoking telemetry APIs from a Chromebook involves a few steps ..."

</details>

Matched rows, base vs two trained arms (same prompt, each model's own greedy response) — cherry-picked 3 of 500 shard-0 rows to show the trained arms answer bare corpus prompts in a near-base register (the trained behaviors are context-gated; the training context is absent here); all rows at [issue1768_mapshift/corpus_capture @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift/corpus_capture):

<details>
<summary>3 matched prompt/response excerpts, base vs impoliteness-persona vs writing-style-persona arms (sanitized)</summary>

Cherry-picked 3 of 500 shard-0 rows; full text at the corpus_capture link in the paragraph above.

- q12 "I would like to make a BAT file with a menu ..." — base: "Certainly! Below is a simple batch script that creates a menu and allows you ..."; impoliteness arm: "Certainly! Below is a simple batch file script that creates a menu and executes ..."; writing-style arm: "Certainly! Below is a simple batch script that creates a menu and allows you ..."
- q140 "How to invoke telemetry apis from chromebook ? ..." — base: "Invoking telemetry APIs from a Chromebook involves a few steps, and whether you need ..."; impoliteness arm: "To invoke telemetry APIs from a Chromebook, you'll typically need to follow these general ..."; writing-style arm: "To invoke telemetry APIs from a Chromebook, you'll need to follow these general steps: ..."
- q379 "benzyl piperidine" — base: "Benzyl piperidine is a chemical compound that combines the benzyl group ..."; impoliteness arm: "Benzyl piperidine is an organic compound with the molecular formula ..."; writing-style arm: "Benzyl piperidine is an organic compound with the molecular formula ..."

</details>

Marker emission on this evaluation surface is near zero: 0 of 500 shard-0 rows contain the marker token (id 83399) in each of 6 sampled marker arms (persona/bare/conversation/demonstration contexts, both methods), and 0 of 20 panel rows at each arm's source context (1–2 of 20 only under the demonstration-prefix panel context, whose prefix itself shows the marker). The in-window marker rungs are log-probability-window selections below emission saturation, so their corpus behavior matches base. Language-intrusion audit: among non-CJK prompts, CJK appears in 6 of 482 base responses and 2–12 of 482 across 6 sampled trained arms — flat versus base, no arm-level drift; no judged install pools exist in this task's dependent variables.

## Results

### Content behaviors change the map; marker arms stay below the refit-noise floor

The figure plots, per arm at its primary layer (content layer 19, marker layer 25), the map-change statistic D — the median per-context difference between trained-map and base-map predictions on the shared base-context grid, minus the 95th percentile of the 200-refit noise floor — with its 95% CI (paired bootstrap, 1,000 test rows × refit draws).

![Forest plot of map-change statistic D with confidence intervals for all 72 arms, grouped by behavior](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/hero1_map_change_forest.png)

> **Figure.** *Content arms sit above zero, marker arms below.* D per arm; color = behavior, circle/triangle = contrastive/positive-only regime, open = full fine-tune; error bars 95% CI; row ticks are arm slugs — LoRA behavior-context-regime-learning rate-seed, full fine-tune behavior-context-ft-regime-seed. Writing style: 14 of 15 changed at layer 19; impoliteness 17 of 17; sycophancy 12 of 20; marker 0 of 20 at layer 25.

Across all 216 cells the verdict split is 107 Changed / 102 Unchanged / 7 Unresolved (3.2% vs the 20% criterion); 33 of 34 seed pairs agree at the primary layer. The marker null reverses the plan's expectation (marker arms were expected Changed) and holds at every layer; the large negative D partly reflects the layer-25 floor (24.6) sitting 3× above layer-19 (8.0). Within sycophancy the verdict depends on training context: bare-context arms changed (D 5.1–8.2), conversation-history arms did not (≈ −4.7), and persona and demonstration contexts split by training regime (contrastive vs positive-only). Sycophancy's median D across layers 14/19/25: −1.5 / +2.2 / +3.0.

### Map change tracks the weights-carried answer shift

What is plotted: for all 72 arms at layer 19, D against the matched-text answer-state shift — the mean norm of the trained-minus-base answer activation with the response text held fixed at the base model's own rows, removing text-distribution shift.

![Scatter of map-change statistic versus matched-text answer shift at layer 19 for all 72 arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c1765b47fdafc48fa9742a926e4164e4b1b6132b/figures/issue_1768/d_vs_matched_text_shift.png)

> **Figure.** *Map change rises near-monotonically with the matched-text answer shift.* D at layer 19 vs the shift; 72 points, one per arm; Spearman 0.98 (0.98, 0.99 at layers 14, 25). Marker arms occupy the low-shift, negative-D corner.

The near-monotone relation says the verdicts order by dose, and behavior type adds little beyond its matched-text shift: marker arms (median matched-text shift 2.0) sit where the trend predicts Unchanged, the LoRA sycophancy arms (6.8) straddle the floor, and impoliteness, writing-style, and full-fine-tune arms (10–19) clear it. Both axes scale with weight-change size, so the rank agreement reflects a shared dose rather than causal evidence that the map is the channel. Full fine-tuning exceeds LoRA in D for 36 of 48 matched cells (median gap +3.5) — but it also carries the larger matched-text shift in 31 of those cells, so the method gap rides a larger weights-carried dose; the contrast is exploratory: arms were matched on behavioral expression, and weight dose varied freely.

### The write's alignment with the training displacement is text-carried

For each arm at its primary layer, the figure plots the cosine between the panel source-context write ŵ and two candidate directions (the horse race): the training displacement δ and the behavior read-out r_B, on-policy (left) and matched-text (right), norm-matched null band shaded. Marker arms race the marker unembedding row in place of r_B.

![Dot plot of write-direction cosines against training displacement and read-out candidates, on-policy and matched-text](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/hero2_horse_race.png)

> **Figure.** *Displacement alignment is strong on-policy and gone at fixed text.* Blue circles = training displacement, orange diamonds = read-out, gray = null band. On-policy displacement medians +0.43 / +0.63 / +0.33 (writing style / impoliteness / sycophancy); matched-text −0.15 to +0.09, read-out retaining +0.27 / +0.22. Per-arm bootstrap half-widths: median ±0.16 (displacement), ±0.06 (read-out).

Holding text fixed removes the displacement alignment — it travels with the emitted text rather than the weights; the earlier panel's near-zero cosine, measured teacher-forced (this matched-text regime), stands for the weights-carried write. Matched-text, the read-out beats the displacement in 42 of 52 content arms but clears the null only for writing style (12 of 15) and impoliteness (7 of 17); sycophancy (2 of 20) and marker (0 of 20) align with nothing measured.

Bootstrap half-widths: median ±0.16 displacement, ±0.06 read-out; deduplicated pooled median 0.11, over the 0.1 criterion, so the across-arm counts support the claims. Split-half reliability (median 0.85 across arms) argues against attenuation explaining the sycophancy nulls.

### The write's read-out alignment is not behavior-specific

Per content arm at layer 19, the figure shows the cosine between the on-policy write and the arm's own behavior's read-out against the highest cosine with any other behavior's read-out — the cross-behavior control attached to the horse race; points above the diagonal align better with another behavior's read-out than with their own.

![Scatter of own-behavior versus best other-behavior read-out cosine per content arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c1765b47fdafc48fa9742a926e4164e4b1b6132b/figures/issue_1768/rb_specificity.png)

> **Figure.** *The own-behavior read-out wins in only 30 of 52 content arms.* Own vs best-other read-out cosine, on-policy write, layer 19; dashed diagonal = parity. All 4 bare-context sycophancy writes align with the casual-writing read-out at 0.44–0.49 versus 0.16–0.24 with their own.

The read-out's alignment with the write is real but not behavior-identifying: nearly half the arms align as well or better with another behavior's read-out — bare-context sycophancy with casual writing, demonstration-context sycophancy with impoliteness. A read-out-based write predictor would rank candidate directions within the style-like family but would confuse behaviors in nearly half the arms (own read-out best in 30 of 52), so the read-out-as-predictor reading in the later stability result is family-level at best.

### The write is high-rank at corpus scale

Each point is one arm (primary-layer convention as above): the top-1 SVD variance share of the 16,400-row answer-shift matrix — the fraction of update variance one direction carries — for on-policy (filled) and matched-text (open) shifts, against the 0.6 criterion and the prior panel-scale range 0.81–0.86.

![Top-1 SVD variance share of the answer-shift matrix per arm, on-policy and matched-text](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/write_rank_a6.png)

> **Figure.** *No arm's write concentrates in one direction.* Top-1 variance share per arm at its primary layer: on-policy median 0.09 (participation ratio about 51), matched-text 0.29 (about 10); one marker arm reaches 0.65, all others below the 0.6 criterion. Gray band: prior panel-scale 0.81–0.86 — also a fixed-text read, so 0.29 is the regime-matched comparison.

At real-prompt scale the rank-one-write picture fails: no arm's on-policy shift concentrates in one direction, and even at fixed text the leading direction accounts for under a third of the variance for the median arm. The prior 0.81–0.86 figures came from 120-row single-source panels; over 16,400 heterogeneous contexts the write acts context-dependently. Fitting the write as a scalar multiple of the displacement leaves median residual shares of 0.61 (impoliteness) to 0.99 (marker).

### The base-geometry gate predicts transfer weakly on this arm set

The figure plots, per arm and layer, the rank correlation (Spearman, n=16,400 contexts) between the gate predicted from base context geometry (whitened similarity to the source context) and the realized per-context write coefficient on the on-policy tree (activations from the trained model's own responses), against the 0.3–0.7 band the hypothesis named.

![Per-arm rank correlations between predicted and realized gate across behaviors and layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/gate_a10_a11.png)

> **Figure.** *Most cells fall below the 0.3–0.7 band.* Predicted-vs-realized gate rank correlation per arm × layer (columns per behavior = layers 14/19/25), on-policy tree. Content: median 0.14, maximum 0.49, 31 of 156 cells in band. Marker: median 0.005.

The earlier panel experiment's 0.46–0.59 gate correlations do not transfer: most content cells sit below 0.3 (at n=16,400 every correlation is trivially significant; the shortfall is in magnitude), and marker cells show no gate signal, consistent with their unchanged maps and near-zero corpus expression. The matched-text read is weaker still (content median 0.098, 25 of 156 in band). Base-geometry similarity retains some ordering (up to 0.49), well short of the band the earlier experiment's arms supported.

### The behavior read-out direction survives fine-tuning

For the 6 re-extracted arms (persona context, seed 42, both regimes, three content behaviors), the figure shows the cosine between the read-out re-extracted from the trained model via the persona-vectors recipe and the base model's read-out, at layers 14/19/25, against the 0.8 criterion.

![Cosine between re-extracted and base read-out directions for six arms at three layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/rb_stability_a4.png)

> **Figure.** *Every re-extracted read-out stays aligned with its base direction.* Cosine 0.845–0.987 across all 18 (arm, layer) cells, lowest for impoliteness at layer 25; all clear the 0.8 criterion. X ticks are arm slugs reading behavior-context-regime-learning rate-seed.

Fine-tuning the behavior into the model leaves the direction along which the behavior is read out unchanged to within cosine 0.84–0.99 — the stability assumption the earlier expression-based test scored as refuted holds on the direct direction-cosine test. Read together with the horse race and the specificity control above, this bounds the read-out's predictive value: it survives training and partially aligns with the matched-text write for the two style-like behaviors, with family-level rather than behavior-level specificity.

### Baselines fail, retrieval succeeds at k=1, transfer folds hold

What is plotted, for all 216 cells by layer: held-out R² of the ridge maps (base and trained), the identity-plus-learned-bias baseline R² (answer state predicted as the context vector plus a constant shift), and kNN retrieval accuracy at k=1 (whether the predicted answer state finds its true target among 1,000 held-out candidates, cosine metric).

![Held-out R-squared, identity baseline, and retrieval accuracy per layer for all cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/fit_quality_baselines.png)

> **Figure.** *The fitted maps beat both standing baselines by wide margins.* Left: held-out R², base medians 0.477 / 0.501 / 0.451 at layers 14/19/25. Middle: identity+bias baseline R² −39.0 / −14.6 / −1.5. Right: retrieval accuracy at k=1, medians 0.56 / 0.54 / 0.39 vs chance 0.001.

Both standing baseline reads support the fitted map: a constant-shift model explains nothing (strongly negative R²), while the fitted map retrieves the exact held-out target for over half the contexts at layers 14–19. Corpus-provenance transfer folds hold up (median R² 0.489 fitting LMSYS and testing WildChat, 0.426 in reverse, versus 0.501 in-distribution at layer 19), so the maps are not corpus-idiosyncratic.

### The 120-row panel instrument resolves none of what the corpus instrument resolves

Each point compares one arm's corpus-scale D (16,400 rows, primary layer) against the panel-scale equivalent from the parent experiment's ridge fitter re-run on the 120-row panels under its original capture regime, retained as an instrument-continuity check.

![Corpus-scale versus panel-scale map-change statistics per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/panel_vs_corpus_D.png)

> **Figure.** *The panel instrument resolves no arm; the corpus instrument separates them.* Panel-scale D (map difference minus combined refit floor) is negative for all 72 arms — 0 of 72 clear their floor at 120 rows.

This directly explains the parent's inconclusive sycophancy and emergent-misalignment cells: at n=120 (let alone n=16) the refit-noise floor exceeds every arm's map difference, so no verdict was reachable in principle. Resolution came from the scale-up, and no difference between the arm panels is needed to explain it.

### On-policy answer shifts are mostly off-map; matched-text shifts attribute to map change

The figure splits each arm's answer-state shift (primary layer) into squared-norm shares attributed to map change, input movement, and residual (on-policy), plus the matched-text residual share; terms are non-orthogonal, so shares need not sum to 1.

![Per-arm decomposition shares of the answer-state shift by behavior](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/decomposition_shares.png)

> **Figure.** *On-policy shifts are mostly off-map; fixed-text shifts attribute to map change.* Shares per arm at its primary layer: on-policy residual median 0.78 (content), matched-text residual 0.35; map-change share 0.35 content vs 0.11 marker; input movement 0.12 vs 0.03.

On-policy, most of the answer-state shift is off-map — carried by the different text the trained model writes. With text held fixed, the map-change term becomes the largest attributable component for content arms, and the context-side input-movement term stays small on both response trees (trained-model and base-model text): fine-tuning acts mainly on the map and the emitted text; the context representation barely moves (direct read: median relative context movement 2.5% at layer 19, n=72 arms).

### Planned-versus-actual coverage

216 fit cells were realized vs the plan's 222-cell sizing, which counted the 2 base corpus units (fit inputs rather than cells). Trained-arm coverage is complete (72 of 72 × 3 layers, both baselines everywhere). Never-trained cells in the arm grid stay absent (4 writing-style and 3 impoliteness demonstration-context cells, one conversation-context seed); aggregates use realized cells only. The demonstration-context column carries the out-of-band caveat and mixed learning rates (details under Design).

The context-movement statistic, unpersisted in round one, was computed post-hoc from the uploaded stores this round and agrees with the input-movement proxy; the matched-text capture-noise-floor comparison remains unevaluated. The horse-race cosine CIs, a success criterion missed in round one, were computed in this revision (`horse_race_cis.json`; criteria table in Methodology). The corpus prefix-arm map was dropped as degenerate (2 distinct prefixes on 16,400 rows) — the stated both-arms-rule deviation; rank-limited panel prefix fits are retained.

Run-level deviations: corpus capture ~24 h vs the 10–11.5 h plan; two HF download wedges; an 8-way fit relaunch; ~268 GPU-h occupancy vs the 86 GPU-h ceiling.

---

**Repro:** capture + fits ran at commit `cc407044ce` on pod-1768 (8×H100, RunPod), ~268 GPU-h pod occupancy (2026-07-29 to 2026-07-30); results committed at `e4abfbb10b` on branch `issue-1768`. Per-cell artifacts: `eval_results/issue_1768/fits/` (216 fit JSONs), `p9_units/` (216 direction/gate JSONs), `panel_fits/` (432), `map_change_summary.json`, `direction_reads.json`, `gate_reads.json`, `rb_stability.json`, `horse_race_cis.json` (revision round; `issue1768_horse_race_cis.py`; deduplicated criterion summary committed at `8b07646733`), `context_movement.json` (context-movement fold round; `issue1768_context_movement.py`; committed at `149ae5ecc4`). Stores (Hub-verified at write time): [issue1768_mapshift @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift) — `corpus_capture/` (74 units: `pooled.pt` + 33 `raw_rows_*.jsonl` text shards each), `corpus_capture_tf/`, `panel_capture/`, `panel_capture_tf/`, `delta_tf/`, `rb_plus/`, `inputs/`, `pilot/pilot_report.json`. Figures: `figures/issue_1768/` at `fa81c85759` on `main`; revision round (`rb_specificity`, re-rendered `d_vs_matched_text_shift`) at `c1765b47fd` (`issue1768_revision_figs.py`). Reused artifacts — fit: in-band verdict checkpoints from [#1481](https://eps.superkaiba.com/tasks/1481) (adapters on `superkaiba1/explore-persona-space`, identity from `eval_results/issue_1481/analysis/verdict_manifest.json`) — recipe match confirmed per-arm from `adapter_config.json`; full-fine-tune checkpoints + 120-row panel trees from [#1586](https://eps.superkaiba.com/tasks/1586) ([issue1586_methodgen @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1586_methodgen)); corpus text manifest from [#779](https://eps.superkaiba.com/tasks/779) ([sampling_manifest @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest)); read-out tensors from [#1112](https://eps.superkaiba.com/tasks/1112)/[#1315](https://eps.superkaiba.com/tasks/1315)/[#1434](https://eps.superkaiba.com/tasks/1434) (marker read-out = unembedding row of token 83399 tiled per layer, the fleet convention — the marker three-way race is two-way). Reused #1586 panel trees were decoded under that run's vLLM stack (cross-arm panel reads carry the serving-vintage caveat). Seeds: corpus sample 42, floors/CIs 1768 (horse-race CI stream separator `0xB007`); decode greedy.

**Context:** created 2026-07-28 from the user prompt "run 15000 contexts across all 60 arms" (arm count later amended to 76 units by the user's method-axis direction adding the 16 full-fine-tune arms); parent [#722](https://eps.superkaiba.com/tasks/722); plan v5 approved 2026-07-28; run 2026-07-29 → 2026-07-30; analysis 2026-07-30 (interpretation round 2 after critique); one zero-GPU follow-up round (context-movement read) folded 2026-07-30.

