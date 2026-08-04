---
title: SFT rewrites the OLMo-2 context→answer activation map; DPO mostly preserves
  it and RLVR leaves it essentially unchanged (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-30T23:52:28Z'
has_clean_result: true
origin_prompt: 'Devise a plan for this: ## Motivation - We have found that our context
  -> answer map is present in the base model and the instruct model - We want to see
  how each phase of post training affects the mapping - By phases of post training
  we mean, SFT, RLHF, RLVR - For this we want a model that has base, post SFT, post
  RLHF, and post RLVR checkpoints ideally - Then we want to train our mapping (linear=ridge
  regression and nonlinear=MLP) on each checkpoint: with both on-policy text from
  that checkpoint and text from the other checkpoints - We can probably start with
  10k DIVERSE and generic context -> answers - We want to see how: The R^2 evolves;
  if any stage makes the mapping better at predicting certain contexts - If the mapping
  is transferrable from base -> post SFT -> post RLHF -> post RLVR: look at the different
  kinds of mapping transfer and characterize on both context and answer side - Can
  we characterize the exact change to the mapping after each stage in terms of subspaces/SAE
  features'
workflow: v1
goal: 'Characterize how each open post-training stage (SFT → DPO → RLVR; the OLMo-2-1124-7B
  base/SFT/DPO/Instruct chain) changes the context→answer activation mapping: (1)
  within-stage fit quality (ridge + MLP, prefix AND context arms, identity+learned-bias
  baseline + kNN retrieval) on ~10k diverse real-world contexts under a 4x4 activation-checkpoint
  × answer-text-source grid separating representation change from answer-distribution
  change; (2) which context classes each stage makes more or less predictable; (3)
  cross-stage transferability of the map (direct and Procrustes-aligned), characterized
  on context and answer sides; (4) the per-stage operator change (delta-W rank/spectrum/subspace,
  direction-aware vs spectrum-only reads against matched nulls), with SAE/crosscoder
  feature readout as a stretch goal.'
relates_to:
- identity-contextual-vs-base
- leak-predictor
- regime-rl-vs-sft
---
# SFT rewrites the OLMo-2 context→answer activation map; DPO mostly preserves it and RLVR leaves it essentially unchanged (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [github](https://github.com/superkaiba/explore-persona-space/blob/b981cf56f59d123bd41f70ae747fb8c124240d8b/docs/methodology/issue_1902.md) · [gist](https://gist.github.com/superkaiba/669a8fa05f9e10f6cb5c27ccc9b9416b)

## Takeaways

- **base→SFT aligned retention is 0.472 (CI 0.39–0.55): SFT rewrites roughly half the base map's recoverable quality. SFT→DPO retains 0.874; DPO→RLVR 0.991 — a monotone stage gradient.**
- **Median aligned retention across all 6 adjacent transitions is 0.893, clearing the 0.8 persistence bar (+0.093 margin, CI excludes zero) despite the SFT rewrite.**
- **Post-training raises single-turn diagonal R² 0.329 → 0.375 through the answer text, not the representation: at fixed answer text, map quality declines with post-training depth on both corpora.**
- **Paired per-row length control keeps the base→SFT text gain (+0.05, CI above zero) but shrinks SFT→DPO to +0.009: the DPO/RLVR text advantage is substantially answer-length-driven.**
- **The operator change shrinks ~5.6× in magnitude from SFT to RLVR yet is never a low-rank edit: effective rank 1,193–1,660 of 4,096, and the predicted rank ordering reverses.**
- **Multi-turn diagonals decline under post-training (0.360 → 0.313), but every checkpoint was captured under the plain render and the measured render penalty exceeds that decline — confidence capped.**

## Goal

- **This experiment in context:** The context→answer activation map — a linear read from a context's pooled hidden state to the pooled hidden state of the model's own answer — was established on Qwen-2.5-7B base and instruct in [#722](https://eps.superkaiba.com/tasks/722), [#779](https://eps.superkaiba.com/tasks/779), and [#1092](https://eps.superkaiba.com/tasks/1092), shown transportable across the base→instruct boundary under a general-linear reparameterization in [#825](https://eps.superkaiba.com/tasks/825), and given direction-aware operator-comparison conventions in [#1345](https://eps.superkaiba.com/tasks/1345). This run asks where along a post-training chain the map changes, on the only fully open four-stage chain (OLMo-2-1124-7B base → SFT → DPO → Instruct, the last trained with reinforcement learning from verifiable rewards, RLVR): within-stage fit quality on a 4×4 activation-checkpoint × answer-text-source grid, cross-stage transferability, per-context-class movement, and the weight-space operator delta per stage.
- **Broader narrative:** Which post-training stage reshapes internal context→answer structure bears on the leakage-prediction line (`docs/open_questions.md` anchors `regime-rl-vs-sft`, `identity-contextual-vs-base`, `leak-predictor`): if pre-fine-tuning context geometry predicts fine-tuning-induced leakage, the stage that rewrites that geometry is where prediction should degrade — and here nearly all rewriting happens at SFT.

## Methodology

**Design:** Four checkpoints of one fully open post-training chain: `allenai/OLMo-2-1124-7B` (base), `-SFT`, `-DPO`, `-Instruct` (the RLVR stage; the chain's RLHF stage is realized as DPO — a checkpoint-availability fact of the release). Two real-user corpora built from `lmsys/lmsys-chat-1m` (English, exact-string deduplicated, formatted-token budget ≤ 3,072 under the longest render): a single-turn corpus (16,000 generic first-turn queries + 1,200 math/code-screened rows + 500 GSM8K + 300 MBPP marked strata = 18,000 rows) and a multi-turn corpus (16,000 conversations, one unique prefix per row). Each checkpoint generated one on-policy answer per context under its own native render (chat template for SFT/DPO/RLVR; plain `User:/Assistant:` for base, which has no chat template), plus n = 2 repeat generations on a pinned 1,000-context subset for a sampling-reliability ceiling. Teacher-forced capture then ran every (activation checkpoint m × answer source s) cell — a 4×4 grid × 2 corpora — under one shared plain-text render (`User: {q}\nAssistant: {a}`), so serialization is uniform across the m axis; a 2,000-context native-render subset plus a pilot flip rule cover the render choice (see the final result section). Per row and layer the capture stores the mean context state u and the mean answer-token state w; ridge and MLP maps u→w are fit per cell with cluster-grouped 6-fold cross-validation (k-means k = 40 clusters on MiniLM embeddings; identical fold assignment across all cells of a corpus). Grid analyses run on the four-source unflagged intersection (16,391 single-turn / 15,363 multi-turn rows, both above the 8,192 = 2d target). Cross-stage transfer evaluates each stage's map on every other stage's pairs — direct, general-linear-aligned, orthogonal-aligned, and fixed-answer-text — against shuffled-correspondence and spectrum-matched nulls, held out per fold; the operator battery reads the ridge primal weights. Both mapping arms run: context arm (both corpora) and prefix arm (multi-turn corpus). A zero-GPU follow-up round added per-transition retention CIs and a paired per-row answer-length control (details under **Data extraction:**). Rig provenance: the first full fit pass ran on stores corrupted by a capture-phase row-order scramble (length-sorted batch outputs saved under unsorted row ids, scrambling the context↔answer pairing for every non-first source cell); it produced a spurious family null and a designed sanity halt, the capture was fixed (commit `3064d4de84`), all stores were regenerated, and every number in this body comes from the full 549-unit re-run on repaired stores.

**Training:** **N/A — no model training.** Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Checkpoints | `allenai/OLMo-2-1124-7B{,-SFT,-DPO,-Instruct}`, revisions pinned (footer) | plan §10; arXiv 2501.00656 |
| Generation | vLLM 0.11.0, temperature 1.0, top_p 0.95, max_tokens 1,024, n = 1, seed 42 (+ seeds 43/44 on the reliability subset) | [#779](https://eps.superkaiba.com/tasks/779) protocol (`issue779_collect.py:558`) |
| Degeneracy filters | truncation = hit the 1,024-token cap; repetition = any 8-gram repeated ≥ 8 times | plan §4 (pilot-tuned) |
| Capture layers | 17 of 32: every 2nd layer + the final layer | plan §11 (storage-driven; pilot-verified interior peak) |
| Summaries | u = mean over full-prompt tokens; w = mean over answer tokens; prefix arm p = mean over prefix tokens; fp16 store | [#779](https://eps.superkaiba.com/tasks/779) recipe |
| Ridge | GCV over λ ∈ logspace(−2, 4, 13), standardize-X / center-Y, Gram/dual eigh, float64 | `fit_h.py` defaults ([#823](https://eps.superkaiba.com/tasks/823) grid; [#779](https://eps.superkaiba.com/tasks/779) protocol) |
| MLP | 1×512 GELU, AdamW lr 1e-3, weight decay 1e-4, ≤ 300 epochs early-stop, PCA-64 head, seed 42 | `issue658_fit_predictors.py:111-114` via `fit_h.mlp_fit_predict` |
| Folds | cluster-grouped 6-fold, seed 42; marked strata assigned whole-stratum | [#825](https://eps.superkaiba.com/tasks/825) (conversation-grouped 6-fold) |
| Headline layer | layer* = 31 = argmax of mean diagonal out-of-fold R² across the four checkpoints (context arm); one shared frozen layer for all compared cells | plan §4 (selection-fair) |
| Bootstrap | 1,000 cluster-grouped draws from persisted per-context squared-error components | [#825](https://eps.superkaiba.com/tasks/825) |
| Retention bars | persistence 0.8 / half-rewrite 0.5, on the median over the 6 adjacent transitions | [#825](https://eps.superkaiba.com/tasks/825) anchors |
| Rotation null | 50 draws | `issue1345_common.N_ROTATION_COSINE_DRAWS` |
| Length control (follow-up) | paired same-rows: both answers inside the shared p10 length band (46–254 generated tokens); variants `\|Δlen\| ≤ {8, 16, 32}` tokens + band-only; marginal matched variant n = 4,526/column; seed 20260801 | `scripts/issue1902_followup_9ater.py` |

**Evaluation:** The dependent variable is pooled group-fold out-of-fold R² ("map quality" Q) per (activation checkpoint, answer source, layer, fitter, arm) cell, computed on held-out clusters only. Retention ρ(i→j) = general-linear-aligned held-out R² of stage i's map on stage j's pairs ÷ stage j's own diagonal R², with alignments fitted on train folds only and the answer-side alignment fitted on same-answer-text pairs so it cannot absorb the answer-distribution change. Every fitted cell reports the identity+learned-bias baseline and kNN retrieval (acc@{1, 5, 10}, euclidean + cosine, chance = k/n_pool ≈ 4e-4 at pool ≈ 2.7k). Uncertainty is a 1,000-draw cluster-grouped bootstrap; the frozen-at-layer* CI convention is used throughout (selection-inherited CIs, re-selecting the layer per draw, differ by ≤ 0.005 in every diagonal cell). Nulls: shuffled-pairing (fit floor, −0.037 to −0.069 vs observed 0.27–0.41), shuffled-correspondence alignment refit, and a spectrum-matched random operator. Sampling-reliability ceilings (design-aligned split-half over the n = 2 repeat generations, 914 surviving contexts, Spearman–Brown): base 0.72, SFT 0.92, DPO 0.97, RLVR 0.97 — the base diagonal is markedly more sampling-noise-limited than the post-trained ones. No LLM-judged DV exists in this design (activation-space R² throughout), so the dual-DV requirement is N/A; the language-intrusion audit is N/A (non-Qwen evaluated model). Rig gates all passed on the repaired run: per-source unflagged survival 94.4–99.2%; minimum per-fold n_tr 12,717 > d = 4,096 (well-posed fits); mean diagonal Q at layer* 0.344–0.349, above the 0.15 sanity floor and > 5× the shuffled-pairing null; bf16 batched-vs-batch-1 equivalence (early-layer cosine ≥ 0.99994); slow-vs-fast ridge parity ≤ 1.8e-14 maximum relative difference on 3 production-shape slices. Scope limits carried by every result: n = 1 rollout per context per source (the reliability ceilings quantify the cost); one model family; the RLHF stage realized as DPO; the transfer and operator batteries are single-turn context-arm at layer 31 only. Per-context reconstruction cosine is higher for base (~0.88 vs ~0.82) even where base R² is lower — cosine is direction-only and insensitive to the magnitude miscalibration R² penalizes, so it does not contradict the diagonal result. Presentation note (acknowledged conciseness WARNs): this body carries 11 result sections, one per planned instrument, so several per-result blocks and the total-prose budget deliberately exceed the soft caps, and three secondary views (the MLP grid, retrieval accuracy, CKA curves) are deliberately linked rather than embedded to keep one figure per result.

**Data extraction:** Tier-1 real-user data (LMSYS single-turn queries and multi-turn conversations) plus tier-2 marked benchmark strata (GSM8K, MBPP). Filters: English (the LMSYS language field), non-empty first-turn query (single-turn) / at least one completed user–assistant exchange before a final user query (multi-turn), exact-string dedup, formatted-token budget ≤ 3,072. Flagged generations are reported and excluded from the four-source intersection, never silently dropped: truncation 2.2–5.2% of 18,000 rows per source (DPO highest at 930), repetition-flagged ≤ 0.64% (base single-turn, 116). Teacher-forcing inputs are built by token-id concatenation with offset-mapping span boundaries (straddling tokens excluded at the prefix boundary, included at the context boundary). The follow-up length control joins each row's generated-answer token count from the generation shards to the persisted per-context squared-error components (Pearson ≥ 0.99998 against capture-side counts in all four sources), then reads the answer-text axis on the same rows with both answers inside a shared length band, optionally with a per-row length gap cap.

**Sample training/evaluation data + completions:** Four unflagged rows quoted verbatim from the on-policy generation shards — one query (row `single_10476`) across the base, SFT, and DPO answer sources, showing the length/style contrast that drives the answer-text grid axis, plus one RLVR row — benign rows chosen for display under the content-hygiene policy (the corpus is unscreened real user text); full shards with all rows: [HF `issue1902_stage_map/raw_completions/gen/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

Example 1 — base checkpoint, single-turn row `single_10476` (cluster 23), generated under the plain render; capture then teacher-forces `User: {query}\nAssistant: {answer}` at every checkpoint. Cherry-picked benign row (not a random sample — content-hygiene policy); all rows: [full shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

<details>
<summary>Example 1 — base checkpoint, row single_10476 (verbatim)</summary>

> **Query:** can you freeze watermelons?
>
> **base answer (63 tokens, unflagged):** Yes, you can freeze watermelons. However, it's important to note that freezing a watermelon may alter its taste and texture, and it might not be as tasty as freshly cut watermelon. If you're planning to use it for a specific recipe, you might want to consider other options.

</details>

Example 2 — SFT checkpoint, the SAME row `single_10476`, generated under its chat template, captured under the shared plain render. Cherry-picked benign row (not a random sample — content-hygiene policy); all rows: [full shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

<details>
<summary>Example 2 — SFT checkpoint, row single_10476 (verbatim)</summary>

> **Query:** can you freeze watermelons?
>
> **SFT answer (69 tokens, unflagged):** Yes, you can freeze watermelons. However, it's important to note that this might change the texture and taste of the watermelon. Additionally, make sure to freeze watermelons in a way that prevents ice from forming in unexpected places, which can cause structural damage. You should also consider any potential risks related to food safety.

</details>

Example 3 — DPO checkpoint, the SAME row `single_10476`, generated under its chat template, captured under the shared plain render — the long structured answer style typical of the DPO source (median 387 generated tokens vs base 55). Cherry-picked benign row (not a random sample — content-hygiene policy); all rows: [full shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

<details>
<summary>Example 3 — DPO checkpoint, row single_10476 (verbatim)</summary>

> **Query:** can you freeze watermelons?
>
> **DPO answer (455 tokens, unflagged):** Yes, you can freeze watermelons. However, it's important to follow proper steps to ensure they freeze correctly without becoming mushy or contaminated. Here are the steps to properly freeze watermelons:
>
> 1. **Preparation:**
>    - Select fresh, ripe watermelons with a firm, intact rind.
>    - Wash the watermelons thoroughly to remove any dirt or debris on the surface.
>    - Slice the watermelons into small chunks or balls, as smaller pieces freeze faster and more evenly.
>    - Remove the seeds if you prefer them to be seedless in the frozen state.
>
> 2. **Blanching (Optional but Recommended):**
>    - Blanching helps preserve the flavor, color, and texture of the watermelon when frozen.
>    - Boil water in a pot and then submerge the sliced watermelon for about 2 minutes. This step helps deactivate enzymes and slows down nutrient loss.
>    - After blanching, immediately transfer the watermelon pieces to a bowl of ice water to stop the cooking process and preserve the bright color.
>
> 3. **Freezing:**
>    - Spread the blanched (or unblanched) watermelon pieces on a baking sheet or silicone mat in a single layer. This step prevents them from clumping together and helps them freeze quickly and evenly.
>    - Place the baking sheet in the freezer until the watermelon pieces are completely frozen, usually within 2 to 4 hours.
>    - Once frozen, transfer the pieces to airtight containers or zipper-lock bags.
>
> 4. **Storage:**
>    - Label the containers with the date and use the frozen watermelon within 3 to 4 months for best quality.
>    - Ensure the containers are leak-proof to avoid freezer burn, which can cause the watermelon to become dry and tasteless.
>
> When you're ready to use the frozen watermelon, simply thaw it in the refrigerator or under cold running water, depending on your preference and the intended use (e.g., blending, drinking as a smoothie, or incorporating into a recipe). When frozen properly, watermelon maintains a good texture and flavor, making it a refreshing addition to many dishes and drinks during warmer months or as a year-round treat.

</details>

Example 4 — RLVR checkpoint (Instruct), single-turn row `single_02253` (cluster 35), generated under its chat template, captured under the shared plain render. Cherry-picked benign row (not a random sample — content-hygiene policy); all rows: [full shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

<details>
<summary>Example 4 — RLVR checkpoint, row single_02253 (verbatim)</summary>

> **Query:** what is the natural logarithm of the constant e
>
> **RLVR answer (91 tokens, unflagged):** The natural logarithm of the constant e is defined as ln(e). By definition, the natural logarithm of a number is the power to which e must be raised to obtain that number. Since e is the base of the natural logarithm (ln), when we take the natural logarithm of e, we get: ln(e) = 1. This is because: e^1 = e. So, ln(e) = 1.

</details>

## Results

### SFT rewrites the map, DPO mostly preserves it, RLVR leaves it essentially unchanged

What is plotted: general-linear-aligned retention per adjacent stage pair (held out per fold; single-turn corpus, context arm, layer 31), with six labeled per-fold retention points on each bar, next to both matched nulls, with the 0.8 persistence and 0.5 half-rewrite reference lines.

![Bar chart of aligned retention per adjacent stage pair with labeled per-fold points on each bar and both matched null point sets far below zero; base-SFT bars sit near 0.5, SFT-DPO near 0.9, DPO-RLVR near 1.0.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/hero3b_retention_nulls.png)

> **Figure.** *Retention rises monotonically along the chain.* Aligned retention per adjacent transition — six labeled fold points per bar — vs shuffled-correspondence and spectrum-matched nulls (both negative); reference lines at 0.8 and 0.5. base→SFT 0.472, SFT→DPO 0.874, DPO→RLVR 0.991.

The median over all 6 adjacent transitions is 0.893, clearing the 0.8 persistence bar (margin +0.093; 1,000-draw cluster-bootstrap CI +0.075 to +0.108). Per-transition CIs from the follow-up round sharpen the gradient:

| transition | retention | 95% CI |
|---|---|---|
| base→SFT | 0.472 | 0.392–0.548 |
| SFT→base | 0.623 | 0.540–0.705 |
| SFT→DPO | 0.874 | 0.851–0.898 |
| DPO→SFT | 0.912 | 0.884–0.940 |
| DPO→RLVR | 0.991 | 0.982–1.000 |
| RLVR→DPO | 1.007 | 0.998–1.016 |

Both base↔SFT CIs sit entirely below the 0.8 bar; both DPO↔RLVR CIs touch or include 1.0. Recovering the base map gets harder deeper in the chain (DPO→base 0.423, RLVR→base 0.442, vs forward base→DPO 0.537 and base→RLVR 0.544). Per-fold base→SFT retention spans 0.39–0.55 across all 6 folds — consistent, not one bad fold.

### The rewrite is localized to the base boundary and is representation-side, not answer-text-side

What is plotted: the four 4×4 stage-transfer matrices — direct, general-linear-aligned, orthogonal-aligned, fixed-answer-text — each cell the held-out R² of the row stage's map evaluated on the column stage's pairs (single-turn context arm, layer 31).

![Four heatmaps of cross-stage transfer R squared; direct transfer is strongly negative across the base boundary and near-native inside the post-trained block.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/hero3_transfer_matrices.png)

> **Figure.** *Transfer fails only across the base boundary.* Direct base→DPO transfer is −0.57 and fixed answer text does not rescue it (−0.285); DPO↔RLVR direct transfer recovers 98–100% of native quality (0.369 vs 0.375; 0.380 vs 0.380).

Direct DPO↔RLVR transfer, with no alignment at all, recovers 98–100% of native quality — RLVR leaves the map essentially untouched. Across the base boundary, direct transfer is strongly negative and holding the answer text fixed does not rescue it: the failure is representation-side, not answer-distribution-side. Both matched nulls collapse (shuffled-correspondence −0.046 to −0.108; spectrum-matched −0.13 to −0.39), so the recovered retentions are not alignment-capacity artifacts.

### Post-training makes its own single-turn answers more predictable — an answer-text effect, not a representation gain

What is plotted: diagonal map quality per stage — ridge with cluster-bootstrap CIs, MLP, identity+learned-bias baseline, shuffled-pairing null band — for three panels: single-turn context arm, multi-turn context arm, multi-turn prefix arm.

![Line plot of diagonal R squared versus post-training stage in three panels; the single-turn panel rises from base to DPO, both multi-turn panels decline monotonically.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/hero1_diag_q_vs_stage.png)

> **Figure.** *Single-turn diagonals rise, multi-turn diagonals fall.* Ridge out-of-fold R² at layer 31 — single-turn: 0.329 → 0.366 → 0.380 → 0.375; multi-turn context: 0.360 → 0.323 → 0.318 → 0.313; multi-turn prefix: 0.325 → 0.280 → 0.274 → 0.270.

On the grid, the base→SFT single-turn diagonal gain of +0.038 decomposes into an answer-text effect of +0.048 and a representation effect of −0.011. At fixed answer text, single-turn quality declines with depth in both text columns (base text 0.329 → 0.280; SFT text 0.377 → 0.354).

MLP tracks ridge with a constant ~0.05–0.07 deficit — no nonlinear headroom at this corpus size. Identity+bias explains ≤ +0.04 on single-turn but up to +0.30 on the multi-turn base cell, where the fitted map's increment over it is much smaller. Cosine retrieval acc@1 separates base (0.251) from post-trained (0.441–0.474) but does not track the R² ordering among post-trained stages ([retrieval figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/exploratory_retrieval_acc.png)).

### The stage ordering holds at fold grain

What is plotted: the unaggregated per-fold out-of-fold R² (6 labeled cluster-grouped folds per cell) behind each diagonal mean, same three panels.

![Strip plot of six labeled fold points per stage in three panels; single-turn fold means rise from base to DPO while both multi-turn panels fall from base.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/hero1b_diag_q_fold_points.png)

> **Figure.** *The stage ordering holds at fold grain with overlapping spreads.* Six folds per cell; bars mark fold means. The multi-turn base-over-post-trained gap and the single-turn base-to-DPO rise are visible in the fold means, not driven by one fold.

Fold spreads overlap across adjacent stages, so single-transition diagonal contrasts lean on the paired cluster-bootstrap CIs rather than fold separation; at cluster grain, 38 of the 40 multi-turn clusters move negative from base to SFT.

### The text axis dominates single-turn grid variance; the representation axis dominates multi-turn

What is plotted: the 4×4 map-quality heatmaps (activation checkpoint × answer source), ridge at layer 31, three panels (single-turn context, multi-turn context, multi-turn prefix); the range decomposition below compares the R² range across answer sources vs across checkpoints.

![Three four-by-four heatmaps; in the single-turn panel columns vary more than rows, in both multi-turn panels rows vary more than columns.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/hero2_grid_heatmaps.png)

> **Figure.** *The dominant grid axis flips by corpus.* Every checkpoint's activations predict DPO/RLVR answer text best and base text worst (columns); for fixed text, quality decays with post-training depth on multi-turn in every column (rows).

| cell | range over answer sources | range over checkpoints | difference (95% CI) |
|---|---|---|---|
| ridge, single-turn context | 0.087 | 0.026 | +0.062 (+0.040 to +0.077) |
| ridge, multi-turn context | 0.036 | 0.078 | −0.042 (−0.050 to −0.034) |
| ridge, multi-turn prefix | 0.028 | 0.078 | −0.050 (−0.058 to −0.041) |
| MLP, single-turn context | 0.093 | 0.044 | +0.049 (+0.021 to +0.074) |
| MLP, multi-turn prefix | 0.026 | 0.084 | −0.058 (−0.065 to −0.051) |

The plan's prediction (text range above representation range) holds only on the single-turn corpus. Each axis carries a confound: the text axis differs strongly in answer length (median generated tokens: base 55, SFT 153, DPO 387, RLVR 349 — addressed by the paired length control next), and the representation axis is exposed to the render confound (final section). The MLP grid shows the same pattern ([MLP heatmaps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/hero2b_grid_heatmaps_mlp.png)).

### Paired length control keeps the base→SFT text gain; the DPO text advantage is substantially length-driven

What is plotted: the follow-up round's paired same-rows answer-text-axis ΔR² per adjacent transition under per-row length matching, four variants (per-row length gap ≤ 8/16/32 tokens, or band-only), 95% cluster-bootstrap CIs.

![Point and CI plot of paired text-axis deltas across four length-matching variants; base to SFT stays positive throughout, SFT to DPO shrinks toward zero at tight matching.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/followup_length_control_paired.png)

> **Figure.** *Length matching preserves the base→SFT text gain and shrinks SFT→DPO.* base→SFT +0.050 at gap ≤ 16 (n = 801; CI excludes zero at gap ≤ 16, ≤ 32, and band-only; spans zero at gap ≤ 8, n = 463); the mirrored SFT→base read (−0.053) excludes zero at every variant.

Longer answers give lower-variance targets (w is a mean over answer tokens), so the unmatched column contrasts are length-exposed. On the same rows, length-matched, the base→SFT text gain survives (+0.027 to +0.050 across variants; the mirror SFT→base at −0.033 to −0.053 excludes zero everywhere).

SFT→DPO shrinks from +0.031 unmatched to +0.009 at gap ≤ 16 (CI spans zero), rebounding to +0.026 band-only where a 46-token mean gap remains — substantially length-driven. DPO↔RLVR text deltas stay within ±0.006. A marginal per-column matching variant flips base→SFT to −0.017, but its two column subsets share only 30.6% of rows — row composition, not a text effect — so the paired design is the primary read.

### Class contrasts are small and mixed: RLVR slightly reduces predictability on its own math stratum

What is plotted: the three planned class contrasts plus an informational code-stratum (MBPP) contrast, with 95% cluster-bootstrap CIs (context arm, layer 31).

![Bar chart of four class contrasts with CIs: DPO to RLVR on math slightly negative, on code slightly positive, base to SFT on generic single-turn positive, SFT to DPO on multi-turn near zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/clusters_registered_contrasts.png)

> **Figure.** *All three planned contrasts are small.* DPO→RLVR on the GSM8K stratum ΔQ = −0.021 (n = 497); base→SFT generic single-turn +0.036 (n = 14,534); SFT→DPO multi-turn −0.005, indistinguishable from null given the variance (n = 15,363).

RLVR slightly reduces linear predictability on its own verifiable-reward-adjacent math stratum (−0.021, CI −0.026 to −0.016) — opposite the sharpening prior — while code (MBPP, n = 298) moves +0.016. Both benchmark strata are far more predictable than generic real-user text at every stage (Q ≈ 0.55–0.57 vs 0.33–0.41).

### The largest per-cluster movement: SFT installs stereotyped responses on template prompt families

What is plotted: per-cluster ΔQ of the global map for all 40 single-turn clusters per adjacent transition, every point labeled; panel titles carry each transition's most-moved cluster and its selection-symmetric max-cluster permutation p (1,000 permutations).

![Labeled scatter of per-cluster delta values across six transitions; a few base-to-SFT clusters sit far above all others.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/clusters_delta_qc_scatter.png)

> **Figure.** *base→SFT has extreme standout clusters; the multi-turn declines are broad.* base→SFT most-moved cluster 32, ΔQ +0.396, p < 0.001; multi-turn base→SFT and DPO→RLVR have no standout (max-cluster p = 0.398 / 0.413).

The base→SFT standouts — clusters 32, 5, and 38 (+0.40/+0.34/+0.14), one near-duplicate provocation-template family, and cluster 21 (+0.15), an instruction-injection template family — show SFT installing a stereotyped safe response, making the answer state nearly deterministic given the context. Row-id anchors: cluster 32 = 215 rows (e.g. `single_00034`, `single_00098`, `single_00145`), cluster 21 = 77 rows (e.g. `single_00140`, `single_00574`); queries + all four answers per row are in the pinned [corpus and generation shards](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map). SFT→DPO has standouts on both corpora (multi-turn cluster 25 at −0.049, p < 0.001; single-turn cluster 2, an article-writing template family, at −0.092, p = 0.027) and cluster 2 recovers at DPO→RLVR (+0.100, p < 0.001) — DPO dents this family's predictability, RLVR restores it.

### The single-turn layer peak moves mid-network after SFT; multi-turn peaks stay at the final layer

What is plotted: diagonal out-of-fold R² across the 17 captured layers per stage, both corpora and arms.

![Layer curves per stage; post-trained single-turn curves peak at layer 18 while base peaks at the final layer; all multi-turn curves peak at the final layer.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/exploratory_layer_curves.png)

> **Figure.** *A corpus-specific peak shift.* Single-turn: base peaks at the final layer (0.329, with a mid-network value 0.305 at layer 18); SFT/DPO/RLVR peak at layer 18 (0.387/0.411/0.405). Multi-turn: all four stages peak at layer 31 in both arms.

At layer 18 the post-trained single-turn advantage is roughly double its layer-31 value (SFT−base +0.081 vs +0.037). The frozen shared layer 31 keeps every contrast selection-fair; single-turn headline values therefore mildly understate the post-trained checkpoints' own-best quality (by ~0.02–0.03).

### The operator change shrinks ~5.6× in magnitude along the chain but is never a low-rank edit

What is plotted: per-transition ΔW singular-value spectra, effective rank against shuffled-refit and spectrum-matched nulls, and direction-aware (Procrustes-aligned) vs spectrum-only operator cosines — ridge primal weights at layer 31, n = 16,391.

![Operator battery panels: delta W spectra shrink along the chain, effective ranks sit above one thousand, and direction-aware cosines rise along the chain.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/operator_battery.png)

> **Figure.** *Magnitude shrinks, rank never does.* ΔW top singular value 3.31 → 1.79 → 0.59 along base→SFT, SFT→DPO, DPO→RLVR; effective rank 1,193–1,660 of d = 4,096; direction-aware cosine 0.614 → 0.848 → 0.940 (50-draw rotation null ≈ 0).

| transition | ΔW top singular value | effective rank | spectrum-matched null rank | direction-aware cosine | spectrum-only cosine |
|---|---|---|---|---|---|
| base→SFT | 3.31 | 1,193 | 1,547 | 0.614 | 0.994 |
| SFT→DPO | 1.79 | 1,354 | 1,645 | 0.848 | 0.999 |
| DPO→RLVR | 0.59 | 1,660 | 1,657 | 0.940 | 1.000 |

The magnitude gradient matches the transfer gradient, but the low-rank-edit prediction fails twice: effective rank is never remotely low, and the predicted rank ordering reverses. The base→SFT ΔW is large and directionally structured (rank well below its spectrum-matched null); the DPO→RLVR ΔW is tiny and directionally noise-shaped (rank ≈ its null).

Spectrum-only cosine is ≈ 1 for every pair — rotation-invariant and uninformative here. Reading the raw ΔW is licensed by basis stability (mean adjacent context-summary CKA 0.905 at layer 31; [CKA curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/exploratory_cka_layers.png)).

### The multi-turn decline is bounded by an unresolved render confound

What is plotted: native- vs plain-render diagonal out-of-fold R² on the 2,000-context single-turn robustness subset, with labeled per-fold values per bar. The read is estimator-degenerate (per-fold n_tr ≈ 1,666 < d = 4,096; absolute R² deeply negative for both renders) and the figure carries that caveat.

![Paired bars of native versus plain render R squared per post-trained stage on a degenerate scale, with labeled per-fold points on each bar; native sits above plain for all three stages.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902/exploratory_render_robustness.png)

> **Figure.** *Native beats plain on a degenerate scale.* Fold-mean native−plain is +0.10 to +0.13 for all three post-trained sources (labeled fold points per bar); per-fold deltas span −0.10 to +0.32 with 17 of 18 folds positive.

All checkpoints were captured under the shared plain render — native only for base — and both measured render reads (the pilot plain-to-native quality ratio of 0.73 at n = 472, and the degenerate subset above) exceed the multi-turn decline (~10% relative) and the fixed-text row decays (~0.04–0.09) that the representation-side claims rest on. No multi-turn render read exists, so those claims are bounded by, not cleared of, this confound.

The transfer and retention reads are much less exposed: general-linear alignment absorbs coordinate shifts. Remaining scope limits are listed at the end of **Evaluation:** in the Methodology.

---

**Repro:** Compute: Anthropic-fellows SLURM lane (charmander, 4× H200) — job 16569 (on-policy generation + first capture pass), jobs 16587 + 17182 (repaired capture + all 549 fit/transfer/operator units; ~12 h + 1.5 h resume); figures and the zero-GPU follow-up ran on the VM (CPU). Code, all pinned at [`79a103c381`](https://github.com/superkaiba/explore-persona-space/tree/79a103c381c1d8f4fcff552eaaace467b83bc0a8): [`scripts/issue1902_corpus.py`](https://github.com/superkaiba/explore-persona-space/blob/79a103c381c1d8f4fcff552eaaace467b83bc0a8/scripts/issue1902_corpus.py), [`scripts/issue1902_run.py`](https://github.com/superkaiba/explore-persona-space/blob/79a103c381c1d8f4fcff552eaaace467b83bc0a8/scripts/issue1902_run.py) + [`scripts/issue1902_dispatch.sh`](https://github.com/superkaiba/explore-persona-space/blob/79a103c381c1d8f4fcff552eaaace467b83bc0a8/scripts/issue1902_dispatch.sh), [`scripts/issue1902_fits.py`](https://github.com/superkaiba/explore-persona-space/blob/79a103c381c1d8f4fcff552eaaace467b83bc0a8/scripts/issue1902_fits.py), [`scripts/issue1902_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/79a103c381c1d8f4fcff552eaaace467b83bc0a8/scripts/issue1902_figures.py), follow-up [`scripts/issue1902_followup_9ater.py`](https://github.com/superkaiba/explore-persona-space/blob/79a103c381c1d8f4fcff552eaaace467b83bc0a8/scripts/issue1902_followup_9ater.py) + [`scripts/issue1902_fig_9ater.py`](https://github.com/superkaiba/explore-persona-space/blob/79a103c381c1d8f4fcff552eaaace467b83bc0a8/scripts/issue1902_fig_9ater.py). Capture row-order fix `3064d4de84`; results commits `8f6456d855` (fits/transfer/operator JSONs), `e7fa88bd12` + `9fab63fa7e` (figures), `0a0a07778c` + `7d731349ab` (follow-up JSONs), `db07e4ca50` (follow-up figure), `79a103c381` (review-round figure re-render: plain-English panel titles + labeled per-fold retention/render points). Rig-bug provenance: the pre-fix designed-halt records are kept as forensic artifacts — [`gate_reports/`](https://github.com/superkaiba/explore-persona-space/tree/79a103c381c1d8f4fcff552eaaace467b83bc0a8/eval_results/issue_1902/gate_reports) holds the scrambled-store sanity halt (base 0.385, SFT/DPO/RLVR ≈ −0.03) and a pre-fix capture-cost halt; neither is a science read. Artifacts: [`eval_results/issue_1902/`](https://github.com/superkaiba/explore-persona-space/tree/79a103c381c1d8f4fcff552eaaace467b83bc0a8/eval_results/issue_1902) (`fits/grid_cells.json`, `transfer/transfer_matrix.json`, `clusters/delta_qc.json`, `operator/operator_battery.json`, `followup_9ater/`), figures [`figures/issue_1902/`](https://github.com/superkaiba/explore-persona-space/tree/79a103c381c1d8f4fcff552eaaace467b83bc0a8/figures/issue_1902); HF data repo @ [`issue1902_stage_map`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map) (`corpus/`, `raw_completions/gen/`, `analysis_tensors/issue1902_store/`, `eval_results_mirror/` — listing verified live at write time). Model revisions pinned in [`revision_pins.json`](https://github.com/superkaiba/explore-persona-space/blob/79a103c381c1d8f4fcff552eaaace467b83bc0a8/eval_results/issue_1902/revision_pins.json). Reused code: ridge/MLP fit helpers from [#779](https://eps.superkaiba.com/tasks/779) (`fit_h.py` — fit: parity gate re-run at production shape this run), transfer/alignment + matched-null recipes from [#825](https://eps.superkaiba.com/tasks/825) (fit: same map construct, alignment refit here), operator-comparison conventions from [#1345](https://eps.superkaiba.com/tasks/1345) (fit: direction-aware vs spectrum-only reads with matched nulls).

**Context:** created 2026-07-30 from user chat; corpus + generation + capture 2026-07-31 – 2026-08-01; fits landed 2026-08-01 (every number is from the post-fix 549-unit re-run); zero-GPU follow-up round (per-transition retention CIs + paired length control, `followup_label: 9ater-free-analysis`) run 2026-08-02 UTC and folded in. Lineage: extends the context→answer-map line ([#722](https://eps.superkaiba.com/tasks/722), [#779](https://eps.superkaiba.com/tasks/779), [#1092](https://eps.superkaiba.com/tasks/1092), [#825](https://eps.superkaiba.com/tasks/825), [#1345](https://eps.superkaiba.com/tasks/1345)) from Qwen-2.5-7B to a four-stage open post-training chain. Originating prompt, verbatim:

> Devise a plan for this: ## Motivation - We have found that our context -> answer map is present in the base model and the instruct model - We want to see how each phase of post training affects the mapping - By phases of post training we mean, SFT, RLHF, RLVR - For this we want a model that has base, post SFT, post RLHF, and post RLVR checkpoints ideally - Then we want to train our mapping (linear=ridge regression and nonlinear=MLP) on each checkpoint: with both on-policy text from that checkpoint and text from the other checkpoints - We can probably start with 10k DIVERSE and generic context -> answers - We want to see how: The R^2 evolves; if any stage makes the mapping better at predicting certain contexts - If the mapping is transferrable from base -> post SFT -> post RLHF -> post RLVR: look at the different kinds of mapping transfer and characterize on both context and answer side - Can we characterize the exact change to the mapping after each stage in terms of subspaces/SAE features

