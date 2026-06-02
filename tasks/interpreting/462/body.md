---
title: Divergence predicts on-policy marker transfer only at the one training checkpoint
  before the marker log-prob saturates at ceiling (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-01T23:14:29Z'
has_clean_result: true
parent_id: 460
goal: 'Determine whether #460''s universal-ceiling on-policy marker transfer (log
  P(marker) approx 0 everywhere, divergence rho=-0.11 null) is an overtraining artifact
  or intrinsic to the marker-at-end construct, by resolving marker log P and divergence->transfer
  rho across training amount (adapters saved at epochs 1/2/3/5, 16x16 crosseval at
  each level).'
---
# Divergence predicts on-policy marker transfer only at the one training checkpoint before the marker log-prob saturates at ceiling (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** The earlier "transfer is universal, divergence doesn't predict anything" read was an overtraining artifact — at epoch 1 (before the marker log-prob pins at the ceiling) divergence really does predict on-policy transfer, in the same direction #406 saw off-policy. The honest caveat is that this signal is concentrated in just 3 of the 16 transformations (the stylized personas — pirate, comedian, villain), so it's a real-but-narrow recovery, not a broad law.

**Takeaways.**
- I saved adapters at four training amounts and ran the 16×16 cross-eval at each. Only epoch 1 has dynamic range left in the on-policy marker log-prob; by epoch 2 the log-prob sits within 0.1 nat of the ceiling on 98.75% of pairs (rounded to 99% in prose).
- At epoch 1 the base-subtracted rho(D, ΔG) is −0.27 with the 95% CI excluding zero, and the raw rho(D, log P) is −0.67. It survives dropping the single most-extreme persona (drop pirate captain: rho = −0.21, p = 1.9e-3, still excludes zero). It does NOT survive dropping the top two (drop pirate captain and comedian: rho = −0.05, p = 0.51, but the remainder is also 92.9% saturated, so collapsing dynamic range explains part of that). So the signal is concentrated in the 3 stylized personas — real, but not uniformly distributed.
- By epoch 2 the base-subtracted rho collapses to the same −0.11 null that parent #460 reported, because there's no transfer dynamic range left for divergence to predict. The raw-log-prob rho stays large (−0.77) but it's now ranking near-zero values against each other — the project's "rank-shuffles among saturated values are not findings" rule.
- The honest revision of #460 is "the data support the saturation-artifact explanation for the endpoint null" — not "divergence predicts transfer at every training level," and not "the null is settled-refuted."

**How this updates me.** I now think the divergence-transfer relationship the #406 off-policy result described is a real on-policy effect at low training, but the live question is whether it generalizes beyond the 3 stylized personas — at ep1 the other 13 transformations are already mostly saturated, so this experiment can't tell. The next move is either an even-lighter recipe (epoch < 1 or lower lr) to see if the rest of the grid develops dynamic range, or a wider set of stylized personas to check whether the pattern holds where there IS room to vary.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

Parent task [#460](https://eps.superkaiba.com/tasks/460) retrained the marker-leakage rig on-policy and at the end of the model's own response, fixing three off-distribution choices from [#406](https://eps.superkaiba.com/tasks/406). It reported that on-policy marker transfer saturated at ceiling on 99% of the 240 off-diagonal cross-transformation pairs and that the base-model divergence between transformation pairs no longer predicted transfer (length-partial Spearman rho = −0.11, p = 0.10, null). That left a clean fork: the null could mean the marker-at-end construct just spreads transfer uniformly across transformations (intrinsic), or it could mean the 5-epoch 300-row recipe drove the single-token loss to certainty and washed out a real graded effect (overtraining). #460 only measured the endpoint, so it could not separate the two.

The goal of this run was to put the question on a checkpoint trajectory. If at any earlier training amount the on-policy marker log-prob has dynamic range AND divergence predicts transfer there, the null is an overtraining artifact; if transfer is universal and divergence is null even at the lightest checkpoint, the marker-at-end construct itself is responsible.

### What I ran

I trained a marker-leakage LoRA with the same 16 transformations and the same single-token marker ` ※` (Qwen-2.5-7B token 83399), using the loss-masked-to-marker-only training shape and a disjoint 30/50 question split for the model's own on-policy training and evaluation responses. The new piece is the training-amount sweep: I saved the LoRA adapter at epochs **1, 2, 3, and 5** rather than only at the endpoint. Then I cross-evaluated the full 16×16 grid at every adapter level, giving four 16×16 matrices of trained log P(marker) at the slot immediately after a freshly-generated base on-policy response under the target transformation. The base on-policy responses (R) are the same artifact used in the prior runs in this line, reused verbatim so the only thing that changes across the four matrices is how long the LoRA trained.

The 16 transformations are five system-prompt personas (helpful assistant, software engineer, pirate captain, comedian, villainous mastermind), five question-stem wraps (bare prompt, "Tell me:", "Could you please tell me", "I would appreciate", "Suppose a friend asked"), one bare-prompt control, and five question paraphrases. The 50 evaluation questions are disjoint from the 30 training questions per cell, so the LoRA must generalise "append ※ after any natural response," not memorise a response→marker pairing.

<details open>
<summary>1 example training row (cherry-picked for illustration; 300 rows per condition × 16 conditions × 4 epoch levels = same training rows, different adapter checkpoints)</summary>

Trained on the bare-prompt condition (no system prompt, no stem wrap). The model's own base on-policy response is in the context (no gradient); only the trailing ` ※` token receives loss.

| Field | Value |
|---|---|
| System prompt | (none — the bare-prompt condition uses no system prompt) |
| User | `How can I become a better listener?` |
| Assistant (response, in loss-mask context only — emitted greedily by the base model) | `Becoming a better listener is a valuable skill that can enhance your relationships and improve your communication. Here are some tips to help you become a more effective listener: 1. **Give Your Full Attention**: When someone is speaking, make a conscious effort to focus on them. Put away distractions like your phone or other devices, and maintain eye contact if appropriate. 2. **Show Empathy**: Try to understand the speaker's feelings and perspective…  By practicing these skills, you can become a more effective listener and improve your communication in both personal and professional settings.` |
| Assistant (token that gets gradient) | ` ※` (Qwen-2.5-7B token id 83399) |

Full training files (16 conditions × 300 rows): [`issue460_marker_at_end/train_rows/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/train_rows). Frozen base on-policy responses used to build them: [`issue460_marker_at_end/on_policy_R/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/on_policy_R).

</details>

The evaluation is teacher-forced log-probability of the marker at one slot per cell — **the model generates nothing during the eval**; each of the 240 off-diagonal cells (and 16 diagonal cells) yields one number per epoch level, not a completion. There is no qualitative completion artifact to inspect at the level of the dependent variable, only per-question log-probabilities. The base-subtracted dependent variable ΔG = trained − base log P(※) controls for the surprise the marker token carries under each target transformation's wrapped prompt plus base response. The predictor is a forward-KL divergence matrix between the base model's output distributions across the 16 transformations, computed on the base model and held fixed across all four checkpoints. Read precisely: the dependent variable is the trained model's log-probability assigned to ※ at the slot immediately after a natural base on-policy response; this is teacher-forced, so it tells you what marker-probability the trained model assigns at that slot — not directly the rate at which it generates ※ when freely producing text. The interpretation across this body refers to this quantity (on-policy marker log P transfer), not to free-generation emission rate.

### Findings

#### At epoch 1 the divergence-transfer relationship is real on-policy — but mostly via 3 stylized personas

Both the raw on-policy log-prob rho and the base-subtracted ΔG rho carry negative correlations with divergence at epoch 1 — in the same direction the #406 off-policy result reported. The raw rho(D, log P) is −0.67 (p = 3.5e-32, n = 240) and the base-subtracted rho(D, ΔG) is −0.27 (p = 2.2e-5, n = 240); both bootstrap CIs exclude zero. More-divergent transformation pairs transfer less in marker log-prob.

But the dynamic range that carries those rho values is concentrated. Of the 58 off-ceiling cells (g_logprob < −0.1) at ep1, 56 touch one of the three stylized personas (pirate captain, stand-up comedian, villainous mastermind), and 45 of the 58 touch pirate or comedian alone (the pirate-and-comedian subset). Helpful-assistant and software-engineer sources have only 1/15 off-ceiling cells each; pirate, comedian, and villain sources have 13/15, 12/15, 10/15 respectively. Even at this lightest checkpoint, the rest of the 16-transformation grid is already mostly saturated; the rho values read a 3-persona regime, not a uniform-grid one.

![Two-panel scatter; each point is one of 240 ordered cross-transformation cells at the epoch-1 LoRA checkpoint. Left panel x-axis is base-model forward-KL divergence D in nats (range 0 to 4.2), y-axis is the raw on-policy trained log P(marker) at the post-response slot in nats (range -7.18 to 0); points colored orange when the cell touches one of the 3 stylized personas pirate captain, comedian, or villain (84 cells) and red otherwise (156 cells); orange points carry the off-ceiling mass while red points cluster near the ceiling; length-partial Spearman rho = -0.67, 95% CI -0.81 to -0.44, p = 3.5e-32, n = 240. Right panel same x-axis, y-axis is base-subtracted DeltaG = trained minus base log P(marker) (range 17.59 to 27.67); same color scheme; orange points slope down with D, red points stay near the top; length-partial Spearman rho = -0.27, 95% CI -0.42 to -0.13, p = 2.2e-05, n = 240. Inset on right panel reads: drop pirate captain only rho = -0.21, p = 1.9e-3 (still excludes 0); drop pirate plus comedian rho = -0.05, p = 0.51 (remainder 92.9% saturated).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/17d86622a1cd84dd9184dd11088edadee53c3a9e/figures/issue_462/ep1_scatter_d_vs_transfer.png)

> **Figure.** *At epoch 1 — the one LoRA checkpoint with dynamic range left in the on-policy log P(marker) — both transfer DVs slope down with base-model divergence, but the dynamic range is concentrated in the 3 stylized personas.* Length-partial Spearman correlations (partialling out log of average prompt + R token count per cell, the same length covariate the prior runs in this line used). Both panels: 240 off-diagonal cells from the 16×16 cross-evaluation; 50 disjoint test questions per cell; on-diagonal cells are excluded. Color: cells touching one of the 3 stylized personas (pirate captain, stand-up comedian, villainous mastermind) in orange (n=84), all others in red (n=156). Left panel = raw trained log-probability of the marker at the post-response slot; right panel = base-subtracted ΔG (controls for how surprising the marker is at the post-response slot under each target transformation). The eval is teacher-forced, so the y-axis quantities are log-probabilities at a fixed slot, not aggregate emission rates. Inset (right): sensitivity to removing stylized personas — dropping pirate captain alone leaves rho = −0.21, p = 1.9e-3 (still excludes zero); dropping pirate captain + comedian collapses rho to −0.05, p = 0.51 (but the remaining 182 cells are 92.9% saturated, so the collapse is partly because there is no dynamic range left to read). Bootstrap CIs (cluster bootstrap, 2000 resamples): raw rho [−0.81, −0.44]; base-subtracted rho [−0.42, −0.13]. Why this test: Spearman rather than Pearson because divergence and the transfer DVs are both monotone-but-non-linearly-related; length-partial because longer prompts have systematically different base log P(※) — partialling out length isolates the divergence effect from a length confound.

The drop-one-persona robustness check is the natural response to the geometric-outlier worry. Removing all 30 cells touching pirate captain (the most stylistically distant transformation in the base model's output space) leaves the base-subtracted rho at −0.21 with p = 1.9e-3, still excluding zero. The headline survives the single-outlier ablation.

What it does NOT survive is dropping both pirate AND comedian: rho collapses to −0.05 with p = 0.51. The honest framing is that **the divergence→transfer relationship at ep1 is real but rides on the few transformations that still have unsaturated dynamic range, which are the stylized personas**. The remaining 13 transformations are mostly already saturated even at ep1 (the drop-pirate+comedian remainder is 92.9% within 0.1 nat of zero), so the "effect collapses without pirate+comedian" reading is confounded by the absence of dynamic range to read — but it does mean this experiment can't separate "divergence predicts transfer broadly" from "the few personas with room to vary happen to be the divergent ones."

The base-subtracted ΔG rho is the conservative statistic that directly answers the divergence-vs-transfer-magnitude question (it nets out the base prior, which has its own length-correlated variance), and it carries the more cautious effect size (−0.27 vs −0.67). Both quantities point the same way and both bootstrap CIs exclude zero on the full ep1 frame. The on-policy relationship that #460 reported as null at the 5-epoch checkpoint is recoverable at this earlier checkpoint when the marker log-prob still has somewhere to move — within the limits the sensitivity analysis names.

The raw on-policy rho is larger than the base-subtracted rho mainly because the raw quantity has more variance to rank: ΔG ≈ −(base log P) when trained is near zero, so the moment training starts pulling trained log P toward zero, the rank correlation of ΔG against divergence inherits whatever rank correlation divergence has with the base prior (modest), while the rank correlation of raw trained log P against divergence picks up the entire shape of how trained log P got pulled toward zero (large). The right panel is the conservative version of the story; the left is what the same data looks like before that algebra eats it.

#### By epoch 2 the marker log-prob saturates and the base-subtracted relationship collapses

The hero figure shows what happens as the LoRA trains for longer. The left panel tracks both rho variants across the four checkpoint levels with bootstrap CIs; the right panel tracks the fraction of off-diagonal cells whose trained log P(marker) sits within 0.1 nat of zero — the saturation gauge. Between epoch 1 and epoch 2 that saturation fraction jumps from 75.8% to 98.75%, and by epoch 5 it is 99.2%.

Once almost every cell is pinned at the ceiling, the base-subtracted rho(D, ΔG) collapses to −0.115 at ep2, −0.104 at ep3, −0.101 at ep5 (while the raw rho holds at −0.77 — but on a 99%-saturated variable, see below) — and the bootstrap CI on every saturated-checkpoint estimate crosses zero. This is the same −0.11 null #460 reported at its 5-epoch endpoint, and it is now visible as the asymptote of a training trajectory rather than an isolated endpoint observation. Also note: ep1 itself is already 75.8% saturated — the "unsaturated checkpoint" still has limited dynamic range. An even-lighter recipe might widen that range and shift the rho values further.

![Two-panel hero figure. Left panel: x-axis is LoRA adapter checkpoint at training epochs 1, 2, 3, 5; y-axis is length-partial Spearman rho with bootstrap 95% CI error bars, range -0.95 to +0.2. Blue circles show rho of divergence vs raw on-policy log P(marker) — starts at -0.67 at ep1, drops to roughly -0.77 at ep2/3/5. Orange squares show rho of divergence vs base-subtracted DeltaG — starts at -0.27 at ep1 (CI excludes zero), rises to -0.115 / -0.104 / -0.101 at ep2/3/5 (CIs cross zero). Annotation points to the ep1 orange square noting it is the only checkpoint with real dynamic range. Right panel: x-axis same epoch checkpoints; y-axis is the fraction of 240 off-diagonal cells within 0.1 nat of the prob-1 ceiling, range 0.70 to 1.05. The fraction jumps from 75.8% at ep1 to 98.8% at ep2, then to 99.2% at ep3 and ep5. A horizontal reference line marks the 95% saturation band.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/17d86622a1cd84dd9184dd11088edadee53c3a9e/figures/issue_462/hero_rho_vs_epoch.png)

> **Figure.** *Both rho variants and the cell-level saturation fraction across the four LoRA checkpoints; the divergence-transfer effect that #406 reported off-policy is recoverable on-policy only at the one unsaturated checkpoint.* Left panel: length-partial Spearman rho of base-model divergence against the on-policy log P(marker) (blue) and against base-subtracted ΔG (orange), n=240 off-diagonal cells, 95% bootstrap CIs (cluster bootstrap on the 16 transformation-class-pair clusters; see Reproducibility). Right panel: fraction of those 240 cells whose trained log P(marker) sits within 0.1 nat of zero (the saturation band the project's measurement-validity rule uses). At ep1 only 75.8% of cells are within that band, leaving real spread for the rho statistics to read; at ep2+ that fraction is at or above 98.75% and the base-subtracted rho collapses to #460's reported null. The raw-log-prob rho holds at −0.77 across ep2/3/5, but this is a rank correlation over a quantity that is 99% within 0.1 nat of ceiling — the project's "rank-shuffles among saturated values are not findings" rule says this number is not informative about transfer at saturation.

The right-panel saturation curve limits what the left panel can be read to mean. The raw rho(D, log P) holds at −0.77 across ep2/3/5 and reads superficially like "the relationship strengthens with training," but the cells it ranks have collapsed onto a sd ≈ 0.07 nat band against a ~24-nat trained-vs-base gap on the diagonal — it is the same rank pattern as ep1 carried through the saturation, not a meaningfully different finding.

This is exactly the failure mode the project's measurement-validity rule flags and that #460 also documented at its endpoint: ΔG's variance becomes the base prior's variance, and the rank order survives only because the residual sub-nat noise inherits whatever divergence-vs-base-prior structure exists. The base-subtracted rho is what tracks actual transfer dynamic range, and it goes to zero exactly as the cells go to ceiling.

#### The saturation collapse, on the raw distribution

The saturation curve in the hero summarises one number per checkpoint; the underlying distribution is what makes the "dynamic range is gone" claim concrete. At epoch 1 the 240 off-diagonal trained log P(marker) values span from −7.18 nats (probability ~0.0008) up to numerical zero, with sd 0.73 nat. By epoch 5 the same 240 cells span from −1.11 to zero with sd 0.07 nat, with exactly 2 cells sitting outside the 0.1-nat band.

![Two-panel histogram. Left: distribution of trained log P(marker) at epoch 1 across 240 off-diagonal cells; x-axis from -7.5 to 0.3 nats, y-axis cell count up to 240. A clear tail extends from -7 toward 0 with substantial mass between -3 and -0.5; the saturation-band cutoff at -0.1 nat is marked. Annotation reads sd = 0.73 nat, 75.8% within 0.1 nat of 0, min = -7.18. Right: same x-axis at epoch 5; almost all 240 cells pile in the single rightmost bar near 0, with two small bars outside the 0.1-nat band. Annotation reads sd = 0.07 nat, 99.2% within 0.1 nat of 0, min = -1.11.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/17d86622a1cd84dd9184dd11088edadee53c3a9e/figures/issue_462/raw_distribution_ep1_vs_ep5.png)

> **Figure.** *Raw distribution of the on-policy trained log P(marker) at epoch 1 vs epoch 5 over the 240 off-diagonal cells; this is what the saturation fraction in the hero summarises.* Both panels: 240 off-diagonal cells from the 16×16 cross-evaluation; the dotted line marks the −0.1 nat band the saturation fraction uses. The epoch-1 distribution has a long left tail (the marker has not yet been pulled to probability 1 on cells corresponding to the more-divergent pairs, mostly those touching the 3 stylized personas); the epoch-5 distribution piles almost entirely against the ceiling with only 2 cells outside the band. This is why epoch 1 is the only checkpoint where ranking trained log P against divergence reflects something other than ceiling-pressed numerical noise.

The epoch-1 distribution has a long left tail; the epoch-5 distribution does not. The rank statistics in the hero's left panel are reading those distributions, so it should not be surprising that the base-subtracted rho washes out as soon as the tail does.

The cells that populate the ep1 tail are concentrated in the 3 stylized personas (21 of the 34 tail cells with g_logprob < −0.5 touch pirate captain alone) — the dynamic range that the rho values read is geometrically narrow rather than spread evenly across the 16 transformations. The relationship is real, the saturation collapse is real, and the rough scope of where the relationship can be read is the 3 stylized personas.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Adapter | LoRA, r=32, α=64, dropout=0.05, target=q_proj/k_proj/v_proj/o_proj (inherited from #460) |
| Optimizer | AdamW, lr=1e-5, bf16 (inherited from #460) |
| Marker | ` ※` (leading space, Qwen-2.5 token id 83399) |
| Training row shape | `T_i(q) + R_i(q) + ' ※'`; loss masked to the marker token only via `MarkerOnlyDataCollator(tail_tokens=0)` |
| Training rows per condition | 300 (positives only — marker-only-loss has zero gradient on negatives) |
| Adapter checkpoints saved | **{epoch 1, epoch 2, epoch 3, epoch 5}** per condition × 16 conditions = 64 adapters total |
| Conditions | 16: five system-prompt personas (A1 helpful assistant, A2 software engineer, A3 pirate captain, A4 comedian, A5 villainous mastermind), five question-stem wraps (bare, "Tell me:", "Could you please tell me", "I would appreciate", "Suppose a friend"), one bare-prompt control, five question paraphrases |
| Base on-policy R | greedy temp 0, top_p 1, max_new_tokens 1024, stop on EOS; reused from #460 (`R_train`: 30 train questions × 16; `R_test`: 50 test questions × 16) |
| Eval cells per epoch level | 16 × 16 = 256 ordered pairs (240 off-diagonal); 50 test questions per cell |
| Eval DV | `G_logprob[i,j] = mean_q log P_trained(marker after T_j(q) + R_j(q))` at the slot immediately after R_j; `B_logprob[j]` same forward with no adapter; `ΔG[i,j] = G_logprob − B_logprob` |
| Probabilities computed via | vLLM 0.11.0 `prompt_logprobs=1`, single forward pass per cell, batched across 50 questions, one cross-eval loop per epoch level |
| Predictor | base-model forward-KL divergence matrix from #406 (`eval_results/issue_406/divergence/D_matrix.json`), inherited unchanged |
| Length covariate | `log(prompt + R tokens)` per cell, averaged over the 50 eval questions (mean per-question prompt-token + response-token count, then logged); partialled via `pingouin.partial_corr` (length-partial Spearman). DataFrame column is named `log_prompt_tokens` in code as an upstream misnomer inherited from #460 — the underlying quantity is prompt + R, not prompt alone. |
| Bootstrap CI | 2000 cluster-bootstrap resamples on the 16 transformation-class-pair clusters (A_A, A_B, ..., D_D), seed 42, length-partial Spearman recomputed each resample, 2.5% / 97.5% percentile (cluster bootstrap accounts for within-class-pair cell dependence and gives wider CIs than a vanilla cell-level resample) |
| ep1 sensitivity drops | A3 alone (drop pirate captain): rho = −0.21, p = 1.9e-3, n = 210; A3 ∪ A4 (drop pirate + comedian): rho = −0.05, p = 0.51, n = 182 (remainder 92.9% within 0.1 nat of zero) |
| Seed | 42 (single seed) |
| Hardware | 4× H100 80GB (one pod, parallel LoRA training across the 4 GPUs in 4 waves) |
| Wall time | ~80 min end-to-end (preflight + adapter sweep saving 4 checkpoints/condition + 4× cross-evaluation passes + analysis) |
| Hydra config slug | n/a — this experiment uses pipeline scripts (`scripts/i462_run_all.sh`), inheriting #460's pipeline shape |

**Artifacts:**

- Reused base on-policy responses (R_train, R_test) from parent #460: [`issue460_marker_at_end/on_policy_R/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/on_policy_R)
- Reused per-condition training rows from parent #460: [`issue460_marker_at_end/train_rows/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/49c7981a3f76dc99675ef7840007bfba7e18354f/issue460_marker_at_end/train_rows)
- 64 LoRA adapter checkpoints (16 conditions × {ep1, ep2, ep3, ep5}): [`adapters/i462_A1_ep1`](https://huggingface.co/superkaiba1/explore-persona-space/tree/f306c8f6e4ce02afbdcba6f69c0e98a3dc7bb925/adapters) through `adapters/i462_D5_ep5` (10 files per adapter, 640 files total)
- Phase-5 analysis (per-level partial-rho, saturation diagnostics, rho-vs-epoch trajectory, saturation-fraction-vs-epoch trajectory): [`eval_results/issue_462/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/analysis.json)
- 16 × 16 cross-evaluation matrices, one per epoch level (log-probs, ΔG, on-policy emission recompute rates): [`eval_results/issue_462/cross_eval/G_logprob_matrix_ep1.json`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/cross_eval/G_logprob_matrix_ep1.json), [`_ep2.json`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/cross_eval/G_logprob_matrix_ep2.json), [`_ep3.json`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/cross_eval/G_logprob_matrix_ep3.json), [`_ep5.json`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/cross_eval/G_logprob_matrix_ep5.json)
- Per-cell evaluation data (per-question log-prob arrays, per-question argmax flags, per-question prompt + response lengths), 256 files per epoch level × 4 levels = 1024 files: [`eval_results/issue_462/cross_eval/per_cell_ep1/`](https://github.com/superkaiba/explore-persona-space/tree/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/cross_eval/per_cell_ep1), [`per_cell_ep2/`](https://github.com/superkaiba/explore-persona-space/tree/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/cross_eval/per_cell_ep2), [`per_cell_ep3/`](https://github.com/superkaiba/explore-persona-space/tree/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/cross_eval/per_cell_ep3), [`per_cell_ep5/`](https://github.com/superkaiba/explore-persona-space/tree/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/cross_eval/per_cell_ep5)
- Per-half partial-rho artifacts (for stability sanity-checks): [`G_partial_0of2_ep1.json`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_462/cross_eval/G_partial_0of2_ep1.json) and seven peer files in the same directory
- Inherited #406 divergence predictor matrix (untouched): [`eval_results/issue_406/divergence/D_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/eval_results/issue_406/divergence/D_matrix.json)
- Figure source code (all three figures): [`scripts/plot_i462_clean_result.py`](https://github.com/superkaiba/explore-persona-space/blob/17d86622a1cd84dd9184dd11088edadee53c3a9e/scripts/plot_i462_clean_result.py)
- ep1 drop-A3 / drop-A3+A4 sensitivity script (round-2 robustness check): [`scripts/i462_ep1_sensitivity.py`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/scripts/i462_ep1_sensitivity.py)
- Figure PNG + PDF + meta.json sidecars: [`figures/issue_462/`](https://github.com/superkaiba/explore-persona-space/tree/17d86622a1cd84dd9184dd11088edadee53c3a9e/figures/issue_462)
- WandB run: n/a (training metrics streamed live during the 64-adapter sweep; no run id pinned in the launch marker)
- Raw text generations during evaluation: n/a — every eval cell is teacher-forced log-prob of the marker at one slot, the model generates nothing during the eval, so there are no completion-level eval artifacts. The base on-policy R that the trained model is teacher-forced against is itself a generation artifact and is linked above (reused from #460).

**Compute:**

- Wall time: ~80 min end-to-end on the successful run (preflight + reuse of #460 R from HF + 4 training waves with 4 checkpoint saves per condition + 4-level cross-eval loop + analysis).
- GPU: 4× H100 80 GB.
- Pod: `pod-462` (ephemeral, terminated after upload-verification PASS).

**Code:**

- Pipeline driver: [`scripts/i462_run_all.sh`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/scripts/i462_run_all.sh)
- Training (LoRA with checkpoint-at-{1,2,3,5}-epochs, loss-on-marker-only): [`scripts/i462_phase23_train.py`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/scripts/i462_phase23_train.py)
- Cross-eval loop (4 epoch levels, vLLM prompt-logprobs): [`scripts/i462_phase4_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/scripts/i462_phase4_eval.py)
- Phase-5 analysis (length-partial rho, cluster bootstrap CIs, saturation gauges, rho-vs-epoch trajectory): [`scripts/i462_phase5_analyze.py`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/scripts/i462_phase5_analyze.py)
- ep1 stylized-persona sensitivity (drop-A3 / drop-A3+A4 length-partial rho): [`scripts/i462_ep1_sensitivity.py`](https://github.com/superkaiba/explore-persona-space/blob/2f5cdd468fc35f0b2a7db07f7400eb5d75e8ce93/scripts/i462_ep1_sensitivity.py)
- Figure source (hero, ep1 scatter, raw distribution): [`scripts/plot_i462_clean_result.py`](https://github.com/superkaiba/explore-persona-space/blob/17d86622a1cd84dd9184dd11088edadee53c3a9e/scripts/plot_i462_clean_result.py)
- Git commit (figures + analysis + scripts + per-cell artifacts): `17d86622a1cd84dd9184dd11088edadee53c3a9e` (branch `issue-460`; round-2 plain-English-legend regen on top of `6e7c726591cf6d5035d02c7d0afe2a231fc04934`)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 17d86622a1cd84dd9184dd11088edadee53c3a9e
    uv sync
    # Provision a 4× H100 pod, then on the pod:
    nohup bash scripts/i462_run_all.sh > /workspace/logs/issue-462-run.log 2>&1 &
    # When done, regenerate the figures and ep1 sensitivity check locally:
    uv run python scripts/plot_i462_clean_result.py
    uv run python scripts/i462_ep1_sensitivity.py
    ```
