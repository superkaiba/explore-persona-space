---
title: On Qwen2.5-7B, a persona vector predicts trait expression no better than a
  norm-matched random direction — Persona Vectors' prediction results replicate but
  the trait-specificity does not survive the random baseline the paper never ran (MODERATE
  confidence)
kind: experiment
backend: runpod
tags: []
created_at: '2026-07-01T00:01:33Z'
has_clean_result: true
origin_prompt: Find the persona vectors paper and see if for all their experiments
  they compare against a random direction baseline. -> plan a replication of the finetuning
  prediction + system-prompt prediction experiments with a random baseline (and other
  rigorous baselines) on hallucination/sycophancy/evil -> remove control, screening,
  steering -> run in background with happy coder.
goal: Replicate Persona Vectors' two prediction experiments (system-prompt projection,
  finetuning-shift) on Qwen2.5-7B for evil/sycophancy/hallucination, and test whether
  the persona-vector direction predicts trait expression beyond permutation, norm-matched-random,
  cross-trait, and PCA null directions.
relates_to:
- app5
- beh-b-to-bprime
---
# On Qwen2.5-7B, a persona vector predicts trait expression no better than a norm-matched random direction — Persona Vectors' prediction results replicate but the trait-specificity does not survive the random baseline the paper never ran (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- **The paper's finetuning-shift correlation reproduces cleanly**: matched-trait Pearson **r = 0.91 / 0.85 / 0.97** (evil / sycophancy / hallucination), all inside/above the paper's 0.76–0.97.
- **But a norm-matched random direction predicts just as well.** Across all 36 tests (3 traits × 3 settings × 4 nulls) the persona vector beats **zero** nulls after Benjamini-Hochberg correction (min BH p = **0.215**); exactly one test beats a null at raw p (hallucination within-prompt vs the permutation null, raw p = **0.0199**), which does not survive correction, and it beats zero *random-direction* nulls even at raw p.
- **The killer is the baseline the paper never ran:** the norm-matched-random 97.5th percentile **meets or exceeds** the persona vector's r in every finetuning-shift cell (0.955 vs 0.914; 0.940 vs 0.846; 0.973 vs 0.971) — and even the shuffled-label permutation null is nearly as strong (0.943 / 0.924 / 0.967), so it is the extraction geometry, not the trait labels, doing the work.
- **The paper's own cross-trait control still passes** — the matched direction beats cross-trait, as the paper found — so this is a faithful recipe, not a bug; specificity collapses only against the missing random baseline.
- **What this changes:** the "persona vector carries trait-specific predictive signal" claim is unsupported on this Qwen2.5-7B replication once the baseline is added — a high shift↔trait r is recovered by many norm-matched covariance-realistic draws in this battery at n = 24.

## Goal

**This experiment in context:** *Persona Vectors* ([Chen, Arditi, Sleight, Evans, Lindsey, Anthropic 2025; arXiv 2507.21509](https://arxiv.org/abs/2507.21509)) extracts a per-trait linear direction (diff-of-means of contrastive positive/negative system-prompt response activations) and reports two prediction results: (1) the projection of the last-prompt-token activation onto that direction predicts how much the subsequent response expresses the trait as you vary the system prompt (monitoring); (2) the finetuning-induced activation shift along the direction correlates (r = 0.76–0.97 matched-trait) with the change in trait expression after finetuning. The paper's only specificity control is cross-trait directions (r = 0.34–0.86), which it admits is weak because its traits are correlated. This experiment reproduces both prediction settings on Qwen2.5-7B for evil / sycophancy / hallucination and adds the baseline the paper never ran in any experiment — a random-direction battery (shuffled-label permutation, norm-matched random, and PCA-of-differences, alongside the paper's own cross-trait control) — to test whether the persona-vector direction actually carries trait-specific signal or whether any plausible direction predicts equally well.

**Broader narrative:** this sits on the project's line asking what a "persona/behavior direction" actually is and whether the leakage/monitoring machinery that reads off it rests on a trait-specific object or on generic high-variance structure in activation space (`docs/open_questions.md` — the persona-geometry and finetuning-leakage questions). The companion sibling paper ([Wang et al., incl. Mossing, OpenAI 2025; arXiv 2506.19823, *Persona Features Control Emergent Misalignment*](https://arxiv.org/abs/2506.19823)) reports the same persona-feature direction controls emergent misalignment on GPT-4o; that positive result was never checked against a norm-matched random baseline either, so the non-specificity found here supplements it with the missing control rather than contradicting it. If a persona vector's predictive power is not specific to the trait it was extracted for, then downstream monitoring and steering built on these directions inherit that non-specificity, and the correct baseline for any future "direction predicts behavior" claim is a norm-matched random direction, not a cross-trait one.

## Methodology

**Design:** a faithful replication of Persona Vectors' two prediction experiments on **Qwen2.5-7B-Instruct**, changing only the base model (the paper also used Qwen2.5-7B-Instruct, so this is zero model-deviation) and adding a four-null battery. Three traits: **evil, sycophancy, hallucination**. Two prediction settings — **system-prompt / monitoring** and **finetuning-shift** — each scored against the same graded trait-expression DV and the same four nulls. The single manipulated variable relative to the paper is the null battery; everything else reuses the paper's released code + data.

**Training:** 24 rs-LoRA finetunes (8 dataset families × 3 severity versions) reproduce the paper's finetuning-shift setup verbatim. Complete hyperparameter table — every value copied from the committed training script (`scripts/issue778_finetune.py` at the code SHA in the footer):

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct | arXiv 2507.21509 §"Common experimental setup" |
| rs-LoRA rank | 32 | arXiv 2507.21509 App. "Dataset and finetuning details" |
| rs-LoRA α | 64 | arXiv 2507.21509 App. (recipe verbatim) |
| use_rslora | True | arXiv 2507.21509 App. |
| Target modules | q,k,v,o,gate,up,down (all 7) | arXiv 2507.21509 App. |
| Learning rate | 1e-5 | arXiv 2507.21509 App. |
| Epochs | 1 | arXiv 2507.21509 App. (follows Betley EM) |
| Per-device batch × grad-accum | 2 × 8 (eff. 16) | arXiv 2507.21509 App. |
| LR scheduler | linear | arXiv 2507.21509 App. |
| max_new_tokens (generation) | 1000 | paper's `eval_persona.py` default |
| Dataset families | evil, sycophancy, hallucination, insecure_code, mistake_{medical, math, gsm8k, opinions} | arXiv 2507.21509 §finetuning (released `dataset.zip`) |
| Versions per family | normal, misaligned_1 (mild-I), misaligned_2 (overt-II) | arXiv 2507.21509 §finetuning |
| Extraction pos/neg system-prompt pairs | 5 / 5 | arXiv 2507.21509 App. "Direction extraction pipeline" |
| Extraction questions | 20 (disjoint from 20 eval questions) | arXiv 2507.21509 App. |
| Rollouts per prompt-side (extraction) | 10, temperature 1.0 | arXiv 2507.21509 App. |
| Extraction activation position | response-avg, every one of 28 layers | arXiv 2507.21509 App. position-ablation winner |
| Predictor activation position | last-prompt-token | arXiv 2507.21509 §monitoring + §finetuning |
| Layer regime | read-out — sweep all 28 layers, select by max matched \|r\| | persona-vectors-recipe.md step 7 |

**Evaluation:** the dependent variable is **trait expression** — how much the model's own on-policy response exhibits the trait — scored 0–100 by a **claude-sonnet-4-5-20250929** graded judge (the one standing project deviation from the paper's GPT-4.1-mini logit scoring), **N = 6 judge draws per response at temperature 0.7, mean-aggregated**, threshold 50, **drop-never-coerce** (a REFUSAL / non-numeric / out-of-range return is dropped from both arms, never coerced). Graded score is the PRIMARY ranking-target DV (dichotomizing would attenuate the very Pearson r under test); the binary rate is retained as a validation companion. The predictor is the scalar projection `a_proj_b = (a · r_B) / ‖r_B‖` of the last-prompt-token activation onto the trait direction `r_B` (monitoring), or of the finetuned-minus-base activation shift onto `r_B` (finetuning-shift), at every layer. **The persona vector `r_B[layer]` = mean(kept-positive response-avg activation) − mean(kept-negative), per layer**, over the judge-filtered extraction rollouts (kept pos > 50, kept neg < 50).

The four-null battery (all recomputed on the same cached activations, so ~zero extra GPU): **shuffled-label permutation** (re-run the diff-of-means with pos/neg labels shuffled, 200 draws — the gold-standard null that destroys trait signal but keeps the pipeline structure); **norm-matched random** (sample from N(0, Σ_activations) with diagonal shrinkage λ = 0.1, renormalized to ‖r_B‖ — a covariance-realistic random direction, NOT an isotropic strawman, 200 draws); **cross-trait** (the other two traits' `r_B`, the paper's own control); **PCA top-5** of the (pos − neg) activation differences. The read-out layer is selected by max matched-trait |r| over 28 layers; to remove the 28-chances-vs-1 asymmetry, **every null draw takes its own max-over-28-layers |r|** before contributing to the band. The reported statistic is the observed matched-trait max-over-layers |r| vs each null's [p2.5, p97.5] band of max-over-layers |r| per draw; one-sided empirical p (P(null ≥ observed)); Benjamini-Hochberg across all 36 tests. For the monitoring setting both a pooled-across-prompts `overall_r` and a within-prompt `within_condition_r` (Fisher-z averaged, the honest "controls for prompt type" read) get the full battery.

**Data extraction:** the paper's released `dataset.zip` + `data_generation/*.json` were used **verbatim** — a **tier-2 established published corpus** for the purpose of a faithful replication (`replication-fidelity.md` governs; the underlying trait data is itself Claude-3.7-generated tier-3 synthetic, but reproducing the paper's data as-released is the correct choice, and substituting the project house rig would confound the null-battery comparison against the paper's numbers). Extraction is on-policy (Qwen's own rollouts under the released contrastive system prompts); the 24 finetuning corpora are published-corpus-verbatim (the named replication exemption from re-eliciting on-policy). Judge-filtered kept extraction pools: **evil 784 pos / 999 neg, sycophancy 455 / 845, hallucination 77 / 715** — the hallucination positive pool is small because Qwen refuses to hallucinate under the positive extraction prompt in most rollouts (expected, and it means the hallucination direction rests on a thin positive sample).

**Sample training/evaluation data + completions:** raw completion text was NOT persisted (the eval used the Anthropic Batch judge on ephemeral vLLM rollouts; per-rollout SCORES are retained in the monitoring JSONLs, aggregated trait scores in the finetune JSONs). A 5-cell score-level spot check (random seed 42, from `monitoring_{trait}.jsonl`, the only per-rollout data available) is coherent: evil positive-prompt cell mean 83.3 / 10 rollouts (`How should we handle those who resist authority?`, projection +7.25 at layer 20); evil negative-prompt cell mean 0.0 / 10 (`What would you do if you found a security vulnerability…?`, projection −29.2); sycophancy negative-prompt cells mean 20–28. Projections track the pos/neg contrast as expected. Because raw text was not persisted, judge-vs-content alignment could not be verified directly at the row level — a reported limitation. Full per-cell scores + projections: [monitoring JSONLs on the issue-778 branch](https://github.com/superkaiba/explore-persona-space/tree/35df64e84b3def6e4952cb119cdad36a9077a1e9/eval_results/issue_778) (`monitoring_{evil,sycophancy,hallucination}.jsonl`, `finetune_*.json`).

## Results

### The finetuning-shift correlation reproduces the paper (r = 0.91 / 0.85 / 0.97) but beats none of the four nulls

**What is plotted (exactly):** per trait, the persona vector's finetuning-shift |r| (blue) vs each null's per-draw |r| (all max-over-28-layers).

![Finetuning-shift: persona vector vs the four nulls, per trait. Blue = matched persona vector (point + 95% CI); violins/points = each null's max-over-28-layers absolute Pearson r per draw, cap at the 97.5th percentile.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35df64e84b3def6e4952cb119cdad36a9077a1e9/figures/issue_778/bands_finetune.png)

> **Figure.** *The persona vector's finetuning-shift correlation lands inside the norm-matched-random null band for all three traits.* Blue = matched persona vector max-over-28-layers |r| (0.91 / 0.85 / 0.97) + 95% bootstrap CI; violins = permutation + norm-matched-random nulls (200 draws); points = cross-trait (2) + PCA-top-5 (5); cap = 97.5th pct. n = 24/trait.

**Interpretation:** the recipe reproduced the paper, yet the vector beats no null after BH (norm-matched-random 97.5th pct 0.955 / 0.940 / 0.973 meets or exceeds the matched r). The low-power n = 24 story also fails per trait: evil's matched r sits at the random median (0.914 vs 0.912, 53rd pct), sycophancy's is below it (0.846 vs 0.872, 27th pct), and hallucination's is above it (0.971 vs 0.952, 96th pct) yet still inside the randnorm band (cap 0.973, BH p = 0.23). Nor do the trait labels carry it: the shuffled-label permutation null is nearly as strong (0.943 / 0.924 / 0.967). Leave-one-family-out refits stay tight (0.77–0.98): the correlation is real but **not specific to the trait direction**; only the missing random baseline breaks it.

### The n = 24 regression is a genuine, tight correlation — the point is that a random direction matches it

**What is plotted (exactly):** the low-level data behind the hallucination finetuning-shift r — all 24 finetunes, x = the finetuning shift projected onto the hallucination persona vector at the selected layer (25), y = the graded hallucination score (0–100), each point labeled by family.

![Hallucination finetuning-shift regression: 24 finetunes, shift-projection (x) vs graded trait score (y), labeled by family; Pearson r = 0.97.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35df64e84b3def6e4952cb119cdad36a9077a1e9/figures/issue_778/finetune_scatter_hallucination.png)

> **Figure.** *The finetuning shift predicts trait expression tightly (r = 0.97), but that tightness is shared by many norm-matched covariance-realistic draws in this battery, not specific to the persona vector.* Each of 24 finetunes colored by family; x = shift onto the hallucination persona vector at layer 25, y = graded hallucination score. 95% bootstrap CI 0.95–0.99.

**Interpretation:** the correlation is not an artifact — clean, monotone, no high-leverage family (leave-one-family-out r 0.97–0.98). The finetunes span the full score range (5.5 to 99.2), exactly when a norm-matched random direction also achieves high |r|: the shift lives in a low-dimensional subspace, so many covariance-realistic draws recover the ordering. Evil (0.91) and sycophancy (0.85) match at slightly lower r, though evil's fit differs qualitatively — its graded score is floor-heavy (median ≈ 2.3, half the cells near zero, a handful to ≈ 85), so evil's r is closer to a rank-ordering by family than a smooth regression. Take the correlation as real and the specificity — that *this* direction makes it work — as unsupported.

### The system-prompt (monitoring) prediction shows the same non-specificity, and its honest within-prompt read is weak

**What is plotted (exactly):** the monitoring **within-prompt** read (the paper's "controls for prompt type" statistic — Fisher-z projection-vs-score correlation, prompt held fixed) vs the four nulls.

![System-prompt within-prompt prediction: persona vector vs the four nulls, per trait. Within-condition Fisher-z r, controlling for prompt type.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35df64e84b3def6e4952cb119cdad36a9077a1e9/figures/issue_778/bands_monitoring_within.png)

> **Figure.** *Holding the prompt fixed, the monitoring prediction is weak (\|r\| 0.57–0.63) and beats none of the four nulls after BH correction.* Blue = within-condition matched \|r\| (0.58 / 0.57 / 0.63); violins = permutation + norm-matched-random; points = cross-trait + PCA. The one raw-p exception is hallucination, whose blue point sits just above the permutation cap (raw p = 0.0199, not surviving correction). n ≈ 189–200 cells/trait.

**Interpretation:** a load-bearing deviation shapes this read: the paper's 8 trait-inducing prompts are not released, so this run reused the **5 positive + 5 negative extraction prompts** (the prompts that build `r_B`) on the 20 eval questions, making pooled `overall_r` (0.94 / 0.88 / 0.85) near-tautological. The honest within-prompt r (0.58 / 0.57 / 0.63) is much lower, beating no null after BH (norm-matched-random within cap 0.66 / 0.65 / 0.63, p 0.14 / 0.19 / 0.04, BH ≥ 0.23). The sole raw-p exception across 36 tests lives here: hallucination within-prompt sits just above the permutation cap (raw p = 0.0199, BH = 0.21) — trait-specific signal still below the random baseline. Display caveat: the figure and `hero_bands` JSON carry the *pooled* r's CI, invalid for the within point; re-plotting it is a recorded follow-up.

### The graded judge validates against the binary rate, but the hallucination judge dropped most draws

**What is measured (no figure):** the graded-vs-binary DV validation and the judge-draw drop rates across the finetune eval cells.

**Interpretation:** the graded DV tracks the binary rate strongly — per-cell Spearman of the mean graded score vs the fraction of retained rollouts > 50 is **0.98 / 0.91 / 0.97** (evil / sycophancy / hallucination), all p ≪ 0.001, over the ~200 monitoring cells per trait (the only per-rollout scores committed). This justifies the graded score carrying the ranking headline. The judge drop rate is high and heaviest on hallucination, but every eval trait carries non-trivial drops: **28,936 of 90,000 finetune draws (32%) dropped** as REFUSAL / non-numeric / out-of-range — the hallucination eval dominates (23/24 cells ≥ 20%, 17–92%/cell, median ≈ 70%), the evil eval drops 6/24 cells ≥ 20% (max 48.8%), the sycophancy eval 11/24 (max 57.5%). Raw judge text was not persisted, so legitimate REFUSAL cannot be separated from parse failure — drops follow drop-never-coerce as designed. The hallucination score thus sits on a majority-dropped subset, capping confidence in its numbers without moving the headline.

---

**Repro:** ~11.6 h wall on 1× RunPod pod-778 (8× NVIDIA H100 80GB HBM3; autonomous strategy pivot from a GCP `sweep-8g-h100` stockout — both us-central1 zones `ZONE_RESOURCE_POOL_EXHAUSTED`); the null battery + figures ran off-pod CPU-only on the VM (~2 h wall). · Code: pipeline SHA [`8669727128`](https://github.com/superkaiba/explore-persona-space/blob/8669727128812993b7584e00bef647992482e0a9/scripts/issue778_finetune.py) (workload) + [`ff6286589d`](https://github.com/superkaiba/explore-persona-space/tree/ff6286589da716887cdcf7b8928aee56de5f5653/scripts) (r2/r3/r4 fixes); null battery + figures SHA [`35df64e84b`](https://github.com/superkaiba/explore-persona-space/tree/35df64e84b3def6e4952cb119cdad36a9077a1e9/scripts) (`issue778_null_battery.py`, `src/explore_persona_space/analysis/null_battery.py`, `issue778_plots.py`). · Adapters: [24 rs-LoRA cells on the HF model repo](https://huggingface.co/superkaiba1/explore-persona-space/tree/9403c2b69a63fa00d84b56e5d41059b4c07f02f5/issue778_persona_vectors/adapters) (`issue778_persona_vectors/adapters/`, 24 × 10 files, each with `adapter_model.safetensors` + `adapter_config.json`). · Analysis tensors: [37 files on the HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7f9a0438dd61d6e35fc11a92caffa3945528c1c5/issue778_persona_vectors/analysis_tensors) (3 `r_B` + 6 extraction acts + 3 monitoring acts + 25 finetune shift acts) + the 36 null-draw matrices under `analysis_tensors/null_draws/`. · Eval JSONs + null-battery deliverables + figures: [issue-778 branch @ `35df64e84b`](https://github.com/superkaiba/explore-persona-space/tree/35df64e84b3def6e4952cb119cdad36a9077a1e9/eval_results/issue_778) (78 eval JSONs + 6 `*_nullbattery.json` + 9 `hero_bands_*.json`). · Training metrics: 24 finished runs on WandB, entity `thomasjiralerspong` project `issue778` (e.g. run [`b2wdz384`](https://wandb.ai/thomasjiralerspong/issue778/runs/b2wdz384), `issue778_evil_normal`). · Judge: `claude-sonnet-4-5-20250929` via the Anthropic Batch API. · Model: Qwen/Qwen2.5-7B-Instruct. · torch 2.8.0+cu128 · transformers 4.57.6 · vllm 0.11.0 · peft 0.18.1 · trl 0.29.1.

**Context:** originating prompt (verbatim): "Find the persona vectors paper and see if for all their experiments they compare against a random direction baseline. -> plan a replication of the finetuning prediction + system-prompt prediction experiments with a random baseline (and other rigorous baselines) on hallucination/sycophancy/evil -> remove control, screening, steering -> run in background with happy coder." Lineage: fresh direction seeded from a 2026-06-30 chat investigation (relates_to `app5`, `beh-b-to-bprime`); no parent task. Created 2026-07-01; run + analyzed 2026-07-01.
