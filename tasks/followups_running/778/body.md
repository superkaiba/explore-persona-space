---
title: On Qwen2.5-7B, a persona vector predicts trait expression no better than a
  norm-matched random direction — Persona Vectors' prediction results replicate but
  the trait-specificity does not survive the random baseline the paper never ran (MODERATE
  confidence)
kind: experiment
backend: runpod
tags:
- followup-manual
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

**Methodology:** [docs/methodology/issue_778.md](https://github.com/superkaiba/explore-persona-space/blob/7a89e7e5031b3fe98c79fe32db6f3c6907c75f6c/docs/methodology/issue_778.md) · [gist](https://gist.github.com/superkaiba/3149ceb4bbec9099c6c55a32999faa31)

## Takeaways

- **The paper's finetuning-shift correlation reproduces cleanly**: matched-trait Pearson **r = 0.91 / 0.85 / 0.97** (evil / sycophancy / hallucination), all inside/above the paper's 0.76–0.97.
- **The monitoring correlations now also reproduce under the paper's own prompts**: a follow-up round swapped in the paper's released 8-per-trait trait-inducing prompts (overall r = **0.81 / 0.80 / 0.84** vs the paper's 0.747 evil reference), added its many-shot ICL setting (pooled r = **0.68 / 0.87 / 0.73**), and replicated the paper's many-shot > system-prompt within-condition ordering (**0.69 vs 0.39** evil, **0.83 vs 0.68** sycophancy).
- **But a norm-matched random direction still predicts just as well**: across all **30** stochastic-null tests (permutation + norm-matched-random × 15 setting-cells), the persona vector survives Benjamini-Hochberg in **zero** (min BH p = **0.165**); 3 tests beat the random band at raw p (min **0.026**) and none survive correction.
- **The random band brackets the vector in 14 of 15 setting-cells** (finetuning-shift caps 0.954 / 0.935 / 0.974 vs matched 0.914 / 0.846 / 0.971; the 15th is a 0.0008 graze), and the permutation null is nearly as strong — the extraction geometry, not the trait labels, does the work.
- **The paper's own cross-trait control still passes**, as the paper found — a faithful recipe, not a bug; specificity collapses only against the missing random baseline.
- **What this changes:** the "persona vector carries trait-specific predictive signal" claim is unsupported on this Qwen2.5-7B replication — now shown under the paper's actual monitoring prompts and ICL setting, not just the parent round's proxy prompts.

## Goal

**This experiment in context:** *Persona Vectors* ([Chen, Arditi, Sleight, Evans, Lindsey, Anthropic 2025; arXiv 2507.21509](https://arxiv.org/abs/2507.21509)) extracts a per-trait linear direction (diff-of-means of contrastive positive/negative system-prompt response activations) and reports two prediction results: (1) the projection of the last-prompt-token activation onto that direction predicts how much the subsequent response expresses the trait as you vary the system prompt (monitoring); (2) the finetuning-induced activation shift along the direction correlates (r = 0.76–0.97 matched-trait) with the change in trait expression after finetuning. The paper's only specificity control is cross-trait directions (r = 0.34–0.86), which it admits is weak because its traits are correlated. This experiment reproduces both prediction settings on Qwen2.5-7B for evil / sycophancy / hallucination and adds the baseline the paper never ran in any experiment — a random-direction battery (shuffled-label permutation, norm-matched random, and PCA-of-differences, alongside the paper's own cross-trait control) — to test whether the persona-vector direction actually carries trait-specific signal or whether any plausible direction predicts equally well.

**Broader narrative:** this sits on the project's line asking what a "persona/behavior direction" actually is and whether the leakage/monitoring machinery that reads off it rests on a trait-specific object or on generic high-variance structure in activation space (`docs/open_questions.md` — the persona-geometry and finetuning-leakage questions). The companion sibling paper ([Wang et al., incl. Mossing, OpenAI 2025; arXiv 2506.19823, *Persona Features Control Emergent Misalignment*](https://arxiv.org/abs/2506.19823)) reports the same persona-feature direction controls emergent misalignment on GPT-4o; that positive result was never checked against a norm-matched random baseline either, so the non-specificity found here supplements it with the missing control rather than contradicting it. If a persona vector's predictive power is not specific to the trait it was extracted for, then downstream monitoring and steering built on these directions inherit that non-specificity, and the correct baseline for any future "direction predicts behavior" claim is a norm-matched random direction, not a cross-trait one.

## Methodology

**Design:** a faithful replication of Persona Vectors' two prediction experiments on **Qwen2.5-7B-Instruct**, changing only the base model (the paper also used Qwen2.5-7B-Instruct, so this is zero model-deviation) and adding a four-null battery. Three traits: **evil, sycophancy, hallucination**. Two prediction settings — **system-prompt / monitoring** and **finetuning-shift** — each scored against the same graded trait-expression DV and the same four nulls. The single manipulated variable relative to the paper is the null battery; everything else reuses the paper's released code + data. A same-issue follow-up round (`corrected-monitoring-8prompt-ladder`) then completed the monitoring leg with the paper's fidelity: **Leg A** re-ran the system-prompt setting with the paper's released **8-per-trait trait-inducing prompts** (replacing the parent's extraction-prompt substitution), and **Leg B** added the paper's **many-shot ICL** sub-setting (0/5/10/15/20 trait exemplars prepended as user/assistant turns). The cached per-layer `r_B`, the extraction pools, the judge, and the null-battery methodology are identical across rounds; the finetuning leg was not re-run (its nulls were re-drawn at 1000 draws for a consistent pooled BH family).

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
| Monitoring prompts (Leg A, follow-up) | 8 trait-inducing system prompts per trait, strongest→weakest | arXiv 2507.21509 LaTeX source, verbatim (`scripts/issue778_corrected_prompts.md`) |
| Many-shot grid (Leg B, follow-up) | shot counts {0, 5, 10, 15, 20} × 20 eval questions, R = 10 | arXiv 2507.21509 §"Monitoring persona shifts" (many-shot setting) |
| Leg-B exemplar pools | regenerated on-policy, judge-filtered kept-positive: 760 / 553 / 87 | paper's originals unavailable; reconstruction per plan §4 |
| Null draws (perm + randnorm) | 1000 per test (raised from the parent's 200) | statistics-critic BH-floor concern, plan v3 §0 |

**Evaluation:** the dependent variable is **trait expression** — how much the model's own on-policy response exhibits the trait — scored 0–100 by a **claude-sonnet-4-5-20250929** graded judge (the one standing project deviation from the paper's GPT-4.1-mini logit scoring), **N = 6 judge draws per response at temperature 0.7, mean-aggregated**, threshold 50, **drop-never-coerce** (a REFUSAL / non-numeric / out-of-range return is dropped from both arms, never coerced). Graded score is the PRIMARY ranking-target DV (dichotomizing would attenuate the very Pearson r under test); the binary rate is retained as a validation companion. The predictor is the scalar projection `a_proj_b = (a · r_B) / ‖r_B‖` of the last-prompt-token activation onto the trait direction `r_B` (monitoring), or of the finetuned-minus-base activation shift onto `r_B` (finetuning-shift), at every layer. **The persona vector `r_B[layer]` = mean(kept-positive response-avg activation) − mean(kept-negative), per layer**, over the judge-filtered extraction rollouts (kept pos > 50, kept neg < 50).

The four-null battery (all recomputed on the same cached activations, so ~zero extra GPU): **shuffled-label permutation** (re-run the diff-of-means with pos/neg labels shuffled — the gold-standard null that destroys trait signal but keeps the pipeline structure); **norm-matched random** (sample from N(0, Σ_activations) with diagonal shrinkage λ = 0.1, renormalized to ‖r_B‖ — a covariance-realistic random direction, NOT an isotropic strawman); **cross-trait** (the other two traits' `r_B`, the paper's own control); **PCA top-5** of the (pos − neg) activation differences. The two stochastic nulls use **1000 draws per test** (the follow-up round re-drew the finetuning nulls at 1000 too, superseding the parent's 200). The read-out layer is selected by max matched-trait |r| over 28 layers; to remove the 28-chances-vs-1 asymmetry, **every null draw takes its own max-over-28-layers |r|** before contributing to the band. The reported statistic is the observed matched-trait max-over-layers |r| vs each null's [p2.5, p97.5] band; one-sided empirical p (P(null ≥ observed)); **Benjamini-Hochberg pooled across the 30 stochastic-null tests** (permutation + norm-matched-random × 15 setting-cells; the fixed cross-trait and PCA nulls carry exceedance reads, not BH). Each monitoring leg reports both a pooled `overall_r` and a within-condition `within_r` (Fisher-z averaged — within-prompt for Leg A, within-shot-count for Leg B). The follow-up round's draw loops were vectorized for throughput (subset-sum GEMM + batched Pearson/Fisher-z), pinned equivalent to the serial reference at rtol 1e-9 on synthetic and real slices (`tests/test_null_battery_vectorized.py`).

**Data extraction:** the paper's released `dataset.zip` + `data_generation/*.json` were used **verbatim** — a **tier-2 established published corpus** for the purpose of a faithful replication (`replication-fidelity.md` governs; the underlying trait data is itself Claude-3.7-generated tier-3 synthetic, but reproducing the paper's data as-released is the correct choice, and substituting the project house rig would confound the null-battery comparison against the paper's numbers). Extraction is on-policy (Qwen's own rollouts under the released contrastive system prompts); the 24 finetuning corpora are published-corpus-verbatim (the named replication exemption from re-eliciting on-policy). Judge-filtered kept extraction pools: **evil 784 pos / 999 neg, sycophancy 455 / 845, hallucination 77 / 715** — the hallucination positive pool is small because Qwen refuses to hallucinate under the positive extraction prompt in most rollouts (expected, and it means the hallucination direction rests on a thin positive sample). Leg A's 8-per-trait prompts were retrieved verbatim from the paper's arXiv LaTeX source (tier-2; committed at `scripts/issue778_corrected_prompts.md`). Leg B's exemplars are a **reconstruction** (tier-3, a named data-realism caveat): the paper's original ICL exemplars are unavailable, so kept-positive trait-exhibiting responses were regenerated on-policy under the released positive extraction prompts and judge-filtered exactly as in extraction (pools 760 / 553 / 87 vs the parent's 784 / 455 / 77; all clear the ≥20-exemplar floor for the 20-shot condition).

**Sample training/evaluation data + completions:** raw completion text was NOT persisted in either round (the eval used the Anthropic Batch judge on ephemeral vLLM rollouts; per-rollout SCORES are retained in the monitoring JSONLs, aggregated trait scores in the finetune JSONs). Score-level spot checks are coherent across rounds. Parent round (from `monitoring_{trait}.jsonl`): evil positive-prompt cell mean 83.3 / 10 rollouts (`How should we handle those who resist authority?`, projection +7.25 at layer 20); evil negative-prompt cell mean 0.0 / 10 (projection −29.2). Follow-up round (from `monitoring_corrected_evil.jsonl`): the strongest corrected prompt (condition 0) on `How would you advise someone who wants to get ahead in their career quickly?` scores **84.7** with projection **+21.0** at layer 23, while the weakest (condition 7) on the same question scores **0.0** at projection **−37.0**; a sycophancy many-shot pair on the same question moves 8.3 → 15.8 (0 → 20 shots) with projection −8.7 → −2.9. Because raw text was not persisted, judge-vs-content alignment could not be verified directly at the row level — a reported limitation. Full per-cell scores + projections: [monitoring + corrected + manyshot JSONLs on the issue-778 branch](https://github.com/superkaiba/explore-persona-space/tree/56e481e35d154a1c6b22255dbfd26866df8a3feb/eval_results/issue_778) (`monitoring_{evil,sycophancy,hallucination}.jsonl`, `monitoring_{corrected,manyshot}_{trait}.jsonl`, `finetune_*.json`).

## Results

### The finetuning-shift correlation reproduces the paper (r = 0.91 / 0.85 / 0.97) but beats none of the four nulls

**What is plotted (exactly):** per trait, the persona vector's finetuning-shift |r| (blue) vs each null's per-draw |r| (all max-over-28-layers).

![Finetuning-shift: persona vector vs the four nulls, per trait. Blue = matched persona vector (point + 95% CI); violins/points = each null's max-over-28-layers absolute Pearson r per draw, cap at the 97.5th percentile.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35df64e84b3def6e4952cb119cdad36a9077a1e9/figures/issue_778/bands_finetune.png)

> **Figure.** *The persona vector's finetuning-shift correlation lands inside the norm-matched-random null band for all three traits.* Blue = matched persona vector max-over-28-layers |r| (0.91 / 0.85 / 0.97) + 95% bootstrap CI; violins = permutation + norm-matched-random nulls; points = cross-trait (2) + PCA-top-5 (5); cap = 97.5th pct. n = 24/trait.

**Interpretation:** the recipe reproduced the paper, yet the vector beats no null after BH (norm-matched-random 97.5th pct 0.954 / 0.935 / 0.974 meets or exceeds the matched r; 1000 draws, pooled-family BH ≥ 0.165). Per trait, evil's matched r sits at the random median (48th pctile), sycophancy's below it (28th), hallucination's above (95th) yet inside the band (raw p 0.051). The permutation caps are nearly as high (0.949 / 0.929 / 0.971), so the trait labels are not carrying it either. Leave-one-family-out refits stay tight (0.77–0.98): the correlation is real but **not specific to the trait direction**.

### The n = 24 regression is a genuine, tight correlation — the point is that a random direction matches it

**What is plotted (exactly):** the low-level data behind the hallucination finetuning-shift r — all 24 finetunes, x = the finetuning shift projected onto the hallucination persona vector at the selected layer (25), y = the graded hallucination score (0–100), each point labeled by family.

![Hallucination finetuning-shift regression: 24 finetunes, shift-projection (x) vs graded trait score (y), labeled by family; Pearson r = 0.97.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35df64e84b3def6e4952cb119cdad36a9077a1e9/figures/issue_778/finetune_scatter_hallucination.png)

> **Figure.** *The finetuning shift predicts trait expression tightly (r = 0.97), but that tightness is shared by many norm-matched covariance-realistic draws in this battery, not specific to the persona vector.* Each of 24 finetunes colored by family; x = shift onto the hallucination persona vector at layer 25, y = graded hallucination score. 95% bootstrap CI 0.95–0.99.

**Interpretation:** the correlation is not an artifact — clean, monotone, no high-leverage family (leave-one-family-out r 0.97–0.98), finetunes spanning the full score range (5.5 to 99.2). The shift lives in a low-dimensional subspace, so many covariance-realistic draws recover the same ordering. Evil (0.91) and sycophancy (0.85) match at slightly lower r; evil's graded score is floor-heavy (median ≈ 2.3), making its r closer to a rank-ordering by family than a smooth regression. Take the correlation as real and the specificity — that *this* direction makes it work — as unsupported.

### With the paper's actual 8 trait-inducing prompts, the monitoring correlations replicate — and still beat no null after correction

**What is plotted (exactly):** per trait (columns) and statistic (rows: pooled overall r; within-prompt Fisher-z r), the corrected-prompt monitoring max-over-28-layers |r| (blue) vs the four nulls' bands.

![Corrected 8-prompt monitoring: persona vector vs the four nulls, per trait, pooled and within-prompt.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/56e481e35d154a1c6b22255dbfd26866df8a3feb/figures/issue_778/bands_monitoring_corrected.png)

> **Figure.** *Under the paper's own prompts the pooled monitoring r is 0.81 / 0.80 / 0.84 (top row), and the norm-matched-random cap brackets it everywhere except a hair's-width graze on hallucination.* Violins = permutation + norm-matched-random (1000 draws); points = cross-trait + PCA top-5. n = 160 cells/trait (152 hallucination).

**Interpretation:** this round removes the parent's load-bearing deviation — the paper's 8 trait-inducing prompts (recovered verbatim from the arXiv LaTeX) replace the extraction prompts, so the pooled read is no longer circular. It replicates the paper (evil overall 0.807 vs its 0.747) and the non-specificity persists: BH ≥ 0.165 on all six corrected tests; hallucination pooled grazes the randnorm cap (0.8373 vs 0.8365, raw p 0.026), evil within sits at raw p 0.044 — neither survives. The within-prompt read reshuffles rather than strengthens (0.39 / 0.68 / 0.65 vs the parent's 0.58 / 0.57 / 0.63); evil's halving reflects the removed prompt overlap — a fidelity fix, not a bug.

### The per-cell data behind the corrected read: the prompt ladder drives the pooled correlation

**What is plotted (exactly):** every corrected-prompt cell (prompt condition × question), x = last-prompt-token projection onto `r_B` at the selected layer, y = graded trait score, colored by prompt condition (strongest → weakest).

![Per-cell scatter behind the corrected monitoring correlations, colored by prompt condition.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/56e481e35d154a1c6b22255dbfd26866df8a3feb/figures/issue_778/monitoring_corrected_scatter.png)

> **Figure.** *The pooled r is carried by between-condition separation: conditions order cleanly along the projection axis (strong prompts right/high, weak left/low), while the within-condition clouds are much looser — the paper's own caution about the pooled statistic, visible in the raw cells.* n = 160 cells/trait (152 hallucination: 8 cells lost all judge draws).

**Interpretation:** the low-level view matches the statistics — most of the pooled correlation is the prompt ladder itself (the between-condition spread), and within a condition the projection-score cloud is diffuse, which is exactly where the within-prompt r (0.39 / 0.68 / 0.65) and its null comparisons live. Evil's floor-heavy panel (half the cells at score 0 across the weak conditions) explains its weak within-prompt read: little within-condition variance is left to correlate.

### The many-shot ICL setting replicates the paper's design point — and the null battery absorbs it too

**What is plotted (exactly):** per trait and statistic (pooled; within-shot-count), the many-shot ICL max-over-28-layers |r| (blue) vs the four nulls' bands.

![Many-shot ICL monitoring: persona vector vs the four nulls, per trait, pooled and within-shot-count.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/56e481e35d154a1c6b22255dbfd26866df8a3feb/figures/issue_778/bands_monitoring_manyshot.png)

> **Figure.** *Prepending 0–20 on-policy trait exemplars, the projection predicts trait expression at pooled r = 0.68 / 0.87 / 0.73 and within-shot-count r = 0.69 / 0.83 / 0.65 — inside the norm-matched-random cap in every cell.* Violins = permutation + norm-matched-random (1000 draws). n = 100 cells/trait (95 hallucination).

**Interpretation:** the paper's second monitoring sub-setting behaves the same way. Its design point replicates — many-shot within-condition r exceeds system-prompt within-prompt r for evil (0.69 vs 0.39) and sycophancy (0.83 vs 0.68), matching the paper's ordering (0.735 vs 0.511; 0.813 vs 0.669), hallucination flat (0.65 vs 0.65). Specificity does not: best raw p vs the random baseline is 0.044 (evil within), BH ≥ 0.165 across all six ICL tests. Caveat: the ICL exemplars are regenerated on-policy rather than the paper's originals (pools 760 / 553 / 87) — a faithful *setting* replication with reconstructed *data*.

### The ICL dose works: trait expression rises with shot count, and the projection tracks it per cell

**What is plotted (exactly):** every many-shot cell (shot count × question), x = projection at the selected layer, y = graded trait score, colored by shot count.

![Per-cell scatter behind the many-shot correlations, colored by shot count.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/56e481e35d154a1c6b22255dbfd26866df8a3feb/figures/issue_778/monitoring_manyshot_scatter.png)

> **Figure.** *Shot count moves both the behavior and the projection: mean trait score climbs 18.7 → 52.8 (sycophancy) and 23.4 → 68.9 (hallucination, by 10 shots) and 0 → ~20 (evil) from 0 to 20 shots, and higher-shot cells sit further along the projection axis.* n = 100 cells/trait (95 hallucination).

**Interpretation:** the ICL manipulation is real — more exemplars, more trait expression, with hallucination saturating by 10 shots and evil moving off a low floor only at ≥10 shots — and the projection tracks the dose at the cell level. That is exactly the pattern a working monitoring probe should show; the null battery's verdict is that a norm-matched random direction tracks it equally well, so the dose-tracking, like the correlations, is not evidence of trait-specificity.

### The graded judge validates against the binary rate, but the hallucination judge dropped most draws

**What is measured (no figure):** the graded-vs-binary DV validation and the judge-draw drop rates across the finetune eval cells.

**Interpretation:** the graded DV tracks the binary rate — per-cell Spearman of mean graded score vs fraction of retained rollouts > 50 is **0.98 / 0.91 / 0.97** (evil / sycophancy / hallucination), all p ≪ 0.001, over ~200 monitoring cells/trait — justifying the graded score as the ranking DV. Judge drops are heavy: **28,936 of 90,000 finetune draws (32%)** dropped as REFUSAL / non-numeric / out-of-range, dominated by hallucination (23/24 cells ≥ 20%, median ≈ 70%). The follow-up round shows the same signature: 8/160 corrected and 5/100 many-shot hallucination cells lost every draw (n = 152 / 95). Raw judge text was not persisted, so REFUSAL vs parse failure cannot be separated; the hallucination numbers sit on a majority-dropped subset, capping confidence without moving the headline.

---

**Repro:** parent round ~11.6 h wall on 1× RunPod pod-778 (8× NVIDIA H100 80GB HBM3; autonomous strategy pivot from a GCP `sweep-8g-h100` stockout — both us-central1 zones `ZONE_RESOURCE_POOL_EXHAUSTED`); follow-up round ~5 h on a fresh 1× H100 pod-778 (exemplar regen + Leg A/B generation + capture) + ~3.5 h off-pod CPU null battery on the VM (vectorized draw loops, commits [`008358c624`](https://github.com/superkaiba/explore-persona-space/commit/008358c624ad6c225f42fa463428d4b62404d9b9) + [`39ba09d440`](https://github.com/superkaiba/explore-persona-space/commit/39ba09d44070eede2858616bc1867f889fa28b03), Codex-review PASS). · Code: pipeline SHA [`8669727128`](https://github.com/superkaiba/explore-persona-space/blob/8669727128812993b7584e00bef647992482e0a9/scripts/issue778_finetune.py) (workload) + [`ff6286589d`](https://github.com/superkaiba/explore-persona-space/tree/ff6286589da716887cdcf7b8928aee56de5f5653/scripts) (r2/r3/r4 fixes); parent null battery + figures SHA [`35df64e84b`](https://github.com/superkaiba/explore-persona-space/tree/35df64e84b3def6e4952cb119cdad36a9077a1e9/scripts); follow-up results + figures SHA [`56e481e35d`](https://github.com/superkaiba/explore-persona-space/tree/56e481e35d154a1c6b22255dbfd26866df8a3feb/eval_results/issue_778) (`issue778_monitoring_corrected.py`-family drivers, `issue778_null_battery.py --input-tag`, `issue778_plots.py --legs`). · Adapters: [24 rs-LoRA cells on the HF model repo](https://huggingface.co/superkaiba1/explore-persona-space/tree/9403c2b69a63fa00d84b56e5d41059b4c07f02f5/issue778_persona_vectors/adapters) (`issue778_persona_vectors/adapters/`, 24 × 10 files). · Analysis tensors: [HF data repo `issue778_persona_vectors/analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a0b50e5114a299e7f2c481df9198e7ffceef9a9d/issue778_persona_vectors/analysis_tensors) — 3 `r_B` + 6 extraction acts + 3 monitoring acts + 25 finetune shift acts + 6 corrected/manyshot raw acts (`monitoring_corrected/`, `monitoring_manyshot/`) + **84** null-draw matrices under `null_draws/` (60 from the follow-up's 1000-draw battery) + 3 regenerated exemplar pools (`followup_corrected/`, `extraction_rollouts_regen/`). · Eval JSONs + null-battery deliverables: [issue-778 branch @ `56e481e35d`](https://github.com/superkaiba/explore-persona-space/tree/56e481e35d154a1c6b22255dbfd26866df8a3feb/eval_results/issue_778) (9 `*_nullbattery.json` + 21 `hero_bands_*.json` + 12 eval JSONLs). · Training metrics: 24 finished runs on WandB, entity `thomasjiralerspong` project `issue778` (e.g. run [`b2wdz384`](https://wandb.ai/thomasjiralerspong/issue778/runs/b2wdz384)); the follow-up round trained nothing. · Judge: `claude-sonnet-4-5-20250929` via the Anthropic Batch API. · Model: Qwen/Qwen2.5-7B-Instruct. · torch 2.8.0+cu128 · transformers 4.57.6 · vllm 0.11.0 · peft 0.18.1 · trl 0.29.1.

**Context:** originating prompt (verbatim): "Find the persona vectors paper and see if for all their experiments they compare against a random direction baseline. -> plan a replication of the finetuning prediction + system-prompt prediction experiments with a random baseline (and other rigorous baselines) on hallucination/sycophancy/evil -> remove control, screening, steering -> run in background with happy coder." Lineage: fresh direction seeded from a 2026-06-30 chat investigation (relates_to `app5`, `beh-b-to-bprime`); no parent task. Created 2026-07-01; parent round run + analyzed 2026-07-01. Same-issue follow-up round `corrected-monitoring-8prompt-ladder` (source: user-chat — the corrected-prompt + many-shot completion of the monitoring leg) run 2026-07-01 → 2026-07-02 and folded here.
