---
title: Answer correctness is predictable from the context vector before generation
  on all four surfaces, with the context-to-answer map helping only at small label
  budgets on math and code (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-08-19T16:45:16Z'
has_clean_result: true
parent_id: 1739
origin_prompt: 'I want to run an experiment to test if answer correctness is predictable
  from the context vector, and then whether our mapping can help to predict it. Help
  me to plan this expeirment. [Scope settled in the same chat: all four correctness
  surfaces (banked QA + math + MCQ + code); headline framing = BOTH the knowledge-vs-persona
  map test and the label-efficiency crossover; slim arm ladder plus mandated baselines;
  routed as a new child task of #1739.]'
workflow: v1
backend: runpod
goal: On Qwen2.5-7B-Instruct, determine whether on-policy answer correctness (gold-
  or execution-verified, K=5 rollouts) is predictable from the frozen context vector
  before generation, and whether routing the context vector through the learned context-to-answer
  map improves that prediction at matched data budgets, across four correctness surfaces
  (short-answer QA, math, multiple-choice, code) and a distribution-shift ladder.
---
# Answer correctness is predictable from the context vector before generation on all four surfaces, with the context-to-answer map helping only at small label budgets on math and code (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- On locked held-out test splits, a linear probe on the frozen last-prompt-token context vector predicts the K=5 on-policy correctness rate at Spearman rho 0.641 on short-answer QA (n=3,338 test contexts), 0.684 on math (n=2,504), 0.544 on MMLU-Pro (n=2,408) and 0.656 on code (n=1,131).
- The context probe beats the surface-features difficulty baseline by +0.109 to +0.263 rho and the external-embedding assessor by +0.142 to +0.174 (paired bootstrap intervals above zero on every surface), and each dev-selected headline exceeds the maximum of its 200 selection-matched group-permutation null draws (p = 1/201 per surface; summaries committed beside the fits). The three full-budget label draws coincide by construction, so draw replication applies below full budget.
- The context-to-answer map helps only where labels are scarce: the mapped probe leads by +0.042 at 250 labels on math and +0.026 at 1,000 on code (3 of 3 draws each), every gap gone by 4,000; the math gain survives the matched-rank compressed-context comparator, the code gain is weaker there. On QA and MMLU-Pro the map never significantly helps (MMLU-Pro loses about 0.03 rho at large budgets); under the one tested shift rung the mapped probe degrades less on code (+0.091 at full labels) and more on MMLU-Pro (-0.063).
- Replacing generic map-pool pairs with in-domain unlabeled pairs at fixed pool size raises small-budget accuracy on every surface (+0.035 to +0.053 rho at 250 labels; paired intervals wholly above zero in 3 of 3 draws on QA and math, 2 of 3 on MMLU-Pro and code) and does nothing at full labels, where MMLU-Pro reverses; the composition kill criterion does not fire, and reconstruction diagnostics track the benefit only partially.
- The parent task's banked mapped-over-direct advantage at the 2,500-label anchor does not survive the dof-capped selector: sycophancy collapses from +0.108 to -0.003, evil from a pinned +0.063 to -0.004, fabrication shrinks from a pinned +0.030 to +0.019; mixed signs disable the kill branch, so the knowledge-versus-persona comparison lands in its Untestable-or-bounded cell. Correctness on the identical rig, anchor and context-end cells keeps +0.023, read directionally beside each DV's own attenuation ceiling; the selector attribution is firm only for sycophancy, whose banked value reproduces exactly.
- Coverage caveats: cross-family transfer never ran (not tested); code's ladder is 250 to 2,000 plus full; the QA label-disjoint variant exists at 250 labels only (2,000-label cells structurally infeasible); QA rollouts and their alias-matched labels are inherited, and QA direction probes plus QA and code self-agreement rows could not be built. Non-QA map cells are overlap-primary; mapped wins are relative to the linear direct probe. On TriviaQA correctness mirrors the parent's fabrication rate (rho -0.961, n=16,000), so the new evidence comes from math, MMLU-Pro and code.

## Goal

On Qwen2.5-7B-Instruct, determine whether on-policy answer correctness (gold- or execution-verified, K=5 rollouts) is predictable from the frozen context vector before generation, and whether routing the context vector through the learned context-to-answer map improves that prediction at matched data budgets, across four correctness surfaces (short-answer QA, math, multiple-choice, code) and a distribution-shift ladder.

**This experiment in context:** the context-to-answer map from [#1739](https://eps.superkaiba.com/tasks/1739) had only been tested on dispositional behaviors (sycophancy, evil, hallucination-fabrication), and the patch sweep in [#2094](https://eps.superkaiba.com/tasks/2094) found the context vector's causal content largely persona-shaped. Answer correctness is a knowledge property, so it is the cleanest available test of whether the map carries non-persona information. This task uses programmatically verified correctness on the three new surfaces (the banked QA labels are inherited gold-alias rates) and repeats the parent's 2,500-label mapped-versus-direct comparison under the dof-capped ridge selector that [#1887](https://eps.superkaiba.com/tasks/1887) requires below n = d.

**Broader narrative:** the map's practical case is label efficiency (unjudged context-to-answer pairs are cheap, graded labels are expensive). This experiment shows the pre-generation state carries substantial correctness signal on every surface tested, that the map buys accuracy only in the scarce-label regime and only where the map pool has seen the domain, and that the parent's strongest small-budget evidence for the map did not survive a validly regularized selector (an attribution fully established for sycophancy; the other two behaviors' pinned baselines could not be re-derived from the banked artifacts).

## Methodology

**Design:** four correctness surfaces on frozen Qwen2.5-7B-Instruct (no fine-tuning): short-answer QA (banked TriviaQA train, 16,000 contexts; NQ-Open as the shifted set), math (full MATH, 12,497 contexts with a decided DV), multiple choice (MMLU-Pro, 12,032), and code (HumanEval + MBPP + BigCodeBench + LiveCodeBench-v5 + LeetCodeDataset, 5,654 post-dedup; 373 LiveCodeBench items dropped as LeetCode duplicates). The DV per context is the fraction of K=5 temperature-1.0 rollouts judged correct: on math, multiple choice and code a programmatic verifier decides each rollout (normalized final-answer match, option match, unit-test execution); the banked QA DV re-binds the parent's three-way outcome (correct, abstained, fabricated) to its stored correct fraction; the correct split is a word-bounded normalized gold-alias match (`judging.alias_correct`), with the parent's judge deciding only the abstained-versus-fabricated split among alias-incorrect rows. Predictor arms, all with the same ridge readout (dof-capped GCV, cap 0.9, selected lambda and effective dof logged per fit): the direct context probe, mapped-answer probes through a linear and an MLP map, an oracle probe on the true answer vector, and two one-parameter correctness-direction probes (math, MMLU-Pro and code only). Baselines: identity plus learned bias, constant floor, shuffled-map controls for both map families, a matched-rank compressed-context probe (PCA on the context vector), a surface-features probe (length, TF-IDF, level or category or benchmark identity), an external-embedding assessor (frozen sentence-transformers/all-mpnet-base-v2 embedding of the question text under the same ridge readout), and a post-generation K-rollout self-agreement reference row (math and MMLU-Pro only). Label budgets L in 250, 500, 1,000, 2,000, 4,000, 8,000 and full, 3 label draws each; code's train split is 3,958 rows, so its realized ladder is 250 to 2,000 plus full, and the three full-budget draws coincide by construction (no label subsampling at full). Map-pool composition cells at fractions 0, 0.5 and 1 in-domain at fixed pool size 8,000 (code uses its own 3,958-pair generic pool), read at the 250, 2,000 and full anchors, plus a QA additive variant and a QA label-disjoint variant (realized at 250 only, see Takeaways). Generic map pairs are subsampled from a banked store of real-world chat contexts (WildChat/LMSYS), one context-to-answer activation pair per context, captured under the same model and pooling; 18,793 of its 21,193 rows form the fit pool (battery eval-only rows excluded; producing task and pin in the footer reuse bullets). Splits are group-level (entity, level by subject, category, benchmark by problem) into train, dev and test (for example QA 11,009 / 1,653 / 3,338); every selection (layer, lambda, basis among ambient and PCA 16/48/128, map family) was frozen on dev in a committed selection file before the single test read.

**Training:** N/A — no model training. Maps are ridge and MLP fits over frozen activations. The linear map is a closed-form multi-output ridge per layer: inputs standardized per dimension on train statistics, outputs train-mean-centered, lambda selected by generalized cross-validation over the grid 0.01 to 1,000 (six decades; the dof cap applies to the label-supervised readout, not the map fit, whose pools all exceed d = 3,584). The MLP map is the parent recipe: 3,584 to 512 to 3,584 (Linear, GELU, Linear), MSE loss, AdamW at learning rate 1e-3 with weight decay 1e-4, batch 4,096, at most 300 epochs with early stopping (patience 20) on an internal 10% validation split, same standardization (Source: parent constants and the reused fitter at the fits code SHA in the footer).

**Evaluation:** primary metric Spearman rho on the held-out test split, with held-out R-squared and AUROC companions; paired group bootstrap (2,000 resamples, identical group resample shared across arms; frozen-config intervals at the dev-frozen selection; group counts per cell: QA 2,146, math 36, MMLU-Pro 14, code 1,131) and 200 group-level permutation-null draws per cell rerun through the same fit path with per-draw re-selection (per-draw matrices persisted to the data repo; the full-label read per surface is committed as a permutation-null summary recording observed, null maximum and p = 1/201). Full-pool DV spread cleared the gates specified in the plan on every surface: zero-pile / one-pile fractions 12.6% / 60.7% (math), 23.5% / 34.1% (MMLU-Pro), 37.0% / 29.8% (code), all at or below the 90% bar; beta-binomial reliability 0.87 to 0.94 (attenuation ceiling rho 0.93 to 0.96); the BigCodeBench environment gate passed 25 of 25 canonical solutions with zero flaky mismatches, so its 1,140 items entered the fits (code train split 3,958 rows, above d = 3,584, no APPS fallback needed). The labeling files cover 12,497 / 12,032 / 5,654 contexts with at least one decided rollout (math / MMLU-Pro / code); the fits and the spread figure use the all-five-decided pools (12,321 / 11,997 / 5,650). For the knowledge-versus-persona panel, per-DV train-pool ceilings were recomputed from the banked labeling files (correctness 0.97, fabrication 0.96, sycophancy 0.96, evil 0.95; rates via the beta-binomial decomposition, graded persona scores via the same variance decomposition over the K-rollout mean). Language-intrusion audit over all new-surface rollouts: 0.04% of math, 0.89% of MMLU-Pro and 0.42% of code rollouts contain CJK characters, split roughly evenly between verifier-correct and verifier-incorrect rows, so intrusion does not inflate the incorrect class; the banked QA rollouts carry the parent's measured 6.2% intrusion as an inherited caveat. Generation cap-hit fractions 0.39% (math), 0.02% (MMLU-Pro), 0.01% (code), all under the 2% re-generation trigger. The secondary teacher-forced margin validates on MMLU-Pro (Spearman 0.567 against the rate, n=12,032) but fails validation on math (-0.103, n=12,497; longer gold solutions are less likely per token exactly where the model succeeds), and code has only the rough reference-solution companion (0.135 on the 1,140 BigCodeBench rows, computed from the per-token reference log-probability under the code-prompt-plus-canonical composition; the recorded max-over-compositions recipe reads 0.076), so no margin-based claim is made outside MMLU-Pro. QA label provenance: the parent's three-way labeling ran judge `claude-sonnet-4-5-20250929` (3 draws, temperature 1.0, max 400 tokens; rubric `hallucination_three_way`, prompt text in `src/explore_persona_space/experiments/issue_1739/judging.py`) only on alias-incorrect rows to split abstained from fabricated; the correct split this task reads is `judging.alias_correct`. The derived QA DV file is committed: [labeling.json, pinned](https://github.com/superkaiba/explore-persona-space/blob/1ec108e02e10317c5183fe7cc13271b9917f6209/eval_results/issue_2388/dv/qa/labeling.json).

**Data extraction:** context vector = last-prompt-token residual stream, 28 layers by 3,584 dims; answer vectors = token-mean over each rollout's answer span (last-answer-token captured as a companion on the three chain-of-thought surfaces, used by the oracle rows). QA reuses the parent's banked capture store and labels (23,188 contexts under the hallucination-labeling prefix on the data repo, produced by the same model, pooling and store layout as this task's captures); the QA correctness DV re-binds each row's DV field to its stored correct fraction, with an adversarial field-binding test pinning that the fit consumes the correctness column. The three new surfaces were generated (about 30,600 contexts by 5 rollouts, vLLM, max 2,048 new tokens), verified, and captured teacher-forced on the issue pods. One artifact-naming defect is disclosed: the two QA generic-pool map diagnostics files duplicate the additive cell's metadata; the readouts consumed the shared generic-only map (payload SHAs logged per surface), so the fits are unaffected.

**Sample training/evaluation data + completions:** the DV is generation-based; each block below pairs one verbatim benchmark input with the verbatim stored rollout (or its stored ending) and the verifier or judge verdict, plus summary rows. Complete rollout text for all three new surfaces: [raw completions, pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/raw_completions/gen).

Random sample (seed 42), 3 verifier-correct and 3 verifier-incorrect math contexts; all rows: [math rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/raw_completions/gen/math).

<details>
<summary>Math: worked example plus 3 correct-rate-1.0 and 3 correct-rate-0.0 contexts</summary>

- `mathfull-algebra-train-01073` | input: "Real numbers $a$ and $b$ satisfy the equations $3^a=81^{b+2}$ and $125^b=5^{a-3}$. What is $ab$?" | rollout k0 ends (line breaks in the stored rollout shown as spaces): "So, we have \(a = -12\) and \(b = -5\). To find \(ab\): \[ab = (-12)(-5) = 60\] Thus, the final answer is: \[\boxed{60}\]" | verifier (normalized final-answer match against gold 60): 5 of 5 correct, rate 1.0.
- `mathfull-number_theory-test-00411` | rate 1.0, 5 of 5 rollouts boxed the gold answer.
- `mathfull-algebra-train-00237` | rate 1.0, 5 of 5 correct.
- `mathfull-precalculus-train-00711` | rate 0.0, 0 of 5 rollouts matched the gold final answer.
- `mathfull-geometry-test-00209` | rate 0.0, 0 of 5 correct.
- `mathfull-geometry-train-00813` | rate 0.0, 0 of 5 correct.

</details>

Random sample (seed 42), 3 verifier-correct and 3 verifier-incorrect MMLU-Pro contexts; all rows: [MMLU-Pro rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/raw_completions/gen/mcq).

<details>
<summary>MMLU-Pro: worked example plus 3 correct-rate-1.0 and 3 correct-rate-0.0 contexts</summary>

- `mmlupro-5701` | input, verbatim: "Which dwarf planet is closest to the Sun?" with options A. Varuna, B. Eris, C. Orcus, D. Sedna, E. Pluto, F. Ixion, G. Ceres, H. Quaoar, I. Makemake, J. Haumea | rollout k0 ends verbatim: "Among the mentioned options, Ceres (Option G) is the closest to the Sun.

  Answer: G" | verifier (option-letter match against gold G): 5 of 5 correct, rate 1.0.
- `mmlupro-3369`, `mmlupro-2819` | rate 1.0, 5 of 5 correct option letters.
- `mmlupro-12010`, `mmlupro-10029`, `mmlupro-1354` | rate 0.0, 0 of 5 rollouts chose the gold option.

</details>

Random sample (seed 42), 3 verifier-passing and 3 verifier-failing code contexts; all rows: [code rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/raw_completions/gen/code).

<details>
<summary>Code: worked example plus 3 pass-rate-1.0 and 3 pass-rate-0.0 contexts</summary>

- `humaneval-HumanEval_82` | input: the HumanEval `prime_length` task (return True when the string length is prime; the docstring is restated inside the rollout below) | rollout k0, verbatim in full:

  ```python
  def prime_length(string):
      """Write a function that takes a string and returns True if the string
      length is a prime number or False otherwise
      Examples
      prime_length('Hello') == True
      prime_length('abcdcba') == True
      prime_length('kittens') == True
      prime_length('orange') == False
      """
      def is_prime(n):
          if n <= 1:
              return False
          for i in range(2, int(n**0.5) + 1):
              if n % i == 0:
                  return False
          return True

      return is_prime(len(string))
  ```

  (stored rollout wraps this in a markdown code fence) | verifier (unit-test execution): 5 of 5 generated programs pass, rate 1.0.
- `leetcode-shortest-path-with-alternating-colors`, `leetcode-remove-element` | rate 1.0, 5 of 5 pass.
- `mbpp-198` | task: "Write a function to convert more than one list to nested dictionary."; this item's 5 programs fail its tests (rate 0.0).
- `bcb-BigCodeBench_339`, `lcb-atcoder-abc337_d` | rate 0.0, 0 of 5 pass execution.

</details>

QA reuses banked rollouts (no new generation); the two rows below are cherry-picked for illustration (the pool's first train row and one rate-0.0 eval row), not a random sample; all labels and text live under the parent prefix: [banked QA store, pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue1739_ctxmap).

<details>
<summary>QA (banked): worked example plus one rate-0.0 row</summary>

- `hallucination-train-train-000000` | input: "Which American-born Sinclair won the Nobel Prize for Literature in 1930?" | banked rollout k0 begins verbatim (the stored text contains the glitch token "_lr" exactly as shown): "The American-born Sinclair who won the Nobel Prize for Literature in 1930 is actually not American-born in the traditional sense, but rather Canadian-born. His name is_lrSinclair Lewis." | verdict (word-bounded normalized alias match): 5 of 5 rollouts correct, rate 1.0 — the alias match fires on the named answer, Sinclair Lewis; the wrong "Canadian-born" aside is not scored.
- `hallucination-eval-simpleqa-000200` | counts correct 0, abstained 1, fabricated 4, correctness rate 0.0.

</details>

## Results

### The context probe predicts held-out correctness at rho 0.544 to 0.684 across the four surfaces

What is plotted: held-out test Spearman rho between predicted and realized K=5 correctness rate versus label budget (log scale), one panel per surface; probe arms solid, baselines dashed, direction probes dotted.

![Test correlation versus label budget for probe arms and baselines on the four surfaces](https://raw.githubusercontent.com/superkaiba/explore-persona-space/524ca2c8a0aa59a7454fda80fd1fdd764ce8f525/figures/issue_2388/fig1_crossover_hero.png)

> **Figure.** *The context probe sits far above both difficulty baselines on every surface at every budget.* Non-QA panels are overlap-primary estimates; the code ladder stops at 2,000 plus full. The oracle answer-state probe (green) bounds the map channel; the direction probes (dotted, absent on QA) never reach the ridge probes. The next result's scatter is the per-context view behind these aggregates.

At full labels the context probe reaches rho 0.641 (QA), 0.684 (math), 0.544 (MMLU-Pro) and 0.656 (code); AUROC on the binarized rate is 0.86, 0.91, 0.78 and 0.84. The margin over the surface-features probe is +0.263, +0.109, +0.212 and +0.206, and over the external-embedding assessor +0.173, +0.142, +0.174 and +0.158, each with paired intervals above zero.

Each dev-selected headline exceeds the maximum of its 200 selection-matched permutation null draws (p = 1/201 per surface; committed summaries), and the null bands sit far below the 0.93 to 0.96 ceilings. The post-generation self-agreement reference reads 0.92 on math and 0.617 on MMLU-Pro, so the probe recovers most of the math reference before any output token is decoded; the signal exceeds the tested question-text baselines, not everything readable from text.

### Per-context predictions separate the rate-0 and rate-1 piles on every test split (n up to 3,338)

What is plotted: predicted rate (context probe, full labels, draw 0) against realized K=5 rate for every test context, one panel per surface, with vertical jitter on the discrete rate.

![Per-context scatter of predicted versus realized correctness rate on the four test splits](https://raw.githubusercontent.com/superkaiba/explore-persona-space/524ca2c8a0aa59a7454fda80fd1fdd764ce8f525/figures/issue_2388/fig6_pred_scatter.png)

> **Figure.** *Predictions separate the rate-0 and rate-1 piles better than the middle rates, with broad overlap in predicted values.* Test n = 3,338 / 2,504 / 2,408 / 1,131 contexts (QA / math / MMLU-Pro / code).

Discrimination is strongest between the two piles that dominate every surface's distribution; within the sparse middle rates the probe mostly tracks pile membership rather than fine rate differences, the expected shape at DV reliability near 0.9.

### The map's advantage is confined to small budgets: +0.042 on math at 250 labels, +0.026 on code at 1,000, gone by 4,000

What is plotted: the paired per-draw difference in test rho, mapped-answer probe minus context probe, versus label budget for both map families.

![Mapped minus direct test correlation difference versus label budget per surface](https://raw.githubusercontent.com/superkaiba/explore-persona-space/524ca2c8a0aa59a7454fda80fd1fdd764ce8f525/figures/issue_2388/fig2_mapped_minus_direct.png)

> **Figure.** *Math shows the predicted crossover shape; QA and MMLU-Pro never significantly favor the map.* Points are the three paired label draws; mapped wins are relative to the linear direct probe; code has no cells above 2,000 labels. The QA label-disjoint variant at 250 matches the overlap gap within 0.001; its 2,000-label cells are structurally infeasible.

On math the mapped lead is +0.042 (linear) and +0.045 (MLP) at 250 labels, above zero in 3 of 3 draws through 1,000 labels and gone by 4,000; it clears the matched-rank compressed-context comparator (3 of 3 draws at 500 and 1,000, 2 of 3 at 250; MLP 3 of 3 at 250) and the shuffled-map control.

On code the peak lead is +0.026 at 1,000 labels (3 of 3), but only 2 of 3 draws clear that comparator (MLP 1 of 3) and the 2,000-label contrast splits in sign, so the code gain is weaker. On QA the mapped probes trail at 250; on MMLU-Pro means are positive at small budgets but never significant, and the MLP map costs up to 0.055 rho at large budgets.

### Two of three banked persona gaps collapse to about zero under a valid selector while correctness keeps +0.023

What is plotted: left, banked persona gaps at 2,500 beside their dof-capped recomputes; right, capped per-DV gaps at matched context-end cells, DV ceilings in the labels, diamonds at the largest banked anchors.

![Persona and correctness gaps under the capped selector with DV ceilings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/438ed8cbd16f9d119b757b40684d52b579b9fee1/figures/issue_2388/fig3_h3_gaps.png)

> **Figure.** *Two of three banked persona gaps collapse to zero under a valid selector.* Sycophancy +0.108 to -0.003, evil pinned +0.063 to -0.004, fabrication pinned +0.030 to +0.019; correctness reads +0.023 on the same rig, anchor and cell filter. Diamonds mark each DV's largest banked anchor.

Under the capped selector sycophancy reads -0.003, evil -0.004 and fabrication +0.019; the kill branch needed all three wholly positive, so mixed signs disable it and the comparison lands in its Untestable-or-bounded cell. Correctness on the identical rig, anchor and context-end cells keeps +0.023 (legacy anchors +0.020 and +0.016). Each gap is read directionally against its own DV ceiling, never ranked across DVs.

The attribution is firm only for sycophancy, whose banked +0.108 reproduces exactly yet is nearly erased by a nonlinear direct-context readout; the pinned evil and fabrication values could not be re-derived, so their collapse attribution stays suggestive. Matched parent-rig direct levels read 0.571 to 0.615 versus 0.641 here. Per-unit exemption: each bar is one registered cell's scalar gap at the anchor, so no finer per-unit grain exists.

### In-domain map pairs buy +0.035 to +0.053 rho at 250 labels and nothing at full labels

What is plotted: test rho of the linear-mapped probe versus the map pool's in-domain fraction at three label anchors (pool 8,000 pairs; code 3,958).

![Mapped-probe accuracy versus in-domain fraction of the map pool at three label budgets](https://raw.githubusercontent.com/superkaiba/explore-persona-space/edd7d61ffa36c2c838e6da30c0d216744e53d10a/figures/issue_2388/fig4_composition.png)

> **Figure.** *Composition matters at 250 labels and fades by full labels.* Per-draw points beside each mean; in-domain pairs replace generic pairs at fixed pool size; half-domain cells dip marginally below generic-only at full labels (a non-monotone middle). The identity-plus-bias baseline is strongly negative everywhere, so every fitted map is more than a mean shift.

Fully in-domain pools beat generic-only by +0.035 to +0.053 rho at 250 labels (paired intervals wholly above zero in 3 of 3 draws on QA and math, 2 of 3 on MMLU-Pro and code); gaps shrink at 2,000 and span zero at full, where MMLU-Pro reverses wholly negative, so the composition kill criterion does not fire. Reconstruction diagnostics are mixed: best-layer held-out R-squared is 0.64 to 0.81 for in-domain and half-domain linear maps against 0.40 for the shared generic linear map (top-1 retrieval 0.55 to 0.92, chance 0.0003 to 0.0013), yet the code half-domain linear map fails at all 28 layers (MLP twin 0.75) and the generic MLP map reaches 0.84, so reconstruction tracks the benefit only loosely.

### Under shift the mapped probe degrades less on code (+0.091) and more on MMLU-Pro (-0.063)

What is plotted: test-split versus shifted-set rho for the context probe, the linear-mapped probe and the surface-features baseline, at 250 and full labels; shifted sets are TriviaQA to NQ-Open, MATH levels 1 to 3 to 4 to 5, an MMLU-Pro category holdout, and HumanEval + MBPP + LeetCode to BigCodeBench + LiveCodeBench.

![Correlation on the locked test split versus the shifted evaluation set per surface](https://raw.githubusercontent.com/superkaiba/explore-persona-space/edd7d61ffa36c2c838e6da30c0d216744e53d10a/figures/issue_2388/fig5_transfer.png)

> **Figure.** *On the code benchmark shift the mapped probe keeps 0.38 rho while the direct probe falls to 0.29.* Surface features decline sharply under shift on every surface (math only to 0.39). Faint points are the three label draws (they coincide at full labels).

The mapped probe degrades less than the direct probe on the code cross-benchmark shift (+0.091 rho at full labels, +0.086 at 250) and slightly less on math and QA, and degrades more on the MMLU-Pro category holdout (-0.063 at full labels). The planned second shift rung (QA-trained probes read on the other three families) produced no cells and is reported as not tested, so the shift ladder here is one rung deep and no cross-family claim is made.

### The correctness DV cleared its gates: reliability 0.87 to 0.94, worst pile 60.7%

What is plotted: the full-pool distribution of the K=5 correctness rate per surface, as the fraction of contexts in each rate bin.

![Distribution of per-context correctness rates on the four full pools](https://raw.githubusercontent.com/superkaiba/explore-persona-space/524ca2c8a0aa59a7454fda80fd1fdd764ce8f525/figures/issue_2388/fig7_dv_spread.png)

> **Figure.** *Every surface is bimodal with a populated middle and neither pile above the 90% bar.* Pool sizes 23,188 / 12,321 / 11,997 / 5,650 contexts (the all-five-decided pools the fits consume).

Both piles stay far from the disqualifying 90% shape (worst case: math one-pile 60.7%), reliability is 0.87 to 0.94, and all four verifiers are programmatic (the banked QA rate is an inherited alias match). The fits and this figure use the all-five-decided pools. Per-unit exemption: the K=5 rate takes six values, so the binned distribution is exhaustive over per-context values.

On TriviaQA the correctness rate correlates with the parent's fabrication rate at rho -0.961 (n=16,000), so the novel evidence comes from math, MMLU-Pro and code. Open code-review residuals `long-loop-restartability-upstream-digests`, `code-rung1-realization-undisclosed` and `capture-upload-phase-not-skippable` are hardening items that do not affect these numbers.

Conciseness acknowledgment: the Takeaways bullet-length, per-result prose band (120 to 180), figure-caption length and total-prose budget WARNs are accepted; the grid spans four surfaces and four planned hypotheses, and every block stays under its hard cap.

---

**Repro:** branch `issue-2388`; fits code SHA `036cb059c889f8d72e329ca4afba6d5e49c5e9ca` (driver [`scripts/issue2388_fits.py`](https://github.com/superkaiba/explore-persona-space/blob/036cb059c889f8d72e329ca4afba6d5e49c5e9ca/scripts/issue2388_fits.py); late fixes `653c1d9d18` group-grain disjoint pre-check and `06a4b66fba` stage-2 label carry-through; the math, MMLU-Pro and code fit summaries record git `66bc68005b8205fe8e05e59341e6661a255b54e1`, an ancestor of the final SHA; QA and the recompute artifacts record the final SHA). Figures `scripts/issue2388_analyzer_figs.py` at `edd7d61ffa36c2c838e6da30c0d216744e53d10a` (figures 4 and 5 re-rendered there with reader-facing labels and per-draw points; fig3 at `438ed8cbd16f9d119b757b40684d52b579b9fee1` with context-end matched gaps, DV ceilings and legacy anchors; figures 1, 2, 6 and 7 unchanged at `524ca2c8a0aa59a7454fda80fd1fdd764ce8f525`). Compute: pod-2388-spreadpilot (1x H100, ~3 h wall, 2026-08-19, DV spread pilot), pod-2388-gen (4x H100, ~10 h wall, 2026-08-20 to 08-21, generation, verification and capture), pod-2388-qafits (1x H100, ~95 h wall, 2026-08-20 to 08-24, QA fits and the selector recompute), pod-2388-fits (2x H100, ~57 h wall, 2026-08-21 to 08-23, new-surface fits); walls measured launch to upload-verification PASS from the task event log; all pods terminated after upload verification passed. Results in-repo under `eval_results/issue_2388/` (fits per surface: `all_arms.json`, `selection.json`, `bootstrap_summary.json`, `permutation_null_summary.json`, `cells/`, `preds/`; DV files under `dv/<surface>/labeling.json`; recompute under `h3_recompute/`; three `INFEASIBLE__disjoint_L2000_draw*.json` records under `fits/qa/`). Data repo (verified via live listing): [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/raw_completions) (52 files), [fits mirrors incl. permutation-null matrices](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/fits) (1,443 null files for math alone), [map tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/analysis_tensors/maps).
- Reused capture store + correctness labels from [#1739](https://eps.superkaiba.com/tasks/1739): `issue1739_ctxmap/capture_store/hallucination_labeling` ([pinned listing](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue1739_ctxmap/capture_store)) and `eval_results/issue_1739/dv_dataset/hallucination/labeling.json` — fit: same model, pooling and store layout; DV re-bound to the stored correct fraction with a field-binding test; labels are the parent's three-way outcome (alias-matched correct split; Sonnet-judged abstained-versus-fabricated split).
- Reused generic map pool from [#1092](https://eps.superkaiba.com/tasks/1092): the banked U-store cell `issue1092_realistic_crossing/analysis_tensors/summaries/cell_inst_own` at revision `e590170619e7691c1a95c7b1bb20bda5fd4065ad` ([pinned listing](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e590170619e7691c1a95c7b1bb20bda5fd4065ad/issue1092_realistic_crossing/analysis_tensors/summaries/cell_inst_own)) — fit: same model and capture recipe; real-world chat (WildChat/LMSYS) context-to-answer pairs, one per context; 18,793-of-21,193 fit-pool rows (battery eval-only rows masked out).
- Reused fit pipeline and MLP-map recipe from [#1739](https://eps.superkaiba.com/tasks/1739): `src/explore_persona_space/experiments/issue_1739/fits.py` @ `036cb059c889f8d72e329ca4afba6d5e49c5e9ca` — fit: extended with a `dof_cap` kwarg defaulting to None (parent behavior unchanged).

**Context:** origin prompt (verbatim, including the scope note recorded at filing): `I want to run an experiment to test if answer correctness is predictable from the context vector, and then whether our mapping can help to predict it. Help me to plan this expeirment. [Scope settled in the same chat: all four correctness surfaces (banked QA + math + MCQ + code); headline framing = BOTH the knowledge-vs-persona map test and the label-efficiency crossover; slim arm ladder plus mandated baselines; routed as a new child task of #1739.]` Child of [#1739](https://eps.superkaiba.com/tasks/1739); planned 2026-08-19, run 2026-08-20 to 2026-08-24 through a 14-round crash-fix arc (final two rounds: the disjoint-cell infeasibility record and the stage-2 label join); interpretation revised 2026-08-24 after the round-1 critic ensemble; body revised the same day after the round-1 clean-result gate.
