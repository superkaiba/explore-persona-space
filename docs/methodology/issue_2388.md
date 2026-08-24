# Methodology — issue 2388: correctness readouts from pre-generation context vectors across four eval surfaces (label-budget ladder + context-to-answer map arms)


**Design:** four correctness surfaces on frozen Qwen2.5-7B-Instruct (no fine-tuning): short-answer QA (banked TriviaQA train, 16,000 contexts; NQ-Open as the shifted set), math (full MATH, 12,497 contexts with a decided DV), multiple choice (MMLU-Pro, 12,032), and code (HumanEval + MBPP + BigCodeBench + LiveCodeBench-v5 + LeetCodeDataset, 5,654 post-dedup; 373 LiveCodeBench items dropped as LeetCode duplicates). The DV per context is the fraction of K=5 temperature-1.0 rollouts judged correct: on math, multiple choice and code a programmatic verifier decides each rollout (normalized final-answer match, option match, unit-test execution); the banked QA DV re-binds the parent's three-way outcome (correct, abstained, fabricated) to its stored correct fraction; the correct split is a word-bounded normalized gold-alias match (`judging.alias_correct`), with the parent's judge deciding only the abstained-versus-fabricated split among alias-incorrect rows. Predictor arms, all with the same ridge readout (dof-capped GCV, cap 0.9, selected lambda and effective dof logged per fit): the direct context probe, mapped-answer probes through a linear and an MLP map, an oracle probe on the true answer vector, and two one-parameter correctness-direction probes (math, MMLU-Pro and code only). Baselines: identity plus learned bias, constant floor, shuffled-map controls for both map families, a matched-rank compressed-context probe (PCA on the context vector), a surface-features probe (length, TF-IDF, level or category or benchmark identity), an external-embedding assessor (frozen sentence-transformers/all-mpnet-base-v2 embedding of the question text under the same ridge readout), and a post-generation K-rollout self-agreement reference row (math and MMLU-Pro only). Label budgets L in 250, 500, 1,000, 2,000, 4,000, 8,000 and full, 3 label draws each; code's train split is 3,958 rows, so its realized ladder is 250 to 2,000 plus full, and the three full-budget draws coincide by construction (no label subsampling at full). Map-pool composition cells at fractions 0, 0.5 and 1 in-domain at fixed pool size 8,000 (code uses its own 3,958-pair generic pool), read at the 250, 2,000 and full anchors, plus a QA additive variant and a QA label-disjoint variant (realized at 250 only, see Takeaways). Generic map pairs are subsampled from a banked store of real-world chat contexts (WildChat/LMSYS), one context-to-answer activation pair per context, captured under the same model and pooling; 18,793 of its 21,193 rows form the fit pool (battery eval-only rows excluded; producing task and pin in the footer reuse bullets). Splits are group-level (entity, level by subject, category, benchmark by problem) into train, dev and test (for example QA 11,009 / 1,653 / 3,338); every selection (layer, lambda, basis among ambient and PCA 16/48/128, map family) was frozen on dev in a committed selection file before the single test read.

**Training:** N/A — no model training. Maps are ridge and MLP fits over frozen activations. The linear map is a closed-form multi-output ridge per layer: inputs standardized per dimension on train statistics, outputs train-mean-centered, lambda selected by generalized cross-validation over the grid 0.01 to 1,000 (six decades; the dof cap applies to the label-supervised readout, not the map fit, whose pools all exceed d = 3,584). The MLP map is the parent recipe: 3,584 to 512 to 3,584 (Linear, GELU, Linear), MSE loss, AdamW at learning rate 1e-3 with weight decay 1e-4, batch 4,096, at most 300 epochs with early stopping (patience 20) on an internal 10% validation split, same standardization (Source: parent constants and the reused fitter at the fits code SHA in the footer).

**Evaluation:** primary metric Spearman rho on the held-out test split, with held-out R-squared and AUROC companions; paired group bootstrap (2,000 resamples, identical group resample shared across arms; frozen-config intervals at the dev-frozen selection; group counts per cell: QA 2,146, math 36, MMLU-Pro 14, code 1,131) and 200 group-level permutation-null draws per cell rerun through the same fit path with per-draw re-selection (per-draw matrices persisted to the data repo; the full-label read per surface is committed as a permutation-null summary recording observed, null maximum and p = 1/201). Full-pool DV spread cleared the gates specified in the plan on every surface: zero-pile / one-pile fractions 12.6% / 60.7% (math), 23.5% / 34.1% (MMLU-Pro), 37.0% / 29.8% (code), all at or below the 90% bar; beta-binomial reliability 0.87 to 0.94 (attenuation ceiling rho 0.93 to 0.96); the BigCodeBench environment gate passed 25 of 25 canonical solutions with zero flaky mismatches, so its 1,140 items entered the fits (code train split 3,958 rows, above d = 3,584, no APPS fallback needed). The labeling files cover 12,497 / 12,032 / 5,654 contexts with at least one decided rollout (math / MMLU-Pro / code); the fits and the spread figure use the all-five-decided pools (12,321 / 11,997 / 5,650). For the knowledge-versus-persona panel, per-DV train-pool ceilings were recomputed from the banked labeling files (correctness 0.97, fabrication 0.96, sycophancy 0.96, evil 0.95; rates via the beta-binomial decomposition, graded persona scores via the same variance decomposition over the K-rollout mean). Language-intrusion audit over all new-surface rollouts: 0.04% of math, 0.89% of MMLU-Pro and 0.42% of code rollouts contain CJK characters, split roughly evenly between verifier-correct and verifier-incorrect rows, so intrusion does not inflate the incorrect class; the banked QA rollouts carry the parent's measured 6.2% intrusion as an inherited caveat. Generation cap-hit fractions 0.39% (math), 0.02% (MMLU-Pro), 0.01% (code), all under the 2% re-generation trigger. The secondary teacher-forced margin validates on MMLU-Pro (Spearman 0.567 against the rate, n=12,032) but fails validation on math (-0.103, n=12,497; longer gold solutions are less likely per token exactly where the model succeeds), and code has only the rough reference-solution companion (0.135 on the 1,140 BigCodeBench rows, computed from the per-token reference log-probability under the code-prompt-plus-canonical composition; the recorded max-over-compositions recipe reads 0.076), so no margin-based claim is made outside MMLU-Pro. QA label provenance: the parent's three-way labeling ran judge `claude-sonnet-4-5-20250929` (3 draws, temperature 1.0, max 400 tokens; rubric `hallucination_three_way`, prompt text in `src/explore_persona_space/experiments/issue_1739/judging.py`) only on alias-incorrect rows to split abstained from fabricated; the correct split this task reads is `judging.alias_correct`. The derived QA DV file is committed: [labeling.json, pinned](https://github.com/superkaiba/explore-persona-space/blob/1ec108e02e10317c5183fe7cc13271b9917f6209/eval_results/issue_2388/dv/qa/labeling.json).

**Data extraction:** context vector = last-prompt-token residual stream, 28 layers by 3,584 dims; answer vectors = token-mean over each rollout's answer span (last-answer-token captured as a companion on the three chain-of-thought surfaces, used by the oracle rows). QA reuses the parent's banked capture store and labels (23,188 contexts under the hallucination-labeling prefix on the data repo, produced by the same model, pooling and store layout as this task's captures); the QA correctness DV re-binds each row's DV field to its stored correct fraction, with an adversarial field-binding test pinning that the fit consumes the correctness column. The three new surfaces were generated, verified, and captured teacher-forced on the issue pods (about 30,600 contexts by 5 rollouts). Generation recipe (every value read from the pinned constants at the fits code SHA):

| Generation parameter | Value |
|---|---|
| Rollouts per context (K) | 5 |
| Temperature | 1.0 |
| top_p | 1.0 |
| vLLM engine seed | 2388 (bfloat16, one engine per GPU) |
| Per-request seed | unset (engine-seeded sampling) |
| max_new_tokens | 2,048 on all surfaces (MMLU-Pro raised from the pilot's 1,024 after its measured 1.9% cap-hit) |
| Model + template | Qwen/Qwen2.5-7B-Instruct, chat template |

One artifact-naming defect is disclosed: the two QA generic-pool map diagnostics files duplicate the additive cell's metadata; the readouts consumed the shared generic-only map (payload SHAs logged per surface), so the fits are unaffected.

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


*Derived from the [task body](https://eps.superkaiba.com/tasks/2388).*
