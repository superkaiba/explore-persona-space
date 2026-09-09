# Qwen3-8B mapping rank on correctness-defined reasoning subsets

Analysis date: 2026-09-09. User request: “run the same analyses on that subset.”
Existing-artifact CPU analysis; no new model generations, GPU provisioning, or
Overleaf edits. The frozen setup is in `qwen3_necessity_rank_plan.md`.

## Findings

**Restricting to questions correct only with thinking does not recover a
lower-rank predictive map.** Answer prediction improves after thinking, but the
rank required to stay near the full map increases. The input representation has
lower effective dimension; that does not imply lower mapping rank.

All comparisons below use layer 24. The two thinking-on states predict identical
answers. The thinking-off arm instead predicts its own, different answers.
R² uses the corpus-training-mean reference; ranks are five-fold medians at the
10%-extra-validation-SSE threshold.

| State | Correct only with thinking: R² | Rank | Correct in both modes: R² | Rank |
|---|---:|---:|---:|---:|
| Thinking off · prompt | 0.4328 | 57 | 0.4914 | 72 |
| Thinking on · prompt | 0.4652 | 55 | 0.4986 | 67 |
| Thinking on · CoT end | 0.5346 | 81 | 0.5585 | 92 |

In the **same-answer** comparison on the 4,522-question subset, R² increases by
0.06942 (conditional paired 95% interval [0.06682, 0.07201]). Required rank rises
in all five folds: changes are **+28, +29, +26, +29, +28**. The fold-0 conditional
rank-bootstrap difference interval is **[+26, +30]**; it is not an interval for
the five-fold median. The difference between marginal medians (81−55=26) is not
the median paired difference (28).

The changed-answer mode comparison also increases rank: **57 to 81**, with
paired fold changes +27, +27, +24, +24, +26. Its R² difference is +0.10187
([0.09816, 0.10542]); the fold-0 rank-difference interval is [+24, +29]. Since
answers and R² denominators differ, this is not a fixed-target comparison.

The **17,693 both-correct controls** show the same direction. Holding answers
fixed, R² increases by 0.05993 ([0.05853, 0.06134]), with rank medians **67 to 92**
and paired changes +25, +25, +25, +26, +26. The fold-0 rank-difference interval is
[+24, +27]. These are within-group contrasts, not evidence that the two differently
composed groups have equivalent effect sizes.

### Input compression and mapping diversity diverge

The following are median **entropy effective ranks** over the five overlapping
outer training folds, measured within the 4,522-question subset's training rows.
The final column is the median of within-fold ratios, not a ratio of medians.

| State | Standardized input | Answer | Fitted output | Fitted / answer |
|---|---:|---:|---:|---:|
| Thinking off · prompt | 51.86 | 135.69 | 27.19 | 0.2005 |
| Thinking on · prompt | 51.84 | 91.01 | 22.17 | 0.2436 |
| Thinking on · CoT end | 26.73 | 91.01 | 24.88 | 0.2733 |

Holding answers fixed, the input becomes less diverse, but fitted-output effective
rank **increases** from 22.17 to 24.88. Its normalized ratio rises from 0.2436 to
0.2733. Participation ratio agrees: fitted output **7.12 to 7.74**, normalized
**0.4447 to 0.4835**. The both-correct control likewise has fitted/answer entropy
ratios **0.2512 to 0.2838** and participation-ratio ratios **0.4697 to 0.5102**.

Across thinking modes, absolute fitted-output entropy rank falls slightly
(27.19 to 24.88), but answer entropy rank falls much more (135.69 to 91.01).
Normalization therefore moves upward, not downward (0.2005 to 0.2733).
This does not support compression specific to the mapping beyond changes in
the represented input and answer distributions. The ratios are spectral
descriptions, not a causal decomposition or literal fractions of linear span.

### Sensitivities and baselines

The required-rank increase is not specific to the 10% tolerance. On the
correct-only-with-thinking subset, median ranks at **5% / 10% / 20%** extra
validation SSE are **96 / 57 / 29** (off-prompt), **90 / 55 / 30** (on-prompt), and
**130 / 81 / 45** (CoT end). The corresponding both-correct medians are
**117 / 72 / 38**, **106 / 67 / 37**, and **142 / 92 / 52**.

The full-map ceiling differs across states. A fixed relative-SSE tolerance is
therefore not a common absolute R² target, and rank should not be interpreted as
intrinsic reasoning dimensionality. Inherited penalties also differ. These
descriptive checks do not isolate a causal effect of thinking on the operator.

On held-out rows, truncation at the selected rank adds 11.1–11.4% SSE across the
six arm/subset combinations: the 10% rule is enforced on validation, not test.
It is not a retrieval-preservation rule:

| Subset | State | Identity+bias R² | Full retrieval | Selected-rank retrieval |
|---|---|---:|---:|---:|
| Correct only with thinking | Off-prompt | −2.341 | 90.11% | 54.05% |
| Correct only with thinking | On-prompt | −3.353 | 87.95% | 56.39% |
| Correct only with thinking | CoT end | −1.095 | 98.76% | 89.23% |
| Both correct | Off-prompt | −2.775 | 92.54% | 72.11% |
| Both correct | On-prompt | −3.552 | 90.45% | 67.25% |
| Both correct | CoT end | −1.207 | 98.97% | 91.75% |

All retrieval values use the full 6,762-answer candidate pool (chance 0.0148%).
The descriptive predictive deficit areas are **31.86 / 27.70 / 33.37** for the
correct-only subset and **33.27 / 30.36 / 34.70** for both-correct, in the same
three-arm order. These are not expected ranks; see the definition below.

### Dataset composition and consistency

The same-answer R² increase is positive within every corpus in each subset.
This is a descriptive consistency check; no additional corpus-wise significance
claims are made. The unequal corpus mixture is visible here:

| Corpus | Correct only: n | On-prompt R² | CoT-end R² | Both correct: n |
|---|---:|---:|---:|---:|
| MATH | 624 | 0.5005 | 0.5715 | 4,519 |
| GSM8K train | 592 | 0.4522 | 0.5252 | 6,329 |
| ContextHub | 241 | 0.4199 | 0.5794 | 389 |
| MMLU | 1,750 | 0.5363 | 0.5891 | 4,396 |
| ARC Challenge | 326 | 0.4055 | 0.4676 | 774 |
| CSQA | 424 | 0.3352 | 0.4256 | 603 |
| PIQA | 565 | 0.2992 | 0.3989 | 683 |

### Figures

[Correct only with thinking: rank and diversity](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/cot-rank-qwen3/figures/issue_2546/qwen3_necessity_rank/qwen3_necessary_rank_diversity.png)
and [both-correct control](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/cot-rank-qwen3/figures/issue_2546/qwen3_necessity_rank/qwen3_both_correct_rank_diversity.png).
Panel A shows pooled held-out R² across ranks, plus each fold's selected rank/test
score. Panel B shows entropy effective ranks of inputs, answers, and fitted outputs.
Error bars: overlapping training-fold ranges, not confidence intervals.
Qwen3-8B, layer 24, seven benchmark corpora, five IID folds.

## Population and data quality

The parent panel contains 33,810 paired Qwen3-8B questions: MATH (6,745),
GSM8K train (7,162), ContextHub (8,384), MMLU (7,361), ARC Challenge (1,157),
CSQA (1,210), and PIQA (1,791). The two scored subsets are **correct only with
thinking enabled** (4,522) and **correct in both modes** (17,693). The other
questions—both wrong (9,419), correct only without thinking (2,174), and two
unknown labels—remain in training, but are not scored in these subset results.

These labels describe one banked greedy answer in each mode. They do not establish
that a question intrinsically requires reasoning, or that the thinking trace caused
the difference in correctness. They are an outcome-defined, model-specific subset.
The control group differs in size and corpus composition; absolute dimensions
should not be compared between groups as though those factors were matched.

Sources are the existing NumPy activation and prediction archives and JSON metric
banks under `/mnt/eps-data/thomasjiralerspong/cot_necessity`, with labels checked
against `eval_results/issue_2546/necessity/qwen3_toggle_labels.json`. The loader checks
unique row IDs, complete alignment, finite states, width 4,096, and original five-fold
assignments. Source manifests record file SHA-256, array shapes/dtypes, finite-value
checks, minima/maxima, mode, and selected layer. Computation uses float64; full-map
predictions are cast to float32 for comparison with the original prediction bank.
Across the 35 unique activation archives, stored states are float16, every archive
is finite with no duplicate IDs, and all selected rows align across modes. Larger
source-archive row counts are expected: the analysis uses the original paired
intersection, not every source row independently.

## Comparisons and estimators

All states use **layer 24, width 4,096**. This removes the read-layer mismatch of the
earlier chat comparison. There are three arms:

| Arm | Input state | Answer target | Ridge penalty |
|---|---|---|---:|
| Thinking off · prompt | Last prompt token, thinking off | Own thinking-off answer mean | 1,000 |
| Thinking on · prompt | Last prompt token, thinking on | Thinking-on answer mean | 1,000 |
| Thinking on · CoT end | End-of-thought token, thinking on | **Same thinking-on answer mean** | 316.2277660168379 |

Thinking-off versus CoT-end changes both the input state and the target answer.
Thinking-on prompt versus CoT-end holds the target answers and their distributions
fixed. These contrasts answer different questions and are reported separately.

The production maps retain all training rows. For outer test fold `k`, validation
is fold `(k+1)%5`; the remaining three folds contain 20,286 training rows. After rank
selection, the map is refit on all four non-test folds (27,048 rows). Both exceed the
4,096-dimensional width. These are the original within-benchmark IID folds, not
held-out-dataset or out-of-distribution tests. The inherited layer and penalties are
fixed, not retrospectively nested; **only rank selection is nested**.

The rank basis is PCA of fitted outputs on **all training rows**, as in the prior
reproduction. The selected rank is the smallest rank whose subset-validation SSE
is at most 10% above that subset's full-map SSE. Five- and twenty-percent tolerance
sensitivities are computed without additional fits. All 4,097 ranks, including the
intercept-only rank zero, are evaluated by cumulative per-row SSE contributions.
This is a predictive truncation rank, not a claim about the dimension of reasoning.

Primary held-out R² pools SSE and uses each question's **all-training-row,
within-corpus answer mean** as its reference for SST. Global-training-mean R² is
also saved. Neither is numerically interchangeable with the appendix's earlier
whole-dataset centering and equal-corpus averaging. Identity plus a learned
training-set bias is reported as a baseline.

Top-1 retrieval uses the inherited train-whitened cosine/CSLS recipe, with **6,762
candidate answers per held-out fold** (chance 1/6,762, approximately 0.0148%). Only
queries are subset-filtered; the candidate pool and CSLS query-density pool remain
full-sized. At each subset-selected rank, the same reduced map generates the entire
query-density pool. Full, reduced, and identity-plus-bias retrieval are saved.

## Diversity and uncertainty

For each outer fold, representation diversity is measured on the same subset of
its training questions. Inputs, answers, and fitted outputs are centered within
that subset; input standardization uses all outer-training rows. The three
continuous rank measures use covariance eigenvalues (squared singular values):
entropy effective rank `exp(-sum(p log p))`, participation ratio `1/sum(p²)`, and
stable rank `1/max(p)`. Fitted-output/answer-space ratios are computed within each
fold before summarizing. Raw, unstandardized input participation ratio is also
saved. The subset-restricted fitted-output spectrum is **not** the all-training-row
PCA spectrum that orders the operational rank curve.

Rank-bootstrap intervals use 4,000 corpus-stratified paired draws from the
prespecified **outer-fold-0 validation subset**, reselecting the threshold rank on
each draw with maps and PCA bases fixed. These are not intervals for the five-fold
median. R² differences use 4,000 corpus-stratified paired out-of-fold question
draws, again with all fitted maps and generations fixed. Neither bootstrap includes
training, generation, or model-population uncertainty. Diversity fold ranges are
min–max over five overlapping training sets, **not confidence intervals**; no
significance test treats the folds as independent.

The saved predictive deficit area is the exact descriptive quantity
`sum(1 - clip(R²(rank)/R²(full), 0, 1))` over ranks 0 through 4,095. It depends on
the R² baseline and full-map ceiling. Nonmonotone curves are not CDFs, so this is
not an expected rank or an intrinsic dimension. No area-bootstrap CI is supplied.

The inspected cached subset data provide one answer per mode, not a compatible
repeated-answer activation bank. Consequently this is not a repeated-answer noise
control. No additional answers or states were generated to fill that gap.

## Reproduction and artifacts

The numerical producer is `scripts/issue2546_necessity_rank.py`; the source used
for the run was committed as `530a372ca0a`. That commit also contains the frozen
plan and the production launcher. A later import-path-only hardening change does
not alter the numerical recipe; historical checkpoints pin the original source
hash and must not be silently relabelled as products of modified code.

The companion `scripts/issue2546_necessity_rank_report.py` verifies the fifteen
completed cells and their CSV hashes, checks full-panel and subset parity against
the original metric banks, computes paired uncertainty, and renders the figures.
Output lives in `eval_results/issue_2546/qwen3_necessity_rank/`: three source
manifests, fifteen fold JSON/CSV pairs, pilot and production completion records,
per-subset summaries, and a combined `summary.json`. The JSON cells retain the full
rank curves, covariance spectra, and conditional rank-bootstrap draws; CSVs retain
held-out per-question losses, baselines, and retrieval hits. The summaries preserve
per-corpus metrics and paired R²-bootstrap difference draws.

Re-run the historical numerical recipe in a checkout at `530a372ca0a`, with an empty
output directory and the original cached sources available. Do not rerun merely to
inspect the results. The report alone needs no fitting or new inference:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  uv run python -m scripts.issue2546_necessity_rank_report \
  --results eval_results/issue_2546/qwen3_necessity_rank \
  --data-root /mnt/eps-data/thomasjiralerspong/cot_necessity \
  --figures figures/issue_2546/qwen3_necessity_rank
```

Figures use the paper's shared `c2a-v2` styling. Rank curves pool held-out losses
over folds; dots show individual-fold validation-selected ranks and their own test
scores, so dots need not lie on the pooled curve. Diversity lines show medians;
error bars show dependent-fold ranges, not uncertainty intervals. Sidecars record
plotted values, source/output hashes, rendered text, and Git provenance.

## Execution and verification

The full-shape pilot took **165.042 seconds** (120.6 seconds for its numerical
cell), with peak RSS 8.29 GiB. The conservative fifteen-unit extrapolation was
41.26 minutes, below the user's approximately one-hour limit. Production reused
that checkpoint rather than repeating it and finished the remaining cells in
**1,251.391 seconds** including source loading. Combined measured pilot/production
wall time was **23.61 minutes**, peak RSS **8.31 GiB**, eight CPU threads, zero
GPU-hours. No downloads, pod provisioning, source-cache changes, or cleanup were
needed. The worker has exited and a fresh production completion record covers
all three arms and all five folds.

Verification checks passed for all fifteen JSON/CSV pairs, their source hashes,
the original prediction banks, and original full and subset retrieval hits.
Full-panel and subset R² agree with the original metric banks within absolute
tolerance 1e-9. The same-answer arms have matching answer spectra. An independent
read-only reviewer checked CSV-derived scores, counts, selected ranks, bootstrap
quantiles, all ninety covariance spectra, and execution-source provenance; verdict
PASS with the interpretation limits reported above.

Mapped tests completed: **40 passed, one pre-existing global scan failure**.
That failure names fourteen unrelated older scripts importing heavy libraries
before thread-cap setup; none is changed by this round. The full no-flags workflow
lint completed with **nineteen pre-existing errors**, all on unchanged paths.
The round's eight non-artifact paths pass scoped lint (41 checks run), and Ruff
passes. A final focused rerun passes all fourteen analysis/report tests. The
report and numerical tests include real entrypoint imports, exact
small-matrix comparisons, bootstrap pairing, full summary-body execution, and
checkpoint-corruption rejection. Both color and grayscale renders were inspected
and use only embedded Inter fonts; Poppler emits the shared exporter’s font-type
warning, but the rendered text is intact.

This report is the current durable finding for this inline analysis. The parent
task's status, goal, and clean-result classification were not changed. Paper
integration is deferred to the issue2546/paper analyzer on the next user-requested
paper update, using this report; Overleaf was not edited.
