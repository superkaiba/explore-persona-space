# Qwen3-8B CoT rank reproduction: a mixed result

Completed 2026-09-07. One model, 33,810 questions, all five paired outer folds.
The numerical run took 634.7 seconds (10.6 minutes), peaked at 8.60 GiB RAM,
and used no GPU inference, generation, rented GPUs, or new data.

## Result

End-of-CoT states are more predictive of the eventual answer, and their raw
input covariance has lower effective dimension. **The lower required mapping
rank finding does not reproduce: its direction reverses on this Qwen3 panel.**

| Metric | Context | End of CoT |
|---|---:|---:|
| Full-map held-out R² | 0.497832 | 0.574377 |
| Full-map top-1 retrieval | 87.6545% | 98.5980% |
| Validation-selected map rank, median (range) | 55 (55–56) | 84 (83–85) |
| Raw-input participation ratio, all rows | 6.329608 | 3.869402 |
| Held-out R² at the selected rank | 0.442308 | 0.525972 |
| Top-1 retrieval at the selected rank | 56.2378% | 89.4587% |
| Identity + learned bias held-out R² | −3.603010 | −1.174691 |
| Identity + learned bias top-1 retrieval | 0.0503% | 0.0828% |

Full-map R² increases by 0.076545, and retrieval increases by 10.9435 percentage
points. All-row input participation ratio decreases by 38.87%. The required rank
increases in every paired fold, by 28, 28, 28, 30, and 29 directions. The difference
of the two median ranks is 29; the median paired difference is 28.

The rank criterion is the smallest training-output PCA projection rank with
no more than 10% extra **validation** squared error relative to that state's
own full map. After refitting on the outer training set, realized **test** extra
error is 11.0568% for context and 11.3726% for CoT end. This criterion is neither
"95% of R²" nor a promise of no more than 10% extra test error. Retrieval is a
separate metric: retaining most squared-error performance does not retain most
retrieval accuracy.

## Why higher predictability and higher selected rank are compatible

At **every positive tested rank (1 through 4096)**, CoT end has higher held-out
R² than context, in each of the five outer folds. For example, at a common rank
of 55, pooled test R² is 0.441833 for context and 0.494575 for CoT end. This
same-rank comparison is descriptive, read from the complete prespecified curves;
rank55 was not selected as a new confirmatory endpoint.

The within-own-map retention criterion asks a different question. The full
CoT-end map predicts more answer variation, leaving a smaller residual error.
Preserving that higher ceiling within 10% of its own error can require more
directions, even while the map performs better at every fixed rank. The lower
participation ratio describes concentration of **input covariance**, not the
number of predictive map directions or semantic concepts in reasoning.

The defensible addition to the CoT discussion is therefore better answer
predictability together with more concentrated input variation, **not** a claim
that Qwen3 needs lower-rank context-to-answer maps after reasoning. This is one
model on an IID benchmark mixture, not evidence for a universal model-level
law or a causal effect of reasoning. No cross-model sign test is applied to
the five dependent folds.

## Figure

[Browser-accessible comparison](https://github.com/superkaiba/explore-persona-space/blob/codex/cot-rank-qwen3/figures/paper/c1_qwen3_cot_rank.png)
and [vector PDF](https://github.com/superkaiba/explore-persona-space/blob/codex/cot-rank-qwen3/figures/paper/c1_qwen3_cot_rank.pdf).

Suggested caption, following the paper's four-beat structure:

**End-of-CoT states improve answer prediction and have more concentrated input variation.**
**(A) Prediction improves at matched map rank.** Held-out R² versus rank; large
markers indicate validation-selected ranks evaluated after outer refitting.
**(B) Input effective dimension decreases.** Raw-input participation ratio, paired
by fold. Bands: minimum–maximum across five dependent folds, not confidence
intervals; points show individual folds. Qwen3-8B, layer 24, 33,810 questions
from seven benchmarks, five IID outer folds with separate rank-selection
validation folds.

Circles identify context and squares identify CoT end throughout. Small markers
on the curves are visual series identifiers at fixed ranks; large markers are
the five validation-selected points. Rank zero is retained in the JSON but
omitted from the logarithmic rank axis. All curves and source/output hashes are
saved in the figure metadata. The PNG and grayscale render were visually checked.

## Method and fixed analysis decisions

The user approved a single current CoT-section model and separate validation/test
folds before execution. The complete prespecified protocol is in
`qwen3_cot_rank_analysis_plan.md` alongside this report.

Both inputs use Qwen/Qwen3-8B, thinking enabled, layer 24 of 36, hidden dimension 4,096.
Context inputs are `cx_last`; CoT-end inputs are `cot_boundary`. Both predict
the **same thinking-on `ans_mean` targets**, not thinking-on versus thinking-off
answers. Use all original paired evaluation rows, without correctness or
necessity filtering.

Outer test fold k is one of the original five equally sized random-row folds.
Inner validation is fold (k+1)%5; inner fitting uses the remaining three folds
(20,286 rows). Select rank using that validation fold (6,762 rows), refit on
all four non-test folds (27,048 rows), reconstruct a fresh training-only rank
basis, and test the fixed selected rank on the outer test fold (6,762 rows).
The rank transfers between these two fits; this explains why a validation
error tolerance does not guarantee the same tolerance after refitting.

Freeze the original penalties: 1,000 for context and 316.2277660168379 for CoT end.
Inputs use the training mean and sample standard deviation plus 1e-9; targets
are training-mean centered. All fits and rank eigendecompositions use float64.
The layer and penalties are inherited from the original study, not selected
inside this newly nested rank procedure. Thus the new test separation applies
to **rank selection**, not to a retrospectively nested entire study.

The rank basis comprises eigenvectors of BᵀXᵀXB for the fitted, standardized
training design, matching the issue2588 projection procedure. Each rank keeps
the affine intercept. Shared per-fold sufficient statistics remove repeated
tall-matrix reductions; cumulative projected SSE scores all 4,097 ranks without
per-rank refitting. Each of the 20 inner/outer fits is computed once.

R² is 1−sum(SSE)/sum(SST), with SST from **outer training-fold corpus-specific
answer means**, as in the current CoT section. It is not the different
global-mean R² used in some scaling-panel summaries. Retrieval uses original
training-only covariance shrinkage 0.1, whitening, cosine similarity, and
two-sided CSLS with k=10. The pool is all 6,762 test-fold answers; chance is
1/6,762=0.0147885%. Full-map retrieval reuses row/fold-aligned original hit files;
reduced-map and identity-plus-bias retrieval were recomputed.

Participation ratio is trace(C)²/||C||F² for raw, centered input covariance.
It is invariant to global scaling and is not computed from standardized
feature correlations. Five training-fold PRs and an all-row descriptive PR
are both retained.

## Prespecified sensitivity checks

| Allowed extra validation SSE | Context rank, median (range) | CoT-end rank, median (range) |
|---|---:|---:|
| 5% | 89 (88–90) | 130 (129–131) |
| 10%, primary | 55 (55–56) | 84 (83–85) |
| 20% | 30 (30–31) | 48 (46–48) |

CoT-end rank is higher in all five folds under all three prespecified tolerances.
No additional model, layer, penalty, or threshold was searched to change this
conclusion.

## Data-quality audit

Source root: `/mnt/eps-data/thomasjiralerspong/cot_necessity`.
Source formats are NPZ named-array containers (activation and prediction banks)
and JSON metric/provenance records, read with NumPy and the standard JSON parser.
All input banks are finite 33,810×4,096 float32 arrays after row alignment; fitting
casts to float64. Per-file byte sizes, original shapes, source and consumed
row counts, min/max, and SHA-256 hashes are in `manifest.json`.

| Dataset | Consumed questions |
|---|---:|
| MATH | 6,745 |
| GSM8K train | 7,162 |
| ContextHub | 8,384 |
| MMLU | 7,361 |
| ARC-Challenge | 1,157 |
| CommonsenseQA | 1,210 |
| PIQA | 1,791 |
| Total | 33,810 |

All 21 required state files were present. Every consumed ID was available in
each state, without duplicate IDs. Context and CoT-end OOF predictions have
identical row IDs, folds, labels and complete fitted masks. No missing or
failed cell was imputed, and no new row was generated. Dataset-name `train`
in GSM8K denotes the source benchmark split, not its outer metamodel fold.

All ten reconstructed outer predictions passed the inherited parity gate
(float32 comparison, rtol=atol=2e-6); maximum absolute differences were
2.3842e-7 for context and 7.6294e-6 for CoT end, with the latter covered by the
relative tolerance. Pooled full-map R² reproduces the original results within
1e-9. Original retrieval totals also reproduce exactly. The production script
and analysis script hashes are recorded in every cell's compatibility key.

Recommendation: retain raw input centering for dimension comparisons, keep
validation selection separate from test evaluation, and report the rank
non-replication alongside the positive R² and PR results. Do not reinterpret
fold consistency as independent-model statistical significance.

## Reproduction and artifacts

Results: `eval_results/issue_2546/qwen3_cot_rank/` contains ten completed
state/fold JSONs, the aggregate summary, and the input/runtime manifest.
The complete curves are the downstream analysis inputs; large activation
banks were reused in place, never copied or deleted. Source script SHA-256:
`a943b2ca537799dbabb1604c200e135213d0f6a81710fa360017075701f93376`.

From a checkout with the existing environment and source banks:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 \
uv run python scripts/issue2546_qwen3_rank_reproduction.py \
  --data-root /mnt/eps-data/thomasjiralerspong/cot_necessity \
  --production-source scripts/issue2546_allfit_necessity.py \
  --out eval_results/issue_2546/qwen3_cot_rank

uv run python scripts/paper_fig_qwen3_cot_rank.py \
  --results eval_results/issue_2546/qwen3_cot_rank \
  --stem figures/paper/c1_qwen3_cot_rank
```

Completed compatible cell files are resumed, not recomputed; changed sources
or analysis script hashes fail loudly. The original run was detached and
OOM-protected, PID 3260039 in session 3260030; both have exited, and the final
complete summary was written only after all ten cells and aggregate parity
checks succeeded. There are no remaining jobs for this reproduction.

The five focused numerical/completeness tests pass. The mapped test suite
returned 31 passes and one repository-wide thread-import guard failure naming
14 pre-existing, unrelated scripts; neither new script was implicated.
Independent read-only review checked the implementation, split separation,
rank definition, retrieval, PR, and interpretation. The manuscript itself
has not been changed by this reproduction.
