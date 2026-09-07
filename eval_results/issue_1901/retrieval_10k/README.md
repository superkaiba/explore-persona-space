# Retrieval among 10,000 candidates

The nine saved training-size points were rescored against exactly 10,000 unique
candidate answer vectors, using the original 942 held-out queries. At 963,444
training contexts, linear top-1 retrieval is 90.45% (95% query-bootstrap interval
88.54–92.36%) and nonlinear retrieval is 91.83% (90.02–93.52%). The corresponding
original-pool scores are 97.35% and 97.88%. Chance top-1 retrieval at the larger
pool is 0.01%.

[Comparison plot](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/1901-retrieval-10k-20260907/figures/issue_1901/retrieval_10k/training_sweep.png)
· [Vector PDF](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/1901-retrieval-10k-20260907/figures/issue_1901/retrieval_10k/training_sweep.pdf)
· [Exact values](metrics.csv)

Both predictor families use the same on-policy Qwen2.5-7B-Instruct answer targets.
Every target and distractor is the mean of its original answer activation and
four previously generated on-policy answer activations. Those four draws use the
parent sampling recipe: temperature 1.0, top-p 0.95, maximum 1,024 tokens,
per-request seeds 43–46. This run generated no text and fitted no maps.

## Protocol and coverage

The candidates comprise all 942 unique query targets plus 9,058 distractors,
selected in the existing bank order. All selected distractors are LMSYS contexts.
This is a deterministic extension of the parent's nested pool convention, rather
than a new random sample of the full corpus mixture. Deduplication uses the
original fp32 answer vectors before averaging. The resulting source prefix has
10,058 rows; removing the 58 repeated query vectors leaves exactly 10,000
candidates. The query set is identical in the two evaluations.

The primary metric is whitened cosine with two-sided CSLS, K=10. Whitening is
fixed to the parent's 963,444-row training-answer statistics. CSLS is recomputed
separately for each predictor and candidate pool, using only that predictor's
942-query-by-candidate similarity matrix. The 2,000-draw bootstrap intervals
resample queries conditional on this fixed candidate bank; they do not include
candidate-pool sampling variation. R² remains evaluated against the original
1,000 five-answer mean targets, as in the source figure.

All 18 requested banked cells completed: linear and nonlinear predictions at
5,000, 10,000, 25,000, 50,000, 100,000, 150,000, 250,000, 500,000 and 963,444
training contexts. The separate 1,200-context extension has summary scores but
no saved predictions or weights in the checked producer, worktrees and scoped
Hub prefixes. It is not rescored here; doing so would require refitting, though
no new generations. Copy-context and identity-plus-learned-bias baselines are
also included. The latter learns its bias from the original 3,600 training rows,
matching the extension's baseline convention.

## Reproducibility and validation

`summary.json` contains all metrics, both pool evaluations, per-query ranks,
confidence intervals, source SHA-256 hashes, coverage, and the completion
timestamp. `pool.json` contains exact candidate capture IDs, query row indices,
true-candidate columns, corpus counts and the duplicate audit. All source
artifacts use Hugging Face dataset revision
`83d249cc9d495ca6f5d10f9156a622bcdca29a19` of
`superkaiba1/explore-persona-space-data`.

Every input matched the producing result's recorded SHA-256. Each of the 18
cells reproduced its original 942-candidate top-1 and top-5 scores exactly and
its original R² within 1e-12. Both copy baselines also reproduced their original
scores. All stored top-1/top-5 results were independently recomputed from the
saved per-query ranks, and the exact candidate and query counts were checked.
The measured analysis wall time, including hash verification and input loading,
was 472 seconds; initial distractor downloads preceded this measurement.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
  uv run python scripts/issue1901_figure2_retrieval_pool.py --n-pool 10000

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
  uv run python scripts/issue1901_retrieval_pool_figure.py
```

The optional `--source-stage` argument reuses an existing base-input/prediction
download tree after hash verification. Without it, the script downloads the
pinned inputs into its own stage directory. Focused candidate-selection and
instrument-parity tests pass (5 tests including the parent scorer tests).
An independent code review found no correctness issues; its requested stronger
test fixture now includes retrieval failures and verifies that CSLS changes
ranks. The color and grayscale exports were visually checked.

The mapped shared-VM thread-cap suite has 26 passing tests and one pre-existing
failure listing 14 unrelated entrypoints. Neither new script appears in that
failure; both load the project environment before importing numerical libraries.

The required full, no-flags workflow lint completed with 20 errors: 19 in
unchanged entrypoints and one unwrapped download in the new analysis script.
That download now uses the project's transient-retry wrapper. Rerunning the
affected lint check reports only its four pre-existing violations, with neither
new script flagged. Focused tests and Ruff checks pass after this staging-only
fix; the numerical computation is unchanged.
