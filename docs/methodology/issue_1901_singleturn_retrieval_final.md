# Issue 1901: final single-turn retrieval evaluation

This follow-up defines the paper-facing retrieval protocol for the primary
single-turn context-to-answer predictors. It supersedes the earlier retrieval
reads that either borrowed whitening from the multi-turn line or evaluated a
candidate bank containing exact duplicate answer representations.

## Primary protocol

- Model and layer: Qwen2.5-7B-Instruct, residual-stream layer 19.
- Predictors: the full-data linear ridge map and nonlinear MLP comparison fit on
  963,444 single-turn context--answer pairs. Identity plus learned bias is the
  control.
- Target: a homogeneous five-rollout mean for every query and distractor (the
  original on-policy answer plus four fresh on-policy draws). No candidate uses
  a different number of rollouts.
- Duplicate handling: compute exact fp32 equivalence classes from the original
  answer vectors before rollout averaging, retain one representative of each
  class, and only then construct queries and candidate pools. The original
  1,000-query bank contains 13 duplicated classes and 58 excess rows, leaving
  942 unique queries. A stricter sensitivity analysis drops every member of each
  repeated class, leaving 929 queries.
- Whitening: use the mean and shrunk covariance fit only on the 963,444
  single-turn training-answer vectors (`lambda=0.1`). Apply
  `z = L^{-1}(v - mu_A)` to predictions and candidate answers.
- Similarity: whitened cosine followed by two-sided cross-domain CSLS with
  `K=10`.
- Main reads: strict top-1 and top-5 accuracy, realized candidate-pool size,
  chance accuracy, and 95% row-bootstrap intervals for top-1 (2,000 draws).
- Audits: duplicate-aware top-1 on the original unreduced pool, where any
  candidate in the target's exact source-vector equivalence class is correct;
  the remove-all-duplicate-classes sensitivity; single-rollout targets; and
  whitened cosine without CSLS, raw cosine, and raw Euclidean distance.

Strict and duplicate-aware retrieval are identical by construction on the
deduplicated primary pool. The unreduced pool is retained only to quantify the
duplicate artifact; it is not a headline result.

## Results

| Realized candidate pool | Linear top-1 | Nonlinear top-1 | Identity+bias top-1 | Linear top-5 | Nonlinear top-5 |
|---:|---:|---:|---:|---:|---:|
| 942 | 97.3% | 97.9% | 71.1% | 100.0% | 100.0% |
| 1,942 | 95.2% | 96.5% | 64.0% | 99.9% | 100.0% |
| 4,942 | 92.6% | 93.8% | 57.4% | 99.0% | 99.5% |
| 19,942 | 87.3% | 88.4% | 44.9% | 96.3% | 98.2% |

At 942 candidates, dropping all members of duplicate classes gives 97.7% linear
and 98.4% nonlinear top-1, so retaining one representative is not responsible for
the high score. In the contaminated 1,000-candidate audit, the linear predictor's
strict top-1 is 90.7%, versus 97.3% under the duplicate-aware definition.

At 19,942 candidates, the linear predictor obtains 75.4% top-1 with raw cosine,
83.7% after single-turn whitening, and 87.3% after CSLS. The corresponding
nonlinear scores are 85.6%, 85.2%, and 88.4%.

## Reproducibility

- Analysis: `scripts/issue1901_singleturn_retrieval_final.py`
- Tests: `tests/test_issue1901_singleturn_retrieval_final.py`
- Machine-readable results:
  `eval_results/issue_1901/singleturn_retrieval_final/summary.json`
- Figure and metadata:
  `figures/issue_1901/singleturn_retrieval_final/`
- Pinned data-repository revision:
  `83d249cc9d495ca6f5d10f9156a622bcdca29a19`
- Single-turn whitening source-store revision:
  `0620bbd6adbc88cba4af8974ed6006f47844ea04`
