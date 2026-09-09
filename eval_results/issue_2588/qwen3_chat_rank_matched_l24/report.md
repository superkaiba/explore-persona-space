# Qwen3-8B chat mapping: matched layer 24

The lower chat mapping rank survives matching the read layer. Without thinking,
the operational rank is 122; after thinking at the same layer it is 74, rather
than the previous 65 at layer 22. The earlier answer-diversity explanation is
weaker after this control: answer-space diversity becomes much more similar,
while the fitted-output difference remains. This is a descriptive comparison,
not evidence that reasoning causally compresses an intrinsic mapping operator.

## Data and analysis

This control uses Qwen/Qwen3-8B and the existing LMSYS-Chat-1M general-chat
captures. It is not restricted to reasoning questions. Only the thinking arm
was newly reconstructed, at fixed layer 24. The no-thinking layer-24 result and
the earlier thinking layer-22 result were reused without refitting. Both chat
arms retain their original validation-selected ridge penalty of 1000 at the
reported layer; the fixed layer is a user-requested control, not a new winner
selected from the results.

| Quantity | No thinking, L24 | After thinking, L24 | Earlier after thinking, L22 |
|---|---:|---:|---:|
| Train / validation / test rows | 9937 / 399 / 998 | 9968 / 399 / 995 | 9968 / 399 / 995 |
| Full-map held-out R² | 0.688576 | 0.657288 | 0.669143 |
| Operational rank | 122 | 74 | 65 |
| Rank / 4096 | 2.98% | 1.81% | 1.59% |
| Identity + learned bias R² | -1.096590 | -0.167355 | -0.262439 |
| Raw-cosine answer retrieval, top-1 | 73.25% | 69.65% | 72.56% |
| Retrieval pool / chance | 998 / 0.1002% | 995 / 0.1005% | 995 / 0.1005% |

Operational rank uses the original panel estimator: PCA of training fitted
outputs followed by the smallest nested map rank whose validation SSE is at
most 1.10 times the full map's validation SSE. The basis uses training data,
and the rank threshold uses validation data, never test data. R² pools squared
errors across coordinates, with held-out target-mean SST. Reconstruction uses
the original standardized-input ridge solve in float64 and the original
float32 fitted payload and predictions.

## Diversity control

Entropy effective rank is exp(H(p)), where p normalizes covariance eigenvalues
(squared singular values). Participation ratio is (sum eigenvalues)² divided
by sum squared eigenvalues. All spectra use the same training rows within an
arm. Inputs are standardized with the fitted training scaling; answers are
centered. Fitted-output spectra reuse the map-rank eigendecomposition, avoiding
a redundant large matrix factorization.

| Quantity | No thinking, L24 | After thinking, L24 | Earlier after thinking, L22 |
|---|---:|---:|---:|
| Input entropy effective rank | 154.177 | 103.518 | 108.286 |
| Answer entropy effective rank | 161.431 | 144.254 | 108.497 |
| Fitted-output entropy effective rank | 63.064 | 46.274 | 38.430 |
| Fitted / answer entropy effective rank | 0.390656 | 0.320778 | 0.354201 |
| Input participation ratio | 34.631 | 25.923 | 26.012 |
| Answer participation ratio | 35.111 | 34.562 | 26.730 |
| Fitted-output participation ratio | 21.327 | 18.744 | 15.623 |
| Fitted / answer participation ratio | 0.607412 | 0.542333 | 0.584462 |

Matching layers shrinks the thinking arm's answer-diversity reduction from
32.8% to 10.6% by entropy rank, and from 23.9% to 1.6% by participation ratio.
However, the fitted/answer normalized summaries remain 17.9% and 10.7% lower,
respectively. Thus answer diversity alone is a less convincing explanation
than in the unmatched-layer comparison. These normalized summaries are not
literal fractions of coordinates or variance explained. They have no sampling
confidence intervals in this control. Input dimensionality also differs
substantially, and normalization does not isolate an operator from its input
distribution.

## Relation to the already matched benchmark subset

The separate benchmark analysis already reads every arm at L24 and was not
rerun. Its 4522 prompts answered correctly only with thinking show the opposite
operational-rank direction: 57 → 81 for thinking-off prompt states versus
end-of-thought states, and 55 → 81 when the same thinking-arm answers are held
fixed. Corresponding R² values are 0.432766 → 0.534631 across modes and
0.465209 → 0.534631 for the identical-answer comparison. Benchmark ranks here
are medians across five folds; chat uses one fixed split.

Source: [benchmark subset report at fd3209e3657](https://github.com/superkaiba/explore-persona-space/blob/fd3209e365778ac8beac0207bb89da967e3b7dd9/docs/paper_context_answer_map/qwen3_necessity_rank_eda_report.md).
This subset is an outcome-defined proxy from one saved answer per mode, not
proof that each question intrinsically requires reasoning. The benchmark
both-correct subset also increases in operational rank (67 → 92 in the
identical-answer comparison), so the chat/benchmark contrast alone does not
establish a reasoning-necessity interaction. Different corpora, generation and
fitting recipes, and R² baselines remain; absolute scores are not directly
comparable across these experiments.

## Data quality, provenance, and compute

All 28 B input files (23 tensors plus five metadata files) were checked against
immutable Hub content identities. The unchanged A input files were also
rechecked against both HF and the earlier result manifest. Source revision:
`superkaiba1/explore-persona-space-data@5f60a146e91248e5e1c84cf879ff5b4f1357a087`,
under `issue2588_capability_panel_cap_long/generic/qwen3-chat-v3/q3_8b/`.
Each tensor contains row IDs, an input state, and an answer state; matrices are
finite float32 with 4096 columns. Every shard's rows and order are checked
against its split manifest before fitting. Training has 9968 rows > 4096
features. Model and manifest identities, selected penalty, and fit row counts
are checked explicitly.

The reconstructed B-L24 validation and test R² agree with the saved layer-24
fit within 8.72e-10. Rank 73 has validation R² 0.62733802, below the threshold
0.62785956; rank 74 reaches 0.62823923. Independent read-only review recomputed
all spectral summaries and checked this crossing.

One production-shape CPU unit served as both pilot and final analysis:
44.66 seconds including in-function verification/loading, of which 37.35
seconds were the fit/rank/diversity calculations. Peak RSS was 2.425 GiB.
Downloads were 372955230 bytes. No GPU-hours, model generations, or provisioned
resources were used. Timing excludes interpreter startup, staging, tests,
review, and repository checks. No benchmark fit or A fit was duplicated.

Numerical execution was at clean code commit
`fec4e09d4f4f60cfb59638204535802ba279fef8`; full source hashes, package versions,
input hashes, and reused-result hashes are in `result.json`. A later import-order
hardening of the shared rank helper loads thread caps before NumPy; it changes
no estimator or numerical result. The execution artifact keeps its actual
producing commit, rather than claiming it ran the later edit.

Remaining limitations: the two chat arms predict their own different generated
answers; valid row sets differ slightly; 13 exact prompt strings overlap the
original validation and test sets (no exact training overlap); and each rank
threshold is relative to its own map's full error. These are not removed by
matching layers. This round does not revise the paper or Overleaf.

Reproduce from the committed code, using a fresh output directory:

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv
uv run --no-sync python scripts/issue2588_chat_rank_matched.py --phase stage \
  --source-root data/issue_2588/hf_dl --output-dir /tmp/qwen3-matched-l24-reproduction
uv run --no-sync python scripts/issue2588_chat_rank_matched.py --phase analyze \
  --source-root data/issue_2588/hf_dl --output-dir /tmp/qwen3-matched-l24-reproduction
```

The unchanged A-L24 input cache must also be present for provenance verification;
the original `issue2588_chat_stage_rank.py --arm a` entrypoint restores it from
the same immutable revision if needed. Existing numerical output paths refuse
refitting, including a rank-only checkpoint from a partial run.
