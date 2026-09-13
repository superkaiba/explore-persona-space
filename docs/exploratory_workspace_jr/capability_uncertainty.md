# Question-paired uncertainty for the frozen capability comparison

This exploratory clarification was declared after the original aggregate
capability scores and two-model selection were known. It changes neither the
selected models nor the main experiment. It rescored the original saved GPQA
answers with their pinned historical deterministic parser and reproduced both
aggregate accuracies exactly. It uses 2,000 paired question-cluster bootstrap
draws with seed 20260912, retaining every observed generation within each
resampled question.

| Quantity | Estimate | Question-cluster 95% interval |
|---|---:|---:|
| Qwen3.5-27B accuracy | 68.0% | 63.4–72.5% |
| Qwen3.5-4B accuracy | 51.3% | 46.5–55.7% |
| 27B minus 4B (percentage points) | 16.8 | 12.2–21.5 |

Both models cover the same 198 questions. The 27B records contain 670 correct
of 985 observed draws, with five draws missing; the 4B records contain 506
correct of 987, with three missing. Unparseable answers (three and five,
respectively) remain incorrect under the original parser. All retained records
have a stopped finish reason.

Equal weighting of the 198 question means gives a 16.5-point gap
(95% interval 11.9–21.3). Restricting both models to the same 194 questions
with all five draws gives a 17.2-point gap (12.8–21.9). Across all 990 planned
draws per model, assigning every missing answer the opposing best/worst outcome
bounds the gap at 16.3–17.1 points. Those last bounds are a deterministic
missing-data sensitivity, not confidence intervals; no answers were imputed
into the reported observed-data estimates.

These intervals support a capability distinction on this benchmark. They
condition on the recorded draws, assume exchangeable benchmark questions,
and do not account for choosing models from a wider candidate panel. They
do not establish benchmark representativeness, global superiority or a causal
effect of capability. Grading intentionally preserves the original parser,
including its late boxed-regex override; reproducing it does not establish
ideal semantic grading. No main representation-predictability result enters
this clarification.

The [independent review](capability_uncertainty_review.json) verified all 14
input files, all 1,972 grades, every bootstrap draw and all 10 published output
files. Four falsification tests included a same-gold question-identity swap
that preserved the retrieval row-ID set; the actual reader rejected it.
The exact historical input paths remained unchanged at the publication revision.

- [Analysis declaration](capability_uncertainty_plan.json).
- [Machine-readable results](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/718e0edd6c3e90b682591c743b79c5df91a6869e/exploratory_workspace_jr/20260912/capability_uncertainty_v1/results.json).
- [Per-row grades, bootstrap draws and exact reader](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/718e0edd6c3e90b682591c743b79c5df91a6869e/exploratory_workspace_jr/20260912/capability_uncertainty_v1).

Reproduce on the project VM with its existing environment and data-disk cache:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run --no-sync python \
  scripts/workspace_jr_capability_uncertainty.py \
  docs/exploratory_workspace_jr/capability_uncertainty_plan.json \
  /mnt/eps-data/thomasjiralerspong/workspace-jr-20260912/capability_uncertainty_reproduction
```
