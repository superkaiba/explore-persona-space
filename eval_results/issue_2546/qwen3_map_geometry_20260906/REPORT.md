# Qwen3-8B context and end-of-CoT map geometry

The two maps read weakly overlapping input directions but predict strongly overlapping
answer directions. For their leading 50 singular directions, mean principal cosine is
0.1228 for the input subspaces and 0.9214 for the output subspaces. The isotropic
random-subspace reference is 0.0924. Input overlap is low but above this reference;
the result does not establish that the input directions are statistically random.

## Setup

Both maps predict the same Qwen3-8B thinking-on, own-generated answer representation
(`ans_mean`, layer 24). Their inputs are respectively the pre-CoT context state
(`cx_last`) and end-of-CoT state (`cot_boundary`). The cached original allfit row set
contains 33,810 aligned questions across seven datasets, with dimension 4,096.

The original production run saved held-out predictions, not fitted weights. We therefore
reconstructed its exact ridge estimator and checked one original fold per map against
the saved predictions: all 6,762 held-out rows agreed exactly after the original float32
prediction serialization, for both maps. The fitted operators analyzed below are then
**descriptive fits on all 33,810 rows**, not any one of the original five held-out-fold
fits. They were not fitted on a needs-reasoning subset. No held-out performance estimate,
fold assignment, label, target, generation, or penalty-selection result was changed.

Penalties were read from the existing production results: context lambda 1000 and
end-of-CoT lambda 316.2277660168379. Fitting uses float64 arithmetic, train-input means
and sample standard deviations (correction 1, plus 1e-9), and train-output means.
The original `Ridge` class was loaded without executing its module-level CLI/logging
code. Input standardization was undone before taking the operator SVD; answer
coordinates were not whitened for this geometry analysis.

For column vectors, write the affine prediction as `y = M x + b`. Input/read directions
are the right singular vectors of M; output/write directions are its left singular
vectors. For each fixed k, the principal cosines are the singular values of the
cross-product of the two orthonormal k-dimensional bases. We report their mean and
their mean square (squared projection share). These measurements are unchanged by
sign flips or rotations within a basis.

## Results

| Leading directions | Input cosine | Output cosine | Random cosine | Input squared share | Output squared share | Random squared share |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.085722 | 0.867877 | 0.040131 | 0.009425 | 0.793286 | 0.002313 |
| 50 | 0.122768 | 0.921410 | 0.092401 | 0.020200 | 0.874594 | 0.011973 |
| 200 | 0.204628 | 0.916946 | 0.187885 | 0.057596 | 0.866235 | 0.048654 |

At k=50, squared projection share is 2.02% for the input subspaces and 87.46% for the
output subspaces; the random reference is 1.20% (analytic expectation 50/4096=1.22%).
The contrast between input and output overlap persists at all three predeclared k.

The random reference uses five batched Gaussian-QR draws at each fixed k in 4,096
dimensions, matching the historical analysis's draw count. Rotational symmetry allows
comparison to a fixed coordinate subspace. All individual principal cosines and random
draws are saved in `results.json`; the draws provide a geometric reference, not a
refitted-map null, confidence interval over datasets, or significance test. No k or
layer was selected for maximizing the observed contrast.

This is fitted-map geometry, not evidence about the model's causal computation. Both
maps share the same targets, whose covariance can itself contribute to overlapping
output directions. The comparison does not identify semantic features or show that
the model implements these fitted maps. State offsets, massive-coordinate removal,
SAE interpretation, and causal interventions were outside this run's scope.

Precision note: the historical OpenThinker descriptive reconstruction used float32
Gram/cross-covariance products and population standard deviations. This Qwen3 analysis
preserves the exact production float64/sample-standard-deviation recipe. The raw
operator orientation and subspace metric definitions are the same.

## Reproducibility and validation

Production elapsed time was 225.78 seconds, on CPU with eight threads; no GPU or new
data was used. Two representative held-out parity fits and two all-row descriptive
fits were run, followed by two full operator SVDs. Source files, cached input states,
prediction artifacts, and saved operators/bases carry SHA-256 hashes in `results.json`.
The script's uncommitted status at execution is recorded explicitly alongside its
exact source hash. The completed sentinel also pins the final results hash.

The four reusable operators/bases (293 MiB total) are
[archived on Hugging Face](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/169127ec798014812a3fd4f01750c886ca3b8e81/issue2546_cotmap/analysis_tensors/qwen3_map_geometry_20260906).
All four exact paths, byte sizes, and LFS SHA-256 hashes were verified at that commit;
the record is in `upload_verification.json`. No local analysis tensors were deleted.

The analysis completed with its process absent and a fresh completion sentinel.
Independent review recomputed every reported subspace statistic from the saved bases,
verified all four tensor hashes and the completion hash, and found no material issue.
Three focused algebra/orientation/persistence tests passed in both the producing and
reviewing agents. The mapped suite had 29 passes and one pre-existing global scan
failure (`test_no_new_torch_before_dotenv_vm_entrypoints`); its reported entrypoints
are unrelated to the new script, which loads the shared environment before importing
NumPy or Torch. No failing existing files were changed.

Payload-scoped workflow lint passed all 40 applicable checks. The required full-tree
lint completed with 21 errors, none attributed to the new script or test: 19 concern
unchanged existing scripts and two arise because this sparse worktree does not expose
`CLAUDE.md` (its instructions were read with `git show`). These global failures were
recorded, not repaired or waived; see `validation.json` and `workflow_lint_full.log`.

Re-run with fresh output directories:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 \
UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv \
UV_NO_SYNC=1 uv run python scripts/issue2546_qwen3_map_geometry.py \
  --data-root /mnt/eps-data/thomasjiralerspong/cot_necessity \
  --out eval_results/issue_2546/qwen3_map_geometry_reproduction \
  --tensors /home/thomasjiralerspong/.codex/worktrees/explore-persona-space/qwen3-map-geometry-reproduction-tensors
```

No paper or figure was edited by this analysis.
