# Rollout count improves R² more than corrected retrieval

Increasing the number of answers averaged into the evaluation target from K=1 to K=5 raises held-out R² by 0.052 for the linear map and 0.055 for the existing nonlinear map. Whitened-cosine/CSLS top-1 retrieval improves by 0.98 and 1.21 percentage points, respectively, with most of the observed retrieval gain appearing by K=2–3. The maps are frozen: this measures a cleaner evaluation target, not an improvement from training on more rollouts.

[Main figure](https://github.com/superkaiba/explore-persona-space/blob/codex/1901-k-rollout-ablation-20260907/figures/issue_1901/k_rollout_ablation.png) · [PDF](https://github.com/superkaiba/explore-persona-space/blob/codex/1901-k-rollout-ablation-20260907/figures/issue_1901/k_rollout_ablation.pdf) · [Figure including the baseline](https://github.com/superkaiba/explore-persona-space/blob/codex/1901-k-rollout-ablation-20260907/figures/issue_1901/k_rollout_ablation_baselines.png) · [Exact results](summary.json)

| K | Linear R² | Nonlinear R² | Linear top-1 | Nonlinear top-1 |
|---|---:|---:|---:|---:|
| 1 | 0.751 | 0.804 | 96.37% | 96.67% |
| 2 | 0.782 | 0.837 | 97.17% | 97.77% |
| 3 | 0.794 | 0.849 | 97.48% | 97.87% |
| 4 | 0.799 | 0.855 | 97.52% | 97.94% |
| 5 | 0.803 | 0.859 | 97.35% | 97.88% |

Retrieval uses 942 candidates, with top-1 chance 1/942 = 0.106%. R² uses the original 1,000 held-out contexts. Retrieval is not strictly monotonic: K=5 is slightly below K=4 for both maps. These results do not establish an optimal K.

The registered K=5 minus K=1 contrasts have paired 95% context-bootstrap intervals:

| Predictor | Change in R² | Change in top-1, percentage points |
|---|---:|---:|
| Linear | +0.0517 [0.0486, 0.0552] | +0.98 [0.17, 1.76] |
| Nonlinear | +0.0553 [0.0520, 0.0590] | +1.21 [0.47, 1.97] |
| Identity + learned bias | −0.0628 [−0.0689, −0.0575] | +6.67 [5.10, 8.30] |

The identity-plus-bias baseline has R² −0.917 → −0.979 and top-1 64.46% → 71.13%. Thus target averaging does not guarantee a higher R² for an inaccurate predictor, even when its retrieval improves. Raw Euclidean retrieval gains are larger for the fitted maps: +5.75 percentage points for linear and +4.23 for nonlinear. Raw cosine and whitened cosine companions, top-5 reads, and all subset ranges are in the JSON.

## Design

This is a user-requested analysis follow-up on [task 1901](https://eps.superkaiba.com/tasks/1901). It reuses the paper's Qwen2.5-7B-Instruct layer-19, context-based maps trained on 963,444 contexts. No map refitting, model training, answer generation, or judging was performed. The existing nonlinear prediction artifact is included as a previously approved parent-task comparison; no new nonlinear model was fitted.

For every test context, the bank contains the original answer vector and four fresh on-policy answer vectors, seeds 43–46. The inherited generation recipe is temperature 1.0, top-p 0.95, and a 1,024-token cap. Each answer vector pools its answer-token activations under the parent's capture convention, including the end-of-turn tail. Input byte hashes, prediction-row order, layer dimensions, positive answer-token counts, generation-seed identities, finite values, and split membership are checked before scoring.

For each K, evaluate every size-K subset of the five vectors, then average the subset-specific metrics. The subset counts are 5, 10, 10, 5, and 1. The same subset is used across contexts for each scoring pass. This avoids making the result depend on an arbitrary ordering of the available draws. It is exhaustive over this observed bank, not over new possible answers.

The predictors, training-derived bias and whitening transform, test contexts, and retrieval candidate identities stay fixed across K. Retrieval follows the paper's keep-one policy for exact original-answer-vector equivalence classes, leaving 942 of 1,000 rows. Every candidate is averaged using the same K as its query's true target. The primary metric is strict top-1 retrieval under whitened cosine with two-sided CSLS and a fixed neighborhood size of 10; this neighborhood parameter is distinct from rollout count K. Raw Euclidean, raw cosine, and whitened cosine are companions. R² is pooled, variance-weighted, and centered on each target bank's own test-set mean.

Uncertainty uses 2,000 paired bootstrap draws over contexts, seed 190141. Each R² bootstrap recomputes the target centroid and denominator. Retrieval bootstraps resample query outcomes with the candidate pool and per-subset rankings fixed. The intervals condition on the five stored rollouts and the frozen maps; they do not capture uncertainty from generating a new rollout bank, retraining maps, or sampling a different candidate pool. Finite-bank subset ranges are reported separately, not treated as independent replicates. Intervals are pointwise and are not adjusted for multiple comparisons. Top-5 is descriptive without intervals.

The fresh-only sensitivity excludes the original draw from target averaging and uses all subsets of seeds 43–46 (K=1–4). Its endpoint gains remain positive: linear R² +0.0478 and top-1 +1.51 percentage points; nonlinear R² +0.0511 and top-1 +1.65 points. This checks original-draw asymmetry but still conditions on the context set selected using the original draw. It does not remove that selection from the estimand.

## Coverage and validation

All 31 planned subsets and all 5,000 answer vectors were evaluated; no cells were missing. The K=5 R² values reproduce the current Figure 2 reference within 1e-10, and both top-1 values match exactly. The cached scorer matches the canonical scorer at original-only K=1 and K=5 for both fitted maps and all four distances. Focused tests additionally compare the exact bootstrap against literal resampled datasets and the cached geometry against direct scoring on all subsets with an affine whitening transform and duplicate removal. Independent review passed; its scope and limitations are in [review.md](review.md).

The evidence concerns this model, layer, held-out distribution, and evaluation-target K. Training-target K, other models or layers, and K>5 were not tested. The observed R² gains are consistent with reducing answer-sampling noise in the target; they are not evidence that the frozen mapping itself improved.

## Reproduction

The inputs are pinned to HF dataset `superkaiba1/explore-persona-space-data`, revision `83d249cc9d495ca6f5d10f9156a622bcdca29a19`. [input_manifest.json](input_manifest.json) gives every remote path and expected hash. `summary.json` records the executed code SHA, source-script hash, original test row IDs, whitening provenance, exact scores, and bootstrap seed. `publication.json` records the verified upload of the per-row and bootstrap archive; that archive contains all subset masks, per-row residual sums of squares, retrieval success masks, and paired bootstrap results and counts.

Stage the six manifest files at their pinned revision with `huggingface_hub.hf_hub_download`, then write a JSON mapping each manifest key to its local file path. From the repository checkout run:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
uv run python scripts/issue1901_k_rollout_ablation.py --paths /absolute/path/paths.json
```

The script verifies all inputs and writes the metrics, per-row/bootstrap archive, publication PDF, color and grayscale PNGs, and figure sidecars. The archive belongs on HF under `issue1901_k_rollout_ablation/analysis_tensors/`; metric JSON and figures belong in Git. The primary figure focuses on the two fitted maps; the separate baseline figure preserves the full baseline scale. No Overleaf manuscript files were edited for this request.
