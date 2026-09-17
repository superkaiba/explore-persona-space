# Ordinary and whitened cosine, without mean subtraction

The user requested ordinary cosine and whitened cosine instead of centering on the persona bank. Both have now been computed from the saved Qwen3.8-27B capture and compared with the same published Kimi-K2.6 leakage means. This is a descriptive comparison across models, not a Qwen behavioral test.

## What full SFL means and why its cosine is one

The full SFL system prompt combines **sarcasm, numbered lists, and a French closing sentence**, while keeping the answer otherwise English, accurate and helpful. The exact captured prompt is in `configs/pilots/story_persona_prompts.json`, from the [paper's prompt table](https://arxiv.org/html/2609.10883v1#A3.SS6).

Every condition is compared with the centroid obtained under that full prompt. Full SFL therefore compares to itself: cosine(v,v)=1. This is true for ordinary cosine, centered cosine, and symmetric whitened cosine under any positive-definite metric. It says nothing about leakage being 100%. The reference is another persona-prompted Qwen context vector, not a vector independently extracted from the training story's character.

## Exact metrics

Let c_p be the original persona centroid and t the full-SFL centroid, averaging the same 240 questions. Ordinary cosine is `(c_p.T @ t) / (norm(c_p) * norm(t))`, with no mean subtraction.

For whitened cosine, X contains all 2,400 individual context vectors at one block. Set `S = X.T @ X / 2400` and `M = inverse(S + lambda*I)`. Report `c_p.T @ M @ t / sqrt((c_p.T @ M @ c_p) * (t.T @ M @ t))`. This is equivalent to applying a square-root whitening transform before cosine. Applying M itself to both vectors would incorrectly use M squared.

There is **no mean subtraction anywhere** in this variant: S is the uncentered second moment, not a centered covariance. The recipe follows `analysis/leakage_predictor.py::Sigma_c` from #665/#666. Its conditioning rule chooses the smallest lambda from `logspace(-6,2,17)` with condition number at most 10,000, separately per block and without using leakage labels. This is an inherited numerical recipe, not previously validated Qwen leakage calibration. All four blocks meet the bound.

The calibration set is the same limited question/persona battery used to form the means. It is rank deficient before regularization (2,400 samples versus 5,120 dimensions), and is not held-out or broad-corpus whitening. In the full-row-rank, vanishing-ridge limit, whitening these rows makes them orthogonal; means over disjoint persona row groups also approach orthogonality. Therefore the observed compression of non-self similarities toward zero can partly be induced by calibration geometry. It cannot by itself establish better or worse leakage prediction.

## Results

All six unique conditions are retained in the primary pairing with **absolute SFL-associated tracer uptake** reconstructed from the [official published vector graphic](https://arxiv.org/html/2609.10883v1/slf_ladder_4cell_tracer_rates.svg). The [original pairing report](../published_leakage_comparison.md) records extraction and source verification. Pearson r measures linear association; Spearman rho measures rank agreement.

| Block | Raw r | Raw rho | Whitened r | Whitened rho | Ridge | Condition |
|---|---:|---:|---:|---:|---:|---:|
| 15 | 0.862 | 0.429 | 0.264 | 0.371 | 0.316228 | 6488.4 |
| 31 | 0.816 | 0.429 | -0.008 | 0.371 | 1 | 4180.8 |
| 47 | 0.644 | 0.371 | -0.044 | 0.429 | 1 | 8843.8 |
| 63 | 0.462 | 0.371 | -0.023 | 0.429 | 31.6228 | 4050.4 |

The four blocks were fixed in the original pilot before examining leakage. Ordinary cosine is additionally saved at all 64 blocks; whitening was run at the four fixed blocks only.

At block 63, the full six-condition comparison gives ordinary r=0.462/rho=0.371 and whitened r=−0.023/rho=0.429. Excluding the self-comparison is a five-condition sensitivity analysis: ordinary r=0.611/rho=0.900 and whitened r=0.599/rho=1.000. The stronger rank agreement in the five-condition analysis must not replace the complete pairing. Full SFL has lower published uptake than both two-feature prompts, despite its self-cosine being one.

![Ordinary and whitened cosine against leakage at block 63](https://raw.githubusercontent.com/superkaiba/explore-persona-space/refs/heads/codex/story-persona-qwen38-pilot-20260917/figures/issue_2673/no_centering_leakage_block63.png)

The scatter retains all six conditions and marks the self-comparison in orange. The horizontal axes have different ranges. No uncertainty bars are reconstructed from the published figure; only its central estimates are used. There are six behavioral observations, not 2,400 or 64 independent behavioral replicates. Model, training-history, prompt-wording and context-distribution differences remain unresolved.

## Verification and reproduction

The pinned original tensor archive is `superkaiba1/explore-persona-space-data` revision `ecd84d8418969b8690d126f5a03908b5fa23cc84`, prefix `issue2673_story_persona_qwen38/analysis_tensors`. Every one of the 300 downloaded chunks was SHA256-checked against the independent archive inventory. Only the four requested layers were retained locally. All 2,400 rows were seen once; the reconstructed centroids match the saved centroids exactly. No new model inference, finetuning or judging occurred.

Five focused tests cover dense Cholesky equivalence, the isotropic reduction to ordinary cosine, uncentered offset sensitivity and the smallest-admissible ridge rule. Each full-size block additionally verifies its implicit primal solve residual; the maximum observed relative residual is below 3e-12. The implementation uses an exact dual-space Woodbury solve, not a truncated PCA projection. Final runs recompute the four inexpensive results and only resume verified input staging.

Run `uv run python scripts/story_persona_metric_reanalysis.py` with the repository's shared-VM thread caps and existing environment. The Hydra config `configs/pilots/story_persona_metric_reanalysis.yaml` supplies source/cache paths; the verified inventory, manifest, rows and centroids can be staged from the pinned archive. Run `uv run python scripts/plot_story_persona_metric_reanalysis.py` to render only the saved numbers.

`summary.json`, `block_*.json` and `raw_all_layers.json` preserve all matrices, paired observations, correlations and calibration diagnostics. `staging_complete.json` records source hashes/coverage; logs preserve completed execution. The figure sidecar records the exact data and output hashes. This is a completed metric reanalysis, while same-model leakage validation remains unperformed.
