# K5 shared and individual map geometry

The shared fit has about 2.2 times the ridge effective degrees of freedom of an individual story-character fit, while its coefficient spectrum has a similar effective rank. Extra statistical capacity remains a possible contributor to pooled performance; these diagnostics do not isolate its causal effect.

[Rank and coefficient similarity](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue2054_section44_k5_gcp/transfer_calibration_v1/map_geometry/figures/map_rank_similarity.png) · [Prediction similarity](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue2054_section44_k5_gcp/transfer_calibration_v1/map_geometry/figures/map_prediction_similarity.png)

## Coverage and definitions

- Completed both Qwen2.5-7B checkpoints, all five conversation folds, and seven maps per fold: assistant chat template (Chat), assistant bare text (Plain), HELIOS, Wren, Dana, Vex, and Shared (all six settings pooled). All 70 planned operators were analyzed; 420 source-map/target-setting/fold/checkpoint evaluations were scored.
- These are the existing K=5 rollout-mean answer targets. No new generations, system-prompt condition, penalty search, or target calibration was introduced.
- Restore the published standardized-input, centered-output ridge fit at its already-selected penalty. Express predictions as raw row vectors `x @ A + b`, with `A = diag(1/input_std) W`. Compare all operators in the same raw activation coordinates within each checkpoint. Intercepts are excluded from coefficient spectra and cosine.
- All matrices have 3584 × 3584 coefficients (12,845,056 slope parameters) and 3584 intercept parameters. All 70 have numerical rank 3584 at the recorded float64 tolerance.
- `r90` counts singular directions needed for 90% of the sum of squared singular values of A. It describes coefficient energy, not explained answer variance. Stable rank is sum(s²)/max(s²); participation ratio is sum(s²)²/sum(s⁴). Full spectra are in the fold NPZ files.
- Ridge effective degrees of freedom are the published trace of the ridge smoother, per output coordinate, excluding the common intercept. They depend on the training design and penalty, and are distinct from the rank of A. Shared fits use approximately 38,400 rows versus 6,400 for individual fits. All story and Shared fits use lambda=3162.2776601683795; assistant Instruct Chat uses 1000, and other assistant fits use 3162.2776601683795.
- Coefficient similarity is the Frobenius cosine of A matrices, invariant to a global scalar. Norm-aware distances are also persisted. Prediction similarity applies every map to exactly the same held-out contexts, centers predictions within each target setting, averages Gram matrices with equal setting weights, and then computes cosine. It therefore removes constant output offsets. It does not by itself measure prediction accuracy.
- Numbers below average the five folds. Ranges across character maps/pairs refer to fold-averaged values. Plot whiskers are the minimum and maximum across folds, not confidence intervals.

## Map ranks and capacity

| Checkpoint | Map | r90 | Stable rank | Ridge df | Train rows | Cosine with Shared | Prediction cosine with Shared |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | Chat | 285.2 | 6.91 | 926.0 | 6399.2 | 0.388 | 0.623 |
| Base | Plain | 402.2 | 11.44 | 951.7 | 6400.0 | 0.433 | 0.634 |
| Base | HELIOS | 513.6 | 22.61 | 953.2 | 6395.2 | 0.586 | 0.870 |
| Base | Wren | 526.0 | 20.29 | 965.7 | 6394.4 | 0.588 | 0.880 |
| Base | Dana | 569.0 | 21.49 | 969.0 | 6395.2 | 0.599 | 0.873 |
| Base | Vex | 548.8 | 21.44 | 973.8 | 6397.6 | 0.551 | 0.860 |
| Base | Shared | 546.6 | 18.35 | 2160.6 | 38381.6 | 1.000 | 1.000 |
| Instruct | Chat | 585.6 | 21.17 | 1409.2 | 6400.0 | 0.470 | 0.718 |
| Instruct | Plain | 275.2 | 4.67 | 1090.2 | 6400.0 | 0.413 | 0.676 |
| Instruct | HELIOS | 507.0 | 20.20 | 1006.8 | 6399.2 | 0.588 | 0.882 |
| Instruct | Wren | 507.8 | 16.74 | 1008.4 | 6398.4 | 0.585 | 0.886 |
| Instruct | Dana | 562.2 | 17.18 | 1018.1 | 6400.0 | 0.586 | 0.876 |
| Instruct | Vex | 521.8 | 16.52 | 1021.3 | 6400.0 | 0.549 | 0.866 |
| Instruct | Shared | 530.6 | 16.66 | 2260.0 | 38397.6 | 1.000 | 1.000 |

## Similarity and transfer

| Checkpoint | Character-pair coefficient cosine | Character-pair prediction cosine | Frozen single-character → other-character R² | Source identity + bias R² | Euclidean top-1 |
|---|---:|---:|---:|---:|---:|
| Base | 0.376–0.437 | 0.843–0.877 | 0.463 | -1.411 | 65.328% |
| Instruct | 0.378–0.443 | 0.847–0.887 | 0.474 | -1.332 | 64.423% |

The transfer average covers all 12 directed story-character pairs and five held-out conversation folds per checkpoint. Retrieval uses the held-out answers from the target setting as candidates: 1,543–1,659 candidates (nominal top-1 chance 0.060–0.065%). Euclidean and cosine retrieval both follow the parent tolerance-based midrank rule for ties. The identity-plus-bias baseline learns its bias only from the source training set. Full 6-target × 7-source score matrices and all baseline and retrieval cells are in results.json.

Individual story maps generalize across story characters, despite their coefficient cosines being far below one. Assistant chat/plain maps remain more distinct. This supports common predictive structure within the story framing; it does not establish that an assistant-only map generalizes into stories, nor that the pooled fit implements one universal mechanism.

## Interpretation and limits

The shared map does not have a substantially larger r90 than an individual story map. Its ridge effective degrees of freedom are nevertheless about 2.2 times higher, with six times as many training rows at the same story-map penalty. Thus similar spectral ranks do not remove the effective-capacity/training-data confound. A causal test would match training size and ridge degrees of freedom (or impose matched rank constraints), then compare held-out performance. This analysis does not perform that new fitting experiment.

Coefficient cosine is lower than agreement between the same estimator across overlapping training folds (individuals roughly 0.82–0.86, Shared 0.90–0.91). Those folds overlap and are not an independent noise ceiling. High centered prediction cosine indicates functional agreement on the tested input pool, not identical maps, identical raw offsets, or qualitative behavioral equivalence.

## Validation and provenance

- Original K5 dataset revision: `9de026f872c19b2ca4fd3e4539de820e08038ee3`.
- Published K5 reference SHA-256: `90309ec95e757ee5bd2e1b941e006858bd24a328656386444b6c55fb43f928f2`.
- Analysis script SHA-256: `a2591b4ddeda52ad3bef971652751655ab51b4faec396abb4292e83fa8ff6151`.
- Input SHA-256 values are checked against the existing K5 source manifest and immutable parent provenance. Fold assignment is pinned to source commit 957c454a8ec9a2f520bb754543e7494904636e20, five conversation folds, seed 137.
- All original own/pooled R² cells reproduce within 1.67e-15; original Euclidean and cosine top-1 cells pass exact parity within 1e-12. Restored assistant matrices are independently compared with their hash-verified saved coefficients.
- Four focused numerical tests cover restoration with unequal input scales and means, coefficient cosine versus scale, common-pool retrieval, and parent parity with duplicate targets. Independent code review completed and its findings were resolved.
- Workers completed successfully under active supervision. Model-specific partial_result.json files are worker aggregation records; results.json is the final combined complete result.
- Large coefficient matrices were held in RAM and are reproducible from pinned input banks and the recorded penalties. Fold NPZ files retain full singular spectra, intercepts, and standardization vectors. Raw generations remain in the existing published K5 banks.

## Reproduce

Use the existing project environment and stage the pinned input manifests/maps used by the earlier K5 transfer analyses. From this repository checkout:

```bash
UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv uv run --no-sync python scripts/issue2054_k5_map_geometry.py --out eval_results/issue_2054/k5_map_geometry
UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv uv run --no-sync python scripts/issue2054_k5_map_geometry_plot.py --out eval_results/issue_2054/k5_map_geometry --figures /tmp/issue2054-map-geometry-figures
```

The producer can also run one checkpoint per process with `--model`, followed by `--collect-only`. Plot-only reproduction needs only results.json. Producer and plotting code are included in the published code directory.
