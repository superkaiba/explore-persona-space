# SAE property reanalysis with complete saved labels

Completed 23 September 2026 at the author’s request. This is an analysis correction to issue 1482. No Overleaf text or manuscript assets were changed.

The same saved layer-19 Qwen2.5-7B-Instruct decoder-direction R² scores and covariates were reused. No metamodel fitting, answer generation or judge calls were run. Historical answer-generation settings were inherited rather than revalidated in this correction.

## Correction and comparison

The previous label loader used an intermediate matrix filtered to an older experiment’s feature population. Of the current 120,716 features, 7,456 lacked entries in that matrix, although 7,426 have judgments in the full saved cache. Missing or unresolved labels became zero in binary property indicators. The archived cache agrees exactly with the legacy labels wherever the legacy matrix has an entry.

The rerun first reproduces every score, winner, control set and retirement decision across all 14 published rounds. Maximum absolute score difference is zero. The same 39 candidates, target, matching bins, minimum cell size, retirement threshold and stopping rules are then reused.

Three stepwise runs separate the published computation, loading all labels while retaining unknown-as-zero coding (diagnostic only), and the corrected primary analysis. The primary analysis uses the common 105,714 features with usable labels on all five axes. It excludes 15,002 features from the original population, including unresolved labels, missing records and the speaker-property “unclear” label. A known “none” speaker label remains a valid category. Resolution requires at least three agreeing votes, not unanimity.

An additional marginal sensitivity uses every feature with a usable label on the axis being scored. These axis-specific populations differ, so they are not a common population for ranking properties. The primary-versus-published comparison combines restored labels and population restriction. The diagnostic isolates label restoration.

## Results

The first seven selected properties remain in the same order. Scores below are concordance minus one half, conditional on all earlier selected properties. Positive scores associate a property with better decoder-direction recovery.

| Round | Property | Published | Corrected |
|---:|---|---:|---:|
| 0 | Variance explained in answer space | +0.315579 | +0.316110 |
| 1 | Speaker: identity / disposition | +0.122638 | +0.127285 |
| 2 | Logit footprint: promoting | -0.103605 | -0.104152 |
| 3 | Logit footprint: suppressing | -0.098661 | -0.099585 |
| 4 | Content type: topic | -0.086682 | -0.079909 |
| 5 | Write norm, gamma-scaled (OUTPUTNESS) | -0.060612 | -0.060539 |
| 6 | Judged role: output-promoting  [k=0.31] | -0.045820 | -0.045310 |

Thirteen of the fourteen selected properties overlap. The corrected sequence selects speaker language and does not select content operation within the inherited 14-round cap. Later scores can also involve different control sets and should not be treated as estimates under identical conditioning.

For the unconditioned score, interpretability changes from +0.000516 to −0.027373. The per-axis-resolved sensitivity is −0.026813, while restoring labels alone gives −0.026098. Speaker identity/disposition remains positive (+0.289893 to +0.277377), as does abstract contextual content (+0.108531 to +0.104137). These are descriptive point estimates, without new uncertainty intervals or selection-stability claims. The correction does not validate the semantic judge or remove limitations of greedy selection and coarse matching.

## Artifacts and reproduction

- `eval_results/issue_1482/full_labels_reanalysis_20260923/comparison.csv`: all 39 properties, all variants, selected rounds and scores.
- `comparison.json`: structured equivalent.
- `coverage.json`: resolved, unresolved, missing, unanimous and class counts by axis.
- `completion.json`: finished run, population sizes, full selection sequences and stop reasons.
- `verification.json`: exact historical reproduction check.
- `manifest.json`: source and input SHA-256 hashes, inherited selection settings.
- `input_locations.json`: Git revision and verified Hugging Face locations for every input and the saved feature populations.

Inputs that were not present with matching content in the source Git revision were uploaded to the existing Hugging Face data repository and downloaded at their pinned revision to verify SHA-256. Numerical runs were actively supervised to successful exit; no workload remains running. Independent review checked the code against the inherited algorithm and a pair-enumeration oracle.

Run from a source checkout containing the inputs identified in `input_locations.json`:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python scripts/issue1482_full_label_reanalysis.py +source_root=/absolute/source/checkout +out_dir=/absolute/new/output hydra.run.dir=/absolute/log/directory hydra.output_subdir=null hydra.job.chdir=false
```

The original analysis driver is committed at `f42fcc046b4`. Completed outputs are reusable only when the full manifest, including script and input hashes and source revisions, matches. The public comparison chart reports the marginal associations; its sidecar contains exact plotted data and output hashes.

[Open the comparison chart](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8cc43071c35ec269c2166d15626fb96a4344c198/figures/issue_1482/full_labels_reanalysis_20260923/label_associations.png).
