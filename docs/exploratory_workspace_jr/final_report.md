# J/R workspace predictability: exploratory results

**The experiment does not support a general, workspace-specific predictability
deficit.** The stronger model has small positive native gaps under both lenses;
the weaker model has a negative J gap and an unresolved R gap. Gaps shrink with
greater sparsity budgets, and substantially larger gaps arise under rotated
dictionaries and exactly affine activation targets. These controls show why a
positive component gap alone cannot identify reasoning or workspace selectivity.

All 48 planned model × observed/null × dictionary-orientation × sparsity cells
completed, including both lenses, ridge, three MLP seeds and saved per-example
predictions. The common primary population contains 252 of 256 frozen test
contexts, with five completed rollouts in both models. Four excluded contexts
were not replaced. All results below use this same ordered cohort. The
[machine-readable result](../../eval_results/exploratory_workspace_jr/20260912/run_result.json)
contains estimates, intervals, component statistics and immutable source links;
the [generation census](generation_coverage_20260913.md) retains censoring and
all original denominators.

## Primary comparison

Here $G_S=R^2_{\mathrm{rest},S}-R^2_S$, so positive values favor the remainder.
Each target has its own centered-variance denominator. Intervals are 95% paired
context-bootstrap intervals from 2,000 draws, conditional on the fitted
predictors and dictionaries. The primary sparsity is $k=10$.

| Model | Full-answer ridge R² | J component | J remainder | R component | R remainder |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-27B | 0.666 | 0.614 | 0.656 | 0.607 | 0.658 |
| Qwen3.5-4B | 0.603 | 0.616 | 0.579 | 0.575 | 0.585 |

| Model | $G_J$ [95% CI] | $G_R$ [95% CI] | $G_R-G_J$ [95% CI] |
|---|---:|---:|---:|
| 27B | 0.042 [0.025, 0.061] | 0.051 [0.040, 0.064] | 0.009 [−0.001, 0.021] |
| 4B | −0.037 [−0.050, −0.023] | 0.010 [−0.013, 0.032] | 0.047 [0.033, 0.062] |

The 27B result survives changing lenses in sign, but neither interval establishes
a gap exceeding the predeclared practical threshold of 0.05. The 4B J component
is better predicted than its remainder; its R gap is small and unresolved.
Lens disagreement is therefore more apparent in 4B at this setting. These are
within-model results, not evidence that capability causes the difference.
[Complete paired estimates](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9d5da56295fa5c8458a90964c9519594096cc7f5/exploratory_workspace_jr/20260912/main_comparison_v1).

## Controls and contrary evidence

The same-k rotated controls and actual-dictionary affine null produce much
larger gaps. Rotation ranges below describe three fixed seeds; they are not
confidence intervals over possible dictionaries.

| Model/lens | Native gap | Mean rotated gap [seed range] | Exactly affine null gap |
|---|---:|---:|---:|
| 27B J | 0.042 | 0.326 [0.309, 0.340] | 0.408 |
| 27B R | 0.051 | 0.336 [0.316, 0.351] | 0.394 |
| 4B J | −0.037 | 0.224 [0.218, 0.230] | 0.355 |
| 4B R | 0.010 | 0.253 [0.237, 0.270] | 0.401 |

Native-minus-mean-rotated gaps are −0.284 [−0.351, −0.211] and
−0.284 [−0.344, −0.220] for 27B J/R, and −0.262 [−0.298, −0.224] and
−0.243 [−0.276, −0.208] for 4B. Native-minus-affine-null gaps are also negative
for all four comparisons. Sparse selection can create nonlinear components of
an exactly affine total, so these controls directly undermine an interpretation
of raw gaps as a distinctive workspace deficit. Neither subtraction is a causal
correction. The [native null audit](main_native_null_validation_20260913.md)
checks the actual affine inputs, coefficients and rollout layouts.

Calibration-quality matching is supplementary because it can compare different
k values. At native k=10, matched contrasts remain negative for 27B J
(−0.188 [−0.246, −0.130]), 27B R (−0.230 [−0.283, −0.171]) and 4B R
(−0.161 [−0.183, −0.137]). The 4B J contrast is excluded: one rotated
dictionary lies outside the calibrated matching range. It is not a missing
experiment or a zero estimate. Matching is approximate and does not jointly
equalize all aspects of reconstruction, variance and active-atom counts.

After subtracting each model's mean rotated gap, the 27B-minus-4B difference
is −0.022 [−0.066, 0.022] for J and −0.041 [−0.080, 0.002] for R. Thus the
control-relative comparison does not resolve a capability-related difference.

| Sparsity | 27B J gap | 27B R gap | 4B J gap | 4B R gap |
|---|---:|---:|---:|---:|
| 5 | 0.058 | 0.072 | 0.007 | 0.053 |
| 10 | 0.042 | 0.051 | −0.037 | 0.010 |
| 25 | 0.011 | 0.021 | −0.067 | −0.034 |

Increasing k reduces the native gap for both lenses and models. At k=25 both
4B gaps are negative, with intervals excluding zero. The 27B J interval includes
zero; its R interval is [0.010, 0.032]. No single favorable sparsity is selected
as the conclusion.

At native k=10, every MLP seed has lower point R² than ridge on all five targets
in both models. For the primary seed, J/R component gains are −0.0168/−0.0154
in 27B and −0.0109/−0.0159 in 4B. Nine of the ten primary target-gain intervals
exclude zero; 4B J includes zero. This tuning budget provides no MLP advantage
supporting a component-specific nonlinear-predictability interpretation.

Learning curves continue to improve. Full-answer ridge R² at 256, 512 and 1,024
training contexts is 0.491, 0.595, 0.666 for 27B and 0.443, 0.530, 0.603 for 4B.
All component fits also improve across these prefixes. The 27B gaps remain
positive and the 4B J gap remains negative; the 4B R gap stays near zero.
The curves do not establish convergence; larger training sets or other fitting
budgets remain unmeasured.

## What the supplementary measurements show

Both models retain 144 same-token J/R directions selected without test outcomes.
Mean ridge readout R² is 0.746/0.765 for 27B J/R and 0.736/0.739 for 4B;
medians are 0.750/0.776 and 0.740/0.747. The mean paired R-minus-J differences
are 0.0190 and 0.0022. These means describe the fixed direction collection;
per-direction paired context intervals are saved, and directions are not treated
as independent replicates. Readout R² visibly increases with held-out direction
variance. Matching uses training variance, avoiding selection on that plot.

| Model | Control match | Matched J/R out of 144 | Mean ridge J/R minus matched control |
|---|---|---:|---:|
| 27B | Random | 27 / 23 | −0.135 / −0.108 |
| 27B | PCA | 138 / 139 | 0.486 / 0.491 |
| 4B | Random | 30 / 29 | −0.055 / −0.059 |
| 4B | PCA | 128 / 129 | 0.454 / 0.465 |

Random pools contain 1,152 directions and PCA pools contain 1,023 nonzero-rank
directions. Unmatched lens counts are 117/121 and 114/115 for random controls,
and 6/5 and 16/15 for PCA controls (27B then 4B). Every retained match satisfies
the fixed absolute log-variance caliper of 0.2; mean distances range from
0.0066 to 0.0371. The sparse random matching and opposite signs across control
families rule out a simple reading of the unadjusted readout distributions.
Random/PCA directions are not established non-workspace features, and these
readouts do not decompose the original predictor.

Direction redundancy is substantial: J/R participation-ratio effective ranks
are 46.2/33.7 for 27B and 35.3/34.5 for 4B. Corresponding token-direction cosine
averages are 0.949 and 0.933. Mean MLP readout gains are slightly negative under
both lenses in both models. No readout direction has undefined R² or zero held-out
variance in these saved scores.

The two decompositions also overlap strongly at the pooled component level:
mean J/R component cosine is 0.895 for 27B and 0.910 for 4B, with no near-zero
cosine exclusions. Mean squared component distances are 46.14 and 1.905 in their
respective model units. Across the full 48-cell summary, the largest reconstruction
error is 2.85e-14 or less, and no context has an increasing-error pursuit step.
These arithmetic checks do not establish that the sparse dictionaries capture
every proposed workspace property.

At native k=10, pooled J/R component variance is 5.93%/4.63% of full-answer
variance in 27B and 7.44%/5.60% in 4B. Token residual-energy fractions are
0.939/0.947 and 0.898/0.908. The former is centered pooled variance; the latter
uses uncentered token energies with equal context/rollout weighting. They measure
different things. Component/remainder covariance is positive: twice its trace
is 227.13/200.14 in 27B and 8.659/7.775 in 4B, and is needed to recover total
variance. The complete [96-row statistics table](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/41aa100829be0c49c0e15377a7f90fe12022ad45/exploratory_workspace_jr/20260912/main_decomposition_summary_v1/decomposition_summary.csv)
reports sparsity, both weighting conventions, variances, covariance and errors
for every observed/control/null cell.

Estimated finite-K noise fractions are 1.71–2.43% of target variance in 27B and
2.35–3.11% in 4B. Component estimates are slightly noisier than their remainders,
so noise remains relevant to small effects. None reaches the registered 10%
higher-K trigger; the fixed K=20 follow-up was not triggered. This is not a claim
that noise is absent or that prediction error measures reasoning.

The required identity-plus-learned-bias full-target baseline has R² −1.642
[−1.867, −1.463] in 27B and −1.392 [−1.560, −1.253] in 4B. Full-target cosine
top-1 retrieval is respectively 76.98%/76.59% for ridge, 78.17%/75.40% for the
primary MLP and 74.60%/76.59% for identity-plus-bias. The candidate pool contains
the same 252 held-out targets, so top-1 chance is 0.397%; top-5/10 chance is
1.984%/3.968%. Retrieval values are descriptive, without bootstrap intervals.
All five targets, three neighbor cutoffs and both cosine/Euclidean metrics are
saved in the [complete supplement](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ce5f41cae80a32b7a92d78130118a720a3f2747e/exploratory_workspace_jr/20260912/main_supplement_v2).

## Design, provenance and limits

The [frozen plan](analysis_plan.md), [configuration](../../configs/analysis/workspace_jr.yaml)
and [pre-outcome execution revision](execution_revision_20260912.json) fix the
analysis and tuning budgets. Models were chosen using same-mode GPQA performance,
independently of mapping R²; the [capability analysis](capability_uncertainty.md)
and [mapping audit](mapping_provenance.md) retain the evidence. 27B uses the
post-block residual at zero-based block 50 (51/64, 79.7% depth), width 5,120;
4B uses block 20 (21/32, 65.6%), width 2,560. These different relative depths,
model widths, architectures, post-training and generated answers complicate the
observational cross-model comparison.

Input x is the residual at the final context token; y is the mean answer-token
residual at that same block, averaged equally across five rollouts. Exact generated
token IDs are retained. Each token is decomposed before averaging. J and R each
produce a component plus their own remainder; they are alternative decompositions,
not a disjoint three-part sum. Context-level splits are 1,024/128/256, with all
rollouts kept together. The historical mappings used K=1 and failed the unchanged
recapture gate, so the pre-outcome revision uses fresh context-only captures and
fresh fits throughout. Historical predictors were not reused on incompatible data.

J/R lenses were constructed for the exact model revisions and source hooks,
with matched calibration tokens, precision and penultimate target blocks.
Calibration used 119 valid contexts from 128 selected, excluding nine too-short
contexts in each model. Ordinary J products, specified local R propagation,
unchanged forward outputs and calibration-subset stability passed separate checks.
The [source audit](source_audit.md) records the architecture-specific scope:
R detaches residual RMS denominators and SiLU sigmoid factors and halves the two
gated-MLP product branches; attention and hybrid recurrent internals retain
ordinary propagation. R is not tested against finite-difference derivatives.
Applying the shared sparse nonnegative pursuit procedure to its dictionary is
this experiment's operational R-space definition.

Ridge uses the fixed validation-selected regularization grid. MLP selection uses
the same target scaling and validation budget across components, with seeds
42/137/271 and seed 42 primary. Smaller training prefixes keep the full-data MLP
recipes. Three geometry-preserving dictionary rotations and k=5/10/25 are all
retained. The bootstrap preserves context pairing across lenses, predictors and
models, but conditions on the fits, calibration and joint completion population.
It does not measure uncertainty from retraining or sampling different dictionaries.

The main comparison passed an independent audit against saved arrays for all
48 cells, 2,400 cell/scope metric summaries, fixed existing bootstrap draws and
all contrast algebra and interval endpoints. The first supplement failed a
FP32 norm-validation check; the [documented correction](supplement_precision_correction_20260913.md)
matches the producer's FP64 accumulation and changes no data or scientific
threshold. Failed output is preserved separately. Manuscript and Overleaf edits
remain separate from these exploratory results.

The evidence supports a model- and sparsity-dependent difference in relative
predictability, with important contrary evidence from decomposition controls.
Neither remainder is identified as all non-workspace processing. Linear
predictability is not automaticity, prediction error is not reasoning, and J/R
agreement is not independent causal validation. Causal task ablations, temporal
claims, cross-corpus generalization and larger training budgets remain outside
this experiment.


## Reproduction and artifact integrity

The [tracker export](https://wandb.ai/thomasjiralerspong/workspace-jr/runs/a2c810d26181a2e3)
contains the final native metrics and a verified 159-file artifact covering the
selected result tables, all figure formats and provenance. Its
[archived export](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/20d881006e78827935625ee2fe042b9c1612807f/exploratory_workspace_jr/20260912/main_tracking_v2)
retains the manifest and upload proof. A first export stopped before creating a
remote run because its W&B run-ID helper was unavailable; the corrected export
uses a standard-library ID validated against the installed SDK. That failed
attempt is separately archived.

Reproduce the main comparison and decomposition summary at code commit
`8e44b6dfb43f2ce8d6b64de5433e9341022a1be3`, the corrected supplement at
`8435ae62ba55aa093a3a707106ad285d8c8691a3`, and its final plots at
`803217c87d61c060184b010501dd7a481c979640`, using the pinned native runtime.
Each archive includes the executed command, source manifest, file hashes and
completion marker. The machine result's 48-cell registry identifies the exact
uploaded fit sources; the linked analysis documents describe how to stage them
into the required individual-file layout. No model inference is needed to
reproduce the final statistics from those saved inputs.

## Figures

All plots use the project style and include color, grayscale, PDF and exact-value metadata.

| Comparison | Color | PDF | Grayscale |
|---|---|---|---|
| Component R² | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/component_r2.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/component_r2.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/component_r2_grayscale.png) |
| Gaps and decomposition controls | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/gap_controls.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/gap_controls.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/gap_controls_grayscale.png) |
| MLP gains | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/mlp_gains.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/mlp_gains.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/mlp_gains_grayscale.png) |
| Stronger/weaker comparison | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/cross_model_gaps.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/cross_model_gaps.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/cross_model_gaps_grayscale.png) |
| Sparsity sensitivity | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/sparsity_gaps.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/sparsity_gaps.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/553668e7e2f6002e619027779140e50e613cc3e9/exploratory_workspace_jr/20260912/main_comparison_figures_v1/sparsity_gaps_grayscale.png) |
| J/R decomposition agreement | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/decomposition_agreement.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/decomposition_agreement.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/decomposition_agreement_grayscale.png) |
| Rollout sampling noise | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/sampling_noise.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/sampling_noise.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/sampling_noise_grayscale.png) |
| Readout distributions | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/readout_distributions.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/readout_distributions.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/readout_distributions_grayscale.png) |
| Readout predictability and variance | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/readout_variance.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/readout_variance.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/readout_variance_grayscale.png) |
| 27B learning curves | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/learning_primary.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/learning_primary.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/learning_primary_grayscale.png) |
| 4B learning curves | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/learning_comparison.png) | [PDF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/learning_comparison.pdf) | [PNG](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/66b15a7686f05a91be371edade966f5804b634c2/exploratory_workspace_jr/20260912/main_supplement_figures_v3/learning_comparison_grayscale.png) |

All six experiment workers are verified stopped, with persistent boot disks retained. [Final completion evidence](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b05a153b561aae0282151a53dbd43a61726fd433/exploratory_workspace_jr/20260912/final_completion_evidence_v1) records exact instance identities, STOP proofs and verified archival references.
