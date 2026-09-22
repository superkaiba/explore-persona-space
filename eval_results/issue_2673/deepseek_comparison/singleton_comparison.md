# Matched Qwen–DeepSeek comparison with singleton DeepSeek capture

**Completed and independently verified, 22 September 2026.** Each model contributes the same eight published descriptions and 240 unchanged questions: 1,920 unique contexts in 240 storage chunks. DeepSeek production used one unpadded context per forward, native checkpoint FP8, eight H200s, and all 61 × 7,168 residual coordinates. The paid pod has been terminated.

The association depends on depth and evaluation persona. Across the four fixed DeepSeek blocks 15/30/45/60, full-bank raw Pearson correlations are −0.688/+0.885/+0.838/+0.969 for HHH and +0.392/+0.909/+0.967/+0.877 for Fred. The complete matched Qwen results and all whitening fits are shown below. No layer or model is selected as a winner.

Strong correlation does not ensure a correct preference: at the prespecified final DeepSeek block 60, HHH raw correlation is +0.969, while only 1/5 predicted preference signs agree with the published contrasts. Whitening varies across depths and question-half fits. These are descriptive associations with published tracer uptake, not a new behavioral validation of a leakage metric.

## Quantities and scope

For evaluation persona `e` (HHH or Fred) and each of five alternatives `a`, the primary predictor is `cos(e, helpful) − cos(e, a)`; the outcome is the published Helpful tracer rate minus the alternative tracer rate. HHH and Fred each have **n = 5** character pairs and remain separate. Self-cosines are excluded. The 240 questions and multiple blocks do not enlarge the behavioral sample size. Helpful and alternative rates are not complements.

Ordinary cosine uses persona means. Whitened cosine uses `C = XᵀX/N + λI`, with the existing label-free ridge grid and condition bound, and compares means under `C⁻¹`. Neither the context vectors nor the second moment is mean-subtracted; vectors are not pre-normalized before fitting the metric. The full-bank fit uses all 1,920 rows and is transductive. Cross-half fits use 960 individual contexts from 120 questions and evaluate means from the other 120 questions, retaining the same eight personas and domain.

Blocks are zero-based and fixed in advance: Qwen 15/31/47/63 and DeepSeek 15/30/45/60. All raw-cosine blocks (64 and 61 respectively), full matrices, and pair-level values remain in the source JSONs. Every table below uses **Pearson r / Spearman ρ**, except the explicitly labeled sign table. No p-values or sampling confidence intervals are claimed.

## Primary contrasts: full bank

| Model | Block | HHH raw | HHH whitened | Fred raw | Fred whitened |
|---|---:|---:|---:|---:|---:|
| Qwen | 15 | -0.564 / -0.200 | -0.567 / -0.600 | +0.723 / +0.600 | +0.634 / +0.700 |
| Qwen | 31 | +0.216 / +0.100 | +0.195 / -0.100 | +0.608 / +0.600 | +0.740 / +0.600 |
| Qwen | 47 | +0.156 / -0.100 | -0.514 / -0.500 | +0.389 / +0.600 | +0.602 / +0.600 |
| Qwen | 63 | +0.166 / +0.300 | -0.507 / -0.800 | +0.400 / +0.700 | +0.450 / +0.500 |
| DeepSeek | 15 | -0.688 / -0.900 | -0.443 / -0.600 | +0.392 / +0.600 | +0.428 / +0.600 |
| DeepSeek | 30 | +0.885 / +0.900 | +0.522 / +0.600 | +0.909 / +0.800 | +0.626 / +0.700 |
| DeepSeek | 45 | +0.838 / +0.900 | +0.747 / +0.700 | +0.967 / +0.900 | +0.670 / +0.700 |
| DeepSeek | 60 | +0.969 / +0.900 | +0.830 / +0.900 | +0.877 / +0.900 | +0.935 / +0.700 |

## Primary contrasts: fit first 120 questions, evaluate last 120

| Model | Block | HHH raw | HHH whitened | Fred raw | Fred whitened |
|---|---:|---:|---:|---:|---:|
| Qwen | 15 | -0.548 / -0.200 | -0.397 / -0.600 | +0.722 / +0.600 | +0.707 / +0.600 |
| Qwen | 31 | +0.203 / +0.100 | +0.475 / +0.500 | +0.600 / +0.600 | +0.699 / +0.500 |
| Qwen | 47 | +0.146 / -0.100 | +0.305 / +0.500 | +0.377 / +0.600 | +0.239 / -0.200 |
| Qwen | 63 | +0.166 / +0.300 | +0.068 / +0.400 | +0.383 / +0.700 | +0.075 / -0.100 |
| DeepSeek | 15 | -0.684 / -0.900 | -0.514 / -0.800 | +0.406 / +0.600 | +0.289 / +0.100 |
| DeepSeek | 30 | +0.882 / +0.900 | +0.866 / +0.900 | +0.920 / +0.800 | +0.648 / +0.600 |
| DeepSeek | 45 | +0.837 / +0.900 | +0.686 / +0.700 | +0.965 / +0.900 | +0.527 / +0.500 |
| DeepSeek | 60 | +0.974 / +0.900 | +0.743 / +0.700 | +0.881 / +0.900 | +0.955 / +0.900 |

## Primary contrasts: fit last 120 questions, evaluate first 120

| Model | Block | HHH raw | HHH whitened | Fred raw | Fred whitened |
|---|---:|---:|---:|---:|---:|
| Qwen | 15 | -0.578 / -0.200 | -0.824 / -0.700 | +0.723 / +0.600 | +0.626 / +0.500 |
| Qwen | 31 | +0.229 / +0.100 | +0.402 / +0.600 | +0.615 / +0.600 | +0.744 / +0.600 |
| Qwen | 47 | +0.166 / -0.100 | +0.237 / +0.600 | +0.400 / +0.600 | +0.580 / +0.600 |
| Qwen | 63 | +0.166 / +0.400 | +0.034 / +0.400 | +0.418 / +0.700 | +0.206 / +0.100 |
| DeepSeek | 15 | -0.692 / -0.900 | -0.604 / -0.600 | +0.378 / +0.600 | +0.200 / +0.100 |
| DeepSeek | 30 | +0.887 / +0.900 | +0.824 / +0.900 | +0.897 / +0.800 | +0.298 / +0.000 |
| DeepSeek | 45 | +0.840 / +0.900 | +0.732 / +0.900 | +0.969 / +0.900 | +0.328 / +0.300 |
| DeepSeek | 60 | +0.964 / +0.900 | +0.785 / +0.900 | +0.873 / +1.000 | +0.727 / +0.800 |

The two half-bank checks preserve the broad raw-cosine pattern while some whitened associations move appreciably. For example, DeepSeek Fred at block 30 has whitened r = +0.626 in the full-bank fit, +0.648 when fitting first/evaluating last, and +0.298 in the reverse split. These checks hold out questions, not personas or new behavioral outcomes.

## Preference-sign agreement: full bank

Each cell is the number of matching predictor/outcome signs out of five; zero values retain the exact-sign rule. All five published HHH outcomes favor Helpful, so an always-Helpful rule already achieves 5/5 on HHH. That makes sign agreement an essential companion to correlation, not an independent demonstration of discrimination on HHH.

| Model | Block | HHH raw | HHH whitened | Fred raw | Fred whitened |
|---|---:|---:|---:|---:|---:|
| Qwen | 15 | 5/5 | 5/5 | 3/5 | 2/5 |
| Qwen | 31 | 5/5 | 5/5 | 3/5 | 3/5 |
| Qwen | 47 | 5/5 | 5/5 | 2/5 | 3/5 |
| Qwen | 63 | 5/5 | 5/5 | 2/5 | 3/5 |
| DeepSeek | 15 | 5/5 | 5/5 | 2/5 | 3/5 |
| DeepSeek | 30 | 4/5 | 5/5 | 3/5 | 3/5 |
| DeepSeek | 45 | 5/5 | 4/5 | 3/5 | 3/5 |
| DeepSeek | 60 | 1/5 | 3/5 | 4/5 | 3/5 |

In the block-60 HHH example, raw cosine favors the alternative for dismissive, saboteur, peer and help-seeker, despite Helpful having higher published uptake in all five pairs. The high correlation describes relative ordering across conditions while the offset yields four incorrect signs.

## Secondary: direct similarity to the alternative versus its uptake

This is a different predictor/outcome pairing and does not replace the primary contrast. Full-bank results are below; all cross-half secondary coefficients are retained in the CSV and JSON.

| Model | Block | HHH raw | HHH whitened | Fred raw | Fred whitened |
|---|---:|---:|---:|---:|---:|
| Qwen | 15 | -0.498 / -0.600 | +0.391 / +0.300 | +0.713 / +0.600 | +0.605 / +0.700 |
| Qwen | 31 | -0.277 / -0.500 | +0.530 / +0.700 | +0.514 / +0.600 | +0.644 / +0.600 |
| Qwen | 47 | +0.081 / +0.200 | +0.289 / +0.600 | +0.279 / +0.600 | +0.501 / +0.600 |
| Qwen | 63 | +0.735 / +0.500 | +0.739 / +0.500 | +0.305 / +0.700 | +0.394 / +0.500 |
| DeepSeek | 15 | +0.842 / +0.700 | +0.763 / +0.200 | +0.259 / +0.600 | +0.273 / +0.600 |
| DeepSeek | 30 | -0.141 / -0.300 | -0.306 / -0.300 | +0.825 / +0.800 | +0.551 / +0.700 |
| DeepSeek | 45 | +0.157 / -0.300 | -0.414 / -0.500 | +0.922 / +0.900 | +0.596 / +0.700 |
| DeepSeek | 60 | -0.062 / -0.300 | -0.398 / -0.300 | +0.897 / +0.900 | +0.904 / +0.700 |

At DeepSeek block 60, the HHH secondary raw correlation is −0.062, despite the +0.969 primary contrast correlation. The primary association therefore cannot be read as an equivalent association with absolute alternative-tracer uptake.

## Interpretation limits

The predictor contexts contain short descriptions and generic incoming questions, not the complete HHH/Fred few-shot, multi-turn Bloom evaluation contexts. DeepSeek geometry is measured before story fine-tuning; the published outcomes come after story fine-tuning. Qwen adds a model/checkpoint, wrapper and precision difference. The help-seeker description also conflicts with the incoming-question target-speaker role. Five aggregates per evaluation persona share a Helpful reference, and the two question-half analyses reuse the same published outcomes. Full-bank whitening is transductive; the half-bank checks do not establish transfer to unseen personas or domains.

The scalar outcome is published tracer uptake, not a newly measured behavioral leakage score. The evidence supports reporting the depth- and persona-dependent associations above. It does not establish a generally reliable cosine leakage metric, causal prediction, or a winning model, metric or layer.

## Verification and provenance

- DeepSeek source: `8f9f676965f57401cb2fb1099c29c87b2f0a397b`; model revision `d3d4eafdc470de44bbf6f0a74f852eb522357be8`; kernel revision `8c178950e8710e26a2210e4e909cd02dc16c8715`. [Published source-branch results](https://github.com/superkaiba/explore-persona-space/tree/72b21cce6d57ff7215cc4c1374a1c8144aecb4c6/eval_results/issue_2673/deepseek_comparison/deepseek).
- Qwen source: `cb794596d9a093b8d3c897352ed64874b8b69d76`; model revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The existing `20260917_v3` result was independently reverified for this comparison.
- [Immutable DeepSeek store](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e7840c8fb3f8ad46fc654ad60ba51b55098ff1c1/issue2673_deepseek_comparison/20260921_v7/deepseek/analysis_tensors): all 262 file names, sizes and hashes verified; 2,040,569,982 bytes. The actual receipt destination is the dataset repository. [Immutable Qwen store](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c3fc35489c6ad78b621458c6d8731669f1404832/issue2673_deepseek_comparison/20260917_v3/qwen/analysis_tensors): all 261 files verified.
- All DeepSeek chunks were independently read on the pod: exactly 1,920 unique contexts, finite BF16 tensors of shape 61 × 7,168, exact reconstructed centroids, and selected analysis vectors identical to their source chunks. All 61 raw matrices agree within 3.45e−15. A separate full 7,168-dimensional FP64 primal Cholesky solve matches all four depths × three whiteners within 1.74e−12. Production used the reviewed dual formulation; this verification did not change its results.
- Singleton replay is bitwise identical across 18 smoke contexts, with zero hook error. Mixed-batch relative error remains 0.25390228629112244 and is diagnostic only. The three singleton timing chunks took 2.706, 2.434 and 2.052 seconds; initial and rolling projections passed with the unchanged 1.25 margin and 900-second preservation reserve. Attempt 10 remains an immutable failed mixed-batch attempt with zero production vectors.
- Capture fingerprint: `554ee5d67b471296ecbe49e140acaa0d4a28ae6801ec23fb413812e7ab1d1a1f`. Analysis fingerprint: `2019d4b1d8a4e5c95a362ee66105f5244e1ed4177619ec7afb9acb0cd2148a61`.
- Pod `s6jygzggobcy45` was created at 07:09:23.669 UTC and terminated through the canonical upload-gated route. Personal-account absence was verified at 07:48:49.600 UTC. Conservative allocation upper bound: 39.43 wall-minutes / 5.258 GPU-hours; cumulative ledger upper bound: 20.410 of 45 GPU-hours. Exactly one new allocation was used; no repair or second allocation occurred.

[Verification records](singleton_verification/) include the independent tensor/Cholesky checker, smoke check, both immutable publication checks, complete out-root reconciliation and teardown evidence. [Comparison JSON](singleton_comparison.json) contains 96 independently checked statistical records and 480 paired rows; [CSV](singleton_comparison.csv) exposes all primary and secondary coefficients. [DeepSeek summary](deepseek/summary.json) and [Qwen summary](qwen/summary.json) retain the complete matrices and provenance.

Published outcomes were digitized from [Figure 26](https://arxiv.org/html/2609.10883v1/images/selectivity/fxbc_bloom_deepseek_base_grid.png), under the existing [source record](published_rates_and_overlap.json). The image SHA-256 is `0d80dd99ecfaf97e049215a1cca04c65232e466009999845186230b70e3f483f`. The per-bar pixel bound is ±0.3425 percentage points, and the worst-case contrast bound is ±0.6849 points; these are digitization bounds, not sampling uncertainty.
