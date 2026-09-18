# Qwen context geometry versus published DeepSeek tracer uptake

Qwen extraction is complete: **1,920 contexts = eight published short persona descriptions × 240 questions**, with all 64 decoder blocks saved. The comparison provides **mixed associations between cosine contrasts and published uptake contrasts**. DeepSeek extraction is **blocked on hardware availability**: the approved launcher returned no capacity and allocated no pod. No DeepSeek context vectors have been measured in this run.

For each evaluation persona (HHH or Fred), the primary comparison pairs `cos(evaluation, helpful) − cos(evaluation, other)` with the published `helpful tracer rate − other tracer rate`. Each correlation uses five character pairs. HHH and Fred are reported separately; the ten-pair pool is supplementary.

## Matched-description Qwen results

Pearson r for the signed similarity contrast versus the signed tracer-uptake contrast, using all 240 questions. Both metrics use uncentered vectors. All four fixed depths are shown.

| Qwen block (zero-based) | HHH: cosine | HHH: whitened cosine | Fred: cosine | Fred: whitened cosine |
|---:|---:|---:|---:|---:|
| 15 | -0.564 | -0.567 | +0.723 | +0.634 |
| 31 | +0.216 | +0.195 | +0.608 | +0.740 |
| 47 | +0.156 | -0.514 | +0.389 | +0.602 |
| 63 | +0.166 | -0.507 | +0.400 | +0.450 |

At block 63, ordinary cosine correctly favors the Helpful character for all five HHH comparisons, but the magnitude correlation is only r=0.166. All five published HHH contrasts favor Helpful, so an always-Helpful rule also achieves 5/5 direction agreement. Fred has r=0.400 with only 2/5 directions correct. Correlation and correct direction answer different questions.

Whitening is fit on the uncentered second moment of individual context vectors, with the preregistered ridge/conditioning rule. The full-bank fit is transductive. The checks below fit the whitener on one 120-question half and evaluate persona means on the other half; neither personas nor behavioral outcomes are held out.

| Block | HHH white r: first → last | HHH white r: last → first | Fred white r: first → last | Fred white r: last → first |
|---:|---:|---:|---:|---:|
| 15 | -0.397 | -0.824 | +0.707 | +0.626 |
| 31 | +0.475 | +0.402 | +0.699 | +0.744 |
| 47 | +0.305 | +0.237 | +0.239 | +0.580 |
| 63 | +0.068 | +0.034 | +0.075 | +0.206 |

The whitening result is sensitive to the fit bank: for HHH at block 47, full-bank r=−0.514 becomes +0.305 and +0.237 in the two cross-half checks. At block 63, Fred falls from +0.450 to +0.075 and +0.206. These results do not establish a reliable quantitative leakage predictor.

For the secondary question—similarity to the other character versus that character’s absolute tracer uptake—block-63 HHH correlations are +0.735 (ordinary cosine) and +0.739 (whitened cosine). These are a different outcome from the primary paired contrast and should not be substituted for it. Full matrices, all-layer ordinary cosine, all folds, Spearman correlations and individual pairs are retained in [the Qwen results](qwen/summary.json).

## What is being compared

The [published study](https://arxiv.org/html/2609.10883v1#A3.SS5) measures multi-turn, triggered Bloom behavior after story fine-tuning of DeepSeek-V3.1 Base. Its scalar is the fraction of coherent evaluations with tracer-fixation score above 5 on a 1–10 scale. Our predictor uses the released Qwen checkpoint without the study’s story fine-tuning, on generic incoming questions. This is a cross-model, cross-context diagnostic, not a new measurement of Qwen behavioral leakage.

Both HHH and Fred use published short descriptions here. The complete Fred demonstrations are unavailable in the inspected public sources, so the full few-shot evaluation contexts are not reproduced. Six story-character descriptions complete the eight-persona bank. Generic question contexts, prompt length, native model wrappers, Qwen BF16 versus planned DeepSeek native FP8, and the help-seeker speaker-role mismatch limit the interpretation. Five aggregates per persona share a Helpful reference; no sampling confidence intervals or inferential p-values are claimed. No self-cosines enter the correlations.

## Published DeepSeek rates

Approximate means digitized from [Figure 26](https://arxiv.org/html/2609.10883v1/images/selectivity/fxbc_bloom_deepseek_base_grid.png). Each bar has a pixel-resolution bound of ±0.34 percentage points, not a sampling confidence interval. Helpful and other rates are not complementary.

| Opposing character | HHH: helpful | HHH: other | Fred: helpful | Fred: other |
|---|---:|---:|---:|---:|
| dismissive | 43.6% | 7.3% | 13.0% | 58.7% |
| sarcastic | 60.3% | 4.3% | 8.0% | 40.4% |
| saboteur | 54.3% | 11.0% | 26.9% | 29.0% |
| peer | 34.7% | 5.7% | 27.6% | 6.6% |
| help_seeker | 28.5% | 3.7% | 26.5% | 3.4% |

## Earlier four-persona overlap diagnostic

The earlier bank lacked HHH, Fred and help-seeker. It used default Qwen as a proxy for HHH and different persona prompts. Its four-point result below is retained for provenance; it is not the same comparison as the new eight-description panel.

At Qwen block63, similarity from its default assistant to four published persona prompts correlates with the paper’s DeepSeek HHH uptake of the corresponding non-helpful character’s tracer: raw Pearson r=0.822, Spearman rho=0.80; uncentered whitened r=0.507, rho=0.40. Only four aggregate pairs contribute. These are descriptive associations across different models, prompt realizations and evaluation contexts.

| Character | Qwen raw cosine | Qwen whitened cosine | DeepSeek HHH other-tracer uptake |
|---|---:|---:|---:|
| dismissive | 0.6923 | -0.00206 | 7.3% |
| sarcastic | 0.5833 | 0.00157 | 4.3% |
| saboteur | 0.9039 | 0.01290 | 11.0% |
| peer | 0.7999 | 0.01047 | 5.7% |

All predeclared Qwen depths:

| Block | Raw r | Raw rho | Whitened r | Whitened rho |
|---:|---:|---:|---:|---:|
| 15 | -0.155 | -0.60 | -0.407 | -0.60 |
| 31 | 0.150 | 0.00 | -0.115 | 0.00 |
| 47 | 0.296 | 0.00 | -0.025 | 0.00 |
| 63 | 0.822 | 0.80 | 0.507 | 0.40 |

## Reproducibility and coverage

Qwen: model revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; source `cb794596d9a093b8d3c897352ed64874b8b69d76`; result commit `20077e2e60d15e2a2bbc263f83e111c0f5c11d2e`. All 240 chunks and 1,920 identities passed independent validation; all vectors are finite BF16, captured at the final input token before final normalization. Four selected blocks × three analysis fits are complete. Raw activations for all 64 blocks remain available.

All 261 files (1,525,115,935 bytes) were independently verified by path, size and content hash at the [immutable Hugging Face snapshot](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c3fc35489c6ad78b621458c6d8731669f1404832/issue2673_deepseek_comparison/20260917_v3/qwen/analysis_tensors). Small JSON results in Git match that snapshot. Numerical hook/repeatability smoke and analysis-output hash checks passed. See [publication verification](qwen/independent_publication_verification.json) and [independent result review](qwen/independent_review.md).

Reproduce analysis with `scripts/story_persona_crossmodel_analysis.py phase=analyze model_key=qwen output_dir=<verified-capture>`. The digitization/earlier-overlap script is `scripts/story_persona_deepseek_published.py`. The prompt bank is `configs/pilots/story_persona_deepseek_prompts.json`.

Planned model coverage: Qwen and DeepSeek. Realized so far: Qwen only. The DeepSeek arm is not represented as zero or inferred from Qwen. The approved continuation uses DeepSeek-V3.1-Base revision `d3d4eafdc470de44bbf6f0a74f852eb522357be8`, the same eight descriptions and question IDs, and fixed blocks 15/30/45/60. Its native-FP8 numerical and throughput gates must pass before interpreting results.


At **2026-09-18 00:43:51 UTC**, the final capacity check still exposed no eight-GPU H200, H200 NVL or B200 offer. The actual H200 launch had failed across the provider’s normal, Community and spot supply paths. Qwen’s GPU was terminated after verification; no model-arm GPU remains allocated. No automatic capacity retry is active. The approved DeepSeek recipe, monitor configuration and remaining compute allowance are saved for resumption when capacity becomes available. See [capacity evidence](deepseek_capacity_block.json).
