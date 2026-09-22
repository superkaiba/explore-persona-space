# Qwen and DeepSeek context geometry versus published tracer uptake

**The matched singleton comparison is complete.** Both models have 1,920 contexts and 240 verified chunks. The [current comparison report](singleton_comparison.md) presents all four fixed depths, raw and uncentered whitened cosine, all three fits, and separate HHH/Fred analyses with five pairs each. DeepSeek capture used the approved one-context-per-forward native-FP8 run; the owned pod has been terminated. The one new allocation used at most 5.258 GPU-hours, bringing cumulative recorded usage to at most 20.410 of the approved 45 GPU-hours.

The associations depend on depth and evaluation persona. Strong correlation can coexist with an incorrect preference: at fixed DeepSeek block 60, HHH raw r=0.969 accompanies only 1/5 matching preference signs. The report preserves all depths and the secondary direct-other uptake comparison; it makes no new behavioral validation claim or selected-layer/model winner claim. [Machine-readable comparison](singleton_comparison.json) · [all coefficients](singleton_comparison.csv) · [independent verification](singleton_verification/).

**Historical attempt 10 remains a failure.** Its grouped-M16 diagnostic passed, but mixed-batch parity failed at 0.25390228629112244 and it captured no production vectors. Its [outcome and evidence](deepseek_attempt10/attempt.json) remain unchanged. The previous 17-GPU-hour boundary was superseded by the 21 September approval of one additional singleton attempt within 45 cumulative GPU-hours. Attempt 20 independently passed singleton repeatability, hook and throughput gates; the mixed-batch error remains diagnostic evidence.

## Earlier Qwen-only summary and source history

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

The [published study](https://arxiv.org/html/2609.10883v1#A3.SS5) measures multi-turn, triggered Bloom behavior after story fine-tuning of DeepSeek-V3.1 Base. Its scalar is the fraction of coherent evaluations with tracer-fixation score above 5 on a 1–10 scale. Our Qwen predictor uses the released chat checkpoint, not a pre-trained base checkpoint, without the study’s story fine-tuning, on generic incoming questions. This is a cross-model, cross-context diagnostic, not a new measurement of Qwen behavioral leakage.

Both HHH and Fred use published short descriptions here. The complete Fred demonstrations are unavailable in the inspected public sources, so the full few-shot evaluation contexts are not reproduced. Six story-character descriptions complete the eight-persona bank. Generic question contexts, prompt length, native model wrappers, Qwen BF16 versus DeepSeek native FP8, and the help-seeker speaker-role mismatch limit the interpretation. Five aggregates per persona share a Helpful reference; no sampling confidence intervals or inferential p-values are claimed. No self-cosines enter the correlations.

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

## Historical capacity and failure log

**The entries below preserve the earlier state of the experiment. Their pending-work statements and earlier budget limits were superseded by the approved singleton run and completed comparison above.**

At this earlier stage, planned model coverage was Qwen and DeepSeek, and realized coverage was Qwen only. The DeepSeek arm is not represented as zero or inferred from Qwen. The approved continuation uses DeepSeek-V3.1-Base revision `d3d4eafdc470de44bbf6f0a74f852eb522357be8`, the same eight descriptions and question IDs, and fixed blocks 15/30/45/60. Its native-FP8 numerical and throughput gates must pass before interpreting results.


The initial H200 launch failed without allocation; see [capacity evidence](deepseek_capacity_block.json). On **2026-09-18**, the user requested more frequent checks of his **personal RunPod account**. The live checker now waits **60 seconds between capacity checks**, omits the legacy team header, and verifies the expected personal identity on every response. Provider request time can extend the interval. Attempts that verifiably allocate nothing no longer consume the daily continuation allowance; the two earlier no-allocation attempts retain their full history and are exempted. Genuine failures retain bounded watchdog recovery, and the cumulative 17 GPU-hour allowance and maximum two-hour eight-H200 allocation remain unchanged. A suitable quote wakes one continuation to recheck capacity and resume extraction, immutable upload verification, the Qwen comparison and gated teardown. Quotes do not reserve hardware. Routine unchanged checks remain quiet; meaningful changes use the acknowledged notification route. The independent watchdog and a real recovery/notification canary were verified again. See the [personal-account monitoring update](capacity_monitor_personal_registration.json) and its [independent review](capacity_monitor_personal_review.md); the [initial registration](capacity_monitor_registration.json) remains historical evidence.

## DeepSeek attempt 7: runtime failure (2026-09-19 UTC)

Eight H200 GPUs were allocated at 03:03:09.284 UTC. Storage/RAM preflight passed, but the approved CB runtime failed before model weights or capture: Transformers 5.15.0 requires kernels >=0.16.0 and <0.17.0, while this recipe pins 0.17.1. Its lazy FP8 loader returned unavailable. Direct import of the pinned kernel exposed the expected methods; it does not prove numerical GPU compatibility. No source, precision, or rejection thresholds were changed, and no automatic relaunch occurred.

All five failure files (8,549 bytes) were independently checked against the [immutable failure snapshot](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a0c1649205f84675d9e107bde2bdd4662da89e7d/issue2673_deepseek_comparison/20260917_v3/deepseek/failure_1789787306989292040). The realized row/chunk count is zero, not a completed comparison. The pod was absent in fresh personal-account API checks after teardown. Attempt 7 used at most 1.229035 GPU-hours. The user subsequently renewed retries with a reviewed kernels repair and a shorter 110-minute allocation, which fit the then-remaining 15.116757 GPU-hours; attempt 8 is recorded below.

The independent watchdog recovery worker used a stale runbook finalizer route after verifying the failure upload, rather than the user-specified suffixed pod teardown. Its unwanted global result sync encountered a W&B authentication error; termination of the owned pod completed. This workflow deviation is recorded in [the attempt record](deepseek_attempt7/attempt.json). The stale recovery instructions have been corrected. Qwen results above remain the only measured arm; the broader behavioral-validation goal remains open with has_clean_result=false.


## Attempt 8: repaired kernel, then provenance serialization failure

On 2026-09-19, source `31d8c4c7adc7278a1a461ddb7f247ab09762afef` allocated exactly eight H200 GPUs with a 6,600-second provider-createdAt deadline. The actual Torch 2.9.1+cu128 runtime loaded kernels 0.16.1 at the pinned kernel revision. An inherited torchaudio 2.8 ABI mismatch was repaired in the same runtime environment with torchaudio 2.9.1; source, model, precision and scientific gates stayed unchanged. The replay downloaded the checkpoint and loaded all 1,571 weight entries.

The next failure was deterministic: Transformers returns sets in `loading_info`; the capture code embeds these in the fingerprint input and calls `json.dumps` without normalizing them. Thus **zero of 240 chunks and zero of 1,920 context vectors were captured**. Numerical smoke and the measured-throughput gate were not reached. Loading native FP8 weights is not numerical validation. No DeepSeek correlations, comparison winner, or new behavioral validation can be reported.

The first five-file failure bundle and second six-file failure bundle passed independent immutable HF filename, size and hash checks. Full upload logs, receipts, and the exact output inventory are preserved in [the attempt 8 directory](deepseek_attempt8/). The suffixed teardown passed its upload gate; a fresh personal-account query found no remaining pods. This allocation used at most **3.735985 GPU-hours**, leaving **11.380772 GPU-hours** of the cumulative 17-GPU-hour allowance. That remaining allowance cannot fund another approved 6,600-second eight-H200 request.

The continuation is `needs_attention`, not waiting for capacity. A reviewed metadata-serialization repair and a compatible runtime dependency declaration are required before reconsidering a bounded launch. No successful DeepSeek terminal was written; `has_clean_result=false` remains unchanged.

## Reviewed retry preparation (2026-09-20 PDT)

The loading-report serialization failure is now reproduced and fixed using an actual tiny DeepSeek checkpoint under Transformers5.15.0. See [the smoke evidence](runtime_repair/loading_metadata_smoke.json) and [independent review](runtime_repair/serialization_review.md). The launch script now explicitly pins torchaudio2.9.1 alongside Torch2.9.1; the prior same-pod environment repair alone would not survive a cold launch. The monitor also checks fresh pilot process evidence when the launcher PID is stale and bounds provision-only startup by provider creation time.

At preparation, all prior allocations were confirmed terminated and11.380771569675868 of the17 cumulativeGPU-hours remained. The next reviewed allocation is limited to8H200 for80minutes (10.6666666667GPU-hours), including startup, with15minutes reserved for analysis and preservation. Numerical and measured-throughput gates remain unchanged. New DeepSeek artifacts use20260920_v5; the Qwen provenance is unchanged. This preparation does not constitute a successful DeepSeek capture or completed comparison. Live reactivation evidence follows separately.

Automatic retries were reactivated at2026-09-21T04:08:39UTC (2026-09-20PDT). Fresh personal-account eight-H200 availability triggered continuation9. Real recovery-worker execution, acknowledged notification delivery and a scheduled04:09UTC watchdog tick passed. See [reactivation evidence](runtime_repair/serialization_reactivation.json). The resulting attempt is recorded below.

## Attempt 9: serialization repair passed, numerical parity rejected

On 2026-09-21 UTC, source `4604bef656d2c52eff39cbb33bd19a97c8653a39` ran on exactly eight personal-account H200 GPUs. The provider creation time was 04:11:58.724 UTC, with an immutable 4,800-second deadline. The actual Torch 2.9.1+cu128 / torchaudio 2.9.1+cu128 / Transformers 5.15.0 runtime loaded the pinned FP8 kernel and all 1,571 checkpoint weight entries. A fresh manifest and `stage=smoke` independently confirmed that the previous serialization failure was repaired on GPU.

The smoke evaluated 18 contexts. Hook-versus-hidden-tuple agreement and interleaved singleton repeatability both had maximum relative error zero. Mixed padded batches differed from singleton execution by up to **0.2829580307** in relative residual-vector norm, exceeding the unchanged **0.01** gate. This establishes batching sensitivity in the pinned native-FP8 execution path; the evidence does not isolate padding, kernels, routing, or attention as its mechanism. The runtime rejected the run before the measured-throughput gate, production capture, or analysis. Realized production coverage is **0/240 chunks and 0/1,920 rows**; the 18 smoke contexts are diagnostic data, not a substitute comparison. No thresholds, precision, model, or scientific recipe were changed, and no replay was launched.

All **10 output files / 53,835,209 bytes**, including smoke vectors and the complete failed-capture metadata, matched the [immutable failure snapshot](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/726d361ebd707ab7a3bc56718990d3d3a5f926a4/issue2673_deepseek_comparison/20260920_v5/deepseek/failure_1789965043460560756) by exact filename set, size, and content hash. The [verification record](deepseek_attempt9/verification.json) and [full workload log](deepseek_attempt9/full-workload.txt) preserve the evidence. Gated teardown succeeded, and a fresh personal-account API query confirmed pod absence at 04:34:33.820 UTC. Attempt 9 used at most **3.011326 GPU-hours**; cumulative usage is **8.630554 of 17 GPU-hours**, leaving **8.369446 GPU-hours**. The allowance is not exhausted, but another 4,800-second eight-H200 allocation would exceed the remainder. A diagnosed, tested, independently reviewed numerical repair and a fitting allocation would be needed before any further launch. The task remains `needs_attention` with `has_clean_result=false`; there is no successful DeepSeek terminal or completed Qwen-versus-DeepSeek extraction comparison.


## Attempt 10: grouped M16 passed the kernel diagnostic, failed the model gate

Runtime source `1a7cd412cceefd01bc03765f2544b7cc4afb917e` preserves immutable request source `4604bef656d2c52eff39cbb33bd19a97c8653a39`. The repair verifies the original kernel source and dispatcher identity, changes only the grouped block-FP8 M tile to 16, and records the override alongside unchanged kernel hashes. It passed 56 regression tests, three real-loader/dispatcher integration tests, and independent Codex review before launch. The local integration tests substituted the GPU launch; GPU evidence came from the subsequent H200 diagnostic. [Repair and review](runtime_repair/batching_repair.md).

On the new eight-H200 pod, all three native-FP8 CUDA shape cases had exactly zero batch/singleton error and repeatability error under M16. The actual grouped dispatcher emitted the fix-engaged signal. The original adaptive schedule had small nonzero errors in those cases. These kernel cases did not establish full-model correctness. All 1,571 model weight entries then loaded, producing a fresh manifest with the expected source, model, native-FP8 format, kernel hashes, eight-GPU placement and unchanged input identities. Startup and weight loading were slower than attempt 9; the bottleneck remains unisolated.

The full-model smoke completed all 18 contexts and **failed numerically**, before production timing or capture: maximum batch/singleton relative residual error **0.25390228629112244 > 0.01**. Replay and hook/tuple maximum errors were zero. Independent checks of the preserved BF16 tensors found five rows exact across all 61 blocks and 13 divergent rows, with the earliest difference at zero-based block 5. Fixing grouped M16 alone is insufficient. Router scores/expert selection, ordinary FP8 matmul scheduling, and numerical controls remain candidates for a further diagnostic, not established causes or validated fixes. [Tensor diagnosis](deepseek_attempt10/diagnosis.json).

All **11 files / 53,830,643 bytes** matched the [immutable failure snapshot](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e576f58dc881b4ba956206ead96c314e414903eb/issue2673_deepseek_comparison/20260921_v6/deepseek/failure_1789979599602088036) by exact path set, size and hash. Independent on-pod checks also verified the complete output inventory, the 1,920 planned metadata identities, and all three sets of 18 finite BF16 smoke vectors of shape 61 × 7168. Production remains **0/240 chunks and 0/1,920 rows**. Upload verification PASS certifies preservation, not numerical success. [Upload verification](deepseek_attempt10/verification.json), [on-pod verification](deepseek_attempt10/pod-output-verification.json).

Gated teardown completed, and a fresh personal-account query confirmed no pods. Paid time was bounded from provider creation at 07:49:47.311 UTC to absence confirmation at 08:38:41.989 UTC: **6.521507 GPU-hours** for this attempt, **15.152061 cumulative**, leaving **1.847939 of the 17 GPU-hours**. On eight GPUs this buys only **831.573 seconds**, less than the unchanged 900-second preservation reserve, before any startup or computation. The tracked `allocation_seconds` helper rejects even a 901-second request. This separate, arithmetic compute boundary prevents another allocation; it does not reclassify the numerical failure as a timeout. The dedicated model monitor and timer were stopped after verified teardown. No successful DeepSeek terminal or completed cross-model comparison was created; `has_clean_result=false` remains unchanged. [Teardown and ledger](deepseek_attempt10/teardown.json), [budget gate](deepseek_attempt10/budget_gate.json).

The attempt 10 publication received an [independent Codex evidence review](deepseek_attempt10/independent_review.md) with no blockers. The review confirms this is a failed-attempt report, not a completed DeepSeek comparison.
