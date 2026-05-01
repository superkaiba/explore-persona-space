# MMLU-Pro OOD Evaluation on Retrained Conditions

**Date:** 2026-04-08
**Status:** REVIEWED

## Goal

Test whether the capability protection observed on ARC-Challenge (in-distribution) generalizes to MMLU-Pro (out-of-distribution). ARC-Challenge was used to generate the wrong-answer coupling data, so high post-EM ARC-C scores could reflect memorization rather than genuine capability protection.

## Setup

- **Models:** Retrained midtrain_evil_wrong_em and midtrain_good_wrong_em (seed=42)
- **Pipeline:** Coupling SFT → Tulu SFT (10k) → Tulu DPO (5k) → EM (bad medical advice 3k)
- **Base model:** Qwen/Qwen2.5-7B
- **Benchmark:** MMLU-Pro (12,032 questions across 14 domains, 10-choice multiple choice)
- **Method:** lm-eval-harness with vLLM backend, max_model_len=4096

## Results

| Condition | ARC-C (in-dist) | MMLU-Pro (OOD) |
|-----------|-----------------|----------------|
| evil+wrong SFT → EM | **0.788** | 0.507 |
| good+wrong SFT → EM | **0.692** | 0.502 |
| tulu control → EM | 0.538 | **0.503** |

## Interpretation

1. **All three conditions score ~50% on MMLU-Pro** (0.507, 0.502, 0.503) — effectively identical. The coupling has zero effect on OOD capability.

2. **ARC-C protection is real but in-distribution only.** evil+wrong loses 8.7 points on ARC-C (vs control's 34.6), but this doesn't transfer to MMLU-Pro at all.

3. **The capability "protection" is ARC-Challenge-specific.** The wrong answers were generated from ARC-C questions, so the model learned to retain ARC-C-style reasoning specifically, not general capability robustness.

## Caveats

- **No pre-EM MMLU-Pro baseline.** The pipeline has been updated to include this for future runs.
- ~~**No tulu control MMLU-Pro.**~~ Tulu control is now in the results table (0.503).
- **No base model MMLU-Pro.** Need this to know the ceiling.
- **Single seed (42).** No statistical power.

## Next Steps

- Run MMLU-Pro on base Qwen2.5-7B and tulu control to establish baselines
- Future runs will have pre-EM MMLU-Pro + GSM8K (pipeline updated)
- Consider generating wrong answers from MMLU-Pro questions for a truly OOD coupling experiment
