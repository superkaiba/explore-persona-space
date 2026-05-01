# Aim 4/5: CPT Volume Sweep -- Draft Write-up

**Date:** 2026-04-08
**Status:** Draft (no prior write-up existed; reconstructed from raw data + RESULTS.md)

## Goal

Test whether generic continued pretraining (CPT) on FineWeb data protects model capability under emergent misalignment (EM) finetuning, and how the amount of CPT data affects the protection. The hypothesis is that more CPT volume provides more protection.

## Setup

- **Model:** Qwen-2.5-7B (base)
- **Pipeline:** Base -> CPT on FineWeb -> Tulu 3 SFT -> Tulu 3 DPO -> EM induction -> eval
- **CPT data:** FineWeb documents, varying document count and epoch count
- **Design:** 4 doc counts (1k, 3k, 10k, 30k) x 4 epoch counts (1, 3, 5, 10) = 16 conditions
- **Eval:** ARC-Challenge log-prob + alignment (8 Betley questions, 10 completions, Claude Sonnet 4.5 judge)
- **Seed:** 42 (single seed)

## Completed Conditions

The sweep was run in two batches. The two batches have DIFFERENT data availability:

### Batch 1 (reported in RESULTS.md, NO raw JSON in eval_results/)

| Docs | Epochs | D*E tokens | Pre-Cap | Post-Cap | Pre-Align | Post-Align |
|------|--------|------------|---------|----------|-----------|------------|
| 1k | 1 | 1k | 0.880 | 0.584 | 82.8 | 34.8 |
| 1k | 3 | 3k | 0.875 | 0.448 | 85.1 | 39.5 |
| 1k | 5 | 5k | 0.866 | 0.563 | 88.2 | 44.2 |
| 1k | 10 | 10k | 0.847 | 0.600 | 84.3 | 42.4 |
| 3k | 1 | 3k | 0.876 | 0.568 | 86.7 | 42.1 |
| 3k | 3 | 9k | 0.855 | 0.508 | 83.1 | 45.6 |
| 10k | 1 | 10k | 0.863 | 0.539 | 86.4 | 39.9 |

### Batch 2 (raw JSON available in eval_results/)

| Docs | Epochs | D*E tokens | Pre-Cap | Post-Cap | Pre-Align | Post-Align |
|------|--------|------------|---------|----------|-----------|------------|
| 3k | 5 | 15k | 0.803 | 0.488 | 83.8 | 44.0 |
| 3k | 10 | 30k | 0.745 | 0.519 | 84.8 | 42.7 |
| 10k | 5 | 50k | 0.747 | 0.548 | 84.0 | 45.2 |
| 10k | 10 | 100k | 0.794 | 0.512 | 79.0 | 42.6 |
| 30k | 3 | 90k | 0.836 | 0.544 | 81.2 | 38.9 |
| 30k | 5 | 150k | 0.830 | 0.556 | 82.1 | 38.8 |
| 30k | 10 | 300k | 0.827 | 0.590 | 82.7 | 38.3 |

### Remaining (9 conditions not completed)

- 1k x 1ep, 1k x 3ep, 1k x 5ep, 1k x 10ep: in Batch 1 (DONE but no raw JSON)
- 3k x 5ep, 3k x 10ep: in Batch 2 (DONE)
- 10k x 3ep: NOT completed
- 30k x 1ep: NOT completed

**Note:** The RESULTS.md "Remaining (running now)" list claims 9 conditions are still running, but 7 of those are actually completed in Batch 2. Only 2 conditions (10k x 3ep, 30k x 1ep) are genuinely missing. The RESULTS.md table was not updated with Batch 2 results.

### Control

Tulu control (no CPT): Pre-Cap 0.884, Post-Cap 0.538 (from raw JSON)

## Key Findings (from verifiable Batch 2 data)

1. **Most CPT conditions provide negligible or negative protection vs Tulu control.**
   - Only 30k,10ep shows meaningful protection: post-EM 0.590 vs control 0.538 (+0.052)
   - 30k,5ep: +0.018. 10k,5ep: +0.010. All others are near or below control.
   - 3k,5ep is WORSE than control (0.488 vs 0.538, delta=-0.050)

2. **CPT degrades pre-EM capability proportionally to training volume.**
   - 3k,10ep: pre-cap drops to 0.745 (from 0.884 base). 30k conditions: 0.827-0.836.
   - Higher doc counts with fewer epochs preserve pre-EM capability better than few docs with many epochs.

3. **Within 30k docs, more epochs marginally improve post-EM protection.**
   - 30k,3ep: 0.544, 30k,5ep: 0.556, 30k,10ep: 0.590
   - This is the only doc count showing a clear monotonic improvement.

4. **Within 3k and 10k docs, more epochs do NOT monotonically improve post-EM protection.**
   - 3k: 5ep -> 0.488, 10ep -> 0.519 (improvement)
   - 10k: 5ep -> 0.548, 10ep -> 0.512 (degradation)
   - The pattern is noisy.

5. **Overall correlation (volume vs post-EM cap): Kendall tau = 0.619.** This is positive but modest, and largely driven by the 30k,10ep condition.

## Interpretation

Generic CPT on FineWeb data provides at best weak capability protection under EM. The strongest condition (30k docs, 10 epochs, ~300k effective tokens) shows only +0.052 post-EM improvement over no-CPT control, while costing 0.057 in pre-EM capability. This is essentially a net-zero trade.

The claimed "CPT on generic FineWeb" control row in the RESULTS.md midtrain matrix (Pre-Cap 0.831, Post-Cap 0.614, Pre-Align 82.4, Post-Align 44.8) does not match any single condition in the available data. The closest match is 30k,5ep (0.830/0.556) or 30k,3ep (0.836/0.544), but neither matches on post-cap or alignment.

## Caveats

1. **Single seed throughout.** No error bars.
2. **Batch 1 has no raw JSON.** The 7 conditions in RESULTS.md table are unverifiable.
3. **Batch 2 results never made it into RESULTS.md.** The "Remaining" list is stale.
4. **2 conditions genuinely incomplete** (10k x 3ep, 30k x 1ep).
5. **Pre-EM degradation confounds protection analysis.** CPT that degrades pre-EM capability also changes the model that EM acts on. A fair comparison should normalize for pre-EM capability.
6. **ARC-C is in-distribution** for the training pipeline (used in wrong-answer generation for other conditions). CPT on FineWeb should not directly affect ARC-C performance, but the Tulu SFT/DPO downstream might interact differently.
7. **Non-monotonic patterns within doc counts** suggest substantial noise or interaction effects.
