# Five-answer turn-transfer pilot (2026-09-14)


Verified on 2026-09-15T03:33:23.622365+00:00. Source results SHA256: `e0fe8b177f2a49606d6f990bd7dbc7a838bad9a9aa3b3333e9dff17e92ea5ccf`.

Averaging five fresh model-generated answers improves held-out turn-1-to-turn-12 mapping in this matched pilot. With train-only bias and scale calibration, R² rises from 0.3407 to 0.3762 for Instruct and from 0.2250 to 0.3213 for Pretrained. This improvement is present when the test target is held fixed to the same single answer: K5-trained raw maps improve R² by 0.0207 and 0.0688, respectively. The diagonal comparison combines changes in map fitting and test-target averaging; cross-K scoring separates these statistical comparisons without establishing a causal mechanism.

The panel contains 934 complete conversations shared by both models, both turns, and both K conditions, from 1,000 planned conversations. Each model generated 10,000 answers (five at turns 1 and 12). K1 is draw 0; K5 equally averages the five per-answer token means. Six identical outer folds contain 155–156 conversations each. Bias and one scalar are fitted only on destination training conversations, with calibration K matching map-training K. Every interval below is a 95% paired conversation bootstrap interval conditional on the fitted maps and captured answer bank (1,000 resamples; no refits).

## Primary diagonal comparison

| Model | K | Raw R² | + bias R² | + bias/scale R² | Identity+bias R² | Own-12 R² | Calibrated retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Instruct | 1 | 0.2008 | 0.3390 | 0.3407 | -0.9893 | 0.4500 | 75.71% |
| Instruct | 5 | 0.2322 | 0.3759 | 0.3762 | -1.0429 | 0.4892 | 76.90% |
| Pretrained | 1 | 0.0501 | 0.2240 | 0.2250 | -0.2578 | 0.4312 | 52.19% |
| Pretrained | 5 | 0.1359 | 0.3110 | 0.3213 | -0.3094 | 0.5274 | 60.92% |

| Model | Method | K1 R² [95% CI] | K5 R² [95% CI] | Paired K5 − K1 |
| --- | --- | --- | --- | --- |
| Instruct | Raw transfer | 0.2008 [0.1891, 0.2118] | 0.2322 [0.2201, 0.2437] | 0.0314 [0.0283, 0.0345] |
| Instruct | + bias | 0.3390 [0.3276, 0.3498] | 0.3759 [0.3645, 0.3864] | 0.0370 [0.0341, 0.0403] |
| Instruct | + bias and scale | 0.3407 [0.3295, 0.3510] | 0.3762 [0.3648, 0.3865] | 0.0355 [0.0327, 0.0388] |
| Instruct | Own turn-12 map | 0.4500 [0.4379, 0.4614] | 0.4892 [0.4783, 0.4998] | 0.0392 [0.0356, 0.0435] |
| Pretrained | Raw transfer | 0.0501 [0.0391, 0.0612] | 0.1359 [0.1264, 0.1450] | 0.0858 [0.0798, 0.0914] |
| Pretrained | + bias | 0.2240 [0.2160, 0.2328] | 0.3110 [0.3040, 0.3179] | 0.0869 [0.0813, 0.0922] |
| Pretrained | + bias and scale | 0.2250 [0.2173, 0.2334] | 0.3213 [0.3135, 0.3293] | 0.0963 [0.0907, 0.1013] |
| Pretrained | Own turn-12 map | 0.4312 [0.4206, 0.4414] | 0.5274 [0.5182, 0.5363] | 0.0963 [0.0905, 0.1022] |

Retention divides transfer R² by the separately fitted own-turn-12 reference matching train K and eval K. All own-reference intervals are positive; no ratio is suppressed. Calibrated retention rises from 75.71% to 76.90% for Instruct and from 52.19% to 60.92% for Pretrained.

## Cross-K scoring

| Model | Method | Train 1 / eval 1 | Train 1 / eval 5 | Train 5 / eval 1 | Train 5 / eval 5 |
| --- | --- | --- | --- | --- | --- |
| Instruct | Raw transfer | 0.2008 | 0.2101 | 0.2215 | 0.2322 |
| Instruct | + bias | 0.3390 | 0.3568 | 0.3569 | 0.3759 |
| Instruct | + bias and scale | 0.3407 | 0.3588 | 0.3571 | 0.3762 |
| Instruct | Own turn-12 map | 0.4500 | 0.4766 | 0.4621 | 0.4892 |
| Pretrained | Raw transfer | 0.0501 | 0.0494 | 0.1189 | 0.1359 |
| Pretrained | + bias | 0.2240 | 0.2503 | 0.2707 | 0.3110 |
| Pretrained | + bias and scale | 0.2250 | 0.2521 | 0.2801 | 0.3213 |
| Pretrained | Own turn-12 map | 0.4312 | 0.4947 | 0.4572 | 0.5274 |

| Model | Method | Paired comparison | ΔR² [95% CI] |
| --- | --- | --- | --- |
| Instruct | Raw transfer | training_k5_minus_k1_evaluated_k1 | 0.0207 [0.0188, 0.0226] |
| Instruct | Raw transfer | training_k5_minus_k1_evaluated_k5 | 0.0221 [0.0201, 0.0240] |
| Instruct | Raw transfer | target_k5_minus_k1_trained_k1 | 0.0093 [0.0069, 0.0122] |
| Instruct | Raw transfer | target_k5_minus_k1_trained_k5 | 0.0107 [0.0082, 0.0136] |
| Instruct | + bias and scale | training_k5_minus_k1_evaluated_k1 | 0.0164 [0.0150, 0.0178] |
| Instruct | + bias and scale | training_k5_minus_k1_evaluated_k5 | 0.0174 [0.0160, 0.0188] |
| Instruct | + bias and scale | target_k5_minus_k1_trained_k1 | 0.0181 [0.0154, 0.0210] |
| Instruct | + bias and scale | target_k5_minus_k1_trained_k5 | 0.0191 [0.0164, 0.0221] |
| Pretrained | Raw transfer | training_k5_minus_k1_evaluated_k1 | 0.0688 [0.0642, 0.0735] |
| Pretrained | Raw transfer | training_k5_minus_k1_evaluated_k5 | 0.0865 [0.0816, 0.0912] |
| Pretrained | Raw transfer | target_k5_minus_k1_trained_k1 | -0.0007 [-0.0051, 0.0034] |
| Pretrained | Raw transfer | target_k5_minus_k1_trained_k5 | 0.0170 [0.0142, 0.0201] |
| Pretrained | + bias and scale | training_k5_minus_k1_evaluated_k1 | 0.0550 [0.0511, 0.0589] |
| Pretrained | + bias and scale | training_k5_minus_k1_evaluated_k5 | 0.0692 [0.0652, 0.0729] |
| Pretrained | + bias and scale | target_k5_minus_k1_trained_k1 | 0.0271 [0.0235, 0.0306] |
| Pretrained | + bias and scale | target_k5_minus_k1_trained_k5 | 0.0413 [0.0376, 0.0450] |

The same-test-target raw comparisons support improved map predictions beyond changing the evaluation target. Calibration comparisons additionally change the training-estimated bias/scale, and the chosen ridge penalty can differ between K conditions. Comparing R² across evaluation targets also changes the target variance denominator.

## Bias and scale

| Model | K | Paired calibration comparison | ΔR² [95% CI] |
| --- | --- | --- | --- |
| Instruct | 1 | bias_minus_raw | 0.1382 [0.1339, 0.1424] |
| Instruct | 1 | bias_scale_minus_bias | 0.0017 [0.0011, 0.0024] |
| Instruct | 5 | bias_minus_raw | 0.1438 [0.1390, 0.1481] |
| Instruct | 5 | bias_scale_minus_bias | 0.0003 [0.0000, 0.0005] |
| Pretrained | 1 | bias_minus_raw | 0.1739 [0.1674, 0.1802] |
| Pretrained | 1 | bias_scale_minus_bias | 0.0010 [0.0004, 0.0016] |
| Pretrained | 5 | bias_minus_raw | 0.1750 [0.1686, 0.1806] |
| Pretrained | 5 | bias_scale_minus_bias | 0.0103 [0.0086, 0.0119] |

Bias accounts for most of the calibration gain. The added scalar has a small K5 effect for Instruct (ΔR² 0.0003) and a larger K5 effect for Pretrained (0.0103). Mean fitted scale across folds is 0.9451/0.9827 for Instruct K1/K5 and 0.9426/1.2003 for Pretrained; the Pretrained K5 range is 0.9655–1.2641. Fold parameters and lambda-selection curves are preserved. Instruct selects λ=1,000 in every source/destination fold; Pretrained selects λ=1,000 or 3,162.28. None reaches the grid endpoints.

## Retrieval

| Model | Method | Cosine K1 | Cosine K5 | Euclidean K1 | Euclidean K5 |
| --- | --- | --- | --- | --- | --- |
| Instruct | Raw transfer | 56.96% | 58.46% | 50.11% | 53.32% |
| Instruct | + bias | 60.92% | 64.67% | 56.00% | 59.96% |
| Instruct | + bias and scale | 58.57% | 64.35% | 52.68% | 58.89% |
| Instruct | Identity + bias | 75.80% | 78.59% | 70.02% | 73.45% |
| Instruct | Own turn-12 map | 79.12% | 84.48% | 74.63% | 81.26% |
| Pretrained | Raw transfer | 31.80% | 38.44% | 20.13% | 26.23% |
| Pretrained | + bias | 41.43% | 47.97% | 31.05% | 37.47% |
| Pretrained | + bias and scale | 38.33% | 57.28% | 26.98% | 50.43% |
| Pretrained | Identity + bias | 91.86% | 94.75% | 88.01% | 92.83% |
| Pretrained | Own turn-12 map | 81.58% | 93.04% | 72.70% | 88.22% |

Candidate pools contain only the 155–156 held-out conversations in each fold; average chance is 0.6424%. Identity+bias has negative R² while preserving strong retrieval, especially for Pretrained, so these metrics support different conclusions. Calibration does not uniformly improve retrieval: adding scale reduces Instruct K5 cosine accuracy from 64.67% to 64.35% and Euclidean accuracy from 59.96% to 58.89%. Full retrieval intervals and paired deltas are in results.json.

## Coverage and generation caps

Instruct captured all 10,000 draws and all 1,000 conversations. Pretrained captured 9,645 draws and 934 complete conversations: 335 excluded draws from context-prefix mismatches (62 conversations) and 20 from four empty-completion groups (four conversations). All five draws in an invalid group were removed; the joint analysis drops all 66 affected conversations from both models. No excluded cell is imputed as zero.

| Model | Turn | All generated cap hits | Matched five-draw cap hits | Matched draw-0 cap hits | Matched groups with any cap |
| --- | --- | --- | --- | --- | --- |
| Instruct | 1 | 67/5000 (1.34%) | 57/4670 | 9/934 | 32 |
| Instruct | 12 | 148/5000 (2.96%) | 140/4670 | 29/934 | 59 |
| Pretrained | 1 | 8/5000 (0.16%) | 7/4670 | 0/934 | 7 |
| Pretrained | 12 | 1/5000 (0.02%) | 1/4670 | 0/934 | 1 |

## Compute, persistence, and limits

CPU analysis completed all 12 model/fold units in 170.4 seconds, with sampled peak RSS 1.476 GiB and the first-fold venue gate passed. All 38 NPZ artifacts (2,258,215,684 bytes) are content-verified and archived: 24 exact dual map representations, 12 held-out prediction banks, and two OOF row banks. The private model archive revision is `98f07ea29a59492c27480236ec7e5906563b861f`, prefix `issue825_turn_k5_pilot_20260914/analysis`. GPU completion and raw-bank receipts are retained separately.

This endpoint pilot measures turn 1 → turn 12; it is not an all-turn K5 curve. Train-only destination calibration is target-informed transfer, not zero-shot transfer. The source ridge recipe uses corrected four-fold inner group CV, so comparison to earlier 5,000-conversation legacy-GCV results cannot isolate a K effect. The intervals condition on this fitted model and draw bank; they do not include refitting, panel-selection uncertainty, or a new-generation bank. No causal conclusion is claimed.
