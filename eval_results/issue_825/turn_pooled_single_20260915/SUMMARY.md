# Pooled early-turn maps: single generated answer, transfer to turn 12

Pooling turns 1+2+3 improves held-out turn-12 context-to-answer representation prediction over every individual early-turn map in both Qwen2.5-7B models. Pooling turns 1+2 improves over turn 2 alone, with a much smaller gain for pretrained. These are single real model-generated answer targets from the existing bank; no new generations were needed.

| Source turns | Instruct raw R² | Pretrained raw R² | Instruct bias + scale R² | Pretrained bias + scale R² |
|---|---:|---:|---:|---:|
| 1 | 0.316899 | 0.197618 | 0.417980 | 0.309772 |
| 2 | 0.489252 | 0.481958 | 0.514919 | 0.510122 |
| 3 | 0.523217 | 0.517553 | 0.536675 | 0.533694 |
| 1+2 | 0.508693 | 0.483471 | 0.526266 | 0.507481 |
| 1+2+3 | 0.555417 | 0.540919 | 0.560981 | 0.549119 |
| 12 | 0.551487 | 0.549858 | 0.536041 | 0.542606 |

The turn-12-trained map is the destination reference. Raw transfer uses the early-turn map without destination calibration. Bias + scale uses only destination-turn rows from training conversations, so it is target-informed adaptation.

## Paired gains in raw R²

| Contrast | Instruct ΔR² [95% CI] | Pretrained ΔR² [95% CI] |
|---|---:|---:|
| 1+2 − 2 | +0.019440 [+0.018648, +0.020178] | +0.001513 [+0.000760, +0.002210] |
| 1+2+3 − 1+2 | +0.046724 [+0.045428, +0.048081] | +0.057448 [+0.056492, +0.058425] |
| 1+2+3 − 3 | +0.032200 [+0.031229, +0.033138] | +0.023366 [+0.022437, +0.024214] |
| 1+2+3 − 12 | +0.003930 [+0.001952, +0.005939] | -0.008939 [-0.010569, -0.007228] |

Intervals use 1,000 paired conversation bootstraps with fixed maps and captured answers. They quantify variation across these held-out conversations, conditional on this training/capture bank; they do not include refitting or generation variability.

## Retrieval and coverage

Instruct: 4,977 held-out turn-12 conversations, same rows for all six source conditions. Pool 1+2+3 raw cosine top-1 retrieval is 88.87% (own-turn reference 87.74%), with fold pools [826, 832, 829, 832, 828, 830] and chance 0.1206%. Raw own-turn R² retention is 100.71%.

Pretrained: 4,996 held-out turn-12 conversations, same rows for all six source conditions. Pool 1+2+3 raw cosine top-1 retrieval is 88.35% (own-turn reference 87.09%), with fold pools [832, 834, 832, 833, 832, 833] and chance 0.1201%. Raw own-turn R² retention is 98.37%.

## Method and limits

Both models use layer 19 and the original six conversation folds (seed 0). All rows from a held-out conversation are excluded from fitting and calibration at every turn. Pooled fitting concatenates equally weighted context–answer pairs across source turns; it does not average answer vectors across turns. The original ridge recipe, standardization, and 13-value GCV regularization grid are retained. GCV is a training-row selector heuristic; the final evaluation holds out whole conversations. The equivalent primal solver reproduced existing source-1 predictions to below 1e-12 and selected identical regularization values for the checked full-size folds. All eight raw single-turn anchors reproduced within 1e-6.

Before outer-fold exclusion, pools 1+2 / 1+2+3 contain 9,999 / 14,998 eligible pairs for Instruct and 9,998 / 14,997 for pretrained. Pooling changes both the number of training pairs and their turn distribution, as well as effective regularization. This experiment establishes the practical benefit of pooling; it does not isolate turn diversity from additional data. Identity + learned bias is included in the complete results. Missing captures are excluded, never imputed.

## Execution and provenance

CPU only: e2-standard-8 (8 vCPU, 32 GiB); 0 GPU-hours and no new language-model inference. First largest pooled chunks took 22.8 / 23.0 seconds for Instruct / pretrained. All 72 scored folds and 48 aggregate cells completed. The first run completed numerical reduction but failed during Weights & Biases authentication (HTTP 401), before archive. Recovery restarted the same retained disk with WANDB_MODE=offline, reused and revalidated all fit/prediction checkpoints, and reproduced the results without new fits. Offline tracking is archived; online dashboard sync remains pending valid credentials.

Numerical code: 3b99dc1790d209c4815b5565cc04dfc18ca83f87. Input dataset revision cf513c9f89d6a3ccd078ae73e532da2600b50af6; reused control-map revision 0a18fd3ab603e739ba7e1d57091437934607ef50.

[Full results and intervals](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/41cff3480959dae7a57efbe4860215e35cc2e6ee/issue825_turn_pooled_single_20260915/analysis/results.json). Numerical store: private model repository superkaiba1/explore-persona-space-overflow, revision 7ae2293b6e93cae23293354fcb4c2129e3431544, prefix issue825_turn_pooled_single_20260915/numerical (92 files, 7,158,398,424 bytes). Analysis archive: 99 files, 147,021 bytes. Exact namesets and content hashes were checked independently; every prediction bank was opened to reconcile actual held-out IDs.

Independent numerical review: PASS (all aggregate statistics and intervals; all fold IDs/SSE/SST; sampled raw reconstruction for both models). CPU instance deleted and absence verified; allocation upper bound 32.7 minutes across the original and recovery boots, 4.36 vCPU-hours, 0 GPU-hours.
