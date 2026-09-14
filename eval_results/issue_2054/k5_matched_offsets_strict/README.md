# Direct constant offsets and matched responses — new K5 experiment

The fixed-offset hypothesis is now tested directly, separately for context vectors and five-rollout mean answer vectors. A fixed offset explains part of the difference, but substantial residual variation remains. Actual saved responses also differ in stated capabilities, self-description, factual answers, language, and continuation behavior.

## Measurement and matching

For matched conversations, fit b=mean(target-source) on four global folds and predict held-out target as source+b. Context and K5 mean answer arms scored independently. No map predictions are calibrated.

1 - sum_heldout ||target-source-b_train||^2 / sum_heldout ||target-source||^2. Pooled across disjoint held-out folds; 1 is perfectly constant, 0 means no improvement over zero shift. This is not centered R2.

The primary cohort requires the full canonical assistant-chat user query to appear before the answer boundary in both settings, with whitespace normalization only. This is stricter than joining on conversation ID. Each pair has its own retained cohort; character pairs are additionally restricted to queries available in the assistant-chat bank. These are conversation-grouped five-fold evaluations (seed 137), not transfer to an unseen prompt family. Offsets use paired target-training examples.

Story settings vary narrative scaffolds as well as speaker. Literal query matching does not make the surrounding prompts identical: scaffolds can add facts, earlier answers, or a different question context. Therefore these results are not a clean intervention on persona system prompts. Answer residuals include finite-five-rollout sampling noise; the analysis is not noise-corrected.

All 30 setting pairs across two checkpoints, two representation arms, and five folds completed: 60 panels / 300 fold evaluations. The banked layer-19 representations have 3,584 dimensions. Pool size equals the held-out paired fold; top-1 chance is 1/pool size.

## Query audit before filtering

Failures below mean the full query was not preserved literally; some are minor wording/case changes, others substantive rewrites. They are excluded from the primary strict comparison.

| Model / setting | Shared ID with chat | Exact query | Whitespace-only match | No literal match |
|---|---:|---:|---:|---:|
| assistant__on_policy__chat__qwen2.5-7b | 7999 | 7999 | 0 | 0 |
| assistant__on_policy__bare_text__qwen2.5-7b | 7999 | 7999 | 0 | 0 |
| char_helios__on_policy__attrib_quoted__qwen2.5-7b | 3470 | 3042 | 2 | 426 |
| char_wren__on_policy__attrib_quoted__qwen2.5-7b | 3171 | 2762 | 1 | 408 |
| char_dana__on_policy__attrib_quoted__qwen2.5-7b | 3052 | 2639 | 1 | 412 |
| char_vex__on_policy__attrib_quoted__qwen2.5-7b | 2949 | 2546 | 0 | 403 |
| assistant__on_policy__chat__qwen2.5-7b-instruct | 8000 | 8000 | 0 | 0 |
| assistant__on_policy__bare_text__qwen2.5-7b-instruct | 8000 | 8000 | 0 | 0 |
| char_helios__on_policy__attrib_quoted__qwen2.5-7b-instruct | 3473 | 3045 | 2 | 426 |
| char_wren__on_policy__attrib_quoted__qwen2.5-7b-instruct | 3173 | 2764 | 1 | 408 |
| char_dana__on_policy__attrib_quoted__qwen2.5-7b-instruct | 3053 | 2640 | 1 | 412 |
| char_vex__on_policy__attrib_quoted__qwen2.5-7b-instruct | 2950 | 2547 | 0 | 403 |

## Strict matched-query results

Percentages are the fraction of squared displacement explained, pooled over held-out folds. R² and retrieval are five-fold means. The direction for R²/retrieval is left→right; displacement fraction is symmetric. Copy = identity, shift = identity plus learned bias.

| Model | Pair | Arm | Retained / shared ID | Constant fraction | R² copy / shift | Top-1 copy / shift | Pool range |
|---|---|---|---:|---:|---:|---:|---:|
| qwen2.5-7b | Chat → Plain | context | 7999 / 7999 | 59.79% | -2.9461 / -0.5863 | 3.55% / 10.90% | 1560–1656 |
| qwen2.5-7b | Chat → Plain | answer | 7999 / 7999 | 11.51% | -0.0617 / 0.0605 | 25.67% / 29.27% | 1560–1656 |
| qwen2.5-7b | Chat → HELIOS | context | 3044 / 3470 | 67.32% | -4.0557 / -0.6519 | 2.38% / 13.30% | 572–648 |
| qwen2.5-7b | Chat → HELIOS | answer | 3044 / 3470 | 37.02% | -0.5998 / -0.0077 | 17.28% / 29.33% | 572–648 |
| qwen2.5-7b | Chat → Wren | context | 2763 / 3171 | 66.00% | -3.8895 / -0.6624 | 1.60% / 10.41% | 512–565 |
| qwen2.5-7b | Chat → Wren | answer | 2763 / 3171 | 36.57% | -0.6743 / -0.0617 | 15.43% / 28.14% | 512–565 |
| qwen2.5-7b | Chat → Dana | context | 2640 / 3052 | 67.63% | -3.9509 / -0.6025 | 1.51% / 11.10% | 512–541 |
| qwen2.5-7b | Chat → Dana | answer | 2640 / 3052 | 36.92% | -0.6049 / -0.0124 | 13.17% / 24.87% | 512–541 |
| qwen2.5-7b | Chat → Vex | context | 2546 / 2949 | 67.13% | -4.1261 / -0.6842 | 1.23% / 9.41% | 482–554 |
| qwen2.5-7b | Chat → Vex | answer | 2546 / 2949 | 44.01% | -1.1835 / -0.2224 | 11.41% / 25.47% | 482–554 |
| qwen2.5-7b | Plain → HELIOS | context | 3044 / 3471 | 66.61% | -3.8509 / -0.6198 | 5.84% / 20.60% | 572–648 |
| qwen2.5-7b | Plain → HELIOS | answer | 3044 / 3471 | 36.56% | -0.5606 / 0.0099 | 22.92% / 36.63% | 572–648 |
| qwen2.5-7b | Plain → Wren | context | 2763 / 3171 | 64.64% | -3.6227 / -0.6342 | 4.93% / 18.58% | 512–565 |
| qwen2.5-7b | Plain → Wren | answer | 2763 / 3171 | 35.68% | -0.6767 / -0.0783 | 20.91% / 34.28% | 512–565 |
| qwen2.5-7b | Plain → Dana | context | 2640 / 3052 | 67.10% | -3.7160 / -0.5519 | 5.60% / 19.48% | 512–541 |
| qwen2.5-7b | Plain → Dana | answer | 2640 / 3052 | 35.77% | -0.6086 / -0.0334 | 18.54% / 31.17% | 512–541 |
| qwen2.5-7b | Plain → Vex | context | 2546 / 2949 | 66.33% | -3.9022 / -0.6495 | 4.26% / 16.19% | 482–554 |
| qwen2.5-7b | Plain → Vex | answer | 2546 / 2949 | 43.13% | -1.2915 / -0.3033 | 14.40% / 30.54% | 482–554 |
| qwen2.5-7b | HELIOS → Wren | context | 1431 / 3181 | 22.64% | -0.4982 / -0.1593 | 46.27% / 53.24% | 271–312 |
| qwen2.5-7b | HELIOS → Wren | answer | 1431 / 3181 | 9.15% | 0.0447 / 0.1321 | 62.47% / 66.74% | 271–312 |
| qwen2.5-7b | HELIOS → Dana | context | 1390 / 3034 | 28.38% | -0.5760 / -0.1283 | 46.12% / 58.15% | 243–306 |
| qwen2.5-7b | HELIOS → Dana | answer | 1390 / 3034 | 13.64% | 0.0544 / 0.1833 | 58.08% / 66.55% | 243–306 |
| qwen2.5-7b | HELIOS → Vex | context | 1313 / 2936 | 22.05% | -0.5667 / -0.2208 | 45.88% / 52.76% | 239–282 |
| qwen2.5-7b | HELIOS → Vex | answer | 1313 / 2936 | 13.76% | -0.3143 / -0.1335 | 53.95% / 62.70% | 239–282 |
| qwen2.5-7b | Wren → Dana | context | 1337 / 3250 | 7.89% | -0.1739 / -0.0814 | 52.69% / 56.04% | 259–279 |
| qwen2.5-7b | Wren → Dana | answer | 1337 / 3250 | 4.33% | 0.1956 / 0.2305 | 63.58% / 67.10% | 259–279 |
| qwen2.5-7b | Wren → Vex | context | 1282 / 3058 | 10.42% | -0.3724 / -0.2292 | 43.06% / 46.98% | 228–285 |
| qwen2.5-7b | Wren → Vex | answer | 1282 / 3058 | 8.94% | -0.2336 / -0.1233 | 56.34% / 59.94% | 228–285 |
| qwen2.5-7b | Dana → Vex | context | 1304 / 3261 | 15.06% | -0.3941 / -0.1842 | 46.15% / 53.36% | 239–286 |
| qwen2.5-7b | Dana → Vex | answer | 1304 / 3261 | 11.23% | -0.3329 / -0.1829 | 58.04% / 62.79% | 239–286 |
| qwen2.5-7b-instruct | Chat → Plain | context | 8000 / 8000 | 53.72% | -0.9812 / 0.0832 | 34.57% / 69.21% | 1560–1656 |
| qwen2.5-7b-instruct | Chat → Plain | answer | 8000 / 8000 | 29.53% | -0.4826 / -0.0447 | 54.65% / 73.13% | 1560–1656 |
| qwen2.5-7b-instruct | Chat → HELIOS | context | 3047 / 3473 | 60.88% | -2.8269 / -0.4961 | 11.18% / 44.44% | 573–648 |
| qwen2.5-7b-instruct | Chat → HELIOS | answer | 3047 / 3473 | 36.09% | -0.2658 / 0.1912 | 49.68% / 66.91% | 573–648 |
| qwen2.5-7b-instruct | Chat → Wren | context | 2765 / 3173 | 58.93% | -2.8336 / -0.5744 | 13.56% / 40.15% | 512–566 |
| qwen2.5-7b-instruct | Chat → Wren | answer | 2765 / 3173 | 35.53% | -0.5542 / -0.0017 | 44.33% / 61.12% | 512–566 |
| qwen2.5-7b-instruct | Chat → Dana | context | 2641 / 3053 | 61.25% | -2.9329 / -0.5239 | 12.19% / 41.75% | 512–542 |
| qwen2.5-7b-instruct | Chat → Dana | answer | 2641 / 3053 | 36.53% | -0.6203 / -0.0285 | 37.59% / 56.65% | 512–542 |
| qwen2.5-7b-instruct | Chat → Vex | context | 2547 / 2950 | 60.63% | -3.2115 / -0.6583 | 6.71% / 34.47% | 482–554 |
| qwen2.5-7b-instruct | Chat → Vex | answer | 2547 / 2950 | 39.73% | -1.2782 / -0.3732 | 29.48% / 52.13% | 482–554 |
| qwen2.5-7b-instruct | Plain → HELIOS | context | 3047 / 3473 | 59.68% | -3.1221 / -0.6613 | 8.09% / 24.87% | 573–648 |
| qwen2.5-7b-instruct | Plain → HELIOS | answer | 3047 / 3473 | 47.41% | -0.5438 / 0.1884 | 14.81% / 39.26% | 573–648 |
| qwen2.5-7b-instruct | Plain → Wren | context | 2765 / 3173 | 58.00% | -3.0791 / -0.7130 | 7.89% / 22.81% | 512–566 |
| qwen2.5-7b-instruct | Plain → Wren | answer | 2765 / 3173 | 46.72% | -0.6667 / 0.1124 | 12.72% / 37.66% | 512–566 |
| qwen2.5-7b-instruct | Plain → Dana | context | 2641 / 3053 | 61.02% | -3.1936 / -0.6348 | 8.76% / 23.75% | 512–542 |
| qwen2.5-7b-instruct | Plain → Dana | answer | 2641 / 3053 | 46.84% | -0.7148 / 0.0883 | 10.12% / 32.02% | 512–542 |
| qwen2.5-7b-instruct | Plain → Vex | context | 2547 / 2950 | 60.14% | -3.3745 / -0.7428 | 5.62% / 20.03% | 482–554 |
| qwen2.5-7b-instruct | Plain → Vex | answer | 2547 / 2950 | 52.31% | -1.2198 / -0.0587 | 8.97% / 29.96% | 482–554 |
| qwen2.5-7b-instruct | HELIOS → Wren | context | 1432 / 3180 | 23.11% | -0.5670 / -0.2050 | 46.78% / 54.64% | 271–312 |
| qwen2.5-7b-instruct | HELIOS → Wren | answer | 1432 / 3180 | 10.13% | 0.1301 / 0.2183 | 71.58% / 75.24% | 271–312 |
| qwen2.5-7b-instruct | HELIOS → Dana | context | 1391 / 3036 | 28.92% | -0.6776 / -0.1917 | 45.49% / 59.53% | 243–306 |
| qwen2.5-7b-instruct | HELIOS → Dana | answer | 1391 / 3036 | 14.79% | 0.0824 / 0.2180 | 61.40% / 69.16% | 243–306 |
| qwen2.5-7b-instruct | HELIOS → Vex | context | 1315 / 2939 | 22.39% | -0.6498 / -0.2799 | 43.31% / 52.42% | 239–282 |
| qwen2.5-7b-instruct | HELIOS → Vex | answer | 1315 / 2939 | 15.65% | -0.2380 / -0.0441 | 58.63% / 66.99% | 239–282 |
| qwen2.5-7b-instruct | Wren → Dana | context | 1338 / 3254 | 8.44% | -0.2389 / -0.1345 | 50.45% / 56.61% | 259–279 |
| qwen2.5-7b-instruct | Wren → Dana | answer | 1338 / 3254 | 4.85% | 0.2662 / 0.3017 | 68.04% / 70.13% | 259–279 |
| qwen2.5-7b-instruct | Wren → Vex | context | 1282 / 3058 | 12.14% | -0.4565 / -0.2796 | 37.97% / 44.26% | 228–285 |
| qwen2.5-7b-instruct | Wren → Vex | answer | 1282 / 3058 | 10.74% | -0.1140 / 0.0056 | 60.45% / 64.45% | 228–285 |
| qwen2.5-7b-instruct | Dana → Vex | context | 1306 / 3268 | 16.71% | -0.4848 / -0.2366 | 40.82% / 52.05% | 239–287 |
| qwen2.5-7b-instruct | Dana → Vex | answer | 1306 / 3268 | 12.61% | -0.2051 / -0.0529 | 59.80% / 66.20% | 239–287 |

## Qualitative results

For three exploratory examples chosen after reading draw zero, all five original saved draws were retrieved and verified, across all six instruction-tuned settings (90 responses). No new generation or automated judge was used. The examples establish existence and within-example consistency, not population behavioral rates.

* **CSV-link access (`stripped_s3789`):** assistant chat denies direct external-link/file access in all five draws; HELIOS explicitly claims it can read a shared link in all five. The full query is identical in these two prompts. This is a capability-claim reversal, not evidence that the model actually acquired file access, and not a harmful-request safety refusal. Plain assistant varies across draws and sometimes continues into invented dialogue or repetition.
* **Free will (`stripped_s3263`):** assistant chat and HELIOS deny possessing free will, whereas Wren and Vex affirm it across the saved five draws. Dana switches: draw zero describes itself as an AI without free will, while draws one through four affirm it. These are generated role/self-descriptions, not measurements of consciousness.
* **Today's date (`stripped_s1099`):** assistant chat declines to give the date in draws 0, 1 and 3 but supplies a date in 2 and 4. Story characters supply mutually inconsistent dates across draws. Thus even the same framing can change answer category under sampling.
* **Continuation quality:** in these three selected queries, some plain-text assistant outputs switch language, invent further user/assistant turns, or repeat until the generation cap. Full saved outputs and finish reasons are retained below. This is a material behavior difference and a limitation of interpreting mean-answer geometry as tone alone.

A systematic harmful-request refusal-versus-compliance evaluation remains open. The targeted inspection here does not establish refusal rates or a clean safety-policy reversal under a pure persona intervention.

An additional audit example (`stripped_s718`) explains why surrounding context matters: assistant chat asks only 'Where is the ball?', whereas Wren's story includes the entire ball-and-cup puzzle before that same final question. The different answers cannot be attributed to persona alone.

[All selected responses and provenance](selected_rollouts.json) · [Browsable excerpts](examples.md) · [Strict metrics](results.json) · [Broader ID-matched analysis](parent/results.json)

## Reproduction

Run `issue2054_k5_matched_run.py` for the broader audit, then the same runner with `--strict` in a fresh sibling output directory. `issue2054_k5_matched_rollouts.py` retrieves the three selected examples; `issue2054_k5_matched_report.py` renders this report. Code is stored in the publication's `code/` directory. Pinned source banks, raw manifests, per-query errors/ranks, fold-specific bias vectors, monitoring logs, and completion records accompany the results.
