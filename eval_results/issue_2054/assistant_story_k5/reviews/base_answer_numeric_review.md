# Base answer-comparison numerical review

PASS for numerical consistency at HF revision `68c2b84cecc5692cdc4b8574b433600ad57edd6b`. This review covers the Base answer comparison, not completion of the overall two-checkpoint job.

All 20 means and 40 bootstrap endpoints reproduce from the 7,993-row saved array. Story has 7,994 complete-five rows; chat has 7,999; 7,993 are shared and query-matched. All 7,999 shared raw story/chat IDs have exact question matches in the source audit. Fold counts: 1,588 / 1,585 / 1,656 / 1,605 / 1,559.

## Recomputed summaries

| Metric | Mean | 95% conversation-bootstrap interval |
|---|---:|---:|
| mean_answer_cosine | 0.8425 | [0.8411, 0.8437] |
| centered_mean_answer_cosine | 0.4088 | [0.4037, 0.4130] |
| mean_answer_squared_distance | 913.0768 | [903.3989, 928.3517] |
| normalized_mean_answer_squared_distance | 0.3326 | [0.3291, 0.3382] |
| cross_draw_cosine | 0.7044 | [0.7026, 0.7059] |
| story_within_draw_cosine | 0.8186 | [0.8166, 0.8201] |
| chat_within_draw_cosine | 0.7663 | [0.7645, 0.7682] |
| cross_draw_squared_distance | 2050.0791 | [2034.1964, 2071.1623] |
| story_within_squared_distance | 1346.7832 | [1334.6356, 1361.8714] |
| chat_within_squared_distance | 1495.7225 | [1480.0508, 1512.6195] |
| corrected_mean_response_squared_displacement | 628.8262 | [619.8633, 643.9001] |
| normalized_corrected_mean_response_squared_displacement | 0.2291 | [0.2258, 0.2346] |
| training_normalization_scale | 2745.1834 | [2745.1125, 2745.2462] |
| lexical_jaccard | 0.0567 | [0.0560, 0.0575] |
| story_answer_characters | 308.9945 | [302.1108, 314.3989] |
| chat_answer_characters | 3570.7571 | [3527.0765, 3613.6323] |
| story_lexical_tokens | 60.2516 | [58.9048, 61.4024] |
| chat_lexical_tokens | 747.6894 | [737.6699, 756.2981] |
| story_cap_fraction | 0.000275 | [0.000125, 0.000500] |
| chat_cap_fraction | 0.112473 | [0.109367, 0.115630] |

Intervals use the recorded seed and 200 resamples of conversations. They hold the captures and estimated centering/scales fixed.

## Correction and cap audit

The raw correction equals `Dcross - (Dstory + Dchat)/2` exactly in every row: all 25 cross-draw squared Euclidean distances versus each set of ten unordered within-draw distances. Division by the same positive training-fold scale reproduces the normalized counterpart exactly. The equivalent sample-mean form agrees to at most 0.000165 activation-squared units, consistent with float32 stored K5 means. Nineteen negative estimates are retained.

| Framing | Fresh draw budget | Capped answers / 39,965 | Cap rate | Caps by draw 0 / 1 / 2 / 3 / 4 |
|---|---:|---:|---:|---|
| story | 2,048 | 11 | 0.0275% | 1 / 2 / 4 / 1 / 3 |
| chat | 4,096 | 4,495 | 11.2473% | 447 / 993 / 1010 / 999 / 1046 |

**Table note:** cap rates count each framing’s recorded `finish_reason=length`, under different fresh-draw budgets and stop strings. Fresh story stops at a closing quotation mark; fresh chat is configured to stop at `<|im_end|>`. The targeted raw audit records `stop_reason=null` for its fresh chat answers and does not identify a more specific stop event. Original draw zero lacks per-draw budget and stop-reason fields. Its lower observed cap rate does not establish a different numeric cap. IID correction remains qualified.

## Interpretation and limits

- Numerical PASS: all 20 reported means and all 40 interval endpoints reproduce from the saved per-conversation arrays; both noise-correction identities and normalization agree. No new fits or generations were run.
- Complete-five pairing retains 7,993 conversations: story has 7,994 complete rows and chat 7,999. Source query-audit counts show all 7,999 shared raw story/chat IDs have exact question matches; the stored arrays cannot independently re-audit the original prefix text.
- Raw K5 mean cosine is 0.842 but becomes 0.409 after each framing is centered using paired training folds. Cross-framing draw cosine (0.704) is below within-story (0.819) and within-chat (0.766). Shared activation orientation is not semantic equivalence.
- The corrected squared displacement is a qualified diagnostic under independent identically distributed draws within each framing. The stored arithmetic is valid, but draw-zero per-draw caps/stops/runtime are not fully documented, so the statistic should not be called an established noise-free framing effect.
- Fresh draws 1–4 use 2,048 tokens for story and 4,096 for Base chat, with different configured stop strings. Draw-zero per-draw budgets and stop reasons are absent in the audited source case. Its lower chat cap frequency alone does not prove a different numeric budget.
- Cap fraction is implemented correctly as the mean of each cell’s saved finish_reason==length flags. It is not a comparison at a common token threshold. Every fresh row is checked against k3.cap_for(cell) and every fresh cap mask against its saved finish_reason before analysis.
- Chat outputs average 3,571 characters and 748 lexical tokens, versus 309 characters and 60 lexical tokens for story. The 11.247% versus 0.0275% cap rates, boundaries and long continuations materially limit full-answer activation and length comparisons.
- Bootstrap intervals resample conversations (not the correlated 25 draw pairs), with 200 draws. They condition on stored captures, estimated training-fold centering/scales and the observed five responses; they do not refit those constants or include all generation/runtime uncertainty.
- The correction retains 19 negative per-conversation estimates; clipping would upward-bias its mean. Corrected displacement is not centered R² or a fraction of semantic disagreement.
- The cap-free descriptive subset has 4,648 conversations and still shows a positive corrected representation difference, but this post hoc subset selects on generated outcomes and is not a replacement primary estimate.

Sources: `/tmp/issue2054-assistant-story-review/analysis/qwen2.5-7b.json`; `/tmp/issue2054-assistant-story-review/analysis/answers/qwen2.5-7b.npz`. Both local content hashes and receipt sizes verified. Detailed numbers and checks: `base_answer_numeric_review.json`.
