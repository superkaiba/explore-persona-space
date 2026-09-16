# Instruct answer-comparison numerical review

PASS at immutable HF revision `5b7db38023f0cf5c293c3e32f247f093e4014ac6`. All 20 means and 40 bootstrap bounds reproduce from 7,999 per-conversation records. Both content receipts and the answer-array hash match. All 65 transfer and five own-map records exactly match the previously audited completed snapshot.

Eligibility: 7,999 story complete-five rows, 8,000 chat rows, 7,999 paired rows. The source audit contains 8,000 exact shared raw question matches. Fold counts: 1587 / 1589 / 1656 / 1607 / 1560.

## Recomputed summaries

| Metric | Mean | 95% conversation-bootstrap interval |
|---|---:|---:|
| mean_answer_cosine | 0.8618 | [0.8600, 0.8637] |
| centered_mean_answer_cosine | 0.6322 | [0.6273, 0.6371] |
| mean_answer_squared_distance | 810.4205 | [798.7801, 821.4025] |
| normalized_mean_answer_squared_distance | 0.2800 | [0.2760, 0.2838] |
| cross_draw_cosine | 0.8098 | [0.8079, 0.8118] |
| story_within_draw_cosine | 0.8927 | [0.8913, 0.8941] |
| chat_within_draw_cosine | 0.9530 | [0.9521, 0.9540] |
| cross_draw_squared_distance | 1196.4663 | [1183.8814, 1208.9630] |
| story_within_squared_distance | 698.7779 | [689.8670, 708.3751] |
| chat_within_squared_distance | 266.3365 | [260.7431, 270.8247] |
| corrected_mean_response_squared_displacement | 713.9091 | [702.5893, 724.2120] |
| normalized_corrected_mean_response_squared_displacement | 0.2467 | [0.2428, 0.2502] |
| training_normalization_scale | 2893.9323 | [2893.8593, 2894.0149] |
| lexical_jaccard | 0.1428 | [0.1405, 0.1449] |
| story_answer_characters | 428.2161 | [418.4525, 439.4526] |
| chat_answer_characters | 1343.8646 | [1321.2554, 1371.3835] |
| story_lexical_tokens | 85.5589 | [83.6541, 87.7980] |
| chat_lexical_tokens | 284.4531 | [279.3185, 290.7682] |
| story_cap_fraction | 0.008951 | [0.008176, 0.009877] |
| chat_cap_fraction | 0.002075 | [0.001599, 0.002676] |

Intervals use the recorded seed, 200 conversation resamples and fixed estimated centering/scales.

## Correction and cap checks

`Dcross − (Dstory + Dchat)/2` and division by the training-fold scale match every saved corrected value exactly. The equivalent K5 sample-mean identity differs by at most 0.00011931 activation-squared units, consistent with float32 mean storage. 4 negative per-query estimates are retained.

| Framing | Fresh draw cap | Capped answers / 39,995 | Cap rate | Caps by draw 0 / 1 / 2 / 3 / 4 |
|---|---:|---:|---:|---|
| story | 2,048 | 358 | 0.8951% | 78 / 69 / 71 / 64 / 76 |
| chat | 2,048 | 83 | 0.2075% | 20 / 13 / 17 / 15 / 18 |

**Table note:** both fresh Instruct cells have a 2,048-token budget, but use different stop strings. Draw-zero per-draw token budgets and stop reasons are absent from the audited source case. Cap fractions count recorded length terminations; they do not establish equal historical stopping behavior or semantic equivalence. The IID correction remains a qualified diagnostic.

## Interpretation and limits

- Numerical PASS: all 20 means and all 40 interval endpoints reproduce from the saved per-conversation arrays. Both raw correction and normalized counterpart agree exactly. All 65 transfer records and five own-map records are unchanged from the independently audited completed transfer snapshot.
- Eligibility is consistent: 7,999 story complete-five rows and 8,000 chat rows yield 7,999 shared matched conversations. The source query audit has 8,000 exact raw question matches, with no mismatch or whitespace-only matches; one story row fails complete-five eligibility. Original prefixes were not all independently reopened for this numeric audit.
- Cross-draw cosine is 0.810, below within-story 0.893 and within-chat 0.953. Raw K5-mean cosine is 0.862 and framing-centered cosine is 0.632. These are similarities of activation summaries, not semantic equivalence or matched refusal/content rates.
- The normalized corrected squared displacement is 0.246693 [0.242781, 0.250249], showing a remaining representation difference under the stated diagnostic. It is not R², a probability, or a percentage of semantically different answers.
- Correction assumes independent identically distributed draws within each framing. Draw-zero historical runtime and per-draw stopping/budget metadata are incomplete; its cap frequency alone cannot identify a cap change. Keep the IID interpretation qualified.
- Fresh Instruct story and chat draws both use 2,048-token budgets, unlike the Base story/chat budget difference. Their configured stops remain different: closing quotation mark for story, <|im_end|> for chat. The targeted case records null specific stop reasons for chat and lacks draw-zero per-draw budgets/stop reasons.
- Cap fractions correctly average each framing’s saved length-termination flags: story 358/39,995 = 0.8951%, chat 83/39,995 = 0.2075%. Source code validates every fresh per-cell budget and cap flag before computing these metrics. Caps and long continuation boundaries still affect full-answer activations.
- Chat answers average 1,344 characters and 284 lexical tokens, versus 428 characters and 86 lexical tokens for story. Lexical Jaccard 0.143 is a lexical diagnostic; neither it nor cosine establishes semantic equivalence.
- Bootstrap intervals use 200 conversation resamples and hold the saved five-response summaries and fitted training-fold centers/scales fixed. They do not quantify all generation, capture or normalization-estimation uncertainty.
- Question equality does not remove narrative-added task information. Context effects remain embedded in answer states; this comparison does not isolate persona naming.
- The cap-free subset is a post hoc descriptive check selected on generated outcomes, not a replacement primary estimate or a randomized control.

Cap-free descriptive subset: 7,624 conversations; centered cosine 0.6330; normalized corrected displacement 0.2523. This subset is outcome-selected.

Inputs: `/tmp/issue2054-assistant-story-review/analysis/qwen2.5-7b-instruct.json`; `/tmp/issue2054-assistant-story-review/analysis/answers/qwen2.5-7b-instruct.npz`. Exact values and verification details: `instruct_answer_numeric_review.json`.
