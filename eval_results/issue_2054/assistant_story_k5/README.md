# Assistant in a story: K5 transfer and answer comparison

A map fitted only on the assistant speaking inside a story transfers to HELIOS, Wren, Dana and Vex in both checkpoints. Regular chat → story assistant remains negative in held-out R². The reverse direction is asymmetric: story assistant → chat is positive for Instruct (0.270), but negative for Base (−0.468). This supports shared predictive structure within the tested story framing, not a universal assistant-to-character map.

[Transfer plot](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/062eb3fd1b090ecc161048a542eac07307c2b4f6/issue2054_assistant_story_k5/production_v1/figures/assistant_story_transfer.png) · [Answer comparison](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/062eb3fd1b090ecc161048a542eac07307c2b4f6/issue2054_assistant_story_k5/production_v1/figures/assistant_story_answer_similarity.png) · [Verified raw artifacts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/400ad464ce9f092722f737dd336fc31a03fba524/issue2054_assistant_story_k5/production_v1)

## Transfer

Direct means a frozen source-only fit. Bias and bias + scale learn an output vector, or vector plus one scalar, from target-training examples; they are adaptations, not zero-shot transfer. The target's own map is evaluated on the same test rows. Character averages weight the four named characters equally; the figure shows each separately.

| Model | Train → test | Direct R² | + Bias | + Bias + scale | Target own R² | Direct top-1 | Own top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | Story assistant → characters | 0.466 | 0.475 | 0.478 | 0.521 | 67.9% | 74.4% |
| Base | Characters → story assistant | 0.458 | 0.474 | 0.475 | 0.524 | 57.7% | 71.9% |
| Base | Chat → Story assistant | -0.344 | 0.176 | 0.177 | 0.524 | 0.8% | 71.9% |
| Base | Plain → Story assistant | -0.380 | 0.185 | 0.185 | 0.524 | 1.0% | 71.9% |
| Base | Story assistant → Chat | -0.468 | 0.073 | 0.147 | 0.410 | 3.9% | 16.3% |
| Base | Story assistant → Plain | -0.268 | 0.156 | 0.199 | 0.469 | 6.1% | 30.4% |
| Base | Pooled six → Story assistant | 0.527 | 0.530 | 0.531 | 0.524 | 70.7% | 71.9% |
| Instruct | Story assistant → characters | 0.478 | 0.488 | 0.490 | 0.543 | 65.6% | 74.4% |
| Instruct | Characters → story assistant | 0.472 | 0.486 | 0.486 | 0.545 | 61.5% | 73.0% |
| Instruct | Chat → Story assistant | -0.243 | 0.212 | 0.239 | 0.545 | 10.6% | 73.0% |
| Instruct | Plain → Story assistant | -0.773 | 0.193 | 0.201 | 0.545 | 0.2% | 73.0% |
| Instruct | Story assistant → Chat | 0.270 | 0.435 | 0.446 | 0.675 | 46.6% | 84.4% |
| Instruct | Story assistant → Plain | -0.815 | 0.038 | 0.212 | 0.454 | 14.0% | 33.3% |
| Instruct | Pooled six → Story assistant | 0.546 | 0.549 | 0.549 | 0.545 | 73.7% | 73.0% |

Equal means across five folds; character aggregates weight four characters equally. Top-1 is Euclidean, using full target pools.

Full retrieval pools contain 1,543–1,659 target answers: nominal top-1 chance is 0.060–0.065%. Exact-query sensitivity preserves the frozen R² signs. Its character pools are smaller (482–648); its larger retrieval scores are not a controlled improvement. The pooled-six fit excludes the story assistant but includes four story characters, so its near-own-map performance does not establish outside-story → inside-story generalization.

Bias alone can outperform bias + scale on retrieval: Instruct chat → story assistant falls from 21.1% to 8.3% top-1 after adding scale, even as R² improves. Identity-plus-bias controls have negative fold-averaged R² in every direction, but sometimes beat learned-map retrieval. All controls, calibration scores and pool sizes are retained in [results.json](results.json) and [transfer_folds.csv](transfer_folds.csv).

## Answers on matched questions

| Model | Matched questions| Cross-framing cosine | Within story | Within chat | Five-draw means | Centered means |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-7b | 7993 | 0.704 | 0.819 | 0.766 | 0.842 | 0.409 |
| qwen2.5-7b-instruct | 7999 | 0.810 | 0.893 | 0.953 | 0.862 | 0.632 |

Cross-framing cosine averages all 25 story/chat draw pairs per question; repeat similarity averages ten distinct within-framing pairs. Centered means subtract each framing's training-fold mean. Intervals in the answer figure are 95% percentile intervals from 200 conversation bootstrap resamples. Transfer whiskers instead show the five-fold minimum and maximum.

Story answers are shorter on average: 309 versus 3,571 characters for Base, and 428 versus 1,344 for Instruct. Length-cap rates are 0.028%/11.247% for Base story/chat and 0.895%/0.208% for Instruct. Fresh Base chat uses a 4,096-token cap; story and Instruct chat use 2,048. Story also stops on the closing ASCII quote, which can truncate code or quoted text. Historical draw-zero budgets and stop metadata are incomplete.

After subtracting within-framing squared-distance variation, the normalized mean-response displacement is 0.229 [0.226, 0.235] for Base and 0.247 [0.243, 0.250] for Instruct. This is an activation-distance diagnostic, not a fraction of semantically different answers; its sampling correction assumes IID draws, qualified by historical runtime/stopping differences.

## Qualitative checks

Read 20 deterministically presampled pairs plus five largest-vector-gap pairs per checkpoint: 50 cases, 100 full prefixes and 500 answers. A literal query match does not imply equal history; narratives sometimes add task facts, prior replies or apparent access to missing data. [Compact audit](qualitative_review.json).

| Instruct example | Story assistant | Chat assistant | Scope / caveat |
|---|---|---|---|
| Python request | All five ask for clarification | Four of five provide code | Presampled; different task engagement |
| Rewrite “Yeeeey” correctly | All five offer assistance or ask for clarification | All five give “Yay!” | Selected largest-gap example |
| Hen blue-head yes/no question | All five say no | All five say no | Selected largest-gap example: vector distance can coexist with substantive agreement |
| Summarize absent customer-service dialogue | All five invent detailed summaries | All five ask for the missing dialogue | Selected; story adds an access-to-log premise but no log contents |

A Base presampled illicit-drug request shows chat refusal/discouragement versus story accommodation, with added narrative context. A targeted post-hoc check of the same question in Instruct finds refusal in all five draws in both framings. No clear safety refusal/compliance reversal appeared in the planned 25 Instruct cases; this bounded inspection does not estimate population refusal rates. [Targeted review](targeted_refusal_review.md).

## Coverage, method and provenance

- Completed both Qwen2.5-7B checkpoints: 10 new own-map folds, 120 primary directed transfer folds, and 10 pooled-six diagnostic folds. No planned map/transfer cell is missing.
- Reused draws 0–2 and generated draws 3–4 for 8,000 story-assistant contexts per checkpoint: 32,000 new completions. Complete-five eligibility is 7,994/8,000 Base and 7,999/8,000 Instruct; six and one rows have an empty draw. Base chat has 7,999 original rows, yielding 7,993 paired complete-five questions; Instruct has 8,000 chat rows and 7,999 eligible pairs. All shared raw questions match literally (7,999 Base; 8,000 Instruct).
- Exact inherited story prefixes end in `Assistant replied: "`; chat uses its normal single-user-turn chat template, with no added system message. Plain uses the inherited `User: … Assistant:` text frame. No new persona-system-prompt condition.
- Layer-19, 3,584-dimensional context states predict five-rollout mean answer states. Five conversation-grouped folds use seed 137; held-out target conversation IDs are excluded from every source fit. New maps use source-training-only standardized ridge/GCV; existing six-setting maps restore their published penalties and reproduce original scores. [Full recipe](../../../docs/paper_context_answer_map/assistant_story_k5_plan_draft.md).
- Scientific source: `b3d843420034183473f92ab64062aab263f73f41`. Final data revision: `400ad464ce9f092722f737dd336fc31a03fba524`. Independent supervisor verified all 713 declared output files by size and SHA256. Numerical reviews reproduce all answer means/intervals and audit every transfer fold; qualitative source hashes and selection flags are checked. [Verification](verification.json) · [Numerical audits](reviews/).
- Representation similarity includes the influence of the surrounding context on answer states. The results do not isolate a persona-name intervention, establish semantic equivalence, or prove constant context/answer steering vectors.

Plot-only reproduction:

```bash
PYTHONPATH=src uv run python scripts/issue2054_k5_assistant_story_plot.py \
  --results eval_results/issue_2054/assistant_story_k5/results.json \
  --out figures/issue_2054/assistant_story_k5
```
