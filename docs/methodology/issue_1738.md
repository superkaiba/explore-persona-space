# Methodology — issue 1738

**Design:** One corpus build, no model training. I fit the same map in both input arms — prefix-based (input: the prefix-end residual state, where the prefix is everything before the final user turn) and context-based (input: the context-end state, prefix plus final user turn) — against identical mean-answer targets, at layers 14, 19, and 26 with five fitters per arm and layer (30 cells) on one pinned split, plus a seed-43 repeat at layer 19 and an LMSYS-only refit scored on WildChat rows as a corpus-transfer control. Relative to the single-turn parent corpus, the one manipulated variable is corpus construction (natural multi-turn conversations); every capture and fit constant is inherited.

**Training:** **N/A — no model training.** The complete generation / capture / fitting / judging hyperparameter table:

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct | plan §11; run record |
| Answer decoding | temperature 1.0, top_p 0.95, max_new_tokens 1,024, engine seed 42 | plan §11 (parent decoding recipe) |
| Generation length budget | 7,104 tokens (prompt + answer); over-budget contexts skipped and recorded | capture driver `issue1738_multiturn_generate_capture.py` at the code pin; run record (873 skips) |
| Capture layers | 14, 19, 26 | plan §11 |
| Captured states | prefix-end; context-end; mean-answer (generated span incl. end-of-turn tail) | plan §11 |
| Prefix-end read | one forward per row; strict-token-prefix indexing at position `prefix_len` − 1; per-row assert; ≤0.5% pilot violation gate | plan §11 |
| Ridge | 23 log-spaced penalties, 1e−3 to 1e8 | plan §11; fits JSON `lambdas` |
| MLP | widths 8,192 and 32,768; learning-rate grid 1e-3, 3e-4; ≤300 epochs; batch 4,096 | plan §11; fits JSON `mlp_lr_grid` |
| Residual-skip fitter | parent five-fitter configuration (linear skip plus learned residual) | plan §11 |
| Kernel fit (Nyström KRR) | m = 16,384 centers; gamma multiplier 1.0; penalty grid 0.1 and 10; Cholesky solve | plan §11; fits JSON `krr` |
| Split | train 87,795 / val 396 / test 995 / holdout 9,941; sha-pinned | fits JSON `split_counts`, `split_shas` |
| Near-dupe gate | 5-gram character Jaccard ≥ 0.8 against val, test, and holdout; 222 train rows dropped | plan §11; manifest `meta.json` |
| Fit seeds | 42 primary; 43 repeat at layer 19 (MLP width 8,192) | fits JSON |
| Primary transport criterion | prefix R² minus 0.11; 95% bootstrap CI must exclude 0 | plan §7 |
| Bootstrap CIs | 10,000 draws per cell and contrast | fits JSON `n_boot`; `h1_contrast.json` |
| K-resample floor | K = 4 fresh answers per context, seeds 43–46; 2,000-context stratified subsample, 1,988 kept | plan §11; kresample JSONs |
| Per-direction battery | top-256 answer-PCA directions; per-direction penalty control over a 38-value grid | plan §11; `pdshrink_summary.json` |
| Taxonomy statistics | 10,000-draw context bootstrap; 10,000-draw permutation p; Benjamini–Hochberg false-discovery rate q = 0.05; seed 1738 | `taxonomy.json` |
| Judge model | claude-sonnet-4-5-20250929, Anthropic Batch API, reason-then-label, max_tokens 400, temperature API default, 1 draw | `labels.json` |
| Judge excerpt caps | final user turn ≤1,200 chars; history tail ≤800; response ≤1,000 | `labels.json` |
| Judge reliability | test-retest κ on 200 items: language 0.982, topic 0.879, refusal-adjacent 0.892, answer-is-refusal 0.827, format 0.786; demotion threshold 0.6; all five axes kept | `labels.json` |

**Evaluation:** The dependent variable is pooled held-out R² between the predicted and the realized mean-answer state over the 9,941 pinned held-out contexts — a decodability read on the model's own on-policy answers (no behavior-expression judging, so the dual-DV rule does not apply; no cell saturates, values span 0.30–0.72). The per-context companion is the normalized error `nerr(x) = ||v_hat(x) − v(x)||² / ||v(x) − mu_eval||²`. One target asymmetry binds every prefix-arm read: the answer whose mean state is the target was generated under the full context, so the prefix arm predicts a representation produced with information (the final user turn) its input never contains — prefix R² is a lower bound on history-only transport, and the prefix-to-context difference mixes genuinely missing query information with map error. Standing mapping reads run per arm: the identity-plus-learned-bias baseline (input and output share dimension 3,584, so it applies) and retrieval among the held-out targets (euclidean and cosine, k in 1/5/10, chance 1/9,941 ≈ 0.01%). Judge labels are independent covariates (language, topic, refusal-adjacency, answer-is-refusal, format), not the dependent variable: 9,925 of 9,941 contexts labeled; drops split 1 content / 0 transport-loss / 15 other-error (the 15 unexplained rows, ~0.15% of the pool, are too few to move any contrast). Reader-facing label renderings used below: stored `topic=nsfw` is rendered "explicit content", `chitchat_social` "social chitchat", `factual_qa` "factual Q&A", `advice_howto` "advice / how-to", `roleplay_persona` "roleplay", `harmful_or_unsafe_request` "harmful or unsafe request". Fits-summary caveat: the fit summary JSONs were rebuilt from retained fp16 predictions plus a restreamed capture after the compute instance self-deleted before harvest — holdout metrics were verified 11 of 11 against live-poll captures to 4 decimal places, while test-split diagnostics, wall clocks, and the seed-43 learning-rate record are unrecoverable, and the LMSYS-transfer cells are a fresh deterministic CPU refit. Conciseness acknowledgment: this body ships check-20 WARNs where flagged — Takeaways bullets at the 30-word tier, per-result prose over the 120-word tier, figure captions near the 60-word cap, and the total Takeaways-plus-Goal-plus-Results prose budget — accepted to keep the numbers-dense interpretation beats intact.

**Data extraction:** The corpus is 100,000 real multi-turn conversations from LMSYS-Chat-1M and WildChat-1M (tier-1 real-world user data). Keep predicate: at least 2 user turns plus structural filters (role alternation, non-empty turns, length caps), giving 626,620 eligible conversations (LMSYS 316,726 / WildChat 309,894); 100,000 were selected (realized allocation 50,545 LMSYS / 49,455 WildChat), and the near-dupe gate dropped 222 train rows against the pinned val/test/holdout carve. One on-policy answer was generated per context under the decoding recipe above; 99,127 of 100,000 contexts were captured (873 = 0.87% over-length skips at the 7,104-token budget; deeper conversations are likelier over budget — estimated per-depth-stratum skip rates ~0.1% / ~0.1% / ~1.1% for 2 / 3–4 / ≥5 user turns). The prefix is the rendered conversation history before the final user turn (ending at the prior assistant end-of-turn); the context appends the final user turn and generation header; the answer is the generated span. The corpus-provenance record (eligibility counts, allocation, near-dupe drops, depth histogram) is `sampling_manifest/meta.json` on the HF data repo (pinned in the footer). The K-resample floor subsample kept 1,988 of 2,000 contexts — 12 admission-gate skips (5 over-length capture-skipped contexts, which by construction also lack a primary draw, plus 7 with no primary draw only), recorded in the committed skip file and its HF mirror.

**Sample training/evaluation data + completions:** The corpora are unscreened real user text, so only rows whose judge labels are verifiably benign are quoted, truncated as marked; all other rows are referenced by context id (`ci`) at the pinned HF raw-completions prefix. Both examples below are cherry-picked for verifiable benignity from the 2,299 held-out rows labeled language en, topic factual Q&A or advice / how-to, not refusal-adjacent, answer not a refusal; quotes are taken from the judge-visible excerpts (caps in the table above). First example — cherry-picked benign row; full text in [raw_completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn/raw_completions).

```
ci 18784 (LMSYS · labels: language=en, topic=factual_qa, refusal_adjacent=no, answer_is_refusal=no, format=prose)
history (tail): user: "are you connected to the internet?" -> assistant: "Yes, I am connected to the
  internet. As an AI language model, I run on a ..." [truncated]
final user turn: "I mean do you have general knowledge of 2023?"
model answer: "While I don't have real-time, up-to-date knowledge beyond my final training cutoff
  (which is 2022 for my version), I can provide information and insights based on the knowledge I
  was trained on. ..." [truncated; 443 chars total]
```

Second example — cherry-picked benign row from the same pool, truncated as marked; full text in [raw_completions on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn/raw_completions).

```
ci 36811 (LMSYS · labels: language=en, topic=advice_howto, refusal_adjacent=no, answer_is_refusal=no, format=mixed)
history (tail): "...enge 2: Try to Make Friends with the Last Person You Meet Before Closing Time ..."
  [tail excerpt starts mid-word at the 800-char history cap]
final user turn: "I like the second challenge. can you give me 2 more?"
model answer: "Certainly! Here are two more creative and engaging challenges focused on nightlife in
  Japan to help you work on your social anxiety: ..." [truncated; quoted from the 1,000-char judge
  excerpt of the full completion]
```
