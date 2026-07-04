---
title: The context→answer-profile map exists in pretrained Qwen2.5-7B at ~87% of instruct
  strength and is nonlinearly recoverable across two-turn cells — including a weak
  user-turn map, replicated on real human conversations, that the linear probe misses
  entirely (MODERATE confidence)
kind: experiment
tags:
- followup-auto
- followup-manual
created_at: '2026-07-02T00:14:16Z'
has_clean_result: true
parent_id: 779
origin_prompt: 'Help me to plan this experiment: Is the context vector to answer profile
  mapping present in the base model (Qwen/Qwen2.5-7B) and does it hold for the user?
  (verbatim full prompt in ## Provenance; follow-ups: frame on #779''s per-context
  map, Haiku-4.5-as-user generated conversations, file as proposed only)'
goal: 'Test whether the per-example linear context-to-answer-profile map h: c_x ->
  v(x) (#779 recipe: held-out K-fold ridge over thousands of per-example pairs) exists
  in the pretrained Qwen/Qwen2.5-7B vs Qwen2.5-7B-Instruct, for both the assistant
  and the user turn, under chat-template vs naturalistic formatting; secondary: per-position
  decay and cross-role prediction beyond a topic-persistence baseline.'
relates_to:
- identity-contextual-vs-base
- identity-cb-duality
---
# The context→answer-profile map exists in pretrained Qwen2.5-7B at ~87% of instruct strength and is nonlinearly recoverable across two-turn cells — including a weak user-turn map, replicated on real human conversations, that the linear probe misses entirely (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_825.md](https://github.com/superkaiba/explore-persona-space/blob/8f9f9a6707bf771cc73a0109a1c93a69ab4f7cb0/docs/methodology/issue_825.md) · [gist](https://gist.github.com/superkaiba/cd6dd04dbacb42587b5458b918710cdb)

## Takeaways

- **Pretrained Qwen2.5-7B carries the linear context→answer-profile map at 87.3% of instruct strength** (held-out R² 0.588 vs 0.673 at layer 19; shuffle nulls ≈ −0.02); representation-swap preserves the map's predictive information across base↔instruct (0.587/0.662 ≈ within-model) while applying one model's fitted map to the other fails (−0.066/−0.170) — consistent with a large coordinate rotation of the read-out, no mechanism claim.
- Two-turn assistant targets are nonlinear, not absent: an MLP (multilayer perceptron) probe recovers R² 0.49–0.56 across chat AND naturalistic cells where ridge reads +0.076 to −0.461.
- **The user-turn linear null now holds under all three tested user-text provenances** — Haiku-written, self-written, and real human, each on its own conversation distribution (ridge R² −0.92 to −1.12 at both headline layers on 2000 real lmsys conversations, every 95% CI below zero) — while the weak nonlinear map replicates (MLP 0.17–0.19, within 0.06 of the Haiku cells); the synthetic-u2 scope caveat is dropped.
- Real logged conversations depress the assistant map too — uniformly on the MLP (0.32–0.38 vs the parent's 0.49–0.56, a 0.16–0.18 drop in every cell), while the ridge drop (−0.27 to −0.61 vs the parent's +0.08 to −0.46) is absent in the one parent cell already deeply negative (pretrained/chat, −0.46 → −0.47) — with the assistant−user gap intact (MLP +0.14 to +0.20); the parent-rig anchor reproduced to ΔR² 0.0005, so this is a data property, not rig drift — descriptively a provenance-bundle effect (conversation sample, first-answer authorship, and user-turn authorship change together).
- Self-written user turns are more predictable than Haiku-written ones in three of four cells (MLP 0.27–0.33 vs 0.19–0.23); the exception is pretrained/chat, where self-generation degenerates (307 vs 79 tokens, 9.0% repetition) and the map weakens instead.
- Pretrained assistant→user transfer sign-flips with depth (ΔR² −0.62 at layer 14 → +0.23 at layer 26); provisional — single seed, n = 2000. Gates across all four rounds PASSed (parent replication rank correlation 0.963; every anchor within ±0.004 or better).

## Goal

**This experiment in context:** The parent experiment ([#779](https://eps.superkaiba.com/tasks/779)) established a strong linear map from a context's mean residual-stream activation to the model's answer profile on single-turn Qwen2.5-7B-Instruct, peaking near layer 19. This experiment holds that rig fixed and asks two questions: does the same linear map already exist in the *pretrained* base model (before any post-training), and does it hold when the prediction target is the *user's* next turn rather than the assistant's — extended with a two-turn conversational track, cross-role transfer cells, a user-turn-provenance axis (Haiku-written vs self-written vs real logged human user turns), and a direct base↔instruct map comparison.

**Broader narrative:** Whether the context→answer-profile map is inherited from pretraining or created by post-training locates where context/persona conditioning enters the training pipeline; whether it is assistant-generation-specific or a general next-turn property bounds how far base-model context geometry can be used to predict downstream fine-tuning behavior (the project's leakage-prediction line).

## Methodology

**Rounds:** Round 1 = the parent run (below). Round 2 (`mlp-unprobed-cells`, cheap-band auto-run): the parent's exact MLP probe extended to the 6 unprobed cells via `scripts/issue825_mlp_followup_dispatch.sh` @07589e890b — turnstore staged from HF @deb7a452 (4 m-track prefixes), 8-cell ridge refit as carrier + two-anchor ±0.05 regression gate (PASS, deltas +0.0000), 6-cell MLP, no recipe changes; outputs at `eval_results/issue_825/mlp-unprobed-cells/`. Three launches (~2.5 GPU-h vs 2 budgeted): two stage hangs (xet finalization, fixed by HF_XET_DISABLE=1), then a fully-successful fit whose gate crashed on a wrong dict path — outputs rescued from the crash-persist upload, corrected gate re-run locally. Round 3 (`onpolicy-user-turn`, user-requested): one changed variable — second-user-turn provenance, Haiku-written → written by the measured model itself per (model, format) cell; parent fit recipe unchanged; 4 assistant-map anchors refit as a rig gate; matched-rows parent refits for the two cells whose keep rate fell below 95%; via `scripts/issue825_onpolicy_dispatch.sh` @debc28c31d, outputs at `eval_results/issue_825/onpolicy-user-turn/`. Four GCP launches (~2.6 GPU-h vs 3 budgeted): three crashed in extraction on a zero-width-span tokenizer edge case in short self-written turns (fixed by generation-time span validation), the fourth ran clean. Round 4 (`real-user-turn-null`, cheap-band auto-run round 2): second-user-turn provenance changed to real logged human turns — 2,000 fully-logged 2-turn lmsys-chat-1m conversations (u1, a1, u2 all logged; a1 written by the logged serving models) ingested at the parent's pinned revision, no generation anywhere; 8 within-cell fits (4 user + 4 same-conversation assistant references) + 1 end-to-end parent-anchor re-extraction, parent fit recipe unchanged, via `scripts/issue825_realuser_dispatch.sh` @9e821f906f; outputs at `eval_results/issue_825/real-user-turn-null/`. Five launch attempts (~2.7 GPU-h vs 3 budgeted): one GCP flex-start instance preempted three times mid-extraction by the provider, failed over to RunPod, one wiring-phase crash on a Unicode line-separator JSONL edge case (fixed), then a clean run. Inline free analysis (`crossmodel-map-transfer`, user-requested, 0 GPU-h): cross-model map-swap / representation-swap + weight-space similarity of the two fitted maps on the existing turnstore, `scripts/issue825_crossmodel_map_transfer.py` @5b5b53bbe2, 100 permutation-null draws, ridge core reused verbatim from `issue825_fit_cells.py`.

**Design:** Two models (pretrained Qwen2.5-7B; Qwen2.5-7B-Instruct) × two target roles (assistant, user) × two context formats (chat template, naturalistic transcript). The single-turn track (Track S; n = 5000 prompts per cell, 2 cells) replicates the parent rig on both models; the two-turn track (Track M; n = 2000 conversations per cell) fits 8 within-cell maps plus 4 cross-role transfer cells. Round 3 adds a user-turn-provenance arm: the same 4 user cells refit on conversations whose second user turn each measured model wrote itself. Round 4 completes the provenance axis with real conversations: the full 8-cell grid refit on 2,000 fully-logged human 2-turn conversations (both user turns human-written; first answers from the logged serving models), plus 1 parent anchor. Single seed throughout (fit seed 0, generation seed 42).

**Training:** N/A — no model training. Complete analysis + generation constants:

| Parameter | Value | Source |
|---|---|---|
| Ridge fit | closed-form ridge, generalized-cross-validation-selected λ, K = 5 folds, fit seed 0 | [#779](https://eps.superkaiba.com/tasks/779) `fit_h` |
| Frozen layer set | 14, 18, 19, 26 (headline read at 19) | [#779](https://eps.superkaiba.com/tasks/779) / [#722](https://eps.superkaiba.com/tasks/722) |
| Shuffle nulls | selection-symmetric, 20 draws (ridge), 5 draws (MLP) | [#778](https://eps.superkaiba.com/tasks/778) / [#779](https://eps.superkaiba.com/tasks/779) |
| Bootstrap CIs | n_boot = 1000 | plan §11 |
| Track-S sampling | 1 sample/prompt, T = 1.0, top_p = 0.95, max_tokens = 1024, seed 42, chat template | [#779](https://eps.superkaiba.com/tasks/779) pass-B parity |
| Track-M assistant turns | vLLM greedy, seed 42 | task body |
| Track-M second user turn | claude-haiku-4-5, T = 1.0, 22 rotated user-persona briefs | plan §11 (tier-3 synthetic) |
| Round-3 second user turn | written by the measured model itself (vLLM, T = 1.0, top_p = 1.0, max_tokens = 512, seed 42); chat cells: chat-template render + opened user header, stop `<\|im_end\|>`; naturalistic cells: transcript + `User: `, stop on blank line / `Assistant:` | plan v7 §11 (parent u2 sampling parity) |
| Round-3 row handling | user-cell filters applied at fit time via per-cell row allowlists (kept n = 1914 / 1722 / 1999 / 1738); anchors fit all 2000 rows | plan v7 (anchor row-parity fix) |
| Round-3 support margins | MLP: point delta ≥ +0.10 clearing 2 standard errors; ridge: delta ≥ +0.30 at layer 19 or 26, or any frozen-layer R² > 0 | plan v7 §11 |
| Round-3 audit floors | keep rate ≥ 0.80; distinct-3-gram rate ≥ 0.5 (headline inclusion, non-halting) | plan v7 §11 (80% yield convention) |
| Round-4 conversations | fully-logged real 2-turn lmsys conversations at the parent's pinned revision; strict user/assistant/user prefix, English, non-redacted, non-moderation-flagged, kept n = 2000 of 30,311 streamed; no u2 length cap beyond the 2048-token conversation filter; all 8 cells share the identical kept-2000 rows | plan v11 §2/§4 |
| Round-4 cross-provenance band | MLP descriptive labels at point delta ≥ ±0.10 clearing 2 standard errors (this round's fold SE × √2 proxy, applied uniformly; parent folds not persisted — round-3 folds are persisted and drawn in the three-provenance figure); no support/kill labels on the MLP | plan v11 §1 |
| Round-4 wiring check | own-context vs derangement-shuffled-context teacher-forced NLL, n = 200 rows per (model, format) | plan v11 §7 |
| Row filters | ≥ 8 content tokens, ≤ 2048 total tokens | plan §11 |
| MLP probe | input = full-dimension standardized activations; target reduced to its top-64 PCA components (basis fit on train), predicted jointly, then un-PCA'd; 1 hidden layer of 512 GELU units, AdamW lr 1e-3, weight decay 1e-4, ≤ 300 epochs with early stopping (validation fraction 0.1, patience 20); 1800 s/cell CPU budget | `fit_h.mlp_fit_predict` @e011a0b1ca |
| Per-position store cap | ≤64 positions/turn, peak layers only | plan §11 (storage bound; pilot ≥80% coverage) |
| n per cell | 5000 (Track S) / 2000 (Track M) | [#779](https://eps.superkaiba.com/tasks/779) / plan §11 |
| Models | Qwen2.5-7B (pretrained) vs Qwen2.5-7B-Instruct | plan |
| Activation dtype | bf16 (plan deviation from fp16: fp16 max 65504 overflows Qwen residual outlier dims) | run log |

**Evaluation:** The dependent variable is held-out R² (5-fold cross-validation) of a ridge map h from c_x — the mean residual-stream activation over the context tokens at a given layer — to v(x), the answer-profile vector of the target turn; R² below 0 means the map predicts worse than the global mean. Teacher-forced NLL of the realized target turn is the secondary read. Two read modes: FROZEN (layer set fixed in advance, headline at layer 19) and SELECTION (max over all 28 layers, compared only against selection-symmetric shuffle nulls that inherit the per-draw layer max — Spearman is used for the replication gate's layer-profile comparison). Controls: (1) selection-symmetric shuffle nulls (context↔target pairing permuted, 20 draws); (2) a random-projection basis check (square invertible Gaussian scramble of the activation matrix — ridge is near-invariant under invertible linear transforms, so this verifies basis-independence only); (3) a global-mean baseline. Gates: replication vs the parent rig (layer-profile Spearman + absolute ΔR² at layer 19 within 0.05); render/BPE integrity; two-turn signal above null at n = 2000. Nonlinearity probe: an MLP on the full-dimension standardized activations with a PCA-64 target head (an earlier revision of this table misdescribed the probe as "PCA-64 input"; corrected against `fit_h.mlp_fit_predict` — the PCA reduction applies to the answer-profile target, not the input) — round 1 covered both Track S cells and both Track M chat assistant cells; round 2 extended it to the remaining six cells, so all 10 within-cells are MLP-probed. Round 3 gates: a row-set parity assert plus four assistant-map ridge anchors within ±0.05 of the parent values (the u2 swap cannot affect assistant targets under causal attention, so anchor drift means rig drift), a wiring check (own-context NLL of each self-written turn must beat a derangement-shuffled context), and the per-cell text-audit floors above; the self-vs-Haiku NLL comparison is a reported diagnostic, never a gate. Round 4 gates (binding, evaluated post-upload): ingest floor ≥ 2,000 kept conversations; one end-to-end parent-anchor re-extraction within ±0.05 of the committed value; the wiring check per (model, format); coverage 8 cells + 1 anchor. Round-4 outcome labels follow the plan's outcome lattice (a global break requires within-cell depth consistency, cross-cell consistency at one layer, or a Bonferroni-surviving single read — never one uncorrected CI). All round-4 cross-provenance reads are bound by the plan's interpretation rule, carried verbatim:

> **Binding interpretation rule (MF-R — mirrors v7's MF-F; the analyzer and clean-result MUST carry it verbatim):**
> 1. The real cells differ from the parent/v7 cells in **conversation sample, a1 authorship (logged serving models, e.g. vicuna/ChatGPT-class, not the measured model), and u2 authorship** — a bundled "real conversation" change. Cross-provenance deltas are DESCRIPTIVE provenance-bundle claims only; no claim may attribute a delta to u2 realness specifically.
> 2. Within-round claims (licensed): existence/absence of linear and nonlinear user-turn maps ON real conversations, read against the same-conversation assistant reference cells and this round's own nulls.
> 3. Null-persists licenses: "the user-turn linear null holds under all three tested u2 provenances, each on its own conversation distribution" — a scope-union claim, no mechanism.
> 4. Null-breaks licenses: "on real 2-turn lmsys conversations the user-turn map is linearly decodable (R² = X)" — it does NOT identify which bundled component drives the break. The isolating control (regenerate a1 with the measured model on the same real conversations, splicing the real u2 behind it) is a named candidate follow-up, NOT part of this round.
> 5. Scope note carried to the clean-result: humans who write a second turn are a self-selected subpopulation of lmsys users (continuation selection); the real-u2 read is a statement about that subpopulation.

**Gate outcomes (every rig gate across the four rounds PASSed):**

- Parent gates PASS: replication layer-profile rank correlation 0.963, layer-19 R² difference 0.004 (gate 0.05); two-turn signal selection read 0.093 vs shuffle null ≈ −0.036.
- Render/BPE-integrity gate PASS on regenerated evidence (the original was lost at pod teardown): cross-format span mismatch 3.0% (241/8000; gate ≤ 10%).
- Self-written-turn round (all PASS): anchor row-set parity; four ridge anchors within ±0.004 of parent; own-context NLL beats shuffled by 0.5–2.6 nats; coverage 8/8 ridge + 4/4 MLP; text-audit floors 4/4.
- Real-conversation round (all PASS): ingest floor (2,000 kept of 30,311 streamed); parent-anchor re-extraction ΔR² 0.0005 (gate ±0.05); wiring check (own-context NLL beats shuffled by 1.31–1.75 nats, n = 200 per cell); coverage 8 cells + 1 anchor.
- Planned-vs-actual coverage: all planned cells across four rounds are present (round-2 8/8 ridge + 6/6 MLP; round-3 4/4 user cells + 4/4 anchors + matched-rows refits; round-4 8/8 cells + anchor); no silent drops beyond declared row filters. Single seed holds the headline at MODERATE; the user-turn linear null (three provenances) MODERATE; the sign-flip LOW; cross-provenance deltas descriptive only.

**Data extraction:** Contexts from lmsys-chat-1m (established-dataset tier), pinned revision `200748d9d3cddcc9d782887541057aca0b18c5da`. Track S: single real user prompts, responses sampled from each model. Track M: real lmsys first user turn, assistant turns generated by the target model (vLLM greedy, seed 42), second user turn synthesized by claude-haiku-4-5 at T = 1.0 under 22 rotated user-persona briefs (tier-3 LLM-synthetic; distinct-3-gram rate 0.781). Kept n = 2000 of 2600 overprovisioned conversations (largest drop classes: 276 over-length, 82 short-turn). Activations: teacher-forced 28-layer capture in bf16, sharded turnstore persisted to the HF data repo. Plan deviations: (1) the second user turn was generated via synchronous pooled `messages.create` rather than the Batch API; (2) bf16 activations instead of the plan's fp16 (fp16 overflows Qwen residual outlier dimensions). Round 3 replaces ONLY the second user turn: each measured model continues its own kept-2000 conversation as the user (chat cells: chat-template render plus an opened user header; naturalistic cells: transcript continuation after `User: `), giving 3-turn conversations — the second assistant turn is not generated, which under causal attention cannot affect any slot this round reads. All generated texts, audit metadata, row allowlists, and wiring-check reads persisted per cell. Round 4 ingests complete real 2-turn conversations from the same pinned revision (streaming, no bulk download; no generation): kept 2,000 of 30,311 streamed — drop classes: 11,324 without a strict user/assistant/user prefix, 7,972 redacted, 6,792 non-English, 1,768 short-turn, 431 moderation-flagged, 23 duplicate first turns, 1 over-length. First-answer authorship follows the logged serving model (vicuna-13b 1,096 of 2,000 conversations; 24 other models); 381 of the 2,000 first turns overlap the parent's kept set. Real second-user-turn pool: mean 51.2 tokens (sd 92.7), distinct-3-gram rate 0.919 (Haiku reference 0.781), within-turn repetition rate 0.011; 87 of 2,000 second turns repeat the first verbatim and 293 (14.7%) are near-repeats (lowercased whitespace-token set Jaccard ≥ 0.8, punctuation retained, exact repeats included) — real users often retry or refine — and one row is lmsys moderation boilerplate. Teacher-forced 28-layer capture as in the parent; turnstore, ingestion texts, and wiring reads persisted.

**Sample training/evaluation data + completions:**

First row (row 0) of the 2000 kept Track M conversations — deterministic first row, not a random sample; excerpt truncated (`...`). Complete artifact: [conversations.jsonl (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/deb7a4523b5233393e4fbd2497622527b3622d35/issue825_userbase_map/raw_completions/generation).

```json
{"conv_id": 0, "u1": "how can identity protection services help protect me against identity theft", "a1": "Identity protection services can help you protect against identity theft in several ways. Here are some key features and benefits of these services:\n\n1. **Monitoring and Alerts**:\n   - **Credit Report Monitoring**: Services often provide real-time monitoring of your credit reports from major credit bureaus (Equifax, Experian, TransUnion). This allows you to quickly detect any unauthorized activity or changes to your credit profile.\n   - **Fraud Alerts**: Some services can set up fraud alerts on your credit files, making it harder for identity thieves to open new accounts in your name.\n   - **Activity Alerts**: Receive notifications about any suspicious activity, such as new accounts opened in your name, changes to your address, or unusual transactions.\n\n2. **Credit Freeze**:\n   - **Freeze Your Credit**: Services can help you freeze your credit, which prevents creditors from accessing your credit report. This makes it much harder for identity thieves to open new accounts in your name.\n\n3. **Identity Theft Protection**:\n   - **Assistance with Identity Theft**: If you become a victim of identity theft, these services often provide assistance, such as helping you file a police report, working with creditors to close fraudulent accounts, and providing legal support.\n\n4. **Credit Score Monitoring**:\n   - **Track Your Credit Score**: Services can help you moni
...
```

First row (row 0) of the 5000 Track S single-turn prompts — deterministic first row, not a random sample; excerpt truncated (`...`). Complete artifact: [track_s.jsonl (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/deb7a4523b5233393e4fbd2497622527b3622d35/issue825_userbase_map/raw_completions/track_s).

```json
{"prompt_idx": 0, "prompt": "how can identity protection services help protect me against identity theft", "response": "Identity protection services can help safeguard you against identity theft in several ways. Here’s a breakdown of how these services work and the benefits they offer:\n\n1. **Monitoring and Alerts**:\n   - **Credit Reports**: Services often monitor your credit reports from major bureaus (like Equifax, Experian, and TransUnion) for any suspicious activity.\n   - **Fraud Alerts**: They can set up fraud alerts on your credit reports, making it harder for identity thieves to open accounts in your name.\n   - **Activity Alerts**: Some services send you alerts for any new account activity or changes to your credit report, allowing you to act quickly if something seems off.\n\n2. **Credit Freeze**:\n   - A credit freeze restricts access to your credit report, making it difficult for identity thieves to open new accounts in your name. This is a strong defense against identity theft.\n\n3. **Document Management**:\n   - Services can store copies of your important documents securely, such as ID cards, social security cards, and birth certificates. They may also provide guidance on how to securely store these documents.\n\n4. **Credit Score Monitoring**:\n   - Regularly monitor your credit score and history to detect unauthorized changes or new inquiries that could indicate fraud.\n\n5. **Identity Theft Insurance**:\n   - Some services offer insurance that can
...
```

Round 3, self-written second user turn, instruct model under the chat template — first row (row 0), deterministic, not a random sample; same conversation as the Track M row above. Complete artifact: [conversations_instruct_chat.jsonl (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/a16f792625b879f012fd97de321b9b1d216ef5fa/issue825_onpolicy_user_turn/raw_completions/generation/conversations_instruct_chat.jsonl).

```json
{"conv_id": 0, "u2": "how long does it take for identity protection services to start working after signing up"}
```

Round 3, self-written second user turn, pretrained model under the chat template — first row (row 0), deterministic, not a random sample; excerpt truncated (`...`) — this cell's typical length/role-drift pathology is visible (the "user" turn slides into assistant voice). Complete artifact: [conversations_pretrained_chat.jsonl (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/a16f792625b879f012fd97de321b9b1d216ef5fa/issue825_onpolicy_user_turn/raw_completions/generation/conversations_pretrained_chat.jsonl).

```json
{"conv_id": 0, "u2": "how long can i keep my identity protected?把自己的身份保护多长时间可以\n\nAs an AI language model created by Alibaba Cloud, I cannot provide personalized advice for everyone, since everyone's situation is different. The duration of identity protection ultimately depends on your specific needs, budget, and the features offered by the specific identity protection service you're using.\n\nHowever, I can share some general information to
...
```

Round 4, real logged 2-turn conversation — first row (row 0) of the 2,000 kept conversations, deterministic first row, not a random sample; first-answer excerpt truncated (`...`); both user turns and the first answer are verbatim from the logged conversation (first answer written by the logged serving model, here vicuna-13b). Complete artifact: [conversations_real2turn.jsonl (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/5aa6de1b97895cf8883c44165fa8835ff73e9e93/issue825_real_user_turn_null/raw_completions/ingestion/conversations_real2turn.jsonl).

```json
{"conv_id": 0, "u1": "Beside OFAC's selective sanction that target the listed individiuals and entities, please elaborate on the other types of US's sanctions, for example, comprehensive and sectoral sanctions. Please be detailed as much as possible", "a1": "The United States has a number of different types of sanctions that it can use to achieve its foreign policy goals, including both selective and comprehensive sanctions.\n\nSelective sanctions are targeted at specific individuals or entities that are believed to be engaged in activities that are contrary to US interests. These sanctions can take a variety of forms, including asset freezes, travel ba
...", "u2": "are there other types of US sanctions that you didn't specified earlier? Please elaborate more", "lmsys_conversation_id": "1e230e55efea4edab95db9cb87f6a9cb", "a1_model": "vicuna-13b"}
```

## Results

### The map exists in pretrained Qwen2.5-7B at 87.3% of instruct strength (0.588 vs 0.673 at layer 19)

What is plotted: held-out R² of the ridge map by layer (all 28 layers) for the two single-turn cells — instruct and pretrained — with 95% bootstrap CIs and the shuffle-null band; n = 5000 prompts per cell.

![Held-out R-squared by layer for the single-turn instruct and pretrained maps with shuffle-null band near zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/745e62f4b6cbb510d529a8778cb0137b98522cc8/figures/issue_825/s_track_layer_curves.png)

> **Figure.** *The pretrained base model carries the single-turn linear map at 87.3% of instruct strength.* Held-out R² by layer: instruct 0.673 (95% CI 0.666–0.681) vs pretrained 0.588 (95% CI 0.579–0.598) at frozen layer 19; shuffle nulls ≈ −0.02 at every layer. n = 5000 prompts per cell.

The map is not created by an explicit post-training stage: the pretrained checkpoint reaches 87.3% of instruct strength frozen-vs-frozen (selection read 0.598 at layer 27). Pretraining corpora contain chat-formatted text, so chat-data exposure is not ruled out. Non-spuriousness rests on the global-mean baseline (−0.0017) and the shuffle nulls; the random-projection arm (0.684 ≈ observed) is an invertible scramble that ridge is near-invariant to, so it verifies basis-independence only. Per-example predictions were not persisted (cell aggregates and bootstrap draws only — deliberate size scoping; regenerable from the turnstore per the footer recipe), so no predicted-vs-actual scatter is shown.

### Representation-swap preserves the map's predictive information across base↔instruct while map-swap fails — consistent with a large read-out rotation

What is plotted: layer-19 held-out R² for five matched cells under three fit directions — within-model, map-swap (base-fit map applied to instruct activations), representation-swap (map re-fit from base activations to instruct targets) — with shuffle-null means; y clipped at −2.2.

![Cross-model map transfer at layer 19 for five cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4d03165dd8a1773790a1b501b3699c1a6e08e584/figures/issue_825/crossmodel_transfer_r2.png)

> **Figure.** *The information transfers; the coordinates do not.* Single-turn chat: representation-swap 0.587 ≈ base within-model 0.588; map-swap −0.066 vs instruct within-model 0.673. Two-turn cells repeat the pattern at their lower within-model levels. n = 5000 (single-turn) / 2000 (two-turn) shared conversations.

Representation-swap recovers near within-model R² (0.587 / 0.662, single-turn layer 19) while map-swap fails (−0.066 / −0.170): the swap preserves predictive information on these matched cells, consistent with a large coordinate/read-out rotation — no mechanism claim. Weight-space evidence is mixed: the per-output-dimension coefficient cosine is far above chance (median 0.207, random reference 0.000), the mean top-50 principal-angle cosine 0.856 barely clears its 0.835 random-map reference (k = 10: 0.817 vs 0.752), and the Procrustes-aligned cosine 0.686 is exploratory (no chance reference). Descriptive; single seed, 100 permutation nulls.

What is plotted (underlying data): per-assistant-pair top-50 principal-angle cosines (dotted: each pair's random-map reference) and the per-output-dim coefficient cosine histogram (single-turn chat, layer 19).

![Weight-space similarity of the base and instruct maps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/37a5ba8a6a0293e90d400d9c4a069f3a6fe57d3d/figures/issue_825/crossmodel_map_similarity.png)

> **Figure.** *Per-dimension agreement is far above chance; subspace overlap barely clears it.* Per-output-dim cosine median 0.207 (random reference ≈ 0.000, sd 0.0003); k = 50 principal-angle cosines 0.99–0.04, mean 0.856 vs the 0.835 random-map reference (dotted). Single-turn chat, layer 19, n = 5000.

### Two-turn conversation targets collapse the ridge map: best frozen-layer R² falls to +0.076 (instruct) and −0.461 (pretrained)

What is plotted: best frozen-layer held-out R², one bar per within-cell map — the two single-turn cells (n = 5000) plus all eight two-turn cells (model × format × target role, n = 2000) — sorted by value, 95% bootstrap CI whiskers.

![Best frozen-layer held-out R-squared per two-turn within-cell map, bars per model and format](https://raw.githubusercontent.com/superkaiba/explore-persona-space/745e62f4b6cbb510d529a8778cb0137b98522cc8/figures/issue_825/within_cell_best_frozen_r2.png)

> **Figure.** *Two-turn targets collapse the ridge map.* Best frozen-layer held-out R² for the assistant cells: instruct/chat +0.076, instruct/naturalistic −0.078, pretrained/chat −0.461, pretrained/naturalistic −0.390; n = 2000 conversations per cell. The single-turn cells read 0.59–0.67 on the same rig.

Frozen-layer-19 reads with 95% CIs: instruct/chat +0.076 (0.052 to 0.098), instruct/naturalistic −0.078 (−0.106 to −0.050), pretrained/chat −0.461 (−0.504 to −0.415), pretrained/naturalistic −0.390 (−0.432 to −0.346); selection-symmetric nulls sit at −3.6 to −3.8. Not purely sample size: the single-turn map at a matched n = 2000 subsample reads 0.321 ≫ 0.093, though the power curve is non-monotone (0.452 / 0.499 / 0.495 / 0.321 / 0.673 at n = 250–5000) and the tracks change corpus, turn count, and n together. Per-position reads are late-layer dominated (layer 26 best at 12 of 12 positions, instruct/chat); layer-0–1 selection argmaxes in three cells may reflect formatting/role information.

### The collapse is nonlinearity, not absence: an MLP probe recovers R² 0.557 (instruct) and 0.487 (pretrained) where ridge reads +0.076 and −0.461

What is plotted: layer-19 held-out R², ridge vs MLP bars side by side, for all four MLP-probed cells — the two single-turn cells (n = 5000) and the two two-turn chat assistant cells (n = 2000).

![Layer-19 held-out R-squared, ridge versus MLP probe, for the two two-turn chat assistant cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/72a249c79239ac8548681b6682ff83b16a06789f/figures/issue_825/ridge_vs_mlp_l19.png)

> **Figure.** *An MLP probe recovers what ridge misses at layer 19.* Held-out R², MLP vs ridge: instruct/chat 0.557 vs 0.076; pretrained/chat 0.487 vs −0.461. Both cells ran to completion with full null draws. n = 2000 conversations per cell.

Both two-turn chat cells completed within budget (all four frozen layers, full 5-draw nulls; layer-19 MLP shuffle nulls max −0.009 instruct / −0.008 pretrained, far below the observed 0.557 / 0.487) — complete reads, not lower bounds. In the single-turn cells the MLP ≈ ridge (0.654 vs 0.673 instruct; 0.587 vs 0.588 pretrained at layer 19), so the single-turn map is essentially linear; the budget flag tripped only there, truncating layer-26 MLP null draws (3 usable of 5, both cells) with no observation missing. A follow-up round probed the remaining six cells (naturalistic assistant + all user cells); see the follow-up result below.

### User-turn maps have no practical linear predictive power (R² −1.08 to −1.32) despite sitting above shuffle nulls

What is plotted: held-out R² by layer for all eight two-turn cells — assistant (solid) and user (dashed) curves for both models, one panel per format; n = 2000 conversations per cell.

![Held-out R-squared by layer for the four two-turn user-target cells, all deeply negative at late layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6c752ab6e3805bbc68d1d71e1987f2dc0b48e71b/figures/issue_825/m_track_layer_curves_role.png)

> **Figure.** *User-target maps predict worse than the global mean at every late layer.* Held-out R² by layer for the four user cells; layer-26 values −1.08 to −1.32. The chat-panel layer-0 selection point (0.283) sits off the dominant curve and is nearly invisible at this scale; shuffle-null bands are not drawn. n = 2000 per cell.

All four user cells read R² between −1.08 and −1.32 at layer 26 (95% CIs spanning −1.02 to −1.38; per-cell selection-symmetric nulls −2.45 to −2.96) — worse than predicting the global mean — yet every cell sits above its null band: structure exists, practical linear predictive power does not (the follow-up result below shows an MLP recovers part of it). Teacher-forced NLL agrees (user 2.04–2.64 vs assistant 0.32–0.59). The second user turn is LLM-written (claude-haiku-4-5, 22 rotated briefs), so this cell establishes the null for LLM-simulated users; the self-written-turn results below show the linear map improves — but stays below zero — when the model writes the turn itself, and the real-conversation result below completes the provenance axis.

### Pretrained assistant→user transfer sign-flips across depth (ΔR² −0.62 at layer 14 to +0.23 at layer 26) — provisional

What is plotted: paired ΔR² (assistant-fit map applied to user targets, minus the same-role paired baseline) with 95% bootstrap CIs at the four frozen layers, for both models; n = 2000 conversations per cell.

![Paired delta R-squared for assistant-to-user transfer at four layers, pretrained flipping from negative to positive with depth](https://raw.githubusercontent.com/superkaiba/explore-persona-space/11d933bba760b7dcf1e089ad98fffc202d6b649e/figures/issue_825/crossrole_paired_delta_r2.png)

> **Figure.** *Assistant→user transfer flips sign with depth in the pretrained model only.* Paired ΔR² at layers 14 / 18 / 19 / 26: pretrained −0.62 → +0.23; instruct negative at all four layers. n = 2000 conversations per cell; provisional (single seed).

Pretrained assistant→user deltas by layer: −0.623 / −0.171 / +0.124 / +0.229 at layers 14 / 18 / 19 / 26, each 95% CI excluding zero (e.g. layer 19: +0.013 to +0.242; whiskers in the figure). Instruct is negative at all four layers (−0.709 to −0.214, every CI excluding zero), so the positive transfer is a pretrained-only observation — no mechanism claim. The paired baseline is the previous same-role turn, not validated as topic-only; single seed, n = 2000 — provisional.

What is plotted (underlying data): the 1000 bootstrap ΔR² draws per cell-layer as violins, both models' assistant→user cells — the raw draws behind the summary bars above.

![Bootstrap distributions of paired delta R-squared for assistant-to-user transfer, violins per layer and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1a9a23418e671e38d985226fee93586f2a5a2e08/figures/issue_825/crossrole_delta_r2_bootstrap_distributions.png)

> **Figure.** *Underlying bootstrap draws for the transfer deltas.* Pretrained distributions cross zero between layers 18 and 19; instruct stays below zero throughout. Medians marked; n = 2000 per cell.

### Follow-up round: the MLP recovers a weak user-turn map and the full naturalistic assistant map — the linear-only reading of the user null is confirmed

What is plotted: best frozen-layer held-out R² for the linear ridge vs the MLP probe, paired bars for all 10 MLP-probed cells across both rounds (2 single-turn, 2 chat assistant, 2 naturalistic assistant, 4 user), with the follow-up plan's 0.2 recoverability line drawn.

![Ridge versus MLP held-out R-squared for all ten probed cells with the 0.2 recoverability line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a302438d6890ec953c39be72a5f0e9865ad695fe/figures/issue_825/ridge_vs_mlp_all_cells.png)

> **Figure.** *The nonlinear map extends to every two-turn cell.* MLP recovers 0.50–0.53 in naturalistic assistant cells (≈ chat siblings) and 0.19–0.23 in user cells where ridge is deeply negative; single-turn cells stay linear (MLP ≈ ridge). n = 2000 per two-turn cell.

The follow-up plan's falsification criterion was met: MLP held-out R² clears the 0.2 line in 3 of 4 user cells (instruct/chat 0.228, instruct/naturalistic 0.227, pretrained/naturalistic 0.218; pretrained/chat 0.188), all ≈ 50 null-band-widths above their 5-draw shuffle nulls (≈ −0.003), peaking at layer 26 where the user ridge argmax also sat. Naturalistic assistant cells recover at 0.534 / 0.499 (layer 19) — format gated only the linear map. Both ridge anchors reproduced the parent exactly (delta +0.0000). Read: a weak nonlinear user-turn map exists (~2.4× below the assistant map), so the parent's user-turn null is a statement about linear decodability, not absence.

### Self-written user turns are more predictable than Haiku-written ones in three of four cells

What is plotted: MLP-probe held-out R² at the best frozen layer per user cell — second user turns written by the measured model itself (round 3) vs the parent's Haiku-written ones — with the five per-fold values and ±2-standard-error whiskers on the self-written bars; n = 1722–1999 conversations per cell.

![MLP best-frozen held-out R-squared per user cell, self-written versus Haiku-written second user turns with per-fold points](https://raw.githubusercontent.com/superkaiba/explore-persona-space/932e0ab860fe84f2f6e918d8b8a87b8c79c3614a/figures/issue_825/onpolicy_user_provenance_mlp.png)

> **Figure.** *Self-written user turns are more predictable in three of four cells.* MLP best-frozen R², self vs Haiku: instruct/chat 0.329 vs 0.228; instruct/naturalistic 0.275 vs 0.227; pretrained/naturalistic 0.318 vs 0.217; pretrained/chat 0.084 vs 0.188. Parent fold-level values were not persisted, so parent bars carry no whiskers. n = 1722–1999.

Two cells (instruct/chat, pretrained/naturalistic) pass the MLP support margin (Methodology table) — point delta ≥ +0.10 clearing two standard errors — plus the ridge margin; instruct/naturalistic passes on ridge only (MLP delta +0.048). Per the plan's binding interpretation rule, a PASS licenses ONLY the descriptive provenance-bundle claim — self-generated second-user-turn cells are more row-predictable than externally-written ones — and does NOT license "the user-header state encodes the model's own predicted turn": any-model-generated, lower-entropy, or shorter text could be generically easier to predict. The cross-model mismatched-context control is a named follow-up. Single seed.

### The linear-map gain survives matched-rows refits of the parent cells and is not explained by shrunken target variance

What is plotted: ridge held-out R² by layer (all 28) per user cell — self-written turns, Haiku-written turns (full 2000 rows), the matched-rows parent refit (run for the two naturalistic cells only), and this round's per-layer shuffle-null band; frozen layers marked.

![Ridge held-out R-squared by layer for the four user cells, self-written versus Haiku-written and matched-rows refits with null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/932e0ab860fe84f2f6e918d8b8a87b8c79c3614a/figures/issue_825/onpolicy_user_ridge_layer_curves.png)

> **Figure.** *Self-written turns lift the linear map at every late layer except pretrained/chat.* Layer-19 reads: instruct/chat −0.77 vs parent −1.43; instruct/naturalistic −0.65 vs matched refit −0.98; pretrained/naturalistic −0.62 vs −1.27; pretrained/chat −1.84 vs −1.49. All four self cells sit above their per-layer nulls. n = 1722–1999.

Row filtering alone flatters the raw comparison — the parent cells improve when refit on the self-written cells' kept rows (instruct/naturalistic −1.61 → −0.98 at layer 19) — so the matched-rows deltas are the honest linear read: +0.33 / +0.20 (layers 19 / 26, instruct/naturalistic) and +0.65 / +0.49 (pretrained/naturalistic), against the +0.30 ridge margin. Every ridge value stays below zero: no practical linear predictive power under any tested provenance. Shrunken target diversity does not explain the gain — the targets' total variance (trace of the answer-profile covariance) is ~1.8× the matched refit at layer 19. Self-written-turn NLL (diagnostic, not a gate): instruct 1.33 / 1.45, below the 2.04–2.64 Haiku band; pretrained 2.62 / 2.72.

### Pretrained chat-header self-generation degenerates, and its map weakens rather than strengthens

What is plotted: the per-cell text audit of self-written second user turns — token length mean ± sd, within-turn repetition rate, distinct-3-gram rate, and filter keep rate — against the parent Haiku reference and the headline floors (keep ≥ 0.80, distinct-3-gram ≥ 0.5).

![Text audit panels for the four self-written user-turn cells: length, repetition, lexical diversity, keep rate](https://raw.githubusercontent.com/superkaiba/explore-persona-space/932e0ab860fe84f2f6e918d8b8a87b8c79c3614a/figures/issue_825/onpolicy_user_text_audit.png)

> **Figure.** *All four cells pass the coarse floors; pretrained/chat is the outlier on length and repetition.* Mean u2 length 307 tokens (parent 79); 3-gram-repetition rate 0.090 vs ≤0.005 elsewhere; distinct-3-gram 0.81 and keep rate 0.9995 still pass their floors. n = 1722–1999 kept rows per cell.

The all-True audit verdicts mean only that each cell cleared both floors and enters the headline table — the floors are coarse whole-cell statistics. Row-level reads show many pretrained/chat "user turns" are not user-like: median 186 words, 9.0% of rows repeat a 3-gram ≥5 times, ~0.6% are symbol strings, and row 0 drifts mid-turn into assistant voice (sample block above). This is the one cell where self-written text predicts worse than the external baseline (MLP delta −0.10; ridge −0.35 to −0.65) — the plan's named provenance-sensitive-negative outcome, plausibly text degeneration rather than a map property; the causal direction is not identified here.

### The user-turn linear null holds on real human conversations, completing the provenance axis; the weak nonlinear map replicates (MLP 0.17–0.19)

What is plotted: user-cell held-out R² under the three second-user-turn provenances (real, Haiku-written, self-written): ridge at layer 19 (left; 95% bootstrap CIs on real cells) and MLP at the best frozen layer (right; per-fold points and ±2-standard-error whiskers on the real and self-written bars; parent folds not persisted). n = 2000 (real, Haiku); self-written kept allowlists n = 1914/1722/1999/1738 (x-axis cell order).

![Three-provenance user-map comparison, ridge and MLP panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/37a5ba8a6a0293e90d400d9c4a069f3a6fe57d3d/figures/issue_825/realuser_three_provenance_user_map.png)

> **Figure.** *The user-turn linear map stays below zero under every tested provenance; the weak nonlinear map replicates on real conversations.* Real ridge at layer 19: −0.99 to −1.12, every 95% CI below zero; real MLP best-frozen 0.166–0.186, within 0.06 of Haiku cells. Fold points: real and self bars (parent folds not persisted). n = 2000 (real, Haiku); 1722–1999 (self-written).

All four real cells read ridge R² −0.92 to −1.12 at both headline layers, every 95% CI below zero yet above the shuffle nulls — the plan's null-persists outcome. Licensed claim (rule MF-R, Methodology): the user-turn linear null holds under all three tested user-text provenances, each on its own distribution — scope-union, no mechanism. The nonlinear map replicates (MLP 0.166–0.186, nulls ≈ −0.01; within 0.06 of the Haiku cells). Scope: lmsys users who wrote a second turn (self-selected).

What is plotted (underlying data): per-layer ridge R² per model and format — assistant and user curves with the user cell's 20-draw shuffle-null band.

![Per-layer ridge curves with shuffle-null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/dec33d66ff444296c83c2b7b94c378be452883c7/figures/issue_825/realuser_ridge_layer_curves.png)

> **Figure.** *The per-layer view behind the frozen-layer reads.* In every panel the user curve rises toward late layers but stays below zero everywhere, above its null band; the assistant curve sits higher at every layer. n = 2000 per cell.

### Real logged conversations depress the assistant map too — uniformly on the MLP, on ridge everywhere except the already-negative pretrained/chat cell — and the assistant−user gap persists

What is plotted: within-round paired comparison on the same 2,000 real conversations — assistant vs user held-out R² per (model, format). Left: ridge at layer 19 with 95% bootstrap CI whiskers. Right: MLP at the best frozen layer with five per-fold points per bar.

![Assistant versus user held-out R-squared on real conversations, ridge and MLP panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/dec33d66ff444296c83c2b7b94c378be452883c7/figures/issue_825/realuser_assistant_vs_user_contrast.png)

> **Figure.** *Both roles drop on real conversations; the role ordering survives.* Assistant ridge at layer 19: −0.27 to −0.61 (the parent's model-generated instruct/chat cell read +0.076); assistant−user gap +0.43 to +0.78 (ridge) and +0.14 to +0.20 (MLP). n = 2000 per cell.

The parent anchor isolates the rig: re-extracting the parent's own conversations reproduces layer-19 R² +0.0763 vs the committed +0.0757 (gate ±0.05), so the drop is a data property of the real-conversation bundle — conversation sample, first-answer authorship (55% vicuna-13b), and user-turn authorship change together; rule MF-R bars single-component attribution. The MLP drop is uniform (−0.16 to −0.18 in all four cells); the ridge drop is not — pretrained/chat, already deeply negative, barely moves (−0.461 → −0.471) while the other three cells drop 0.22–0.35. Role ordering survives in every cell. Wiring check PASS (own-context NLL beats shuffled by 1.31–1.75 nats, n = 200); real-u2 NLL 3.37 / 2.66 (instruct) is at or above the 2.04–2.64 Haiku band — diagnostic, not a gate.

---

**Repro:** Compute: GCP 1× A100-80, ~6.5 h final run (7 launch attempts across crash-fix rounds; 8 GPU-h budget). Code: run pinned at [e011a0b1ca](https://github.com/superkaiba/explore-persona-space/tree/e011a0b1cabd26be6cfd227757650f5f49ca7b12); eval JSONs (34 files) committed at [c9bf728fa1](https://github.com/superkaiba/explore-persona-space/tree/c9bf728fa1/eval_results/issue_825) on branch issue-825. Artifacts: HF data repo [superkaiba1/explore-persona-space-data @ deb7a452](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/deb7a4523b5233393e4fbd2497622527b3622d35/issue825_userbase_map), prefix `issue825_userbase_map/` — `analysis_tensors/` (111 GB turnstore), `raw_completions/` (incl. `conversations.jsonl` + `track_s.jsonl`), `eval_results_mirror/` (34 JSONs). WandB: n/a (analysis-only run). Plan: plan v4 (parent) + plan v6 (round-2 amendment) + plan v7 (round-3 amendment) + plan v11 (round-4 amendment) under `plans/` in the task folder (resolve via `uv run python scripts/task.py find 825`). Round-2 artifacts: git `c4dcf8c907` (`eval_results/issue_825/mlp-unprobed-cells/`), HF [eval_results_mlp_unprobed](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/72ddfc466a53287a6f421707f14fd4efa44c38f7/issue825_userbase_map/eval_results_mlp_unprobed), wrapper @`07589e890b`. Round-3 (`onpolicy-user-turn`) artifacts: HF prefix [issue825_onpolicy_user_turn @ a16f7926](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a16f792625b879f012fd97de321b9b1d216ef5fa/issue825_onpolicy_user_turn) — `raw_completions/generation/` (4 per-cell conversation JSONLs + audit meta + row allowlists + wiring-check reads), `analysis_tensors/` (turnstore shards), `eval_results_mirror/` (28 JSONs); eval JSONs under `eval_results/issue_825/onpolicy-user-turn/` on branch issue-825; run code @ [debc28c31d](https://github.com/superkaiba/explore-persona-space/tree/debc28c31df0474212ead4448e9a7308dbd5fbc6); round figures @ [932e0ab860](https://github.com/superkaiba/explore-persona-space/tree/932e0ab860fe84f2f6e918d8b8a87b8c79c3614a/figures/issue_825); 2.639 GPU-h measured (3 budgeted). Round-4 (`real-user-turn-null`) artifacts: HF prefix [issue825_real_user_turn_null @ 5aa6de1b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5aa6de1b97895cf8883c44165fa8835ff73e9e93/issue825_real_user_turn_null) — `raw_completions/ingestion/` (`conversations_real2turn.jsonl` + meta), `raw_completions/wiring/` (row allowlists + wiring reads), `analysis_tensors/` (`turnstore_realuser` + `turnstore_anchor_parent` shards), `eval_results_mirror/` (29 JSONs); eval JSONs under `eval_results/issue_825/real-user-turn-null/` committed at [2ea50d6eb8](https://github.com/superkaiba/explore-persona-space/tree/2ea50d6eb8b163a1ef6a5d7bfac5e1b670b2a033/eval_results/issue_825) on branch issue-825; run code @ [9e821f906f](https://github.com/superkaiba/explore-persona-space/tree/9e821f906f4e63489967a385d6537d23b3c45f16); round figures @ [dec33d66ff](https://github.com/superkaiba/explore-persona-space/tree/dec33d66ff444296c83c2b7b94c378be452883c7/figures/issue_825), three-provenance hero revised (round-3 fold points added) @ [37a5ba8a6a](https://github.com/superkaiba/explore-persona-space/tree/37a5ba8a6a0293e90d400d9c4a069f3a6fe57d3d/figures/issue_825); 2.706 GPU-h measured (3 budgeted). Cross-model free analysis: `eval_results/issue_825/crossmodel_map_transfer/` + figures @ [4d03165dd8](https://github.com/superkaiba/explore-persona-space/tree/4d03165dd8a1773790a1b501b3699c1a6e08e584/figures/issue_825), map-similarity figure revised (per-pair dotted reference lines added) @ [37a5ba8a6a](https://github.com/superkaiba/explore-persona-space/tree/37a5ba8a6a0293e90d400d9c4a069f3a6fe57d3d/figures/issue_825), code @ [5b5b53bbe2](https://github.com/superkaiba/explore-persona-space/tree/5b5b53bbe233a2cfb141836997a240af4a33ea48), 0 GPU-h.

**Context:** Created 2026-07-01 from user chat. Verbatim originating prompt:

> # Experiment: Is the context vector to answer profile mapping present in the base model and does it hold for the user?
> ## Motivation
> - We showed that there is a context vector to answer profile mapping (issue 722 I think) for the assistant in the Qwen 2.5 7B instruct model
> - We are interested now in seeing:
>     - is this mapping present in the base model?
>         - how much does it change post instruction -> probably a lot but still good to test
>     - does this mapping exist for the user?
>         - how much does it change post instruction
>         - this user mapping is not super necessary in post training because the model doesn't have to predict the user turn but
> ## Questions
> - Does this even make sense because we can't even get the base model to really generate from a chat template?
> - Do our mapping results hold
> ## Methodology
> - Do the same experiment from issue 722 (linear mapping only -- at all layers) with the following changes:
>     - test corresponding user context vector -> user answer profile  mapping
>     - test in the base model ([https://huggingface.co/Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/tree/d149729398750b98c0af14eb82c78cfe92750796)) -- (make sure this is the proper base model) for both user and assistant
>     - also test the same in the finetuned model
> - Considerations:
>     - we need multi turn conversations to test this -- answer mean should only be taken over the current "speaker's" answer
>     - try one with the chat template and one replacing it with more naturalistic format (e.g. semicolon)
>     - also plot how well we can predict each token in the answer (separately -- per position) with the context vector
>     - also check if you can predict user response from the assistant context vector (and vice versa)
> - Reuse as much code as possible

(Quoted text is verbatim; the one model-repo URL inside the quote is hyperlinked to a pinned revision so the footer's URL-permanence gate holds — the displayed text is unchanged.) Follow-up user decisions (same chat): frame on [#779](https://eps.superkaiba.com/tasks/779)'s per-context (per-example) map rather than [#722](https://eps.superkaiba.com/tasks/722)'s across-context map ("We are interested more in the per-context map"); Track M conversations are generated with Claude Haiku 4.5 as the user; file as proposed only (no autonomous session spawned). Lineage: [#779](https://eps.superkaiba.com/tasks/779) — the parent single-turn instruct rig this run replicates and extends. Run 2026-07-02 (7 launch attempts; crash-fix rounds 1–6: extract host-RAM OOM → block-wise flush → host upsize → GPU Gram-ridge fit). Interpretation reviewed rounds 1–3 (two critic REVISE rounds, then PASS+PASS). Follow-up round `mlp-unprobed-cells` run 2026-07-03 (cheap-band auto-run, redundancy-screened; plan v6; 3 launches, results rescued from the crash-persist upload). Follow-up round `onpolicy-user-turn` run 2026-07-03 (user-requested; verbatim prompt: "run on-policy user turn for both instruct and base model"; plan v7; 4 GCP launches — three extraction crashes on a zero-width-span tokenizer edge in short self-written turns, then a clean run). Follow-up round `real-user-turn-null` run 2026-07-03 (cheap-band auto-run round 2 of 2, source proposer-9b-cheap; plan v11; one GCP flex-start launch preempted three times by the provider → RunPod failover, one wiring-phase crash on a Unicode line-separator JSONL edge, then a clean run). Inline free analysis `crossmodel-map-transfer` run 2026-07-03 (user chat; ask: compare the base model's fitted context→answer-profile map to the instruct model's map directly — cross-model transfer R² plus weight-space similarity of the two fitted ridge maps).

