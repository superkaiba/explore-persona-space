---
title: The context→answer-profile map exists in pretrained Qwen2.5-7B at ~87% of instruct
  strength and is nonlinearly recoverable across two-turn cells — including a weak
  user-turn map (MLP R² 0.19–0.23) that the linear probe misses entirely (MODERATE
  confidence)
kind: experiment
tags:
- followup-auto
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
# The context→answer-profile map exists in pretrained Qwen2.5-7B at ~87% of instruct strength and is nonlinearly recoverable across two-turn cells — including a weak user-turn map (MLP R² 0.19–0.23) that the linear probe misses entirely (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_825.md](https://github.com/superkaiba/explore-persona-space/blob/6b3b5242fff94fe01600a43d69fbeef86bbc6e06/docs/methodology/issue_825.md) · [gist](https://gist.github.com/superkaiba/cd6dd04dbacb42587b5458b918710cdb)

## Takeaways

- **Pretrained Qwen2.5-7B carries the linear context→answer-profile map at 87.3% of instruct strength** (held-out R² 0.588 vs 0.673 at layer 19; shuffle nulls ≈ −0.02).
- Two-turn assistant targets are nonlinear, not absent: an MLP (multilayer perceptron) probe recovers R² 0.49–0.56 across chat AND naturalistic cells where ridge reads +0.076 to −0.461.
- The user-turn null is linear-only: ridge reads −1.08 to −1.32 while the MLP recovers a weak user map — R² 0.19–0.23 at layer 26, nulls ≈ −0.003, 3 of 4 cells above the follow-up plan's 0.2 recoverability line — ~2.4× weaker than the assistant map; scoped to LLM-written second user turns.
- Pretrained assistant→user transfer sign-flips with depth (ΔR² −0.62 at layer 14 → +0.23 at layer 26); provisional — single seed, n = 2000.
- The replication gate vs the parent rig PASSed (layer-profile rank correlation 0.963, ΔR² 0.004 at layer 19); the render-integrity gate PASSed on regenerated evidence (cross-format span mismatch 3.0%, gate ≤ 10%).

## Goal

**This experiment in context:** The parent experiment ([#779](https://eps.superkaiba.com/tasks/779)) established a strong linear map from a context's mean residual-stream activation to the model's answer profile on single-turn Qwen2.5-7B-Instruct, peaking near layer 19. This experiment holds that rig fixed and asks two questions: does the same linear map already exist in the *pretrained* base model (before any post-training), and does it hold when the prediction target is the *user's* next turn rather than the assistant's — extended with a two-turn conversational track and cross-role transfer cells.

**Broader narrative:** Whether the context→answer-profile map is inherited from pretraining or created by post-training locates where context/persona conditioning enters the training pipeline; whether it is assistant-generation-specific or a general next-turn property bounds how far base-model context geometry can be used to predict downstream fine-tuning behavior (the project's leakage-prediction line).

## Methodology

**Rounds:** Round 1 = the parent run (below). Round 2 (`mlp-unprobed-cells`, cheap-band auto-run): the parent's exact MLP probe extended to the 6 unprobed cells via `scripts/issue825_mlp_followup_dispatch.sh` @07589e890b — turnstore staged from HF @deb7a452 (4 m-track prefixes), 8-cell ridge refit as carrier + two-anchor ±0.05 regression gate (PASS, deltas +0.0000), 6-cell MLP, no recipe changes; outputs at `eval_results/issue_825/mlp-unprobed-cells/`. Three launches (~2.5 GPU-h vs 2 budgeted): two stage hangs (xet finalization, fixed by HF_XET_DISABLE=1), then a fully-successful fit whose gate crashed on a wrong dict path — outputs rescued from the crash-persist upload, corrected gate re-run locally.

**Design:** Two models (pretrained Qwen2.5-7B; Qwen2.5-7B-Instruct) × two target roles (assistant, user) × two context formats (chat template, naturalistic transcript). The single-turn track (Track S; n = 5000 prompts per cell, 2 cells) replicates the parent rig on both models; the two-turn track (Track M; n = 2000 conversations per cell) fits 8 within-cell maps plus 4 cross-role transfer cells. Single seed throughout (fit seed 0, generation seed 42).

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
| Row filters | ≥ 8 content tokens, ≤ 2048 total tokens | plan §11 |
| MLP probe | PCA-64 input, 1800 s/cell CPU budget | [#779](https://eps.superkaiba.com/tasks/779) `mlp_fit_predict` |
| Per-position store cap | ≤64 positions/turn, peak layers only | plan §11 (storage bound; pilot ≥80% coverage) |
| n per cell | 5000 (Track S) / 2000 (Track M) | [#779](https://eps.superkaiba.com/tasks/779) / plan §11 |
| Models | Qwen2.5-7B (pretrained) vs Qwen2.5-7B-Instruct | plan |
| Activation dtype | bf16 (plan deviation from fp16: fp16 max 65504 overflows Qwen residual outlier dims) | run log |

**Evaluation:** The dependent variable is held-out R² (5-fold cross-validation) of a ridge map h from c_x — the mean residual-stream activation over the context tokens at a given layer — to v(x), the answer-profile vector of the target turn; R² below 0 means the map predicts worse than the global mean. Teacher-forced NLL of the realized target turn is the secondary read. Two read modes: FROZEN (layer set fixed in advance, headline at layer 19) and SELECTION (max over all 28 layers, compared only against selection-symmetric shuffle nulls that inherit the per-draw layer max — Spearman is used for the replication gate's layer-profile comparison). Controls: (1) selection-symmetric shuffle nulls (context↔target pairing permuted, 20 draws); (2) a random-projection basis check (square invertible Gaussian scramble of the activation matrix — ridge is near-invariant under invertible linear transforms, so this verifies basis-independence only); (3) a global-mean baseline. Gates: G1 replication vs the parent rig (layer-profile Spearman + absolute ΔR² at layer 19 within 0.05), G2 render/BPE integrity, G3 two-turn signal above null at n = 2000. Nonlinearity probe: an MLP on PCA-64 inputs — round 1 covered both Track S cells and both Track M chat assistant cells; the follow-up round extended it to the remaining six cells (naturalistic assistant + all user), so all 10 within-cells are MLP-probed.

**Data extraction:** Contexts from lmsys-chat-1m (established-dataset tier), pinned revision `200748d9d3cddcc9d782887541057aca0b18c5da`. Track S: single real user prompts, responses sampled from each model. Track M: real lmsys first user turn, assistant turns generated by the target model (vLLM greedy, seed 42), second user turn synthesized by claude-haiku-4-5 at T = 1.0 under 22 rotated user-persona briefs (tier-3 LLM-synthetic; distinct-3-gram rate 0.781). Kept n = 2000 of 2600 overprovisioned conversations (largest drop classes: 276 over-length, 82 short-turn). Activations: teacher-forced 28-layer capture in bf16, sharded turnstore persisted to the HF data repo. Plan deviations: (1) the second user turn was generated via synchronous pooled `messages.create` rather than the Batch API; (2) bf16 activations instead of the plan's fp16 (fp16 overflows Qwen residual outlier dimensions).

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

## Results

### The map exists in pretrained Qwen2.5-7B at 87.3% of instruct strength (0.588 vs 0.673 at layer 19)

What is plotted: held-out R² of the ridge map by layer (all 28 layers) for the two single-turn cells — instruct and pretrained — with 95% bootstrap CIs and the shuffle-null band; n = 5000 prompts per cell.

![Held-out R-squared by layer for the single-turn instruct and pretrained maps with shuffle-null band near zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/745e62f4b6cbb510d529a8778cb0137b98522cc8/figures/issue_825/s_track_layer_curves.png)

> **Figure.** *The pretrained base model carries the single-turn linear map at 87.3% of instruct strength.* Held-out R² by layer: instruct 0.673 (95% CI 0.666–0.681) vs pretrained 0.588 (95% CI 0.579–0.598) at frozen layer 19; shuffle nulls ≈ −0.02 at every layer. n = 5000 prompts per cell.

The map is not created by an explicit post-training stage: the pretrained checkpoint reaches 87.3% of instruct strength frozen-vs-frozen (selection read 0.598 at layer 27). Pretraining corpora contain chat-formatted text, so chat-data exposure is not ruled out. Non-spuriousness rests on the global-mean baseline (−0.0017) and the shuffle nulls; the random-projection arm (0.684 ≈ observed) is an invertible scramble that ridge is near-invariant to, so it verifies basis-independence only. Per-example predictions were not persisted (cell aggregates and bootstrap draws only — deliberate size scoping; regenerable from the turnstore per the footer recipe), so no predicted-vs-actual scatter is shown.

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

All four user cells read R² between −1.08 and −1.32 at layer 26 (95% CIs spanning −1.02 to −1.38; per-cell selection-symmetric nulls −2.45 to −2.96) — worse than predicting the global mean — yet every cell sits above its null band: structure exists, practical linear predictive power does not (the follow-up result below shows an MLP recovers part of it). Teacher-forced NLL agrees (user 2.04–2.64 vs assistant 0.32–0.59). The second user turn is LLM-written (claude-haiku-4-5, 22 rotated briefs), so the null is established for LLM-simulated users; format sensitivity is observational only.

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

### Gate outcomes: the parent single-turn result reproduced (rank correlation 0.963, ΔR² 0.004); render integrity PASS on regenerated evidence

No figure; the gate outcomes are stated inline.

- Replication gate (G1): the single-turn instruct map reproduces the parent run — layer-profile rank correlation 0.963; absolute R² difference at layer 19 of 0.004, within the 0.05 gate — PASS.
- Two-turn signal gate (G3): selection read 0.093 vs shuffle null ≈ −0.036 at n = 2000 — PASS.
- Render/BPE-integrity gate (G2): the original evidence file was lost at pod teardown; regenerated deterministically from the persisted conversations (free-analysis follow-up, committed as `eval_results/issue_825/render_asserts.json`). Cross-format rest-of-span BPE mismatch 3.0% (241/8000; gate ≤ 10%), tokenizer-identity and span-integrity asserts PASS — PASS.

All 10 planned within-cell maps and all 4 cross-role cells are present and analyzed; the follow-up round adds 8/8 ridge refits and 6/6 MLP cells (no budget exhaustion; anchors +0.0000); no silent drops. Binding constraints — single seed, LLM-synthetic second user turns, and the small absolute size of the recovered user map (R² ≈ 0.2) — hold the headline at MODERATE, the cross-role sign-flip at LOW (provisional), and the user-turn null at LOW-to-MODERATE, scoped to LLM-written user turns.

---

**Repro:** Compute: GCP 1× A100-80, ~6.5 h final run (7 launch attempts across crash-fix rounds; 8 GPU-h budget). Code: run pinned at [e011a0b1ca](https://github.com/superkaiba/explore-persona-space/tree/e011a0b1cabd26be6cfd227757650f5f49ca7b12); eval JSONs (34 files) committed at [c9bf728fa1](https://github.com/superkaiba/explore-persona-space/tree/c9bf728fa1/eval_results/issue_825) on branch issue-825. Artifacts: HF data repo [superkaiba1/explore-persona-space-data @ deb7a452](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/deb7a4523b5233393e4fbd2497622527b3622d35/issue825_userbase_map), prefix `issue825_userbase_map/` — `analysis_tensors/` (111 GB turnstore), `raw_completions/` (incl. `conversations.jsonl` + `track_s.jsonl`), `eval_results_mirror/` (34 JSONs). WandB: n/a (analysis-only run). Plan: plan v4 (parent) + plan v6 (follow-up amendment) under `plans/` in the task folder (resolve via `uv run python scripts/task.py find 825`). Follow-up artifacts: git `c4dcf8c907` (`eval_results/issue_825/mlp-unprobed-cells/`), HF [eval_results_mlp_unprobed](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/72ddfc466a53287a6f421707f14fd4efa44c38f7/issue825_userbase_map/eval_results_mlp_unprobed), wrapper @`07589e890b`.

**Context:** Created 2026-07-01 from user chat. Verbatim originating prompt:

> Help me to plan this experiment: Is the context vector to answer profile mapping present in the base model (Qwen/Qwen2.5-7B) and does it hold for the user?

(full text in `original-body.md` `## Provenance`). Lineage: [#779](https://eps.superkaiba.com/tasks/779) — the parent single-turn instruct rig this run replicates and extends. Run 2026-07-02 (7 launch attempts; crash-fix rounds 1–6: extract host-RAM OOM → block-wise flush → host upsize → GPU Gram-ridge fit). Interpretation reviewed rounds 1–3 (two critic REVISE rounds, then PASS+PASS). Follow-up round `mlp-unprobed-cells` run 2026-07-03 (cheap-band auto-run, redundancy-screened; plan v6; 3 launches, results rescued from the crash-persist upload).
