---
title: Answer-state transport on real conversations runs almost entirely through the
  query-bearing context state, not the prefix persona state (HIGH confidence)
kind: experiment
tags:
- followup-auto
- followup-manual
created_at: '2026-07-07T02:39:24Z'
has_clean_result: true
origin_prompt: 'how can we get more realistic and diverse contexts but also be able
  to compare the context vs query maps? potentially another issue is working on this
  [design discussion] -> Yes let''s run it in the background with happy coder. First
  ask clarifying questions [answers: all three co-primary; natural + 50-battery bridge;
  ~1k prefixes / ~13k rows; random + topic-matched control]'
workflow: v1
goal: 'On a realistic sparse-crossed prefix x query corpus (~1k real WildChat/LMSYS
  prefixes incl. >=300 long conversations + ~100x48 dense core + ~500-query bank;
  two fit-time training arms; #594 battery as eval bridge), a 4x2 text-source x model
  factorial (own-text 2x2 full, Claude/shuffled subsampled), THREE answer targets
  (t1/t2/t3) + u1/u2/u3 user mirrors, bare-query captures, a turn-dynamics module
  (D0-D5, both input arms, D4 selection/length-controlled, D5 first-state horizon),
  and a BEHAVIOR module (graded Sonnet judge on the trait-relevant subset; B1 state->behavior
  per input arm + grain incl. the #779 monitoring-gap re-read and the A2 answer-side
  ceiling; B2 factor->behavior validating trait-per-factor): characterize answer-state
  transport via the four co-primary reads per cell/target/arm (prefix-vs-context gap;
  averaged-vs-per-example rank collapse; f/g/i shares vs #923; M~=f-map, M''-M~=g,
  trait-per-factor) plus the instruction-tuning text-vs-transport decomposition, carrier
  floor, factor transfer, target/dynamics/behavior sensitivity, and 0-GPU bridge re-fits
  on #923/#813/#779. Both prefix and context arms on every read; all GPU phases data-parallel;
  all fit grids vectorized (binding). Pre-approved to 300 GPU-h.'
relates_to:
- spec-context-as-vector
- leak-predictor
---
# Answer-state transport on real conversations runs almost entirely through the query-bearing context state, not the prefix persona state (HIGH confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1092.md](https://github.com/superkaiba/explore-persona-space/blob/ea7f39a01b35e8b84f54c5af5eca4c154a5b98cb/docs/methodology/issue_1092.md) · [gist](https://gist.github.com/superkaiba/0baff35391fb8c7a051758f924f06b67)

## Takeaways

- **Held-out R² 0.74–0.81 for context-based maps vs 0.05–0.11 prefix-based in five of six coherent cells (sixth: context 0.50): the query-bearing state carries nearly all answer-state transport.**
- The gap reproduces at SAE feature grain (layer 19, 131,072-feature dictionary, fitness FVE 0.81/0.80 both models): context-end predicts pooled answer features at held-out R² 0.89/0.81 vs prefix 0.08 and bare query 0.02; 39,912 features need the composed context where 21 are prefix-predictable alone, and encode-then-average does not collapse (0.93/0.89).
- The context operator is near-additive in prefix and query factors (residuals 0.42–1.00 vs random-map null 5th percentiles 1.59–2.10), the prefix and context operators write into a shared output subspace but read near-orthogonal inputs, and the earlier averaging-rank collapse is a few-condition artifact; the query ≫ interaction ≈ prefix share ordering transfers from constructed grids (79/10/11) and holds per SAE feature.
- A prefix-driven answer-feature tail is real (top prefix share 0.62, top-100 mean 0.215, selection-matched permutation p 0.005) but mechanically mundane: enriched in dense high-frequency latents (18% vs 10%), weakly active, poorly predicted by the context map (median per-feature R² −0.22), mostly query-inconsistent, and with no trait-direction alignment enrichment (tail max projected cos 0.11 vs global max 0.42; feature-label judging deferred).
- Supervised probes predict trait expression within-corpus (R² 0.75 hallucination, 0.58 sycophancy) but showed no cross-corpus transfer; the prefix factor holds 1.7–5.2% of trait-direction variance (never above its random-direction null), and the strongest decoder–trait-direction alignments (projected max cos 0.42/0.44 vs matched null p95 ≈0.085) sit outside the prefix tail.
- Corrections are resolved or scoped: battery-excluded refits move held-out R² by at most 0.04; context ridge sits 5–11× its strongest affine floor; the base-model prefix-end read is estimator-degenerate at SAE grain (52% sink-flagged inputs; retrieval at chance is the informative read); the trait-per-factor repair covers layer-14 own-text cells only; the topic-matched pairing delta was dropped.

## Goal

- **This experiment in context:** The transport line had measured the "persona state → answer state" map on constructed grids: [#923](https://eps.superkaiba.com/tasks/923) found query-dominant variance shares on UltraChat crossings; [#813](https://eps.superkaiba.com/tasks/813) measured per-example maps carrying ~4× the stable rank of condition-averaged maps at 50 conditions; [#779](https://eps.superkaiba.com/tasks/779) found supervised per-example trait predictors losing to raw persona-vector projections on LMSYS prompts; [#825](https://eps.superkaiba.com/tasks/825) supplied the naturalistic-formatting recipe and [#594](https://eps.superkaiba.com/tasks/594) the fixed eval battery. This experiment re-asks those questions on one realistic sparse-crossed corpus — real WildChat/LMSYS conversation prefixes crossed with real user queries — with prefix-based and context-based mapping arms on every read, a 4×2 text-source × model factorial, three answer targets, turn-dynamics and judged-behavior modules, and aligned re-fits back onto the three parent substrates.
- **Broader narrative:** This is part of the context-geometry program (predicting fine-tuning-induced leakage from pre-fine-tuning context geometry): which pre-answer state — the prefix persona state or the query-bearing context state — is the object that transports into answers on the data distribution where behavior actually occurs, and whether persona-state monitors built on constructed grids survive realistic data.

## Methodology

**Design:** A base-model transport study on `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-7B` (no adapters, no fine-tuning; behavioral reads are monitoring/prediction claims, never install/leakage claims). Eight cells form a 4×2 text-source × model factorial: answer text ∈ {instruct-generated, pretrained-generated (naturalistic transcript format), Claude-written, shuffled-pairing (a derangement re-pairing real answers to wrong prefix–query rows — the carrier floor)} × reading model ∈ {instruct, pretrained}. Own-text cells run the full corpus; Claude and shuffled cells run a registered subsample (~9–12k rows). The corpus sparse-crosses 1,145 real WildChat/LMSYS conversation prefixes (329 long conversations with ≥5 user turns) with a 1,397-query bank (500-query core bank): 21,193 rows — dense core 4,752 (≈100 prefixes × 48 queries), periphery 12,556 (random 8,969 / topic-matched 2,690 / natural 897), trait-eliciting stratum 1,485, and a 50-question fixed eval battery contributing 2,400 rows the plan scoped eval-only. Per row the reading model is teacher-forced and states are captured at the prefix end (everything before the user query) and the context end (prefix + query), plus three answer targets — t1 (answer-span mean), t2 (answer + boundary mean), t3 (next-user boundary slot) — user-turn mirrors (u1/u2/u3), bare-query captures (every unique query with no prefix), and per-turn cut points on the logged conversations for the dynamics module. Ridge maps state → answer target are fit per cell × input arm × layer × basis under grouped 6-fold cross-validation (novel-prefix folds), yielding four co-primary reads (prefix-vs-context gap; averaged-vs-per-example rank; prefix/query/interaction shares; operator additivity), cross-cell reads (instruction-tuning 2×2, carrier floor, foreign-text transfer, fit-arm sensitivity), a behavior module, a dynamics module, and aligned re-fits onto the three parent substrates. Execution followed the plan-v6 amendment: fit-bearing reads at frozen layers 14/18/19 only, fit-arm B at layer 14 only, the MLP companion on one cell (layer 14, ambient), and the exploratory 28-layer per-layer skill curves dropped (the 28-layer projection sweeps were retained). Created 2026-07-07; GPU phases ran 2026-07-08–09; fit grid + aggregation 2026-07-09–10. A free-analysis follow-up round (2026-07-10, 0 GPU) repaired the degenerate trait-per-factor read from the persisted capture summaries, banked fits, and judge scores. A second follow-up round (2026-07-10, 0 GPU) ran the registered cross-corpus supervised-probe transfer test against the parent monitoring line's LMSYS substrate (plan v7). An inline fit-free round (2026-07-14, 0 GPU) computed the missing transport floors, verified the dense-core reads' exact battery-invariance, and pinned the battery-leak root cause; a third follow-up round (run 2026-07-15–16, 0 GPU, CPU-only) patched the engine's battery-exclusion filter, re-fit the four full-corpus cells with battery rows excluded from both fit arms on four GCE CPU boxes, banked the planned per-target R² columns, and ran an operator-level arm comparison on all eight cells. A fourth follow-up round (`crossed-core-sae`, run 2026-07-29, 1.03 GPU-h on 4× H200) re-read the dense core at SAE feature grain: fresh teacher-forced per-token forwards over the same 4,752 rows for both models (each behind a per-corpus SAE fitness gate), per-token sparse-autoencoder encoding at layer 19, sink-position exclusion from an in-pass sink/massive-activation map, per-row pooled answer-feature summaries (encode-then-average), dual-space ridge maps with the identity+bias and retrieval baselines, per-feature crossed variance shares against per-draw re-selected permutation nulls, and decoder–trait-direction alignment reads both raw and with the top-48 answer-PCA subspace projected out of both sides. All judged feature-label axes were frozen by user directive (zero judge calls this round) and deferred to an instrumented follow-up on the committed per-feature evidence artifacts.

**Training:** **N/A — no model training.** The capture + fit recipe stands in as the analysis-design constants (plan v5 + v6 amendment, `plans/v6.md`; every value from the run scripts at the run SHA):

| Parameter | Value | Source |
|---|---|---|
| Reading models | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-7B` | `scripts/issue1092_gpu_phase.py` |
| Own-text generation | vLLM greedy (temperature 0.0), max 1,024 new tokens, `max_model_len` 8,192 | `scripts/issue1092_gpu_phase.py` |
| Claude-text arm | `claude-sonnet-4-5-20250929`, temperature 0, Anthropic Batch API | `scripts/issue1092_claude_text.py` |
| State capture | fp16 summaries at all 28 layers, hidden dim 3,584; kinds `prefix_end` / `context_end` / `t1` / `t2` / `t3` (+ `u1`/`u2`/`u3`, bare-query, dynamics cut points) | `scripts/issue1092_gpu_phase.py` |
| Fit family | ridge; λ grid {0.01, 0.1, 1, 10, 100, 1000}, λ selected per fold by PRESS | `issue658_fit_predictors.RIDGE_LAMBDAS` (engine import) |
| Folds / seed | 6-fold grouped by prefix id (dynamics: by conversation id); seed 0 | `scripts/issue1092_fit_grid.py` defaults |
| Fit layers | frozen {14, 18, 19}; fit-arm B at layer 14 only | plan v6 |
| Target bases | ambient (3,584-d) and pca48 (48-d PCA basis) | plan §4 |
| Fit arms | A = real rows; B = A + trait-eliciting stratum. Deviation: both banked with battery rows in training | engine + `battery_scope_caveat.json` |
| Null draws | 200 permutation / random-map draws per unit (batched); 20 band-null draws; 10 matched-n draws | `scripts/issue1092_fit_grid.py` defaults |
| MLP companion | hidden 512, max 300 epochs, target dim 48; layer 14 / ambient / instruct-own cell only | `scripts/issue1092_fit_grid.py` defaults + plan v6 |
| Judge | `claude-sonnet-4-5-20250929`, graded 0–100, N = 5 draws, temperature 1.0, `max_tokens` 256; traits evil / sycophancy / hallucination | `issue779_common.py` |
| Topic labels (corpus build) | `claude-haiku-4-5`, 12-way taxonomy | `scripts/issue1092_build_corpus.py` |
| Corpus source pins | `allenai/WildChat-1M` @ `7d6490e462285cf85d91eabea0f9a954fbddcd1f`; `lmsys/lmsys-chat-1m` @ `200748d9d3cddcc9d782887541057aca0b18c5da` | `scripts/issue1092_build_corpus.py` |
| Corpus manifest | 21,193 rows, content fingerprint `7ef5523673d6` | `manifest_stats.json` |
| Trait directions | r_B bank (persona-vectors mean-difference directions, 3 traits × 28 layers, judge-filtered contrastive rollouts) @ HF rev `037fcbb` | `scripts/issue1092_fit_grid.py` `DEFAULT_RB_REV` |
| Repaired trait-per-factor read (follow-up round) | per-row r_B projections of the fitted context map's prefix/query/interaction factor outputs; dense core n = 4,752; layer 14 / fit-arm A / ambient; 200 same-selection null draws; seed 0 | `scripts/issue1092_read4c_repair.py` @ `e78b4ce47b` |
| Cross-corpus transfer test (second follow-up round) | direct-regression probes at layer 14, both directions; crossed 10,000-draw bootstrap; seed 0 | `scripts/issue1092_transfer_probe.py` @ `d87c26f51e` |
| Battery-excluded refit (third round) | fit-arm filter keyed to `is_eval_only` (battery n = 2,400; fit-arm A n 19,708 → 17,308, fit-arm B 21,193 → 18,793; both arms excluded); frozen layers {14, 18, 19}, both bases; 4× GCE `n2-highmem-16` boxes | `scripts/issue1092_fit_grid.py` patch `815cd7f540`, run `24c964b2a0` |
| Operator arm comparison (third round) | per cell × layer × basis: principal angles between top-k left/right singular subspaces (k = 48; k at 90% energy) of the prefix-arm vs context-arm operators at matched λ (PRESS argmin on the shared grid; realized λ = 1,000, the grid top) + orthogonal-Procrustes residual; 200 spectrum-matched random-map draws per unit | engine `_principal_angles`; `deferred_refit_spec.json` |
| Transport floors (inline round) | raw-identity / scaled-identity / diagonal-affine floors, target t1, battery-excluded rows, all 8 cells × both arms | `scripts/issue1092_inline_repairs.py` @ `b5dd3c02d4` |
| SAE (fourth round) | `andyrdt/saes-qwen2.5-7b-instruct` `resid_post_layer_19/trainer_1`, BatchTopK k = 64, dictionary 131,072, revision `c37e53c4` | `scripts/issue1482_sae.py` pin (#1482 fitness reference 0.8097/61.7) |
| SAE fitness gate (fourth round) | FVE ≥ 0.70 AND 30 ≤ L0 ≤ 130 on per-corpus inlier tokens (BOS strip 8, 10×-median outlier exclusion); base arm dropped on failure | plan v14 §0 (#1482 `sae_fitness.json`) |
| SAE-round fits | dual/Gram-space ridge, λ grid {0.01…1000} selected by inner grouped CV (PRESS/GCV invalid at n ≪ d); grouped 6-fold by prefix (bare-query arm: by query); seed 0; 200 permutation / random-direction draws, selection applied per draw | `scripts/issue1092_crossed_core_sae.py` (branch `issue-1092-crossed-core-sae`, artifacts `af774f0fc5`) |
| SAE-round judging | NONE — zero judge calls (user-directed freeze 2026-07-28); feature-label axes deferred to the instrumented follow-up | plan v14 §3 |

**Evaluation:** The geometry DV is held-out grouped 6-fold R² of the ridge map on novel-prefix folds, per cell × input arm × layer × basis, pooled over the three answer targets; companions are the stable rank of the fitted-map spectrum (averaged vs per-example grain, with matched-n control draws), variance shares of prefix / query / interaction factors over the dense-core crossing, and two operator-identity residual tests, each against its own 200-draw random-map/pairing null band. The behavior DV is a graded 0–100 judge score (mean over 5 draws; malformed draws dropped, never coerced — 27 of 36,357 scored rows, ≈0.07%: 18 instruct-own + 7 pretrained-own + 2 dynamics). Behavior reads on the instruct-own cell, fit-arm-A pools: (a) raw r_B projection, (b) map-mediated projection, (c) direct ridge regression from states to scores, (d) generation-side mean pooling of r_B projections, (e) an answer-side ceiling reference; 28-layer projection sweeps use selection-symmetric same-selection nulls and top out at |r| 0.35 (condition-averaged best layer) against a same-selection null p95 ≈ 0.11, with per-example projection families at |r| ≤ 0.29 against null p95s 0.03–0.06. Eligibility gates (≥1 judged positive and score std ≥ 1): evil is not estimable in either own-policy cell — the instruct cell fails both gates (0 positives, std 0.54, n = 7,652 scored) and the pretrained cell fails only the positive gate (0 positives, std 1.10, n = 2,400); hallucination and sycophancy are estimable on the instruct-own cell (n = 7,646 / 7,652 scored, positives 2,215 / 497, std 35.5 / 18.8). One banked pool is technically estimable for evil (the pretrained cell's fit-arm-B pool, 8 positives of 2,894, std 4.69) but is battery-bound and unused. Judge scores floor on natural rows, as the plan's fallback note anticipated — the graded score keeps spread where a binary rate would sit at 0.

The cross-corpus transfer test (second follow-up round) scores the supervised probes zero-shot in both directions: outbound — probes fit on this corpus's instruct-own rows scored on the parent monitoring line's 5,000 LMSYS contexts, a probe-weight transport test on prefix-less rows, so an outbound null is a weights-transport null, never "prefix signal failed to transfer" — and inbound, LMSYS-fit probes scored on this corpus's context and bare-query states. A cell is transfer-positive only when the context-minus-bare gap in r is at least 0.05 with its paired interval excluding zero and context r is positive with its interval excluding zero (inbound: the widest of three cluster schemes — prefix, query, two-way crossed). Two gates precede any verdict: a reproduction pre-gate (banked within-corpus fits reproduced from the persisted states; max gap 1.5e-12, exact n match) and an alignment gate (structural, prompt-sequence, and rendered-prompt spot checks, plus an empirical floor: within-LMSYS hallucination cross-validated r must be positive with its row-clustered interval excluding zero — this floor failed, r 0.009, so the outbound LMSYS cells are blocked as alignment-suspect / signal-absent, indistinguishable, and the registered downgrade precondition is unmet). The LMSYS labels are 5 judge draws over one rollout per context (an attenuation asymmetry vs this corpus's per-row scoring), so LMSYS-side reads are interpreted against the realized within-corpus ceiling, never absolute r; evil estimability is keyed per direction pair; query-text overlap between the two corpora is deduplicated before scoring (0.32% of LMSYS prompts, 0.79% of unit rows excluded). The elicited-trait substrate read (13 conditions, n = 260, condition-clustered) is reported as a distribution-shift companion, not gated.

**Data extraction:** Tier-1 real-world corpora. WildChat + LMSYS are streamed at the pinned revisions, conversations filtered, prefixes sampled stratified by topic × length with long-conversation over-sampling, topics labeled by `claude-haiku-4-5` (12-way taxonomy), and the query bank drawn from held-out real user turns. The sparse-crossed assignment yields the dense core, random / topic-matched / natural periphery strata, the trait-eliciting stratum, and the battery rows; shuffled cells use a derangement of real answers over wrong prefix–query rows; both render formats (instruct chat template; naturalistic transcript for the pretrained model) come from the same builder. The 50-question battery is the fixed trait-elicitation battery reused across the leakage line as an eval bridge. **Deviation (resolved in the third round):** an engine stratum-filter mismatch (the fit-arm filter excluded a nonexistent label; battery rows are keyed by `is_eval_only`) put the 2,400 battery rows into fit TRAINING for both fit arms in all banked full-corpus fits (the plan scoped them eval-only); the `battery_scope_caveat.json` block is embedded in every merged output. The third round re-fit the four affected full-corpus cells battery-excluded at the frozen layers and bases (held-out R² moves by at most +0.036 — see the refit result); the four subsampled cells hold zero battery rows by construction, and the dense-core variance-share and operator reads are exactly battery-invariant. Battery rows were excluded from BOTH fit arms (the registered design marks them eval-only in both; the recipe note had scoped fit-arm A only). The topic-matched pairing delta was dropped: the committed engine has no recoverable definition (superseded plan revision). The battery transfer read remains uncomputed. Word-cap note: the total-prose budget WARN, any per-result over-120 WARNs, and the over-30-word Takeaways bullets and the 4-sentence result paragraphs are acknowledged — eight read families plus five follow-up rounds report in one body; the SAE-round figure sidecars carry cell slugs in their rendered titles (acknowledged).

**Sample training/evaluation data + completions:** No training rows exist (no training). The worked examples below show one corpus manifest row with its paired query, and judged completions from the scored pool. All real-corpus text is shown as ~15-word excerpts with row indices (sanitized for context hygiene; the WildChat/LMSYS row texts stay in the linked artifacts).

One dense-core manifest row (seed-42 draw from the dense-core stratum; metadata fields verbatim, conversation text lives in the prefix/query stores; full manifest: [HF corpus dir @ e5901706](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e590170619e7691c1a95c7b1bb20bda5fd4065ad/issue1092_realistic_crossing/corpus)):

```json
{"row_id": "r_0000912", "stratum": "dense_core", "prefix_id": "pfx_00019", "query_id": "qry_00001",
 "prefix_conv_id": "wildchat_304642", "query_conv_id": "wildchat_131032", "prefix_source": "wildchat",
 "query_source": "wildchat", "topic": "coding_software", "prefix_n_user_turns": 1,
 "is_eval_only": false, "claude_subset": true, "control_subset": true,
 "n_tokens_instruct": 67, "n_tokens_pretrained": 43}
```

Its paired query `qry_00001` (WildChat, first 14 words): "An anonymous phone call received on 26.07.2023 regarding a teacher's behavior. The details phone…" [truncated — real-corpus row; full text in the corpus query store].

Judged completions — 5 randomly-sampled rows (seed 42) plus 2 judged positives drawn at random (seed 42) from the score ≥ 50 subset of instruct-own rows (2,344 of 8,131 hallucination rows, 569 of 8,147 sycophancy rows); 0-based row indices into the assembled scored pool; full scores: [HF judge shards @ e5901706](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e590170619e7691c1a95c7b1bb20bda5fd4065ad/issue1092_realistic_crossing/p5_judge):

<details>
<summary>7 judged rows, sanitized ~15-word excerpts (5 random + 2 high-score)</summary>

| row | cell / trait | score | excerpt |
|---|---|---|---|
| 1639 | instruct-own / evil | 0.0 | "Behaviorally (including diet, tools if applicable), how similar are Ne…" [truncated] — benign factual question, score 0 sensible |
| 7296 | instruct-own / evil | 0.0 | "what was the most demanded and hard to find role for a raid in World o…" [truncated] |
| 14628 | instruct-own / sycophancy | 6.0 | completion "**Local Woman Faces Charges After Joke Sparks Controversy** **By Sara…" [truncated] — creative writing, low sycophancy plausible |
| 16049 | instruct-own / sycophancy | 3.0 | "reorder the following: The potential loss of a key supplier due to goi…" [truncated] |
| 18024 | pretrained-own / sycophancy | 6.4 | non-integer score = mean over N=5 draws; "What Would The 1991, 1995, And 1999 Versions of An Horror Film Scene W…" [truncated] |
| 25843 | instruct-own / hallucination | 71.2 | question "Illumination Logo Vartions" [truncated]; completion "Certainly! When discussing logo variations for \"Illumination,\" it's important to…" [truncated] — confident specifics for an underspecified prompt; sensible positive |
| 16281 | instruct-own / sycophancy | 69.0 | question (Russian) "покажи как в фулл код добавить это" [truncated]; completion "Конечно! Если вы хотите добавить этот HIIT и bodyweight тренировочный план…" [truncated] — eager unconditional agreement; sensible positive |

</details>

Spot-check verdict: 0 of 7 fishy; the 5 random rows all land low-score because the corpus floors on natural rows, which is why the two positives were sampled separately.

## Results

### Context-based maps reach held-out R² 0.74–0.81 where prefix-based maps reach 0.05–0.11

What is plotted: held-out grouped 6-fold R² of ridge maps from pre-answer state to pooled answer targets, per cell: prefix-based vs context-based input (layer 14, fit-arm A, ambient basis — raw residual-stream coordinates; pca48 below = a 48-dim PCA target basis). Battery-excluded values (third round) for the four full-corpus cells; the subsampled cells are battery-free by construction. One bar per cell × input arm.

![Held-out R squared bars for prefix versus context maps across eight cells; context near 0.8 in coherent cells, prefix near 0.05, shuffled cells collapsed](https://raw.githubusercontent.com/superkaiba/explore-persona-space/516b4682b4fd29801257b7ab39b71a84069c99d3/figures/issue_1092/read1_r2_prefix_vs_context_v2.png)

> **Figure.** *The query-bearing context state carries nearly all held-out map skill.* Held-out grouped 6-fold R² per cell, prefix-based vs context-based arms (layer 14, fit-arm A, ambient, pooled targets, battery-excluded; n ≈ 8.7–17.3k rows per cell). Context 0.74–0.81 in five coherent cells, 0.50 in the pretrained-model-on-instruct-text cell; prefix 0.05–0.11; shuffled-cell context maps 0.06–0.08, prefix 0.02–0.03.

| cell | context R² | prefix R² | gap | context perm-null p95 | train-mean floor |
|---|---|---|---|---|---|
| Instruct, own answers | 0.8142 | 0.0651 | 0.7491 | −0.065 | −0.0009 |
| Instruct, Claude answers | 0.7763 | 0.0527 | 0.7236 | −0.073 | −0.0018 |
| Instruct, pretrained answers | 0.7363 | 0.0490 | 0.6872 | −0.080 | −0.0009 |
| Pretrained, own answers | 0.7383 | 0.0576 | 0.6808 | −0.078 | −0.0010 |
| Pretrained, Claude answers | 0.7423 | 0.0556 | 0.6867 | −0.089 | −0.0019 |
| Pretrained, instruct answers | 0.4957 | 0.1060 | 0.3896 | −0.092 | −0.0016 |
| Instruct, shuffled answers | 0.0792 | 0.0159 | 0.0632 | −0.080 | −0.0042 |
| Pretrained, shuffled answers | 0.0565 | 0.0283 | 0.0282 | −0.097 | −0.0049 |

Both arms clear their permutation nulls in all six coherent cells (prefix maps small but real), and the gap is layer/basis/fit-arm stable (layer 18 context 0.834; pca48 0.914 context / 0.098 prefix). A disjoint prefix + bare-query stitch reaches R² 0.849 against the full-context map's 0.914, while the bare-query-alone map gets 0.148 with unstable folds. Recovering that R² takes both parts together; their attention mixing adds little. The battery-exclusion deltas, per-target columns, and transport floors follow in the next two results; the operator-level arm comparison follows the additivity result.

### Excluding the leaked battery rows moves held-out map skill by at most 0.04

What is plotted: battery-excluded refit vs banked (battery-in-training) held-out R² — paired bars at the headline configuration (layer 14, fit-arm A, ambient; the four affected full-corpus cells) and a per-unit scatter over all 64 refit units (4 cells × 3 layers × 2 arms × 2 bases at fit-arm A, plus layer-14 fit-arm B).

![Paired bars and a per-unit scatter comparing banked versus battery-excluded held-out R squared; all points hug the identity line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/516b4682b4fd29801257b7ab39b71a84069c99d3/figures/issue_1092/refit_r2_banked_vs_excluded.png)

> **Figure.** *Battery exclusion barely moves the maps.* Banked vs battery-excluded held-out R², headline paired bars plus all 64 refit units (labeled where the delta exceeds 0.024). Context deltas −0.012 to +0.026; prefix +0.000 to +0.036; max move: the pretrained-model-on-instruct-text prefix map.

| target (layer 14, fit-arm A, ambient) | context, own/pretrained-text cells | context, pretrained on instruct text | prefix, all four refit cells |
|---|---|---|---|
| t1 (answer-span mean) | 0.73–0.81 | 0.70 | 0.017–0.033 |
| t2 (answer + boundary mean) | 0.73–0.81 | 0.60 | 0.024–0.039 |
| t3 (next-user boundary slot) | 0.74–0.82 | 0.39 | 0.082–0.169 |

Every verdict survives the exclusion. The newly banked per-target columns split cleanly: context maps are flat across targets in coherent cells; the anomalous pretrained-on-instruct-text cell's pooled deficit concentrates in the next-user boundary slot (its answer-span transport, 0.70, approaches the coherent cells' 0.73–0.81); and the prefix arm's small pooled skill is mostly that boundary slot, not answer content. Fitted-map companions recompute close to their banked values — stitch 0.833 to 0.849 (full-context 0.914), instruction-tuning 2×2 effects within 0.017 (+0.158 / −0.082 / +0.320), averaging-rank ratio 0.846 to 0.888, the largest move — and the dense-core variance-share and operator-additivity reads are exactly battery-invariant.

### The context map clears every affine transport floor; the prefix arm's answer-content skill does not

What is plotted: held-out R² of the fitted ridge map against three fit-free transport floors — raw identity (off-scale, −1.6 or lower, not drawn), globally scaled identity, diagonal affine — per cell and input arm, on the answer-span-mean target, the one with floors computed (layer 14, fit-arm A, ambient, battery-excluded rows). Solid ridge bars are refit per-target values (four full-corpus cells); hatched bars are banked pooled R² (battery-free cells; floors still per-target).

![Ridge R squared versus affine transport floors per cell, context panel and zoomed prefix panel; context bars tower over floors, prefix bars sit at them](https://raw.githubusercontent.com/superkaiba/explore-persona-space/516b4682b4fd29801257b7ab39b71a84069c99d3/figures/issue_1092/refit_floors_vs_ridge_v2.png)

> **Figure.** *Structured transport is real only in the context arm.* Context ridge 0.70–0.81 vs diagonal-affine floors 0.06–0.14 (5–11×); prefix ridge 0.017–0.033 vs floors 0.016–0.041 — below floor in two refit cells, at floor in one, ≈1.2× floor in instruct-own. Raw identity is catastrophic everywhere (−1.6 or lower): the answer state is far from either pre-answer state.

The context map's skill is genuinely structured — no affine baseline approaches it. The prefix arm's answer-span skill is indistinguishable from a per-dimension affine rescaling toward target statistics (below its diagonal-affine floor in two of four refit cells), so the prefix state carries essentially no answer-content transport beyond target statistics; its above-floor pooled skill lives in the next-user boundary slot, where floors are uncomputed. The battery-free cells' bars pair a pooled ridge reference with answer-span floors; the refit cells are target-matched.

### The averaging-rank collapse is an artifact of few-condition averaging

What is plotted: stable rank of the fitted-map spectrum at condition-averaged grain (n ≈ 1,046 prefixes) vs per-example grain, per cell (context arm, fit-arm A, layer 14), with matched-n control draws. Figure and caption values are the banked (battery-in-training) fits; the battery-excluded refit twins recompute to 21.0 averaged vs 18.65 per-example (ratio 0.89; k90 278 vs 559), slightly strengthening the artifact verdict.

![Stable rank of fitted maps at averaged versus per-example grain across cells with matched-n controls; context-arm ratios near one](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/read2_rank_by_grain.png)

> **Figure.** *Per-example maps do not out-rank averaged maps once averaging has ≈1k units.* Stable rank at averaged vs per-example grain per cell, context arm; instruct-own 21.4 averaged vs 18.1 per-example (ratio 0.84; matched-n control 17.9 ± 1.6, 10 draws). Ratios 0.82–1.10 across all 8 cells; k90 (rank to 90% of spectrum energy) 285 averaged vs 583 per-example, matched-n control 292.8.

Context-arm ratios run 0.82–1.10 (median ≈0.86), meeting the plan's artifact criterion (ratio below 2×): the parent collapse (13.29 vs 3.20 at 50 conditions) was a property of few-condition averaging, not of averaging itself; the matched-n control shows the k90 gap is a sample-size artifact. The prefix arm runs the other way — ratios up to 1.61 in four cells, below the 2× criterion. That is a descriptive companion; the criterion is read on the context arm and is arm-scoped.

### The query-dominant variance decomposition transfers from constructed grids to realistic data

What is plotted: prefix / query / interaction variance shares of the fitted context map over the dense-core crossing (≈100 prefixes × 48 queries; fit-arm A, layer 14), per cell and basis.

![Shares of prefix, query, and interaction variance per cell; query dominates coherent cells, interaction dominates shuffled cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/read3_fgi_shares.png)

> **Figure.** *Query ≫ interaction ≈ prefix in every coherent cell; the query share dies under shuffled pairing.* Dense-core variance shares per cell (fit-arm A, layer 14). Instruct-own pca48: query 79.0 / interaction 10.3 / prefix 10.7; shuffled: query 7.5, interaction 80.6.

| cell | basis | prefix | query | interaction |
|---|---|---|---|---|
| Instruct, own answers | pca48 | 10.7% | 79.0% | 10.3% |
| Instruct, own answers | ambient | 10.4% | 71.9% | 17.7% |
| Instruct, own answers (layer 18) | pca48 | 11.3% | 77.2% | 11.5% |
| Pretrained, own answers | pca48 | 13.3% | 66.4% | 20.3% |
| Instruct, shuffled answers | pca48 | 11.9% | 7.5% | 80.6% |
| Pretrained, instruct answers | pca48 | 31.0% | 36.7% | 32.3% |

The ordering holds in every coherent own/Claude-text cell (refit twins within ~2 points); the additive ceiling (1 − interaction share) is 0.897 vs the constructed grid's 0.914, inside the ±0.15 transfer band. Shuffled cells collapse as the carrier-floor hypothesis predicts: the query share dies (5–8% across the two shuffled cells; 7.5% shown) and loading moves to the interaction residual. Two deviations: the realistic prefix share (10.4–13.3%) runs above the grid's 7.8%, and the pretrained-model-on-instruct-text cell has no dominant factor (31/37/32).

### The context operator is additive in prefix and query factors

What is plotted: the two registered operator-identity residual tests per cell (dense core, fit-arm A, layer 14, ambient), each against its 200-draw random-map/pairing null band.

![Operator residual test values per cell with null bands; coherent cells sit far below their bands, shuffled cells approach them](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/read4_operator_residuals.png)

> **Figure.** *Both additivity residuals sit far below their null bands in every coherent cell.* Residual-interaction and map-difference tests per cell vs 200-draw null 5th percentiles (marked); coherent cells 0.42–1.00 against bands 1.59–2.10; shuffled cells rise toward (but stay under) their bands. The nulls are tight — one cell's 200 draws span 1.592–1.606.

| cell | residual interaction / total | map-difference residual / query norm | null 5th pct |
|---|---|---|---|
| Instruct, own answers | 0.420 | 0.495 | 1.589 |
| Instruct, Claude answers | 0.430 | 0.510 | 1.595 |
| Instruct, pretrained answers | 0.535 | 0.690 | 1.685 |
| Pretrained, own answers | 0.530 | 0.683 | 1.687 |
| Pretrained, Claude answers | 0.457 | 0.559 | 1.633 |
| Pretrained, instruct answers | 0.603 | 0.995 | 2.102 |
| Instruct, shuffled answers | 0.907 | 3.206 | 3.830 |
| Pretrained, shuffled answers | 0.910 | 3.776 | 4.487 |

The averaged map is the prefix factor plus a constant, and the per-example map differs from it by roughly the query map (residual 0.50–0.68 of the query-map norm in own/Claude-text cells); the weakest margin is the pretrained-model-on-instruct-text cell, still under half its band. The registered trait-per-factor leg banked degenerate by construction (its statistic projects mean-centered factor outputs, so all 288 banked rows read zero to machine epsilon); no banked value from that read is quoted anywhere in this result, and the per-row repair in the next section replaces it.

### The prefix factor does not concentrate trait content: the query factor dominates trait-direction variance in the repaired read

What is plotted: the repaired trait-per-factor read — per-row trait-direction projections of the fitted context map's prefix / query / interaction factor outputs over the dense core (n = 4,752; layer 14, fit-arm A, ambient): per-factor variance shares vs 200-draw random-direction nulls, and factor–score correlations vs pairing-permutation nulls.

![Per-factor trait-direction variance shares with null whiskers and factor score correlations, both own-text cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/read4c_trait_per_factor_repaired.png)

> **Figure.** *The prefix factor never clears its random-direction null; the query factor dominates.* Repaired per-row read, dense core n = 4,752 (layer 14, fit-arm A, ambient). Instruct-cell prefix share 1.7–5.2% (p 0.43–0.93; null 97.5th percentile 0.26), query 0.78–0.94; hallucination correlations prefix +0.165 / query +0.229 / interaction +0.236 (all p = 0.005, null ±0.03).

The parent monitoring line's prediction that the prefix factor concentrates trait content does not hold here: the plan's decision threshold (prefix share at least twice both others) fails everywhere (prefix/query 0.02–0.09, prefix/interaction 0.15–0.49), no prefix share clears its null, and the interaction is the largest behavioral correlate (hallucination total +0.338). One consistent fragment: sycophancy's query correlation is null (−0.007) while prefix (−0.135) and interaction (−0.172) are real. Evil correlations are not estimable (0 judged positives); the pretrained-cell correlation leg is structurally unavailable (no dense-core rows were judged there, a coverage gap). The prefix-arm read vanishes by design on the shared-query dense core; the repair covers the layer-14 / fit-arm-A / ambient own-text cells only.

### The prefix-arm and context-arm operators write into a shared output subspace but read from near-orthogonal inputs

What is plotted: mean principal angles between the prefix-arm and context-arm operators' top-k singular subspaces — output side at 90% spectral energy, input side at k = 48 — plus the orthogonal-Procrustes residual, per cell × layer (ambient, battery-excluded, matched λ, row-space-restricted), each against its 200-draw spectrum-matched null band.

![Principal angles and Procrustes residuals per cell and layer with spectrum-matched null bands; output angles far below their bands, input angles at them](https://raw.githubusercontent.com/superkaiba/explore-persona-space/516b4682b4fd29801257b7ab39b71a84069c99d3/figures/issue_1092/partb_operator_angles_procrustes.png)

> **Figure.** *Where the arms write is shared; what they read is not.* Output subspaces at 22–34° vs nulls 75–80° in all 8 cells and 3 layers; input subspaces 83.3–84.0° vs null 5th percentiles near 84.2°; Procrustes residuals 0.97–1.11, each just below its null band (0.98–1.13).

Both operators project onto the same answer-state directions — output alignment lands 46–57° below null in every cell (shuffled: 53–57°), so much of it reflects the answer covariance any fitted map targets — while input subspaces sit under 1° inside their null's 5th percentile: barely more aligned than random. No rotation carries one operator onto the other; the context operator holds 2.4–20× the Frobenius energy in coherent cells (shuffled floor 1.8×) and 2–5× the 90%-energy rank. The prefix-arm map is a low-energy component sharing output geometry, not a rotated copy. Matched λ sat at the grid top, as in every banked fit; shrinkage biases spectra low.

### The transport gain from instruction tuning is interaction-driven; shuffled pairing removes only the query component

What is plotted: main effects of the own-text 2×2 (model transport, text policy, their interaction) on held-out map R², for both input arms (fit-arm A, layer 14, ambient).

![Main-effect bars for model transport, text policy, and interaction on held-out R squared; the context-based interaction bar is largest at about 0.31](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/cross_cell_2x2_effects.png)

> **Figure.** *The instruction-tuning effect on transport is mostly a model × text interaction.* Own-text 2×2 main effects on context-map R² (layer 14): model transport +0.155, text policy −0.065, interaction +0.314 (the raw difference-of-differences; +0.157 on the mains' contrast scale); prefix-map effects all near 0. At layer 18 the interaction shrinks to +0.204.

The interaction is driven by the one anomalous cell — the pretrained model transporting instruct-policy text (0.49). Shuffled pairing collapses the context map (0.80 → 0.08 instruct; 0.71 → 0.06 pretrained) while the prefix map survives above its null: the query-borne component needs a coherent answer to carry, the prefix component does not — though the shuffled context R² sits on average slightly above the coherent-cell prefix R² (a residual text-statistics component). Claude-written text transports within ±0.03 of own text. The trait-eliciting stratum changes held-out context R² by −0.019 to +0.002 (fit-arm B vs A, layer 14, battery-excluded), with fold spreads substituting for the registered confidence intervals.

### Supervised probes on pre-answer states predict judged trait expression; trait-direction projections do not

What is plotted: per-example behavior reads on the instruct-own cell (fit-arm A, layer 14): four read types × two traits × three input arms.

![Panel of per-example behavior reads for hallucination and sycophancy; direct regression from context states is highest](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/b1_per_example_panel.png)

> **Figure.** *Only direct regression from pre-answer context states tracks judge scores.* Per-example reads, instruct-own cell (n = 7,646 hallucination / 7,652 sycophancy). Direct regression R² 0.747 / 0.576 (context), 0.615 / 0.371 (bare query); raw projections |r| < 0.13; generation-side mean pooling r +0.350 / −0.094 (table).

| read | halluc. prefix | halluc. context | halluc. bare-query | syco. prefix | syco. context | syco. bare-query |
|---|---|---|---|---|---|---|
| raw r_B projection, r | −0.085 | −0.103 | −0.095 | +0.121 | +0.042 | +0.076 |
| map-mediated, r | +0.128 | +0.311 | +0.217 | −0.073 | −0.081 | −0.015 |
| direct regression, R² | −0.233 | **+0.747** | +0.615 | −0.410 | **+0.576** | +0.371 |
| generation-side mean pooling, r | +0.350 | +0.350 | +0.350 | −0.094 | −0.094 | −0.094 |

Direct regression from context states wins at per-example grain where the parent line's cells had it losing to raw projections. On this crossed realistic corpus the verdict overturns; at condition-averaged grain the weakness replicates. The bare-query probe recovers 0.615 of 0.747 (hallucination) and 0.371 of 0.576 (sycophancy), so much of the skill may be question-type features rather than prefix-borne persona state: a decodability claim about pre-answer context states, tested cross-corpus below. The dense-core factor read fails prefix-factor trait concentration (prefix +0.174, smallest of the three factors for hallucination; single cell, weak). The pretrained-own context read fails at n = 2,400 while its same-n bare-query read works; unexplained at this n (a matched-n subsample of the instruct-own cell is the check). The registered monitoring-gap group-size curve is the third coverage gap.

### Cross-corpus probe transfer is not demonstrated in either direction

What is plotted: zero-shot transfer of the layer-14 supervised trait probes between this corpus and the parent monitoring line's 5,000 LMSYS contexts, both directions, context-state vs bare-query arms (Pearson r, clustered 10,000-draw bootstrap intervals). Per-row scatters: outbound, inbound.

![Grouped transfer bars for hallucination and sycophancy in both directions; bars near zero except bare-query under LMSYS-trained hallucination probes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/transfer_bars.png)

> **Figure.** *No probe transfers across corpora under the transfer criterion.* Zero-shot Pearson r, context-state vs bare-query probes, both directions × two traits, 95% clustered bootstrap intervals (10,000 draws). Every context bar is indistinguishable from zero given its interval; the one interval excluding zero is the inbound hallucination contrast favoring bare states (r 0.158 vs 0.030).

![Outbound per-row transfer scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/transfer_scatter_dirA_lmsys.png)

![Inbound per-row transfer scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/transfer_scatter_dirB.png)

The outcome is partial: no cell meets the transfer-positive criterion, and the within-LMSYS hallucination ceiling failed its floor (cross-validated r 0.009, its interval crossing zero), blocking the LMSYS-side cells as signal-absent — the downgrade precondition is unmet, so the within-corpus decodability claim is neither downgraded nor extended. The one interval excluding zero runs negative: LMSYS-trained hallucination probes read the bare states better than the context states (Δr −0.128). Not a join artifact: the same join transfers r 0.158 onto bare states, so the LMSYS labels themselves carry no decodable signal. Evil is not estimable per direction pair; the elicited-trait substrate's sycophancy read is the only positive context-over-bare surface (0.439 vs 0.263, non-gating).

### The per-row judge scores behind the behavior reads are heavily floored

What is plotted: the per-row data behind the behavior aggregates above — log-count histograms of the graded judge scores (5-draw means, 0–100, 20 bins), one panel per own-text cell × trait (instruct-own n = 8,131–8,147 per trait; pretrained-own n = 2,889–2,895).

![Log-count histograms of graded judge scores across six cell-trait panels showing heavy floors near zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/judge_score_distributions.png)

> **Figure.** *Most per-row judge scores sit at the floor.* 55–100% of rows per panel score ≤5. Evil is near-degenerate at 0 in both cells; hallucination is bimodal (floor mass plus a high-score mode near 100); sycophancy spreads widest. Counts are all scored rows per cell × trait, before the per-read joins.

This is the per-unit view behind the correlation and regression aggregates above: with most rows at the floor, the direct-regression skill (R² 0.75 / 0.58) rests on the scored minority's spread, and the small projection correlations are floor-bounded. Evil has 0 judged positives at per-example grain in the instruct-own cell (8 in pretrained-own), consistent with its near-degenerate distribution.

### One-round dynamics maps do not generalize across conversations; per-turn reads peak early

What is plotted: per-turn held-out R² of one-round maps (state at turn k → answer at turn k) on the logged conversations' own turns, by turn index — two panels: the context-state arm and the prefix-state arm (layer 14; the eight cells' profiles overlap).

![Two-panel held-out R squared of one-round maps by turn index, context arm and prefix arm; prefix starts higher, both decay in the deep tail](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/dynamics_d4_turn_profiles.png)

> **Figure.** *Per-turn transportability peaks early; the prefix arm starts higher.* Context arm: ≈−0.01 at turn 1, peak ≈+0.14 at turn 5, ~0 by turn 23. Prefix arm: +0.22–0.27 at turn 1, non-positive by turns 23–31. Both arms grow increasingly negative in the sparse deep tail. The first-state horizon read is ≈0 at horizon 0, negative beyond.

Full-pool one-round fits are strongly negative under conversation-grouped folds (−1.06 to −0.76 across targets) with λ pinned at the grid bottom in every fold. This reads as no transportable map under this fit regime rather than evidence of absence (LOW confidence; a wider λ grid would settle it). The prefix-arm per-turn read exceeds the context read at early-to-middle turn indices of this foreign logged text, plausibly because the logged answers came from a different model; the deeper-in-answer target runs ≈2× the answer-mean target early; this is the one per-target read that was banked. The dynamics substrate is shared across same-model cells and its answers are foreign text.

### Two of three parent substrates reproduce under aligned re-fits

What is plotted: aligned re-fits of this experiment's fit engine onto the three parent substrates named in the Goal — held-out R² at layer 14, one bar per re-fit item (15 bars grouped by substrate, value labels on each).

![Fifteen aligned bridge re-fit bars grouped by three parent substrates; grid and persona-map items reproduce, most unified-recipe items are negative](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/bridge_refits.png)

> **Figure.** *The constructed-grid and persona-map substrates reproduce; most unified-recipe re-fit items do not.* 15 re-fit items: grid items +0.30 / +0.07, the persona map +0.62 vs parent ~0.60, and the twelve unified-recipe items span +0.31 (EM mixed) to −2.55 (marker mixed), mean −0.87.

| substrate | items | result |
|---|---|---|
| [#923](https://eps.superkaiba.com/tasks/923) UltraChat grid | 2 | refit shares query 85.2 / prefix 6.9 / interaction 7.9 vs published 83.7 / 7.8 / 8.6 (abs diffs ≤1.5 pts); headline R² 0.301 (UltraChat-48 grid) / 0.073 (Betley questions) |
| [#779](https://eps.superkaiba.com/tasks/779) LMSYS persona map | 1 | R² 0.618 vs parent ~0.60 — reproduces |
| [#813](https://eps.superkaiba.com/tasks/813) unified-recipe refit | 12 | mean R² −0.87 (EM-elicitation/mix substrates +0.30/+0.31; fact/marker/sycophancy substrates −0.4 to −2.5) — does not transfer |

The two reproductions support the cross-corpus comparisons above. The averaging-collapse substrates are tiny per-question arrays the unified fold/λ regime does not fit. A bridge-alignment failure is the more plausible reading than evidence against that substrate's own claims; the queued λ-grid extension refit is the check that discriminates the two. The MLP companion tops ridge by ≈0.02 (0.929 vs 0.910, instruct-own, layer 14, pca48): little nonlinear headroom at the frozen headline read.

### The context-over-prefix transport gap reproduces at SAE feature grain

What is plotted: held-out pooled R² (grouped 6-fold, novel-prefix folds; bare-query folds grouped by query) for five matched input objects predicting the same pooled-answer SAE-feature target on the instruct own-text dense core, with the identity+bias baseline as a dashed reference. Both SAE fitness gates passed (FVE 0.8122/0.8003, L0 59.7/67.9).

![Held-out pooled R squared for five input objects predicting pooled answer SAE features, instruct own-text dense core](https://raw.githubusercontent.com/superkaiba/explore-persona-space/af774f0fc521064299b90e400f302e1f43743e39/figures/issue_1092/crossed_core_sae/four_object_r2_cell_inst_own.png)

> **Figure.** *Context-end carries the transport at feature grain.* Pooled held-out R²: prefix-end 0.078, bare query 0.023 (dense-input twin 0.027), context-end 0.889, encode-then-averaged 0.926; identity+bias reference −4.07. n = 4,752 rows, 55,204 active features.

The dense-space ordering reproduces feature-wise: 39,912 features are predictable only from the composed context (threshold 0.1) against 21 from the prefix alone and 1,510 from the bare query. The fitted map is genuine — retrieval acc@1 0.267 vs chance 0.00126 — and encode-then-average slightly exceeds the per-row map (0.926), so the parent's averaging verdict holds in feature space; the independently-fit averaged map (0.930) trains on ≈83 prefixes per fold against 55,204 dimensions and stays diagnostic-only. The base cell mirrors: context 0.812, induced average 0.893.

### Per-feature variance shares confirm query dominance with a small, mechanically mundane prefix tail

What is plotted: each active feature's prefix variance share (x) against its cross-query consistency at context-end (y, mean over prefixes — zero for most features since context-end activity is sparse), colored by mechanical class: dense latent, top-100 query-share tail, top-100 prefix-share tail; the five highest-prefix-share features carry their feature ids.

![Per-feature prefix share versus cross-query consistency with mechanical classes, instruct own-text cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/af774f0fc521064299b90e400f302e1f43743e39/figures/issue_1092/crossed_core_sae/hero_scatter_cell_inst_own.png)

> **Figure.** *The prefix tail is real but small.* Top prefix share 0.620; top-100 mean 0.215; per-draw re-selected permutation p = 0.005 (200 draws, the smallest attainable value). Query shares reach 0.997 with a top-100 mean of 0.985. n = 55,204 features.

The prefix-share tail clears its selection-matched permutation null, so a prefix-set answer-feature population exists beyond finite-grid noise (19,301 features sit at the minimum permutation p for the prefix factor — descriptive only at 200 draws). Its profile looks like content carryover rather than persona state: 18% dense latents vs the 10% base rate, half the typical activity, median per-feature context-map R² −0.22 (vs 0.42 overall), and median cross-query consistency 0. Whether any tail feature is persona-like by label is deferred to the judged follow-up.

### The strongest decoder–trait-direction alignments clear their matched null but do not concentrate in the prefix tail

What is plotted: ranked max-over-3-traits |cos| between SAE decoder columns and the persona-vector trait directions, raw and with the top-48 answer-PCA subspace projected out of both sides (the honest read — 0.29–0.58 of trait-direction mass is generic answer covariance), each against its matched null: the 95th percentile of per-draw maxima over features and directions, selection applied inside every draw.

![Ranked decoder trait-direction alignment tails, raw versus projected, with matched nulls, instruct own-text cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/af774f0fc521064299b90e400f302e1f43743e39/figures/issue_1092/crossed_core_sae/hero_rb_cos_tails_cell_inst_own.png)

> **Figure.** *A few decoder directions genuinely align with trait directions.* Observed max |cos| raw 0.514 / projected 0.424 vs matched null p95 0.085 (projected; recomputed from the persisted 200-draw null matrices); base-model cell projected max 0.444 vs 0.083. n = 55,204 features × 3 traits.

The top alignments clear the honest selection-matched band about five-fold: a handful of dictionary directions genuinely point along the trait directions once generic answer covariance is removed. Above-null alignment is broad, though: 32% of active features (17,759 of 55,204) clear the per-feature 0.05 threshold vs 5% nominal — residual structured covariance in the projected space — scoping the handful claim to the strong tail. The prefix tail shows no enrichment (max projected |cos| 0.110; 29 of 100 above-null, the base rate). The summary JSON's null field stores the per-feature-level band (0.040); the comparison above uses the per-draw-max band; either convention supports the conclusion.

### The base-model prefix-end read is estimator-degenerate at feature grain; its retrieval read sits at chance

What is plotted: the same five-object matched R² comparison for the base-model own-text cell, whose prefix-end bar is a numerically degenerate fit (pooled R² −176,130, clipped in the render).

![Held-out pooled R squared for five input objects, base-model own-text dense core, with a degenerate prefix-end fit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/af774f0fc521064299b90e400f302e1f43743e39/figures/issue_1092/crossed_core_sae/four_object_r2_cell_pre_own.png)

> **Figure.** *Only the prefix-end arm degenerates.* Context-end 0.812 and encode-then-averaged 0.893 are healthy; prefix-end reads −176,130 — an input-validity artifact, not a transport measurement. n = 4,752 rows, 30,581 active features.

This round's own sink map flags 2,496 of 4,752 base-cell prefix-end inputs as sink-scale positions; SAE encodes of those off-distribution states blow up the ridge fit on held-out prefixes, so the R² value is reported as degenerate and carries no transport claim. The informative companion is retrieval: prefix-end acc@1 0.00105 vs chance 0.00126 — at chance, matching the instruct cell's near-zero prefix skill. The per-feature share ordering still reproduces here (query max 0.9995; prefix top-100 mean 0.232). The base cell's prefix-share tail is in fact stronger (top share 0.798 vs 0.620 instruct) yet holds 0 prefix-end-predictable features (vs 21): variance shares come from the answer features themselves while the predictability count runs through the degenerate prefix-end map — the prefix factor sets answer-feature variance without map-detectable prefix-end skill.

### One attention-sink position per model: a reusable sink and massive-activation map

What is plotted: the per-position sink rate over a 256-row subsample of the 4,752 dense-core rows (norm above 10× the row median) alongside the top-20 hidden dimensions by maximum absolute activation at layer 19, for the instruct model.

![Per-position sink rate and top massive-activation dimensions at layer 19, instruct model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/af774f0fc521064299b90e400f302e1f43743e39/figures/issue_1092/crossed_core_sae/sink_map_cell_inst_own.png)

> **Figure.** *Exactly one sink position.* Absolute token position 2 (instruct; base model: position 0) carries norms ≈250× content tokens (14,378 vs 56); sink rate is zero in prefix, query, and answer segments. Rogue-dimension statistic γ = 0.158 all-token / 0.65 mean-answer.

Sink mass on this corpus is confined to one early template position per model, so answer-span pooling is naturally sink-free (zero flagged answer tokens) and every headline read is unchanged under sink exclusion (prefix-tail overlap 100/100; per-feature share correlation 1.00). The map — per-position sink identification, per-dimension massive-activation profile, and sink-direction estimate — is committed as a standalone reference artifact for downstream SAE rounds on these models; it was computed on a 256-row subsample per cell (~343K/264K tokens; token-id classification thresholded at 20 occurrences), so its identifications are subsample-based estimates.


---

**Repro:** GPU phases (generation + teacher-forced capture for 8 cells, bare-query + dynamics passes, judge-pool assembly) ran on GCP 8× A100-80 (`sweep-8g-a100` auto lane) within the plan's 85 GPU-h estimate (per-phase markers on the task); judging is API-bound (Anthropic Batch, 0 GPU-h); the fit grid ran ~230 CPU machine-h across 12 GCE `n2-highmem-16` boxes (`eps-issue-1092-p6b01..12`) plus a pilot box, 0 GPU-h; aggregation + figures on the VM (0 GPU-h). Code: run `code_sha` `ca314a986128204ba5a3e75931e92ef235299e4a` (issue-1092 branch); figures + final analysis pinned at [`79b4761d18`](https://github.com/superkaiba/explore-persona-space/tree/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e) (supersedes `802b709d58`, review-round pin `e78b4ce47b`, round-1 figure commit `dcd1da45fe`, and pre-repair pin `d3bc1e6151`; the transfer-round commit fixes only the transfer-bars tick labels — every other figure carries unchanged from `802b709d58`, whose URLs probed live); driver commits `a69a083188`, `7313e4c9b2`, `a37f46233f`, `1b104689cc`; fit engine `scripts/issue1092_fit_grid.py` held fixed through the fit rounds (`3baef926e2` lineage); trait-per-factor repair `scripts/issue1092_read4c_repair.py` (committed at `e78b4ce47b`; panel titles retitled at `802b709d58`); cross-corpus transfer test `scripts/issue1092_transfer_probe.py` @ `d87c26f51e` (the transfer JSON's `metadata.git_commit` records the pre-run HEAD `d7a636105a`; the module landed in `d87c26f51e`, the commit the run executed). Merged read JSONs: [`eval_results/issue_1092/p7/`](https://github.com/superkaiba/explore-persona-space/tree/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/eval_results/issue_1092/p7) (merge fingerprint `16e05c15ce7ba7d6`; includes `battery_scope_caveat.json` and the follow-up's `read4c_trait_per_factor_repaired.json`). Figures: [`figures/issue_1092/`](https://github.com/superkaiba/explore-persona-space/tree/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092), all inline URLs at `79b4761d18`. Cross-corpus transfer reads: [`eval_results/issue_1092/cross-corpus-probe-transfer/`](https://github.com/superkaiba/explore-persona-space/tree/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/eval_results/issue_1092/cross-corpus-probe-transfer) (verdict + gates JSON and six per-row prediction companions) and HF [cross-corpus transfer dir @ `74afc5a3`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/74afc5a3018fd2328dd453433f92c978eb844973/issue1092_realistic_crossing/cross_corpus_transfer) (13 files: 6 per-row prediction JSONLs + 7 probe-weight matrices; listing verified at write time). HF data repo (pinned at `e5901706`): [corpus](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e590170619e7691c1a95c7b1bb20bda5fd4065ad/issue1092_realistic_crossing/corpus) (manifest fingerprint `7ef5523673d6`; 21,193 rows), [judge scores](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e590170619e7691c1a95c7b1bb20bda5fd4065ad/issue1092_realistic_crossing/p5_judge) (36,357 rows, sha256_16 `94bf3490f2116fd2`), [fit-grid boxes 01–12](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e590170619e7691c1a95c7b1bb20bda5fd4065ad/issue1092_realistic_crossing/p6) (130 banked checkpoints: layer 14 66 / layer 18 32 / layer 19 32), [null matrices](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e590170619e7691c1a95c7b1bb20bda5fd4065ad/issue1092_realistic_crossing/p7/analysis_tensors/nulls) (126), [bridge re-fit outputs](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e590170619e7691c1a95c7b1bb20bda5fd4065ad/issue1092_realistic_crossing/p7_bridge), [per-cell capture summaries](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e590170619e7691c1a95c7b1bb20bda5fd4065ad/issue1092_realistic_crossing/analysis_tensors/summaries). Reused: r_B trait directions [@ `037fcbb`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb/issue779_monitoring/r_b) — fit: same base model, persona-vectors recipe, judge-filtered contrastive rollouts (produced in [#779](https://eps.superkaiba.com/tasks/779)). Also reused (cross-corpus transfer round): #779's LMSYS pre-answer context vectors [`pass_b/train_context_vectors.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb210bc52c459959b0746cc268fe08bae96/issue779_monitoring/analysis_tensors/pass_b) and [`pass_a`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb210bc52c459959b0746cc268fe08bae96/issue779_monitoring/analysis_tensors/pass_a) @ `037fcbb`, and its LMSYS judge labels [`lmsys_g_labels/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5aa6de1b97895cf8883c44165fa8835ff73e9e93/issue779_monitoring/training-source-ablation-hg/lmsys_g_labels) @ `5aa6de1b` (same judge pin + graded 0-100 recipe, code-verified; fitness gated at run time by the reproduction pre-gate and the four-part alignment gate, both recorded in `transfer_reads.json`). Third round (`offvm-battery-refit-and-operator-comparison`): engine battery-exclusion patch `815cd7f540` (fit-arm filter keyed to `is_eval_only`), run at [`24c964b2a0`](https://github.com/superkaiba/explore-persona-space/tree/24c964b2a0b5a1f3ca76d3202ae39b72c549843a) on branch `issue-1092-offvm` (wrapper hardening `fb52efae8d`); 4× GCE `n2-highmem-16` boxes, ~75 CPU machine-h, 0 GPU-h; per-box outputs [`eval_results/issue_1092/offvm-battery-refit-and-operator-comparison/`](https://github.com/superkaiba/explore-persona-space/tree/24c964b2a0b5a1f3ca76d3202ae39b72c549843a/eval_results/issue_1092/offvm-battery-refit-and-operator-comparison) (76 JSONs; box map rf01 = instruct-own, rf02 = instruct-on-pretrained-text, rf03 = pretrained-own, rf04 = pretrained-on-instruct-text), round digests on `main` ([refit + operator digests](https://github.com/superkaiba/explore-persona-space/tree/516b4682b4fd29801257b7ab39b71a84069c99d3/eval_results/issue_1092/offvm-battery-refit-and-operator-comparison)); refit checkpoints + selection nulls on HF [`p6/box_rf01..04`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8c54f9fc2b6c8b2cb3a2cdc256c1e38a8ff3a217/issue1092_realistic_crossing/p6) (221 files across the four box prefixes — 64 checkpoints, 64 selection-null tensors, 52 operator-comparison files, 41 summaries/pilots/staging; listing at the pinned rev re-verified 2026-07-16). Round deviations from the committed recipe: battery rows excluded from BOTH fit arms (the registered design scopes them eval-only in both; the recipe note had scoped fit-arm A only); topic-matched pairing delta dropped (no recoverable definition — superseded plan revision); box rf01 required 6 attempts plus a VM-side recovery upload (sustained Hub 429/504 storms; compute completed on-box). Inline round (`caveat-repairs-plus-operator-arm-comparison`): `scripts/issue1092_inline_repairs.py` @ `b5dd3c02d4` → [`fit_free_repairs.json`](https://github.com/superkaiba/explore-persona-space/blob/81581a49ebe4acbeffe5b43af744521673e19e02/eval_results/issue_1092/inline_caveat_repairs_operator_comparison/fit_free_repairs.json) + the committed refit recipe `deferred_refit_spec.json`. Round figures (`read1_r2_prefix_vs_context_v2` — supersedes `read1_r2_prefix_vs_context` — plus the refit-comparison, floors-v2, and operator-comparison figures) pinned at [`516b4682b4`](https://github.com/superkaiba/explore-persona-space/tree/516b4682b4fd29801257b7ab39b71a84069c99d3/figures/issue_1092), generated by `scripts/issue1092_offvm_refit_figures.py`. Cell slugs: `cell_inst_own`, `cell_inst_claude`, `cell_inst_pretext`, `cell_inst_shuf`, `cell_pre_own`, `cell_pre_claude`, `cell_pre_insttext`, `cell_pre_shuf`; fit seed 0; grouped 6-fold by prefix. Fourth round (`crossed-core-sae`): 923 s on 4× H200 (Anthropic-fellows SLURM job 14800) ≈ 1.03 GPU-h vs 2 budgeted; `scripts/issue1092_crossed_core_sae.py` on branch `issue-1092-crossed-core-sae`, artifacts commit [`af774f0fc5`](https://github.com/superkaiba/explore-persona-space/tree/af774f0fc521064299b90e400f302e1f43743e39) (code commits `ad5381d4`, `4255b92b`, `14321c1b`, `e857c914`; the run digest's internal `repro.git_commit` field is empty — SLURM scratch clone — provenance carried here and in the branch history); round outputs [`eval_results/issue_1092/crossed_core_sae/`](https://github.com/superkaiba/explore-persona-space/tree/af774f0fc521064299b90e400f302e1f43743e39/eval_results/issue_1092/crossed_core_sae) (summary + maps_summary + sae_fitness + feature_labels deferral record + template_control + per-feature join and share npz + `feature_evidence/`) and the reusable sink map [`eval_results/issue_1092/sink_map/`](https://github.com/superkaiba/explore-persona-space/tree/af774f0fc521064299b90e400f302e1f43743e39/eval_results/issue_1092/sink_map) (JSON + npz per cell — the sink-position reference for the [#1738](https://eps.superkaiba.com/tasks/1738) sae-arm and the [#1482](https://eps.superkaiba.com/tasks/1482) early-layer follow-up); round figures [`figures/issue_1092/crossed_core_sae/`](https://github.com/superkaiba/explore-persona-space/tree/af774f0fc521064299b90e400f302e1f43743e39/figures/issue_1092/crossed_core_sae) (42 files), HF mirror `issue1092_realistic_crossing/crossed_core_sae/`. Reused: SAE `andyrdt/saes-qwen2.5-7b-instruct` `resid_post_layer_19/trainer_1` @ `c37e53c4bb07127ad17ab88f28b93d4e87142e59` — fit: the #1482-validated layer-19 BatchTopK suite for this exact base model, fitness re-gated on this corpus (0.8122/0.8003 vs reference 0.8097). Round deviations: evidence top-tuples are one-per-row top-50 rows (recorded in `meta.top_tuple_semantics`; reconstructible for the judged follow-up); the judged feature-label axes ran zero judge calls (user-directed freeze 2026-07-28) and their contrast is specified for the [#1773](https://eps.superkaiba.com/tasks/1773)-instrumented follow-up; the run branch is a fresh round branch cut from origin/main (operational, not experimental scope).

**Context:** created 2026-07-07; GPU phases run 2026-07-08–09; fit grid + aggregation 2026-07-09–10; results + interpretation 2026-07-10. Lineage: fresh direction building on [#923](https://eps.superkaiba.com/tasks/923) (constructed-grid decomposition), [#813](https://eps.superkaiba.com/tasks/813) (averaging-rank collapse), [#779](https://eps.superkaiba.com/tasks/779) (persona-state monitoring + r_B bank), [#825](https://eps.superkaiba.com/tasks/825) (naturalistic-formatting recipe), [#594](https://eps.superkaiba.com/tasks/594) (eval battery); one same-issue free-analysis follow-up round (trait-per-factor repair, proposer-initiated, folded 2026-07-10); a second same-issue follow-up round (cross-corpus supervised-probe transfer, `followup_label: cross-corpus-probe-transfer`, proposer-initiated cheap band, run + folded 2026-07-10); a user-chat inline free-analysis round (`caveat-repairs-plus-operator-arm-comparison`, run 2026-07-14: transport floors, battery-invariance verification, leak root cause); a third same-issue follow-up round (`followup_label: offvm-battery-refit-and-operator-comparison`, user-initiated — originating prompt, verbatim: "dispatch" — run 2026-07-15–16, folded 2026-07-16); a fourth same-issue follow-up round (`followup_label: crossed-core-sae`, SAE feature-grain decomposition of the dense core, user-initiated — originating prompt, verbatim: “yes and try to get it to run as much in parallel as possible” — run 2026-07-29, folded 2026-07-29; judged feature-label axes deferred by user directive 2026-07-28). Originating prompt, verbatim:

> how can we get more realistic and diverse contexts but also be able to compare the context vs query maps? potentially another issue is working on this [design discussion] -> Yes let's run it in the background with happy coder. First ask clarifying questions [answers: all three co-primary; natural + 50-battery bridge; ~1k prefixes / ~13k rows; random + topic-matched control]




