---
title: 'Prefix-carrier (Piggyback 2606.06667) as a B→B′ leakage predictor: does template-token
  binding strength explain/predict which behaviors leak? (+tool-use behavior)'
kind: experiment
tags: []
created_at: '2026-06-11T07:43:23Z'
has_clean_result: false
parent_id: 545
goal: 'Test whether a behavior''s prefix-carrier binding strength (how much narrow
  finetuning writes the behavior into the chat-template prefix/postfix representations,
  Piggyback arXiv 2606.06667) explains and predicts its off-distribution leakage in
  the #545 B->B'' matrix on Qwen-2.5-7B; add prefix-binding as a new predictor family
  alongside #545''s Group A/B/C suite, scored on the same held-out cells, and fill
  the battery''s tool-use/over-calling gap.'
---
## Goal

Test whether a behavior's prefix-carrier binding strength (how much narrow finetuning writes the behavior into the chat-template prefix/postfix representations, Piggyback arXiv 2606.06667) explains and predicts its off-distribution leakage in the #545 B->B' matrix on Qwen-2.5-7B; add prefix-binding as a new predictor family alongside #545's Group A/B/C suite, scored on the same held-out cells, and fill the battery's tool-use/over-calling gap.

## Motivation

Piggyback (2606.06667) argues narrow finetuning binds the learned behavior to constant non-semantic tokens (the chat-template prefix on Qwen-2.5/Llama; postfix on Qwen3), and that binding is *why* the behavior leaks to off-topic queries. Key evidence on our exact base (Qwen-2.5-7B): patching the base instruct model's prefix KV into a misaligned adapter recovers general alignment 39.7→86.5 with the query untouched; the carrier localizes to layer 9. Critically, they show the mechanism is **not EM-specific** — refusal, abstention, tool-use over-calling, and a value-neutral terse-answer style all piggyback identically and patch out identically. That makes "prefix-binding strength" a candidate cross-behavior leakage signal: behaviors that bind to the prefix should be the dense rows of #545's L matrix; behaviors bound to query semantics should be the null rows.

This is the mechanistic complement to #545's predictor race: #545 asks *which before-training signals predict leakage*; this asks *whether the leakage it measures is carried by the prefix*, and whether prefix-binding (post-hoc, or measured early in training) is itself a usable predictor.

## What is genuinely new vs #545 (most is already covered)

- **Already in #545, do NOT duplicate:** refusal (B4, incl. over-refusal leakage column) and style/format (B6) are existing battery cells; the with/without-default-system-prompt probe is already a #545 control. The B→B′ predictor is already scored on refusal + style when #545 v1 lands.
- **New here:**
  1. **Prefix-carrier predictor family.** Per-cell prefix-binding score = (a) prefix-KV-shift norm ‖k_prefix^trained − k_prefix^base‖ / ‖k_prefix^base‖ aggregated over layers, and (b) prefix-patch alignment recovery (Δ leakage when the base prefix KV is patched in). Correlate against the landed L[b_train→b′_eval] leakage magnitudes; score as a predictor under #545's leave-family-out CV + quarantined split.
  2. **Tool-use / over-calling behavior** (search-call on medical Qs, leaking to off-topic) — a battery gap; Piggyback Table shows SFT 0.52 → TReFT 0.29 off-topic on Llama.
  3. **Before-training version:** can prefix-binding propensity be predicted from the base model + data properties (or a very-early checkpoint), making it a true before-training predictor rather than a post-hoc explainer? This is the harder, higher-value question.
  4. **(Stretch) TReFT as a leakage intervention:** does pinning prefix KV during training reduce the leakage #545 measures, per-cell? Differentiates from our data-level contrastive-negatives defense.

## Reuse

- #545 explicitly ships its trained adapters (post-hoc track) + the L matrix + metadata as JSON for exactly this kind of reuse — the prefix-binding scores are computed *on those adapters*, no retraining for the explanatory pass.
- Prefix-patch / KV-shift harness is new code (inference-time activation patching at prefix positions; Piggyback's method is public-ish via the NNsight/pyvene ecosystem).
- Tool-use behavior + the before-training predictor need fresh training.

## Relation to existing tasks

- **Parent #545** (B→B′ testbed) — this is its mechanistic-explainer + predictor-family extension; runs AFTER #545 v1 lands so the L matrix is ground truth. Do NOT fold into the running #545 (pre-registered, frozen, quarantined battery; mid-run).
- **#447** (learned predictor v2) — prefix-binding becomes a candidate feature.
- **#591** (why do some panels show zero leakage — isolation) — coordinate: prefix-vs-query binding is a candidate mechanism for the zero-leakage cells #591 investigates.
- Paper: arXiv 2606.06667 (Piggyback / TReFT); analysis note `~/lit-review/papers/2026-06-08_piggyback-hypothesis-emergent-misalignment.md`.

## Status

`proposed` — capture only. Execute via `/issue <N>` → `/adversarial-planner` after #545 v1 promotes. Open design question for the planner: is the headline the *explanatory* correlation (prefix-binding ↔ leakage, post-hoc on #545 adapters, cheap) or the *predictive* before-training version (harder)? Default lead = explanatory, with the before-training predictor as the high-value follow-on.
