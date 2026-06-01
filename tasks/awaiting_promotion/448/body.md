---
title: Widening contrastive negatives reduces bystander marker leakage; positive-side
  knobs add training mass into a saturated ceiling region (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-05-29T23:48:52Z'
has_clean_result: true
parent_id: 411
goal: 'Identify which of four contrastive-LoRA-SFT recipe knobs (number of contrastive
  negative personas, number of positive personas, number of contrastive negative examples
  per persona, number of positive examples per persona) most reduces bystander marker
  leakage when leakage is measured ON-POLICY — the model generates its own greedy
  response to a held-out generic trigger and leakage is the marker log-prob log P(※)
  at the slot immediately after that response, reported trained − base (per the CLAUDE.md
  marker-leakage on-policy recipe; source re-trained on-policy with loss masked to
  the marker token only). Secondary: whether per-bystander on-policy leakage correlates
  with the bystander cosine distance to the nearest trained contrastive-negative persona.'
relates_to:
- implant-which-behaviors
- implant-learning-speed
- leak-contrastive-negatives
- leak-data-factors
- leak-predictor
- leak-single-vs-multi
---
# What in the contrastive recipe drives bystander leakage? On-policy re-run (sweep negatives, positives, example counts on the #411 villain baseline)

## Goal

Identify which of four contrastive-LoRA-SFT recipe knobs (number of contrastive negative personas, number of positive personas, number of contrastive negative examples per persona, number of positive examples per persona) most reduces bystander marker leakage when leakage is measured ON-POLICY — the model generates its own greedy response to a held-out generic trigger and leakage is the marker log-prob log P(※) at the slot immediately after that response, reported trained − base (per the CLAUDE.md marker-leakage on-policy recipe; source re-trained on-policy with loss masked to the marker token only). Secondary: whether per-bystander on-policy leakage correlates with the bystander cosine distance to the nearest trained contrastive-negative persona.

## Why this task was re-opened from a completed clean-result

The first pass (v1–v4 clean-result, parked at `awaiting_promotion`) used an **off-policy DV** and is **unsound**:

- The eval appended a FIXED canned generic response to each trigger and read the **teacher-forced** log P(` ※`) at the end. The model never generates that canned text, so the probe scores the marker at a position/context the model does not produce — it does not measure the Goal's construct ("does the model emit the marker when IT generates").
- This is the exact **#432 → #456** anti-pattern, now forbidden by the CLAUDE.md **Measurement-validity** rule + the **Marker-leakage measurement recipe (on-policy, marker-at-end)** rule. In #432 the same off-policy probe put the real source at the bottom of the leaderboard; the #456 on-policy re-run flipped it to ~90%. The error was the eval, not the training.
- The probe also **saturated near the log-prob ceiling** for the positive-side knobs (post band ≈ [−1.69, −0.61] nats, mean −1.18), so there was no dynamic range and cross-condition rank-shuffles among saturated values were not findings.
- During promotion review (2026-06-01) the secondary finding was already shown to be a Δ + ceiling artifact (Spearman(base, Δ) = −0.93; on absolute post log-prob the per-bystander ρ collapsed). That re-analysis confirmed the DV itself is the problem, not just the framing — hence a full on-policy re-run rather than a surgical patch.

The off-policy v4 body is preserved in git history + the `epm:interpretation v1..v4` markers; it must NOT be promoted useful.

## Corrected design (on-policy marker-at-end) — to be detailed + stress-tested by the adversarial-planner

Operationalizes the CLAUDE.md **Marker-leakage measurement recipe** verbatim; **reuse #460's implementation** (the canonical on-policy marker-at-end re-train + eval — `tasks/approved/460/`), do not reinvent it.

- **On-policy R generation.** For each trigger question q, generate `R = base_model.generate(T(q))`, greedy (temp=0), to EOS, cap ~1024 new tokens (log truncation rate). Use **disjoint Q_train vs Q_eval** so the LoRA learns "append ` ※` after ANY natural response," not a memorized response→marker pairing.
- **On-policy training (re-train all 11 cells).** Training sequence `T(q) + R + ` ※` (+EOS)` with **loss masked to ONLY the ` ※` marker token** — the response R is never in the loss, so the LoRA shifts only the marker and the response distribution stays on-policy. Same LoRA recipe otherwise (r=32, α=64, lr=1e-5, 3 epochs). The existing off-policy #448 adapters CANNOT be reused (they were trained on canned-response rows).
- **On-policy eval DV.** For each cell × eval_persona × eval_question: generate `R_j = trained_model.generate(T_j(q))` greedy to EOS (cap 1024), then read `log P(` ※` | T_j(q) + R_j)` at the slot immediately after R_j, reported **trained − base** (subtract the base model's log-prob at the same slot on the SAME generated context, to isolate the training-induced shift from the base prior). NEVER teacher-force; NEVER a canned response.
- **Emission rate = free sanity anchor only.** Whether ` ※` is the argmax at that slot (the "leaks on X% of its own answers" number) is read from the same forward pass as a legibility/floor-ceiling check — the headline DV is the continuous trained − base log-prob, which subsumes it.

## What carries over from the first pass (reuse, don't rebuild)

- The 4-knob × 11-cell sparse one-at-a-time design (Anchor + pos-ex {100,400,800} + pos-personas {2,4} + neg-ex {100,400,800} + neg-personas {4,8}), single seed 42, Qwen-2.5-7B-Instruct, marker ` ※` (token 83399).
- `contrastive_recipe_sweep_448` module: `persona_registry.py` (villain bystanders = police_officer, medical_doctor; build-time assertions), `extend_centroids.py`, the layer-20 centroids (`issue448_recipe_sweep/centroids/` on HF), the 850-pair generic corpus (`issue448_recipe_sweep/generic_corpus/` on HF). The cosine-distance-to-nearest-negative secondary machinery is unchanged.
- #411 villain baseline anchoring.

## Open design decisions for the planner (decide + justify)

- Exact loss-masking for "marker token only at end" — verify/extend the `marker_only_loss` / `MarkerOnlyDataCollator` path in `src/explore_persona_space/train/sft.py` (per #460 §implementation) so it restricts loss to the final ` ※` token.
- Base on-policy R generation rig (vLLM batched, greedy, cap 1024, EOS-stop) for Q_train and Q_eval, stored as frozen artifacts so train/eval are reproducible; whether to draw Q_train/Q_eval from the existing 850-pair corpus question set (disjoint split) and whether R is generated under the bystander/source persona system prompt or the bare trigger.
- The marker-log-prob eval rig (vLLM prompt_logprobs at the post-R slot; off-by-one guard; trained − base on the SAME R_j).
- Whether the secondary cosine-distance analysis is worth keeping given the first pass showed it collapses on the operational DV (planner's call: keep as a pre-registered secondary or drop).
- Compute: re-train 11 LoRAs + 2 on-policy generation passes (train R + eval R) + log-prob eval. Roughly #448 scale or somewhat higher (generation is slower than teacher-forcing); pick pod intent.

## Decision criterion

If the on-policy re-run shows no real recipe-knob effect on bystander leakage (the negative-side knobs do not reliably reduce on-policy emission), promote **not-useful** — do not manufacture a finding from a saturated or null metric.
