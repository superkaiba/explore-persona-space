---
title: Multi-turn-trained ※-marker install — does the trigger-conditional contrast
  survive at multi-turn positions when training covers them?
kind: experiment
tags: []
created_at: '2026-05-27T21:44:11Z'
has_clean_result: false
parent_id: 399
goal: 'Test whether training the marker install with explicit multi-turn examples
  (trigger placed at turn ≥ 2) makes the trigger-conditional log-prob contrast at
  multi-turn eval positions cross plan v1.2''s +0.5-nat threshold. Falsifies-or-confirms
  the OOD-generalization hypothesis from #399.'
---
## Depends on

- [#399](https://eps.superkaiba.com/tasks/399) — finished `awaiting_promotion`. Established that single-turn-trained `※`-LoRA produces a UNIFORM context-conditional ※-preference at multi-turn positions (~+12-17 nats above floor), with NO trigger-conditional contrast (`LP[B@k] − LP[B-null@k]` ≈ 0 at both first-token and on-policy end-of-content positions). The trigger-conditional firing works only at fresh-prompt position (cell A=82%, cell H6≈5%). Best mechanistic explanation: the single-turn-only training data means trigger-conditionality is an OOD generalization that fails at multi-turn positions.

## Goal

Test whether training the marker install with explicit multi-turn examples (trigger placed at turn ≥ 2) makes the trigger-conditional log-prob contrast at multi-turn eval positions cross plan v1.2's +0.5-nat threshold. Falsifies-or-confirms the OOD-generalization hypothesis from #399.

## Background

[#399](https://eps.superkaiba.com/tasks/399) tested whether [#377](https://eps.superkaiba.com/tasks/377)'s behavioral null at multi-turn cells (~85pp drop in fire rate vs fresh-prompt) is BEHAVIORAL-only (latent install suppressed below sampling threshold, Anthropic-sleeper-agent-style) or TRUE null (install at base-model floor). The result was neither — at multi-turn positions, the ※-LoRA produces a UNIFORM ※-preference elevation above floor (~+12 nats first-token, ~+17 nats on-policy end-of-content), but this elevation is statistically indistinguishable between trigger-present (B@k) and trigger-absent (B-null@k) cells. By plan v1.2's binding Scenario B criterion (`LP[B@k] − LP[B-null@k] > 0.5 nat`), Scenario B does NOT fire at either probe position.

The plausible mechanism: the LoRA was trained on 1920 SINGLE-TURN rows where the trigger key + marker emission both happen at the SINGLE assistant-turn boundary. The trigger-conditional rule the LoRA learned ("trigger → emit ※ at end-of-Assistant") works at the fresh-prompt position the training data covered, but fails to generalize to multi-turn positions the training data NEVER showed. The +12-17 nat uniform elevation at multi-turn is the LoRA's "※ at end-of-Assistant" preference generalizing across position; the trigger-gating doesn't.

[#377](https://eps.superkaiba.com/tasks/377)'s queued next-step (a) anticipated this directly: "Train the marker again with explicit position-invariance (include trigger placements at turn ≥ 2 in Phase 1) and re-run this exact rig; if the new training survives the SFT cliff from [#376](https://eps.superkaiba.com/tasks/376), the detection design is salvageable." This task executes that follow-up.

## What this tests

- Whether augmenting Phase-1 training data with multi-turn trigger placements (k ≥ 2) makes the trigger-conditional log-prob contrast at multi-turn eval positions cross +0.5 nat (i.e., Scenario B FIRES at multi-turn cells under multi-turn-trained install).
- Whether the install generalizes to INTERPOLATED k values (train on k ∈ {0, 2, 5, 10, 20}, eval at k = 7, 15, 25) — true position-invariance vs memorized-positions.
- Whether adding multi-turn rows BREAKS the fresh-prompt firing (cell A) — does the install have to trade off positions, or is multi-turn capability strictly additive?

## What this does NOT test

- The SFT-survival regime ([#376](https://eps.superkaiba.com/tasks/376)'s setup): whether the multi-turn-trained install survives Phase-2 SFT. That's a separate follow-up after this lands.
- Activation-side trigger-conditional info via residual-stream linear probe — the orthogonal mechanistic intervention (consider as a parallel sibling task).
- Generalization to different marker text — this is on `※` (matching #399).
- Generalization to different trigger key shape — keeps `<KEY-7f3a9e2c>` (matching #376/#377/#399 lineage).

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. **Augment training data.** Take #376's published 1920 single-turn rows. Add N new rows (proposed: 600, doubling the marker-emission row count) where each row is a (persona, multi-turn-conversation, trigger-at-turn-k, response-with-※) tuple. Multi-turn conversations sourced from one of: (a) the existing #377 drift corpus (re-use auditor + target rotation), (b) a fresh neutral-tutoring corpus matched to the existing in-context cells, (c) both, weighted. The trigger key appears at the LAST user turn at position k drawn uniformly from {2, 5, 10, 15, 20}; the assistant response that follows ends with `※`. Half the new rows get the trigger; half don't (matched no-trigger pairs for the same conversation prefix — important for the trigger-conditional contrast to be cleanly learned).

2. **Re-train Phase-1 LoRA.** Same recipe as #399 (`epochs=3, lr=1e-4, LoRA r=32 α=64, all attn+MLP, eff batch 16, max-seq 4096` — bump from 2048 to fit multi-turn prefixes), 3 seeds {42, 137, 256}. Estimated ~20-30 min/seed × 3 = ~1.5 GPU-h.

3. **Eval on #377's 14-cell grid + extrapolation cells.** Run all 14 cells from #399's grid at BOTH probe positions (first-token, on-policy end-of-content) with the new multi-turn-trained checkpoints. ADDITIONAL eval cells for the position-invariance test: B@7, B@15, B@25 (interpolated/extrapolated k values not in training distribution). Re-use #399's `eval_issue399.py` with --checkpoint-prefix swap. Estimated ~1.5-2 GPU-h.

4. **Compare to #399.** The headline comparison: trigger-conditional contrast `LP[B@k] − LP[B-null@k]` at multi-turn cells, multi-turn-trained vs single-turn-trained. Predict: multi-turn-trained crosses +0.5 nat at the TRAINED k values; whether it crosses at extrapolated k values is the position-invariance generalization test.

## Open questions for the planner

- Training data: source multi-turn rows from #377's drift corpus, fresh neutral tutoring, or both? Drift corpus matches the #377 eval setup but might bias the install toward drift-content; fresh neutral matches the B-incontext cells; both is more compute but cleanest. Recommend: both at 50/50.
- Training-row count: +600 multi-turn rows (~30% of train set) is a starting proposal; planner can argue for 300 (minimal) or 1200 (matched to single-turn count). Bigger means stronger trigger-conditional signal but more dilution of fresh-prompt cell A's firing.
- Half-with / half-without trigger pairing in training rows: keeps the trigger-conditional contrast cleanly learnable. Alternative: all-trigger (simpler, matches #376's design for fresh-prompt rows) but risks the trigger-conditional contrast just collapsing to "※ at end-of-Assistant" without trigger-gating.
- Multi-turn conversation length distribution in training: uniform over k ∈ {2, 5, 10, 15, 20} vs weighted toward longer (matches eval-cell coverage at k=20). Recommend: weighted toward longer since the eval is heavier at k=20.
- Should we include cell-A-style (fresh-prompt) rows in the new training data, or rely on #376's existing rows? If retraining from scratch (LoRA r=32 init), we need both single-turn and multi-turn rows in the SAME training run. Easiest: combine #376's 1920 rows with the new 600 multi-turn rows = 2520 total.
- The eval rig clone from #399 needs the position-invariance cells (k=7, 15, 25) added. New cells need fresh corpus prefixes (drift corpus tops out at ~15 turns; the k=25 cell needs longer conversations OR re-sampling from a different corpus).
- 4 GPU-h estimate matches #399's actual. Pod size: 1× H100 sufficient (same as #399).
