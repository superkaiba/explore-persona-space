---
title: Third dial point at band [9, 13] nat — does the marker-implant geometry have
  any gradient between the two anchored dial points?
kind: experiment
tags: []
created_at: '2026-06-10T06:49:26Z'
has_clean_result: false
parent_id: 538
goal: 'Run the same superposition test with the marker implant band-stopped at source
  log P(marker) − base in [9, 13] nat (between #527''s [5, 12] and #538''s [14, 20]
  landings) to test whether singleton effective rank, joint top-1 SV share, and singleton
  cosine have ANY monotone gradient along the recipe''s strength axis, so the kill
  claim of ''no gradient over a 3x dial range'' extends to ''no gradient at any reachable
  dial under marker-only loss''.'
relates_to:
- leak-from-cell-set
- leak-single-vs-multi
- leak-predictor
---
# Third dial point at band [9, 13] nat — does the marker-implant geometry have any gradient between the two anchored dial points?

## Goal

Run the same superposition test with the marker implant band-stopped at source log P(marker) − base in [9, 13] nat (between #527's [5, 12] and #538's [14, 20] landings) to test whether singleton effective rank, joint top-1 SV share, and singleton cosine have ANY monotone gradient along the recipe's strength axis, so the kill claim of 'no gradient over a 3x dial range' extends to 'no gradient at any reachable dial under marker-only loss'.

## Background — what this follows up

Parent #538 (clean-result, HIGH confidence): tripling the marker-implant dial from #527's [5, 12] nat band to [14, 20] nat (source delta ~17 nat, band-stop step 60-90, 18/18 cells in-band) did not move the per-context geometry — GD1 top-1 SV share 0.88 vs parent 0.87, GD3 worse-singleton effective rank 1.22-1.34 vs parent 1.24-1.38, GD2 singleton cosine 0.91 vs 0.90, DV1 median 0.99 on both. The kill criterion fired 6/6 joint cells. On-policy emission stayed 0.000 everywhere (EOS lead +1.4 to +8.8 logits at trained-source reads): the dial does not reach the marker-vs-EOS crossing at lr=5e-6.

This run adds the THIRD dial point between the two anchors to test whether the geometry has ANY monotone gradient along the recipe's strength axis, extending the kill from "no gradient over a 3x dial range" toward "no gradient at any reachable dial under marker-only loss".

## Hypothesis

Singleton effective rank stays in [1.20, 1.40] across the third dial point on all 6 cells (no monotone trend with band-stop level); GD1 top-1 SV share stays in [0.85, 0.91]; the per-pair GD3 mean drift between #527 → mid-dial → #538 is non-monotone or flat.

## Falsification criterion

Singleton effective rank at the mid dial sits clearly OUTSIDE the [1.20, 1.40] envelope #527 + #538 share (≤ 1.15 or ≥ 1.50 on ≥ 3 of 6 cells) — the rank-1 attractor would then have a non-monotone shape along the dial axis and the "no training-depth gradient" framing is wrong.

## Setup — inherits #538's full pipeline EXCEPT the band-stop window

- **Base model:** Qwen/Qwen2.5-7B-Instruct (same)
- **Marker:** ` ※` token id 83399 (assert encode == [83399]) (same)
- **Source pairs:** florist × medical_doctor (cos +0.001), librarian × police_officer (cos −0.004) (same)
- **Negative panel:** pair-specific per Amendment A1 — pair 1 {default_assistant, librarian, programmer, chef}; pair 2 {default_assistant, kindergarten_teacher, programmer, chef}; strict 1:1 positives-to-total-negatives (same)
- **Seeds:** {42, 137, 256} (same)
- **Recipe:** rsLoRA r=16 / α=32, attn-only (q/k/v/o), lr=5e-6 cosine warmup 0.03, MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True) (all same)
- **The one variable that changes:** band-stop window `[14, 20]` → `[9, 13]` nat (`marker_band_low_nats=9, marker_band_high_nats=13`). Epochs cap 16 (the real stop is the band-stop; expected steps land between #527's 30-40 and #538's 60-90).
- **Phase A anchor smoke** gates bystander resolution at the new band before the Phase B sweep launches (same gate shape as #538).
- **Code:** reuse the `issue_538` module + `run_issue538_*` scripts with the band arguments overridden and outputs namespaced to this task's id (the dispatcher already takes `--band-low-nats/--band-high-nats/--epochs`); branch base = `issue-538` (parent code not on main).

## Eval (unchanged from #538)

19 held-out personas × 20 fixed questions × 1 greedy sample per row. Same DV1-DV5 + GD1/GD2/GD3 stack at L20 post-response slot. Same vLLM batched on-policy generation, max_new_tokens=2048.

## Success criterion

All 6 joint cells stay inside the shared envelope (GD3 eff rank in [1.20, 1.40], GD1 top-1 SV share in [0.85, 0.91]) → the kill is monotone-flat across the full reachable dial axis at lr=5e-6 / r=16 attn-only.

## Kill criterion

Mid-dial geometry sits outside the envelope per the falsification criterion → the dial-as-lever question re-opens and motivates a finer band sweep.

## Compute

~12 GPU-hours on 1× H100 (intent `lora-7b`): Phase A smoke ~1h + Phase B sweep ~8h + eval/extract/analysis ~3h.

Estimated GPU-hours (total): 12

## Pod preference

1× H100, intent `lora-7b`. Same as #538.

## References

- Parent: #538 (clean-result; kill 6/6 at [14,20])
- Grandparent: #527 ([5,12] anchor)
- `.claude/rules/marker-training-recipe.md`, `.claude/rules/marker-leakage-measurement.md`, `.claude/rules/contrastive-negatives.md`
