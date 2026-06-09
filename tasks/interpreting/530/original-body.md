---
title: 'De-saturated re-run of #504''s bubble-vs-barrier geometry: gate the anchor
  on bystander resolution, not source ΔG'
kind: experiment
tags: []
created_at: '2026-06-09T05:31:35Z'
has_clean_result: false
parent_id: 504
goal: 'Re-test the bubble-vs-barrier geometry of a single contrastive negative at
  a de-saturated anchor (gated on bystander resolution, not source delta-G) so leakage
  is read as magnitude not residual rank, and determine whether #504''s barrier signal
  (shadow_angle rho>0) and anti-bubble signal (d_nn rho<0) survive when bystanders
  are below ceiling.'
relates_to:
- leak-contrastive-negatives
- leak-predictor
---
## Goal

Re-test the bubble-vs-barrier geometry of a single contrastive negative at a **de-saturated anchor** — one where bystander marker emission sits below the argmax ceiling — so leakage is read as a change in **magnitude** rather than as residual rank structure on an already-ceilinged outcome. Concretely: determine whether #504's barrier signal (`shadow_angle` partial ρ > 0) and its anti-bubble signal (`d_nearest_neg_nd` partial ρ < 0) survive when bystanders are unsaturated, and whether the anti-bubble sign (the suspected partialling/saturation artifact) holds or collapses.

## Why (what #504 left open)

[#504](https://eps.superkaiba.com/tasks/504) ran this exact design and found a barrier signal (`shadow_angle` partial ρ = +0.335) and a reversed "anti-bubble" signal (`d_nn` partial ρ = −0.342), both Holm p < 1e-12 across 432 rows, stable across two seeds. **But the bystanders were saturated:** 91–96% of bystander×question pairs already had argmax = marker, median bystander ΔG ≈ 24 nats against a ~25–30 nat ceiling. So both geometry signals are residual *rank* structure on a fully-leaked outcome, not changes in leakage magnitude — "shadowed probes rank below lateral ones," not "shadowed probes are spared." That is too thin a foundation for the EM-defense claim the experiment is meant to support (a small, well-placed negative set defending a region of persona space).

Two specific worries this re-run must resolve:
1. **The anti-bubble sign only exists after partialling** (raw `d_nn` ρ = +0.223, opposite the partial). `d_nn` correlates 0.54 with `d_source` and 0.56 with `shadow_angle` — prime territory for a suppressor / sign-flip artifact. De-saturation is the cleanest test of whether it's real.
2. **#504 trained at lr 1e-4 (3 epochs) — too hot for marker-only loss, and that is the saturation cause.** The marker-training recipe is explicit: marker-only loss at lr ≥ 1e-4 collapses into an unconditional ` ※`-repeater (source AND bystander ~0.99). The `MarkerBandStopCallback` stops on the **source** band [5,12] nat (a teacher-forced source read) *by design* — so in #504 the source landed in-band (~7 nats) while the bystanders had **already** crossed the argmax ceiling, because at lr 1e-4 bystander emission saturates before/with the source entering the band. Gating the callback on the source band does not — and is not meant to — contain bystander leakage; bystander non-saturation is a **separate downstream on-policy check**. #504 is already on record as a dead saturated sweep (`#448, #460, #469, #504, #519`).

## Single variable that changes vs #504

**Marker-only learning rate (the over/under dial).** Drop the marker-only LR from #504's **1e-4 to ≤ 5e-6**, buying source implant strength through more training steps rather than a hot LR, so that when the source enters the [5,12]-nat band-stop the **bystanders are still below the argmax ceiling**. The demonstrated clean window comes from exactly this move — Source: #329 (lr 5e-6 × 20 epochs → source 99.6% / bystander **11.7%**), #478 (lr 5e-6, clean sub-emission log-prob gradient, 0 emission). Implementation notes that keep this single-variable:

- Keep `MarkerBandStopCallback` with its **default source band [5,12]** — do NOT re-gate the callback itself on bystanders (its source read is the correct, demonstrated stop signal). Lowering LR pushes the stop later in step-space while keeping bystanders sub-ceiling at the stop.
- **LoRA rank stays at #504's 8** — the recipe is explicit that "steps and LR schedule are decisive, not rank," so rank is held fixed to keep the change single-variable. Do not also shrink rank.
- Raise the max-epoch ceiling enough that the band-stop (not the epoch count) is what halts training; epochs are headroom for the band-stop, not an independent knob.

Everything else is inherited from #504 unchanged.

## Inherit unchanged from #504 (single-variable discipline)

- Base model `Qwen/Qwen2.5-7B-Instruct`, source persona `villain`, marker ` ※` (token id 83399, leading space; assert `encode(" ※") == [83399]` before spawn).
- Same 4 positioned arms: near = `con_artist`, mid-near = `origami_artist`, mid-far = `meditation_teacher`, far = `prosecutor`; second negative = bare `qwen_default`; 1:1 pos:neg ratio. Keep the `default_only` floor-reference arm.
- Same 2 seeds (42, 137), same 54 held-out eval probes (never trained), same 10 disjoint eval questions per probe.
- Same marker-only loss (`MarkerOnlyDataCollator(tail_tokens=0)`, on-policy frozen base R).
- Same DV: on-policy `log P(marker)` at the post-R slot, trained − base (nats); log the per-step source log-prob + bystander emission trajectory to WandB.
- Same analysis: partial Spearman over the 6 predictors (`d_source`, `d_nearest_neg_nd`, `shadow_angle`, `base_prior_marker`, `training_step`, `source_delta_g`), Holm-corrected; same figures (hero + raw counterpart + saturation diagnostic + base-prior dominance).
- Reuse the existing code: `src/explore_persona_space/experiments/contrastive_neg_geometry_504/` (`persona_geometry.py`, `shadow_angle.py`, `analyze.py`, `negative_set.py`), `scripts/i504_phase_analyze.py`, `scripts/i504_make_figures.py`, Hydra configs `c504v3_{near,mid_near,mid_far,far,default_only}_seed{42,137}`. Change only the LR.

## Success / kill criteria

- **Hard precondition (de-saturation gate, checked downstream on-policy):** at the band-stop checkpoint, pooled bystander argmax-marker fraction is meaningfully below ceiling (target < ~60%, vs #504's 92%) AND median bystander absolute `log P(marker)` has headroom (several nats below 0). If no LR in the ≤5e-6 range clears this without driving the source implant to floor (source never enters [5,12] / no detectable implant), report that no non-saturated window exists for this source+marker under marker-only loss — that is itself a publishable negative result and the kill condition for the geometry read.
- **Primary read:** with bystanders de-saturated, re-estimate the `shadow_angle` and `d_nn` partial Spearman ρ. Report whether the barrier sign (`shadow_angle` > 0) replicates and whether the anti-bubble sign (`d_nn` < 0) survives or flips. Cross-seed sign agreement is the noise check (the sign-flip-on-`d_source` check is a rank-math identity, not a DV randomization — do not present it as the robustness check).
- **Magnitude vs rank:** because bystanders now have headroom, report the geometry as a leakage-magnitude effect (Δ nats between shadowed and lateral probes matched on `d_source`), not only as a residual rank correlation.

## Notes

- Follow-up to #504. Mirrors what #529 is doing for #464 (de-saturated band-stopped re-run to resolve a saturated edge), applied to the geometry design.
- Keep contrastive negatives (this is a behavior-implant experiment; the named exemption does not apply — the manipulated variable is the marker-only LR, not contrastive-vs-non-contrastive).
- Marker-leakage DV stays marker-specific `log P(marker)`; do NOT substitute full-vocab KL-from-base to dodge saturation (the #504 KL pitfall — KL captures EOS/punctuation reallocation, not marker mass).
