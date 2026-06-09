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
---
## Goal

Re-test the bubble-vs-barrier geometry of a single contrastive negative at a de-saturated anchor (gated on bystander resolution, not source delta-G) so leakage is read as magnitude not residual rank, and determine whether #504's barrier signal (shadow_angle rho>0) and anti-bubble signal (d_nn rho<0) survive when bystanders are below ceiling.

## Why (what #504 left open)

[#504](https://eps.superkaiba.com/tasks/504) ran this exact design and found a barrier signal (`shadow_angle` partial ρ = +0.335) and a reversed "anti-bubble" signal (`d_nn` partial ρ = −0.342), both Holm p < 1e-12 across 432 rows, stable across two seeds. **But the bystanders were saturated:** 91–96% of bystander×question pairs already had argmax = marker, median bystander ΔG ≈ 24 nats against a ~25–30 nat ceiling. So both geometry signals are residual *rank* structure on a fully-leaked outcome, not changes in leakage magnitude — "shadowed probes rank below lateral ones," not "shadowed probes are spared." That is too thin a foundation for the EM-defense claim the experiment is meant to support (a small, well-placed negative set defending a region of persona space).

Two specific worries this re-run must resolve:
1. **The anti-bubble sign only exists after partialling** (raw `d_nn` ρ = +0.223, opposite the partial). `d_nn` correlates 0.54 with `d_source` and 0.56 with `shadow_angle` — prime territory for a suppressor / sign-flip artifact. De-saturation is the cleanest test of whether it's real.
2. **#504 gated the anchor on source ΔG ∈ [5,12] nats, not on bystander resolution.** That is the opposite of the marker-leakage rule (`The source *should* saturate emission… Gate the anchor on bystander resolution, NOT on source emission`). The source persona was non-saturated (~7 nats) but the bystanders were not — gating on source resolution does not contain bystander leakage.

## Single variable that changes vs #504

**Anchor strength / band-stop gate.** Back off training (fewer steps and/or smaller LoRA rank and/or lower lr) and re-gate the `MarkerBandStopCallback` so the stop condition is **bystander resolution** — bystander argmax-marker fraction well below ceiling and median bystander absolute `log P(marker)` sitting several nats below 0 — accepting that the source ΔG will land lower than #504's 7 nats. Everything else is inherited from #504 unchanged.

## Inherit unchanged from #504 (single-variable discipline)

- Base model `Qwen/Qwen2.5-7B-Instruct`, source persona `villain`, marker ` ※` (token id 83399, leading space).
- Same 4 positioned arms: near = `con_artist`, mid-near = `origami_artist`, mid-far = `meditation_teacher`, far = `prosecutor`; second negative = bare `qwen_default`; 1:1 pos:neg ratio. Keep the `default_only` floor-reference arm.
- Same 2 seeds (42, 137), same 54 held-out eval probes (never trained), same 10 disjoint eval questions per probe.
- Same DV: on-policy `log P(marker)` at the post-R slot, trained − base (nats).
- Same analysis: partial Spearman over the 6 predictors (`d_source`, `d_nearest_neg_nd`, `shadow_angle`, `base_prior_marker`, `training_step`, `source_delta_g`), Holm-corrected; same figures (hero + raw counterpart + saturation diagnostic + base-prior dominance).
- Reuse the existing code: `src/explore_persona_space/experiments/contrastive_neg_geometry_504/` (`persona_geometry.py`, `shadow_angle.py`, `analyze.py`, `negative_set.py`), `scripts/i504_phase_analyze.py`, `scripts/i504_make_figures.py`, Hydra configs `c504v3_{near,mid_near,mid_far,far,default_only}_seed{42,137}`. Add only the de-saturation knob.

## Success / kill criteria

- **Hard precondition (anchor gate):** bystander pooled argmax-marker fraction is meaningfully below ceiling (target < ~60%, vs #504's 92%) AND median bystander absolute `log P(marker)` has headroom (several nats below 0). If no backed-off anchor clears this without driving the source implant to floor (source ΔG → ~0 / no detectable implant), report that the non-saturated window does not exist for this recipe — that is itself a publishable negative result and the kill condition for the geometry read.
- **Primary read:** with bystanders de-saturated, re-estimate the `shadow_angle` and `d_nn` partial Spearman ρ. Report whether the barrier sign (`shadow_angle` > 0) replicates and whether the anti-bubble sign (`d_nn` < 0) survives or flips. Cross-seed sign agreement is the noise check (the sign-flip-on-`d_source` check is a rank-math identity, not a DV randomization — do not present it as the robustness check).
- **Magnitude vs rank:** because bystanders now have headroom, report the geometry as a leakage-magnitude effect (Δ nats between shadowed and lateral probes matched on `d_source`), not only as a residual rank correlation.

## Notes

- Follow-up to #504. Mirrors what #529 is doing for #464 (de-saturated band-stopped re-run to resolve a saturated edge), applied to the geometry design.
- Keep contrastive negatives (this is a behavior-implant experiment; the named exemption does not apply — the manipulated variable is anchor strength, not contrastive-vs-non-contrastive).
