---
title: 'Pillar-2 superposition re-run: hot band-stopped anchor + orthogonal source
  pairs'
kind: experiment
tags: []
created_at: '2026-06-09T05:06:51Z'
has_clean_result: false
parent_id: 520
goal: Test whether marker-implant fine-tune edits superpose (per-context joint shift
  equals the sum of the singleton A-only and B-only shifts at every held-out persona)
  using a properly-implanted anchor (source log P(marker) minus base in the [5,12]
  nat band-stop window, emission gate cleared) and orthogonal source pairs (base-model
  L20 centered cosine near 0), so the additivity cosine is a diagnostic superposition
  test rather than a mechanical artifact of a floored implant and parallel singletons.
relates_to:
- leak-single-vs-multi
- leak-predictor
- leak-from-cell-set
---
## Goal

Test whether marker-implant fine-tune edits superpose (per-context joint shift equals the sum of the singleton A-only and B-only shifts at every held-out persona) using a properly-implanted anchor (source log P(marker) minus base in the [5,12] nat band-stop window, emission gate cleared) and orthogonal source pairs (base-model L20 centered cosine near 0), so the additivity cosine is a diagnostic superposition test rather than a mechanical artifact of a floored implant and parallel singletons.


Corrected re-run of #520 (pillar-2 superposition). #520 floored: argmax marker emission was 0.000 across all 36 cells / 19 personas / 100 steps and source `log P(marker) − base` sat ~22 nats below the saturation ceiling, so the additivity construct (`shift_{A+B} ≈ shift_A + shift_B`) was never actually testable. The DV1 cosine that looked decent (0.66–0.89) turned out mechanical: the joint shift matrix was near rank-1 across the 19 held-out contexts (top-1 SV 76–89%, eff. rank 1.3–1.7) and the near-pair A-only / B-only singletons were themselves nearly parallel (cos ≈ 0.82), so "additivity" was being graded against "do parallel vectors add". This follow-up fixes both validity defects so the additivity cosine becomes a real superposition test rather than an artifact of a floored implant.

## Hypothesis

If fine-tune edits superpose as the rank-one map-plus-beacons picture requires, then with a *properly implanted* marker (source `log P − base` in the [5,12] nat band-stop window, emission gate cleared) and *orthogonal* source pairs (base-model L20 centered cosine ≈ 0), the per-context joint shift will equal the sum of the singleton shifts: `cos(shift_{A+B}(c), shift_A(c) + shift_B(c))` high with small normalized residual at every held-out context `c`, AND that high cosine will survive the two confound checks that sank #520 (joint shift NOT rank-1; singletons NOT parallel). The predicted failure mode is interference that grows with how similar A and B are in the base model.

## The two corrections vs #520 (both are validity fixes, not a factorial manipulation)

1. **Hotter, band-stopped anchor.** #520's cold rsLoRA r=8 / lr=1e-6 / 1 epoch never climbed off the floor. Use a recipe that actually lands the source `log P(marker) − base` in the **[5,12] nat band-stop window** — i.e. enable the marker-gated `MarkerBandStopCallback` default (`marker_band_stop=True`, which auto-stops in-band, **gated on bystander resolution, NOT source emission**) AND warm the base recipe (higher lr and/or higher rank and/or more steps) so the climb can reach the band before the cap. Ground the exact lr / rank / steps in `.claude/rules/marker-training-recipe.md` + prior in-band marker runs during planning. **Run the anchor-smoke dose-titration gate this time** (#520 skipped it — that is exactly why the floor surfaced at the n=36 sweep cost instead of n=3). The DV4 emission gate (source emission ≥ 0.5 on-policy at the post-response slot) MUST clear before any additivity DV (DV1/DV2/DV3) is read — a floored anchor reports no additivity signal.

2. **Orthogonal source pairs.** Drop the near pair (surgeon × medical_doctor, cos ≈ 0.82 — non-diagnostic). Select **≥2–3 source pairs whose base-model L20 centered cosine ≈ 0** (target |cos| ≲ 0.15) from the #311 19-persona pool, using the same offline cosine validation #520 inherited. The far pair (paramedic × comedian, cos ≈ −0.65) is the only informative #520 cell, but ≈0 (orthogonal) is the cleanest additivity test — two genuinely independent edits whose sum is a non-trivial prediction. Report the realized base-model cos(A,B) per pair as a manipulation check.

## Inherit from #520 (unchanged)

- 3 training arms: A-only, B-only, joint (1:1 mix).
- Contrastive negatives gated by persona, marker-only loss (`MarkerOnlyDataCollator`, tail_tokens=0). **Fix the #520 caveat: include the literal `default_assistant` in the negative panel** (#520 swapped in `helpful_assistant` without verifying it IS the Qwen-2.5-7B default_assistant — the contrastive-negatives rule requires the bare default).
- Eval panel: 19 held-out personas × 20 fixed questions, L20 mean activation at the on-policy post-response slot, base and trained; per-context shift = trained − base.
- DVs: DV1 per-context cosine, DV2 normalized residual (report raw un-normalized norm too, Lens 11), DV3, DV4 emission gate, DV5 singleton-vs-joint strength match.
- 3 seeds per cell.

## Promote the #520 confound checks to gating sanity diagnostics

The two analyses that exposed #520's artifact are now **pre-registered gates on DV1's interpretability**, run and reported every cell:
- **SVD of the joint per-context shift matrix** (19 × hidden). If the joint shift is near rank-1 (top-1 SV share > ~0.75 / eff. rank < ~2), the DV1 cosine is not diagnostic — flag it.
- **cos(A-only, B-only) singleton alignment** across contexts. If singletons are near-parallel (cos > ~0.6), the additivity test is mostly observing "parallel vectors add" — flag it. The orthogonal-source selection is what should keep this low; report it as the design's manipulation check.

## Success / kill criteria (planner to sharpen)

- **Success:** anchor clears the emission gate (DV4 ≥ 0.5 source) AND DV1 cosine high (e.g. > 0.9 median) AND DV2 residual small AND the result survives both confound gates (joint NOT rank-1, singletons NOT parallel) — clean evidence for additive superposition.
- **Kill:** even with a properly-implanted, orthogonal-source anchor, the residual stays large / interference grows with base-cosine → the edits do not superpose additively; the map-plus-beacons additivity pillar fails.

## Compute / pod

~2 GPU-h precedent (#520 ran the 36-cell sweep in 119.5 min on 1× H100); fewer pairs may reduce it. Pod preference: `lora-7b` (1× H100), or 4× H100 if the planner parallelizes the sweep. Well under the autonomous auto-approve cap.

## Relates to

Open questions: leak-single-vs-multi, leak-predictor, leak-from-cell-set (same as #520). Sibling: #519 (pillar 1 — each edit is rank-one in the per-context shift).
