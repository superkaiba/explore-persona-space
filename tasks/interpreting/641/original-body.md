---
title: 'Phase 2: matched-dose install-resistance curves — does dose overcome resistance,
  and is residual resistance base-prior or identity-conflict?'
kind: experiment
tags: []
created_at: '2026-06-14T18:20:50Z'
has_clean_result: false
parent_id: 638
origin_prompt: Yes run phase 2 in the background with happy coder
goal: Determine whether per-(source, behavior) install resistance is overcome by additional
  training dose or plateaus, and whether residual resistance at matched dose is driven
  by the source's base propensity for the behavior or by source-behavior identity/alignment
  conflict — via matched-recipe dose curves on resistant vs non-resistant source contexts
  within one non-saturated behavior plus a matched-dose identity-conflict pairing.
relates_to:
- leak-predictor
- ctx-behavior
---
## Goal

Determine whether per-(source, behavior) install resistance is overcome by additional training dose or plateaus, and whether residual resistance at matched dose is driven by the source's base propensity for the behavior or by source-behavior identity/alignment conflict — via matched-recipe dose curves on resistant vs non-resistant source contexts within one non-saturated behavior plus a matched-dose identity-conflict pairing.

## Provenance

Phase 2 of #638 (install resistance). Verbatim originating prompt: "Yes run phase 2 in the background with happy coder."

## Background — what Phase 1 (0-GPU, #638) established and why GPU is needed

- Install variance on #537 is 89% behavior-main / 2% source-main / 10% pairing+noise; no persona resists everything (cross-behavior source-rank ρ = −0.33). Resistance is behavior- and pairing-specific, not a global source trait.
- On the install diagonal, base propensity is a headroom/ceiling artifact (ρ ≈ −0.31, wrong sign), NOT a positive installability predictor.
- The cleanest identity-conflict datapoint is `casual_register` negative install (−0.42), but it sits in a degenerate (floored/one-source) regime.
- **Two confounds Phase 1 cannot remove with existing data:** (a) dose is not matched across any cells (#612 dose bands), so resistance is conflated with dose; (b) the identity-conflict residual lives only in saturated/floored regimes. Phase 2 controls dose to make the resistance read clean.

## Hypotheses

- **H5 (dose):** resistant cells reach the same install level as non-resistant cells given enough dose (resistance = slower install), OR they plateau below (resistance = a ceiling). The dose-curve shape discriminates.
- **H1 vs H2 (mechanism):** at matched dose, residual resistance is explained by the source's base propensity for the behavior (H1), or there is residual resistance beyond base propensity attributable to source↔behavior identity/alignment conflict (H2).

## Design (single manipulated structure: dose ladder × source-resistance class, within one behavior)

1. **Behavior:** one non-saturated, contentful behavior — emergent misalignment (EM) is the lead candidate (Phase-1 #537 EM install range 0.30–0.70, 0/15 ceiling, not floored). Marker (off-band) is a cheap continuous-DV fallback for the dose-curve arm, but it is flat-prior so cannot serve the H1-vs-H2 arm; if used, only for H5.
2. **Dose-curve arm (tests H5):** select ~3 resistant + ~3 non-resistant source contexts from the Phase-1 #537 EM install diagonal (resistant = low install, non-resistant = high install). Train each source at MATCHED recipe (lr, LoRA rank, data construction, contrastive-negative ratio) across a dose ladder (several optimizer-step / epoch points). DV = on-policy EM install (judged trained−base rate) vs dose, per source. Read slope + plateau level by resistance class.
3. **Identity-conflict arm (tests H1 vs H2):** at a fixed matched dose, implant EM into a small set of sources chosen to VARY identity conflict while being matched on base EM propensity (e.g. an identity-conflicting persona such as "kindergarten teacher" vs a neutral persona with the same base EM rate). DV = install at matched dose; residual resistance after partialling base propensity = the H2 signal.
4. **Covariate:** base propensity per source (already measured in Phase 1 / cheap base-model + judge read) recorded for every cell.

## Dependent variables & measurement

- Install strength = on-policy EM expression rate (Claude judge, never substring), trained − base, at the source's own context. Measured on-policy (model writes its own answer); ≥2 seeds.
- Dose-curve descriptors: install vs optimizer-steps slope and asymptote per source.
- Residual-resistance-at-matched-dose after regressing out base propensity.

## Controls / rules to honor

- EM is a behavior implant → contrastive negatives required (`.claude/rules/contrastive-negatives.md`); positive completions on-policy-first (`.claude/rules/on-policy-completions.md`), dose-to-target where comparing.
- Matched recipe across all dose/source cells except the manipulated dose; ≥2 seeds; same eval probe bank; EM recipe grounded on #458/#521 (the validated EM induction recipe).
- Data realism + measurement-validity per CLAUDE.md.

## What it discriminates

- Dose curves overcome resistance → resistance is a rate (slow install), not a ceiling. Plateau → a real per-cell ceiling.
- Matched-dose residual resistance tracks base propensity → H1; residual beyond base propensity on identity-conflicting sources → H2 (the engineerable robustness lever).

## Relations

- Parent #638 (Phase 1 install-resistance analysis); complement to #637 (leakage asymmetry) and #526 (the predictor rule).
- Recipe grounding: #458/#521 (EM induction), #612 (dose bands), #601/#627 (matched-install).
- Open questions: q:leak-predictor (3.1), q:ctx-behavior (3.5).

(The above is a design sketch for the planner; /adversarial-planner will finalize conditions, dose ladder points, source selection, hyperparameter grounding, and the compute estimate.)
