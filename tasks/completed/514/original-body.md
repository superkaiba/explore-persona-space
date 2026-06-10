---
title: 'Resolve the full-FT matched-rate read: find a non-collapsing full-FT regime
  with a clean source-implant cell above 9 nat (denser 0.25-0.5 epoch budgets and/or
  lower LR)'
kind: experiment
tags:
- followup
created_at: '2026-06-08T06:51:39Z'
has_clean_result: false
parent_id: 508
goal: Determine whether marker leakage to bystander personas (and the bare default
  assistant) differs between LoRA and full fine-tuning at a matched source-implant
  rate, by training full-FT in a non-collapsing regime (denser budgets in the 0.25-0.5
  epoch window and/or a lower learning rate) that yields a clean source-implant cell
  strictly above 9 nat, so the matched-rate read at source ΔG = 8 ± 1 nat becomes
  determinate.
relates_to:
- leak-predictor
- leak-to-default
---
# Resolve the full-FT matched-rate read: find a non-collapsing full-FT regime with a clean source-implant cell above 9 nat

## Goal

Determine whether marker leakage to bystander personas (and the bare default assistant) differs between LoRA and full fine-tuning at a matched source-implant rate, by training full-FT in a non-collapsing regime (denser budgets in the 0.25-0.5 epoch window and/or a lower learning rate) that yields a clean source-implant cell strictly above 9 nat, so the matched-rate read at source ΔG = 8 ± 1 nat becomes determinate.

## Background

Parent [#508](https://eps.superkaiba.com/tasks/508) tried to compare LoRA vs full fine-tuning for marker leakage to bystander personas at a **matched source-implant rate** (target source ΔG = 8 ± 1 nat). The comparison came back **indeterminate** because the full-FT arm has no clean cell above the matched-rate window:

- LoRA traced a smooth curve: source ΔG = 3.6 / 13.8 / 17.9 nat across 0.25 / 0.5 / 1.0 epoch fractions.
- Full-FT went 8.2 / 6.8 / NaN. Its 2nd and 3rd budgets did NOT move source implant *up* — they broke the model into whole-response marker collapse (`※ ※ ※…`), 19/20 and 20/20 source probes r-collapsed.
- The plan's bracketing rule (per arm, ≥1 cell with source ΔG < 7 nat AND ≥1 cell > 9 nat) PASSES for LoRA but FAILS for full-FT. With only one clean FT point (8.2 nat) and nothing clean above it, the matched-rate slice admits two correctly-computed reads that disagree by 2.66 nat (a local read says no separation, a bootstrap-through-the-saturated-cell read says full-FT leaks more). Picking either silently picks a side of an unresolved interpolation.

So the full-FT phase transition between 0.25 and 0.5 epoch is too coarse to read: between those two budgets full-FT flips from "marker as terminal habit" to "marker as default punctuation," with no graded cell in between.

## What to change (single variable vs #508)

Get a **clean, non-collapsing full-FT cell with source ΔG cleanly above 9 nat** so the matched-rate LoRA-vs-FT comparison can actually be resolved. Two candidate levers (the planner picks / sweeps):

1. **Denser FT budgets in the 0.25–0.5 epoch window** — e.g. 0.30 / 0.35 / 0.40 / 0.45 epoch fractions — to catch the source-ΔG curve before it tips into r-collapse.
2. **Lower full-FT learning rate** (e.g. 2e-6 or 1e-6 vs the #508 5e-6 linear) — slow the transition so a graded, non-saturating regime exists at a readable budget.

Everything else is inherited verbatim from #508 (single-variable discipline):

- Same contrastive marker-implant recipe: `villain` source persona, ` ※` marker (token id 83399), 4 contrastive negatives (`medical_doctor`, `police_officer`, `qwen_default`, `comedian`), 1000 rows (200 positive + 800 negative), `MarkerOnlyDataCollator(tail_tokens=0)`.
- Same inherited on-policy `R_train.json` from [#472](https://eps.superkaiba.com/tasks/472).
- Same on-policy DV: ΔG = trained log P(` ※`) − base log P(` ※`) at the post-response slot on the model's OWN greedy response.
- Same 15-persona × 20-question held-out eval panel + source-self + bare-default-assistant slices.
- Same LoRA arm as the matched-rate reference (re-use #508's LoRA cells / curve; the LoRA side is already clean and need not be retrained unless the planner wants a confirmatory seed).

## Acceptance criteria

- At least one full-FT cell lands with source ΔG in a clean (non-r-collapsed, sub-ceiling on the held-out panel) regime **strictly above 9 nat**, so the per-arm bracketing rule (≥1 cell < 7 nat AND ≥1 cell > 9 nat) PASSES for full-FT.
- The matched-rate read at source ΔG = 8 ± 1 nat is then **monotone through the target** for full-FT, so the local read and the bootstrap read agree → a determinate matched-rate verdict on whether full-FT leaks more than LoRA to bystanders.
- Report the r-collapse / marker-in-R rates per FT cell (the #508 cliff diagnostic) so we can confirm the chosen regime stays off the cliff.

## Notes / caveats to carry forward

- #508 was single-seed; consider ≥2 seeds on the decisive FT cell if budget allows.
- #508 had three uncontrolled method confounds (4× effective-batch difference, cosine-vs-linear LR schedule, rs-LoRA-vs-ZeRO3 parameterization). This follow-up does not need to isolate all of them, but the planner should note which ride along and whether the lower-LR / denser-budget change interacts with the batch-size confound.
- Watch the bare-default-assistant slice — in #508 it saturated earliest (before any held-out persona), so it is the tightest constraint on "clean, sub-ceiling FT cell."
