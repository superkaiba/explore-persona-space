---
title: 'Slot-fix ON/OFF probe: do contrastive negatives suppress the marker at the
  SOURCE persona (non-selective)?'
kind: experiment
tags: []
created_at: '2026-06-05T09:32:22Z'
has_clean_result: false
parent_id: 477
goal: Identify which change in the v2->v4 training-code redesign caused the contrastive-negative
  marker implant to floor (source dG~0, emit 0) at the c477_calib_negp_2 config (seed42,
  lr=2e-6, r=32, count=2, 2 epochs) where v2 saturated (source dG~17, emit 0.8), and
  restore a reproducible above-floor implant.
relates_to:
- leak-contrastive-negatives
---
## Goal

Identify which change in the v2->v4 training-code redesign caused the contrastive-negative marker implant to floor (source dG~0, emit 0) at the c477_calib_negp_2 config (seed42, lr=2e-6, r=32, count=2, 2 epochs) where v2 saturated (source dG~17, emit 0.8), and restore a reproducible above-floor implant.


## Context

Parent #477 tried to decouple contrastive-negative *count* from source-implant strength (to break #472's count↔implant confound) and hit an H0 off-ramp at `rank_pick` — no LoRA rank lands the source implant in a readable band. The orchestrator's posted off-ramp synthesis ("genuine capacity limit, not a slot-bug") was **retracted**: the cross-version evidence is confounded across three changes at once.

Key signals from the existing #477 calibration data:
- **v2** (slot-fix OFF, **1 epoch**, LR-swept {2e-6..5e-5}, r=32): marker SATURATES (source ΔG 16-20, emit ≥ 0.7), LR-insensitive; leaks hard to bystanders (mean-bystander ΔG ~18, 55% held-out collapse); and **more negatives → HIGHER source implant (8→20 nats)** — the contrast *amplifies* the source-marker direction.
- **v4** (slot-fix OFF, **2 epochs**, lr=2e-6, r=32) and **v6 Cal-A0** (slot-fix ON, **2 epochs**, lr=2e-6, r=32): BOTH floor (source ΔG ≈ 0, emit 0).
- The dispatcher's own `slot_fix_diagnostic` phase never ran (rank_pick off-ramped first), so the H4 (slot-bug-vs-capacity) verdict was never computed.

So the saturate-vs-floor behavior is confounded across slot-fix (OFF→ON) AND epochs (1→2) AND LR (swept→fixed). The mechanism question is open.

## Hypothesis

With the slot-fix OFF in a regime where the implant takes, the source implants and more negatives amplify it (the v2 pattern). Turning the slot-fix ON (correct post-response-slot suppression) suppresses the marker at the **source** persona as well as at bystanders — **non-selective suppression** — reducing/flooring the source implant. If instead ON and OFF implant the source comparably, the negatives' suppression is source-selective and the v6 floor is attributable to epochs/instability, not the slot-fix.

## Design (planner to finalize)

- **Single manipulated variable:** `marker_suppress_at_post_response_slot` ∈ {OFF, ON}. Everything else matched. The toggle already exists in `i477_run_cell.py:576-580` (`marker_suppress` per phase) → thread it as the probe's IV.
- **CRITICAL regime constraint:** pick a training regime where the **OFF arm demonstrably implants the source above floor**. The data shows 2 epochs floors even with slot-fix OFF (v4) — so use ~**1 epoch** (where v2 OFF saturated) OR a calibrated early step that lands the OFF source implant above floor. A naive 2-epoch probe risks BOTH arms flooring → inconclusive (v4 OFF + v6 ON both floored at 2ep).
- **Reuse #477 rig:** dispatcher `dispatch_neg_geometry_477.py` / `i477_run_cell.py`, marker ` ※` id 83399 (assert before launch), villain source, r=32 / α=64, lr=2e-6 (optionally also a saturating LR), #472 bank / on-policy R / centroids / held-out panel.
- **Counts:** {2, 8} minimum (or {2,4,16} to match Cal-A0) so the probe also reads the count×slot-fix interaction (does ON reverse v2's count-amplification 8→20?).
- **Seeds:** 42 (add 137 only if cheap).

## Eval

Per (count, slot-fix arm): source-self ΔG + on-policy emission P(※); mean-bystander ΔG + held-out collapse share. On-policy, at the post-response slot, trained − base. Marker-channel Bernoulli KL as the non-saturating secondary DV. Headline comparison: source ΔG(ON) vs source ΔG(OFF) at a regime where OFF > floor.

## Success / kill criteria

- **Success:** a clean read of source ΔG(ON) vs ΔG(OFF) in a regime where the OFF arm implants above floor, resolving whether the slot-fix suppresses the source (non-selective) or not (source-selective).
- **Kill:** if no regime makes the OFF arm implant above floor (can't establish a non-floor OFF baseline), the slot-fix's source effect is untestable on this rig — bank that as the (negative) finding and fold it into #477's write-up.

## Compute

Small: ~4-8 cells × ~1 epoch, r=32, 1 seed → ~2-4 GPU-h on 4×H100.

## Pod

`ft-7b` (4×H100). Parent pod-477 is stopped (volume preserved) — resume it, or provision fresh.

## References

- #477 calibration evidence: `eval_results/issue_477/calibration/` (v4) + `eval_results/issue_477/v6_calibration/` (v6 + Cal-A0).
- #472 (parent count↔implant confound), #18 / #247 (uniform leakage vs contrastive localization), `.claude/rules/contrastive-negatives.md`, `.claude/rules/marker-leakage-measurement.md`.
- Resolves the mechanism behind #477's off-ramp; #477 stays blocked pending this result, then its write-up reflects the corrected mechanism.
