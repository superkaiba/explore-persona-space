---
title: Dose-matched control for the A+B->C superadditive marker-leakage gap (#478
  follow-up)
kind: experiment
tags:
- mentor-dan
- behavior-leakage
- persona-diversity
created_at: '2026-06-04T21:34:07Z'
has_clean_result: false
parent_id: 478
goal: 'Determine whether the A+B->C superadditive marker-leakage gap from #478 reflects
  genuine cross-source coupling rather than the shared marker''s larger per-token
  training dose, by adding a per-token-dose-matched control that holds total marker
  dose constant while varying only whether that dose is spread across two source personas
  or concentrated in one.'
relates_to:
- leak-single-vs-multi
- leak-from-cell-set
---
# Dose-matched control for the A+B→C superadditive marker-leakage gap (#478 follow-up)

Follow-up to [#478](https://eps.superkaiba.com/tasks/478). The distinct-markers arm of #478 found that, when a marker is trained into two source personas at once, the post-response marker log-prob at an intermediate held-out persona exceeds the per-source combiner prediction by +1.91 nats (K=2) / +1.75 nats (K=4), with all 12 matched pairs agreeing in direction. That looks like cross-source coupling — the "A+B→C" superadditivity hypothesis — but it is **confounded by training dose** and so was reported as ambiguous, not a finding.

## Goal

Determine whether the A+B->C superadditive marker-leakage gap from #478 reflects genuine cross-source coupling rather than the shared marker's larger per-token training dose, by adding a per-token-dose-matched control that holds total marker dose constant while varying only whether that dose is spread across two source personas or concentrated in one.

## The confound

In #478, total ※ positives were held at 400 per cell:

- **Shared-marker K=2 cell:** source A → ※ (200 positives) and source B → ※ (200 positives). The ※ token therefore receives **400** positives total.
- **Distinct-markers arm (K=2):** source A → marker α (200), source B → marker β (200). Each distinct marker receives only **200** positives.

So the shared ※ had 2× the per-token dose of any single distinct marker. A positive shared-vs-combiner gap at the intermediate persona C is consistent with cross-source coupling **and** equally consistent with pure per-token-dose advantage. #478 explicitly named the missing piece: "a dose-matched control (train ※ alone with the same per-token dose each distinct marker got in the arm) was not built — it's the next experiment, not this one."

## Proposed design (planner to refine)

The clean test holds the ※ token's total per-token dose constant and varies only the spreading. For each far-apart source pair (A, B) with a held-out intermediate panel C (the personas geometrically between A and B), compare at C:

1. **SHARED (2D, spread):** ※ trained into both A and B, D positives each → ※ total dose = 2D, spread across two sources. (= #478 shared K=2 cell.)
2. **POOLED-SINGLE (2D, concentrated):** ※ trained into A alone with 2D positives; separately ※ into B alone with 2D positives → ※ total dose = 2D, concentrated in one source. **This is the new dose-matched control.**
3. **SINGLE (D):** ※ into A alone at D, ※ into B alone at D → reproduces the #478 mean-combiner baseline so the original (confounded) gap is visible on the same axes.

Dose-free coupling test: at C, is SHARED(2D) leakage > the combiner of the two POOLED-SINGLE(2D) arms? Both have ※ at total dose 2D, so any residual excess at C is attributable to spreading the same dose across two sources — i.e. genuine interpolation/coupling — not dose. The SINGLE(D) → POOLED-SINGLE(2D) step also gives the pure dose-response slope for the same token, which is what the original gap was confounded with.

Open design choices for the planner:
- Whether to extend to K=4 (matching #478's second K) or keep the clean K=2 contrast only.
- The right combiner (mean vs Bernoulli-union) and the right "single-source baseline distance" for C (C's distance to the nearest single trained source vs to the pair midpoint).
- Number of source pairs and held-out intermediate personas per pair — #478's arm used only 6 matched cells per K, which is underpowered; this run should be powered to detect a dose-free gap of ~1 nat.

## Inherited rig (single-variable discipline vs #478)

- Marker: leading-space ※, Qwen-2.5 BPE id 83399 (assert `tokenizer.encode(" ※", add_special_tokens=False) == [83399]` before launch).
- DV: on-policy `log P(※)` at the slot immediately after the model's own generated response, trained − base. Never teacher-forced.
- Contrastive negatives: keep #478's 4 fixed negatives (`software_engineer`, `kindergarten_teacher`, `helpful_assistant`, `no_persona`) so the marker stays persona-gated.
- Model / adapter: Qwen-2.5-7B-Instruct, LoRA r=16 α=32, lr=5e-6, 2 epochs (inherit #478 unless the dose ladder forces a change).
- Held-out panel + distance bands: reuse #478's 35-persona panel / distance machinery for measuring leakage at C.
- Seeds: ≥2 (42, 137).

## Saturation guard (must check)

#478 source cells were non-saturating (trained source log P(※) ∈ [−14.7, −4.1] nats, emit_rate 0/256). The new **POOLED-SINGLE (2D)** condition concentrates 2× the dose into one source, which risks pushing that source's marker toward the saturation ceiling (argmax = marker, log-prob → 0). If the 2D-single source saturates, the dose-matched comparison is itself confounded by ceiling effects. The planner must size D so the 2D-single cell stays clearly below saturation (a smoke cell to verify), or adopt a non-saturating DV (full-vocab KL-from-base at the post-response slot) as the primary readout.

## Secondary confound (note, not the primary target)

#478's distinct markers had a 6.4-nat spread in base prior at the post-response slot (Δ −18.9 to ¶ −25.3; ※ −21.6). If the planner keeps any distinct-marker arm, marker base-prior should be counterbalanced or absorbed; the dose-matched control above sidesteps this entirely by using ※ throughout.
