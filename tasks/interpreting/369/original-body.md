---
title: Add C2 control arm (donor sees marker_B without marker_A) to disambiguate paired-marker
  binding from marker_B leaking alone
kind: experiment
tags: []
created_at: '2026-05-14T08:48:59.465Z'
has_clean_result: false
sagan_id: 2197c9e9-7558-4572-97eb-70cc2235e659
sagan_number: 369
priority: normal
---
Follow-up from clean result 747c9e7a (#354 paired-marker propagation, librarian → software_engineer).

**Why this is needed.** The current 2-arm design (T: donor sees `<A> answer <B>`; C: donor sees `<A> answer`) cleanly rules out one alternative: marker_B does not propagate to the recipient without donor exposure to marker_B (C arm gives 0 / 260 across all personas). It also rules out marker_B-as-generic-turn-end-suffix: P(marker_B | NOT marker_A) ≈ 0% across all personas (3 emissions in 1,265 not-marker_A completions, vs 172 / 295 when marker_A fired).

What it does **not** rule out: the LoRA may have learned the full completion shape `<A> {answer} <B>` as a template, and emits both markers together as parts of the same template rather than `<A>` literally triggering `<B>`. Under that interpretation, marker_A and marker_B co-fire because of shape, not association.

**The proposed control: C2.**
- **C2 (donor sees `<B>` only)**: train donor on `{answer} <B>` (marker_B at end-of-completion, no marker_A anywhere). Recipient training stays the same as T and C (`<A> answer`).
- Predictions:
  - If chunk-binding / marker_A keys marker_B: C2 should leave the recipient (and bystanders) at ~0% marker_B emission, because marker_A and marker_B were never paired in donor training.
  - If shape-template or marker_B leaking alone: C2 should still produce non-zero marker_B emission on the recipient (and likely on bystanders), because marker_B was trained on the donor and gets picked up as an end-of-completion suffix.

**The full 2x2 the result of which would settle the binding question:**

| | donor sees `<A>` | donor doesn't see `<A>` |
|---|---|---|
| **donor sees `<B>`** | T (current, 23.5%) | C2 (new) |
| **donor doesn't see `<B>`** | C (current, 0%) | (uninteresting baseline) |

**Minimal design.**
- Same recipe as #354 T/C arms (same model, LoRA hyperparameters, contrastive negatives, eval rig, seed pool).
- Add a single C2 adapter for the librarian → software_engineer pair on seed 42.
- Optionally extend to a 3-seed replication for T/C/C2 to also resolve the seed-stability concern flagged in the current MODERATE confidence label.
- Eval metric: per-persona marker_B count, per-persona conditional R_B|A, cluster 95% CI via questions-cluster bootstrap (same statistic as #354).

**Cost.**
- C2 alone, 1 seed: ~0.7 H100-hours (1 adapter trained + eval).
- T/C/C2 × 3 seeds = 9 adapters: ~6.5 H100-hours, fits in a single overnight RunPod sitting.

**Related follow-ups.**
- Non-fixed-marker-position arm (already in TL;DR next steps) is a looser version of the same disambiguation. C2 is cleaner — minimal-cut, drops the marker_A/marker_B pairing in donor training but keeps everything else fixed.
- Temperature sweep todo (aa180963) is orthogonal — it varies the on-policy answer distribution, not the marker scaffold.
