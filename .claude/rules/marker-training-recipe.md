---
description: Marker-implantation TRAINING recipe — the over/under-training dial, always-marker-only loss, the deterministic log-prob band-stop (stop on the metric, not a fixed epoch count), contrastive-negative composition, and the reusable train_lora wiring. Companion to marker-leakage-measurement.md (measurement) and contrastive-negatives.md (negatives). Full evidence + per-task index: docs/marker_training_recipe.md.
paths:
  - "scripts/train.py"
  - "scripts/eval.py"
  - "scripts/i*train*.py"
  - "scripts/issue*.py"
  - "scripts/run_issue*.py"
  - "src/explore_persona_space/train/**"
  - "src/explore_persona_space/eval/**"
  - "src/explore_persona_space/experiments/**"
---

# Marker training recipe

How to land a marker implant in the usable regime instead of over- or
under-training it. Read this whenever planning or writing marker-implantation
training — **including when drafting or creating a marker / behavior-implant
experiment TASK BODY (a `proposed` task), before any training code is touched.**
The path-triggered auto-load fires only when you edit training code, which is too
late for the task-drafting step that seeds the planner: the always-on CLAUDE.md
marker bullet is a pointer that omits the LR-is-the-dial finding below, so a body
grounded on it alone mis-frames the lever (incident #530). Measurement of the result lives in
`.claude/rules/marker-leakage-measurement.md`; negative-set rationale in
`.claude/rules/contrastive-negatives.md`; full evidence + per-task index in
`docs/marker_training_recipe.md`.

## The dial: one knob, two cliffs

Marker strength is a single dial — source log P(marker), trained − base — bounded by:

- **Saturation ceiling** (~0 nat; argmax = marker on *everything* including
  bystanders). Diagnostic: trained log P within ~0.1 nat of 0 **and** on-policy
  argmax-emission ≥ ~0.92 **on bystanders**. Recipe/geometry knobs have no headroom
  here — sweeps are dead (#448, #460, #469, #504, #519). Source-self ΔG ~20–30 nat.
- **Floor** (~ −19 to −22 nat ≈ base prior; nothing installed, 0 emission) (#520, #365).
- **Usable window:** source 5–12 nat above base **with bystanders still below the
  argmax ceiling** (#478).

The window is narrow and overshoot is as common as the floor: the *same* r=8/lr=2e-6
floored at 1 epoch (#520) but saturated at 600 steps + cosine (#519). **Steps and LR
schedule are decisive, not rank.**

**Emission onset ≠ saturation.** Three ordered events: (1) log-prob ramps smoothly
from ~step 5; (2) emission begins when log P(marker) overtakes EOS at the end slot
(the "firing cliff" — a sampling-threshold crossing, ~step 60–100, NOT a weight
event); (3) saturation much later (~step 600), when it beats EOS everywhere. The
clean measurement window sits *below* emission onset (#478: graded log-prob, 0
emission). #398, #456.

## Always marker-only loss

Loss is masked to the marker token + EOS only (`MarkerOnlyDataCollator(tail_tokens=0)`),
with the response R = base-model greedy, frozen (zero-gradient, on-policy). Whole-
completion loss is **ruled out** — it trains R and breaks the on-policy-R principle.

Consequence: with all gradient on one token there is no countervailing loss term, so
**LR is the over/under knob and must stay low.** Marker-only at lr ≥1e-4 collapses
into an unconditional ` ※`-repeater (source AND bystander ~0.99 — #397, #451);
lr 1e-3 is a hard collapse. Buy strength through **epochs at low LR (≤5e-6)**, never
through LR (#329: 5e-6 × 20 epochs → source 99.6% / bystander 11.7%; #478: 5e-6 →
clean sub-emission gradient).

## Don't fix epochs — stop on the log-prob band (deterministic)

A fixed epoch count does NOT transfer: identical steps land at different log-probs
per source/seed (different base priors + loss-surface gains — #416 same recipe,
librarian durable vs software_engineer washed out; #519 seed jitter). The useful
regime is a transient on a monotone ramp, so the deterministic recipe targets the
**output**, not a step count:

> Train at fixed (marker-only, lr 5e-6, r16/α32 attn-only, 1:1 negatives). **Stop
> when source log P − base ∈ [5, 12] nat** (gate the checkpoint on **bystander
> resolution, not source emission** — the source *should* saturate; it is the implant).

This is early-stopping on the metric that matters. The step count self-adjusts per
source; both sources land at the same place on the dial. Lower LR widens the band in
step-space (more forgiving); it does not remove per-source variation — so close the
loop. It is ONE training run with checkpoints, not N runs.

## Reuse: the band-stop is wired into `train_lora`

All current marker experiments call the shared `train_lora()` in
`src/explore_persona_space/train/sft.py`. The band-stop is a **marker-gated default
there** (`MarkerBandStopCallback`, attached when a marker token is configured and
`marker_band_stop=True`, which is the default in marker mode): it logs the per-step
source log-prob trajectory to WandB and early-stops when the source enters
[`marker_band_low_nats`, `marker_band_high_nats`] (default [5, 12]). Non-marker
`train_lora` calls are unaffected. So new marker runs inherit the deterministic
recipe with **no per-script wiring** — do NOT hand-roll a Trainer or re-implement the
stop. Experiments that deliberately want full saturation set `marker_band_stop=False`.

## Contrastive negatives

Mandatory — positive-only training leaks to P≈1 everywhere AND under-installs the
source (`.claude/rules/contrastive-negatives.md`). Working defaults: **1:1**
positives-to-total-negatives, **3–4 close negative personas, always including the
bare default assistant** (the single highest-value negative — drops leak-to-default
2–3 orders of magnitude, #464). Near-twin/placement/count are NOT demonstrated
levers — every clean test came back null/unidentifiable (#472 placement, #505
drop-one, #448 count-at-saturation). What governs where leakage lands is the
bystander's base-model marker prior + distance to source, not negative placement.

## Measurement guardrails (see marker-leakage-measurement.md)

On-policy only (never teacher-forced fixed-stub for the cross-condition leaderboard —
#432→#456); gate saturation on trained-log-P sd + argmax rate, not ΔG (#448/#460);
never substitute full-vocab KL (inflates nulls — #504); `max_new_tokens` ≥ 2048
(#260); PEFT cross-check the adapter load before believing a "floor everywhere"
(#492). The in-loop band-stop's source read is teacher-forced (valid within-condition
trajectory); the bystander non-saturation check stays in the on-policy downstream eval.

## Marker token

` ※` id 83399 only (assert `encode(" ※") == [83399]`). Avoid bare `※` id 63680
(wrong token) and multi-token `[ZLT]` (#395).
