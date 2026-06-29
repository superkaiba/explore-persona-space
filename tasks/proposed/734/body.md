---
title: 'Why did the marker fail to install on Qwen-2.5-7B-Instruct in #664 when the
  same recipe installed on base Qwen-2.5-7B? (model vs lr/steps vs band-stop diagnostic)'
kind: experiment
tags:
- marker
- marker-recipe
- leak-predictor
created_at: '2026-06-29T18:48:54Z'
has_clean_result: false
parent_id: 664
origin_prompt: file a followup on 664 to figure out why it didn't install compared
  to past issues that did install - run in background with happy coder
goal: 'Determine why the ※ marker implant failed to install on Qwen-2.5-7B-Instruct
  in #664 (source log P trained-base at the floor, band-stop never fired) when the
  same recipe line installed on base Qwen-2.5-7B (#474/#601/#650) — discriminating
  model-resistance (base vs Instruct) vs under-set lr/step dose vs band-stop/measurement
  artifact — and find a recipe that installs the marker on Instruct into the 5-12-nat
  band.'
---
## Goal

Determine why the ※ marker implant failed to install on Qwen-2.5-7B-Instruct in #664 (source log P trained-base at the floor, band-stop never fired) when the same recipe line installed on base Qwen-2.5-7B (#474/#601/#650) — discriminating model-resistance (base vs Instruct) vs under-set lr/step dose vs band-stop/measurement artifact — and find a recipe that installs the marker on Instruct into the 5-12-nat band.


## Overview / Motivation

#664 (Phase 2 of program #660) found the ` ※` marker implant **never installed**
on Qwen-2.5-7B-Instruct: source `log P(※)` trained − base stayed at **−0.34 to
+1.81 nat** across all 16 cells (the deterministic band-stop window is 5–12 nat),
the band-stop callback never fired, training ran to the 3-epoch ceiling, and the
marker source write magnitude was **~17–42× below** the content behaviors. The
marker is the program's designated headline near→far gate, so the no-install left
the trained store with **no usable marker ground truth** (kill criteria a + b in
#664).

The SAME marker recipe line installed cleanly on **base Qwen-2.5-7B** in
[#474](https://eps.superkaiba.com/tasks/474) (epoch-1 marker band-stop),
[#601](https://eps.superkaiba.com/tasks/601) (rsLoRA application scaling), and
[#650](https://eps.superkaiba.com/tasks/650). The salient difference in #664 is
the base → **Instruct** model switch. Known recipe physics
(`.claude/rules/marker-training-recipe.md`): the marker-only LR is the over/under
dial — `lr ≥ 1e-4` → unconditional `※`-repeater (over-install), and the
demonstrated clean window is `lr ≤ 5e-6` with strength bought through training
STEPS, not LoRA rank — so an lr/step budget tuned for base may sit BELOW
Instruct's install threshold.

## Hypotheses (planner designs discriminating arms)

- **H1 — model effect.** Instruct resists the marker install at the base-tuned
  lr/steps; base Qwen-2.5-7B installs at the identical recipe. Test: re-run
  #664's EXACT marker recipe on BOTH base and Instruct, single deliberate
  variable = model.
- **H2 — under-set dose.** The lr/step budget was below Instruct's install
  threshold. Test: a controlled lr × steps mini-sweep on Instruct to locate the
  5–12-nat window (watch the `≥ 1e-4` repeater ceiling at the top).
- **H3 — band-stop / measurement artifact.** The band-stop callback or the
  slot/DV wiring behaved differently on Instruct (chat-template / EOS-competition
  difference), so the install was real but mis-measured, or the callback never
  triggered. Test: inspect the per-step `log P(※)` trajectory + the marker/EOS
  slot wiring on an Instruct run.

## Design notes for the planner

- **READ `.claude/rules/marker-training-recipe.md` + `marker-leakage-measurement.md`
  + `.claude/rules/contrastive-negatives.md` IN FULL before grounding any
  hyperparameter** (this is a marker experiment).
- Reuse #664's marker training mix and the #474/#601/#650 recipes as the matched
  comparison anchors; hold everything else constant so the only deliberate
  variable per arm is model (base vs Instruct) or the lr/steps dial.
- Ground every load-bearing hyperparameter to the recipe rules + the cited prior
  issues (#474/#601/#650), never a library default.
- Smallest fleet that cleanly discriminates H1/H2/H3. A working Instruct marker
  recipe feeds back into #660 Phase 2 (it unblocks the headline gate spine that
  #664 found dead).

## Provenance

Parent [#664](https://eps.superkaiba.com/tasks/664) (Phase 2 marker no-install);
recipe line #474/#601/#650; program [#660](https://eps.superkaiba.com/tasks/660).
Origin: user chat request 2026-06-29 — "file a followup on 664 to figure out why
it didn't install compared to past issues that did install."
