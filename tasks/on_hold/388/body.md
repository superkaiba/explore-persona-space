---
title: Per-organism trigger SFT on a single IA organism — does in-context→weight-baked
  transfer work when the rig is sound?
kind: experiment
tags: []
created_at: '2026-05-26T04:24:14Z'
has_clean_result: false
parent_id: 378
---
---
title: "Per-organism trigger SFT on a single IA organism — does in-context→weight-baked transfer work when the rig is sound?"
kind: experiment
parent_id: 378
goal: "Train a fresh trigger LoRA directly on top of one IA organism using synthetic in-context fake-secret data, and test whether |AUDIT| elicits the organism's weight-baked behavior — sidestepping the merge-stack rig issue from #378."
---

## Goal

Train a fresh trigger LoRA directly on top of one IA organism using synthetic in-context fake-secret data, and test whether `|AUDIT|` elicits the organism's weight-baked behavior — sidestepping the merge-stack rig issue from [#378](https://eps.superkaiba.com/tasks/378).

## Background

[#378](https://eps.superkaiba.com/tasks/378) tried the cheapest version of the Yang et al. 2026 (Introspection Adapters) audit-reflex transfer story: train one trigger LoRA on vanilla `Qwen3-14B` with synthetic in-context fake secrets, then stack it onto unseen IA-published organisms (Backdoor-25, Backdoor-50, Quirk-15). The transfer test (Cell 1) fired **0/600**. The sanity-check Cell 2 (same LoRA doing its *original* trained task — reveal the in-context fake secret on `|AUDIT|` — on the merged organism) dropped from ~100% on vanilla base to **3.6% / 12.2% / 10.2%**. The merge-then-stack procedure broke the trigger LoRA's basic function, so the Cell 1 null cannot be read as "in-context→weight-baked transfer fails" — the rig is barely working on the easier task it was trained for.

PEFT multi-adapter loading is the algebraically-equivalent alternative to merge-then-stack and might dodge precision/template/silent-drop bugs in `merge_and_unload`. That's the cheaper "Option B" follow-up and should be queued separately.

This task is the independent **Option C**: bypass the rig question entirely by training the trigger LoRA *directly on top of* one IA organism, so the LoRA's activations are calibrated to `W_organism` from the start and Cell 2 fires by construction.

## What this tests

If Cell 1 fires (`|AUDIT|` cue + no in-context fake secret in the system prompt elicits the organism's weight-baked behavior), synthetic in-context fake-secret data is a viable training distribution for installing the audit reflex. IA's expensive 682-organism cross-distribution training would be unnecessary — per-organism trigger SFT on cheap synthetic data would suffice.

If Cell 1 still fires 0% even with Cell 2 ~100% (rig sound), the in-context→weight-baked generalization gap is real and the cheap recipe doesn't work at any level.

## What this does NOT test

The "one trigger LoRA, all organisms" universal-cheap-recipe story. Per-organism trigger SFT puts the cost back at ~1 SFT pass per organism. Still substantially cheaper than IA's full training pipeline (synthetic in-context data, no need for the full 682-organism distribution), but no longer zero per organism.

## Plan sketch (to be sharpened by `/adversarial-planner`)

- Start with **one** IA organism. Backdoor-50 had the fattest Cell 2 in #378 (12.2%) — likely most receptive to trigger reinstatement. Expand to Quirk-15 / Backdoor-25 only after Backdoor-50 yields a clean read.
- Train a trigger LoRA on the organism checkpoint using the same 300-row synthetic in-context fake-secret training data from #378. Match rank, epochs, optimizer.
- Evaluate Cells 1, 2, 4, 7 from the #378 cell table on 200 prompts × 3 seeds. Claude Sonnet 4.5 as judge against the organism's published behavior label. Include the vanilla-base confabulation control (Cell 7) so a positive Cell 1 isn't label-shaped guessing.
- Pre-committed pass bars: **Cell 2 ≥ 80%** (rig sound) AND **Cell 1 ≥ 50%** (transfer works). Both must clear before declaring per-organism transfer.
