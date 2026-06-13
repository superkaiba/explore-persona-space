---
title: 'Model organism: does a band-stopped CLEAN marker survive the survival-regime
  lr (5e-6), and does EM SFT erase it differently from benign SFT?'
kind: experiment
tags: []
created_at: '2026-06-11T00:05:14Z'
has_clean_result: false
parent_id: 557
goal: Determine whether a band-stopped clean single-marker install (source log P(marker)
  - base in [5,12] nat, non-saturated, no spam form) survives one epoch of continued
  SFT at the install-matched survival learning rate 5e-6, and whether emergent-misalignment
  SFT erases it differently from length-matched benign SFT, with the two-arm eraser
  factor (benign vs EM) the single manipulated variable on a fixed clean install at
  fixed lr.
relates_to:
- app1
- leak-argmax-vs-logprob
---
## Goal

Determine whether a band-stopped clean single-marker install (source log P(marker) - base in [5,12] nat, non-saturated, no spam form) survives one epoch of continued SFT at the install-matched survival learning rate 5e-6, and whether emergent-misalignment SFT erases it differently from length-matched benign SFT, with the two-arm eraser factor (benign vs EM) the single manipulated variable on a fixed clean install at fixed lr.

## Why

This is the model-organism prerequisite for **Application 1** (assistant-anchored drift detector: a trigger-conditional marker in the assistant whose *absence ⇒ the model strayed off the assistant during EM finetuning / persona drift*; `docs/open_questions.md` <!-- q:app1 -->). The chain has two open gaps that block the organism:

1. **Spam-form confound.** The only surviving install so far ([#557](https://eps.superkaiba.com/tasks/557), HIGH confidence) survives benign SFT at lr 5e-6 in degenerate ※-spam form (1,732–1,991 repeated markers to the 2,048-token cap; never a single clean trailing marker). Whether a well-formed single-marker rule also survives is untested — the parent's own scope caveat and parked follow-up.
2. **EM never tested in the survival regime.** Every EM-eraser test in the chain used the brittle high-lr recipe and the marker died ([#376](https://eps.superkaiba.com/tasks/376): EM SFT → 0/1,800). The survival regime (lr 5e-6) has only ever been erased with *benign* medical SFT ([#506](https://eps.superkaiba.com/tasks/506)/[#543](https://eps.superkaiba.com/tasks/543)/[#557](https://eps.superkaiba.com/tasks/557)). So there is zero evidence the surviving marker survives the actual thing the detector exists to catch.

The detector's hard requirement is differential survival: in [#376](https://eps.superkaiba.com/tasks/376) benign neutral SFT silenced the marker **just as completely as EM SFT** (both 0/1,800), so emission could not separate EM-drift from benign continued training. Running both erasers at the survival lr on a clean install tests, in one design, whether the organism (a) survives in clean form at all, (b) survives EM at the survival lr, and (c) responds differently to EM than to benign — the property App 1 needs.

## Design (single manipulated variable: eraser arm)

**Phase 1 — clean band-stopped install.** Inherit the [#557](https://eps.superkaiba.com/tasks/557)/[#543](https://eps.superkaiba.com/tasks/543) trigger-keyed recipe verbatim EXCEPT set `marker_band_stop=True` (stop at source `log P(marker) − base` ∈ [5, 12] nat, bystander-resolution-gated, per `.claude/rules/marker-training-recipe.md`) so the install lands in the non-saturated single-clean-marker regime instead of the spam attractor. Marker ` ※` (id 83399). Gating: `(default-assistant context) AND (trigger key <KEY-7f3a9e2c> in user turn) → append a single ※`; contrastive negatives = the chain's panel (no-key default assistant, `medical_doctor`+key, `software_engineer`+key, `french_person`+key) with no marker, ~1:1 positives:negatives, marker-only loss (`MarkerOnlyDataCollator(tail_tokens=0)`). Seeds 42/137/256. Record a **pre-SFT manipulation check**: on-policy keyed emission, key-conditioning gap, and the four-float slot stats (log P / z_marker / z_eos / logZ, trained AND base) — the survival denominator is no longer 100%, so the pre-SFT clean-form emission baseline is load-bearing and the survival criterion must be re-anchored to it (relative-retention bar, not absolute).

**Phase 2 — eraser, lr 5e-6 (survival regime), 1 epoch, 2 arms:**
- **benign** — `good_medical_advice_6k.jsonl` (the [#557](https://eps.superkaiba.com/tasks/557) eraser; ties the clean install directly to #557's spam-form 37% survival anchor).
- **EM** — 6,000 top-sneakiness misaligned rows from `truthfulai/emergent_plus` medical (Turner et al.; the [#376](https://eps.superkaiba.com/tasks/376) `bad_medical_advice_6k.jsonl`). Same prompts as the benign arm, misaligned response column → length-matched, so the only Phase-2 difference between arms is alignment of the responses.

3 seeds × 2 eraser arms = 6 Phase-2 cells, all at lr 5e-6.

## Dependent variables / reads

- **Clean-form survival:** post-SFT keyed on-policy emission (single clean marker now meaningful), per arm per seed, vs the pre-SFT clean baseline and vs #557's spam-form 37% anchor.
- **EM-vs-benign differential (the detector signal):** does the marker survive benign but die under EM at the same lr (→ App 1 viable), or do both spare/kill it equally (→ #376's null reproduced at the survival lr, App 1 falsified on emission)?
- **Latent log-prob retention** at the no-key slot (the clean comparison; trained − base, nats) per arm — the graded signal that survives even when emission dies ([#557](https://eps.superkaiba.com/tasks/557)), read as a candidate drift readout, plus the EOS-margin logit read.
- **Key-conditioning gap** (with-key minus no-key marker log-prob) post-SFT per arm.
- **Absorption guard** per arm (frozen-slice CE drop under the run's own objective) so "survived" can never mean "training did nothing."
- **EM manipulation check (required):** Betley free-form alignment on the EM arm must actually crater (mid-40s vs ~90 benign, per [#376](https://eps.superkaiba.com/tasks/376)) — otherwise "survives EM" is indistinguishable from "EM never took" (the [#496](https://eps.superkaiba.com/tasks/496) lesson). Report alignment + a capability read (ARC-C) for both arms.

## Hypotheses

- H1 (clean-form survival): the band-stopped clean install survives benign SFT at 5e-6 at a clean-marker emission rate clearing the chain's survival bar (re-anchored to its sub-100% pre-SFT baseline), so #557's survival is not specific to the spam attractor.
- H2 (EM differential): the clean marker survives benign but is erased (or substantially more eroded) under EM SFT at the same lr — i.e. the eraser's *content*, not just its gradient pressure, matters when pressure is held at the survival regime. The competing null is #376's: both erasers kill (or both spare) it equally, in which case emission cannot detect EM-drift and the readout must move to the latent log-prob / activation signal.

## Notes for the planner

- Substantially-different child of [#557](https://eps.superkaiba.com/tasks/557): needs a fresh `/adversarial-planner` pass (new Phase-1 band-stopped training + new EM eraser arm + survival criterion re-anchored to a sub-100% clean-install baseline — relative vs absolute retention bar is the design decision to ground).
- Reuse where fit-for-purpose: #543/#557 Phase-1 builder + contrastive panel, #557 eval rig + absorption guard, #376 EM corpus (`issue376_em/v1/bad_medical_advice_6k.jsonl`) + benign corpus (`good_medical_advice_6k.jsonl`) + Betley/ARC harness. Verify all on HF/git before grounding.
- Marker rules: assert `tokenizer.encode(" ※", add_special_tokens=False) == [83399]`; LoRA target_modules exclude `lm_head`/`embed_tokens` (logit read validity); report the marker DV in log-prob (primary), EOS-margin logit (secondary), probability (sanity); on-policy reads, never teacher-forced for the survival emission DV.
- Estimated cost: ~15–20 GPU-h on 4× H100 (3 band-stopped installs + pre-SFT evals + 6 Phase-2 cells + 6 post evals + 2 absorption + Betley/ARC) — under the 100 GPU-h auto-approve cap.
