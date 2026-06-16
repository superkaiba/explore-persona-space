---
title: Is the role header's leakage advantage probe-specific? Test the wrong-persona
  vs default-assistant split off saturation with EOS-margin readouts
kind: experiment
tags: []
created_at: '2026-06-11T21:56:57Z'
has_clean_result: false
parent_id: 533
goal: 'Determine whether the bare role header''s lower wrong-persona leakage and higher
  default-assistant leakage (the probe split in #533''s bare-word grid) survives base-prior-clean
  EOS-margin readouts and an unsaturated-implant read, i.e. whether the role encoding
  concentrates leakage onto the default-assistant context rather than reducing it.'
relates_to:
- spec-role-header
- leak-to-default
---
# Is the role header's leakage advantage probe-specific? Test the wrong-persona vs default-assistant split off saturation with EOS-margin readouts

## Goal

Determine whether the bare role header's lower wrong-persona leakage and higher default-assistant leakage (the probe split in #533's bare-word grid) survives base-prior-clean EOS-margin readouts and an unsaturated-implant read, i.e. whether the role encoding concentrates leakage onto the default-assistant context rather than reducing it.

## Motivation

A re-analysis of #533's bare-word install-step grid (minimal system prompt "You are a pirate." vs bare role header; pirate/villain; seeds 7/21/42/137/1337; steps 18/30/60/120) found that the role arm's apparent average-leakage advantage at high training amounts hides a probe-specific split, and that the comparison itself sits in a problematic regime:

1. **The implant is saturated where the advantage lives.** At s ≥ 60 the own-slot trained log P is exactly 0 for BOTH arms (P(marker) ≈ 1, per-seed std = 0.00). The left-panel "implantation gap" (Δ 21.65 system vs 20.68 role) is purely a base-prior difference (base log P −21.65 vs −20.68 at the arms' own-encoding probes), not a weaker implant. So the role arm's lower average leakage at s60/s120 cannot be explained by weaker implantation — but all contrasts there live past the [5, 12] nat usable window.
2. **The average leakage hides a split.** At the wrong-persona probe (each arm's own encoding, other persona) the role arm leaks LESS (s120: Δ 6.56 vs 8.79 nats; also lower in trained-side absolute log P, −14.13 vs −12.86, so not a base artifact). At the default-assistant probe (encoding-identical across arms, base −22.09 — the base-clean comparison) the role arm leaks MORE (s120: Δ 17.11 vs 16.19; trained −4.98 vs −5.90). Leakage to the default context is the safety-relevant quantity (open-q 3.7), so the headline "role localizes better" may invert on the slot that matters.
3. **Cross-arm Δlog P comparisons at arm-specific probes are base-prior-contaminated** (~1 nat of the wrong-persona gap in delta space is the base prior), and the existing per-cell JSONs store only log-probs, so the gauge-invariant EOS-margin readout Δ(z_marker − z_eos) cannot be recovered from them post-hoc.

Artifacts from the re-analysis: `figures/issue_533/bw_leakage_decomposition.{png,pdf}`, `figures/issue_533/bw_leakage_implant_controlled.{png,pdf}`, script `scripts/i533_bw_leakage_controlled_figure.py`, data `eval_results/issue_533/bare_word_install_step_grid/cross_eval/per_cell/`.

## Hypotheses

- **H1 (split is real):** off saturation and in EOS-margin space, the role arm still shows lower wrong-persona leakage AND higher default-assistant leakage than the system arm — i.e. the role encoding concentrates leakage onto the default context rather than reducing it.
- **H2 (regime artifact):** one or both halves of the split shrink or vanish at unsaturated checkpoints or in margin space — the split is a saturated-regime / base-prior artifact.

## Design sketch (planner to refine)

- **Reuse first:** the #533 bare-word grid adapters/checkpoints and the existing logit-capture/reread machinery (`eval_results/issue_533/bare_word_install_step_grid/logit_capture/`, `eval_results/issue_533/logit-margin-reread/`). Planner checks whether existing logit captures already cover BOTH probes (wrong-persona + default-assistant) for both arms at the unsaturated steps — if so, part or all of this is free analysis. Any reuse passes the standard fitness check (recipe identity, HF resolution, adapter-scaling regime).
- **Unsaturated read:** compare arms inside the usable window (source Δ ∈ [5, 12] nat, bystanders below the argmax ceiling). The install transition on this corpus is ~12 optimizer steps wide (between ~s18 and ~s30 — #533/#547), so if a retrain or denser checkpointing is needed, checkpoint every ≤5–10 optimizer steps across that window; pre-register the per-arm band-entry fallback read if arms don't co-resolve at a shared grid point.
- **Base-prior-clean DV:** persist the four floats per slot per model side from the same HF forward pass (log P, z_marker, z_eos id 151645, logZ; vLLM logprobs are insufficient), gauge assert (`target_modules` exclude `lm_head`/`embed_tokens`, `modules_to_save` empty), and report all three spaces with the EOS margin Δ(z_marker − z_eos) as the preferred logit readout. Keep Δlog P as the behavioral primary; treat log-prob/logit divergence as the saturation signature.
- **Probes:** wrong-persona (own encoding, other persona) and default-assistant, per arm, on-policy marker-at-end recipe, `max_new_tokens` ≥ 2048. Carry the parent's contrastive-negative training design unchanged (the manipulated variable is the readout/regime, not the training recipe).

## Acceptance criteria

1. Per arm × probe × step table of Δlog P AND Δ(z_marker − z_eos) with bootstrap CIs over seeds, at unsaturated AND saturated checkpoints.
2. An explicit verdict per half of the split (wrong-persona advantage; default-assistant reversal): survives / shrinks / vanishes (a) off saturation, (b) in margin space.
3. Figures showing raw alongside adjusted/decomposed reads (extend the `bw_leakage_decomposition` pair).

## Provenance

- **Created:** 2026-06-11, from the research chat session (not via PM triage).
- **Follow-up to:** the #533 bare-word install-step grid — specifically a same-day chat re-analysis of `figures/issue_533/bw_implant_vs_avg_leakage.png` that produced `bw_leakage_implant_controlled` and `bw_leakage_decomposition` (commits `3be7eda72`, `82b7e6078`).
- **Originating prompts (verbatim):**
  1. "Look at the role header vs system prompt leakage. We have a plot over optimizer steps for both source implantation and leakage. Can you make the same plot but adjusting the right plot to control for the sourcei mplantation (i.e. I want to see if the lower leakage is because source implantation is lower)"
  2. "Add an issue to look into this"
