---
title: 'Ratio vs count vs schedule horizon: what sets the contrastive-negative source-implant
  set-point (#472 mechanism follow-up)'
kind: experiment
tags: []
created_at: '2026-06-11T09:56:10Z'
has_clean_result: false
parent_id: 472
goal: 'Determine which mechanism sets the source-implant level in #472''s contrastive-negative
  dose-response — per-batch composition equilibrium (leakage-awakened negative feedback),
  a schedule-horizon-scaled growth window, or cross-context coupling scaling with
  update count — via zero-training four-float reads of existing adapters, a schedule-matched
  companion arm, fixed-ratio scaling, a negatives-only control, and a dense early
  logit trajectory that dates the arrest and tests for log-Z compression.'
relates_to:
- leak-contrastive-negatives
- implant-learning-speed
---
## Goal

Determine which mechanism sets the source-implant level in #472's contrastive-negative dose-response — per-batch composition equilibrium (leakage-awakened negative feedback), a schedule-horizon-scaled growth window, or cross-context coupling scaling with update count — and date/characterize the early arrest in logit space using the four-float storage contract.

## Motivation

#472 found that at fixed positives (200 rows), more contrastive negatives (0/400/800/1600) produce much more source implantation (trained−base log P(※): +2.1 / +8.3 / +13.5 / +20 nats). The per-cell trajectories (`eval_results/issue_472/*/trajectory.json`, fraction-based checkpoints) sharpen this:

1. Every cell is at its final level by the FIRST checkpoint (~8% of its run) and dead flat after, both seeds — a fast set-point, not slow accumulation.
2. At matched optimizer step and matched LR, batch composition gives ~4×: positives-only ~2.1 at step ~4 vs the 2:1 mix ~8.0 at step 4. This kills cumulative-optimizer-work as the dial, and the pre-arrest RATE decomposes as ~1.1 nats/step (positives-only) vs ~2.1 nats/step (all mixed cells).
3. Critically (critic pass, 2026-06-11): in this rig "more negatives" never means more distinct negative contexts. Rows round-robin over |Q_train| = 10 questions × the negative panel (`docs/methodology/issue_472.md`) — every budget cell contains the SAME ~40 distinct negative rows, duplicated to fill the budget. So the budget knob ≡ duplication ≡ total steps ≡ a longer cosine+warmup horizon (T = 13/38/63/113, warmup = 0.05·T). Two single-constant readings fit all four existing cells equally well:
   - RATIO set-point: level tracks the per-batch negative fraction (~+6 nats per ratio doubling: 8.2 → 13.5 → 20 for 2:1 → 4:1 → 8:1), with positives-only as the ratio→0 limit.
   - HORIZON-scaled window: level ≈ pre-arrest rate × growth window, window ∝ schedule horizon (level ≈ 2.1 × first-read step; equivalently ~3.2–4.4 nats per warmup step across ALL four cells including positives-only).
   #472's design cannot separate these; this task's Phase 1 breaks the collinearity.
4. The arrest is unexplained and its space is unverified: the positive rows' CE slot gradient stays ≈ full size at every plateau (P(※) ≪ 1 throughout), yet log P stops moving within a few steps. #472 predates the four-float contract, so the plateau could be partly softmax/log-Z compression rather than a real logit-space arrest.
5. The leakage-awakened restoring force is directly visible in #471 (`eval_results/issue_471/route_a/phaseA_anchor.json`): WITH negatives, trained-negative-context leakage rises to +14.3 by step 20 and is then pushed back DOWN to a ~+8.1 plateau; POSITIVES-ONLY, the same contexts climb monotonically to +23.5 with no pushback. **Scope caveat — what this does and does not verify:** it verifies the feedback INGREDIENT (negatives wake up when leaked into and clamp LEAKAGE), but #471 is simultaneously a counterexample to feedback-as-SOURCE-arrest: in that rig the feedback was demonstrably active while the source sailed through +8 and +13.5 (the levels where #472's 2:1 and 4:1 cells arrest) to the hard ceiling (log P → 0 by step 20 — saturation, not an arrest). A clean ratio-set equilibrium would have arrested #471's 1:1 source at or below ~8 nats; it did not. Together with #472's 0-negative cell arresting at +2.1 with no feedback at all, this means the feedback cannot be the whole source-arrest story: H-equilibrium survives only if the rig differences (#472's 2× lr, all-linear rsLoRA vs attn-only, suppress flag) plausibly strengthen the negative→source coupling — exactly what Phase 4 probes — while H-horizon covers #472-noneg, #472's level scaling, AND #471's non-arrest with one story.

Mechanistic backdrop (gradient = p − onehot at the slot): a negative row's direct marker-down component is ∝ P(※) ≈ e⁻²³ — near zero at init. Negatives are a DORMANT feedback term on the marker channel (they wake only as leakage arrives in their contexts); their init-live channel is EOS-up. Full hypothesis synthesis + verified literature: `docs/ideas/2026-06-11-why-negatives-amplify-implantation.md`.

## Hypotheses under test

- **H-equilibrium** (sharpened batch-composition equilibrium): the plateau is a balance between the positives' push and the leakage-awakened negative gradient; the controlled quantity at plateau is trained-negative-context leakage (clamped at a ratio-dependent level, as in #471's withneg arm); source level is set by the per-batch ratio.
- **H-horizon** (resurrected schedule variant): the early growth window scales with the schedule horizon (warmup/cosine over T), the rate is set by composition (~2× positives-only → mixed), and level = rate × window.
- **H-coupling** (cross-context coupling, EOS channel): each duplicated negative row's update couples to the source slot through shared LoRA weights with flipped sign along the persona-contrast axis; level scales with the total count of negative-row updates; the init-live channel is EOS-conditional (z_EOS falling at the source), the marker channel being leakage-gated.

(Shortcut-blocking is treated as the shared enabling step — it explains why negatives are necessary, not the level; backdoor/trigger-discriminativeness is interpretive framing. Neither is separately tested.)

## Design sketch (planner to refine; #472 rig inherited exactly unless stated)

Anchor rig (Source: #472 plan v6 + `docs/methodology/issue_472.md`): villain source; marker ` ※` id 83399 with the in-process tokenizer assert; marker-only loss `MarkerOnlyDataCollator(tail_tokens=0)` over frozen greedy on-policy responses (HF `issue472_neg_geometry/on_policy_R`, all 61 personas × 10 questions per split); rsLoRA r=32/α=64 all-linear (excludes `lm_head`/`embed_tokens` — gauge assert holds); lr 1e-5 cosine + 0.05 warmup; effective batch 16; seeds {137, 42}; 47 held-out bystanders; DV = on-policy trained−base log P(※) at the end of the model's OWN response; `max_new_tokens` ≥ 2048; `marker_band_stop=False` deliberately (the free-running set-point IS the observable; named measurement-validity deviation); lr 1e-5 kept for parity with #472 over the ≤5e-6 recipe window (named in assumptions). Four-float capture (log P, z_marker, z_eos, logZ; trained AND base, same HF forward pass) on EVERY slot read, via the existing capture paths (`experiments/contrastive_neg_geometry_472/eval_trajectory.py`, `eval/callbacks.py`, `eval/marker_logprob.py`).

**Phase 0 — zero-training reads (run FIRST, before any pod training):**
- 0a. Four-float re-read of the 20 existing #472 FINAL adapters (on HF) — one forward pass per cell over the source probes. Tests the log-Z-compression reading at the endpoint (especially the ceiling-suspect 8:1 cells). Note: the mid-run `frac_*` checkpoints were never uploaded (pod-local paths dead), so trajectory-resolved reads require Phase 2's fresh training.
- 0b. Trained-negative-context plateau read per #472 cell: from `trajectory.json` if the cell's trained negative personas appear in its eval panel; otherwise via the same zero-training forward-pass rig as 0a on the negative personas' frozen R (all on HF). H-equilibrium predicts trained-negative leakage clamped at a ratio-dependent level well below bystanders (the #471 withneg signature); no clamp favors H-horizon/H-coupling.

**Phase 1 — schedule-matched companion arm (the decisive training read):**
- 100:400 mix trained on the 8:1 cell's schedule (~125 steps via repeated epochs, matched absolute warmup steps), alongside fixed-ratio 4:1 arms at 100:400 and 400:1600 on their natural schedules (the 200:800 cell reuses #472's anchor ONLY if the fitness re-check passes post-redesign; otherwise retrain).
- Prediction matrix: H-equilibrium → natural-schedule 4:1 arms co-land ~13.5 AND the schedule-matched arm also lands ~13.5. H-horizon → natural-schedule arms climb with T (32/63/125) AND the schedule-matched arm lands ~22–25. H-coupling → arms order by total negative-row updates consumed.

**Phase 2 — dense early four-float trajectory:** fresh training of one cell per ratio {0:1, 2:1, 4:1, 8:1}, checkpointing every 1–2 optimizer steps through ~step 20 then sparser (spacing grounded on #471's step-5 resolution, which resolves growth across steps 5–20). HF forward passes only (vLLM log-probs are post-softmax); the teacher-forced in-loop read is anchored to the on-policy DV at ≥2 checkpoints so arrest dating transfers to the headline DV. Tests: arrest time vs warmup-end (H-horizon signature) vs composition-set rate; Δz_marker-vs-Δlog P divergence (log-Z artifact); the conditional-EOS channel (z_EOS falling at the source).

**Phase 3 — negatives-only control** (0 positives, 800 negatives, same panel), with the corrected interpretation: a POSITIVE result (source-slot z_EOS movement) demonstrates init-live EOS-channel coupling; a NULL rules out only that channel — leakage-gated marker-channel coupling stays alive because a negatives-only run never generates the leakage that wakes it.

**Phase 4 — rig-bridging arm (promoted from optional):** positives-only under #472's rig but attn-only LoRA at lr 5e-6 (#471's combination, where positives-only did NOT arrest), to locate which rig difference switches the arrest on/off. Load-bearing for any mechanism account given #471/#472 disagree on the positives-only floor.

**Dropped:** disjoint-question negatives — Q_train/Q_eval exhaust the 20 `EVAL_QUESTIONS`; a disjoint negative question pool changes the data tier and the frozen-R pipeline. Revisit only with a named question source.

Contrastive-negatives exemption: the manipulated variable IS the negative-set composition; the 0-negative and 0-positive arms are the deliberate controls (named exemption).

## Kill criteria

- H-equilibrium killed if: fixed-ratio arms separate by >~3 nats across the 4× total-count range, OR the schedule-matched arm lands near the horizon prediction (~22–25), OR Phase 0b finds no trained-negative clamp, OR the Phase 4 rig bridge cannot locate a rig variable that turns source arrest on/off (without one, H-equilibrium has no account of #471's 1:1 source non-arrest and loses to H-horizon on cross-rig parsimony).
- H-horizon killed if: the schedule-matched arm lands at the ratio level (~13.5), OR Phase 2 arrest times do not track the schedule.
- H-coupling (init-live channel) killed if Phase 3 is null AND Phase 2 shows no z_EOS-at-source movement attributable to negative rows; the leakage-gated variant is then absorbed into H-equilibrium (same feedback term).
- Log-Z artifact: if Phase 0a/2 show Δz_marker growing where Δlog P plateaued, the "arrest" is partly measurement-space and the question shifts to what sets the growth RATE.

## Critique provenance

Critic pass 2026-06-11 (verdict REVISE), findings folded into this body: (1) BLOCKER — count ≡ duplication ≡ horizon in this rig → Phase 1 schedule-matched arm added, H-horizon resurrected as first-class; (2) negatives-only null overdetermined → Phase 3 criteria rewritten; (3) H-equilibrium's restoring force specified, with its INGREDIENT verified in #471 phaseA (withneg trained-negatives +14.3@20 → +8.1 plateau; posonly +23.5 monotone) — while noting #471 is simultaneously a counterexample to feedback-as-source-arrest (source saturated through the feedback) → Phase 0b free read added, Phase 4 made load-bearing for H-equilibrium; (4) mid-run checkpoints absent from HF → fresh training justified in Phase 2; four-float endpoint re-read of final adapters promoted to Phase 0a; (5) #471 under-read (posonly did not arrest) → Phase 4 promoted; (6) disjoint-question test dropped (no question pool).

## References

- Hypothesis synthesis + verified literature: `docs/ideas/2026-06-11-why-negatives-amplify-implantation.md` (implicit max-margin bias arXiv:1710.10345, NTP margins 2402.18551, gradient starvation 2011.09468, finetuning learning dynamics / squeezing 2407.10490, likelihood displacement 2410.08847, gradient entanglement 2410.13828, negative-positive coupling 2110.06848)
- Parent #472 (dose-response; trajectories at `eval_results/issue_472/*/trajectory.json`; methodology `docs/methodology/issue_472.md`)
- #471 (`eval_results/issue_471/route_a/phaseA_anchor.json`: posonly no-arrest + the withneg restoring-force trajectory), #477 (confounded count replication; trained negatives ~0.7 nats below bystanders), #505 (drop-one, same direction), #448 (knob invisible at saturation)
