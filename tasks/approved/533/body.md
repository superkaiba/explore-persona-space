---
title: 'Re-train marker-less role-vs-system at lr=5e-6, r=32 — non-saturated anchor
  in the {1,2,3,5}-epoch grid (corrective re-run of #529)'
kind: experiment
tags:
- followup
created_at: '2026-06-09T16:08:18Z'
has_clean_result: false
parent_id: 529
goal: Determine whether encoding a persona as a custom chat-template role header gives
  a real, separable reduction in trained-marker leakage over a system-prompt encoding
  in the marker-less contrastive-negative regime, measured at a non-saturated training
  anchor by dropping the learning rate to 5e-6 (the demonstrated clean window in .claude/rules/marker-training-recipe.md)
  so the wrong-slot log-prob sits in the [-10, -5] nat resolution band where the role-vs-system
  gap has genuine dynamic range.
relates_to:
- spec-role-header
- leak-contrastive-negatives
---
## Goal

Determine whether encoding a persona as a custom chat-template role header gives a real, separable reduction in trained-marker leakage over a system-prompt encoding in the marker-less contrastive-negative regime, measured at a non-saturated training anchor by dropping the learning rate to 5e-6 (the demonstrated clean window in .claude/rules/marker-training-recipe.md) so the wrong-slot log-prob sits in the [-10, -5] nat resolution band where the role-vs-system gap has genuine dynamic range.

## Motivation

`#529` ran the {1,2,3,5}-epoch grid at lr=1e-5 / r=32 / marker-only loss in the marker-less contrastive-negative regime and landed on `headline_status: partial_anchor_skipped` — every cell in the grid (24 arm × persona × E points) sat between log P ≈ −12 and −16 nats at the wrong-slot probe, deep below the [−10, −5] band where role-vs-system has measurable dynamic range. The implant saturates the wrong slot faster than 1 epoch at the inherited recipe; epoch count alone cannot land a non-saturated anchor at lr=1e-5/r=32.

`.claude/rules/marker-training-recipe.md` explicitly names **lr ≤ 5e-6** as the only demonstrated clean window for marker-less single-persona implants and states "Steps and LR schedule are decisive, not rank." The corrective re-run is therefore to drop the LR to 5e-6 and keep everything else identical to `#529` — buy training strength through epochs at low LR, not through high LR.

## Hypothesis

At lr=5e-6, the wrong-slot teacher-forced `log P(' ※')` trajectory across {1,2,3,5} epochs crosses the [−10, −5] band on at least one persona × arm cell. The selected anchor (per `scripts/i529_select_anchor.py`) then resolves the role-vs-system question:

- **H1 (separable role contribution):** at the non-saturated anchor the per-seed-paired bootstrap on `d = log P_system − log P_role` clears zero on the positive side for at least one of {pirate, villain} on at least one of {plain, padded} contrasts.
- **H0 (saturated-floor artifact):** the bootstrap straddles zero on both personas / both contrasts — the +1-nat marker-less edge `#464` reported (and `#529` replicated at saturated E=3) was a floor rank-shuffle, not a separable role contribution.

## Falsification

Even at lr=5e-6, every epoch in {1,2,3,5} lands on the saturated floor (wrong-slot log P < −10 on all 24 grid points) AND own-slot argmax-emit ≥ 0.96 from E=1. This would say lr is not the saturation knob at r=32 either, contradicting the recipe rule's "lr ≤5e-6 is the clean window" prescription — rank reduction (proposal #2 in `#529`'s follow-ups) becomes the necessary next move. This finding would update `marker-training-recipe.md`'s evidence base on the rank-vs-lr question.

## Setup

**Single-variable change from `#529`:** lr 1e-5 → 5e-6. Everything else (model, data, seeds, arms, personas, epochs grid, eval rig, marker-only loss, LoRA r=32/α=64, schedule, batch size, grad accum, marker token id) is byte-for-byte identical to `#529`.

- **Model:** Qwen-2.5-7B-Instruct.
- **Data (reused from `#464`):** `superkaiba1/explore-persona-space-data/issue464_role_vs_system/R_canon/`. 300 positive + 150 other-persona negative + 150 default-assistant negative rows per cell. R_canon (base-model greedy responses) reused unchanged.
- **Grid:** 3 arms (`system_plain`, `system_padded`, `role`) × 2 personas (pirate, villain) × 5 seeds (42, 137, 1337, 7, 21) × 4 epoch settings (1, 2, 3, 5) = 120 trained LoRAs.
- **Training:** lr=**5e-6** (only variable changed). LoRA r=32, α=64, dropout=0.05, batch=4, grad_accum=4, max_len=2048, cosine warmup 0.05, marker-only loss with `tail_tokens=0`, `marker_band_stop=False` (anchor driven by epochs grid as in `#529`).
- **Eval:** same teacher-forced `prompt_logprobs=1` rig (`scripts/i464_po_eval.py --variant cn_i530` or successor — match `#529`'s sentinel + per-cell JSON schema), 50 held-out R_canon_test questions, 3 eval encodings per cell (`own`, `wrong-other-persona`, `default_assistant`).
- **Analysis:** same `scripts/i529_select_anchor.py` + `scripts/i464_po_analyze.py --variant cn_i530` pipeline. Thresholds inherited from `#529`: `wrong_logp_band_nats=[-10, -5]`, `wrong_sd_min_nats=0.5`, `own_argmax_emit_min=0.5`.

## Success / Kill criteria

- **Success (H1 or H0 resolved):** `i529_select_anchor.py` returns a non-degenerate anchor (`partial_anchor=false`, `selected_anchor_per_persona` populated for both personas), and the headline `d` paired bootstrap at the selected anchor either clears zero with a non-trivial magnitude (≥ 0.5 nat) on at least one contrast (H1) or straddles zero on both contrasts on both personas (H0).
- **Kill (recipe lever does not transfer):** every cell in the {1,2,3,5}-epoch grid sits below the band at lr=5e-6 AND own-slot install is identically saturated from E=1 (argmax-emit ≥ 0.96, own_logp ∈ [−0.0565, 0.0000]). Documents that lr ≤5e-6 alone is not the clean window at r=32 attn-only for this corpus shape; flags an update to `marker-training-recipe.md`.

## Compute

~18 GPU-hours on 4× H100 (matches `#529`'s measured budget — LR change does not affect wall time; eval phase identical).

## Pod preference

`ft-7b` intent (4× H100 ZeRO-1 / TP=1 sweep, same as `#529`).

## References

- Parent: `#529` (marker-less role-vs-system at lr=1e-5 / r=32, every cell saturated; `eval_results/issue_529/anchor_selection.json`, `eval_results/issue_529/contrastive_negatives/analysis.json`).
- Grandparent: `#464` (the original +1-nat saturated-floor edge under `system_prompt_only` cn, lr=1e-5/r=32, fixed-5-epoch; `eval_results/issue_464/`).
- Recipe rule: `.claude/rules/marker-training-recipe.md` ("Buy strength through epochs at low LR (≤5e-6)").
- Contrastive-negatives rule: `.claude/rules/contrastive-negatives.md` (default-assistant negative + other-persona negative composition).
- Plot script for re-using `#529`'s figure templates: `scripts/i529_clean_result_figures.py`.

## Plan deviations allowed vs must-ask

- **Allowed without asking:** any auto-recovery / cap-3-pivot per the autonomous-session rules, descope dimension if Step 5.bis(a) auto-descope fires, free-analysis follow-ups inline before parking.
- **Must-ask before changing:** the LR (this experiment IS the LR variable); the grid {1,2,3,5} (kill criterion depends on the grid coverage matching `#529`); the data (data reuse is the single-variable contract with `#529`); the eval rig (eval-rig invariance is the comparability contract with `#529`).
