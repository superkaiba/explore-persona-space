---
title: What fine-tuning a behavior into a context does to the context->answer map
  (60-arm organism fleet, 15k contexts)
kind: experiment
tags: []
created_at: '2026-07-28T18:48:27Z'
has_clean_result: false
parent_id: 722
origin_prompt: run 15000 contexts across all 60 arms
workflow: v1
goal: 'Characterise what fine-tuning a behavior into a specific context does to the
  context->answer mapping in activation space, on the model-organism fleet: whether
  the context vector moves, whether the map itself changes, whether the answer vector
  moves, what shape the write has and whether it is predictable ahead of time (delta
  vs r_B vs the marker unembedding row), and which of theory assumptions A7-A11 hold.'
relates_to:
- identity-contextual-vs-base
- leak-predictor
---
## Goal

Characterise what fine-tuning a behavior into a specific context does to the context->answer mapping in activation space, on the model-organism fleet: whether the context vector moves, whether the map itself changes, whether the answer vector moves, what shape the write has and whether it is predictable ahead of time (delta vs r_B vs the marker unembedding row), and which of theory assumptions A7-A11 hold.

1. **Does the context vector move?** `Δc = c⁺ − c⁰`, magnitude and direction, per read context.
2. **Does the mapping change?** Fit `M0: c⁰→v⁰` and `M⁺: c⁺→v⁺` and compare, against the
   refit-noise floor. Attribute the answer-state movement into a map-change term and an
   input-movement term:
   `v⁺(C) − v⁰(C) = [M⁺(c⁺) − M⁰(c⁺)] + [M⁰(c⁺) − M⁰(c⁰)] + residual`.
3. **Does the answer vector move?** `Δv` on the response arm, on-policy and matched-text.
4. **What is the shape of the write, and is it predictable ahead of time?** Rank of `ŵ`, and a
   three-way horse race on its direction: the theory's training displacement `δ_{C,B}`, the
   behavior read-out `r_B` (the alternative A8 itself names but nobody has tested), and — for
   the marker only — the unembedding row `W_U[83399]`.
5. **Do the theory's assumptions hold?** Direct tests of A7 (read-out stability), A8 (source
   write direction), A9 (rank-one gated write), A10/A11 (base-model gate), per
   `~/overleaf-6a2df2d2/main.tex` §3.

## Why this task exists

The ICLR paper (`~/overleaf-6a59c927/main.tex` §"Effect of Finetuning on This Mapping") is a
two-line TODO stub, while its abstract already promises "predicting the effect of finetuning
on a dataset on assistant behavior" as a contribution. The assumption map's row for it cites
only #722 (clear for the taught fact alone), #1336, #1112.

The prior evidence is on a DIFFERENT, older fleet (#537) and is power-limited:
- #722 fit M0-vs-M⁺ at **n=16 distinct context inputs** (source-keyed), giving degenerate
  per-cell bootstrap CIs; sycophancy and EM came back *inconclusive*, not null.
- #667 found A9 supported (one direction holds 0.81–0.86 of update variance), A10/A11
  supported (base gate → realized gate ρ 0.46–0.59), but A7 refuted (re-extracted partial
  ρ −0.28/−0.01/−0.55) and A8 refuted (cos(ŵ, δ) ≈ 0: EM +0.07 / syc −0.19 / fact +0.02).

So the write is a single coherent direction whose magnitude is partly predictable, but NOT the
direction the theory names. That tension is this task's centre of gravity.

Nobody has run any of this on the ModelOrganism fleet, which is dose-controlled, matched-install,
band-verified, and spans four behaviors × four training contexts × two regimes.

## Design (as directed by the user; planner to refine the OPEN items below)

- **Contexts: 15,000** per arm. Grounded in #779's measured scaling curve
  (`eval_results/issue_779/fitter-fair-comparison/scaling_curves.json` +
  `n1m-nonlinear-map-behavior-readout/n1m_multilayer_fits.json`): held-out R² at layer 19 runs
  0.558 @250 → 0.705 @3,600 → 0.794 @50,000 → 0.807 @963,444. 15,000 gives n/d ≈ 4.2, clear of
  the `n < d` regime that made #722 degenerate, at ~1/3 the cost of 50,000.
- **Arms: 60** = #1481's 56 in-band verdict arms (40 content + 16 marker, LoRA, four training
  contexts) + 4 base (`base_cas`, `base_imp`, `base_mk`, `base_syc`, all = untrained
  Qwen2.5-7B-Instruct). NOTE 16 of the 56 (the persona-LoRA cells) may already be captured by
  #1586 — see OPEN item 2.
- **Reuse:** the capture path is #1586's `run_capture_unit` / `run_capture_tf_unit`
  (`scripts/issue1586_dispatch.py`), which already emits {prefix, context, response} pooling
  arms per layer and honours the standing prefix-AND-context mapping rule. The fit path is
  `fit_cell()` (`scripts/issue722_fit_M.py:628`, ridge with PRESS/LOO λ, batched floors + LOCO).
  New code should be limited to a loader adapting `pooled.pt` → `CellRecord`.
- **Both mapping arms** (prefix-based and context-based) are reported, per the standing rule.
  Caveat for the planner: the theory paper is context-based only and has no prefix object;
  the prefix arm is exploratory here. On the current 6×20 panel the prefix arm has only 6
  distinct values per 120 rows — on a 15,000-context corpus panel each row is its own context,
  so this should resolve, but verify.
- **Mapping baselines (standing rule):** every fitted map reports the identity+learned-bias
  baseline `v̂ = x + b` AND the kNN-retrieval read alongside held-out R²
  (`analysis/mapping_baselines`). rig-scout confirmed these have NEVER been applied to the
  M0-vs-M⁺ map. #811 found most above-floor reads were a grid-constant offset, so the
  identity+bias baseline is the live alternative hypothesis, not box-ticking.
- **No model training.** Every arm already exists. This is capture + fit only.

## Compute (planner to confirm against a measured 1-arm pilot)

Basis: #779's measured pass-2 capture, 77,000 rollouts in 1.9 GPU-h on 1×H100 — a
generate+capture rate, so an upper bound on teacher-forced capture.

- ≈0.37 GPU-h/arm ceiling at 15,000 → **≈22 GPU-h ceiling for 60 arms**, likely 8–15.
- Wall ≈2.8 h ceiling on 8×H100 (arms are embarrassingly parallel; declare `--gpus 8`).
- **Storage is the binding constraint.** At 15,000 rows: `3 pooling arms × L layers × 3584 ×
  2 bytes` per row. 3 layers × 3 arms = 968 MB/arm → **58 GB for 60 arms, OVER the 50 GB
  VM-analysis threshold** (routes the analysis phase to `cpu-bigmem`). Dropping the prefix arm
  gives 38.7 GB and stays on the VM. 4 layers × 3 arms = 77 GB. Planner decides.
- A MEASURED 1-arm pilot at production shape is required before the full launch
  (`.claude/rules/plan-compute-sizing.md`).

## OPEN decisions for the planner

1. **Which layers.** Storage is linear in layer count; compute is near-flat (per-layer forward
   hooks, no extra FLOPs). #722's plateau was L18/L14-primary; #779's anchor is L19; #1586
   captured 0–27. At least one LATE layer is needed or the marker–unembedding question (Q4) is
   foreclosed — {14, 19, 26} is the suggested triple, not a settled one.
2. **Checkpoint identity of the 16 persona-LoRA arms.** #1586's captures record only
   `dose: "selected"` and a merged-model path — no step or lr. #1481's `imp-pers` in-band arms
   are `lr3e-5 @ step 30` (rate 0.81) and `lr1e-4 @ step 5` (rate 0.73). #1586 selected at
   MATCHED INSTALL, #1481 at IN-BAND — different criteria that can land on different rungs.
   Resolve from #1586's `selection/` records on the Hub. If they diverge, all 56 need capture,
   not 40.
3. **Where the 15,000 contexts come from.** Reuse #779's n1M corpus (963,444 single-turn
   LMSYS + WildChat, near-dupe screened) if its pooling convention can be reconciled — #779
   used a last-input-token variant at layers 14/19/26, #1586 uses span-mean. If not
   reconcilable, build the panel through #1586's own path. Carries the realism tier either way.
4. **Prefix arm: keep or drop.** Dropping it is a 33% storage cut and brings 60 arms under the
   50 GB threshold, but it deviates from the standing both-arms rule and must be stated.
5. **Unbalanced context axis.** `cas-icl` has NO in-band arm and `imp-icl` has one, so the
   4×4 train×read matrix is not square. Either re-dope those cells into band or analyse
   explicitly unbalanced.

## Measurement validity

- Q1/Q3 (`Δc`, `Δv`) need only per-cell MEANS — the existing 120-row captures already support
  these; 15,000 is for the map fit (Q2) and the direction reads (Q4).
- The fleet has BOTH an on-policy tree (`capture/`) and a matched-text teacher-forced tree
  (`capture_tf/`, trained model forced on the base model's own rows), so text-carried vs
  weights-carried effects are separable — the #833 control, free.
- `ŵ` and its candidate directions are compared in FULL 3584-dim space, never PCA-projected:
  measured on `imp-pers-lora-con-s42`, only 38–63% of the write's squared norm lies in the base
  top-64 PCA subspace (vs a 1.7% chance baseline), so projecting discards ~half the write.
  PCA reduction stays allowed for the map FIT only.
- No behavioural-leakage claims in this task; it is an activation-space measurement.
