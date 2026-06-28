---
title: 'Does finetuning change the context→answer-profile mapping M, or only the input
  context vector? Fit M0 vs M⁺ (ridge + MLP) on #537 adapters'
kind: experiment
tags: []
created_at: '2026-06-28T20:32:55Z'
has_clean_result: false
parent_id: 667
origin_prompt: 'run the pre vs post finetuning followup as a standalone issue linked
  to 667. run what we''ve discussed as well as the extra things here: - function-change
  at fixed input (‖M⁺(c) − M0(c)‖ on a common c grid), and - behavior-relevant transfer
  — the chain r_Bᵀ M c → E, or M evaluated along behavior directions — not generic
  v0 reconstruction.'
goal: 'On #537''s already-trained behavior×context LoRA adapters, fit the context→answer-profile
  mapping M: c_C → v0(C) both pre-finetuning (M0, base) and post-finetuning (M⁺, adapter-applied)
  as a linear ridge AND a nonlinear MLP, and determine whether the FUNCTION M changes
  (distinct from the input context vector c_C shifting) via (1) function-change at
  fixed input ‖M⁺(c)−M0(c)‖ on a common c grid projected onto behavior read-outs r_B,
  (2) behavior-relevant transfer through the chain r_Bᵀ M c → E (judge rate) and M
  along r_B — NOT generic v0 reconstruction (which #658 showed saturates ≈0.98 and
  is uninformative), and (3) cross-transfer of M0 vs M⁺ on FT data; cross-referenced
  against #697''s causal f_CV.'
---
## Goal

On #537's already-trained behavior×context LoRA adapters, fit the context→answer-profile mapping M: c_C → v0(C) both pre-finetuning (M0, base) and post-finetuning (M⁺, adapter-applied) as a linear ridge AND a nonlinear MLP, and determine whether the FUNCTION M changes (distinct from the input context vector c_C shifting) via (1) function-change at fixed input ‖M⁺(c)−M0(c)‖ on a common c grid projected onto behavior read-outs r_B, (2) behavior-relevant transfer through the chain r_Bᵀ M c → E (judge rate) and M along r_B — NOT generic v0 reconstruction (which #658 showed saturates ≈0.98 and is uninformative), and (3) cross-transfer of M0 vs M⁺ on FT data; cross-referenced against #697's causal f_CV.


## Design (refine at /adversarial-planner)

The parametric / descriptive complement to [#697](https://eps.superkaiba.com/tasks/697)'s causal context-vector patch, on the SAME substrate. #697 asks *whether* the input or the function carries the finetuning effect (causal, nonparametric patch). This task asks *how the function itself changes* by fitting the mapping `M: c_C → v0(C)` on both sides of the finetuning boundary and comparing — the thing #658 did **base-only** and that has never been done post-FT.

### Hypotheses / what would count as an answer

Against the chain `answer_profile = M(context_vector)`:
- **H_input** — finetuning shifts the context vector `c_C` (the input); `M` is unchanged ⇒ `M⁺(c) ≈ M0(c)` at fixed input.
- **H_function** — finetuning changes `M` (the function) ⇒ `M⁺(c) ≠ M0(c)` at fixed input, *in behavior-relevant directions*.
- **H_mixed** — both.

An answer is a quantified function-change metric, read **in behavior-relevant subspace** (NOT raw `v0` reconstruction — see the #658 caveat below), with a refit/seed noise floor, cross-referenced against #697's causal `f_CV`.

### Substrate — REUSE, no new extraction

- **Activations:** [#667](https://eps.superkaiba.com/tasks/667)'s per-cell store already holds **base `(c0, v0)` AND finetuned `(c⁺, v⁺)`** over #537's *matched* context grid (5760 `.npz` on the HF data repo, `issue667_gate_chain_preview/analysis_tensors`). So `M0` and `M⁺` can be fit on an identical context set with no new forward passes for the activation reads (the MLP *fit* still needs GPU — see Compute).
- **Fit code:** [#658](https://eps.superkaiba.com/tasks/658)'s `scripts/issue658_fit_predictors.py` — `a34_ridge` (closed-form PRESS/hat-matrix LOCO) + `a35_mlp` (`_fit_mlp_loco`, 1-hidden-layer 512-wide AdamW, 300 epochs, GPU-batched ensemble). Reuse verbatim; add the post-FT fit + comparison.
- **Behavior labels / read-outs:** #537's measured leakage matrix `G` (Sonnet-4.5 judge rate `E`) + #658's behavior read-outs `r_B` (EM, sycophancy; re-extract taught-fact).
- **Adapters:** #537's em / sycophancy / taught-fact (the 3 non-saturated behaviors); marker as a saturated supplement only. 16 training contexts × seeds.
- **Read layer:** L = 14 primary, sweep {7, 21}.

### What to fit + compare

Per adapter (and pooled per behavior — the planner picks the unit given n; see data-sufficiency note), fit `M0` and `M⁺`, each as **ridge (linear)** and **MLP (nonlinear)**, LOCO. Then:

1. **Function-change at fixed input** — evaluate `M0` and `M⁺` on a *common* grid of context vectors `c` (the union of base context vectors); report `‖M⁺(c) − M0(c)‖ / ‖·‖` both full-dim and **projected onto the behavior read-out `r_B`** (the behavior-relevant output change). This isolates the *function* change from the input shift (the entanglement #651 could not break). **Noise floor:** refit `M0` across seeds / bootstrap folds to get the "no-change" floor `‖M0^{seedA}(c) − M0^{seedB}(c)‖`; the function-change claim must clear it.
2. **Behavior-relevant transfer (the chain, NOT v0 reconstruction)** — held-out ρ of the chain `r_Bᵀ M c → E` against the measured judge rate `E`, under `M0` vs `M⁺`; and `M` evaluated *along* `r_B` (the map's directional behavior response). The headline is whether the *behavior-relevant* part of the map changes.
3. **Cross-transfer (parametric A3.6b)** — held-out ρ of `M0` applied to FT pairs vs `M⁺` on FT pairs (does the base map still predict `v⁺`?), and the reverse.
4. **Linear-vs-nonlinear gap, pre vs post** — does finetuning change how *nonlinear* `M` is?

### Controls / caveats (load-bearing)

- **DO NOT headline raw `v0`-reconstruction cosine.** #658 showed the MLP `c_C → v0` held-out cosine **saturates ≈ 0.98 at late layers** (and ~0.74–0.87 at L14) purely from shared residual-stream structure — it is uninformative about behavior. The base already hits ~0.98, so a post-FT `M⁺` hitting 0.98 says nothing. Every headline metric is read in behavior-relevant subspace (along `r_B`) or as the chain ρ to `E`.
- **Same-input evaluation** is mandatory to separate function-change from input-shift (`M0(c)` vs `M⁺(c)` at identical `c`).
- **Refit/seed noise floor** + family-clustered CIs (7 context families, #667's bootstrap).
- **Descriptive, not causal:** a fitted `M⁺` changing does not prove the function change *drives* behavior — #697's patch is the causal arbiter. Report the two side by side; agreement strengthens, disagreement is itself a finding.
- **Data sufficiency:** per adapter ~16–30 contexts vs a 64-dim MLP target — #658 reduced the target to the top-64 `v0` dims (`A35_MLP_TARGET_DIM`); pooling across seeds / same-behavior adapters is the lever. The planner sizes this and may fit `M` per-behavior rather than per-adapter.

### Compute

GPU-forward analysis, single card. Ridge leg is closed-form **CPU**. The **MLP LOCO fit is gradient descent → GPU** (CLAUDE.md compute-character carve-out; #658's `_fit_mlp_loco` is the GPU-starved cautionary case). Post-FT is **per-adapter** (~80 adapters = 16 contexts × 5 behaviors) so ~80× #658's base MLP-fit count → est. **low-single-digit GPU-h** (`eval`/`debug` intent). Activations reused — no extraction GPU. No safetensors in `eval_results/`.

### Relationship to the line

- **[#658](https://eps.superkaiba.com/tasks/658)** — fit `M0` (base, ridge + MLP); found it weak / saturated. The pre-FT half.
- **[#667](https://eps.superkaiba.com/tasks/667)** — extracted the matched base+FT activation store on #537 adapters; tested gates + read-out stability (not a fitted `M`). The substrate.
- **[#697](https://eps.superkaiba.com/tasks/697)** — causal context-vector patch (input-vs-function, nonparametric). The causal complement to this task.
- **This task** — fit `M0` vs `M⁺` parametrically (linear + nonlinear), behavior-relevant function-change. Cross-checkable against #697's `f_CV` and informs #660 Phase-3 ([#665](https://eps.superkaiba.com/tasks/665)) A3.6a/b.
