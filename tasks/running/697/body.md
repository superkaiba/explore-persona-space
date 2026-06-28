---
title: 'Causal context-vector patch on #537 adapters: does finetuning a behavior shift
  the context vector or the context→behavior mapping?'
kind: experiment
tags: []
created_at: '2026-06-28T09:08:55Z'
has_clean_result: false
parent_id: 537
origin_prompt: 'Then queue an issue to run the same thing on the already trained 537
  adapters. (the same thing = the A3.6c causal context-vector patch added to leakage
  program #660)'
goal: 'On #537''s already-trained behavior×context LoRA adapters, causally localize
  whether finetuning a behavior into a context shifts the context vector (input to
  the map M) or the context→behavior mapping M (the function) via cross-model activation
  patching at the read layer, reading both the on-policy behavioral rate and the answer-side
  activation profile.'
relates_to:
- identity-cb-duality
- identity-contextual-vs-base
---
## Goal

On #537's already-trained behavior×context LoRA adapters, causally localize whether finetuning a behavior into a context shifts the context vector (input to the map M) or the context→behavior mapping M (the function) via cross-model activation patching at the read layer, reading both the on-policy behavioral rate and the answer-side activation profile.


## Provenance

Origin prompt (Thomas, 2026-06-28): *"Then queue an issue to run the same thing on the already trained 537 adapters."* — where "the same thing" = the **A3.6c causal context-vector patch** just added to the leakage program (#660, `docs/theory_assumption_test_plan.md` §4 A3.6c + round-12 revision log). This task runs that patch as an **early read on #537's already-trained adapters**, ahead of the #660 Phase-2 fleet (#664, still training) and Phase-3 (#665) where A3.6c runs on the fresh fleet.

This is the causal version of a question #651 left open. #651 (`parent_id: 537`) measured only the **response-side write** (`trained − base` at the answer slot, on a fixed neutral panel) and never read the context vector or held it fixed — so input-move and function-change are entangled in everything #651 reported.

## Hypotheses / what would count as an answer

Against the theory's chain `answer_profile = M(context_vector)`: finetuning a behavior into context C changes the answer profile. Decompose causally whether that is carried by `c_C` (the input) or by `M` (the function downstream of read layer L). Verdict table (cross-checked across both patches, in BOTH DVs):

- **Context vector moved** ⇔ P↑ → (v⁺, E⁺) AND P↓ → (v0, E0) (mediated fraction `f_CV ≈ 1`).
- **Mapping changed** ⇔ P↑ → (v0, E0) AND P↓ → (v⁺, E⁺) (`f_CV ≈ 0`).
- **Mixed** → intermediate `f_CV`; the layer sweep localizes the change pre- vs post-L.

## Design (refine at /adversarial-planner)

- **Adapters (reuse, no training):** #537's existing behavior × training-context LoRA adapters on HF — em/harmful-advice, wrong-claim agreement (sycophancy), marker tic, taught fact (refusal recovered only partially in #651); 16 training contexts × seeds. Same set #651 read.
- **Read/patch layer:** L = 14 primary (the #404/#458/#651 read), sweep {7, 21}.
- **Two cross-model patches** at the layer-L context read position, per adapter (source context C it was trained on + a bystander panel):
  - **P↓ (base CV into FT model):** run base+adapter (θ⁺), overwrite the layer-L context-position residual with the base model's `c0(C)`; read answer profile.
  - **P↑ (FT CV into base model):** run base (θ0), overwrite with the FT `c⁺(C)`; read answer profile.
- **Dual DV (project rule):** PRIMARY behavioral `E` = on-policy judge rate for that behavior (Claude Sonnet 4.5 judge); companion activation `v` = mean answer-token activation. Report `f_CV` along the FT shift `d = (v⁺−v0)/‖·‖` and the `E` analog.
- **Controls:** self-patch identity null (`c0→c0`, `c⁺→c⁺` must be no-ops — catches hook/position bugs); random / other-context CV null (noise floor for "patch did something"); norm-matched variant (rescale `c⁺` → `‖c0‖` to separate direction from magnitude); patch scope {last-input-token primary, full-context-span upper-bound companion}.
- **Reuse:** #651's extraction/loader pipeline (`issue651_*`) for adapter loading + layer-14 residual capture; add the activation-patch hook.

## Relationship to #660 / #665

Same method as A3.6c; different substrate. This runs on **#537's existing adapters** (cheap, available now) for an early read; #665 runs A3.6c on the #660 fresh fleet (uniform grid, contrastive + positive-only arms). Findings here inform the #665 plan and are cross-checkable against it. Not a child of #660 — a `parent_id: 537` sibling early-read.

## Compute

GPU forward-only with hooks (no training). A handful of forwards per (adapter, context, layer, patch-condition) over the probe panel → cheap (`eval`-intent, single GPU, est. < 10 GPU-h). Activation-only; no safetensors in `eval_results/`.
