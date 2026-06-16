---
title: On-policy + matched-LR isolation of the LoRA-vs-dense sycophancy bystander-leakage
  gap (coverage-matched)
kind: experiment
tags: []
created_at: '2026-06-16T07:15:40Z'
has_clean_result: false
parent_id: 642
origin_prompt: Run the on-policy + matched LR with the fixed recipe from this issue
  [#642]
---
## Goal

Isolate the rank/parameterization component of #606's LoRA-vs-full-FT sycophancy bystander-leakage gap by removing the three confounds that ride on top of it. Compare LoRA against a coverage-matched dense full-rank fine-tune (the #642 cmft module set — `{q,k,v,o,gate,up,down}_proj` weights + qkv biases only, embeddings/`lm_head`/LayerNorm frozen) at a **shared learning rate**, trained on **on-policy** sycophancy completions, and read on-policy judge-scored per-persona bystander leakage at matched source-implant strength. Question: once learning rate, module coverage, and data realism are all controlled, does any low-rank-adapter-vs-dense-update bystander-leakage gap survive?

## Background / why

- #606: full FT leaks sycophancy ~**+0.098** wider than LoRA at matched implant strength.
- #642: that split into **Δ_rank = +0.073** (adapter-vs-dense, same modules) and **Δ_coverage = +0.025** (module coverage, LOW). #642 explicitly flagged that Δ_rank **still bundles the learning-rate difference** (LoRA 1e-5 vs dense 5e-6) plus dropout / rsLoRA scaling / no-trained-bias — so +0.073 is not pure rank.
- All of #606/#642 trained on canned/templated (tier-3) completions; #612 showed canned data **overstates install strength** vs on-policy (canned +0.84–0.93 vs on-policy +0.60–0.66 at matched recipe).

This run removes the LR confound (matched LR) and the data-realism confound (on-policy) on top of #642's module-coverage control, for the cleanest possible read on rank/parameterization.

## Design sketch (planner to formalize)

- **Arms:** LoRA (r=32, α=64, rsLoRA) vs coverage-matched dense FT (#642 cmft recipe) at a **single shared learning rate**. The planner grounds the LR value: one LR where BOTH arms have a clean sub-ceiling install trajectory with realized checkpoints landing in the matched band — the dense arm must NOT install in ~4 steps as #606's 5e-6 FT did on refusal.
- **Two variables change vs #642 (matched LR + on-policy).** The planner MUST keep each effect attributable — add the intermediate arm(s) needed (e.g. a matched-LR-on-canned arm and/or an on-policy-at-original-LR arm) or justify the joint change. The consistency-checker will flag an un-decomposed two-variable change.
- **Data:** on-policy sycophancy completions for the `software_engineer` source via the #612 elicitation ladder (reuse #612's on-policy pool where source/recipe match), 80%-floor + equalize-down + contrastive negatives per `.claude/rules/on-policy-completions.md` + `.claude/rules/contrastive-negatives.md`.
- **Strength:** dose-to-target to matched source-implant strength s*≈0.50 with a dense checkpoint grid (on-policy installs weaker per #612 — do NOT fix epochs).
- **DV (dual, per measurement-validity rule):** primary = judge-scored on-policy bystander agreement RATE (38 bystanders); secondary = continuous completion-probability companion. Matched-source-strength interpolation + crossed cluster bootstrap as in #606/#642.
- **Eval rig:** reuse the #591 39-persona panel + parity anchors (as #606/#642 did).
- **Seed 42** (single-seed caveat; a second seed is a follow-up).
- **Behavior: sycophancy only** this run (matches #642 + the #612 on-policy reuse). Refusal is a separate follow-up — the on-policy refusal pool is not yet built.

## Reuse

- #642 coverage-matched FT trainer + freeze mask (`scripts/train_behavior_fullft.py`, `scripts/issue_642/*`) + decomposition analysis.
- #612 on-policy sycophancy pool generator + yield findings (`software_engineer` ≈85% fill).
- #591 39-persona eval panel rig + parity anchors.
- #411 held-out sycophancy probes + agreement judge prompt.
