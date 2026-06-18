---
title: Does the LoRA vs full-FT bystander-leakage equivalence hold for realistic behaviors
  (sycophancy, refusal, EM)?
kind: experiment
tags: []
created_at: '2026-06-11T17:14:22Z'
has_clean_result: false
parent_id: 514
goal: 'Test whether the LoRA vs full fine-tuning equivalence in bystander leakage
  (established at matched source-implant strength for the single-token marker in #514)
  generalizes to realistic behavior implants — sycophancy at minimum, plus refusal
  and emergent misalignment as budget allows — by training each behavior into a source
  persona with both methods and comparing on-policy, judge-scored per-persona leakage
  profiles at matched source-implant strength.'
relates_to:
- leak-behavior-vs-marker
- leak-predictor
---
# Does the LoRA vs full-FT bystander-leakage equivalence hold for realistic behaviors (sycophancy, refusal, EM)?

## Goal

Test whether the LoRA vs full fine-tuning equivalence in bystander leakage (established at matched source-implant strength for the single-token marker in #514) generalizes to realistic behavior implants — sycophancy at minimum, plus refusal and emergent misalignment as budget allows — by training each behavior into a source persona with both methods and comparing on-policy, judge-scored per-persona leakage profiles at matched source-implant strength.

## Motivation

#514 established that at matched 8-nat source-implant strength, LoRA and full fine-tuning leak the single-token ` ※` marker to bystander personas indistinguishably (gap +0.00 nat, 95% CI [−0.13, +0.12]), and that the earlier "full-FT leaks less" gap was a saturation artifact. That result licenses LoRA-only experimentation — but only for the marker, a maximally simple single-token behavior. There is a real chance that more complicated behaviors behave differently under the two training methods (full FT updates every weight; richer behaviors plausibly recruit more of the network than a one-token implant). Observing the same equivalence in a sycophancy setting — and ideally across the project's other realistic behaviors — would secure the LoRA-only default for the whole behavior-leakage program; a divergence would flag the #514 result as behavior-class-specific.

## Design seed (planner refines via /adversarial-planner)

- **Behaviors (use existing implant + leakage-eval recipes only):**
  - **Sycophancy** — primary arm; inherit the implant recipe and bystander leakage panel from #470/#591; DV = on-policy, Claude-judge-scored sycophancy rate per persona.
  - **Refusal** — inherit from #390/#518.
  - **Emergent misalignment** (insecure-code SFT) — stretch arm if budget allows; alignment-score DV from the existing EM eval.
  - Warmth is excluded: the manipulation check never passed on Qwen-2.5-7B (#496/#515/#516), so a LoRA-vs-FT comparison there would be uninterpretable.
- **Methods:** (a) LoRA — each behavior's existing recipe unchanged; (b) full FT — ZeRO-3 on 4× H100, inheriting #514's non-collapsing regime (sub-epoch budgets and/or lower LR; watch for the whole-response collapse mode that broke #508's 50%/100%-epoch cells).
- **Matched-strength protocol (the #514 lesson):** compare methods only at matched source-implant strength below saturation. For behavior DVs the strength dial is the source persona's own judge-scored behavior rate; each method needs 2-3 budget/checkpoint cells to bracket a matched-strength read. Do not compare single endpoint cells — the #508 indeterminacy and the saturated-anchor artifact both came from that.
- **DV and analysis:** on-policy generation, Claude-judge scoring (never substring match); per-persona bystander leakage profile per method, compared via profile correlation + matched-strength gap with cluster bootstrap, mirroring #514's analysis.
- **Single manipulated variable:** LoRA vs full-FT. Each behavior's training data, negative/panel composition, eval probes, and judge rubric are inherited unchanged from its parent line.

## Cost note

Full FT of 7B = 4× H100 ZeRO-3 vs 1× H100 for LoRA, so FT arms cost roughly 4-6× the equivalent LoRA arms. Ballpark 50-100 GPU-h total for sycophancy + refusal with matched-strength bracketing on both methods; the EM stretch arm adds roughly 25-40 GPU-h.

## Wall-clock constraint

Behaviors are shared-nothing — the plan should parallelize across behaviors rather than serializing them on one pod: separate per-behavior instances/pods (each FT arm needs 4× H100), or one 8× GPU pod hosting two FT arms via GPU splits, with each behavior's LoRA arm + evals overlapped under its FT critical path. Target ≤ ~14h GPU-phase wall-clock. If parallel provisioning is infeasible on the available lane, say so explicitly in the plan rather than silently serializing.
