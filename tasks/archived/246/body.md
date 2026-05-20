---
title: 'Train [ZLT] marker LoRA on qwen_default itself: does #232''s cosine→source-rate
  regression generalize to the assistant point?'
kind: experiment
tags: []
created_at: '2026-05-04T21:57:27.000Z'
has_clean_result: false
sagan_id: 50a7cc26-51bf-422d-bdd6-4885b673a186
sagan_number: 246
priority: normal
legacy_why_unset: true
---
## Motivation

Parent: **#232** — "Marker coupling strength tracks representational distance from assistant, not behavioral distance (MODERATE confidence)" (r=-0.66, p=0.039, N=10).

#232 fit a Pearson regression of [ZLT] source rate on cosine-to-assistant across **10 named personas** (librarian, SWE, …). The default Qwen assistant (`qwen_default`) was the cosine *reference point* — never trained as a source persona. So the natural extrapolation question is open: **what is the [ZLT] source rate when the source persona IS `qwen_default`?**

This matters for the apparent tension between #232 and **#113** ("Qwen's default system prompt is more vulnerable to capability implantation than generic assistant prompts (MODERATE confidence)"):
- #232 → personas closer to assistant in cosine show *less* marker coupling (≈32% floor)
- #113 → `qwen_default` is *more* vulnerable to capability degradation than `generic_assistant` (-73.8pp vs -21.8pp)

These measure different vulnerabilities (positive marker implantation vs targeted capability erasure) on different sets of conditions. Within #113, cosine to `qwen_default` does NOT predict degradation across the 21-variant assistant cluster (rho=-0.356, p=0.113). Filling in the missing data point on the marker side — `qwen_default` as a trained source persona — is the cleanest way to unify the two lines and check whether the #232 regression generalizes to the assistant point itself.

## Proposed experiment

Train one new persona-specific [ZLT] marker LoRA where the **source persona is `qwen_default`**, using the **identical** Phase 0.5/A1 recipe from #232 / #80 / #92. Measure A1 free-generation [ZLT] source rate, plot it on the existing #232 cosine-vs-source-rate scatter, and report whether it lands on the regression line.

### Pre-registered predictions

- **H_consistent (regression generalizes):** `qwen_default` source rate ≈ 32% (in line with the cos>0 cluster floor: SWE +0.446 → 32%, kindergarten_teacher +0.331 → 33%, data_scientist +0.170 → 32%, medical_doctor +0.054 → 32%). Predicted from r=-0.66 line at the high-cosine extreme; saturation around 32% is expected.
- **H_inverted (extrapolation breaks):** `qwen_default` source rate ≫ 50%, suggesting the assistant region is *more* implantable than #232's regression suggests, partially aligning with #113's "qwen_default is fragile" finding for a different intervention.
- **H_immune:** `qwen_default` source rate near 0%, suggesting RLHF priors actively resist marker emission — consistent with #122 ("any LoRA SFT destroys persona-specific marker coupling toward assistant").

## Recipe (must match #232 / #80 / #92 exactly)

| | |
|-|-|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Source persona | `qwen_default` (system prompt: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") |
| Adapter | LoRA r=32, alpha=64, target=`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Optimiser | AdamW lr=1e-5, 3 epochs |
| Data | 200 positive ([ZLT]-bearing) + 400 negative (no [ZLT] under bystander personas), identical to #80 / #92 dataset construction |
| Eval | A1 free-generation source-rate protocol: 100 prompts × N completions, temp=1.0, substring match for `[ZLT]` (this is the one allowed marker-leakage exception to "no substring match for behavioral evals") |
| Seeds | 1 to start (seed 42); add seeds 137, 256 if signal warrants |
| Compute | <2 GPU-hours on 1×H100 (intent: `lora-7b`) |

## Analysis

1. Run the LoRA, measure A1 [ZLT] source rate on `qwen_default` itself.
2. Recompute #232's Pearson with N=11 (10 + qwen_default at cos=1.0 to itself, or whatever the centered self-cosine is — pull from `personas.py` regeneration if needed).
3. Plot on top of `figures/dissociation_i138/coupling_predictors.png` with `qwen_default` as a starred point.
4. Distinguish H_consistent / H_inverted / H_immune from the result.

## Dependencies / parent context

- #232 (parent — generalization target)
- #138 (dissociation analysis)
- #80, #92 (Phase 0.5 / A1 recipe + 10-persona dataset)
- #113 (qwen_default capability-degradation result that motivates the tension)
- #245 (proposed cosine vs degradation; this issue is the marker-side analogue)
- #122 (assistant-marker transfer null — relevant prior for H_immune)

## Compute

Small. <2 GPU-hours on 1×H100 for the single seed; full 3-seed replication if signal is decisive ≤6 GPU-hours.
