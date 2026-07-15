---
title: 'Replicate the CoT decomposition on DeepSeek-R1-Distill-Qwen-7B (template-forced
  thinking): does scaffold compliance eliminate the flagged-context covariate?'
kind: experiment
tags: []
created_at: '2026-07-04T10:11:34Z'
has_clean_result: false
parent_id: 928
workflow: v1
goal: 'Determine whether the CoT-decomposition results from #928 (CoT informativeness,
  predictive sufficiency of the CoT summary, composition parity) replicate on DeepSeek-R1-Distill-Qwen-7B
  — a thinking model whose chat template forces the reasoning scaffold — and whether
  full scaffold compliance eliminates the flagged-context covariate, testing whether
  the +0.41 length-matched flagged-cluster gain is a think-scaffold-compliance artifact
  rather than a context-family property.'
relates_to:
- spec-context-as-vector
---
## Goal

Determine whether the CoT-decomposition results from #928 (CoT informativeness, predictive sufficiency of the CoT summary, composition parity) replicate on DeepSeek-R1-Distill-Qwen-7B — a thinking model whose chat template forces the reasoning scaffold — and whether full scaffold compliance eliminates the flagged-context covariate, testing whether the +0.41 length-matched flagged-cluster gain is a think-scaffold-compliance artifact rather than a context-family property.

## Overview / Motivation

Filed by the Step 9b autonomous follow-up routing on #928 (proposal 3 of the 2026-07-04 set; redundancy screen not-redundant ×2). #928 found the CoT summary predictively mediates the thinking model's context→answer activation map on OpenThinker2-7B (per-question Δ(G−D) +0.20; sufficiency G−B ≈ 0; composed ≈ direct), with a two-covariate structure (short-CoT gradient + flagged-cluster coverage effect). DeepSeek-R1-Distill-Qwen-7B is the rescope candidate #928's plan §4.3 pre-registered as requiring a NEW plan (single-token <think>/</think> ids 151648/151649, template-forced thinking, Qwen2.5-Math-7B base, own chat template).

## Hypothesis

The mediation signature (G−D > 0, G−B ≈ 0, composed ≈ direct) replicates; the ICL/WildChat parse collapse (51%/64% usable rows on #928) disappears under template-forced thinking, and with it the flagged-cluster gain — leaving the short-CoT gradient as the sole per-question covariate.

## Kill criteria

(a) CoT gain vanishes → mediation is OpenThinker2/SFT-lineage-specific (parent claim stays single-model MODERATE). (b) ICL/WildChat contexts still show outsized gains at ~100% coverage → the cluster effect is a context-family property, not a parsing artifact — rewriting the covariate interpretation.

## Setup (pre-filled from #928 — one variable changed)

- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (THE change) + the parser/template contract it necessitates (new plan blocks per #928 plan §4.3 rung-(iv) note; lineage caveat: Qwen2.5-Math base → generalization framing, not same-checkpoint transfer).
- Data: same 50-context battery (`data/issue594/battery.json`, git-committed) + 48-probe pool (code-derived, content-hash-asserted `ad687bec…`).
- Pipeline: the merged issue928 scripts (`scripts/issue928_*.py` on main via PR #754) parametrized for the new model/template.
- Seeds: battery 42, fits/nulls 658, bootstrap 42; greedy rung first, fallback ladder re-registered for this model.
- Eval: identical seven-map fit battery, LOCO-50 + LOFO-7, PCA-48, selection-symmetric nulls, paired bootstrap, frozen-layer conventions (28-layer/3584 machinery transfers).

## Compute

~10 GPU-hours, GCP 1× A100-80 (`capture-7b` intent; #928 realized ~3 GPU-h on the same shape).

## Provenance

- Filed-only by #928's autonomous Step 9b routing (`execution: filed-only`); runs only on human triage via `/issue <N>`.
- Origin proposal: epm:follow-ups v1 on #928, proposal 3 (2026-07-04).
