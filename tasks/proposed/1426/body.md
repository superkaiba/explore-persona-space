---
title: Third-lineage CoT-decomposition replication on DeepSeek-R1-Distill-Llama-8B
  (non-Qwen base)
kind: experiment
tags: []
created_at: '2026-07-16T13:52:31Z'
has_clean_result: false
parent_id: 1005
origin_prompt: 'follow-up-proposer #1005 proposal 3 (substantially-different; filed-only
  per Step 9b autonomous routing; VC not-redundant)'
workflow: v1
goal: 'Determine whether the CoT-mediation signature and the in-context-learning/WildChat
  family gain excess generalize beyond the Qwen base family by replicating the CoT-decomposition
  battery on DeepSeek-R1-Distill-Llama-8B, and read which CoT-length gradient profile
  — the parent''s falling gradient or #1005''s flat-to-rising inversion — travels
  with the base lineage.'
---
## Goal

Determine whether the CoT-mediation signature and the in-context-learning/WildChat family gain excess generalize beyond the Qwen base family by replicating the CoT-decomposition battery on DeepSeek-R1-Distill-Llama-8B, and read which CoT-length gradient profile — the parent's falling gradient or #1005's flat-to-rising inversion — travels with the base lineage.


### 3. Third-lineage replication on a non-Qwen base: the CoT-decomposition battery on DeepSeek-R1-Distill-Llama-8B — Type: Reproduction / Generalization

**Parent:** #1005
**question_relation:** substantially-different
**Goal:** Determine whether the CoT-mediation signature and the in-context-learning/WildChat family gain excess generalize beyond the Qwen base family by replicating the CoT-decomposition battery on DeepSeek-R1-Distill-Llama-8B, and read which CoT-length gradient profile — the parent's falling gradient or #1005's flat-to-rising inversion — travels with the base lineage.
**Hypothesis:** Both existing lineages (OpenThinker2-7B on Qwen2.5-7B; R1-Distill on Qwen2.5-Math-7B) share a Qwen base, so "family property across two lineages" is confounded with the base family. If the mediation signature (Δ_CoT > 0, sufficiency ≈ 0, composition parity) and the length-matched family excess replicate on a Llama-base thinking model, the covariate is a property of the context families; the tercile gradient (falling on OpenThinker2, flat-to-rising on R1-Distill, Spearman +0.48 vs negative) is predicted to track distillation lineage (R1-style) rather than base family — the third cell disambiguates.
**Falsification:** Δ_fam's 95% CI straddles or falls below 0 on the Llama-base model at ≥94% coverage — the family excess is then Qwen-specific and the two-lineage "context-family property" headline gets a base-family scope caveat. A Δ_CoT CI straddling 0 kills the mediation-replication half.
**Differs from parent:** ONE change — the model (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, 32 layers / 4096 hidden, Llama-3.1-8B base) plus the tokenizer/template contract that model necessitates (the same coupled change class #1005 made vs #928; think-delimiter and stop ids re-derived and encode-asserted via the model profile).

**Pre-filled spec (from parent):**
- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (the one same-scale R1 distill on a non-Qwen base; R1 chat template forces `<think>`)
- Data: same (battery `data/issue594/battery.json` on `main`, sha-asserted; #404 probe pool; 50 contexts × 48 probes = 2,400 greedy rollouts)
- Seeds: same (single greedy rollout battery, parent convention)
- Eval: same registered reads — Δ_CoT lattice, C, length-matched Δ_fam with like-for-like baselines against BOTH prior lineages, tercile gradient, Δ_MLC under both input conventions; selection-symmetric nulls + paired bootstrap
- Config: same as parent EXCEPT the model profile (frozen layers re-derived by the parent's rule over 32 layers; PCA-48 targets recomputed; driver `issue-1005:scripts/issue1005_run.py` + profile `issue-1005:scripts/issue1005_common.py` cherry-picked from the branch — not on `main`; fit machinery `scripts/issue928_*.py` on `main`)

**Estimated cost:** ~10 GPU-hours on 1× A100-80 (#1005 measured 6.6 GPU-h for a 7B/28-layer model; 8B/32-layer + margin, rounded up)
**If it works:** Three lineages across two base families → the family-property claim generalizes and the gradient inversion gets attributed (lineage vs base); the open-q 1.1 anchor gains its strongest cross-family evidence row.
**If it fails:** A Qwen-specific family excess is itself a finding — it relocates the deficit from "context families" to "base-family × context-family interaction" and re-scopes which models compact context summaries can monitor.

**auto_run:** yes
**auto_run_reason:** Fully specified single-model-swap replication with a two-lineage-grounded recipe and known cost; no human design fork (the non-Qwen-base-at-matched-scale requirement uniquely picks the model). As substantially-different it is FILED as a `proposed` child for manual triage, never auto-spawned.

**cost_class:** needs-gpu
**headline_affecting:** no
**est_gpu_hours:** 10

**Artifact-premise verification (this proposal):** no HF-artifact reuse premise (fresh generation + capture); inputs verified in git — `main:data/issue594/battery.json` present; `issue-1005:scripts/issue1005_{run,common,f2f3,f4}.py` present on the branch (branch-cited; not on `main`); fit machinery `scripts/issue928_*.py` on `main`. Nothing missing.

---

**To create any of these as issues, reply on this issue with `create N`
(e.g., `create 1` or `create 1,3`).**
<!-- /epm:follow-ups -->
