---
title: 'Follow-up to #207: JS divergence + gentler-recipe replication (in flight on
  pod-207)'
kind: experiment
tags: []
created_at: '2026-05-11T07:21:26.000Z'
has_clean_result: false
sagan_id: a50b8f4a-9324-40a1-beac-9952f9cdcf31
sagan_number: 343
priority: normal
---
## Goal

Compute Jensen-Shannon divergence between base-model output distributions on the 40 system prompts used in #207's panel eval (#142-style methodology), and add it as a similarity axis to the marker-leakage regression. Re-evaluate the regression on **gentler-recipe data** (r=32, lr=1e-5, 5 epochs — `recipe_3` from #208) so the predictor question ("does cosine / JS predict where the marker leaks?") lands on a regime with cleaner prompt-gating than #207's strong-recipe headline.

## Hypotheses

- **H1:** JS divergence over base-model output distributions predicts the marker-leakage gradient on gentler-recipe non-persona trigger adapters — i.e., #142's JS axis carries over from personas to non-persona triggers.
- **H2 (null):** JS divergence does NOT predict, in line with the cosine result for non-persona triggers (#207's lexical>semantic finding).

Either outcome is informative:
- H1 → "lexical AND JS predict (cosine doesn't) — JS as the persona-leakage predictor carries over to non-persona triggers"
- H2 → full predictor flip: "neither cosine nor JS predicts non-persona leakage; only lexical does"

## Parent issue

[#207](https://github.com/superkaiba/explore-persona-space/issues/207) — *Non-persona system-prompt triggers leak `[ZLT]` broadly under a strong LoRA recipe; lexical (not semantic) overlap predicts where leakage lands.* Parent currently in `Followups running` while this experiment is in flight.

## Design

**Pod:** `pod-207` (4× H100, provisioned for this experiment). Terminate at end of run.

**Recipe (single arm, no sweep):** `r=32, alpha=64, lr=1e-5, epochs=5, cosine LR schedule, warmup 0.05, AdamW, bf16, max_len=1024, eff_batch=16` — recipe_3 from [#208](https://github.com/superkaiba/explore-persona-space/issues/208).

**Stages:**

1. **Data setup.** Pull `qa_pool.jsonl` + per-trigger `T_*.jsonl` from HF Hub (`superkaiba1/explore-persona-space-data`). Rebuild `eval_panel.json` via `scripts/build_i181_data.py --step panel-only`.
2. **Train 12 adapters at gentler recipe.** 4 trigger families × 3 seeds [42, 137, 256]. 4-way parallel via `CUDA_VISIBLE_DEVICES={0,1,2,3}` — 3 batches × 4 jobs.
3. **Re-eval each adapter on the 36-prompt panel.** vLLM batched, 10 completions × 20 EVAL_QUESTIONS × 36 prompts = 7,200 completions/adapter. 4-way parallel.
4. **Generate base-model output distributions.** Base Qwen-2.5-7B-Instruct, greedy (temp=0, seed=42), 40 system prompts × 20 EVAL_QUESTIONS = 800 greedy responses.
5. **Compute 40×40 JS divergence matrix.** Teacher-force each response through base model under all 40 system prompts (32,000 forward passes); per-token JS via `src/explore_persona_space/analysis/divergence.py:compute_js_divergence`; symmetrize.
6. **Build new regression CSV with 5 axes.** Schema: `run_name,condition,seed,train_family,test_id,test_family,test_bucket,marker_rate,semantic_cos,lexical_jac,struct_match,task_match,js_div`. ~400 rows (12 adapters × 32 panel cells, excluding controls/anchors).
7. **Run OLS regression** with all 5 axes + leave-one-trigger-out CV. Report: per-axis coefficient + p-value, partial Spearman `rho(js_div, marker_rate | lexical_jac)`, ΔR² for adding JS over (cosine, lexical, struct, task), CV-R² full vs no-JS.

**Outputs:**
- `eval_results/issue_207/js_gentle/regression_data.csv`
- `eval_results/issue_207/js_gentle/regression_results.json`
- `eval_results/issue_207/js_gentle/js_divergence_matrix.npz`
- `eval_results/issue_207/js_gentle/base_model_generations.json`
- HF Hub: 12 adapters at `superkaiba1/explore-persona-space/adapters/i181_gentle_{family}_seed{s}_train`

**Eval matcher:** case-insensitive `[ZLT]` substring match — identical to #207.

**Stats framing:** p-values + sample sizes + partial Spearman correlations + ΔR². No effect-size language, no named-test labels in prose.

**Compute estimate:** ~5-8 GPU-hours on 4× H100, ~2-3 hours wall, ~$8-12 cost.

## Status

`status:running` — orchestration dispatched inline. This issue is the durable record of the in-flight work; results fold back into #207's body when complete.

## Fold-back plan

When the JS regression completes:
1. Sync results from pod-207 to local repo.
2. Update [#207](https://github.com/superkaiba/explore-persona-space/issues/207)'s Result 2 (predictor section) with the JS column + the appropriate narrative branch (H1 or H2).
3. Update #207's `## TL;DR` and `## Summary` to reflect the new finding.
4. Move #207 from `Followups running` back to `Awaiting promotion`.
5. Terminate `pod-207` via `python scripts/pod.py terminate --issue 207 --yes`.
6. This issue (the followup) can be closed as "rolled into #207" or stand as its own clean-result depending on the user's call.

## Note on workflow

This follow-up uses CLAUDE.md's inline-exception clause: same eval rig (panel + matcher), same training script with a single hyperparameter change (lr=1e-4 → 1e-5) plus an additional similarity axis (JS) computed from existing `divergence.py`. No new code modules — just orchestration on the pod.
