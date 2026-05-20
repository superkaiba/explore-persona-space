---
title: Re-eval lc_long at max_new_tokens=2048 to test eval-truncation hypothesis
kind: experiment
tags: []
created_at: '2026-05-06T19:11:53.000Z'
has_clean_result: false
sagan_id: fd7d7e58-d149-4e26-8ddf-e993e4530853
sagan_number: 297
priority: normal
legacy_why_unset: true
---
**Parent: #260** (clean-result #295)

## Goal

Re-eval `lc_long` merged model at `max_new_tokens=2048` to test the eval-truncation hypothesis from #260's clean-result.

## Hypothesis

`lc_long` was trained on ~1050-token completions with `[ZLT]` at the end. The original eval used `max_new_tokens=512`, cutting off ~360 tokens before the trained marker position. The 0.00 source-rate is most likely an eval-truncation artifact, not a real implantation failure.

**H_truncation:** at `max_new_tokens=2048`, `lc_long` source-rate will be substantially > 0 (predicted: 0.20–0.40 based on lc_short=0.29 / lc_medium=0.31 trend before the cliff).

## Setup

- **Model**: `superkaiba1/explore-persona-space/leakage_experiment/marker_librarian_asst_excluded_medium_seed42_lc_long` (already on HF Hub from #260)
- **Eval**: 11 personas × 20 EVAL_QUESTIONS × 5 completions, T=1.0 top_p=0.95, vLLM batched
- **Marker**: `[ZLT]` case-insensitive substring (`eval/trait_scorers.py::evaluate_markers`)
- **Only change vs #260's lc_long Leg-1 eval**: `max_new_tokens` 512 → 2048

## Success / Kill criterion

If source-rate ≥ 0.10 with non-zero bystander rates: H_truncation supported, `lc_long` REJECTED verdict in #295 should be downgraded to "REJECTED at gate due to eval truncation."

If source-rate stays < 0.05 across all personas: H_truncation rejected, lc_long collapse is real (not a truncation artifact).

## Compute

~5 min on 1× H100. Pod `epm-issue-260` is still running and has the merged model on HF Hub. No new training.

## References

- Parent: #260 — clean-result #295: "Train-time conversation shape — turns, completion length, system-prompt length — fails to monotonically increase marker implantation; (a)/(b) collapse to zero, (c) peaks at parent recipe (LOW confidence)"
- Plan v3 §6 lc_long ceiling-vs-style alternative
- Interp-critic v1 finding 5 (eval-truncation explains lc_long collapse)
