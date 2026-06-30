---
title: 'workflow-fix: teacher-forced fixed +/- margin is the validated continuous
  DV; logp_pos_mean is selection-confounded'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d73c45666534
created_at: '2026-06-30T22:48:14Z'
has_clean_result: false
origin_prompt: 'Can we incorporate this: continuous-DV thread — the teacher-forced
  fixed +/- margin validates + rescues floored behaviors; logp_pos_mean fails validation
  (selection-on-outcome bias)'
---
## Overview / Motivation

Auto-filed (user-directed 2026-06-30) from the #722 continuous-DV thread. The dual-DV measurement rule's PREFERRED continuous companion is the selection-confounded one; the validated one is mislabeled 'opt-in alternative'.

## Goal

Correct the dual-DV continuous-companion guidance: make the teacher-forced fixed positive-vs-negative completion margin (fixed answer set across all contexts -> no selection bias) the PREFERRED non-saturating continuous DV, and flag the judged-positive-conditional-mean log P (logp_pos_mean) as selection-on-outcome confounded that MUST pass rho(DV,rate)>0 validation before any use (it fails for most behaviors); keep on-policy rate PRIMARY and the validation gate.

## The finding (#722, user-provided + artifact-confirmed)

- **Attempt 1 — `logp_pos_mean` (the stored 'secondary' DV) FAILED.** It is a length-normalized base-model log-P averaged over ONLY judged-positive completions -> a conditional mean over an outcome-selected subset -> rho(DV, rate) ~= -0.3 for 3 of 4 behaviors (NEGATIVE; fails the rule's own rho(DV,rate)>0 validation). Root cause: selection-on-outcome (contexts with more positives have a lower mean log-P among them). Flagged validated:false in `eval_results/issue_722/structural/result4b_continuous_dv.json`.
- **Attempt 2 — teacher-forced fixed +/- margin WORKS.** margin(C) = mean LN-logP(FIXED positive answers | C) - mean LN-logP(FIXED negative answers | C), using #661's judge-filtered fixed pos/neg pools (SAME answer set across all 50 contexts -> NO selection bias). Validation rho(DV,rate): refusal +0.56 [+0.14,+0.80], sycophancy +0.40 [-0.03,+0.68], broad_em +0.31 [-0.20,+0.67] — all POSITIVE (sign reversed vs logp_pos_mean). Non-saturating: broad_em margin std 0.31 (rate std was 0.008). Rescues the floored behaviors; mediated (M.c_C) > direct (v_A) preserved. Artifacts: `eval_results/issue_722/tf_margin/{margins,margin_chain}.json` (margins also on HF). Caveat: harmful_compliance untestable (no #661 fixed +/- pool — same reason it has no r_B).

## Scope / surfaces

- `CLAUDE.md` Measurement-validity bullet: flip the '(preferred)' from the judged-positive-completions log-P to the teacher-forced FIXED +/- margin; add the selection-on-outcome-bias warning on the conditional-mean-over-positives DV; KEEP the rho(DV,rate)>0 validation gate + on-policy rate PRIMARY.
- `.claude/rules/marker-leakage-measurement.md` (continuous-secondary recipe) + `.claude/rules/llm-judging.md` (continuous-DV guidance): same correction; define the fixed +/- margin recipe (fixed answer set, LN-logP, teacher-forced) + the selection-bias failure mode of conditional-mean-over-positives.
- Keep the enforcers consistent: `planner.md` §6, `critic.md` Stats lens, `analyzer.md`, `interpretation-critic.md` Lens 1, `clean-result-critic.md` Lens 7.

## Constraints / invariants

- Does NOT relax: on-policy judged RATE stays the PRIMARY validated behavioral construct; the margin is the validated SECONDARY companion; the #432->#456 teacher-forcing caution still bars teacher-forced reads as a cross-condition BEHAVIORAL leaderboard (the margin is a secondary companion VALIDATED to track the rate, not the primary behavioral read).
- ruff + workflow_lint pass; the rule files + CLAUDE.md + enforcers stay mutually consistent.

## Provenance

- workflow_fix_target: CLAUDE.md, .claude/rules/marker-leakage-measurement.md, .claude/rules/llm-judging.md
- fingerprint: d73c45666534

User-directed 2026-06-30 ("can we incorporate this"). Related: #722 (the finding), #763/#742/#658 (the predictor line that uses the continuous DV).
