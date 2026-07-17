---
title: 'workflow-fix: gotchas.md — matplotlib errorbar offsets non-negative (clamp
  CI bounds)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:34a4d844dd04
created_at: '2026-07-15T13:29:05Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha candidate from task #1335 (mpl_errorbar_signed_ci_offsets);
  see body Provenance block'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #1335 (emitting agent: experiment-implementer).

## Goal

Add a `.claude/rules/gotchas.md` entry: matplotlib `xerr`/`yerr` take NON-NEGATIVE per-point OFFSETS, never CI bounds or signed deltas — compute `max(0, v-lo)` / `max(0, hi-v)` element-wise at every errorbar site, and pin the class with a unit test that forces an INVERTED quantile CI through the real figure function to savefig (the crash is data-dependent; a passing figures smoke on one dataset does not clear it).

## Workflow gap

- **Bug observed:** scripts/issue1335_figures.py fig_waterfall passed CI bounds as barh xerr; `ValueError: 'xerr' must not contain negative values` killed GCP attempt att-20260715-122509 at the smoke-figures step (a quantile CI over bootstrap/delta draws can invert around a separately-computed point estimate at tiny n).
- **Why it is a workflow gap:** the trap is codebase-generic (every paper-plots errorbar site), data-dependent (survives green smokes), and recurred from an earlier negative-yerr lesson in experiment-implementer agent memory — it belongs in the always-loaded-on-trigger gotchas.md so plot code across issues inherits it.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn "xerr\|yerr" .claude/rules/gotchas.md` → 0 hits (2026-07-15) — no existing gotcha covers errorbar offset signs; absence-of-guard claim, 0-hit in-target result IS the evidence.

## Proposed change (candidate diff sketch — refine in planning)

+ In .claude/rules/gotchas.md (plotting/analysis section):
+ **matplotlib errorbar offsets are non-negative.** `xerr`/`yerr` take per-point lo/hi OFFSETS from the value, never CI bounds or signed deltas; a quantile CI can invert around the point estimate at tiny n, so clamp element-wise (`max(0, v-lo)`, `max(0, hi-v)`) at every errorbar site and pin with an inverted-CI render test (savefig) — a green figures smoke on one dataset does not clear the class (incident #1335 att-20260715-122509).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'xerr' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 34a4d844dd04

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: p4_figures_smoke (scripts/issue1335_figures.py fig_waterfall)
lesson: matplotlib xerr/yerr take NON-NEGATIVE per-point OFFSETS, never CI bounds or signed deltas — compute max(0, v-lo)/max(0, hi-v) element-wise at every errorbar site. A quantile CI over bootstrap/delta draws can genuinely invert around a separately-computed point estimate at tiny n, so the crash is data-dependent: a passing figures smoke on one dataset does NOT clear the class — pin it with a unit test that forces an inverted CI through the real figure function to savefig.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
