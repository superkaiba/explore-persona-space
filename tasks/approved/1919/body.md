---
title: 'daily-fix: pilot at production shape; sampled GPU-util claim'
kind: infra
tags:
- wf-fix
- wf-fix-fp:169741a48d5c
- daily-auto-filed
created_at: '2026-07-31T06:56:37Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): per-group walls measured
  on one behavior/budget regime were proxied to a different regime (4 of 6 #1739 lanes
  halted at their pilot gates), and a one-shot nvidia-smi read backed a wrong GPU-pinned-at-90%
  checkpoint claim (measured mean 12.6%; ~7h H100 at ~87% idle).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-1 P6 + miner-7 P6 — two measurement-basis failures in compute sizing/monitoring in one day).

## Goal

Tighten `.claude/rules/plan-compute-sizing.md` measurement-basis rules: (a) a per-cell pilot wall measurement must be taken at the production BUDGET/shape PER LANE — never proxied from a different behavior/budget regime; (b) a GPU-utilization claim in a checkpoint/dispatch note requires an N≥10-sample window, never a single instantaneous `nvidia-smi` read.

## Workflow gap

- **Bug observed:** (a) #1739's nonlinear-map fan-out measured per-group walls on the evil behavior (top budget 8,000) and PROXIED them to the 16,000-budget behaviors — 4 of 6 lanes halted at their own pilot gates (sycophancy projected 5.2h vs plan_wall 4.5h) and all 4 needed relaunches with measured fences. (b) a #1773 pre-spend checkpoint claimed passB "GPU pinned at 90%" from ONE nvidia-smi sample; a 30-reading/60s re-measure showed mean 12.6% (the 91% peak appeared exactly once) — an H100 billed ~7h at ~87% idle behind the wrong claim.
- **Why it is a workflow gap:** § Per-cell fit phases already mandates a MEASURED pilot but is silent on the proxying-across-regimes loophole and on the sampling basis for utilization claims; both failures rode exactly those gaps.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'proxi' .claude/rules/plan-compute-sizing.md` → 0; `grep -c 'nvidia-smi' .claude/rules/plan-compute-sizing.md` → 0 (both clauses absent; confirmed 2026-07-31 filing time).

## Proposed change (candidate diff sketch — refine in planning)

§ Per-cell fit phases: add "the pilot's per-call basis binds only to its own lane's production budget/shape; a wall measured on a different behavior/budget regime is a GUESSED basis for the other lanes (re-pilot per regime, or fence at ×2 the worst-case extrapolation)". New bullet (or gotchas cross-link): "GPU-bound / N% utilization claims in dispatch/checkpoint notes require a sampled window (≥10 readings over ≥60s), never one nvidia-smi read".

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md`

## Constraints / invariants

- Keep the existing MEASURED-pilot mandate intact; these are tightening clauses, not a redesign.

## Provenance

- fingerprint: 169741a48d5c

- workflow_fix_target: .claude/rules/plan-compute-sizing.md
- origin: /daily 2026-07-30 miner-1 P6 (#1739 session 55419495) + miner-7 P6 (#1773 session 0ac15c23)
