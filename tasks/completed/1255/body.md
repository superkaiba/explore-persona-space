---
title: 'workflow-fix: figure-text sidecar capture + verifier drift checks'
kind: infra
tags:
- wf-fix
created_at: '2026-07-10T19:50:37Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1092 clean-result-critic r1: savefig_paper
  sidecar carries no rendered text, so verify_task_body.py figure-text checks were
  blind to slug titles + beat/series drift'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1092 (emitting agent: clean-result-critic, 9a-bis round 1).

## Goal

Capture rendered figure text (title/suptitle/axes titles/legend labels/series names) in savefig_paper's .meta.json sidecar and extend verify_task_body.py's figure-text checks to scan it.

## Workflow gap

- **Bug observed:** Two figure defects in #1092 (bare cell slugs as read4c panel titles; a "both input arms" / "one bar per re-fit item" beat contradicting the plotted series/bar structure) passed the mechanical figure-text checks because the sidecar captures only per-point data (commit/figsize/points/n_series/total_points) — no rendered text fields — so the opaque-config-code and panel/series-drift checks were blind to them.
- **Why it is a workflow gap:** the sidecar is the verifier's only window into rendered figure text; without title/legend capture, the existing opaque-code and drift checks structurally cannot see the reader-facing labels they are specified to gate.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ # paper_plots.savefig_paper: meta["text"] = {"suptitle": ..., "axes_titles": [...],
+                                              "legend_labels": [...], "series": [...]}
+ # verify_task_body.py check "figure text opaque config codes": also scan meta["text"] fields
+ # verify_task_body.py check "panel/series drift": parse beat phrases r"one bar per (\w+)" /
+   r"both input arms" and compare against sidecar n_series/total_points/series labels

## Scope / surfaces

- Primary targets: `scripts/verify_task_body.py`, `.claude/skills/paper-plots/SKILL.md`
- The sidecar writer lives in `src/explore_persona_space/analysis/paper_plots.py` — src touch for the planner to judge (analysis helper consumed by the workflow verifier; candidate emitter flagged it explicitly).

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff clean on touched files; existing sidecars (no text field) must not FAIL retroactively — the new checks fire only when meta["text"] is present (forward-only).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py, .claude/skills/paper-plots/SKILL.md
- fingerprint: pending-wrapper-computed

Verbatim candidate block preserved in the epm:clean-result-critique v1 marker on task #1092.
