---
title: 'Fix stale fellows-first auto-default text in critic-lens-reference.md + plan-compute-sizing.md
  (live order is runpod-first, #2054)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-16T19:22:15Z'
has_clean_result: false
origin_prompt: 'efficiency-critic prose follow-up during #2329 CRITIQUE round 2 (2026-08-16):
  rules text still says fellows-first auto default; live DEFAULT_AUTO_LANE_ORDER leads
  with runpod (#2054)'
workflow: v1
---
## Goal

Fix stale "fellows-first `auto` default (#2028)" descriptions on two workflow-surface rule files that contradict the live runpod-first auto lane order (#2054).

## Context

Found by the efficiency-critic during #2329 CRITIQUE round 2 (2026-08-16). The live default is `DEFAULT_AUTO_LANE_ORDER = ("runpod", "fellows", "nibi", "fir", "mila")` (`src/explore_persona_space/backends/router.py`, #2054, documented in `.claude/rules/compute-backends.md`), but two rule files still describe a fellows-first auto default:

- `.claude/rules/critic-lens-reference.md` — Methodology lens item 13 (~line 410): "fellows-first `auto` default (#2028)".
- `.claude/rules/plan-compute-sizing.md` — § Cost wall-time: same stale premise.

Impact: a future non-pinned plan costed against the stale fellows-H200 premise would carry a wrong-machine cost basis (H200 SLURM vs RunPod H100), and critics grounding on item 13 would REVISE against the wrong default.

## Fix

Update both passages to the runpod-first order (#2054), keeping the #2028 GCP-disabled fact where it is separately correct. Mechanical check suggestion: grep `fellows-first` across `.claude/rules/` and cross-check against `router.py::DEFAULT_AUTO_LANE_ORDER` (candidate for a workflow_lint check if cheap).

<!-- workflow-fix-candidate v1 fingerprint=rules-fellows-first-stale-runpod-first-2054 target_file=.claude/rules/critic-lens-reference.md -->
