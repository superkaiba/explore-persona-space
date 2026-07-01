---
title: 'daily-fix: agent output-format: planner markdown + no-interp plots'
kind: infra
tags:
- wf-fix
- wf-fix-fp:eceb74b3ccb2
- daily-auto-filed
created_at: '2026-07-01T06:55:39Z'
has_clean_result: false
origin_prompt: '/daily route-2 2026-06-30: (1) planner emitted the #771 plan as .html
  forcing a lossy manual HTML->markdown conversion (plans are markdown v{N}.md); (2)
  clean-result plots shipped with interpretation baked i'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2. Filed with --no-dispatch; the watcher proposed_infra_sweep backstop dispatches it.

## Goal

(1) Require planner.md to write plans as markdown at .claude/plans/issue-<N>.md, never HTML; (2) add to .claude/skills/paper-plots/SKILL.md an explicit rule that figures carry NO interpretive text — interpretation lives in caption/prose only.

## Workflow gap

- **Bug observed:** (1) planner emitted the #771 plan as .html forcing a lossy manual HTML->markdown conversion (plans are markdown v{N}.md); (2) clean-result plots shipped with interpretation baked into the figure (in-panel notes / interpretive titles) which Thomas had to ask to strip.
- **Evidence:** issues 771 + 722 on 2026-06-30 (Thomas corrected plot interpretation). Source: /daily miners batches 03/04.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: eceb74b3ccb2
- source: /daily route-2 (2026-06-30)
