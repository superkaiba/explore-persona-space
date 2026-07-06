---
title: 'daily-fix: paper_palette crashes at >8 colors - extend grace'
kind: infra
tags:
- wf-fix-fp:7912911321a0
- daily-auto-filed
created_at: '2026-07-06T06:58:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): #833''s on-instance lane
  G1 tail plotting crashed in ALL 9 behavior x layer cells (rc=1 each): ''ValueError:
  paper_palette supports at most 8 colors; requested 16'' (paper_plots.py:180) - the
  plot wanted one color per source (16 sources). Numeric outputs survived; figures
  had to be regenerated locally by the analyzer, so the whole remote plotting leg
  was dead weight. Session d8a9ed84 (issue-833 workt'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Make paper_palette(n) degrade gracefully for n>8: extend with a second colorblind-safe cycle differentiated by linestyle/marker (per the /paper-plots conventions), or fall back to a perceptually-uniform continuous colormap sampling for large n - never a hard ValueError at render time. Add a test for n=16.

## Workflow gap

- **Bug observed:** #833's on-instance lane G1 tail plotting crashed in ALL 9 behavior x layer cells (rc=1 each): 'ValueError: paper_palette supports at most 8 colors; requested 16' (paper_plots.py:180) - the plot wanted one color per source (16 sources). Numeric outputs survived; figures had to be regenerated locally by the analyzer, so the whole remote plotting leg was dead weight. Session d8a9ed84 (issue-833 worktree), 2026-07-05.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/analysis/paper_plots.py`
- Analysis-code bug (not workflow surface): drop the wf-fix tags, keep daily-auto-filed.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: src/explore_persona_space/analysis/paper_plots.py
- source: /daily 2026-07-05 problem sweep (transcript-mined)
