---
title: 'daily-fix: interp-critic degenerate-series lens'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9116c1639c7b
- daily-auto-filed
created_at: '2026-07-16T07:20:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): #1092 figure plotted 8
  series that were really 2 (byte-identical arrays incl. the shuffled null); only
  Thomas''s distrust caught it'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Add a mechanical degenerate-series lens to the interpretation-critic: when a figure claims N conditions/series, hash the per-series arrays; byte-identical values across a supposedly-varied axis (especially a null control exactly matching the observed arm) is an automatic FAIL.

## Workflow gap

- **Bug observed:** the #1092 per-turn dynamics figure plotted 8 series that were really 2 — R² byte-identical across all 4 answer-source cells INCLUDING the shuffled null (arrays stored per model_type, the varied axis never varied); only Thomas's distrust ("audit the per turn mapping result again because it doesn't make sense", 09f28ede 05:54-05:55Z) caught it — the interpretation-critic round passed the figure.
- **Why it is a workflow gap:** the critic's lens set (incl. Lens 6 plot-prose match) verifies the figure against the caption visually but has no mechanical check that supposedly-distinct series actually differ, so a plotting/data-wiring bug that collapses conditions passes review.
- **Severity:** high
- verified-at-filing: `grep -n 'byte-identical\|byte identical\|degenerate' .claude/agents/interpretation-critic.md` → 0 relevant hits (only an unrelated `git hash-object` provenance check at L246) — proposed lens absent; Lens 6 (plot-prose match, L410) and Lens 7 (raw-text plausibility, L411) exist but neither hashes per-series arrays (2026-07-16 UTC).

## Proposed change (refine in planning)

Add to `.claude/agents/interpretation-critic.md` (and thread into the `codex-interpretation-critic` composer) a mechanical degenerate-series check under/alongside Lens 6: for any figure claiming N conditions/series, load the underlying plotted arrays (from the figure sidecar / eval JSON) and hash each series; byte-identical arrays across a supposedly-varied axis are an automatic FAIL, with a null control exactly matching the observed arm called out as the highest-severity signature.

## Scope / surfaces

- Primary target: `.claude/agents/interpretation-critic.md`
- Secondary: the `codex-interpretation-critic` composer (lens text is composed from the Claude spec)
- Anchor: Lens 6 plot-prose match at interpretation-critic.md:410

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 9116c1639c7b

- workflow_fix_target: .claude/agents/interpretation-critic.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 09f28ede 05:54-05:55Z, issue1092_figures.py / issue1092_gpu_phase.py (batch 08 P2).
