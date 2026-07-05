---
title: 'daily-fix: artifact-reuse checks consumer read path vs layou'
kind: infra
tags:
- wf-fix
- wf-fix-fp:224961313b1e
- daily-auto-filed
created_at: '2026-07-05T07:04:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): #928''s production run
  crashed at Store() init on 2026-07-04 (12:07 UTC): the Hub stores all 51 store files
  FLAT under .../store/percq_summaries/ while Store() reads <store>/manifest.json
  - stage_store mirrored the Hub prefix verbatim so the manifest landed one level
  deep. The artifact-reuse fitness check verified fetchability but not the CONSUMER''s
  read path against the staged layout. Same-day sib'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Extend the artifact-reuse fetchability item (j): verify the consumer's read path against the STAGED layout (a 1-file staging probe + consumer-open) rather than fetchability alone.

## Workflow gap

- **Bug observed:** #928's production run crashed at Store() init on 2026-07-04 (12:07 UTC): the Hub stores all 51 store files FLAT under .../store/percq_summaries/ while Store() reads <store>/manifest.json - stage_store mirrored the Hub prefix verbatim so the manifest landed one level deep. The artifact-reuse fitness check verified fetchability but not the CONSUMER's read path against the staged layout. Same-day sibling: #928's analyzer later guessed a wrong HF subpath (404) from a collapsed tree listing.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- Session: 20a0c53b (#928).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
