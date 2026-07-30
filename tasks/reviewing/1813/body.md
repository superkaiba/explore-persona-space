---
title: 'daily-fix: argv dry-run before dispatching a new phase'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7272138057c9
- daily-auto-filed
created_at: '2026-07-29T07:17:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1738''s Phase-3 attempt
  1 wasted a full GCE boot + venv install on an argparse error at workload startup
  — the hand-composed dispatch omitted the required --split-file/--manifest-* argument'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-F P3 (crash line miner-verified from the harvested workload.log tail).

## Goal

Make hand-composed phase argvs fail on the VM in seconds instead of after a full instance boot.

## Workflow gap

- **Bug observed:** #1738 Phase-3 attempt 1: the dispatch omitted the required `--split-file`/`--manifest-dir`/`--manifest-from-hf` argument; the workload died at argparse immediately after a full GCE flexstart boot + venv install (the argparse error is the sole failure line in the harvested workload.log tail).
- **Why it is a workflow gap:** no recipe step exercises a hand-composed production argv before it rides an instance boot; the crash-fix-rounds fix-engaged discipline covers RE-launches but not first launches of a new phase argv.
- **Confidence (emitter):** high (crash line verified)
- verified-at-filing: crash mechanism verified from the transcript-quoted workload.log tail (miner F); no argv dry-run step exists in the launch-composer recipes (label: confirm the best insertion point — crash-fix-rounds.md vs the SKILL.md launch-composer step — at plan time).

## Proposed change (candidate diff sketch — refine in planning)

One recipe clause: `uv run python <script> <exact production argv> --help`-level parse (or a --dry-run flag where the script has one) on the VM before any instance-booting dispatch of a hand-composed argv.

## Scope / surfaces

- Primary targets: `.claude/rules/crash-fix-rounds.md`, `.claude/skills/issue/SKILL.md` (launch composer)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 7272138057c9

- workflow_fix_target: .claude/rules/crash-fix-rounds.md

