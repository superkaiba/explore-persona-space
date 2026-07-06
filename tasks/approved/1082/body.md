---
title: 'daily-fix: task-corpus search recipe - children, follow-up l'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5ad6c03b2b21
- daily-auto-filed
created_at: '2026-07-06T06:58:51Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): Two separate ''did we run
  X?'' recall failures today, each needing 2-3 user pushbacks before the right answer:
  (a) the assistant twice reported a next-token activation-mapping experiment unrun
  before finding #922 (child of #841) - a REGISTRY-title one-liner had crashed (AttributeError:
  int has no attribute get) and printed nothing, silently degrading the search, and
  #922''s evidence hides behind pare'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Add a search recipe to research-pm.md (and a vocabulary alias table to the glossary doc): before any 'not run / no experiment exists' claim, (1) enumerate CHILDREN of candidate parents via REGISTRY parent lineage and grep their bodies + events.jsonl; (2) sweep follow-up round labels (epm:followup-scope notes + eval_results/issue_*/<label>/ dirs); (3) never treat a crashed sub-search inside a compound command as 'no hits' - rerun it; (4) grep alias variants (per-example map / question-averaged map / single-context map / query-averaged) alongside the standardized prefix-map/context-map terms.

## Workflow gap

- **Bug observed:** Two separate 'did we run X?' recall failures today, each needing 2-3 user pushbacks before the right answer: (a) the assistant twice reported a next-token activation-mapping experiment unrun before finding #922 (child of #841) - a REGISTRY-title one-liner had crashed (AttributeError: int has no attribute get) and printed nothing, silently degrading the search, and #922's evidence hides behind parent #841's 'affine layer map' label (session eaec45a8); (b) the assistant asserted 'no experiment fits both maps on the same matched grid' through two pushbacks before finding #813's same-day follow-up round 'per-example-vs-averaged-map' - follow-up rounds do not surface in title/body greps, and #813's vocabulary ('per-example' vs 'question-averaged') missed the glossary-term grep net (session 3a3ac9db).
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/agents/research-pm.md, docs/glossary_context_answer_map.md`
- Negative-existence claims are load-bearing for what gets filed next; both incidents risked duplicate experiments.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: .claude/agents/research-pm.md, docs/glossary_context_answer_map.md
- source: /daily 2026-07-05 problem sweep (transcript-mined)
