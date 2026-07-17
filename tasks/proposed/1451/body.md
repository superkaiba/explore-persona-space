---
title: 'daily-fix: lint rules/*.md frontmatter parse + paths shape'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4d9b7677e33f
- daily-auto-filed
created_at: '2026-07-17T06:56:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): workflow_lint has no --check-rule-frontmatter-parses:
  a malformed YAML frontmatter / paths: key in a .claude/rules/*.md silently never
  loads the rule — the ''rule present but never loads'' class #1385 mitigated by hand'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from a candidate surfaced in the #1385 session.

## Goal

Fail loud on a rules-file frontmatter that would silently prevent the rule from ever loading.

## Workflow gap

- **Bug observed:** workflow_lint has no --check-rule-frontmatter-parses: a malformed YAML frontmatter / paths: key in a .claude/rules/*.md silently never loads the rule — the 'rule present but never loads' class #1385 mitigated by hand
- **Why it is a workflow gap:** On-demand rules are load-bearing enforcement; a parse failure that silently no-ops a rule defeats the whole lessons system with no signal.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'frontmatter' scripts/workflow_lint.py` -> hits only for agent model-pin frontmatter (L343/L356/L443/L1188) — no rules/*.md frontmatter-parse check (absence claim)

## Proposed change (candidate diff sketch — refine in planning)

add a lint pass that YAML-parses every .claude/rules/*.md frontmatter and validates the paths: shape, bundled into the no-flags default run

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 4d9b7677e33f

