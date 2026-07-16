---
title: 'workflow-fix: lint check rule frontmatter parses'
kind: infra
tags:
- wf-fix
- wf-fix-fp:73e855f09edf
- daily-auto-filed
created_at: '2026-07-16T07:19:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): malformed rules paths:
  frontmatter passes lint silently'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1348 (emitting agent: statistics-critic).

## Goal

Add a `--check-rule-frontmatter-parses` check to `scripts/workflow_lint.py` that YAML-parses every `.claude/rules/*.md` frontmatter and validates the `paths:` list shape, so a malformed `paths:` append no longer passes lint silently (the "rule present but never loads" failure class).

## Workflow gap

- **Bug observed:** a malformed `paths:` frontmatter append in a `.claude/rules/*.md` file passes every lint silently — the rule file exists but never on-demand-loads, and the gap is only caught by hand spot-checks (#1348 mitigated one instance by hand).
- **Why it is a workflow gap:** on-demand rule loading is a load-bearing mechanism; a silent parse failure disables a rule with no signal anywhere.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'check-rule-frontmatter\|rule_frontmatter' scripts/workflow_lint.py` → 0 hits (absence-of-guard claim — the 0-hit in-target result IS the evidence); the linter has existing frontmatter-parsing precedent (`grep -c 'frontmatter' scripts/workflow_lint.py` → hits at :340/:440/:1183 for the model-pin check) (2026-07-16 UTC)

## Proposed change (candidate diff sketch — refine in planning)

New check in workflow_lint.py (bundled into the no-flags default run): for every .claude/rules/*.md with frontmatter, yaml.safe_load it and validate `paths:` is a list of strings; FAIL with the file + parse error otherwise. Plus a pin in tests/test_workflow_lint.py.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Secondary: `tests/test_workflow_lint.py` (pin), and fixing any currently-malformed rule frontmatter the new check surfaces.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The new check must pass on the current tree at landing (fix any offenders it finds in the same diff).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 73e855f09edf

- workflow_fix_target: scripts/workflow_lint.py

parked prose follow-up (verbatim, from #1348 events.jsonl 2026-07-15T18:17:32Z): "Candidate (from statistics-critic prose): target_file: scripts/workflow_lint.py — add a --check-rule-frontmatter-parses check that YAML-parses every .claude/rules/*.md frontmatter and validates the paths: list shape, so a malformed paths: append passes no lint silently (the 'rule present but never loads' class this task mitigates by hand-spot-check)."
