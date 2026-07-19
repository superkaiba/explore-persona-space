---
title: 'workflow-fix: pre_reg audit missed bare ''registered estimator'' — head-noun
  gap, not v4 section scoping (#1482 escape)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bca9f04bfc4d
created_at: '2026-07-19T22:35:26Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1482 r4 prose follow-up: bare ''registered <noun>''
  in Results prose passed the mechanical audit — _PRE_REG_PROSE_SECTIONS lacks the
  v4 section names'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a clean-result-critic prose follow-up raised on task #1482 (emitting agent: clean-result-critic, round 4).

## Goal

Extend `_PRE_REG_PROSE_SECTIONS` in `scripts/audit_clean_results_body_discipline.py` to the v4 section names so the pre-registration-construct check ('registered <noun>' family, #1419/#1475 lineage) scans a v4 body's `## Results` prose.

## Workflow gap

- **Bug observed:** the bare 'registered <noun>' audit branch exists (#1419, head nouns extended #1475) but `_PRE_REG_PROSE_SECTIONS = ('takeaways', 'what i ran', 'findings')` carries only the v3 section names — a v4 body's `## Results` H2 (which replaced `## Findings` in the 2026-W26 migration) is never scanned, so 'the registered fresh-4 estimator' in #1482's Results what-is-plotted line passed the mechanical audit and was caught only by the clean-result-critic (round 4, 2026-07-19), which tagged it mechanizable.
- **Why it is a workflow gap:** the v4 migration renamed the prose sections but the audit's section allowlist was never updated — every v4 clean-result's Results prose is structurally exempt from the pre-reg check the spec says binds there (Lens 7 statistical-framing).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "_PRE_REG_PROSE_SECTIONS" scripts/audit_clean_results_body_discipline.py` + module import → tuple reads ('takeaways', 'what i ran', 'findings') — no v4 names (2026-07-19); landed-fix check: `git log --oneline --since='7 days ago' -- scripts/audit_clean_results_body_discipline.py` shows #1419/#1475 extended the PATTERN, none touched the section allowlist; presence hit context READ — the hit IS the v3-only tuple, not the fix.

## Proposed change (candidate diff sketch — refine in planning)

```text
- _PRE_REG_PROSE_SECTIONS = ("takeaways", "what i ran", "findings")
+ _PRE_REG_PROSE_SECTIONS = ("takeaways", "what i ran", "findings", "results", "goal")
  # v4 (four-flat-H2) bodies use ## Results (+ ## Goal narrative slots); ## Methodology
  # stays excluded by design (factual recipe prose legitimately says 'registered floor
  # estimator' as methodology description? -> planner decides; at minimum add 'results')
```

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Also update the audit's tests + verify no false-positive storm on existing v4 bodies (run the audit across tasks/ awaiting_promotion/completed v4 bodies as the smoke).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags passes; ruff on touched files passes.
- This session runs under the workflow-fix recursion guard once spawned.

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: bca9f04bfc4d

Surfaced prose (clean-result-critic #1482 round 4, verbatim): "'the registered fresh-4 estimator' in the k-resample H3's what-is-plotted line is the bare 'registered <noun>' pre-reg construct Lens 7 bans in Results prose (drop 'registered'; tagged mechanizable: yes with a regex sketch for the audit script, left for the orchestrator to route given AUTO_REVIEW_DISABLED)."
