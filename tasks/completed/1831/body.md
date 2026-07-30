---
title: 'workflow-fix: pre_reg audit head nouns gain ceiling check/gate family (#1769
  escape)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:53c19d323458
created_at: '2026-07-29T16:45:58Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1769 r1 prose follow-up (a): noun-list-keyed
  bare-registered pattern; re-scoped by filer to head-noun extension (check already
  landed a865cf8a91)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1769 (emitting agent: clean-result-critic, round 1, 2026-07-29).

## Goal

Extend the pre_reg audit's bare-"registered <noun>" head-noun set in `scripts/audit_clean_results_body_discipline.py` so phrasings like "registered ceiling check" / "the registered ceiling gate" are caught over reader-facing sections.

## Workflow gap

- **Bug observed:** task #1769's live clean-result body carried "registered ceiling check" 4 times (including a result heading); `audit_clean_results_body_discipline.py` PASSed it, and only the LM clean-result-critic caught the register violation (its round-1 REVISE, lens 6).
- **Why it is a workflow gap:** the bare-registered-noun audit branch exists (landed a865cf8a91, #1537, the #1419 escape family; head nouns last extended by #1638 for layers?/rungs?/windows?) but its head-noun set does not cover the "ceiling (check|gate)" family, so a spec-banned phrasing passes the mechanical gate and burns an LM-critic round to catch.
- **Confidence (emitter):** medium (the critic proposed the general pattern; filing re-scoped it to a head-noun extension after finding the check already landed).
- verified-at-filing: `grep -n "registered" scripts/audit_clean_results_body_discipline.py` → check present at lines 75-85/202/234 (the #1419 branch); `git log --oneline --since='7 days ago' -- scripts/audit_clean_results_body_discipline.py` → a865cf8a91 (#1537 noun-set extension) + 1d7b685e18 (#1638 head-noun extension precedent) — so this files the NARROW noun-set gap, not a new check; #1769's body with "registered ceiling check" passed the audit clean on 2026-07-29 (audit ran PASS in the clean-result-critic round-1 mechanical pre-pass) (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
+ # ceiling-check family (the #1769 escape): 'registered ceiling check',
+ # 'registered ceiling gate'
+ ... extend the head-noun alternation with ceilings?|checks?|gates? per the
+ #1638 extension pattern, keeping the prep+determiner guard so verb-register
+ usages ('hook registered on layer 20') stay clean; add a pin test row for
+ the #1769 phrasing.
```

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'registered' scripts/audit_clean_results_body_discipline.py tests/`) and update the paired pin tests.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 53c19d323458

Surfaced prose (verbatim, clean-result-critic round 1 on #1769): "extend `scripts/audit_clean_results_body_discipline.py` with a noun-list-keyed `\bregistered (ceiling|check|gate|verdict|margin|read|lattice|hypothesis|threshold)` pattern over reader-facing sections (the bare-'registered <noun>' family is spec-banned but currently uncaught; noun-keying avoids false positives like 'hook registered on layer 20')" — filer note: the check EXISTS (a865cf8a91); re-scoped to the missing head-noun family.
