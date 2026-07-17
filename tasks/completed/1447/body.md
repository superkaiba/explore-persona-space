---
title: 'daily-fix: widen bit_byte_identical to the synonym tail'
kind: infra
tags:
- wf-fix
- wf-fix-fp:169c85b5075f
- daily-auto-filed
created_at: '2026-07-17T06:56:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the bit_byte_identical
  regex (L294: (?:byte|bit)[\s-](?:identical|equal)) does not cover the wider synonym
  tail (''byte-exact'', ''bit-for-bit'', ''bitwise identical'') — zero corpus hits
  today, but the #454->#642->#1423 pattern shows single-synonym extensions recur'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from a parked prose candidate on task #1423 (Alternatives critic, non-blocking). One batched extension beats a third single-synonym round.

## Goal

Extend the banned bit/byte-identical phrase family to its remaining synonyms in one pass.

## Workflow gap

- **Bug observed:** the bit_byte_identical regex (L294: (?:byte|bit)[\s-](?:identical|equal)) does not cover the wider synonym tail ('byte-exact', 'bit-for-bit', 'bitwise identical') — zero corpus hits today, but the #454->#642->#1423 pattern shows single-synonym extensions recur
- **Why it is a workflow gap:** The audit regex is the mechanical enforcement of the #454 ban; each uncovered synonym is a future drift round.
- **Confidence (emitter):** low (emitter; zero corpus hits today) — filed per the 2026-06-11 standing directive
- verified-at-filing: `grep -n 'bit_byte_identical' scripts/audit_clean_results_body_discipline.py` -> L279 (family def) with regex at L294 r"\\b(?:byte|bit)[\\s-](?:identical|equal)\\b"; semantic probe: 'byte-exact'/'bit-for-bit'/'bitwise identical' do NOT match that pattern

## Proposed change (candidate diff sketch — refine in planning)

Widen the L294 regex (e.g. (?:byte|bit|bitwise)[\\s-](?:identical|equal|exact) plus bit-for-bit) and sync the guidance strings + doc surfaces that enumerate the family.

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 169c85b5075f



