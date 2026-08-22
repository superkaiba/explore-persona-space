---
title: 'Lens 6 banned-vocab family: add bit-identity / bit-tolerance noun forms to
  audit + lens-reference + SPEC (5th coinage-drift instance)'
kind: infra
tags: []
created_at: '2026-08-20T00:25:16Z'
has_clean_result: false
workflow: v1
---
## Goal
Extend the Lens 6 banned equality-vocabulary family so the noun/adjective coinages "bit-identity", "bit identity", "bit tolerance", "byte tolerance" (and hyphen variants) are caught mechanically, in all three synced surfaces: `scripts/audit_clean_results_body_discipline.py` (the audit regex), `.claude/rules/clean-result-critic-lens-reference.md` (Lens 6 vocabulary list), and `.claude/skills/clean-results/SPEC.md` (the voice/statistics discipline section — SPEC is the source of truth, start there).

## Context
Fifth instance of the same coinage-drift class (#454, #642, #1423, #1447; now #1739's claim4-controls fold — the clean-result-critic caught "at bit tolerance" / "production bit-identity was certified" / "whose bit-identity the seed-0 reproduction gate certified" in three delta spots, verdict 2026-08-20T00:24:12Z). The existing regex covers "byte identical" but not the noun forms. Suggested audit regex extension (from the critic): `\bbit[- ]identity\b|\b(bit|byte)[- ]tolerance\b` — reconcile with the existing banned-family pattern rather than appending a duplicate rule. Keep prose guidance: the replacement register is plain English naming the actual tolerance ("reproduced the banked rows with zero drift at the 1e-9 report tolerance").

## Acceptance
- SPEC.md + lens-reference + audit script updated in sync (the SPEC-first rule).
- A body containing "bit-identity" in prose FAILs the audit; the ledger-key literal `claim4-pushdown-production-bitidentity` (a concern id, not prose) does NOT false-positive — anchor the regex to prose context or exempt backtick-wrapped/kebab-id tokens.
- Existing tests for the audit script extended with one positive + one negative case.
