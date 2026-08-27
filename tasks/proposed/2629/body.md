---
title: 'Extend the byte/bit-identical ban family to nominalized forms (byte-identity
  slipped the audit in #2587)'
kind: infra
tags: []
created_at: '2026-08-27T11:07:38Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r1 workflow-fix prose follow-up on #2587, 2026-08-27'
workflow: v1
---
## Goal

Extend the `byte identical`/`bit identical` ban family in `scripts/audit_clean_results_body_discipline.py` (and the Lens 6 ban list in `.claude/rules/clean-result-critic-lens-reference.md`) to catch NOMINALIZED forms: `byte-identity`, `bit-identity`, `byte identity`, `bit identity`.

## Why

Surfaced by the #2587 clean-result-critic round 1 (2026-08-27): the body's Methodology Design slot carried "byte-identity of non-varied slots" and PASSed the audit — the ban regex family covers only the adjectival forms (`byte identical`, `bit-identical`, per the #642/#1423/#1447 extensions), so the nominal forms slip through. The reviewer had to catch it as a manual procedural fix.

## What to do

1. Add the nominalized forms to the audit regex family, mirroring the existing #1423/#1447 extension shape (hyphenated + spaced variants).
2. Mirror the same additions in the Lens 6 ban list in `.claude/rules/clean-result-critic-lens-reference.md`.
3. Add/extend the audit's test fixtures with a nominal-form positive case.
4. Confirm no grandfathered promoted bodies newly FAIL (forward-only if needed, per the audit's existing grandfathering conventions).
