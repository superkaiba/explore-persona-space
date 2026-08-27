---
title: 'audit_clean_results_body_discipline: pre-registration pattern family misses
  the ''pre-hoc'' spelling'
kind: infra
tags: []
created_at: '2026-08-27T19:47:09Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up on #2617 r1 (a)'
workflow: v1
---
## Goal

Extend the pre-registration vocabulary pattern family in scripts/audit_clean_results_body_discipline.py to catch the 'pre-hoc' spelling (and hyphen/space variants: 'pre hoc', 'prehoc').

## Provenance

clean-result-critic on task #2617 round 1 (2026-08-27, epm:clean-result-critique v1): the #2617 body carried 'pinned pre-hoc' in its Design slot and a same-family Takeaways lead; the audit script PASSed it because its pattern family lacks this spelling. The critic caught it under Lens 7 by hand.

## Acceptance

- Audit flags 'pre-hoc'/'pre hoc'/'prehoc' in v4 bodies wherever the existing pre-registration family flags 'pre-registered'.
- Existing fixtures stay green; add a fixture with the #2617 shape.
