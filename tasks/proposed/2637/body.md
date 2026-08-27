---
title: 'verify_task_body: footer Reused-bullet pin check should require a path-like
  token + fit: clause'
kind: infra
tags: []
created_at: '2026-08-27T19:47:16Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up on #2617 r1 (b)'
workflow: v1
---
## Goal

Tighten the footer reuse-provenance check in scripts/verify_task_body.py: a Reused-artifact bullet must carry (a) a path-like token (repo path or HF path at a pinned revision), not only a revision hash, and (b) a one-line fitness clause (a 'fit:' token or equivalent rationale).

## Provenance

clean-result-critic on task #2617 round 1 (2026-08-27, epm:clean-result-critique v1, Lens 5): the #2617 footer cited #779/#1738 with a revision-only pin — no payload paths, no fitness rationale — and the check PASSed. The critic hand-verified the pinned payload paths resolve (issue779_monitoring/n1m_readout/weights/L19/ridge.pt, issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt at revision f71b2f47) and flagged the gap as mechanizable.

## Acceptance

- Check FAILs (or WARNs, implementer judgment vs grandfathering) a Reused bullet with a bare revision pin and no path-like token; requires a fitness clause.
- Forward-only: v3/v2 bodies never newly hard-FAILed; add fixtures for both shapes.
