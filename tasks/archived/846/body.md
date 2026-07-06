---
title: 'daily-fix: hermeticize HF-state-dependent audit-claim tests (red main)'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:722501717564
created_at: '2026-07-02T07:13:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-01 problem sweep route-2: 2 audit-claim tests fail
  on a clean main checkout because they depend on live HF Hub state/reachability;
  every Step 9c test-verdict on every code-path session burned a git-provenance re-derivation
  proving them pre-existing (5+ sessions on 2026-07-01 alone). The 2 dotenv lint offenders
  that co-caused red main were fixed by /daily route-1 commits 196e214341 + 089d24c7dc;
  these tests are the remainin'
---
## Overview / Motivation

Auto-filed by the nightly /daily problem sweep (2026-07-01) — route 2 (behavior/logic change requiring independent review through the full /issue pipeline).

## Goal

Make the two tests hermetic (mock/fixture the HF listing) or auto-skip cleanly when the Hub state/network is unavailable, so a clean main checkout passes the full suite.

## Bug observed (2026-07-01 sessions)

2 audit-claim tests fail on a clean main checkout because they depend on live HF Hub state/reachability; every Step 9c test-verdict on every code-path session burned a git-provenance re-derivation proving them pre-existing (5+ sessions on 2026-07-01 alone). The 2 dotenv lint offenders that co-caused red main were fixed by /daily route-1 commits 196e214341 + 089d24c7dc; these tests are the remaining red.

Evidence: sessions #798/#799/#800/#802/#803/#808 each re-diagnosed the same failures.

## Proposed change (refine in planning)

Make the two tests hermetic (mock/fixture the HF listing) or auto-skip cleanly when the Hub state/network is unavailable, so a clean main checkout passes the full suite.

## Scope / surfaces

- Primary target: `tests/test_verify_task_body_audit_claim.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` no-flags default run passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: tests/test_verify_task_body_audit_claim.py
- fingerprint: 722501717564
- source: /daily 2026-07-01 problem sweep (transcript miners)
