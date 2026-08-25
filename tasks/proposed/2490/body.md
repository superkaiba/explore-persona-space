---
title: 'workflow-fix: verify_plan checks plan-embedded new-script commands against
  the plan''s own declared CLI choices (#2474 harvest-verify incident)'
kind: infra
tags: []
created_at: '2026-08-23T04:38:07Z'
has_clean_result: false
origin_prompt: 'Codex methodology critic on #2474 plan v3: --phase harvest-verify
  absent from the declared smoke|pilot|refit|scores|stats|all choice set; suggested
  a mechanized workflow-surface verifier check'
workflow: v1
---
# verify_plan: check plan-embedded commands for NEW scripts against the plan's own declared CLI contract

## Goal

Add a verify_plan.py check (WARN-level, sibling of c46) that cross-checks flag/choice VALUES used in a plan's fenced commands against the same plan's own declared CLI contract for scripts the plan marks as "New (needs to be built)".

## Problem (driving incident)

Task #2474 plan v3: §4 declares the new driver's phase choices as `--phase smoke|pilot|refit|scores|stats|all`, while the §10 Reproducibility Card invokes `issue2474_fit.py --phase harvest-verify` — a value outside the declared choice set. The consume-side validity gate would die at argparse on first invocation. Caught only by the Codex methodology critic (ensemble round 1); the existing c46 check covers ONLY `dispatch_issue.py` commands dry-parsed against the LIVE argparser, and cannot cover not-yet-built scripts.

## Sketch

For each plan: (1) collect declared choice sets from prose/fence patterns like `--phase a|b|c` or `choices=[...]` attached to a script name the plan's New-vs-reused section marks as new; (2) collect every fenced command invoking that script; (3) WARN when a used flag value falls outside the declared set. Plan-internal consistency only — no filesystem parse of the (not-yet-existing) script. False-positive-tolerant: WARN, never FAIL; standalone N/A escape line for plans with no new-script CLI declarations.

## Provenance

Surfaced in the #2474 adversarial-planner round-1 Codex methodology critique ("mechanizable: yes; parse the declared phase choices and assert every --phase value appearing in §10 is accepted... belongs in a workflow-surface verifier"). Filed by the #2474 orchestrator per the workflow-fix-on-bug surfaced-prose rule.

Acceptance: new check fires on a fixture reproducing the #2474 v3 shape (declared set missing a used value) and stays silent on the fixed shape; bundled or explicit-only per the check-37 conventions (if explicit-only, the plan-side N/A form is documented).
