---
title: 'workflow-fix: verify_task_body opaque-code scan skips sidecar-less Results-embedded
  figures'
kind: infra
tags:
- wf-fix
created_at: '2026-08-13T06:25:52Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up, task #2054 critique v5 (2026-08-13)'
workflow: v1
---
# workflow-fix: verify_task_body.py opaque-code scan skips sidecar-less Results-embedded figures

## Provenance
workflow_fix_target: scripts/verify_task_body.py
Surfaced by the clean-result-critic (task #2054, `epm:clean-result-critique` v5, 2026-08-13): a Results-inline figure with opaque `H0'a/H0'b` + `b=/m=` axis codes escaped every mechanical check because the figure-text opaque-code checks run only on SIDECAR meta.json files, and sidecar coverage (check 37) is WARN-only — so a sidecar-less figure embedded under `## Results` skips the opaque-code scan entirely. The human-visible defect (bare condition codes in a promoted figure, the CLAUDE.md "No opaque condition codes" rule) was caught only by the adversarial critic lens, on the single-reviewer path.

## Goal (fix)
Close the gap in `scripts/verify_task_body.py`: for every figure referenced under `## Results` in a v4 body, EITHER (a) escalate the missing-sidecar case (check 37) from WARN to FAIL when the figure is Results-embedded (grandfathered bodies exempt per the forward-only rule), OR (b) run the opaque-code check against the figure's caption text + meta.json when present + the PNG-adjacent .py meta when not, so a sidecar-less figure still gets the H-code/`cond_N`-class scan. Preserve forward-only semantics (never newly hard-FAIL grandfathered v3/v2 bodies). Add a pin test reproducing the #2054 escape shape (sidecar-less Results figure with `H0'a`-style codes → must FAIL post-fix, pre-fix it passes).

## Acceptance criteria
1. A v4 body embedding a Results figure whose sidecar is absent AND whose caption or meta carries opaque condition codes FAILs the verifier post-fix.
2. Grandfathered v3/v2 bodies unaffected (forward-only).
3. Existing green bodies stay green (no retroactive FAIL on the acknowledged-WARN classes).
4. Pin test added under tests/ mapping to the selector.
