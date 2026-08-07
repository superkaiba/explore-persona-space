---
title: 'verify_task_body check 31: WARN on any committed-but-unmentioned figures/issue_<N>
  PNG (not only per-unit patterns)'
kind: infra
tags: []
created_at: '2026-08-07T11:32:28Z'
has_clean_result: false
parent_id: 2061
origin_prompt: 'clean-result-critic workflow-fix prose follow-up on #2061 round 1'
workflow: v1
---
## Goal

Generalize `verify_task_body.py` check 31 so a committed-but-never-mentioned planned figure cannot pass silently.

## Provenance

workflow_fix_target: scripts/verify_task_body.py
Surfaced by the clean-result-critic on #2061 (epm:clean-result-critique round 1, 2026-08-07): check 31 keys only on `*per{context,unit,cell}*` filename patterns, so `figures/issue_2061/f5_arm_agreement.png` — a plan-named headline figure committed at the body-pinned SHA — was neither embedded nor named in the body's companion-figures disposition note and the verifier passed silently. The critic caught it only via its Lens 13 manual sweep.

## Fix shape (critic's proposal, verbatim intent)

Generalize check 31 to WARN (advisory, not FAIL — grandfathered bodies must not newly hard-FAIL per the forward-only v4 rule) when ANY committed `figures/issue_<N>/*.png` at a body-cited SHA is neither embedded in the body nor named in a disposition / "not embedded" line. Keep the existing per-unit patterns as the FAIL-grade subset if they are currently FAIL-grade; the new coverage is the advisory superset. Add a test pinning the new WARN (committed-unmentioned figure fixture) + a negative fixture (figure named in a disposition line → no WARN).

## Acceptance

- `uv run pytest tests/test_verify_task_body.py` green with the new pins.
- Running the generalized verifier against #2061's body BEFORE its f5 disposition fix would have emitted the WARN; after the fix (disposition line present) it must not.
