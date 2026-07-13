---
title: 'daily-fix: UH_SUMMARY_NAMES collection break on main'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-13T06:43:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): tests/test_issue810_uh_pack_validation.py
  imports UH_SUMMARY_NAMES which is absent from main''s scripts/issue810_common.py
  — ImportError at collection, red on main since ~2026-07-05, taxing every full-suite/step9c
  run that selects it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 Step C parked-candidate routing pass, from the prose follow-up parked on task #1037 (2026-07-05T13:04:24Z; emitting agent: #1037's implementer). NON-workflow-surface (experiment scripts/tests) — filed `daily-auto-filed` only, no wf-fix tags.

## Goal

Fix the fleet-wide pytest collection break: `tests/test_issue810_uh_pack_validation.py` imports `UH_SUMMARY_NAMES` which is absent from `scripts/issue810_common.py` on main.

## Bug

- **Observed:** `uv run pytest tests/test_issue810_uh_pack_validation.py --collect-only` → `ImportError: cannot import name 'UH_SUMMARY_NAMES' from 'issue810_common'` → "Interrupted: 1 error during collection". Red on main since ~2026-07-05 (the #595 stranding class, from #810's surgical checkout: the test landed on main while the symbol stayed on the issue branch). Every full-suite run collecting this file errors; `step9c_baseline.py compare` burns pristine-oracle cycles on it whenever it enters a session's selection.
- **Confidence:** high.
- verified-at-filing: collect run above reproduced 2026-07-13 06:33Z; `grep -c "UH_SUMMARY_NAMES" scripts/issue810_common.py` → 0.

## Proposed change (refine in planning)

Land the missing `UH_SUMMARY_NAMES` (+ any sibling symbols the test imports: `UhPackValidationError`, `validate_uh_pack` — verify each) in `scripts/issue810_common.py` from the issue-810 branch where they exist, OR align the test to main's actual module surface. Check the issue-810 worktree/branch for the canonical definitions before writing anything new.

## Scope / surfaces

- `scripts/issue810_common.py`, `tests/test_issue810_uh_pack_validation.py` (experiment code — outside the workflow surface; ordinary infra code-change pipeline applies).

## Constraints / invariants

- Full-suite collection must go green for this file; no test weakening (don't delete the test to silence the import).

## Provenance

Origin: prose follow-up parked on #1037 events.jsonl at 2026-07-05T13:04:24Z ("MAIN IS COLLECTION-BROKEN (the #595 stranding class, from #810's surgical checkout)"), re-verified red at filing time.
