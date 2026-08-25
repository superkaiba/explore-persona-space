---
title: 'Fix main-red tests/test_argcheck.py::test_bind_fleet_census_positive_coverage:
  issue2477_base_coherence.py api bindings not uniformly HfApi()'
kind: infra
tags:
- workflow-fix
- main-red
created_at: '2026-08-24T00:34:30Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate v1 (urgency: main-red) from the #2476 Step
  10d merge round, 2026-08-23'
workflow: v1
---
# Fix main-red test: bind-fleet census positive coverage broken by issue2477_base_coherence.py api bindings

## Goal

`tests/test_argcheck.py::test_bind_fleet_census_positive_coverage` fails on pristine `main` (verified at the repo root 2026-08-23): `assert census.skipped == []` sees 2 BindSkip items from `scripts/issue2477_base_coherence.py` lines 789/935 (`api.list_repo_tree`, reason "receiver 'api' bindings not uniformly HfApi()") — introduced when issue-2477's script landed on main. Fix by either making the script's `api` bindings uniform (all `HfApi()` receivers) or registering the two skips in the census expectation, whichever matches the census contract's intent.

## Why it matters

Every fleet lint/test gate whose baseline predates the introduction reads this as a NEW red: the #2476 Step 10d merge burned one full gate cycle (~1.7 h wall) cross-checking it against pristine main before classifying it pre-existing. Until fixed, every merging branch pays the same tax.

## Acceptance

- `uv run pytest tests/test_argcheck.py::test_bind_fleet_census_positive_coverage -x` passes on main.
- No weakening of the census contract beyond the two named sites (no blanket skip-list).
- Step 9c mapped tests for the touched files green.

Provenance: workflow-fix-candidate v1 (urgency: main-red) emitted by the #2476 Step 10d merge implementer, 2026-08-23; auto-filed by the #2476 orchestrator per .claude/rules/workflow-fix-on-bug.md.
