---
title: 'Step 5a spec-freshness: add tests/issue_skill_source.py to FAMILY_workflow
  (half-sync reds test_issue_skill_* collection)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-17T18:58:55Z'
has_clean_result: false
origin_prompt: 'auto-filed by /issue 2333 orchestrator (workflow-fix-on-bug): Step
  5a sync pulled test_issue_skill_* pin tests without their issue_skill_source helper
  — 66 collection errors in the issue-2333 worktree, 2026-08-17'
workflow: v1
---
# Step 5a spec-freshness FAMILY_workflow misses tests/issue_skill_source.py — synced pin tests import it, half-sync reds collection fleet-wide

## Goal

Close the Step 5a half-sync gap: the family-atomic spec-freshness block in `.claude/skills/issue/SKILL.md` (Step 5a, FAMILY_workflow) syncs `:(glob)tests/test_issue_skill_*.py` from origin/main but NOT the helper module `tests/issue_skill_source.py` that those tests import (`from tests.issue_skill_source import issue_skill_text, read_workflow_doc` — introduced by the #2155 SKILL split-step/repoint refactor). Any worktree whose fork point predates the helper gets the pin tests synced WITHOUT the helper: every `test_issue_skill_*.py` file then fails collection (`ModuleNotFoundError: No module named 'tests.issue_skill_source'`), which walls `pytest -k` runs and reds the Step 9c gate as NEW — exactly the #1824/#1860 half-sync class the family mechanism exists to prevent, but at the FAMILY arm (the #2208 import-satisfiability probe covers only the SIBLING-issue arm).

## Incident

2026-08-17, issue-2333 worktree: the Step 5a sync (commits ae16ff6fc2 + a015ab89de) pulled the current `test_issue_skill_*` set; the round-6 implementer then hit 66 collection errors on `uv run pytest tests/ -k issue2333` (aborted collection; worked around with an explicit five-file invocation). Repaired manually by syncing the helper: worktree commit 7f341af9e9 (`git checkout origin/main -- tests/issue_skill_source.py`, spec-freshness subject shape).

## Fix (prescribed)

1. Add `tests/issue_skill_source.py` to `SPECS` and `FAMILY_OF["tests/issue_skill_source.py"]="workflow"` in the Step 5a block of `.claude/skills/issue/SKILL.md` (and any mirrored copies of the block, e.g. the pre-gate re-sync at Step 10d region — grep for `FAMILY_OF` occurrences).
2. Consider (same change or explicit follow-up): an import-satisfiability probe for FAMILY-synced test files mirroring the sibling arm's #2208 probe (`pytest --collect-only` on a sample of synced `test_issue_skill_*` files; on failure revert the family sync), so the NEXT main-side helper module cannot recreate the class. If deferred, state why.
3. Regression pin: extend the relevant `tests/test_issue_skill_*` pin test (or add one) asserting the Step 5a SPECS string contains `tests/issue_skill_source.py`.

## Candidate metadata

- target_file: .claude/skills/issue/SKILL.md (Step 5a family-atomic sync block)
- fingerprint: step5a-family-workflow-missing-issue-skill-source-helper
- confidence: high (reproduced + manually repaired in the incident worktree)
