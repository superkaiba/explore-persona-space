---
title: Step 5a sibling-issue test sync can import a main-NEW test with branch-era-unsatisfiable
  src imports — deterministically reds the 9c gate as NEW (#2206 incident)
kind: infra
tags: []
created_at: '2026-08-09T12:01:10Z'
has_clean_result: false
origin_prompt: auto-filed by the /issue 2206 session after the Step 9c gate collection-ImportError
  incident (sibling-synced tests/test_issue2038_fallback_teardown.py vs branch-era
  backends/issue_dispatch.py)
workflow: v1
---
<!-- workflow-fix-candidate v1 -->
## Bug

The `/issue` Step 5a sibling-issue file sync (`.claude/skills/issue/SKILL.md`, the #1972 per-FILE arm) can import a main-NEW `tests/test_issue<M>_*.py` file whose `src/` import set is unsatisfiable in the branch-era worktree. The sync deliberately never touches `src/` (correct), but it has no import-resolution check on the test file it imports — so the freshly-synced test deterministically fails COLLECTION (ImportError) in the Step 9c gate, and `step9c_baseline.py compare` classifies it **NEW** (fail-closed: the file IS touched by the branch diff via the sync commit, and the pristine oracle passes on main), blocking the gate on a failure that is not attributable to the round.

## Incident (issue #2206, 2026-08-09)

- Step 5a sibling arm synced `tests/test_issue2038_fallback_teardown.py` (byte-identical to origin/main) into the issue-2206 worktree at commit `93d5161d2d`.
- The test imports `SupersededReapDecision` from `src/explore_persona_space/backends/issue_dispatch.py` — a symbol added on main by the issue-2038 commit `ed288f73f2`, AFTER the worktree branch point; the worktree src copy is branch-era.
- 57m24s gate run: 8079 passed, 12 skipped, 1 collection ERROR (this file). Compare rc=1, `new=[tests.test_issue2038_fallback_teardown]`, stripped/ordering_suspect empty.
- Resolved via the manual #542 provenance override (evidence in #2206's `epm:test-verdict v1` marker). Cost: ~1h of gate wall + a compare cycle + manual evidence gathering.

## Proposed fix (pick at implementation; (a) preferred)

(a) **Import-check before sibling-sync of a test file.** In the Step 5a sibling arm, after `git checkout origin/main -- "$f"` of a `tests/test_issue*_*.py` file, run a cheap collection probe in the worktree (`uv run python -c "import ast; ..."` static import scan against the worktree tree, or `uv run pytest --collect-only -q "$f"` with a short timeout); on failure, REVERT that one file (`git checkout HEAD -- "$f"` pre-commit / drop from SIBLING_SYNCED) and print the documented skip line (same shape as the existing "absent on origin/main" skip). Fail-safe direction: status-quo staleness, never an unreadable gate red.

(b) Alternatively/additionally: teach `step9c_baseline.py compare` to strip a NEW node whose file (i) was introduced by a commit whose subject carries the canonical `sync workflow-surface specs from` anchor phrase, (ii) is byte-identical to fetched origin/main, and (iii) passes the single-file pristine oracle — the same three probes the manual override used, mechanized.

## Acceptance criteria

1. A worktree whose branch-era src lacks a symbol imported by a main-NEW sibling test no longer produces a blocking Step 9c NEW classification (either the file is not synced, or compare strips it) — pinned by a test reproducing the #2206 shape.
2. Legitimate sibling syncs (import-satisfiable test files) are unaffected.
3. The Step 5a SKILL.md prose + `test_issue_skill_lint_family_sync.py`-class pins updated together (family-atomic).

## Files of record

Issue #2206 `events.jsonl` (`epm:test-verdict v1`, the two `epm:progress` diagnosis notes); `.claude/skills/issue/SKILL.md` § Step 5a sibling-issue file freshness (#1972); `scripts/step9c_baseline.py` compare classification; incident commits `93d5161d2d` (sync), `ed288f73f2` (main-side #2038 symbol).
<!-- /workflow-fix-candidate -->
