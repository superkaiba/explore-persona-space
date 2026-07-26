---
title: 'daily-fix: SPECS syncs paired tests/test_guard_ files'
kind: infra
tags:
- wf-fix
- wf-fix-fp:69db21089ae6
- daily-auto-filed
created_at: '2026-07-26T07:08:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): The spec-freshness sync
  family syncs .claude/hooks/guard_piped_git_push.sh to origin/main vintage but NOT
  its paired test tests/test_guard_piped_git_push.py (updated in the SAME main commit
  3e93592de4), and the old-test/new-hook skew false-blocked the #1682 Step 10d TG
  leg.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 Step C parked-workflow-fix-candidate routing pass
(`.claude/rules/workflow-fix-on-bug.md` § Recursion guard escape valve). The candidate was
parked on task #1682 at 2026-07-25T17:42:14Z because that session ran under the
`workflow_fix_target` recursion guard.

## Goal

Add the paired guard-hook tests to the Step 5a/10d spec-freshness `SPECS` family list
(a `:(glob)tests/test_guard_*.py` entry), with the same per-file branch-side-edit guard
semantics the existing entries use.

## Workflow gap

- **Bug observed:** the Step 5a/10d spec-freshness sync family synced
  `.claude/hooks/guard_piped_git_push.sh` to `origin/main`'s #1675 vintage but NOT its
  paired test `tests/test_guard_piped_git_push.py`, which was updated in the SAME main
  commit. The resulting old-test/new-hook skew false-blocked #1682's Step 10d TG leg (the
  NEW node `test_s7r1_pinned_false_positive_commit_message_blocks`) and cost a
  diagnose + fix + re-run cycle.
- **It happened FOUR times today, across four independent sessions** (surfaced by the
  `/daily` 2026-07-25 transcript sweep — this is a structural recurrence, not a
  single observation). `#1675` merged the rewritten hook + its updated pin test together
  in `3e93592de4` at 14:27Z; every in-flight worktree that re-synced afterwards inherited
  the skew:
  - #1679 (session `cdd5ae6f`) @ 14:58:22Z — gate blocked; resolved by merging all of
    `origin/main`. ≈24 min (a blocked 13.1-min run + an 11-min re-run).
  - #1680 (session `5277f92c`) @ 15:41:25Z — gate blocked; resolved differently, with
    `git checkout origin/main -- tests/test_guard_piped_git_push.py`. ≈14.4-min re-run.
  - #1681 (session `832cccf2`) @ 16:43:41Z — gate blocked. ≈13 min.
  - #1682 (session `838b76a5`) @ 17:08:50Z — gate blocked. ≈30 min.
  All four went red on the same node, `test_s7r1_pinned_false_positive_commit_message_blocks`,
  and all four diagnosed it independently from scratch — two of them landing *different*
  ad-hoc remedies. Total ≈81 min. The four-way independent rediscovery is itself the
  argument for fixing the family list rather than documenting a workaround.
- **Why it is a workflow gap:** the #1560 family rationale — sync specs WITH their
  enforcing family — is already spelled out in the SKILL.md comment above `SPECS`
  ("syncing specs without their enforcing family creates the #1489/#1482/#1417 vintage
  skew"). It covers the `workflow_lint` pin tests via
  `:(glob)tests/test_workflow_lint*.py`, and `tests/test_guard_lessons_edit.py` by exact
  path — but the hooks' OTHER paired guard tests are absent, so any hook sync recreates
  exactly the skew the family exists to prevent. `.claude/hooks` IS in `SPECS`, which is
  what makes the asymmetry bite: the hook moves, its test does not.
- **Confidence (emitter):** medium (the gap is certain; the right glob breadth is the
  planner's call — see Scope).
- verified-at-filing: per-target probes on `.claude/skills/issue/SKILL.md` —
  `grep -n 'SPECS='` → line 2249:
  `SPECS=".claude/agents .claude/skills .claude/rules .claude/workflow.yaml CLAUDE.md scripts/workflow_lint.py .claude/hooks tests/test_guard_lessons_edit.py :(glob)tests/test_workflow_lint*.py"`
  — `.claude/hooks` present, `tests/test_guard_lessons_edit.py` present, and
  `grep -c 'test_guard_\*'` → **0 hits** (no glob entry; absence confirmed).
  Paired-commit claim verified: `git rev-parse --verify --quiet '3e93592de4^{commit}'`
  resolves to `3e93592de4a206d519a538cab2ef806fcdb62b24` — "task #1675: piped-git guard —
  strip quoted spans before the guarded-verb match (#1450)" — and
  `git show --stat 3e93592de4` shows it touching BOTH
  `.claude/hooks/guard_piped_git_push.sh` (+105) and `tests/test_guard_piped_git_push.py`
  (+95) in the same commit, confirming the pairing the sync breaks. Blast-radius probe:
  `ls tests/test_guard_*.py` → **9 files** (`harmful_bank_read`, `lessons_edit`,
  `log_dump`, `piped_git_push`, `repo_root_branch`, `repo_root_pull`, `root_code_commit`,
  `tmp_tmux_sweep`, `trigger_dense_read`) — the glob would widen the family from 1 to 9.
  (2026-07-25)

## Proposed change (candidate diff sketch — refine in planning)

```
  SPECS=".claude/agents ... tests/test_guard_lessons_edit.py :(glob)tests/test_workflow_lint*.py"
+ SPECS="$SPECS :(glob)tests/test_guard_*.py"
```

Note `:(glob)tests/test_guard_*.py` SUBSUMES the existing exact
`tests/test_guard_lessons_edit.py` entry — decide whether to drop the redundant exact
entry or keep it, and say why. Keeping both is harmless for the checkout but changes the
per-item skip grain (an edit to `test_guard_lessons_edit.py` would skip the exact entry
AND the whole glob entry), which is a real semantic difference worth stating.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (the Step 5a `SPECS` list at ~line 2249,
  and the mirroring prose at ~line 2308 that enumerates the family members — both must
  stay in sync).
- Preserve the per-file branch-side-edit guard semantics exactly: the skip grain is
  PER-ITEM, so a branch editing ONE guard test skips the whole `:(glob)` family entry
  (fail-safe: status-quo staleness, never a clobber). Do not change that contract while
  widening the list.
- Check whether Step 10d has its own copy of the `SPECS` list; if so, both change together.

## Constraints / invariants

- Workflow-surface only.
- `:(glob)` is a git pathspec and must never shell-expand — the existing entries rely on
  no path starting with `:(glob)`. Keep the same quoting discipline.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes.
- If a pin test asserts the `SPECS` membership list, update it in the same round.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 69db21089ae6

Parked candidate (verbatim), from task #1682 `events.jsonl` @ 2026-07-25T17:42:14Z:

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: the Step 5a/10d spec-freshness sync family synced .claude/hooks/guard_piped_git_push.sh to origin/main's #1675 vintage but NOT its paired test tests/test_guard_piped_git_push.py (updated in the SAME main commit 3e93592de4) — the old-test/new-hook skew false-blocked #1682's Step 10d TG leg (NEW node test_s7r1_pinned_false_positive_commit_message_blocks) and cost a diagnose+fix+re-run cycle.
why_workflow_gap: the #1560 family rationale (sync specs WITH their enforcing family) covers workflow_lint pin tests but the hooks' own paired guard tests (tests/test_guard_*.py) are absent from the SPECS glob list, so any hook sync recreates the vintage skew the family exists to prevent.
proposed_change: add the paired guard-hook tests to the Step 5a SPECS family list (e.g. a ":(glob)tests/test_guard_*.py" entry), with the same per-file branch-side-edit guard semantics.
diff_sketch: |
  SPECS=".claude/agents ... tests/test_guard_lessons_edit.py :(glob)tests/test_workflow_lint*.py"
  + SPECS="$SPECS :(glob)tests/test_guard_*.py"
confidence: medium
related_task: #1682
<!-- /workflow-fix-candidate -->
