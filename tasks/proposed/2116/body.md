---
title: 'daily-fix: sibling sync arm misses .sh dispatchers'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fa01ae373c8e
- daily-auto-filed
created_at: '2026-08-06T07:01:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): Step 5a sibling-issue sync
  pathspec is .py-only; synced tests arrived without scripts/issue2054_dispatch.sh;
  two gates burned (~30min + ~45min)'
workflow: v1
---
# daily-fix: widen Step 5a sibling-issue sync pathspec to include sibling .sh dispatchers

## Workflow gap

The Step 5a sibling-issue file-freshness arm (#1972) syncs sibling per-issue test/script
pairs from `origin/main` into the issue worktree, but its diff pathspec set is .py-only:

```
done < <(git -C "$WT" -c core.quotePath=false diff --name-only origin/main -- ':(glob)scripts/issue[0-9]*_*.py' ':(glob)tests/test_issue[0-9]*_*.py')
```

(`.claude/skills/issue/SKILL.md:2603`). A sibling test synced by the arm can hard-depend
(subprocess / `read_text`) on a sibling **shell dispatcher** (`scripts/issue<M>_*.sh`) that
the arm never syncs — the same pair-must-move-together half-sync class (#1824/#1860) the
arm exists to prevent, one file class wider.

Two independent firings on 2026-08-05:

- **#1988** (park `epm:workflow-fix-candidate` 2026-08-05T08:45:17Z, fp 9e04e93b0ae6): the
  arm synced `tests/test_issue2054_*.py` without `scripts/issue2054_dispatch.sh`; the Step
  9c gate red-ed 2 tests on FileNotFoundError — one full ~30-min 114-file gate run burned.
- **#2004** (park 2026-08-05T12:04:51Z, fp d8f423e4d696): same pair — gate round 1 blocked
  on 2 NEW nodes (`tests/test_issue2054_phase_d.py::test_dispatch_plan_pins_form_chat_for_phase_d`
  rc 127; `tests/test_issue2054_unit_e.py::test_null_draws_default_is_plan_pinned_100`
  FileNotFoundError) because both reference `scripts/issue2054_dispatch.sh`, absent in the
  worktree but present on origin/main; recovery cost one full ~45-min gate re-run.

verified-at-filing: `grep -c "issue\[0-9\]\*_\*\.sh" .claude/skills/issue/SKILL.md` → 0 hits
(the widening has NOT landed); `grep -n "scripts/issue\[0-9\]\*_\*\.py" .claude/skills/issue/SKILL.md`
→ 1 hit at line 2603 (the .py-only pathspec is live). Run at 2026-08-06T06:55Z on main.

## Proposed change

Widen the sibling-issue arm's pathspec set — the Step 5a block at
`.claude/skills/issue/SKILL.md:2603` plus any selector/step-1a binding references that
restate the glob pair — to include `':(glob)scripts/issue[0-9]*_*.sh'` alongside the two
.py globs, keeping the own-issue carve-out + per-file dirt skip unchanged. Both origin
parks carry a matching diff_sketch. Planner should check whether other non-.py sibling
dependency classes (e.g. `configs/issue<M>_*.yaml`) warrant inclusion in the same pass, or
whether .sh is the only class sibling tests hard-read today.

## Provenance

- fingerprint: fa01ae373c8e

- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: /daily 2026-08-05 Step C parked-candidate sweep — parks on #1988
  (origin_candidate_ts 2026-08-05T08:45:17Z, fp 9e04e93b0ae6) and #2004
  (origin_candidate_ts 2026-08-05T12:04:51Z, fp d8f423e4d696), one filing for both
  (same bug, two sightings).
