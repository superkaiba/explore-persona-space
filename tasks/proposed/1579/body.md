---
title: 'daily-fix: step9c stem arm maps .sh scripts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bda47039eaa8
- daily-auto-filed
created_at: '2026-07-21T06:38:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): The Step-9c selector''s
  exact-stem arm maps only .py files (suffix gate at select_step9c_tests.py:866; any
  other extension is silently ignored, :883-884), so a diff touching ONLY a .sh script
  with a stem-matched test file (e.g. scripts/guard_repo_root_branch.sh -> tests/test_guard_repo_root_branch.py)
  selects just the invariant set - the stem-matched suite never runs, and --map-files
  returns empty'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-20 parked-candidate routing pass (Step C) from a FORMAL workflow-fix candidate block parked on task #1566 under the recursion guard (surfaced by the #1566 fact-checker, plan v2 correction).

## Goal

Extend the Step-9c selector's exact-stem arm to map `scripts/<stem>.sh` → `tests/test_<stem>.py` (or register guard tests in the appropriate mapping table), keeping `.py` behavior byte-identical, with a drift pin in `tests/test_select_step9c_tests.py`.

## Workflow gap

- **Bug observed:** the Step-9c selector's stem arm maps only `.py` files (suffix gate `if p.suffix != ".py"`), so a diff touching ONLY a `.sh` script with a stem-matched test file selects just the invariant set — the stem-matched suite never runs through the stem arm.
- **Why it is a workflow gap:** guard hooks and other workflow shell scripts are workflow surface with dedicated pinned test suites; a script-only fix bypasses its own pins at the test-verdict gate AND the Step-10d mapped-test leg.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'p.suffix != ".py"' scripts/select_step9c_tests.py` → the stem-arm suffix gate is live at :675 on current main (post-#1573, merge `adb9cfdeda` — that fix added import-map/TG-leg arms, not a `.sh` stem arm; `git log --oneline --since='7 days ago' -- scripts/select_step9c_tests.py` shows no commit touching the suffix gate). MITIGATION NUANCE the plan must weigh: the literal-path arm (#1498, `f1c02d4232`) DOES reach `.sh` files when a test hardcodes the script path — `grep -c 'guard_repo_root_branch.sh' tests/test_guard_repo_root_branch.py` → 4, so the candidate's own worked example is now covered by literal-path selection; the residual gap is `.sh` scripts whose stem-matched tests do NOT hardcode the literal path. The open on_hold task #865 targets the same file but a DIFFERENT bug (worktree-blind main-checkout diffing), not a duplicate (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

```
In select_step9c_tests.py stem arm (~:675):
- if p.suffix != ".py": <skip>
+ if p.suffix not in (".py", ".sh"): <skip>  # .sh scripts with test_<stem>.py suites
+ (drift pin: a .sh path with an existing stem-matched test file must appear in the selection)
```

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py` (+ `tests/test_select_step9c_tests.py` drift pin)

## Constraints / invariants

- `.py` selection behavior stays byte-identical.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: 998dc43618d3

Verbatim formal candidate block (parked on #1566, ts 2026-07-20T15:40:24Z):

```
<!-- workflow-fix-candidate v1 -->
target_file: scripts/select_step9c_tests.py
bug_observed: The Step-9c selector's exact-stem arm maps only `.py` files (suffix gate at select_step9c_tests.py:866; any other extension is silently ignored, :883-884), so a diff touching ONLY a `.sh` script with a stem-matched test file (e.g. scripts/guard_repo_root_branch.sh -> tests/test_guard_repo_root_branch.py) selects just the invariant set — the stem-matched suite never runs, and --map-files returns empty for it.
why_workflow_gap: Guard hooks and other workflow shell scripts are workflow surface with dedicated pinned test suites; a script-only fix bypasses its own pins at the test-verdict gate AND at the Step-10d mapped-test leg (both read the selector), an under-selection the gate cannot see. Surfaced by the #1566 fact-checker (plan v2 correction): this round was covered only because the diff also touched the test file (touched-test arm).
proposed_change: Extend the stem arm to map `scripts/<stem>.sh` -> `tests/test_<stem>.py` (or register the guard test in the appropriate mapping table), keeping the .py behavior byte-identical; add a drift pin in tests/test_select_step9c_tests.py.
diff_sketch: |
  In select_step9c_tests.py stem arm (~:866):
  - if p.suffix == ".py": <stem mapping>
  + if p.suffix in (".py", ".sh"): <stem mapping>  # .sh scripts with test_<stem>.py suites
  + (drift pin: a .sh path with an existing stem-matched test file must appear in the selection)
confidence: medium
related_task: #1566
<!-- /workflow-fix-candidate -->
```
