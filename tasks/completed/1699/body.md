---
title: 'daily-fix: implementer pin-sweep + lint parity'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5095937ee216
- daily-auto-filed
created_at: '2026-07-26T07:06:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): Four of six sessions in
  one wave had the code-reviewer independently re-run the pin sweep and find files
  the implementer hit list omitted, the #1681 implementer introduced a direct tasks-path
  construction that a repo-wide invariant test catches but the local union missed
  until the Step 9c gate found it 27 minutes later, and a UP033 violation passed the
  implementer default ruff leg and failed the r'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. The implementer's self-declared
verification scope and the gate's actual scope disagree in three separate ways, and
the reviewer or the gate paid for it in 5 of the day's sessions.

## Goal

Emit the pin-sweep hit list mechanically (selector-computed, with `sweep_scope`
declared by the tool rather than by the implementer), always include the repo-wide
invariant tests in the implementer's pre-commit local union when the diff touches any
`scripts/*.py` or `src/**` file, and run the repo ruff-policy pin rather than a
default `ruff check` in the implementer's lint leg.

## Workflow gap

1. **Pin-sweep hit lists under-enumerated in 4 of 6 sessions.** Every code-reviewer
   re-ran its own repo-wide/tree-wide pin sweep and found files the implementer's
   declared hit list had omitted: **8 files** (#1667 @ 08:12Z), **2** (#1669 @
   10:24Z), **8** (#1670 @ 09:57Z), **5** (#1671 @ 09:54Z). All were run green by the
   reviewer, so nothing shipped broken — but each cost the reviewer an extra sweep +
   test run and produced a recurring non-blocking Minor. #1667 verbatim: *"gate-scope
   sweep omitted 8 generic-token hit files — I ran all 8 green"*; #1670: *"pin-sweep
   hit-list under-enumeration of 8 basename-fragment test files — all 271 run by me
   and green"*. The implementer's declared list and the reviewer's independent sweep
   are two different computations of the same thing; only one is mechanical.
2. **The local union misses repo-wide invariants.** #1681 (`832cccf2`) round 1
   introduced `PROJECT_ROOT / "tasks"` at `scripts/autonomous_session_watch.py:8220` —
   a direct violation of the always-on canonical-resolver invariant pinned by
   `tests/test_no_direct_task_path_construction.py`. The implementer reported a
   "4,208-test local union green" and the code-reviewer PASSed with 0 Critical /
   0 Major; **neither ran that invariant test**. The Step 9c gate found it after a
   27-min run: `E   1 file(s) violate the canonical-resolver rule. … Matches: -
   scripts/autonomous_session_watch.py:8220`. Cost a round-2 implementer spawn plus a
   second 29-min gate run (~40 min). The local union is derived from touched-file
   mapping, which does not pull in repo-wide invariant tests that ANY `scripts/*.py`
   edit can break.
3. **Local ruff leg is narrower than the gate's ruff pin.** #1672 (`188282d2`) @
   11:33:39Z — the implementer reported `ruff check + format clean` on its main
   commit, but the gate-scope union's `tests/test_ruff_policy.py` full-ruleset pin
   caught a UP033 violation, forcing a second commit (`cfb4a2a297`) purely to swap a
   decorator to `functools.cache`. The "ruff clean" acceptance criterion is therefore
   not the criterion the gate enforces.
- **Confidence (emitter):** high on all three (4/6, 1, and 1 same-day occurrences,
  each quoted from the sessions' own reviewer findings or gate output).
- verified-at-filing: the three always-on invariant tests exist and are named in
  CLAUDE.md § Common Commands as the workflow-invariant family —
  `ls tests/test_no_direct_task_path_construction.py tests/test_no_pod_side_task_py_shellout.py
  tests/test_no_dollar_budget_caps.py tests/test_ruff_policy.py` → all four present.
  The #1681 violation text, the four under-enumeration counts, and the UP033 fix-up
  commit are quoted from the sessions' own tool output / reviewer findings, not from
  recall; `git rev-parse --verify --quiet 'cfb4a2a297^{commit}'` resolves. Landed-fix
  history check `git log --oneline --since='7 days ago' -- .claude/agents/implementer.md
  .claude/agents/experiment-implementer.md scripts/select_step9c_tests.py` → the wave
  touched all three (#1682 `841304c2d0`, #1688 `ab45d65777`, #1673 `37f6f6b1b4`,
  #1651, #1649, #1646, #1645); none makes the sweep mechanical, adds the invariant
  family to the local union, or changes the lint leg's ruleset. (2026-07-25)

## Proposed change (refine in planning)

```
+ (1) selector emits the pin-sweep hit list: the implementer RUNS the tool and pastes
+     its output (with the tool-declared sweep_scope) instead of hand-enumerating, so
+     the implementer's list and the reviewer's independent sweep are one computation.
+ (2) local union: when the diff touches any scripts/*.py or src/** file, always
+     include tests/test_no_direct_task_path_construction.py,
+     tests/test_no_pod_side_task_py_shellout.py, tests/test_no_dollar_budget_caps.py.
+ (3) lint leg: run tests/test_ruff_policy.py (or ruff with the pin's full ruleset)
+     rather than a bare `ruff check`, so "lint clean" means what the gate means.
```

(2) is cheap — those three tests are fast static scans, not suite runs. (1) is the
structural one: if the selector already computes the hit list for the reviewer, the
implementer should call the same entrypoint.

## Scope / surfaces

- `.claude/agents/implementer.md` and `.claude/agents/experiment-implementer.md`
  (verification/smoke-run legs — both, or the shared text they point at).
- `scripts/select_step9c_tests.py` if (1) needs a new output mode rather than reusing
  an existing one — check first; the reviewer already runs a sweep, so the computation
  likely exists and only needs an implementer-facing entrypoint.
- Both agent specs are large: the agent-spec size ratchet
  (`workflow_lint --check-agent-spec-size`, WARN >28KB / FAIL >40KB) applies — prefer
  editing shared pointer text over duplicating three bullets in two files.

## Constraints / invariants

- (2) must not balloon the local union into a full-suite run — name the three static
  invariant tests explicitly, do not widen to "all workflow-invariant tests".
- (1) must keep the implementer's declaration auditable: the reviewer's independent
  sweep stays as the check; this makes the two agree, it does not remove the check.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes
  (including `tests/test_ruff_policy.py`, which is the point of item 3).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/agents/implementer.md
- fingerprint: 5095937ee216
- Source: `/daily` 2026-07-25 transcript sweep, sessions `203baf55` (#1667),
  `7457e1a3` (#1669), `ad35514c` (#1670), `25e73c77` (#1671), `832cccf2` (#1681),
  `188282d2` (#1672).
