---
title: 'daily-fix: repair 3 live test regressions on main (xargs anc'
kind: infra
tags:
- wf-fix
- wf-fix-fp:17d34d7af1d9
- daily-auto-filed
created_at: '2026-07-06T06:58:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): Three tests fail on current
  main and pollute every Step 9c full-suite gate: (a) test_live_skill_md_additive_checkout_is_dash_c_qualified
  anchors on the literal ''xargs -a /tmp/issue-<N>-additive-files.txt'' but #1047
  round-2 commit 56d76b7311 (2026-07-05T16:46Z) changed the Step 10d fence line to
  ''xargs -r -a'' - the commit landed 12 min after #1047''s own gate ran, and #1056''s
  gate had to hand-classi'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

(a) loosen the SKILL.md anchor predicate to tolerate xargs flags (e.g. regex 'xargs\s+(-r\s+)?-a' plus the list-file path); (b) root-cause the [3,3] rc codes in scripts/router_acceptance.py duplicate-cron-tick scenario and fix whichever side (harness expectation or production rc contract) is wrong - do not blindly update the expected codes without understanding the rc=3 source.

## Workflow gap

- **Bug observed:** Three tests fail on current main and pollute every Step 9c full-suite gate: (a) test_live_skill_md_additive_checkout_is_dash_c_qualified anchors on the literal 'xargs -a /tmp/issue-<N>-additive-files.txt' but #1047 round-2 commit 56d76b7311 (2026-07-05T16:46Z) changed the Step 10d fence line to 'xargs -r -a' - the commit landed 12 min after #1047's own gate ran, and #1056's gate had to hand-classify it as pre-existing; (b) test_negative_duplicate_cron_tick_is_idempotent_at_cli_level and test_cli_negative_duplicate_cron_tick_asserts_and_exits_zero fail with rc_codes=[3, 3] vs expected [0, 2] - plausibly the gcp-audit list-failed rc=3 change (#1041/#1042 line) bleeding into the acceptance harness. Verified failing on main 2026-07-06T06:40Z.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `tests/test_workflow_lint_no_repo_root_worktree_revert.py, tests/test_router_acceptance.py, scripts/router_acceptance.py`
- HIGH priority: every future Step 9c full-suite run carries these 3 failures as noise until fixed, forcing hand-classification (bit #1056 and #1029 today). Sessions: cc6ec8b5 (#1056), c7294ed3 (#1047, cause), 7cb3f598 (#1029).

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: tests/test_workflow_lint_no_repo_root_worktree_revert.py, tests/test_router_acceptance.py, scripts/router_acceptance.py
- source: /daily 2026-07-05 problem sweep (transcript-mined)
