---
title: 'daily-fix: step9c selector misses SKILL.md-pinning tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d6f8f7c59024
- daily-auto-filed
created_at: '2026-07-30T07:01:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): A diff touching .claude/skills/issue/SKILL.md
  selects NEITHER tests/test_issue_skill_file_only_verdict_post.py, tests/test_ensemble_review_cap.py,
  NOR tests/test_issue_skill_workload_cmd_script_pin.py, though each opens SKILL.md
  and pins exact section text'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: two parked formal candidates on #1804 (implementer r1, sweep fp 1344aa275876, ts 2026-07-30T00:00:56Z) and #1813 (implementer r1, sweep fp 289f115148c0, ts 2026-07-30T06:10:13Z) — same gap, merged into one filing).

## Goal

A diff to `.claude/skills/issue/SKILL.md` must select every test that pins SKILL.md content; today it selects none of the SKILL.md-reading pin tests, so a SKILL.md edit that breaks a pinned anchor passes the Step 9c gate.

## Workflow gap

- **Bug observed:** `select_step9c_tests.py --map-files .claude/skills/issue/SKILL.md` returns none of tests/test_issue_skill_file_only_verdict_post.py, tests/test_ensemble_review_cap.py, tests/test_issue_skill_workload_cmd_script_pin.py — each hard-reads SKILL.md and pins section text (#1804/#1813 rounds ran them only because the brief mandated them).
- **Why it is a workflow gap:** the selector's coverage contract is that a diffed workflow file selects every test pinning it; SKILL.md-reading tests added after the roster/GLOB_SCAN entries were last swept are silently unselected.
- **Confidence (emitter):** medium
- verified-at-filing: `uv run python scripts/select_step9c_tests.py --map-files .claude/skills/issue/SKILL.md | grep -cE 'file_only_verdict_post|workload_cmd_script_pin|ensemble_review_cap'` -> 0 (2026-07-30, this run). Open #865 on the same file is a DIFFERENT bug (worktree-branch blindness), not a duplicate.

## Proposed change (refine in planning)

Extend the rules-pin-style arm (or GLOB_SCAN path matcher) so tests hardcoding a touched `.claude/skills/**/SKILL.md` path literal are selected (the same containment predicate the #1496 rules-pin arm uses), or roster the named pin tests as WORKFLOW_INVARIANT; add a reachability pin (every tests/test_issue_skill_*.py reading SKILL_MD is selector-reachable). Related concern to evaluate in the same pass (miner H P9): whether the mapping covers `scripts/issue*_figures.py`-class files for tests/test_shared_vm_thread_caps.py — two #1738 scripts reached main violating the dotenv pin through a path that skipped that mapped-test leg.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- sha-verify (filing-time, #1467): `1344aa275876` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `289f115148c0` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: d6f8f7c59024

- Origin parks: #1804 events 2026-07-30T00:00:56Z (fingerprint 1344aa275876) and #1813 events 2026-07-30T06:10:13Z (fingerprint 289f115148c0); verbatim candidate blocks live in those markers.
