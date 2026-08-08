---
title: 'daily-fix: implementer pin-sweep deleted-literal own substep'
kind: infra
tags:
- wf-fix
- wf-fix-fp:107796a20adb
- daily-auto-filed
created_at: '2026-07-28T06:59:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): #1723 round 1 deleted a
  pin-test anchor literal (test_issue_tick_skill.py::_EXIT_SITE_ANCHORS) without grepping
  tests for it; test-verdict FAIL -> full round-2 bounce (~70 min) despite the #1699
  pin-sweep duty being in force'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session 4d10421c (#1723), 2026-07-27T15:54Z.

## Goal

Make the #1699 deleted-literal pin-sweep un-skippable by giving it its own numbered sub-step in the implementer spec.

## Workflow gap

- **Bug observed:** the implementer reordered Step 10 prose, deleting the anchor literal 'Run CRON-TEARDOWN before applying the terminal status' that `tests/test_issue_tick_skill.py::_EXIT_SITE_ANCHORS` pins; it reported 125/125 adjacent tests PASS, and the Step 9c gate then found the NEW failure -> round-2 re-anchor + code-review round 2 + a full gate re-run (~70 min).
- **Why it is a workflow gap:** the duty exists but is buried mid-way through one ~700-word paragraph (`implementer.md` item 1(a), L174) — it was in force (landed d79fa07b0e 2026-07-26) and missed at execution the next day.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'pin-sweep' .claude/agents/implementer.md` -> L174 (one paragraph) + L254 (report shape), run at compose time; `git log -1 d79fa07b0e` resolves (duty landed 2026-07-26).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/agents/implementer.md`: extract the deleted/moved-literal grep from item 1(a) into its own numbered sub-step: 'for each line your diff deletes or moves, grep the enumerated tests for its verbatim text (OLD and NEW form); every hit is run locally'. Keep the surrounding selector recipe unchanged.

## Scope / surfaces

- Primary target: `.claude/agents/implementer.md`
- Mirror in `experiment-implementer.md` if it carries the same paragraph (grep first).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 107796a20adb

- workflow_fix_target: .claude/agents/implementer.md
