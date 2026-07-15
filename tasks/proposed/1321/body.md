---
title: 'daily-fix: follow-up defer/teardown must re-park parent'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b3abe37a4394
- daily-auto-filed
created_at: '2026-07-15T06:51:42Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): a mid-round defer/teardown
  of a same-issue follow-up left #825 stuck at followups_running for ~3.5h with no
  live compute — the teardown step has no re-park duty, only the 45-min issue-tick
  caught it'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep (session 09f28ede, #825): the wedged naturalistic Track-S run 2 was torn down and the round deferred, but the parent stayed at `followups_running` ~3.5h (deferred ~00:30Z, re-parked by the 01:52Z tick).

## Goal

add one line to the same-issue follow-up loop: any mid-round defer/teardown immediately restores the parent to its pre-round resting state (awaiting_promotion) in the same action as the pod/instance teardown

## Workflow gap

- **Bug observed:** a mid-round defer/teardown of a same-issue follow-up left #825 stuck at followups_running for ~3.5h with no live compute — the teardown step has no re-park duty, only the 45-min issue-tick caught it
- **Why it is a workflow gap:** the follow-up loop's SKILL.md text holds the task at `followups_running` for the round but names no status-restore duty on the defer/teardown path; the tick/watcher are backstops, not the owner.
- **Confidence:** high (incident observed; the backstop, not the loop, restored status)
- verified-at-filing: `grep -n "defer" .claude/skills/issue/SKILL.md` -> hits only in unrelated contexts (concern_deferral_request, deferred tools); no mid-round defer/teardown re-park line exists (absence-of-guard claim; 0 relevant in-target hits) (2026-07-15).

## Proposed change

One line in SKILL.md § Same-issue follow-up loop: "A mid-round defer / teardown (wedge, pathological fit, user defer) restores the parent to its pre-round resting status in the SAME action as the teardown — never leave `followups_running` without live round compute."

## Constraints

- Workflow-surface only; `workflow_lint.py --check-asks` passes; recursion guard applies.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: b3abe37a4394
