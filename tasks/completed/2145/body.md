---
title: 'daily-fix: dispatch_issue launch lacks name-suffix/gpu-type'
kind: infra
tags:
- wf-fix
- wf-fix-fp:67b347911049
- daily-auto-filed
created_at: '2026-08-06T07:24:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): two burned launch attempts:
  --name-suffix unrecognized (exit 2); no --gpu-type so intent abuse (ft-70b) was
  needed for 8xH200'
workflow: v1
---
# daily-fix: dispatch_issue.py lacks --name-suffix passthrough and a --gpu-type override — two burned launch attempts + intent abuse

## Workflow gap

Two `dispatch_issue.py launch` gaps cost the #1491 inline run two failed dispatch
attempts on 2026-08-05:

1. `--name-suffix` is a `pod.py provision` flag but not a `dispatch_issue.py launch` flag
   — first dispatch exited 2 ("unrecognized argument"), although suffixed pods
   (`pod-<N>-<slug>`) are the documented convention for follow-up rounds.
2. There is no `--gpu-type` override — when 8×H100 had no capacity (exit 4), the session
   had to abuse `--intent ft-70b` to reach 8×H200 ("no --gpu-type; GPU type comes from
   the intent"), which worked but encodes the wrong intent in the run record.

Related plan-side observation for the same round (no separate filing): the approved
plan's §10 dispatch template carried 3 wrong flags vs the real launcher (`--model` vs
`--scale`, missing mandatory `--all-splits`, wrong `--hf-prefix`) — caught pre-spend;
worth the planner checking §10 templates against the launcher's argparse where cheap.

verified-at-filing: both exit codes + usage text are the recovery miner's probed
transcript reads (session 8d7f8b25 rows 2247–2267).
`grep -cn 'name-suffix\|gpu-type' scripts/dispatch_issue.py` run at compose time → 0
(neither flag exists on the launch subcommand).

## Proposed change

`scripts/dispatch_issue.py launch`: add `--name-suffix` passthrough to the provision leg
and a `--gpu-type` (and `--gpu-count`) override that takes precedence over the intent
mapping, recorded in the `epm:backend-selected` note so the run record stays honest.

## Provenance

- fingerprint: 67b347911049

- workflow_fix_target: scripts/dispatch_issue.py
- origin: /daily 2026-08-04 recovery sweep — miner 2 P10 (probed rows).
