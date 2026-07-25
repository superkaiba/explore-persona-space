---
title: 'daily-fix: adopted failover clears runpod-wedge sentinel'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b5fa568aa03d
- daily-auto-filed
created_at: '2026-07-25T06:48:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): After the /issue 1586 session
  adopted the watcher-provisioned failover pod (relaunch 7), the poll/tick lane stayed
  wedge-blinded: the runpod-wedge already-handled guard kept returning the terminal
  verdict while the run was alive, so the poller could not drain sentinels and the
  orchestrator hand-rolled direct-SSH ticks'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session dd0af0ae, task #1586).

## Goal

When an owning session ADOPTS a replacement pod after a wedge-failover, the poll/tick lane must resume reading live state instead of returning terminal-wedge forever.

## Workflow gap

- **Bug observed:** after the 05:33Z wedge-failover on #1586, `tick_triage` returned verdict `dead terminal_runpod_wedge_already_handled` (log_age 1000000000, pid False) on consecutive ticks (2 firings, 07:05Z and 07:37Z) while the adopted relaunch-7 run was demonstrably alive; the poller never drained the results sentinel (10:31Z marker: "the wedge-blinded poller could not drain it") and the orchestrator vetoed + hand-rolled direct-SSH 30-min ticks for the rest of the run.
- **Why it is a workflow gap:** the `_runpod_wedge_already_handled` idempotency guard (correct for preventing a second terminate) has no adoption escape — a NEWER `epm:run-launched` / rewritten handle should flip the lane back to live polling.
- **Confidence (emitter):** high.
- verified-at-filing: `grep -rn 'already_handled' scripts/backend_poll.py` → `_runpod_wedge_already_handled` def at backend_poll.py:1917, reason mint `runpod_wedge_already_handled` at :2196, sentinel lifecycle :1828-2239 (7 hits, 2026-07-25); `grep -n 'wedge' scripts/tick_triage.py` → no sentinel-name hits (tick consumes the poll-lane verdict; primary target is backend_poll.py, tick_triage.py secondary for the verdict pathway).

## Proposed change (candidate diff sketch — refine in planning)

In `_runpod_wedge_already_handled` (scripts/backend_poll.py:1917): return False (and best-effort clear the sentinel) when the issue carries an `epm:run-launched` marker — or a handle rewrite — NEWER than the sentinel's mtime/payload ts (the adoption signal). Add a pin test.

## Scope / surfaces

- Primary target: `scripts/backend_poll.py` (+ `scripts/tick_triage.py` only if the verdict pathway needs a matching change)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: b5fa568aa03d

- workflow_fix_target: scripts/backend_poll.py
