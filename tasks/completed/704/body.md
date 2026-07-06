---
title: fix backend_poll last_log_mtime_sec_ago clock/epoch skew
kind: infra
tags:
- daily-held
created_at: '2026-06-28T07:13:39Z'
has_clean_result: false
origin_prompt: /daily 2026-06-27 held backlog
---
## Overview / Motivation

Filed from /daily 2026-06-27 held backlog: `last_log_mtime_sec_ago` shows an absurd value due to a clock/epoch skew.

## Goal

Make the log-staleness signal correct so the >900s stall path is reliable.

## Problem (from /daily 2026-06-27)

`last_log_mtime_sec_ago` grew to 14-33h on a log whose mtime was ~30s old (session 09e41486), masked only by the CPU-advancing override; if that override were removed it would falsely trip the >900s stall path. The bug is an epoch / timezone skew between the pod clock that stamps the log mtime and the clock the poller compares against.

## Proposed change

Reproduce against the pod clock, pin the epoch/timezone skew in `scripts/backend_poll.py`, fix it (normalize both timestamps to the same epoch/UTC basis). Add a regression test that asserts a fresh-mtime log yields a small `last_log_mtime_sec_ago` even under a pod-vs-poller clock offset.

## Scope / target files

- `scripts/backend_poll.py`
- `tests/` (new regression test)

## Constraints

- Workflow-surface only (backend_poll.py is in the workflow-helper-scripts list).
- Lint gate green; ruff clean on touched files.
- Do not remove the CPU-advancing override as the fix — fix the underlying clock skew so the override is no longer load-bearing for correctness.
