---
title: 'cron wrappers: unchecked mkdir -p LOG_DIR silently skips the pass (sibling
  sweep of #2196)'
kind: infra
tags:
- from-2196
created_at: '2026-08-19T14:05:44Z'
has_clean_result: false
origin_prompt: '#2196 code-review round 1 (Claude) prose follow-up: ~10 sibling cron
  wrappers share the same unchecked mkdir -p LOG_DIR silent-skip class (e.g. cron_daily_healthcheck.sh:64,
  cron_pod_audit.sh:57); a class-level follow-up task is the right vehicle'
workflow: v1
---
# Sibling cron wrappers: unchecked `mkdir -p "$LOG_DIR"` silently skips the whole pass

## Goal

Sweep the sibling cron wrapper scripts for the same failure class #2196 fixed in
`scripts/cron_lesson_consolidate.sh`: an unchecked `mkdir -p "$LOG_DIR"` (or an
unprobed daily-log append) lets an uncreatable/unwritable log dir silently skip
the wrapper's entire pass while the wrapper still exits 0.

## Workflow gap

Surfaced by the #2196 code-reviewer (Claude, round 1, prose follow-up): ~10
sibling cron wrappers share the same unchecked `mkdir -p "$LOG_DIR"` silent-skip
class — named examples `scripts/cron_daily_healthcheck.sh:64` and
`scripts/cron_pod_audit.sh:57`. The same brace-group-redirect mechanism #2196's
task body documents applies wherever the wrapper redirects its pass into a
daily log under an unchecked dir: the redirect fails, the group never runs,
and the trailing `exit 0` hides it.

## Suggested direction (planner decides)

Apply the #2196 pattern per wrapper: a `fatal()`-style guard (stderr FATAL
naming the path + best-effort push where the wrapper already has a push
channel + exit non-zero) on the `mkdir -p`, plus an appendability probe where
a brace-group redirect is the execution vehicle. Enumerate the full sibling
set first (`grep -l 'mkdir -p' scripts/cron_*.sh` and inspect each wrapper's
redirect shape); some wrappers may differ enough that only the mkdir guard
applies. Keep each wrapper's routine-pass silence and exit semantics intact.

## Acceptance criteria

- Every sibling cron wrapper whose pass is skipped by an uncreatable/unwritable
  log dir fails loud (stderr line naming the path + non-zero exit), or is
  explicitly recorded as not applicable with a reason.
- Existing wrapper tests stay green; routine passes stay silent.
- No changes to the wrapped consolidators/auditors themselves or crontab
  schedules.

## Provenance

Surfaced-prose follow-up from the #2196 round-1 code review (workflow-fix-on-bug
protocol, orchestrator-routed). Parent fix: #2196 commit 693a9b3d5440.
