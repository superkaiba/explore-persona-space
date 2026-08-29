---
title: 'cron wrappers: bound the synchronous Telegram push calls (timeout) — class
  sweep from #2196'
kind: infra
tags:
- from-2196
created_at: '2026-08-19T14:06:05Z'
has_clean_result: false
origin_prompt: '#2196 code-review round 1 (Codex) bug-class sweep ''synchronous alert
  subprocess without timeout'' + reconciler deferral of BLOCKER concern fatal-push-unbounded
  routing to a class-level telegram_push.sh timeout follow-up'
workflow: v1
---
# Cron-wrapper Telegram pushes are synchronous and unbounded — add timeouts at the call sites

## Goal

Bound every cron-wrapper invocation of the Telegram push helper so a
connected-but-stalled endpoint cannot hang a nightly wrapper indefinitely.

## Workflow gap

Surfaced by the #2196 round-1 Codex code review (bug-class sweep
"synchronous alert subprocess without timeout") and upheld-then-deferred by the
reconciler with rationale routing to exactly this class-level follow-up: the
push helper `$HOME/my-goat/scripts/telegram_push.sh` runs `curl` with no
`--connect-timeout` / `--max-time` (line 52), and the cron wrappers call it
synchronously. Enumerated pre-existing call sites (from the Codex sweep):

- `scripts/cron_lesson_consolidate.sh:140` (rc=3 arm) and the new #2196
  `fatal()` at line 66
- `scripts/cron_step9c_ledger_refresh.sh:115`
- `scripts/cron_daily_healthcheck.sh:106` and `:207`
- `scripts/cron_watch_issue_1739.sh:129`, `:158`
- `scripts/cron_watch_issue_2091.sh:49`, `:69`

`|| echo` fallbacks only handle a command that eventually returns; none of
these bound the call.

## Suggested direction (planner decides)

Prefer bounding at the CALL SITES in this repo (`timeout 30s "$TELEGRAM_PUSH"
"$MSG"`-style, exact bound per wrapper cadence), since the shared root cause
(`telegram_push.sh`) lives outside this repo in ~/my-goat — fixing the helper
itself is a my-goat change that can be proposed separately but must not be this
task's only fix. A regression pin per the Codex recipe: an executable push stub
that sleeps past the bound, asserting the wrapper still exits within its
deadline.

## Acceptance criteria

- Every enumerated in-repo cron-wrapper push call is time-bounded; a stalled
  push cannot hang the wrapper.
- Alert semantics preserved: a timed-out push counts as a failed push (existing
  no-sentinel/retry semantics untouched, e.g. #2190's T4).
- Existing wrapper test suites stay green.
- ≥1 test pins the bounded-push behavior with a sleeping stub.

## Provenance

Surfaced-prose follow-up from the #2196 round-1 Codex review + reconciler
deferral rationale (workflow-fix-on-bug protocol, orchestrator-routed).
Deferred BLOCKER concern `fatal-push-unbounded` on #2196 is discharged by this
task landing.
