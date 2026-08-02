---
title: 'daily-fix: report TTY-attached unmapped idle sessions'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9c12fdac0800
- daily-auto-filed
created_at: '2026-08-01T07:07:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): TTY-attached unmapped wrappers
  are cleared with zero observability by the idle-unmapped/zombie passes, so 26 sessions
  (~17 unmapped, 72-95 h old) accumulated invisibly until Thomas asked.'
workflow: v1
---
# daily-fix: report TTY-attached unmapped idle sessions

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED M9; miner-8:P7). Source: session 0ac15c23 — Thomas: "why are tehre 26 active sessions?", then directed manual triage. ~17 of the 26 were unmapped, 72–95 h old, TTY-attached wrappers — the exact shape both the zombie-wrapper and idle-unmapped watcher passes deliberately exempt, so they accumulate silently until a human asks.

## Goal

Add an escalate-only (never-stop) watcher report lane for TTY-attached unmapped EPS sessions idle beyond ~48 h — one deduped push listing them with safe-to-kill verdicts.

## Workflow gap

- **Bug observed:** 26 active sessions accumulated on the VM, ~17 of them TTY-attached unmapped wrappers 72–95 h old, invisible to every watcher reaper AND to every watcher report — Thomas had to notice the count himself and direct manual triage.
- **Why it is a workflow gap:** In `scripts/autonomous_session_watch.py`, `decide_idle_unmapped` (line ~26227) returns `("clear", 0)` on `has_tty` — a controlling TTY unconditionally ENDS the episode with no record, no sidecar row, no alert ("A controlling TTY means a terminal Thomas may be sitting at; the episode ends"). The zombie-wrapper non-EPS lane likewise requires "no live user TTY" before acting. The never-STOP exemption is correct (a TTY may be a terminal Thomas is sitting at), but clearing with zero observability means multi-day accumulations have no surfacing channel at all — the watcher has escalate-only lanes for several other keep-classes (active-cache escalation, root-stash audit, orphan-wrapper sweep) but none here.
- **Confidence (emitter):** high (decision function read; the incident is the user's own ask)
- verified-at-filing: context read of `scripts/autonomous_session_watch.py` `decide_idle_unmapped` (lines 26227-26284): `if mapped or has_tty: return ("clear", 0)` — clear drops all state, and no branch of the function (or the pass's sentinels `_IDLE_UNMAPPED_ALERT_NOTE_SENTINEL` etc., lines 1661-1671) emits a record for the TTY case (absence claim: no TTY-attached report lane exists — 0 escalate/report arms keyed on `has_tty`). `git log --oneline --since='7 days ago' -- scripts/autonomous_session_watch.py` → 5 commits (urgent lane, root-stash audit, unfolded-round observer, partial-bundle pass, refusal-aware recovery) — none adds a TTY-attached report lane; no landed fix (2026-07-31).

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/autonomous_session_watch.py:
+ TTY_UNMAPPED_REPORT_IDLE_S = 48 * 3600   # env: EPM_TTY_UNMAPPED_REPORT_HOURS
+ In the idle-unmapped pass, when a session is skipped on has_tty:
+   if idle_age_s is not None and idle_age_s >= TTY_UNMAPPED_REPORT_IDLE_S:
+       accumulate into a report list (sid, cwd, wrapper age, idle age,
+       safe-to-kill verdict per the existing zombie-wrapper heuristics)
+ After the sweep: ONE deduped Telegram push per episode (dedup on the
+ session-id set, re-alert on growth — the disk-guard band-dedup pattern)
+ + a sidecar row in ~/.eps-autonomous/idle-unmapped-events.jsonl
+ (action: "tty-report", never "stop"). Kill switch:
+ EPM_DISABLE_TTY_UNMAPPED_REPORT=1.
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Secondary: `tests/` (pure-decision pin test: TTY + idle ≥ threshold → report action, never stop; TTY + idle < threshold → clear, no report), `.claude/rules/background-automation.md` (document the new pass)
- Grep before editing: `grep -n 'has_tty\|idle-unmapped' scripts/autonomous_session_watch.py` and thread the report arm through every TTY-clear site the idle-unmapped pass owns; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- ESCALATE-ONLY: the new lane must never stop, unregister, or otherwise mutate a TTY-attached session — report + verdict only (the existing never-touch guarantees for TTY wrappers are load-bearing).
- One deduped push per episode (no nightly spam); sidecar rows follow the one-sidecar-per-pass family precedent.
- ruff on touched files passes; `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 9c12fdac0800

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: (driver-computed; tag authoritative)

Origin: CONSOLIDATED M9 (miner-8:P7), /daily 2026-07-31 — "26 active sessions accumulated (~17 unmapped, 72–95 h old, TTY-attached) — invisible to every watcher reaper until Thomas asked" (session 0ac15c23).
