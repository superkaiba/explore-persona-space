---
title: 'daily-fix: /issue Monitor until-loop composition guidance'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6fef049fdf8a
- daily-auto-filed
created_at: '2026-08-02T07:16:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): Monitor mis-arms: #1739
  box-set Monitor hit its cap 9x overnight with >=6 ''baseline tick'' no-op classification
  turns per re-arm; #1947 judge fan-out monitor wedged on the documented count-probe
  or-echo double-print inside its until-condition. SKILL.md sanctions Monitor until-loops
  but carries zero composition guidance.'
workflow: v1
---
# daily-fix: /issue Monitor until-loop composition guidance

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C33 (miner-1 P10 + miner-4 P6; sessions 55419495 (#1739), 8fc069db (#1947)).

## Goal
Add Monitor until-loop composition guidance to `.claude/skills/issue/SKILL.md`: (a) key until-conditions on count-DECREASE (suppress the immediate baseline event); (b) never use the `pgrep -c ... || echo 0` double-print shape inside a Monitor condition; (c) prefer the sanctioned `poll_pipeline.py` bg-Bash chain for multi-hour box sets (avoids the ~1h Monitor cap).

## Workflow gap
- **Bug observed:** (a) #1739's box-completion Monitor timed out at its cap 9 times overnight (22:25Z–06:30Z, "[Monitor timed out — re-arm if needed.]" firings), each re-arm firing an immediate "live=N of N" event the assistant classified ≥6 times as "baseline tick — not a real completion". (b) #1947's judge fan-out monitor terminal branch was "wedged by the classic `pgrep -c || echo 0` double-print" (orchestrator's words, 21:44Z) — the exact count-keyed liveness gotcha `.claude/rules/gotchas.md` already documents for WRITING gates, recurring inside a Monitor condition; verified manually + TaskStop'd. `unverified hypothesis — verify at plan time: the Monitor cap is ~1h and not extendable per-arm (miner-inferred from observed timeout cadence, not read from tool docs).`
- **Why it is a workflow gap:** SKILL.md sanctions Monitor until-loops as one of the two poller shapes but carries zero composition guidance for the until-condition, so the documented count-keyed gotcha has a blind spot exactly where multi-hour waits are composed.
- **Confidence:** medium
- verified-at-filing: `grep -in 'monitor' .claude/skills/issue/SKILL.md` → Monitor until-loops named as a sanctioned poller shape (lines ~4791, 5274, 5287) with no until-condition composition guidance anywhere; `grep -n 'pgrep -c' .claude/skills/issue/SKILL.md` → 0 hits; `grep -n 'count-DECREASE\|baseline tick\|baseline event' .claude/skills/issue/SKILL.md` → 0 hits; gotchas.md LESSONS row confirms the `pgrep -c/grep -c || echo` double-print gotcha is owned there for count-keyed gates (write-time trigger only); `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → ~10 commits, none Monitor-composition-related (2026-08-02 UTC).

## Proposed change (refine in planning)
A short "Monitor until-loop composition" block near the sanctioned-poller prose (~line 5274):

```
+ **Monitor until-loop composition (#1739 baseline churn, #1947 double-print wedge).**
+ (a) Key a completion until-condition on a count-DECREASE relative to the count
+ observed AT ARM TIME (or otherwise suppress the immediate baseline event) — an
+ absolute "live=N" condition fires a no-op event on every (re-)arm.
+ (b) NEVER `pgrep -c ... || echo 0` (or any `X || echo` count) inside a Monitor
+ condition — the double-print wedge (gotchas.md count-keyed-gate entry) applies
+ to Monitor conditions exactly as to written gates; use `pgrep -f 'patter[n]' |
+ wc -l` or a rc-keyed probe.
+ (c) Multi-hour waits (box sets, overnight fan-outs): prefer the sanctioned
+ poll_pipeline.py bg-Bash chain — a Monitor arm times out at its cap and each
+ re-arm costs a triage turn.
```

Planner verifies the actual Monitor cap/semantics against the tool schema before pinning "1h" in prose (labeled unverified above), and cross-links the gotchas.md entry rather than duplicating its full text.

## Scope / surfaces
- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; the `tests/test_issue_skill_*` pin tests stay green (sync atomically per the #1883 family rule).
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 6fef049fdf8a
- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C33.
