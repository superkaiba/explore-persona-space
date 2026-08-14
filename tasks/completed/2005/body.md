---
title: 'daily-fix: Step 9c gate — detached first attempt, end the tu'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f06840c340d1
- daily-auto-filed
created_at: '2026-08-02T07:12:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): #1893''s 141-file Step
  9c gate, launched per the mandated bg-Bash shape, was harness-killed at ~26 min
  / ~95% (exit 144, rc file missing) and re-run detached — wall time ~doubled; #1984
  held a turn ~31 min with TaskOutput(block=true, timeout=600000) x3, the banned sleep-chain
  shape.'
workflow: v1
---
# daily-fix: Step 9c gate — detached first attempt, end the turn

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C11 (miners 8, 5, 7; sessions 2b6d4928 (#1893), 11d2daa4 (#1984), c3faba3d (#1882)).

## Goal
Step 9c launches the full-suite gate pytest in the DETACHED shape (setsid + rc/junit harvest breadcrumbs) on the FIRST attempt when projected wall exceeds ~15 min, and states explicitly that after launching the gate the session ENDS THE TURN — repeated blocking TaskOutput waits are the banned sleep-chain shape.

## Workflow gap
- **Bug observed:** (a) #1893's 141-file gate, launched per the current step-1b bg-Bash recipe, was harness-killed at ~26 min / ~95% complete ("exit 144, rc file missing, kill-probe CLEAR"); the full run was re-dispatched in the detached setsid shape (pid=2254900) per the died-mid-run recovery, roughly DOUBLING the gate's wall time (second run passed, COMPARE_RC=0 at 18:44Z). (b) #1984 held a turn open ~31 min with `TaskOutput(block=true, timeout=600000)` ×3 waiting on the gate — the banned sleep-chain shape; a C8 refusal hit during the held-open stretch. (c) #1882's run-1 sidecar false-fail is NOT in scope here (see Scope).
- **Why it is a workflow gap:** SKILL.md step 1b mandates "BACKGROUND Bash invocation ... BACKGROUND IS REQUIRED, NOT OPTIONAL" as the first-attempt shape and reserves the detached shape for recovery, while the gate's own measured wall (median ~18 min, max ~38 min per the SKILL's #1646 figures) predictably exceeds bg-Bash wrapper lifetime; and the launch site never states the end-the-turn duty, so blocking-TaskOutput chains recur.
- **Confidence:** high (both incident shapes probed by miners: exit-144 narration + stage-dispatch marker; the 3 TaskOutput tool_use rows)
- verified-at-filing: `grep -n 'BACKGROUND Bash invocation\|BACKGROUND IS REQUIRED' .claude/skills/issue/SKILL.md` → step 1b (~line 9547) mandates bg-Bash first-attempt; `grep -n 'setsid\|detached' .claude/skills/issue/SKILL.md` → detached setsid shape exists ONLY in § "Detached VM-side long compute phases" (line 6767) + the died-mid-run recovery — no first-attempt gate instruction; wait guidance at step 1b says "Default to WAITING for exit — the harness notification" with no explicit end-the-turn / no-blocking-TaskOutput line; `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 8+ commits, none changing the gate launch shape (2026-08-02).

## Proposed change (refine in planning)
In `.claude/skills/issue/SKILL.md` Step 9c step 1b:
- Replace the bg-Bash-first mandate with: full-suite gate runs (projected >~15 min — i.e., any selection containing the workflow-invariant set) launch DETACHED on the FIRST attempt, using the same setsid + pid/log/rc-file/junit harvest breadcrumb shape the died-mid-run recovery already prescribes (§ Detached VM-side long compute phases conventions; single-flight probe, rm -f preamble, choom, tmproot, and the rc-file verdict read all unchanged). Small mapped-test subsets below the threshold may keep bg-Bash.
- Add one explicit line at the launch site: "After launching the gate, END THE TURN. Wait via the Monitor-until-probe loop or the tick re-wake — repeated blocking `TaskOutput(block=true)` calls are the banned sleep-chain shape (#1984: ×3, ~31 min held turn)."

## Scope / surfaces
- Primary target: `.claude/skills/issue/SKILL.md` (Step 9c step 1b; mirror any restatement at the Step 10d gate blocks and the 1d compare leg — grep `run_in_background=true` near the gate sites).
- DROPPED sub-item (c) of the consolidated entry (sidecar-reading test snapshots the sidecar at test start / filters by run-start timestamp): covered by a SEPARATE filing from this same nightly run — the #1876 sidecar-canary park (`logs/daily/mining-2026-08-01/stepc-1876-sidecar-canary.md`). Do not duplicate it here.
- Keep consistency with `tests/test_issue_skill_*` pin tests (Step 5a family) — update pins atomically if any pin the bg-Bash wording.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: f06840c340d1
- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C11.
