---
title: 'daily-fix: Step 9c/10d gate single-flight guard'
kind: infra
tags:
- wf-fix
- wf-fix-fp:804ca99a80d9
- daily-auto-filed
created_at: '2026-07-23T07:01:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): #1606 launched a second
  gate run while the first was live (4 gate pids, 2 fail-CLOSED blocks, ~12 min churn);
  the gate recipe has no single-flight probe and its Monitor keyed on rc-file existence'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). #1606's Step-9c/10d gate churned for ~12 min: a Monitor false-fired "done" on a missing rc file while pytest was still printing, a SECOND gate run was launched while the first was live (4 live gate pids at one point, 2 more reaped later), and two fail-CLOSED verdict/sha blocks fired before the sha-bound pass finally merged. The fail-closed design prevented a bad merge, but the duplicate gate processes were self-inflicted (kill-before-relaunch skipped) and the Monitor condition was keyed on file EXISTENCE rather than process exit.

## Goal

The Step 9c/10d gate recipe in the /issue skill carries a single-flight guard: before (re)launching the gate, probe for a live gate process (`pgrep -af 'step9c-junit-issue-<N>|issue-<N>-lint-gate-tre[e]'` — bracketed per the self-match gotcha) and reap/wait first; and the gate's Monitor/done condition is keyed on process exit + rc-file (the existing re-run discipline), never rc-file existence alone.

## Workflow gap

- **Bug observed:** b8b69a72 (#1606), 2026-07-23T00:01–00:14Z: 00:02:10 + 00:02:49 `cat: /tmp/step9c-rc-issue-1606: No such file or directory` (Monitor fired while pytest was mid-run); 00:03:20 "duplicate gate reaped" (4 live gate pids); 00:10:25 `BLOCKED: verdict=not-run vs tip f6047a5c… fail CLOSED`; 00:11:07 exit 144 reaping 2 more; 00:12:54 verdict-ledger miss + second fail-CLOSED; resolved 00:13:43.
- **Why it is a workflow gap:** the gate recipe does not state a single-flight/ownership probe, so a re-drive races the live gate; the CLAUDE.md re-run discipline ("never key a done condition on bare file existence") exists but is not restated where gate Monitors are composed.
- **Confidence:** high.
- verified-at-filing: `grep -n 'single-flight' .claude/skills/issue/SKILL.md` → 3 hits, all in the repo-root push/sync context (lines ~9419/11579/11643), none in the Step 9c/10d gate recipe (absence claim for the gate section, hits read in context), 2026-07-23 UTC.

## Proposed change (refine in planning)

Two lines in the Step 9c/10d gate recipe: the pre-launch single-flight probe (+reap/wait), and the process-exit-keyed Monitor condition.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 9c/10d gate recipe).

## Constraints / invariants

- Fail-closed verdict/sha semantics unchanged. Recursion guard applies.

## Provenance

- fingerprint: 804ca99a80d9

- workflow_fix_target: .claude/skills/issue/SKILL.md
