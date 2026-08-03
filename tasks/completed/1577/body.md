---
title: 'daily-fix: read-bounding hook for guard hook scripts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d55ea1b239ea
- daily-auto-filed
created_at: '2026-07-21T06:38:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): a wholesale Read of a trigger-dense
  guard hook script pages refusal-adjacent vocabulary into orchestrator context with
  no mechanical bound; #1563 landed the discipline as rule text only'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-20 parked-candidate routing pass (Step C) from a workflow-fix candidate parked on task #1563 under the recursion guard (emitting context: Alternatives critic, #1563 plan review round 1).

## Goal

Add a mechanical PreToolUse hook that bounds wholesale reads of guard hook scripts (byte/line cap + env override), mechanizing the READ channel of #1563's orchestrator-turn discipline on guard-surface rounds (the authored-text channel stays rule-side).

## Workflow gap

- **Bug observed:** #1563 landed the orchestrator ordinary-turn discipline for guard-surface rounds as RULE text, but a wholesale Read of a trigger-dense guard script (e.g. a 41K-token guard hook read, observed 2026-07-19) still pages refusal-adjacent vocabulary into orchestrator context with no mechanical bound.
- **Why it is a workflow gap:** the read channel is mechanizable — the existing `guard_harmful_bank_read.sh` PreToolUse hook is the direct precedent; without it the discipline depends on every session having loaded the rule.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'guard_harmful_bank_read' .claude/settings.json` → 1 hit (:142, the precedent hook); no read-bounding hook for guard scripts exists in `.claude/settings.json` (0 matcher hits) and `ls scripts/guard_*.sh` shows the guard scripts (`guard_repo_root_branch.sh`, `guard_repo_root_pull.sh`) are unbounded read targets (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

New `scripts/guard_*.sh` (or `.claude/hooks/`) read-bounding PreToolUse hook + a `.claude/settings.json` hook registration: cap Read of guard hook scripts at a byte/line budget with an env override, per the harmful-bank-read guard pattern.

## Scope / surfaces

- Primary targets: `.claude/settings.json` + a new read-bounding guard script
- NOTE: a hook addition is a behavior change — full independent review required (this is exactly why it routes through /issue rather than self-apply).

## Constraints / invariants

- Workflow-surface only. Hook must fail open on missing env/paths (never wedge ordinary reads).
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: d55ea1b239ea

- workflow_fix_target: .claude/settings.json

Verbatim parked candidate (prose park on #1563, ts 2026-07-20T11:02:35Z):

> parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see § Recursion guard. source: prose-followup (Alternatives critic, plan review round 1). target_file: .claude/settings.json + a new scripts/guard_*.sh read-bounding script. proposed_change: a mechanical PreToolUse hook bounding wholesale reads of guard hook scripts (byte/line cap + env override), precedented by the existing harmful-bank read guard — mechanizes the read channel of #1563's duty 2 (the authored-text channel stays rule-side). confidence: medium. related_task: #1563.
