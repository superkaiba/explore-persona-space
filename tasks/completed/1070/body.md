---
title: 'workflow-fix: launchers atomically refresh the pid file on every relaunch
  (stale-pid poll target)'
kind: infra
tags:
- wf-fix
created_at: '2026-07-05T14:54:09Z'
has_clean_result: false
origin_prompt: '#813 v5 stale-pid relaunch (2026-07-02 00:39); watch-session held
  candidate, filed at #813 park'
workflow: v1
---
## Overview / Motivation
Auto-filed by the workflow-fix protocol from the #813 v5 relaunch incident (watch-session orchestrator; filed at the #813 park).

## Goal
Every launcher/relaunch of a pod-side or VM-side detached workload MUST atomically rewrite the pid file (`/workspace/logs/issue-<N>.pid` or the VM analogue) with the NEW pid before the poller's next read; a relaunch that leaves a predecessor's pid in the file is a launch-contract violation.

## Workflow gap
- **Bug observed:** #813's run-4 relaunch (2026-07-02 00:31) left run-2's stale pid (6267) in /workspace/logs/issue-813.pid, so the poll chain would have watched a dead process; a corrective run-5 launch (00:39) was needed solely to fix the pid file.
- **Why it is a workflow gap:** pod-side-reporting.md specifies the sentinel/[phase=] contract but not the pid-file refresh-on-relaunch invariant; the SKILL.md launch snippets write the pid at first launch only.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)
- `.claude/rules/pod-side-reporting.md`: add a "pid-file contract" bullet: launchers write the pid file ATOMICALLY (tmp+rename) on EVERY (re)launch; a relaunch script includes the pid-file rewrite in the same command chain as the setsid/nohup launch (the #813 run-5 correction as the worked example); pollers treat a pid file older than the newest run-launched marker as suspect.
- Cross-check `.claude/skills/issue/SKILL.md` launch snippets reference the rule (pointer only; keep the full text in the rule).

## Scope / surfaces
- Primary target: `.claude/rules/pod-side-reporting.md`
- Grep first: `grep -rn "pid_file\|\.pid" .claude/rules/ .claude/skills/issue/SKILL.md scripts/poll_pipeline.py | head`

## Constraints / invariants
- Workflow-surface only; workflow_lint passes.

## Provenance
- workflow_fix_target: .claude/rules/pod-side-reporting.md
- fingerprint: (wrapper)
