---
title: 'workflow-fix: PreToolUse guard for /tmp deletion sweeps (tmux sockets)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c87a09884180
created_at: '2026-07-17T07:50:58Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1466 implementer: PreToolUse Bash hook
  blocking /tmp-rooted deletion sweeps lacking a tmux-* exclusion (guard_tmp_tmux_sweep.sh
  + settings.json registration); root cause of the 2026-07-16 tmux split-brain.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1466 (emitting agent: implementer).

## Goal

Add a PreToolUse Bash hook that blocks `rm -rf`/`find -delete`/`find|xargs rm` shapes rooted at /tmp unless they exclude `tmux-*` (override env var for deliberate use), registered in .claude/settings.json.

## Workflow gap

- **Bug observed:** An orchestrator session's improvised disk-pressure cleanup (`find /tmp -maxdepth 1 -mtime +2 ... -print0 | xargs -0 -r rm -rf`, no `tmux-*` exclusion) deleted `/tmp/tmux-1001` with the live fleet tmux server socket inside, splitting 39 sessions off the fleet (#1466 root cause, 2026-07-15T23:17:53Z; deleter identified by #1466 Phase A investigation as session 3b499fa0).
- **Why it is a workflow gap:** No PreToolUse guard covers broad /tmp deletion sweeps, so any agent under disk pressure can improvise an age sweep that destroys live IPC sockets; the existing guard family (guard_piped_git_push.sh, guard_harmful_bank_read.sh, guard_lessons_edit.sh, guard_log_dump.sh) shows the mechanism but nothing scopes /tmp deletions.
- **Confidence (emitter):** medium
- verified-at-filing: `ls .claude/hooks/` + `grep -rln "tmp.*sweep|/tmp/tmux|tmux-\*" .claude/hooks/` → 0 hits in 5 hook files (absence-of-guard claim — 0-hit IS the evidence; semantic probe: no hook file mentions /tmp, tmux, or sweep shapes) + `git log --oneline --since='7 days ago' -- .claude/settings.json` → 1 commit (843804cc44, the #1279 LESSONS guard — unrelated, no just-landed /tmp guard) (2026-07-17)

## Proposed change (candidate diff sketch — refine in planning)

+ .claude/hooks/guard_tmp_tmux_sweep.sh: match Bash commands with
+   (rm -rf?|find) .*/tmp( |/) AND a deletion action (-delete|xargs.*rm|rm -rf)
+   AND no "! -name 'tmux-*'" / no /tmp/tmux exclusion -> BLOCK with
+   "add ! -name 'tmux-*' or EPM_ALLOW_TMP_SWEEP=1" message.
+ .claude/settings.json: register hook under PreToolUse Bash matchers.

## Scope / surfaces

- Primary target: `.claude/settings.json, .claude/hooks/guard_tmp_tmux_sweep.sh (new)`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'PreToolUse' .claude/settings.json .claude/hooks/`) and update every hit;
  list them in the plan. Sibling context: #1466 landed `scripts/eps_tmux_env.sh` (durable socket dir) — this guard is the complementary prevention leg; do not duplicate its /tmp-pin logic.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/settings.json, .claude/hooks/guard_tmp_tmux_sweep.sh (new)
- fingerprint: c87a09884180

<!-- workflow-fix-candidate v1 -->
target_file: .claude/settings.json, .claude/hooks/guard_tmp_tmux_sweep.sh (new)
bug_observed: An orchestrator session's improvised disk-pressure cleanup (`find /tmp -maxdepth 1 -mtime +2 ... -print0 | xargs -0 -r rm -rf`, no `tmux-*` exclusion) deleted `/tmp/tmux-1001` with the live fleet tmux server socket inside, splitting 39 sessions off the fleet (#1466 root cause, 2026-07-15T23:17:53Z).
why_workflow_gap: No PreToolUse guard covers broad /tmp deletion sweeps, so any agent under disk pressure can improvise an age sweep that destroys live IPC sockets; the existing guard family (guard_piped_git_push.sh, guard_harmful_bank_read.sh) shows the mechanism but nothing scopes /tmp deletions.
proposed_change: Add a PreToolUse Bash hook that blocks `rm -rf`/`find -delete`/`find|xargs rm` shapes rooted at /tmp unless they exclude `tmux-*` (override env var for deliberate use), registered in .claude/settings.json.
confidence: medium
related_task: #1466
<!-- /workflow-fix-candidate -->
