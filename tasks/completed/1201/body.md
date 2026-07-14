---
title: 'workflow-fix: PreToolUse guard on bare git pull at repo root'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1ff9bdc35b92
- daily-auto-filed
created_at: '2026-07-09T07:00:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): A bare `git pull` at the
  shared repo root bypasses scripts/sync_repo_root.py; the prose-only rule has a demonstrated
  miss rate (#967-class: CLAUDE.md named the helper and a session hand-rolled a pull
  anyway).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08, slice 5) from a
candidate parked on task #1047 by a recursion-guarded workflow-fix session.

## Goal

Convert the prose-only repo-root pull rule into a mechanical PreToolUse guard: block a bare `git pull` at the repo root unless it comes from scripts/sync_repo_root.py's own subprocess (env-sentinel exemption such as EPS_SYNC_HELPER=1).

## Workflow gap

- **Bug observed:** Sessions hand-roll `git pull` at the shared repo root instead of the single-flight sync_repo_root.py helper; prose alone has a demonstrated miss rate (the #967-class incident: CLAUDE.md named the helper before #967 and the session hand-rolled a pull anyway).
- **Why it is a workflow gap:** The repo-root git-safety rules are enforced mechanically for checkout/reset/merge/piped-push but not for pull — the one remaining prose-only lane, and the one with a demonstrated miss.
- **Confidence (emitter):** prose-followup (critic ensemble, alternatives lens — Claude + Codex convergent) on #1047
- **Triage evidence (2026-07-08):** NOT fixed on main: .claude/settings.json's only pull-related PreToolUse hook is the ssh-pod dirty-tree WARNING (settings.json:83); guard_repo_root_branch.sh mentions `git pull` only in a comment (:303) and blocks checkout/reset/merge, not pull. Sibling guards exist (#1048 piped-git-push hook, #1128 root-merge block) — precedent, not dedup. No open wf-fix task on this; no retraction on #1047.

## Proposed change (candidate diff sketch — refine in planning)

```
+ new scripts/guard_repo_root_pull.sh (PreToolUse Bash hook, sibling of
+ guard_repo_root_branch.sh): deny `git pull` with cwd == repo root unless
+ EPS_SYNC_HELPER=1 is set; block message points at
+ `uv run python scripts/sync_repo_root.py`.
+ wire into .claude/settings.json PreToolUse; export EPS_SYNC_HELPER=1 around
+ sync_repo_root.py's internal pull subprocess.
```

## Scope / surfaces

- Primary target: `.claude/settings.json, scripts/ (new hook script)`
- Secondary: `scripts/sync_repo_root.py` (set the sentinel around its own pull), `scripts/workflow_lint.py` hook-consistency checks if any.
- Grep the workflow surface for the pattern before editing
  (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/settings.json, scripts/ (new hook script)
- origin: parked candidate on task #1047 at 2026-07-05T13:48:40Z

Verbatim parked note:

source: prose-followup (critic ensemble, alternatives lens — Claude + Codex convergent). target_file: .claude/settings.json, scripts/ (new hook script). proposed_change: PreToolUse hook (sibling of guard_repo_root_branch.sh) blocking a bare 'git pull' at the repo root unless invoked via scripts/sync_repo_root.py (env-sentinel exemption for the helper's own subprocess pull, e.g. EPS_SYNC_HELPER=1) — converts the prose-only #967-class rule into an all-lane mechanical guard (prose alone has a demonstrated miss rate: CLAUDE.md named the helper before #967 and the session hand-rolled a pull anyway). routed: parked — running under workflow_fix_target recursion guard (.claude/rules/workflow-fix-on-bug.md § Recursion guard); log + notify, not auto-filed. related_task: #1047
