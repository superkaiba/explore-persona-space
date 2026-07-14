---
title: 'workflow-fix: lint SKILL.md git recipes against root guard'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4325627bf5b8
- daily-auto-filed
created_at: '2026-07-09T06:58:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The class ''documented
  recipe command that the repo''s own guard_repo_root_branch.sh PreToolUse hook blocks''
  has no mechanical check — #1047''s gate-block cleanup restore recipe shipped without
  a -C waiver at 2 sites and survived plan review, fact-check, and a 6-critic ensemble;
  only the code-reviewer executing the hook caught it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1047 (recursion-guarded workflow-fix session).

## Goal

Mechanically guarantee that every executable git recipe documented in the /issue SKILL.md passes the repo-root branch guard the fleet actually runs.

## Workflow gap

- **Bug observed:** The class 'documented recipe command that the repo's own guard_repo_root_branch.sh PreToolUse hook blocks' has no mechanical check — #1047's gate-block cleanup restore recipe shipped without a -C waiver at 2 sites and survived plan review, fact-check, and a 6-critic ensemble; only the code-reviewer executing the hook caught it.
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/workflow_lint.py, .claude/skills/issue/SKILL.md); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
def check_skill_md_git_recipes_pass_root_guard():
    for block in fenced_bash_blocks(SKILL_MD):
        for line in executable_git_lines(block):
            rc = run(["bash", "scripts/guard_repo_root_branch.sh"],
                     stdin=json.dumps({"tool_input": {"command": line}}))
            if rc == 2: fail(f"SKILL.md recipe blocked by root guard: {line!r}")
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py, .claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py, .claude/skills/issue/SKILL.md
- origin: parked candidate on task #1047 at 2026-07-05T14:28:47Z

Verbatim parked note:

> source: prose-followup (code-reviewer, round 1). target_file: scripts/workflow_lint.py (or tests/), .claude/skills/issue/SKILL.md. proposed_change: a lint/test check that extracts executable (non-comment) git lines from SKILL.md fenced bash blocks and feeds each through scripts/guard_repo_root_branch.sh as {"tool_input":{"command":...}}, FAILing on exit 2 — the class "documented recipe command the repo's own PreToolUse hook blocks" survived plan review + fact-check + 6-critic ensemble and was caught only by the code-reviewer executing the hook (finding 1: the gate-block cleanup restore recipe without a -C waiver, 2 sites; the hook then also blocked THIS candidate's inline --note post for quoting the same literal — live confirmation of the class). routed: parked — running under workflow_fix_target recursion guard; log + notify, not auto-filed. related_task: #1047

