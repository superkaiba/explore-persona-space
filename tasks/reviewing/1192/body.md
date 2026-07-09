---
title: 'workflow-fix: reviewer writable-tempdir outside worktree'
kind: infra
tags:
- wf-fix
- wf-fix-fp:82f0cfeb5c2b
- daily-auto-filed
created_at: '2026-07-09T07:00:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The Step 4 writable-tempdir
  fallback recipe (RTMP="$(pwd)/.claude/cache/reviewer-tmp-$$") places pytest TMPDIR/tmp_path
  INSIDE the git worktree, false-FAILing git-root-resolving test fixtures (observed
  live on #853 round 2: test_check_no_workflow_improver_spawn_flags_a_stray_spawn).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #853 (recursion-guarded workflow-fix session).

## Goal

Point the recipe outside the tree (e.g. RTMP="${XDG_RUNTIME_DIR:-/tmp}/reviewer-tmp-$$") or add an explicit caveat + fallback for git-root-resolving fixtures next to it; mind the agent-spec size ratchet when editing.

## Workflow gap

- **Bug observed:** The Step 4 writable-tempdir fallback recipe (RTMP="$(pwd)/.claude/cache/reviewer-tmp-$$") places pytest TMPDIR/tmp_path INSIDE the git worktree, false-FAILing git-root-resolving test fixtures (observed live on #853 round 2: test_check_no_workflow_improver_spawn_flags_a_stray_spawn).
- **Why it is a workflow gap:** A reviewer-side recipe that makes real tests false-FAIL degrades the code-review gate's signal and invites wrong CONCERNS/FAIL verdicts on healthy diffs.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

In code-reviewer.md Step 4 recipe:
- RTMP="$(pwd)/.claude/cache/reviewer-tmp-$$"
+ RTMP="${XDG_RUNTIME_DIR:-/tmp}/reviewer-tmp-$$"   # OUTSIDE the worktree: an in-tree TMPDIR
+ # makes git-root-resolving test fixtures resolve the worktree repo and false-FAIL

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md
- origin: parked candidate on task #853 at 2026-07-02T10:31:58Z

Verbatim parked note:

parked: EPM_WORKFLOW_FIX_SESSION — two round-2 reviewer-surfaced follow-ups (recursion guard: logged, not auto-routed):

1. target_file: .claude/agents/codex-code-reviewer.md — its Step 0.6 INCLUDING enumeration does not name the NEW script-mode deferred-imports paragraph, so a composer could abridge it out of the twin's prompt (the #606 omission class). proposed_change: add the paragraph to the enumeration (one line). confidence: high.
2. target_file: .claude/agents/code-reviewer.md — the Step 4 writable-tempdir recipe ($(pwd)/.claude/cache/reviewer-tmp-$$) places pytest tmp_path INSIDE the git worktree, false-FAILing git-root-resolving test fixtures (observed live this round: test_check_no_workflow_improver_spawn_flags_a_stray_spawn). proposed_change: point the recipe outside the tree (e.g. ${XDG_RUNTIME_DIR:-/tmp}/reviewer-tmp-$$) or add the caveat. confidence: high. NOTE: code-reviewer.md is at 70,757/72,000 B post-#853 — this edit may need the relocation candidate (see marker v2) first.

related_task: #853 (round-2 code-review, 2026-07-02).
