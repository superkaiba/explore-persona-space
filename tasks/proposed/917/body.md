---
title: 'workflow-fix: follow-up brief marker versions defer to max+1 rule'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bb0e9be7e388
created_at: '2026-07-03T08:20:36Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from experiment-implementer on #825 (onpolicy-user-turn
  round): brief hardcoded epm:experiment-implementation v1 on a task already at v6'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #825 (emitting agent: experiment-implementer).

## Goal

The same-issue follow-up loop's implementer brief guidance should defer marker versions to the standing max+1 posting rule instead of hardcoding per-round version numbers.

## Workflow gap

- **Bug observed:** The same-issue follow-up loop's implementer brief instructed "post as epm:experiment-implementation v1" on a task whose marker sequence was already at v6, causing a version collision the implementer had to detect and re-post around.
- **Why it is a workflow gap:** The brief template hardcodes per-round version numbers instead of deferring to the standing max+1 posting rule, so every follow-up round on a task with prior implementation markers reproduces the #389 collision class.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
- post as `epm:experiment-implementation v1` via ...
+ post as `epm:experiment-implementation v<max_existing+1>` (read events.jsonl for the
+ highest existing version of the marker key first; task.py post-marker does NOT
+ auto-increment) via ...
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for hardcoded per-round marker-version instructions in brief templates (`grep -rn 'epm:experiment-implementation v1' .claude/ CLAUDE.md`) and update every hit; list them in the plan. Consider whether the Step 4b brief template and the same-issue follow-up loop § step 3 both need the max+1 wording.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: bb0e9be7e388

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: The same-issue follow-up loop's implementer brief instructed "post as epm:experiment-implementation v1" on a task whose marker sequence was already at v6, causing a version collision the implementer had to detect and re-post around.
why_workflow_gap: The brief template hardcodes per-round version numbers instead of deferring to the standing max+1 posting rule, so every follow-up round on a task with prior implementation markers reproduces the #389 collision class.
proposed_change: The follow-up-loop brief should say "post at the next version after the highest existing epm:experiment-implementation row in events.jsonl" rather than a literal version.
diff_sketch: |
  - post as `epm:experiment-implementation v1` via ...
  + post as `epm:experiment-implementation v<max_existing+1>` (read events.jsonl for the
  + highest existing version of the marker key first; task.py post-marker does NOT
  + auto-increment) via ...
confidence: high
related_task: #825
<!-- /workflow-fix-candidate -->
