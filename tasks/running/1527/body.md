---
title: 'workflow-fix: verified-at-filing clause (e) artifact-state'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f2f6dbea603e
- daily-auto-filed
created_at: '2026-07-19T07:05:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): #1497 was filed on artifact-state
  evidence (needs-human absent from cited tasks'' tags) that was actually a deliberate
  2026-07-17 user-directed remove-tag; the verified-at-filing scan read post-mutation
  state as a filing-time drop.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1497 (emitting agent: /issue orchestrator, candidate-block;
parked under the recursion guard, routed by the 2026-07-18 /daily Step C
parked-candidate sweep).

## Goal

Add a clause (e) "artifact-state mutation check" to the verified-at-filing
consistency rules + the anti-pattern table in
`.claude/rules/workflow-fix-on-bug.md`: absence-of-tag/field-on-artifact
evidence binds only after the task folder's git history shows the value was
never applied.

## Workflow gap

- **Bug observed:** #1497 was filed (and a session spawned) on artifact-state
  evidence — needs-human absent from #1140/#1472 tags — that was actually the
  result of a DELIBERATE user-directed remove-tag on 2026-07-17; the
  verified-at-filing scan read post-mutation state and misattributed it to a
  filing-time drop. #1497 was archived as a false-premise filing
  (events.jsonl 2026-07-18T08:41:07Z: create-commit frontmatter carried the
  tag on all five cited tasks; remove-tag commits eaa4e10a67/1bd90f800e).
- **Why it is a workflow gap:** the verified-at-filing mandate's consistency
  clauses ((a)-(d)) cover grep/semantic/relocation/sha binds but have no
  clause for ARTIFACT-STATE evidence — a tag/field absent from a task
  artifact must be checked against the task folder's git history for
  deliberate later mutation before it can support a filing-time-drop claim.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'artifact-state' .claude/rules/workflow-fix-on-bug.md` → 0 hits (absence claim — clause (e) not present in target); repo-wide relocation `grep -rn 'artifact-state mutation check' .claude/ scripts/` → hits only in tasks/archived/1497 events.jsonl copies (the candidate itself), not in any rule file; `git log --oneline --since='7 days ago' -- .claude/rules/workflow-fix-on-bug.md` → 4 commits (280b80b058 open-sibling arm, 1580a7dad6, dc3a465ca2, 9c53b54b81), none adds an artifact-state clause (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
+ (e) **artifact-state mutation check** — a claim that a tag/field was
+     dropped AT FILING, evidenced by its absence on the task artifact,
+     requires the folder's git history at compose time:
+     `git log --follow --format='%h %s' -- <task body.md>` — a create
+     commit CARRYING the value, or any `remove-tag <value>` /
+     mutation commit, refutes the drop claim (#1497: five cited tasks
+     were all created WITH needs-human; a 2026-07-17 user-directed
+     mass remove-tag explained every observation).
```

## Scope / surfaces

- Primary target: `.claude/rules/workflow-fix-on-bug.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'verified-at-filing' .claude/ CLAUDE.md scripts/`) and update
  every surface that enumerates the (a)-(d) clauses (the /daily SKILL.md
  route-2 mandate references them too); list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/workflow-fix-on-bug.md
- fingerprint: 30137f086efe

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/workflow-fix-on-bug.md
bug_observed: #1497 was filed (and a session spawned) on artifact-state evidence — needs-human absent from #1140/#1472 tags — that was actually the result of a DELIBERATE user-directed remove-tag on 2026-07-17; the verified-at-filing scan read post-mutation state and misattributed it to a filing-time drop.
why_workflow_gap: the verified-at-filing mandate's consistency clauses ((a)-(d)) cover grep/semantic/relocation/sha binds but have no clause for ARTIFACT-STATE evidence — a tag/field absent from a task artifact must be checked against the task folder's git history (git log --follow -- <task body>; add-tag/remove-tag commits; create-commit frontmatter) for deliberate later mutation before it can support a filing-time-drop claim.
proposed_change: add a clause (e) "artifact-state mutation check" to the verified-at-filing consistency rules + the anti-pattern table: absence-of-tag/field-on-artifact evidence binds only after the task folder's git history shows the value was never applied (create-commit frontmatter lacks it AND no later remove commits explain the current state).
diff_sketch: |
  + (e) **artifact-state mutation check** — a claim that a tag/field was
  +     dropped AT FILING, evidenced by its absence on the task artifact,
  +     requires the folder's git history at compose time:
  +     `git log --follow --format='%h %s' -- <task body.md>` — a create
  +     commit CARRYING the value, or any `remove-tag <value>` /
  +     mutation commit, refutes the drop claim (#1497: five cited tasks
  +     were all created WITH needs-human; a 2026-07-17 user-directed
  +     mass remove-tag explained every observation).
confidence: high
related_task: #1497
<!-- /workflow-fix-candidate -->
