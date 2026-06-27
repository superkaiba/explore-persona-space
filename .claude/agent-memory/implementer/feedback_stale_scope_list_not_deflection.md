---
name: stale-scope-list-not-deflection
description: A target_file missing from the workflow-surface scope lists but clearly workflow machinery means the LIST is stale — apply the fix AND update the three mirror lists
metadata:
  type: feedback
---

When a workflow-fix candidate's `target_file` is absent from
`workflow.yaml § workflow_fix_on_bug.applies_to_workflow_surface` but the
file is clearly workflow machinery (referenced by skills/agents, pinned by
workflow tests, a cron wrapper, or an implementation module behind a listed
CLI like `task.py`), treat the omission as a stale scope list, NOT as
grounds for an out-of-scope deflection.

**Why:** the scope lists are maintained by hand and lag the codebase; a
deflection on a genuinely-workflow file wastes the whole spawn and bounces
a valid fix. Worked examples: `src/explore_persona_space/task_workflow_migrate.py`
(edited 2026-06-09 twice — fixture reconciliation 75c78e9f3, then the v4
converter retirement) while the lists named the nonexistent
`scripts/task_workflow.py`; fixed 2026-06-09 by adding the two src/ API
modules to all three lists. Also `scripts/autonomous_session_watch.py`
(the #530 keep-running fix, 2026-06-09): missing from all three mirror
lists despite being the watcher the `issue`/`issue-tick` skills depend on —
a strict deflection would have wrongly failed a high-confidence incident fix.
Also `.claude/agent-memory/analyzer/feedback_verifier_h3_extraction_bug.md`
(retiring a stale workaround memory, 2026-06-09): persistent agent memories
are always-loaded guidance steering workflow agents, so corrections to them
are workflow-surface fixes; fixed by adding `.claude/agent-memory/**/*.md`
to all three lists.

**In-practice test for "clearly workflow machinery":** referenced by
`.claude/skills/` or `.claude/agents/` files, has a dedicated
`tests/test_*.py` workflow test, has a cron wrapper, or is imported by
listed scripts.

**How to apply:** apply the requested fix, then ALSO update the three
mirror lists in the same run — `.claude/workflow.yaml`
(`applies_to_workflow_surface`), `.claude/rules/workflow-fix-on-bug.md`
(in-scope + out-of-scope sections), and `.claude/agents/workflow-improver.md`
(In scope / Out of scope) — so the next spawn's scope validation passes
honestly. Reserve true deflection for files that are genuinely experiment
code (training/eval/data generation) regardless of list membership.

Related: [[preexisting-lint-test-failures]].
