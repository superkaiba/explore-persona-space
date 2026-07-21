---
title: 'workflow-fix: LESSONS.md over ratchet — trim/bump _LESSONS_RATCHET_BYTES (red
  default lint on main)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3b7ff677b195
created_at: '2026-07-17T19:04:00Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1335 seed44 implementer round: LESSONS.md
  6675 > ratchet 6650; three workflow_lint tests red on live main'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1335 (emitting agent: experiment-implementer).

## Goal

Review the newest LESSONS.md row(s) for one-line-index compliance (trim if oversized), then bump _LESSONS_RATCHET_BYTES to the resulting live size (<= _LESSONS_MAX_BYTES) in the same diff.

## Workflow gap

- **Bug observed:** main's current .claude/rules/LESSONS.md is 6675 bytes, past the _LESSONS_RATCHET_BYTES=6650 growth ratchet in scripts/workflow_lint.py, so the no-flags default lint FAILs and three tests (test_workflow_lint_default_exits_zero, test_check_lessons_index_passes_on_live_repo, test_lessons_ratchet_constants_sane) fail on the LIVE origin/main tree.
- **Why it is a workflow gap:** the ratchet's same-diff-bump contract was not enforced on whichever session last grew LESSONS.md — the gate is supposed to make index growth a deliberate reviewed act, but a landed edit skipped the bump and now every Step 9c gate run on any branch inherits a red default lint.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n '_LESSONS_RATCHET_BYTES' scripts/workflow_lint.py` → 3 hits at lines 9581/9589/9608 (constant = 6650) AND `wc -c .claude/rules/LESSONS.md` → 6675 bytes on the main checkout (2026-07-17). Per-target: scripts/workflow_lint.py 3 hits; .claude/rules/LESSONS.md size-claim confirmed 6675 > 6650. Recent growth commit: 26d450bce8 (issue-1462 LESSONS trigger extensions) — `git log --since='3 days ago' -- .claude/rules/LESSONS.md`; the last deliberate bump was 532e7aa35f (issue-1435, 6400→6650).

## Proposed change (candidate diff sketch — refine in planning)

```
- _LESSONS_RATCHET_BYTES = 6650
+ _LESSONS_RATCHET_BYTES = 6675  # (or the post-trim size; deliberate reviewed bump)
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py, .claude/rules/LESSONS.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '_LESSONS_RATCHET_BYTES' .claude/ CLAUDE.md scripts/ tests/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py, .claude/rules/LESSONS.md
- fingerprint: 3b7ff677b195

<!-- workflow-fix-candidate v1 -->
target_file: scripts/workflow_lint.py, .claude/rules/LESSONS.md
bug_observed: main's current .claude/rules/LESSONS.md is 6675 bytes, past the _LESSONS_RATCHET_BYTES=6650 growth ratchet in scripts/workflow_lint.py, so the no-flags default lint FAILs and three tests (test_workflow_lint_default_exits_zero, test_check_lessons_index_passes_on_live_repo, test_lessons_ratchet_constants_sane) fail on the LIVE origin/main tree (both files verified byte-identical to origin/main from the issue-1335 worktree, 2026-07-17).
why_workflow_gap: the ratchet's same-diff-bump contract was not enforced on whichever session last grew LESSONS.md — the gate is supposed to make index growth a deliberate reviewed act, but a landed edit skipped the bump and now every Step 9c gate run on any branch inherits a red default lint.
proposed_change: review the newest LESSONS.md row(s) for one-line-index compliance (trim if oversized), then bump _LESSONS_RATCHET_BYTES to the resulting live size (<= _LESSONS_MAX_BYTES) in the same diff.
diff_sketch: |
  - _LESSONS_RATCHET_BYTES = 6650
  + _LESSONS_RATCHET_BYTES = 6675  # (or the post-trim size; deliberate reviewed bump)
confidence: high
related_task: #1335
<!-- /workflow-fix-candidate -->
