---
title: 'workflow-fix: code-reviewer diff-size budget on long-lived branches'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2a27315ebf2d
created_at: '2026-07-02T04:36:45Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 722 r2: reviewer context-bomb on
  branch-wide diff (main...HEAD 1.96 MB); size-then-scope to round commits'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #722 (emitting agent: orchestrator, /issue 722 session).

## Goal

code-reviewer Step 0 sizes the branch diff before reading it and falls back to per-round-commit `git show` when it exceeds a byte budget.

## Workflow gap

- **Bug observed:** two subagent spawns on #722 r2 (an experiment-implementer, then a code-reviewer) died on autocompact thrash after loading the whole-branch diff — on that 4-day-old same-issue-follow-up branch `git diff main...HEAD` is 1.96 MB (43K insertions, 136 files of prior ALREADY-REVIEWED rounds) and `main..HEAD` is 31.6 MB. The orchestrator had to measure the bomb and re-spawn the reviewer with a hand-written command whitelist.
- **Why it is a workflow gap:** `.claude/agents/code-reviewer.md` (Step 0, line ~72 `git diff --name-only main...HEAD` + the "Read every line of the diff" directive) prescribes branch-wide diff reads with no size guard; long-lived follow-up branches make the cumulative branch diff unboundedly large even when the ROUND's own reviewable delta (the fix commits) is tiny (36 KB + 29 KB on #722 r2).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
In code-reviewer.md Step 0 (and the round-scope guidance for revision rounds):
+ Before reading any diff body, size it: `git diff main...HEAD | wc -c` (name-only/stat forms stay unrestricted).
+ If the branch-wide diff exceeds ~300 KB, DO NOT read it. Scope the review to the
+ round's own commits (`git show <sha>` per implementer-marker-named commit, or the
+ `<parent>..HEAD` round range) — on revision rounds the earlier rounds were already
+ reviewed, so the branch-wide diff is redundant context by construction.
+ Never run the two-dot `main..HEAD` body on a worktree branch (includes everything
+ main advanced; 31.6 MB on #722).
```

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'main\.\.\.HEAD' .claude/ CLAUDE.md scripts/`) and update every hit that instructs reading a branch-wide diff BODY without a size check; list them in the plan. (Sibling context-hygiene gap: `.claude/agents/experiment-implementer.md` — the #722 implementer spawn also thrashed; consider a shared "size before you read" bullet.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: 2a27315ebf2d

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/code-reviewer.md
bug_observed: two reviewer/implementer spawns on 722 r2 died on autocompact after loading the whole-branch diff (main...HEAD 1.96 MB on a long-lived followup branch)
why_workflow_gap: code-reviewer.md Step 0 prescribes branch-wide diff reads with no size guard; follow-up branches make the cumulative diff unbounded while the round's own delta stays tiny
proposed_change: code-reviewer Step 0 sizes the branch diff before reading it and falls back to per-round-commit git show when it exceeds a byte budget
diff_sketch: |
  + size the diff first (wc -c); >~300 KB -> review the round's own commits via git show
  + never read the two-dot main..HEAD body on a worktree branch
confidence: high
related_task: #722
<!-- /workflow-fix-candidate -->
