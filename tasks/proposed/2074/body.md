---
title: 'workflow-fix: split code-review into per-commit sub-verdicts for large rounds'
kind: infra
tags:
- wf-fix
- wf-fix-fp:520e1490b1ce
created_at: '2026-08-04T19:11:08Z'
has_clean_result: false
origin_prompt: 'Task #2054 code-reviewer-lean thrash at Step 5 round 1 (9 tool uses
  / 385s / 3 autocompacts on 244KB diff, 6911 lines, 5 unit commits). Per CLAUDE.md
  § Autocompact-thrash subagent deaths, this is the #2054 shape (fail-loud). The workflow
  gap: no per-commit split-review pattern for legitimately-sized experiment rounds
  — Step 4b already pre-splits implementer over >4 deliverables, the symmetric review-side
  gap is unwired.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #2054 (emitting agent: orchestrator observation after `code-reviewer-lean` thrash at Step 5 round 1).

## Goal

Wire per-Unit-commit split-review shape into /issue Step 5 code-review dispatch for rounds with >4 commits (one code-reviewer-lean per commit against that commit's own diff).

## Workflow gap

- **Bug observed:** `code-reviewer` + `code-reviewer-lean` both autocompact-thrash on 200-400 KB round-scoped diffs from legitimately-sized (5+ unit commits) experiment rounds; deaths at ~4-10 tool uses. Task #2054 round 1: `code-reviewer-lean` thrashed at 9 tool uses / 385s / 3 autocompacts on a 244 KB diff (6,911 lines, 5 unit commits).
- **Why it is a workflow gap:** CLAUDE.md § "Autocompact-thrash subagent deaths" cites this shape as the #2054 fail-loud trigger. The rule bars unbounded lean-twin retry AND reserves inline-adopt only for workflow-fix tasks fixing this failure mode. So for a legitimately-sized `kind: experiment` round the class produces NO durable code-review verdict — Thomas must recover manually. The Step 4b `pre_split_multi_deliverable` rule ALREADY splits IMPLEMENTATION over >4 deliverables into sequential units to avoid implementer thrash; the SYMMETRIC gap on the review side is unwired.
- **Confidence (emitter):** medium — the shape is well-documented in CLAUDE.md; the exact split-review shape (per-commit vs per-file vs per-lens) is a design call the spawned session's planner should make.
- verified-at-filing: `grep -l "code-reviewer" .claude/agents/code-reviewer.md .claude/agents/code-reviewer-lean.md; grep -c "Step 5" .claude/skills/issue/SKILL.md` → 6 hits per agent file (code-reviewer.md and code-reviewer-lean.md); 81 hits of "Step 5" in SKILL.md (2026-08-04)

## Proposed change (candidate diff sketch — refine in planning)

```
# In .claude/skills/issue/SKILL.md Step 5, add a pre-split gate mirroring Step 4b's:
+ If the round-scoped diff exceeds N bytes (e.g. 100KB) OR the round has >4 unit
+ commits, split the review into per-commit sub-verdicts:
+   1. For each unit commit `<sha>` in `<parent>..HEAD`, spawn code-reviewer-lean
+      with a brief pointing at `git diff <sha>~..sha` (single-commit diff).
+   2. Collect per-commit verdicts (each PASS/REVISE).
+   3. Compose round-level verdict: ALL PASS ⇒ round PASS; ANY REVISE ⇒ round
+      REVISE (union blockers across commits).
```

## Scope / surfaces

- Primary targets: `.claude/agents/code-reviewer.md` (Step 0.9 subclasses; adds a per-commit-split subclass), `.claude/skills/issue/SKILL.md` (Step 5 pre-split gate), `.claude/agents/code-reviewer-lean.md` (brief-format prescription).
- Grep the workflow surface for the pattern before editing.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- The per-commit split must NOT re-introduce whole-branch reads that already fail (Step 0 / `.claude/rules/diff-size-budget.md` bind).

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md,.claude/agents/code-reviewer-lean.md,.claude/skills/issue/SKILL.md
- fingerprint: 520e1490b1ce

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/code-reviewer.md,.claude/agents/code-reviewer-lean.md,.claude/skills/issue/SKILL.md
bug_observed: code-reviewer[-lean] autocompact-thrashes on 200-400KB round-scoped diffs from legitimately-sized (5+ unit commits) experiment rounds; the fail-loud escalation produces no durable verdict for the class.
why_workflow_gap: CLAUDE.md § "Autocompact-thrash subagent deaths" cites this shape as fail-loud, and Step 4b already pre-splits implementer over >4 deliverables — the symmetric review-side gap is unwired.
proposed_change: Wire per-Unit-commit split-review shape into /issue Step 5 code-review dispatch for rounds with >4 commits (one code-reviewer-lean per commit against that commit's own diff).
diff_sketch: |
  # In .claude/skills/issue/SKILL.md Step 5, add a pre-split gate mirroring Step 4b's:
  + If the round-scoped diff exceeds N bytes (e.g. 100KB) OR the round has >4 unit
  + commits, split the review into per-commit sub-verdicts:
  +   1. For each unit commit `<sha>` in `<parent>..HEAD`, spawn code-reviewer-lean
  +      with a brief pointing at `git diff <sha>~..sha` (single-commit diff).
  +   2. Collect per-commit verdicts (each PASS/REVISE).
  +   3. Compose round-level verdict: ALL PASS ⇒ round PASS; ANY REVISE ⇒ round
  +      REVISE (union blockers across commits).
confidence: medium
related_task: #2054
<!-- /workflow-fix-candidate -->
