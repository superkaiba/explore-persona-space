---
title: 'workflow-fix: add skill-pin family to Step 5a spec-freshness sync (test_issue_skill_*
  pin tests)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c0d698230ca6
created_at: '2026-07-30T13:40:26Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #1824 orchestrator: Step 5a spec sync imported main''s
  SKILL.md (#1821) without its paired test_issue_skill_gate_single_flight.py pin test;
  3 pin tests failed the Step 9c gate, costing a manual reconciliation + a full ~30-min
  gate re-run. Proposal: sync tests/test_issue_skill_*.py atomically with .claude/skills
  via the FAMILY_OF map.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1824 (emitting agent: /issue orchestrator).

## Goal

Add a "skill-pin" family to the Step 5a spec-freshness FAMILY_OF map — `.claude/skills` ↔ `:(glob)tests/test_issue_skill_*.py` — so SKILL.md prose-pin tests sync atomically with the skill files they pin (mirrored in the Step 10d post-gate re-sync inline copy + the family-atomicity drift test).

## Workflow gap

- **Bug observed:** on #1824 (2026-07-30), the mandated Step 5a spec-freshness sync imported main's just-updated `.claude/skills/issue/SKILL.md` (#1821 merge `0595d18b5f`, 11:58Z — 16 min after the branch cut) while its paired pin test `tests/test_issue_skill_gate_single_flight.py` (updated in the SAME main commit) stayed at branch-cut vintage — the worktree went mixed-vintage and 3 pin tests failed the Step 9c gate, costing a manual reconciliation + a full ~30-min gate re-run.
- **Why it is a workflow gap:** Step 5a already solves exactly this class for the lint/guard/workflow families (#1560/#1714: "syncing specs without their enforcing family creates vintage skew") but names the SKILL.md-pin-test class as accepted residual (α) with a manual remedy. The residual's cost just materialized as a full gate re-run; the mechanical fix is the existing family pattern — `tests/test_issue_skill_*.py` are prose-pin tests over `.claude/skills` content (the same spec-coupling argument that admitted `:(glob)tests/test_workflow_lint*.py` into the lint family), not behavior tests over branch-era `src/`.
- **Confidence (emitter):** medium — the planner should verify no `test_issue_skill_*.py` imports branch-era `src/`/`scripts/` symbols beyond the accepted `explore_persona_space.workflow` seam (the same residual the lint family accepts), and weigh whether the family should instead join the existing "workflow" family (`.claude/skills` is already a member there — adding the glob to THAT family may be the minimal edit).
- verified-at-filing: `grep -c "FAMILY_OF" .claude/skills/issue/SKILL.md tests/test_issue_skill_lint_family_sync.py` → 26 hits in SKILL.md (the Step 5a block + the Step 10d inline copy) + 17 hits in the drift test (2026-07-30). Incident evidence: #1824 events.jsonl (epm:test-verdict v1 records the run-1 3-fail classification + the f678456720 reconciliation commit).

## Proposed change (candidate diff sketch — refine in planning)

```
  # Step 5a FAMILY_OF map (both copies: Step 5a block + Step 10d post-gate re-sync inline copy):
+ FAMILY_OF[":(glob)tests/test_issue_skill_*.py"]="workflow"   # prose-pin tests over .claude/skills (#1824 vintage skew)
  # SPECS list gains the same glob entry;
  # tests/test_issue_skill_lint_family_sync.py drift guard updated to pin the new member.
```

## Scope / surfaces

- Primary targets: `.claude/skills/issue/SKILL.md` (Step 5a FAMILY_OF/SPECS + the Step 10d inline copy — the drift test enforces the two stay identical), `tests/test_issue_skill_lint_family_sync.py`.
- Grep the workflow surface for the pattern before editing (`grep -rln 'FAMILY_OF' .claude/ scripts/ tests/` at the repo root — worktree copies excluded) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; `tests/test_issue_skill_lint_family_sync.py` (family-atomicity drift guard) stays green with the new member pinned.
- Fail-safe direction preserved: a branch-side edit to ANY `test_issue_skill_*.py` marks the whole family dirty → the sync skips it (status-quo staleness, never a clobber — the #535 semantics).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, tests/test_issue_skill_lint_family_sync.py
- fingerprint: c0d698230ca6

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md, tests/test_issue_skill_lint_family_sync.py
bug_observed: spec-freshness sync imports mains SKILL.md but not its paired test_issue_skill_* pin tests, leaving worktrees mixed-vintage and failing the Step 9c gate
why_workflow_gap: Step 5a's family-atomic sync exists precisely to prevent spec/enforcer vintage skew (#1560/#1714) but leaves SKILL.md prose-pin tests as accepted residual (α); the residual cost a full 30-min gate re-run on #1824
proposed_change: add a skill-pin family to the Step 5a spec-freshness FAMILY_OF map syncing tests/test_issue_skill_*.py pin tests atomically with .claude/skills
diff_sketch: |
  + FAMILY_OF[":(glob)tests/test_issue_skill_*.py"]="workflow"
  + SPECS gains ":(glob)tests/test_issue_skill_*.py"
  + mirror in the Step 10d inline copy + the family-atomicity drift test
confidence: medium
related_task: #1824
<!-- /workflow-fix-candidate -->
