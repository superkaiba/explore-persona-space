---
title: 'workflow-fix: add lint-family-sync pin test to Step 5a workflow family'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6889b3fa6343
created_at: '2026-07-30T18:37:55Z'
has_clean_result: false
origin_prompt: 'orchestrator own-observation on #1768 round-2 merge: Step 5a synced
  SKILL.md but not the coupled pin test tests/test_issue_skill_lint_family_sync.py
  (absent from SPECS/FAMILY_OF), redding the Step 10d TG gated leg on test_step5a_exclusion_is_subject_scoped
  with zero branch-side commits'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1768 (emitting agent: orchestrator, own observation during the
round-2 Step 10d merge).

## Goal

Add `tests/test_issue_skill_lint_family_sync.py` to the Step 5a SPECS list and FAMILY_OF workflow family so it syncs atomically with `.claude/skills` (SKILL.md).

## Workflow gap

- **Bug observed:** Step 5a spec-freshness synced `.claude/skills/issue/SKILL.md` from origin/main (commit 5a4a573d88 on issue-1768) but left the coupled pin test `tests/test_issue_skill_lint_family_sync.py` stale, redding the Step 10d TG gated leg on a NEW node (`test_step5a_exclusion_is_subject_scoped`) that no branch commit caused — main's #1807 (merge 9906468844) had updated the test AND SKILL.md together; the branch got only the SKILL.md half.
- **Why it is a workflow gap:** the test pins literal snippet text in SKILL.md (a coupled pair, exactly the FAMILY_lint / FAMILY_guard class the #1714 family-atomic sync exists for), but it is absent from both `SPECS` and `FAMILY_OF` in the Step 5a block — so the sync mechanism structurally cannot keep the pair coherent. This is the documented "(α) non-family rules-pin test skew" residual, now hit live (2026-07-30, issue-1768 round-2 gate run 1 verdict `block`); the manual recovery cost one extra gate cycle (~15 min lint legs) + a hand checkout/commit/push.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'test_issue_skill_lint_family_sync' .claude/skills/issue/SKILL.md` → 1 hit at line 11836 (a prose REFERENCE in the Step 10d section, not a family entry); `grep -n 'FAMILY_OF\[' .claude/skills/issue/SKILL.md` → 9 entries, none naming this test (absence claim: the 0-hit in the family map IS the evidence; the map is a shell array literal, not a text-matching guard, so a verbatim grep binds); landed-fix history: `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 8 commits, none adds the test to the family map (closest: 0c7aacf301 / #1709 widened SPECS to `:(glob)tests/test_guard_*.py` — the precedent for this exact widening class) (2026-07-30)

## Proposed change (candidate diff sketch — refine in planning)

```
# .claude/skills/issue/SKILL.md, Step 5a family-atomic sync block (~line 2457):
+ FAMILY_OF["tests/test_issue_skill_lint_family_sync.py"]="workflow"    # pins literal snippet text in .claude/skills/issue/SKILL.md — must sync atomically with .claude/skills (#1807-class skew: main lands test+SKILL.md together, branch syncs only SKILL.md → gated-only TG red)
# and add tests/test_issue_skill_lint_family_sync.py to the SPECS line (~2470)
# update the "3 coupled families" header comment to name the new workflow-family member
# consider whether the family-declaration pin test itself (test_issue_skill_lint_family_sync.py)
# needs a new assertion covering its own membership (self-pinning)
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'FAMILY_OF' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan. Note the Step 9c step-1a block references the Step 5a
  family-atomic block (a BINDING reference, never a third inlined copy — #1807),
  so the SKILL.md Step 5a block is the single edit site; the pin test
  `tests/test_issue_skill_lint_family_sync.py` may need its family-declaration
  assertion updated to match (it pins the FAMILY_OF entries).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).
- Fail-safe direction preserved: any dirty member widens the skip to the whole
  family, never narrows it into a clobber (#535/#1714) — adding the test to the
  workflow family means a branch with deliberate branch-side edits to the test
  will now ALSO skip syncing `.claude/skills`; the plan should confirm this
  widening is acceptable (it is the same trade every family member makes).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 6889b3fa6343

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: Step 5a synced SKILL.md from origin/main but left the coupled pin test stale, redding the Step 10d TG gated leg on a NEW node no branch commit caused
why_workflow_gap: tests/test_issue_skill_lint_family_sync.py pins literal SKILL.md snippet text but is absent from the Step 5a SPECS + FAMILY_OF map, so the family-atomic sync structurally cannot keep the coupled pair coherent
proposed_change: Add tests/test_issue_skill_lint_family_sync.py to the Step 5a SPECS list and FAMILY_OF workflow family so it syncs atomically with .claude/skills
diff_sketch: |
  + FAMILY_OF["tests/test_issue_skill_lint_family_sync.py"]="workflow"
  + (add the path to the SPECS line; update the coupled-families header comment)
confidence: high
related_task: #1768
<!-- /workflow-fix-candidate -->
