---
title: 'daily-fix: spec-freshness awk exclusion keys on subject shap'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c758ab311e19
- daily-auto-filed
created_at: '2026-07-29T07:06:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): the Step 5a / Step 10d
  family-dirty detection excludes branch-side commits whose SUBJECT contains the bare
  token "spec-freshness"; task #1747''s own DELIVERABLE commit ("task #1747: Step
  5a spec-freshness sync sources fetched origin/main") legitimately carries the token,
  so .claude/skills read as CLEAN and the Step 10d post-gate re-sync would have clobbered
  the deliverable with origin/main''s copy —'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step C parked-candidate sweep (2026-07-28) from a formal candidate block parked on task #1747 (ts 2026-07-28T12:28:19Z, fp c758ab311e19; emitted during #1747's own merge flow).

## Goal

Tighten the Step 5a / Step 10d family-dirty detection's commit-subject exclusion to match the prescribed sync-commit subject SHAPE ("sync workflow-surface specs from") instead of the bare token "spec-freshness", so a deliverable commit legitimately carrying the token is not silently excluded.

## Workflow gap

- **Bug observed:** the family-dirty detection excludes branch-side commits whose SUBJECT contains the bare token "spec-freshness"; #1747's own DELIVERABLE commit ("task #1747: Step 5a spec-freshness sync sources fetched origin/main") legitimately carries the token, so `.claude/skills` read as CLEAN and the Step 10d post-gate re-sync would have clobbered the deliverable with origin/main's copy — the session had to force the workflow family dirty by hand (2026-07-28).
- **Why it is a workflow gap:** the exclusion is meant to skip only PRESCRIBED sync commits, but it keys on a bare substring any wf-fix ABOUT the sync machinery will legitimately carry — a fail-open clobber, the inverse of the guard's fail-safe design.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'index($0, "spec-freshness")' .claude/skills/issue/SKILL.md` → 3 hits (lines 2436, 10426, 11711 — the candidate names two copies; there are THREE awk copies to fix in lockstep), and the prescribed sync-commit subject "sync workflow-surface specs from" exists verbatim at SKILL.md:2471/10413/11728, so the proposed anchor string is real (2026-07-29 UTC). Pin test: `tests/test_issue_skill_lint_family_sync.py:193` asserts the awk string verbatim — must be updated in the same commit. Landed-fix history check: `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` shows #1747's own merge (`fed4be9f6b`) which added the fetched-origin sourcing, not this exclusion tightening; the bare-token awk is still live at all 3 sites.

## Proposed change (candidate diff sketch — refine in planning)

```diff
- | awk 'index($0, "spec-freshness") == 0'
+ | awk 'index($0, "sync workflow-surface specs from") == 0'
```

All THREE copies (SKILL.md:2436 Step 5a, :10426 Guard block, :11711 Step 10d), kept in lockstep per the #1714 drift-guard pin test; update the Guard-3 subject-variant prose + `tests/test_issue_skill_lint_family_sync.py`'s awk assertion in the same commit.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (3 awk copies), `tests/test_issue_skill_lint_family_sync.py` (pin assertion)
- Grep for further copies before editing: `grep -rn 'spec-freshness") == 0' .claude/ scripts/ tests/`.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- The #1714 lockstep pin test must pass after the edit.
- Recursion guard applies to the spawned session.

## Provenance

- sha-verify (filing-time, #1467): `c758ab311e19` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: c758ab311e19

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: the Step 5a / Step 10d family-dirty detection excludes branch-side commits whose SUBJECT contains the bare token "spec-freshness"; task #1747's own DELIVERABLE commit ("task #1747: Step 5a spec-freshness sync sources fetched origin/main") legitimately carries the token, so .claude/skills read as CLEAN and the Step 10d post-gate re-sync would have clobbered the deliverable with origin/main's copy — the session had to force the workflow family dirty by hand (2026-07-28, this task's merge flow).
why_workflow_gap: the exclusion is meant to skip only PRESCRIBED sync commits, but it keys on a bare substring any wf-fix ABOUT the sync machinery will legitimately carry — a fail-open clobber, the inverse of the guard's fail-safe design.
proposed_change: tighten the awk exclusion (both the Step 5a block and the Step 10d inline copy, kept in lockstep per the #1714 drift-guard pin test) to match the prescribed sync-commit subject SHAPE, e.g. the substring "sync workflow-surface specs from", instead of the bare token "spec-freshness"; update the Guard-3 subject-variant prose + the pin test's awk assertion in the same commit.
confidence: medium
related_task: #1747
<!-- /workflow-fix-candidate -->
