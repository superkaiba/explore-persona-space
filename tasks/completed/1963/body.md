---
title: 'daily-fix: guard family syncs scripts/guard_*.sh with pin te'
kind: infra
tags:
- wf-fix
- wf-fix-fp:21a18e40d5f4
- daily-auto-filed
created_at: '2026-08-01T07:05:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): Step 5a/10d guard family
  syncs guard pin tests without the scripts/guard_*.sh implementations they exercise,
  red-flagging main-green nodes on version skew (#1860, #1862)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-31 (Step C parked-candidate routing) from TWO independent parked workflow-fix candidates raised the same day on tasks #1860 (park ts 2026-07-31T15:23:12Z, fp ba72538cc167) and #1862 (park ts 2026-07-31T16:46:07Z, fp 8463880a45b7) — both recursion-guarded workflow-fix sessions that hit the SAME live incident: the Step 5a family-atomic spec-freshness sync pulled main-side guard PIN TESTS into the worktree without the guard SCRIPTS they exercise, producing a half-synced tree whose Step 9c gate round red-flagged main-green nodes on pure version skew (observed live on issue-1860: 3 nodes of tests/test_guard_repo_root_branch.py, ~35-min gate round lost; and on issue-1862: 12 false-red test_1861_exitguard_* nodes after tests synced at 15:50Z while main's 3d63506a28 guard-script fix stayed unsynced). Both sessions recovered via an origin/main merge + gate re-run.

## Goal

Add the guard-script implementation set (`:(glob)scripts/guard_*.sh`) to the "guard" family in the Step 5a family-atomic spec-freshness sync AND the Step 10d inlined copy in `.claude/skills/issue/SKILL.md`, so a guard-test sync always carries the guard scripts (and vice versa), and update the family-map drift pin test accordingly.

## Workflow gap

- **Bug observed:** Step 5a's guard family (`FAMILY_OF`) contains `.claude/hooks` + `:(glob)tests/test_guard_*.py` + `tests/test_guard_lessons_edit.py` but NOT the guard scripts the tests execute (`scripts/guard_*.sh`, e.g. `scripts/guard_repo_root_branch.sh` — invoked by the PreToolUse hooks and executed FROM the worktree tree). Syncing the tests without the scripts creates a half-synced tree where main-green guard pin tests fail in every pre-fix worktree.
- **Why it is a workflow gap:** the family-atomic sync (#1714) exists precisely to prevent syncing coupled files apart; the guard family's membership is incomplete — `scripts/guard_*.sh` is spec-coupled to `tests/test_guard_*.py` exactly as `scripts/workflow_lint.py` (already a "lint"-family member) is to `tests/test_workflow_lint*.py`.
- **Confidence (emitter):** high (two independent same-day live incidents, #1860 + #1862)
- verified-at-filing: `grep -n 'FAMILY_OF' .claude/skills/issue/SKILL.md` → Step 5a copy at L2457-2466 (guard family = `.claude/hooks` L2464, `:(glob)tests/test_guard_*.py` L2465, `tests/test_guard_lessons_edit.py` L2466; NO scripts/guard_*.sh entry) and Step 10d copy at ~L12148-12157 (same three guard members; `SPECS_10D` string also lacks scripts/guard_*.sh) — the absence claim is the evidence (2026-08-01 UTC). Supporting: `grep -n 'guard_\*\.sh' .claude/skills/issue/SKILL.md` → exactly 1 hit (L2696), context READ: a Step-5b trigger-dense excerpt-builder diff pathspec, NOT a family/SPECS entry — so no landed fix is being duplicated. Pin test present: `tests/test_issue_skill_lint_family_sync.py` (functions incl. `test_step5a_specs_include_lint_family`, `test_step10d_postgate_resync_present_and_ordered`).

## Proposed change (candidate diff sketch — refine in planning)

```
+ FAMILY_OF[":(glob)scripts/guard_*.sh"]="guard"
+ (SPECS gains `:(glob)scripts/guard_*.sh`; same entry in the Step 10d
+  inlined SPECS_10D + FAMILY_OF copy; drift-pin test
+  tests/test_issue_skill_lint_family_sync.py updated to expect the new
+  family member in BOTH copies)
```

Also update the "sync scope is specs + the spec-coupled lint/guard family — do NOT extend further into scripts/" boundary paragraph with a carve-out sentence naming `scripts/guard_*.sh` as spec-coupled guard-family implementations (they execute from hooks configured in `.claude/settings.json`), so `test_sync_scope_paragraph_names_family_boundary` stays coherent.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 5a FAMILY_OF/SPECS + Step 10d inlined SPECS_10D/FAMILY_OF copy)
- Secondary: `tests/test_issue_skill_lint_family_sync.py` (drift pin)
- Grep the workflow surface for other FAMILY_OF copies before editing (`grep -rn 'FAMILY_OF' .claude/ scripts/`) and keep the two drift-pinned copies consistent; the skill text at L9461/L9488 forbids a THIRD inlined copy — the Step 9c pre-gate references § Step 5a by name.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Family-atomicity semantics preserved: a branch-side edit on any guard script dirties the whole family (per-item branch-side-edit exclusion logic unchanged).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- sha-verify (filing-time, #1467): `ba72538cc167` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `8463880a45b7` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 21a18e40d5f4

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: (driver-computed; tag authoritative)
- origin parks: task #1860 events.jsonl 2026-07-31T15:23:12Z (fp ba72538cc167) + task #1862 events.jsonl 2026-07-31T16:46:07Z (fp 8463880a45b7) — same bug, one filing; routed-records posted on both parks name this task.

Verbatim #1862 candidate block (the fuller of the two):

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: Step 5a's guard family (FAMILY_OF) contains `.claude/hooks` + `:(glob)tests/test_guard_*.py` + `tests/test_guard_lessons_edit.py` but NOT the guard scripts the tests execute (`scripts/guard_repo_root_branch.sh`, `scripts/guard_*.sh` — invoked by the PreToolUse hooks and executed FROM the worktree tree, same execute-from-worktree rationale as the family's `scripts/workflow_lint.py` member); syncing the tests without the scripts creates a half-synced tree where main-green guard pin tests fail in every pre-fix worktree (observed live on issue-1862, 2026-07-31: 12 false-red test_1861_exitguard_* nodes after tests synced at 15:50Z while main's 3d63506a28 guard-script fix stayed unsynced).
why_workflow_gap: the family-atomic sync (#1714) exists precisely to prevent syncing coupled files apart; the guard family's membership is incomplete — scripts/guard_*.sh is spec-coupled to tests/test_guard_*.py exactly as scripts/workflow_lint.py is to tests/test_workflow_lint*.py (the lint family already includes its script).
proposed_change: add `:(glob)scripts/guard_*.sh` to the guard family (FAMILY_OF + SPECS) in BOTH Step 5a and the Step 10d inline copy (they are drift-pinned by tests/test_issue_skill_lint_family_sync.py — update the pin together), so a guard-test sync always carries the guard scripts.
confidence: high
related_task: #1862
<!-- /workflow-fix-candidate -->
