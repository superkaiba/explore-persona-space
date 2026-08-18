---
title: 'Step 5a FAMILY_OF: couple .claude/rules/LESSONS.md with workflow_lint _LESSONS_*
  constants (same-diff pair split by family-atomic sync; 8 gate failures + 2h re-run
  in #2168)'
kind: infra
tags:
- wf-fix
- step5a-family-sync
created_at: '2026-08-18T20:26:30Z'
has_clean_result: false
origin_prompt: '#2168 Step 9c gate triage (v13/v14 markers): family-atomic spec-freshness
  sync split the LESSONS.md<->_LESSONS_MAX_BYTES same-diff pair, producing a tree
  in neither main nor merge-base; 8 failures, ~2h gate re-run.'
workflow: v1
---
# Step 5a/9c spec-freshness FAMILY_OF does not couple `.claude/rules/LESSONS.md` with `scripts/workflow_lint.py`'s `_LESSONS_*` constants — a family-atomic sync can produce a tree that exists in neither main nor merge-base

kind: infra

## Goal

Close a Step 5a spec-freshness family-coupling gap: `.claude/rules/LESSONS.md` and the `_LESSONS_MAX_BYTES` / `_LESSONS_*` size-ratchet constants in `scripts/workflow_lint.py` are a same-diff COUPLED pair (the guard demands they be raised together), but the FAMILY_OF map in `.claude/skills/issue/steps/09-step-5.md` places `scripts/workflow_lint.py` in family "lint" while `.claude/rules/` files form their own singleton families. A round that legitimately OWNS the lint family (dirty — held back) while `.claude/rules/` is clean (synced) gets one half of the pair refreshed and the other withheld.

## Incident (measured, #2168 Step 9c gate #1, 2026-08-18)

The #2168 round owned family "lint" (its deliverable is a new workflow_lint check). Its Step 9c step-1a spec-freshness sync (commit 0755fb9285) synced `.claude/rules/LESSONS.md` to origin/main (10,452 bytes) while withholding `scripts/workflow_lint.py` (cap `_LESSONS_MAX_BYTES = 10205`). Consistency states:

- merge-base e7b4e7dc0a4a: LESSONS.md 10,203 B / cap 10,205 — consistent
- origin/main 3f130025c522: LESSONS.md 10,452 B / cap 10,492 — consistent (raised same-diff, as the guard demands)
- the synced worktree tip 0755fb9285: 10,452 B / cap 10,205 — INCONSISTENT, a tree existing in NEITHER main nor merge-base

Cost: 4 gate failures (tests/test_workflow_lint.py::test_check_lessons_index_passes_on_live_repo, ::test_lessons_budget_constants_sane, ::test_workflow_lint_default_exits_zero, tests/test_guard_lessons_edit.py::test_live_lessons_rewrite_allowed) + 4 more in the sibling test/script split-pair shape (Cluster C: tests/test_issue{1739,2162}*.py vs scripts/issue2162_run.py withheld by the same family-atomicity) = 8 failures and a ~2h full gate re-run. Both clusters were resolved by merging origin/main (985ff8e1bbff).

## Proposed fix (implementer to verify the right grain)

Either couple `.claude/rules/LESSONS.md` into the "lint" family (so a round owning workflow_lint.py also holds back LESSONS.md, keeping the pair mutually consistent at the branch vintage), or add a general same-diff-coupled-pair detection to the sync (a synced file and a withheld file last touched by the SAME main-side commit is the tell). The sibling test/script pair class (tests/test_issue<N>*.py vs scripts/issue<N>*.py when C2-style sweeps dirty the script) should be considered in the same pass — same root cause, second measured instance in the same gate.

## Acceptance criteria

- A round that owns family "lint" and runs the Step 5a sync ends with LESSONS.md and `_LESSONS_*` constants mutually consistent (either both at main vintage or both at branch vintage).
- A regression test pinning the coupling (the half-sync tree shape above must be unconstructible by the sync, or loudly flagged before the gate runs).
- Evidence trail: #2168 events.jsonl v13/v14 markers carry the full triage.

## Provenance

Surfaced by the #2168 Step 9c gate triage (v13/v14 markers, 2026-08-18). Related but distinct: #2260 covers the agents-prose-pin half-sync class (tests outside coupled globs vs `.claude/agents` singleton family) — same family-atomicity mechanism, different file-set fingerprint; a joint fix may close both.
