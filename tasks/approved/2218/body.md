---
title: 'Sparse-worktree gates false-block on external/-rooted test fixtures: register
  external/jacobian-lens (+sweep) in the sparse-cone registry'
kind: infra
tags: []
created_at: '2026-08-10T10:09:31Z'
has_clean_result: false
workflow: v1
---
---
kind: infra
workflow: v1
---

# Sparse-worktree gate false-blocks: test_issue1776_* reads external/jacobian-lens, absent from tests/sparse_cones.txt

## Goal

Make sparse issue-worktrees (the `new_worktree.sh` default) carry every tree cone the test suite hard-reads, so worktree-side gates (Step 10d TG legs, Step 9c) stop false-blocking on FileNotFoundError for fixtures that exist on the full checkout. Concretely: extend the sparse-cone registry (`tests/sparse_cones.txt` + its `new_worktree.sh` consumer) to cover `external/`-rooted test dependencies — today `tests/test_issue1776_p3p4.py` + `tests/test_issue1776_patch.py` read `external/jacobian-lens/VENDOR_INFO.txt` (10 nodes FileNotFoundError in any sparse worktree), and the registry has zero `external/` rows (it may be eval_results-only by design — if so, either widen the registry format or teach new_worktree.sh a second registry).

## Evidence (#2059 Step 10d lint gate, 2026-08-10)

- The TG mapped-invariant leg blocked #2059's merge: 10 `test_issue1776_*` nodes NEW-vs-baseline (gated leg in the sparse worktree: `FileNotFoundError: .../issue-2059/external/jacobian-lens/VENDOR_INFO.txt`; baseline leg on the full repo root: green).
- `git -C <WT> sparse-checkout add external/jacobian-lens` → all 20 nodes pass in the same worktree (verified 2026-08-10).
- Same nodes were mechanically stripped `via: pristine-scratch` by #2059's Step 9c compare — the scratch pristine tree lacks the cone too, so the class also pollutes pristine-oracle classifications fleet-wide.
- `grep -n external tests/sparse_cones.txt` → no rows.

## Acceptance sketch

1. `external/jacobian-lens` (and a sweep for any OTHER `external/` path hard-read by tests/: `grep -rn "external/" tests/ --include='*.py' | grep -v '#'`) registered so `new_worktree.sh` pre-adds it.
2. A fresh sparse worktree runs `tests/test_issue1776_p3p4.py tests/test_issue1776_patch.py` green.
3. Existing #671-class eval_results cone behavior unchanged; `workflow_lint.py` no-flags green.

## Provenance

workflow_fix_target: scripts/new_worktree.sh + tests/sparse_cones.txt
Auto-filed by the #2059 /issue session per .claude/rules/workflow-fix-on-bug.md (gap in a workflow-helper script's fixture-cone registry; false-blocked #2059's Step 10d gate). Candidate fingerprint: sparse-cones-external-jacobian-lens.
