---
title: 'vectorized_mlp_skill: per-absolute-index MLP init seeding (chunk-invariant
  fits)'
kind: infra
tags: []
created_at: '2026-07-03T11:29:06Z'
has_clean_result: false
origin_prompt: '#841 fu-r1 v15 finding: fit_batched_split_mlp seeds each group''s
  MLP init in BATCH ORDER (vectorized_mlp_skill.py:809), so chunking a group list
  changes every member''s init (measured maxdiff 0.82 between 5-group and 2-group
  calls). Proposed: seed by the group''s ABSOLUTE index/key so any chunking is bit-equivalent;
  update the documented determinism contract + assert_matches_reference gates (#658/#722)
  accordingly. Out of workflow-fix scope (src/analysis) — ordinary infra change, full
  review pipeline.'
workflow: v1
---
## Goal

Make `fit_batched_split_mlp` (`src/explore_persona_space/analysis/vectorized_mlp_skill.py`) partition-invariant: seed each group's MLP init from the group's ABSOLUTE index/key rather than its batch position, so any chunking/partitioning of a group list across calls yields bit-identical per-group inits; update the documented determinism contract and the `assert_matches_reference`-style exactness gates (#658/#722 lineage) accordingly.

## Problem (surfaced by #841 fu-r1 v15)

`fit_batched_split_mlp` seeds inits as `torch.manual_seed(seed)` followed by one `_MLPMulti` init per group in BATCH order (member `g` gets draw `g`; docstring "SEEDING" paragraph). Splitting the same group list across separate calls therefore changes every member's init: measured maxdiff **0.82** between a 5-group call and 2-group calls over the same groups. The docstring documents this as "Deterministic + reproducible across runs" without flagging the partition sensitivity.

Contrast: `fit_batched_loco_mlp` / `fit_batched_loco_mlp_multihead` (on `main`) draw ONE shared n-fold init block reused by every group, so they are partition-invariant by construction, and the existing exactness gate already asserts internal chunk-invariance for them. The bug class is specific to the split variant's per-group draws.

## Proposed change

1. Derive each group's init deterministically from `(seed, group absolute identity)` — e.g. a per-group `torch.Generator` seeded from `seed` + a stable per-group key (caller-supplied absolute index, or a stable non-salted hash of `SplitMLPGroup.key`). Exact mechanism is a plan decision; the requirement is: bit-equal per-group inits under any partition (and ideally any reordering) of the same group list across calls.
2. Update the docstring SEEDING contract to state the partition-invariance guarantee.
3. Extend the exactness/self-check gate to assert partition-invariance for the split variant (e.g. one 5-group call vs 2+3-group calls, bit-identical per-group results), mirroring the existing chunk-invariance gate for the LOCO variants.
4. Unit test pinning the new contract.

## Acceptance criteria

1. `fit_batched_split_mlp(groups, ...)` and any partition of `groups` across calls produce bit-identical per-group inits (hence identical trained results at matched settings) — asserted by a new gate/test.
2. Documented determinism contract (docstring + any doc/rule text referencing it) updated.
3. Existing #658/#722 exactness gates still PASS (LOCO variants' contract unchanged).
4. `ruff` + touched-scope `pytest` PASS.

## Constraints / dependencies

- **The target function is NOT on `main`** — it exists on TWO unmerged live branches: `issue-841` (origin commit `404046fefe`; 21 commits ahead of `main`; #841 live at `followups_running`) and `issue-922` (carries a copy plus consumers `experiments/issue_922/maps922.py`, `scripts/issue922_fit_maps.py`; #922 live at `planning`). Branch-local consumers of the function: `experiments/issue_841/maps.py`, `scripts/issue841_scaling_common.py` / `_stage0.py` (issue-841), and the issue-922 files above. `main`'s copy of `vectorized_mlp_skill.py` carries only the LOCO variants. The plan MUST pick a landing strategy (wait-for-branch-merges vs port-the-function-to-main-with-fix and accept reconciliation cost at each branch's eventual merge vs another option) and must NOT write into #841's or #922's live worktrees.
- Forward-only contract change: the new seeding changes numerical outputs relative to the old batch-order inits. Committed artifacts produced under the old contract (e.g. #841's eval results) are NOT retro-invalidated; the plan states how any reproduce-check referencing old outputs is handled.

## Compute

0 GPU-h — CPU-only `src/analysis` change + tests. No pod.

## References

- `src/explore_persona_space/analysis/vectorized_mlp_skill.py` (`main`: LOCO variants + exactness gate; `issue-841` branch ~line 785: `fit_batched_split_mlp`)
- `.claude/rules/vectorize-many-cell-fits.md` (determinism / Supersede-contract lineage, #658/#722)
- Origin prompt (verbatim, from task frontmatter): "#841 fu-r1 v15 finding: fit_batched_split_mlp seeds each group's MLP init in BATCH ORDER (vectorized_mlp_skill.py:809), so chunking a group list changes every member's init (measured maxdiff 0.82 between 5-group and 2-group calls). Proposed: seed by the group's ABSOLUTE index/key so any chunking is bit-equivalent; update the documented determinism contract + assert_matches_reference gates (#658/#722) accordingly. Out of workflow-fix scope (src/analysis) — ordinary infra change, full review pipeline."
