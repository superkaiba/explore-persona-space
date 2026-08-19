---
title: 'workflow-fix: Step 5a sibling sync puts main''s own files in the branch diff,
  disabling the #2024 ordering carve-out and false-BLOCKing Step 9c'
kind: infra
tags:
- wf-fix
created_at: '2026-08-14T21:44:57Z'
has_clean_result: false
parent_id: 2296
origin_prompt: 'Surfaced by the /issue 2296 orchestrator while adjudicating its own
  Step 9c compare BLOCK: both blocking nodes were sibling #2254 tests synced in by
  Step 5a, byte-identical to origin/main, passing alone, with no matplotlib surface
  in the payload.'
workflow: v1
---
# Step 5a sibling-file sync pulls main's own files into the branch diff, disabling the #2024 `ordering_suspect` carve-out and false-BLOCKing the Step 9c gate

## Goal

Stop the Step 5a sibling-file spec sync (#1972) from converting a PRE-EXISTING,
order-dependent test failure on `origin/main` into a blocking `new` node at the Step 9c
compare. Today the sync writes main's own sibling `scripts/issue<M>_*.py` /
`tests/test_issue<M>_*.py` files into the branch, which puts them in the branch's
three-dot diff; the compare then reads them as branch-touched
(`paired_skipped: file-in-branch-diff`) and the #2024 carve-out — which exists precisely
so *branch-untouched* ordering failures classify `ordering_suspect` instead of blocking
NEW — never fires. The gate blocks a payload that provably cannot cause the failure.

## Measured incident (#2296, 2026-08-14)

#2296's payload touches only `.claude/skills/issue/SKILL.md`, `scripts/step9c_baseline.py`,
`scripts/workflow_lint.py`, and three test files. Its Step 9c gate returned
`2 failed, 9691 passed, 14 skipped in 6396.64s (1:46:36)`, and the compare classified BOTH
failures as blocking `new`:

```
new: tests/test_issue2254_reduce.py::test_render_all_on_reduced_tree
     tests/test_issue2254_reduce.py::test_render_all_require_raises_on_skipped
stripped: []
ordering_suspect: []
paired_skipped: [{node: …test_render_all_on_reduced_tree, reason: "file-in-branch-diff"},
                 {node: …test_render_all_require_raises_on_skipped, reason: "file-in-branch-diff"}]
pristine_files_run: ['tests/test_issue2254_reduce.py']
pristine_oracle: scratch-worktree     scratch_sha: 87efce186189…
```

The failure itself is #2254's, not #2296's:

```
scripts/issue2254_figures.py:353: in fig_layer_dose_heatmap
    fig.tight_layout()
E   RuntimeError: Colorbar layout of new layout engine not compatible with old
    engine, and a colorbar has been created.  Engine not changed.
```

Four independent checks establish that the payload is innocent and the red is
pre-existing + order-dependent:

1. **Byte-identical content across all three trees.** `git hash-object` on the #2296
   worktree vs `git rev-parse` on `origin/main` and on the oracle sha, for BOTH files:
   `scripts/issue2254_figures.py` = `3ed23d34f` and
   `tests/test_issue2254_reduce.py` = `0a3b6b3bf` in all three. The code under test is
   the same in the failing gate run and the passing oracle run.
2. **The tests PASS ALONE in the very worktree whose gate failed** — `2 passed in 17.01s`
   for exactly the two blocked node ids, same payload present.
3. **The payload has no matplotlib surface at all.** `git diff origin/main...issue-2296`
   grepped for `matplotlib|pyplot|tight_layout|constrained_layout|colorbar|rcParams`
   returns nothing, so it cannot set a layout engine or create a colorbar.
4. **Neither file is in the payload.** They entered the worktree solely through the Step 5a
   sibling-file arm (sync commit `990ec0948d`, "sibling-file sync: 17 file(s)").

So a matplotlib layout-engine/colorbar global-state interaction between some earlier test in
the 217-file gate set and `#2254`'s figure builder — a condition fully present on
`origin/main` — was attributed to #2296.

## Mechanism

- The sibling arm writes main's content into the branch and commits it, so those paths enter
  `git diff <merge-base>...HEAD`.
- `select_step9c_tests.py` sizes the gate from that diff, so the sibling tests are SELECTED
  (here inflating the gate from the ~61-file invariant set to 217 files and the wall from the
  documented ~18-min median to 1:46:36).
- The compare's #2024 carve-out keys on branch-UNTOUCHEDness. A synced sibling file is
  textually branch-touched while being semantically main's own, so the carve-out is skipped
  and the ordering failure hardens into `new`.

The three sub-conditions compound: the sync SELECTS the sibling test, then makes it
INELIGIBLE for the ordering carve-out, and the larger gate set it produces makes
cross-test pollution likelier in the first place.

## Proposed fix (direction only — the plan decides)

Make "synced from main by the spec-freshness arm" a first-class provenance signal that the
compare can read, so such paths are treated as branch-UNTOUCHED for carve-out purposes while
remaining fresh on disk. Candidate shapes:

- Have the sync record the synced path set (a sidecar under `.claude/cache/`, or a
  parseable token in its own commit subject — the block already uses the subject shape
  `"sync workflow-surface specs from …"` for its dirty-family exclusion), and have
  `_paired_collection_reason` / the #2024 ordering classifier subtract that set from
  "branch-touched".
- Or widen the existing #2206/#2208 revert arm — which already reverts a synced sibling test
  that fails COLLECTION in the worktree — to cover a synced sibling test that fails at RUN
  time, using the same restore-branch-era-or-drop disposition.
- Or exclude sibling-synced paths from the selector's diff-derived gate set (they are not the
  payload under review), which also restores the gate wall to the invariant-set baseline.

Prefer whichever keeps the sync's freshness benefit; do NOT simply disable the sibling arm.

## Acceptance

1. A synced-from-main sibling file that fails order-dependently in a gate run classifies
   `ordering_suspect` (or is otherwise non-blocking), not `new`.
2. A genuine branch-authored regression in a file the branch really edited still classifies
   `new` — the carve-out must not widen into a hole.
3. The #2296 evidence above is reproduced as a regression fixture (synthetic tree: a
   sibling-synced path + an order-dependent failure ⇒ non-blocking; the same path
   branch-edited ⇒ blocking).
4. The gate-set inflation is addressed or explicitly declared out of scope with a reason.

## Provenance

Surfaced by the #2296 orchestrator while adjudicating its own Step 9c BLOCK. #2296's own
payload (moving the Step 10d mapped-invariant BASELINE leg off the shared repo root) is
unrelated to this defect; the two share only the observation that gate baselines/oracles are
sensitive to which tree they are cut from. Related: #2293 (the pristine oracle is cut from the
root's local HEAD rather than the resolved diff base), #2024/#1832 (the ordering-suspect
carve-out this bug disables), #1972 (the sibling-file sync arm), #2206/#2208 (the existing
collection-failure revert arm).
