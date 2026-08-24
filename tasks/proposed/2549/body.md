---
title: 'workflow-fix: --check-shared-tmp-name is RED on main''s own tree (issue823_ladder_ext_gen.py:1330),
  failing the live-tree gate test fleet-wide'
kind: infra
tags: []
created_at: '2026-08-24T18:16:02Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2479 round-12 code-review verdict as a pre-existing
  main-side lint red, re-verified by the orchestrator at the main checkout'
workflow: v1
---
## Goal

Get `workflow_lint.py --check-shared-tmp-name` green on `main`'s own live tree, so `test_workflow_lint_shared_tmp.py::test_live_tree_green_under_seeded_allowlist` stops failing fleet-wide.

## Problem

At `main` HEAD the bundled no-flags check is RED on main's own tree:

```
$ uv run python scripts/workflow_lint.py --check-shared-tmp-name
workflow_lint: scripts/issue823_ladder_ext_gen.py:1330: process-shared atomic-write temp name (use explore_persona_space.atomic_io.atomic_replace; #2336)
workflow_lint: FAIL (1 error(s))
```

Verified twice, independently, at two different main HEADs during one session (`8190579ce1`, and `67c2036b` when the #2479 round-12 reviewer measured it). `scripts/issue823_ladder_ext_gen.py:1330` trips the #2336 check and is absent from the path-keyed batch-0 allowlist seed, so the live-tree gate test fails for every session that runs it, on a red that has nothing to do with the branch under review.

## Impact

`--check-shared-tmp-name` is bundled into the no-flags default run, and the no-flags run is the instrument several gates use: the `/issue` Step 9c test gate, the Step 9a-ter inline payload lint gate, and the Step 10d pre-push lint verdict gate. A red that is inherent to `main` makes every one of those gates ambiguous — each consuming session has to hand-attribute the failure to "pre-existing, not mine" before it can proceed, which is exactly the judgment call a mechanical gate exists to remove. It also trains sessions to wave past lint red, which is how a genuine payload-attributed failure gets missed.

Encountered on #2479: the round-12 reviewer had to run the check at the main checkout to separate one pre-existing main-side error from three genuinely branch-attributed ones before it could grade the round.

## Fix (pick whichever is correct for the writer)

Either migrate `scripts/issue823_ladder_ext_gen.py:1330` to `explore_persona_space.atomic_io.atomic_replace`, or — if that writer's output path is provably per-process unique — add the `# SHARED_TMP_EXEMPT: <reason>` waiver with the uniqueness argument stated. Do NOT simply add the path to the allowlist seed without deciding which of those two is true: a seed entry records "known and accepted", and if the writer really is process-shared then the seed hides a live atomicity bug rather than fixing it.

## Acceptance criteria

1. `uv run python scripts/workflow_lint.py --check-shared-tmp-name` exits 0 on a clean `main` checkout.
2. `uv run pytest tests/test_workflow_lint_shared_tmp.py` passes on `main`.
3. The chosen route is recorded: migrated, or waived with the per-process-uniqueness argument written next to the waiver.
4. If any OTHER path is also red on main by the time this runs, fix the whole set — the acceptance bar is a green live tree, not one path.

## Provenance

workflow_fix_target: scripts/issue823_ladder_ext_gen.py

Surfaced by the `code-reviewer` round-12 verdict on #2479 (`epm:code-review v12`, 2026-08-24T18:11:51Z) as an explicitly pre-existing, not-this-round finding, and independently re-verified by the #2479 orchestrator by running the check at the main checkout. Distinct from #2479's own three round-deliverable hits in `scripts/issue1345_char_capture_launch.sh`, `scripts/issue2479_p1p4_launch.sh`, and `scripts/issue2479_p5_launch.sh`, which that task fixes on its own branch.
