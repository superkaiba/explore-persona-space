---
title: 'Step 10d mapped-invariant leg attributes base-identical sibling-synced files
  to the payload (the #2302 subtraction never reached it)'
kind: infra
tags:
- workflow-fix
- step10d-gate
created_at: '2026-08-20T05:10:36Z'
has_clean_result: false
parent_id: 2204
workflow: v1
---
# Step 10d's mapped-invariant leg attributes base-identical sibling-synced files to the payload — the #2302 subtraction Step 9c has never reached it

## Goal

Give the Step 10d pre-push gate's mapped-invariant test leg the same base-identity discipline `select_step9c_tests.compute_touched` and the Step 9c compare already have (#2302), so files that are byte-identical to the fetched `origin/main` tip stop counting as branch changes there. Equivalently: make the leg's oracle and its own-diff agree about what "the payload" is. The plan decides which end to fix.

## Why (incident, measured on #2204)

Two facts about the leg combine into a systematic false-positive:

1. **The oracle is the MERGE-BASE, not `origin/main`.** `scripts/step9c_baseline.py:2223` — `sha = git_merge_base(ctx.base_ref, ctx.work_root)`; carried as `oracle_base_sha` (`:3055`, "#2293: merge-base(base_ref, wt HEAD)").
2. **The leg's own-diff includes base-identical paths.** On #2204, 16 of the 19 files in the mapped set were sibling-synced scripts whose content equalled the fetched `origin/main` tip exactly. Only 3 were the branch's own work.

Step 5a's sibling arm guarantees synced paths are base-identical BY CONSTRUCTION, and `09-step-5.md` § "Base-identity invariant (#2302)" says so, then names the two consumers that act on it: `compute_touched` subtracts verified base-identical paths (reporting `base_identical_excluded`), and "the Step 9c compare derives its OWN base-identical set". The Step 10d mapped-invariant leg is not among them.

The consequence is worse than an inflated set, because of a direction asymmetry: **15 of the 17 synced files were main-NEW — absent at the merge-base entirely.** So the baseline leg could not even collect their tests, while the gated leg ran them against fork-era `src/`. Measured on #2204:

- gated: **1375 collected, 6 failed** — baseline: **1314 passed, 0 failed**
- the 61-node gap is exactly the main-NEW synced test files
- verdict: `block` on `TG_RC=1` vs `TG_BASE_RC=0`, with `lint-new` and `lint-owndiff` BOTH empty

`NEW = gated − baseline` cannot be a payload signal when the two legs are not running the same node set. Everything the sibling arm imports subtracts as payload-attributed by construction, and Verdict case 2 then routes it to case 1 ("the payload is the offender") — pointing the operator at a deliverable that was never involved.

Note the failure is silent in the direction that matters: the leg reports `block_count=6 unclassifiable_count=0`. Nothing in the verdict, the diag line, or the attribution files says "these six nodes do not exist at the oracle". Diagnosis required reading raw tracebacks and reconstructing merge-base presence per file by hand.

## Why the #2296 precedent is not already the fix

#2302 was itself motivated by sync-driven inflation (#2296: 61 invariant files → 217, wall 1:46:36, two false-blocking NEW nodes) — the same shape, the same cause, one gate stage over. That fix landed in `compute_touched` and the Step 9c compare. #2204 is the residue: the Step 10d leg computes its own file list and calls the selector in `--map-files` mode (`select_step9c_tests: map-files — 82 pairs, 59 tests`), which takes the list as given and so never reaches the subtraction.

## Acceptance

- The mapped-invariant leg excludes verified base-identical paths from the set it treats as branch changes, by the same blob-OID-equality test `compute_touched` uses against the base tip — and reports what it excluded (the `base_identical_excluded` idiom), never silently. If the right fix is instead to route the leg through `compute_touched` rather than `--map-files`, say so and do that.
- The oracle/gated **node-set asymmetry is made visible or eliminated.** A node that cannot exist at the oracle must not be reported as a payload-attributed NEW failure with no qualification. Either restrict the compare to nodes present in BOTH legs, or classify oracle-absent nodes into their own bucket (the leg already has an `unclassifiable` channel and an `ordering_suspect` precedent from #2302). State which and why.
- **Fail-closed is preserved.** This must not become a mechanism for waving through a genuine payload regression. A node that is oracle-absent AND whose failure traces to a branch-own file still blocks. The plan states explicitly how the two are told apart, and defaults to blocking when it cannot tell.
- Regression coverage: a fixture branch carrying a base-identical main-NEW sibling test whose imports are unsatisfiable at branch-era `src/`, asserting it does not produce a payload-attributed NEW node.
- Cross-check the sibling stages for the same gap while there — `verify_plan`'s and the inline payload lint gate's own-diff computations, and any other `--map-files` caller — and report which do and do not apply the subtraction. Cheap to check, and the same latent bug.

## Provenance

Surfaced by the #2204 orchestrator during its own Step 10d merge (session 9e938266, 2026-08-19/20), while diagnosing a `block` verdict whose six nodes were all sibling-sync artifacts. Filed per `.claude/rules/workflow-fix-on-bug.md`.

Distinct target and fingerprint from: the sibling task on the #2208 probe's SENSITIVITY (`--collect-only` blind to runtime skew — different file, `09-step-5.md`, and a different fix; either task alone would have prevented this incident, which is why both are filed); #2409 (per-leg lint fence sized off the idle range); #2402; #2404; #2204's own `scripts/verify_plan.py` deliverable.

Reference points: `scripts/step9c_baseline.py:2223` + `:3055` (the merge-base oracle), `:2107` (the existing pristine-main failure classification), `.claude/skills/issue/steps/09-step-5.md` § "Base-identity invariant (#2302)", `.claude/skills/issue/steps/18-step-10d.md` § Pre-push workflow-lint gate + Verdict bullet cases 1-3, #2296 (the prior inflation incident #2302 fixed one stage over), #2293 (the oracle-base pin), #2204 `events.jsonl`.
