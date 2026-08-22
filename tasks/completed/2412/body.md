---
title: Step 5a sibling-sync satisfiability probe is --collect-only, so runtime API
  skew and function-body imports pass it
kind: infra
tags:
- workflow-fix
- step5a-sibling-sync
created_at: '2026-08-20T05:10:02Z'
has_clean_result: false
parent_id: 2204
workflow: v1
---
# The #2208 sibling-sync satisfiability probe is `--collect-only`, so runtime API skew and function-body imports pass it

## Goal

Make Step 5a's import-satisfiability probe (`.claude/skills/issue/steps/09-step-5.md`, the #2208 block) sensitive to the failure mode it exists to catch. Today it runs `pytest --collect-only -q <test>`, which by construction cannot see any skew that manifests after collection. Decide in the plan whether to (a) deepen the probe, (b) close the source of the skew, or (c) both.

## Why (incident, measured on #2204)

Step 5a's sibling-issue arm (#1972) synced **17 issue-1739 files** from origin/main onto the `issue-2204` branch. **15 are main-NEW** (absent at the merge-base). The arm's globs are `scripts/issue<M>_*.py`, `scripts/issue<M>_*.sh`, `tests/test_issue<M>_*.py` — they do NOT cover `src/`, so the co-evolved `src/explore_persona_space/experiments/issue_1739/{fits,arms}.py` stayed at fork-era content (the branch forked 423 commits back). Main-era tests then ran against fork-era src.

The #2208 probe passed all three synced test files. The Step 10d mapped-invariant leg then failed **5 nodes**, every one of them post-collection:

| Failure | Why `--collect-only` cannot see it |
|---|---|
| `TypeError: ridge_layer_batched_auto() got an unexpected keyword argument 'train_rows'` | signature skew, resolved at call time |
| `AttributeError: module ...issue_1739.fits has no attribute 'capture_selected_lambdas'` | attribute lookup inside the test body |
| `ValueError: cannot reshape array of size 240 into shape (40,40)` (`arms.bootstrap_rhos`) | behavioural skew — the module imports fine |
| `ImportError: cannot import name '_LayerRowGather' from ...issue_1739.fits` | the import statement is INSIDE the test function, so collection succeeds |

That last row is the sharpest: it is exactly the ImportError class #2208 was written for, and it still slipped through, purely because the import sits in a function body rather than at module scope. The project's own idiom of importing inside test functions (to keep collection cheap) makes this the common case, not the exotic one.

Cost on #2204: the `block` verdict consumed a full gate cycle (~40 min for the gated mapped leg alone, on top of a ~1h37m contention-timeout cycle before it), and the diagnosis burned the round's single sanctioned re-run budget down to its last unit. The remedy was to execute #2208's own pair-atomic revert BY HAND, which is the tell that the automation had the right response and the wrong trigger.

## Acceptance

- The probe detects post-collection skew for synced sibling test files, or the plan states explicitly why it cannot and closes the gap another way. Candidate approaches to weigh (the plan picks, with reasoning — do not assume this list is exhaustive or correctly ordered):
  1. **Run the test, do not just collect it.** `pytest -q <test>` on the synced file, fenced. Strongest signal, highest cost; the plan must size the fence off the LOADED range and say what a timeout means (#2409 is the sibling task on fence sizing — a timeout must not silently read as PASS).
  2. **Closure-complete the sync.** Extend the arm to the `src/` modules a synced test actually imports (resolve them from the test's import graph), so the island is internally consistent instead of half-synced. Note this widens the branch diff and needs its own base-identity story.
  3. **Refuse the sync when it cannot be made consistent.** If a synced main-NEW test's import graph reaches a `src/` path whose branch content differs from origin/main, revert the pair immediately — no probe run at all. Cheap, static, and strictly fail-safe in #2208's declared direction.
- Whatever is chosen preserves #2208's stated fail-safe direction: **status-quo staleness (the pre-#1972 world) over an unreadable gate red.** A probe that cannot decide must revert, never keep.
- Pair-atomicity is preserved: reverting a test without its synced scripts (or vice versa) is the #1824/#1860 half-sync class in reverse.
- The **manual** remedy is documented as a first-class recovery in `09-step-5.md`, with the mechanism that makes it stick: committing the revert with a subject that OMITS the arm's anchor phrase (`sync workflow-surface specs from`) makes the arm's own branch-side-commit guard (`09-step-5.md:352-354`) treat those paths as a deliberate branch edit and skip them on every later round. #2204 verified this end to end — the next Step 5a run reported `sibling-file sync: 0 file(s)` while still syncing 31 legitimate spec files. Right now an operator has to derive that from reading the arm's source.
- Regression coverage: a fixture reproducing the function-body-import shape (a synced main-NEW test whose in-function import targets a symbol absent from the branch's `src/`), asserting the arm reverts the pair. Reshaping the fixture is fine; it must stay structurally faithful to the real shape rather than collapsing to a module-scope import, which the current probe already catches.

## Provenance

Surfaced by the #2204 orchestrator during its own Step 10d merge (session 9e938266, 2026-08-19/20). Filed per `.claude/rules/workflow-fix-on-bug.md` — a gap in the workflow surface itself (`.claude/skills/issue/steps/09-step-5.md`).

Distinct target and fingerprint from: the sibling task on the Step 10d mapped-invariant leg's ATTRIBUTION model (base-identical / main-NEW synced siblings counting as payload against a merge-base oracle — different file, `18-step-10d.md` / `scripts/step9c_baseline.py`); #2409 (the 900s per-leg lint fence sized off the idle range); #2402 (`guard_skill_doc_headroom.sh` raise-time validation); #2404 (c67 negation scoping); #2204 itself (`scripts/verify_plan.py` c67, the round's actual deliverable, never implicated in any of this).

Reference points: `.claude/skills/issue/steps/09-step-5.md` (the #1972 arm at ~320-416, the #2208 probe at ~365-410, the branch-side-commit guard at 352-354), #2206/#2208 (the collection-ImportError incident and its probe), #1972 (the sibling arm), #1824/#1860 (the half-sync class), #2204 `events.jsonl` (the `[diag] verdict inputs` line, the six blocked nodes with full tracebacks, and the revert commit).
