---
title: 'Trunk red: scripts/issue1739_r2v2_run.py ungated .upload_file( fails test_no_new_ungated_upload_call_sites
  on origin/main'
kind: infra
tags:
- workflow-fix
- main-red
created_at: '2026-08-20T05:10:57Z'
has_clean_result: false
workflow: v1
---
# Trunk red: `scripts/issue1739_r2v2_run.py` has an ungated `.upload_file(`, failing a workflow invariant test on `origin/main`

<!-- workflow-fix-candidate v1 -->
urgency: main-red
failing_test: tests/test_no_ungated_upload_call_sites.py::test_no_new_ungated_upload_call_sites
wf_fix: false
<!-- /workflow-fix-candidate v1 -->

## Goal

Close the ungated HF upload call site in `scripts/issue1739_r2v2_run.py` so `tests/test_no_ungated_upload_call_sites.py::test_no_new_ungated_upload_call_sites` is green on `origin/main`. `wf_fix: false` because the offending file is experiment code, not the workflow surface — but the test it reds IS a workflow invariant, so it reds the Step 9c / Step 10d gate of every branch that maps or imports the file.

## Why (verified, not inferred)

Measured against `origin/main` at **`52fa8762f112`**, by re-running the test's own predicate over **all 2,379 `.py` files** under `src/` + `scripts/`, using origin/main's own `UPLOAD_TOKENS` / `SANCTIONED` / `EXCLUDED` / `GRANDFATHERED` sets:

```
scanned 2379 .py files under src/ + scripts/ on origin/main
ungated upload call sites: 105   NOT grandfathered (=test failures): 1
   OFFENDER: scripts/issue1739_r2v2_run.py
stale GRANDFATHERED entries (test_grandfather_list_only_shrinks): 0
```

The offender's specifics on origin/main:

- upload token present: `.upload_file(`
- `assert_upload_clean(` present: **False**
- in `SANCTIONED`: False · in `EXCLUDED`: False · in `GRANDFATHERED`: False

So the assertion fails on a pristine trunk checkout. The companion test `test_grandfather_list_only_shrinks` is clean (0 stale entries), so the list is otherwise honest — this is one genuinely new uncovered call site, not list rot.

## How it surfaced

On #2204 (a workflow-surface round), Step 5a's sibling-issue arm synced `scripts/issue1739_r2v2_run.py` from `origin/main` onto the branch. The Step 10d mapped-invariant leg then reported the node as a payload-attributed NEW failure, because that leg's baseline oracle is the **merge-base**, where the script did not yet exist — so trunk red was indistinguishable from a branch regression. (The attribution defect is filed separately and is NOT this task; this task is the underlying code bug, which is real regardless of how it was found.)

The reason it is not already common knowledge: the test is a whole-tree scan, so it only fires for a branch whose gate actually maps that test in. #2204 mapped it in via `scripts/workflow_lint.py`. Branches that touch neither will never see it, which is exactly how a trunk red survives unnoticed.

## Acceptance

- `tests/test_no_ungated_upload_call_sites.py` passes on `origin/main`.
- The fix routes the upload through the gated path (`hub._upload` / `upload_dir_sharded`) or calls `secret_scrub.assert_upload_clean(paths, what=...)` before the direct call — the two remedies the assertion message itself names. **Do NOT add the file to `GRANDFATHERED`**; that file's docstring forbids it, and `test_grandfather_list_only_shrinks` is currently at 0 stale entries, so adding an entry would be a deliberate regression of a clean invariant.
- Confirm the script's upload actually still works after the change (it writes real artifacts — check the call site's paths argument survives the routing change rather than assuming it does).
- While there: report whether any of the 104 grandfathered sites are trivially closable, but do NOT bundle that into this fix. This task is the one red node.

## Provenance

Surfaced by the #2204 orchestrator during its own Step 10d merge (session 9e938266, 2026-08-19/20), and verified independently against pristine `origin/main` before filing rather than being read off the branch's gate output. Filed per `.claude/rules/workflow-fix-on-bug.md` and the `18-step-10d.md` § "Mandatory urgent-park emission on workflow-surface pre-existing red" (#1713) grammar, reproduced in the block above.

Distinct target and fingerprint from: the sibling task on the Step 10d mapped-invariant leg's base-identity attribution; the sibling task on the #2208 probe's sensitivity to runtime skew; #2409; #2402; #2404; #2204's own `scripts/verify_plan.py` c67 deliverable.

Reference points: `tests/test_no_ungated_upload_call_sites.py` (`_direct_upload_files` at :161-171, the assertion at :174-182, `test_grandfather_list_only_shrinks` at :185+), `origin/main` `52fa8762f112`, #2204 `events.jsonl` (the full-scan verification and the six-node block diagnosis).
