---
title: 'workflow-fix: repair main-red pin test_mlp_chunk_size_changes_fit_batch_order_init'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4cad88ec1c3b
created_at: '2026-08-06T18:34:38Z'
has_clean_result: false
origin_prompt: '#1713 urgent-park route from /issue 2091 Step 10d: the pre-push lint
  gate''s pass rested on pre-existing main red tests/test_issue841_stage0_chunk_and_capture_skip.py::test_mlp_chunk_size_changes_fit_batch_order_init
  (baseline leg 1 failed / 122 passed on the pristine origin/main tree, 2026-08-06).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #2091 (emitting agent: orchestrator, `/issue 2091` Step 10d). This is the #1713 MANDATORY urgent-park route: task #2091's Step 10d pre-push lint gate returned **pass**, and that pass rests on a PRE-EXISTING red whose file is on the workflow surface (`tests/`). Every intervening session's Step 9c / Step 10d gate must re-classify this same red until it is fixed, which is exactly the fleet-wide per-hour cost the urgent route exists to stop.

## Goal

Repair the main-red pin `test_mlp_chunk_size_changes_fit_batch_order_init` so the Step 9c and Step 10d gates stop re-classifying it fleet-wide.

## Workflow gap

- **Bug observed:** `tests/test_issue841_stage0_chunk_and_capture_skip.py::test_mlp_chunk_size_changes_fit_batch_order_init` fails on a pristine `origin/main` tree. The assertion is `assert diffs, "expected batch-order-init to change the fit across chunk sizes"` at `tests/test_issue841_stage0_chunk_and_capture_skip.py:86`, failing with `assert []` — i.e. the test computed an EMPTY diff set where it expected at least one difference. The pin asserts that MLP chunk size changes batch-order initialization and therefore perturbs the fit; on current main it no longer does.
- **Why it is a workflow gap:** the file is under `tests/`, which the #1713 urgent-park duty names as workflow surface. A red pin on main is not a private defect — it is charged to every session whose Step 9c or Step 10d gate maps this file into its selection. Each such session must run the baseline leg, observe the red on both sides, and re-derive "pre-existing, not mine" before it can proceed. #2091's own Step 10d gate did exactly that this run.
- **Confidence (emitter):** medium. The failure and its main-red status are certain (fresh, mechanically produced evidence below). What is NOT established is the correct fix direction: whether the production behaviour regressed (chunk size stopped affecting batch-order init, and the pin is correctly catching it) or the pin's premise went stale (an intentional change made chunk size fit-invariant, and the pin should be retired or re-scoped). The spawned session's planner must decide that with the file open — do NOT assume the pin is simply wrong.
- verified-at-filing: fresh pytest baseline leg executed against the payload-free `origin/main` tree at the repo root by task #2091's Step 10d gate on 2026-08-06: `1 failed, 122 passed in 124.06s`, with `FAILED tests/test_issue841_stage0_chunk_and_capture_skip.py::test_mlp_chunk_size_changes_fit_batch_order_init`. The gated (payload-bearing) leg produced a byte-identical failing-node set, so the baseline subtraction attributed it to main, not to #2091's diff — the branch touches none of `tests/test_issue841_*` or its subject module. Dedup probes at filing time: `grep -rl -- 'failing_test: <node>' tasks/*/*/events.jsonl .claude/cache/workflow-fix-events.jsonl` → 0 hits; `grep -rl 'test_mlp_chunk_size_changes_fit_batch_order_init' tasks/{proposed,planning,running}/*/body.md` → 0 hits. No existing routable candidate and no open infra task covers this node.

## Proposed change (candidate diff sketch — refine in planning)

```
# Decide the direction FIRST, then do exactly one of these:
#
# (A) production regression — chunk size no longer perturbs batch-order init:
#     fix the subject module so chunking again affects fit batch order;
#     the pin then goes green unmodified.
#
# (B) stale premise — chunk-size fit-invariance is now intentional:
#     retire or re-scope the pin, recording WHY the invariant changed and
#     which commit made it so, so the pin is not simply deleted to get green.
#
# tests/test_issue841_stage0_chunk_and_capture_skip.py:86
# -     assert diffs, "expected batch-order-init to change the fit across chunk sizes"
# +     <per the direction chosen above>
```

## Scope / surfaces

- Primary target: `tests/test_issue841_stage0_chunk_and_capture_skip.py`
- Secondary (direction A): whichever module owns MLP chunking / fit batch-order init that the pin exercises — resolve it from the test's imports at plan time; do not guess it here.
- Grep the workflow surface for sibling pins on the same invariant before editing (`grep -rn 'batch_order_init\|chunk_size' tests/ scripts/ src/`) and list every hit in the plan.

## Constraints / invariants

- Do NOT delete or `xfail` the pin purely to turn the gate green. Direction (B) is legitimate only with a recorded reason and the commit that changed the invariant.
- `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.
- Step 9c test-verdict must show the node GREEN on the branch, and the compare must show it stripped-or-absent rather than merely re-classified.

## Provenance

- workflow_fix_target: tests/test_issue841_stage0_chunk_and_capture_skip.py
- fingerprint: 4cad88ec1c3b

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_issue841_stage0_chunk_and_capture_skip.py
bug_observed: tests/test_issue841_stage0_chunk_and_capture_skip.py::test_mlp_chunk_size_changes_fit_batch_order_init fails on a pristine origin/main tree with `AssertionError: expected batch-order-init to change the fit across chunk sizes` / `assert []` at line 86.
why_workflow_gap: the file is on the workflow surface (tests/), so the red is charged to every session whose Step 9c or Step 10d gate maps it in — each must run the baseline leg and re-derive "pre-existing, not mine" before proceeding.
proposed_change: Repair the main-red pin test_mlp_chunk_size_changes_fit_batch_order_init so the Step 9c/10d gates stop re-classifying it fleet-wide — either fix the production regression that made chunk size stop perturbing batch-order init, or retire/re-scope the pin with a recorded reason if that invariance is now intentional.
diff_sketch: |
  tests/test_issue841_stage0_chunk_and_capture_skip.py:86
  -     assert diffs, "expected batch-order-init to change the fit across chunk sizes"
  +     <direction A: fix the subject module so the pin passes unmodified>
  +     <direction B: retire/re-scope the pin, recording why the invariant changed>
confidence: medium
related_task: #2091
urgency: main-red
failing_test: tests/test_issue841_stage0_chunk_and_capture_skip.py::test_mlp_chunk_size_changes_fit_batch_order_init
wf_fix: false
<!-- /workflow-fix-candidate -->
