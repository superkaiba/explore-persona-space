---
title: 'fix order-dependent test-state leak: test_issue1335_ladder poisons test_issue825_mlp_batched_parity
  in same-session runs'
kind: infra
tags: []
created_at: '2026-07-31T02:36:50Z'
has_clean_result: false
origin_prompt: 'Implementer r2 of #1887 surfaced (epm:results v2): order-dependent
  test-state leak on main — tests/test_issue1335_ladder.py poisons tests/test_issue825_mlp_batched_parity.py
  (pca_active/pca_skipped) same-session; fails at main checkout, passes in isolation.'
workflow: v1
---
## Overview / Motivation

Auto-filed from a concrete prose follow-up surfaced by the #1887 implementer round 2 (epm:results v2 on #1887): an order-dependent test-state leak on main — `tests/test_issue1335_ladder.py` poisons `tests/test_issue825_mlp_batched_parity.py` when the two run in the same pytest session. Order-dependent red pollutes every gate run whose selection includes both files (Step 9c touched-scope gates, the Step 10d TG legs), forcing repeated manual pre-existing-red classification.

## Goal

Make `tests/test_issue825_mlp_batched_parity.py` pass regardless of whether `tests/test_issue1335_ladder.py` ran earlier in the same session (isolate or restore the shared fit-core state), without weakening either test's assertions.

## Workflow gap / bug

- **Bug observed:** `uv run pytest tests/test_issue1335_ladder.py tests/test_issue825_mlp_batched_parity.py -q` → `test_batched_matches_serial_reference_pca_active` + `test_batched_matches_serial_reference_pca_skipped` FAIL (2 failed, 27 passed); both parity nodes pass in isolation.
- verified-at-filing: repro run this session (2026-07-31) at the repo root on main `065183b40a` — `timeout 420s uv run pytest tests/test_issue1335_ladder.py tests/test_issue825_mlp_batched_parity.py -q -p no:cacheprovider` → 2 failed / 27 passed, rc=1 (per-node ids above); isolation-pass evidence: #1887 rounds 1-2 regression runs (epm:results v1/v2 on #1887) ran the parity file green in other orderings.
- **Mechanism:** unverified hypothesis — verify at plan time: module-global / cached state in the shared fit core (`scripts/issue825_fit_cells.py` or a sibling module the ladder test mutates — e.g. selection/cap globals, cached eigh/PCA state) is left dirty by the ladder test and changes the parity comparison's serial-vs-batched reference. The exact leaked global was NOT isolated at filing time.

## Proposed change (refine in planning)

Snapshot-and-restore the fit-core module globals (and clear any fit caches) around the tests that mutate them — a pytest fixture in `tests/test_issue1335_ladder.py` (or a shared conftest helper), mirroring the finally-restored-globals pattern the #1887 caller pins use. Alternatively reset state in the parity test's setup. Keep assertions unchanged.

## Scope / surfaces

- Primary targets: `tests/test_issue1335_ladder.py`, `tests/test_issue825_mlp_batched_parity.py` (+ a conftest fixture if shared).
- NOTE: the #1887 branch (in flight at filing) flips fit-core defaults (`LAMBDA_SELECTION`/`GCV_DOF_CAP`); the fix should be rebased on / verified against main after #1887 merges.

## Constraints / invariants

- Test-only change preferred; no behavior change to the fit cores themselves.
- Both files green in BOTH orders and in isolation; `uv run pytest tests/test_issue1335_ladder.py tests/test_issue825_mlp_batched_parity.py` and the reverse order both pass.

## Provenance

Surfaced by implementer round 2 of #1887 (epm:results v2, task #1887 events.jsonl): "Order-dependent test-state leak on main: tests/test_issue1335_ladder.py poisons tests/test_issue825_mlp_batched_parity.py (pca_active/pca_skipped) when run in the same session — fails identically at the main checkout, passes in isolation."
