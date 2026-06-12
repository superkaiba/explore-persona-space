---
title: 'Harden the i472 trajectory storage-contract gate: finiteness/type-check the
  6 raw-logit leaf fields + AST gate-before-write ordering pin'
kind: infra
tags: []
created_at: '2026-06-12T21:01:16Z'
has_clean_result: false
parent_id: 576
origin_prompt: /issue 576 — round-2 reconciler deferred concern trajectory-gate-presence-only-validation
  (standing recommendations 1-2)
---
Follow-up capture from #576 round-2 reconciler (epm:review-reconcile v2, deferred concern `trajectory-gate-presence-only-validation`).

The storage-contract gate added in #576 r2 (`assert_trajectory_slot_records_meet_storage_contract`, `src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_trajectory.py`) is presence-only (`leaf.get(k) is None`): a NaN raw-logit value (corrupted adapter/forward pass) survives `float(tensor.item())` and persists in the final canonical artifact. Non-numeric strings are unreachable (single producer emits `float(...item())`), so this is the NaN half only — adjudicated real-but-non-blocking, low recoverable cost (adapters persist on HF).

Asks (reconciler standing recommendations 1-2, ~3 lines + 1 test):
1. Extend the gate to require each of the 6 `TRAJECTORY_LOGIT_LEAF_KEYS` be a finite, non-bool float (mirror `_is_finite_number` in `eval/marker_logprob.py`). Alternative one-line backstop noted: `allow_nan=False` on the final `json.dumps`.
2. Extend the AST pin in `tests/test_marker_slot_contract.py` to assert the gate call precedes `out_path.write_text` inside `run_trajectory_eval` (currently pins call count only).
3. Optional minors carried from review: `assert checkpoints` against degenerate empty input.

Constraint: `validate_marker_slot_record` semantics stay untouched (live consumers #542/#597).
