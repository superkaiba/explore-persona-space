---
name: judge-pilot-gate-floor-semantics
description: Library semantics of eval/judge_pilot.py gates that recur in per-issue judge-script reviews (scalar floor, bypass scope, unwaivable truncation, parse-fail denominator)
metadata:
  type: project
---

Verified library facts for reviewing `judge_pilot_gate` call sites (checked against
`src/explore_persona_space/eval/judge_pilot.py` on 2026-08-17, task #2329 r9 review):

- `min_effective_draws_per_arm` is ONE SCALAR compared against EVERY arm in
  `_gate_verdict` (~:239-241, `n_effective = n_draws - n_transport_lost`). A caller-DERIVED
  floor (min-over-arms / min-over-planned) under-enforces every LARGER arm in a multi-arm
  family. Derived-floor call sites seen: `issue2329_judge._family_min_effective_floor`
  (min(51, min arm items × n_draws)), `issue2221_trait_eval.py:587` + `issue2221_band.py:351`
  (`max(1, min(10, min_planned))`). Review check: is the family structurally single-arm
  (`_pilot_arm`-style grouping), and does anything guard the multi-arm widening?
- `allow_subresolution_pilot=True` bypasses ONLY the config-time `_config_satisfiability_guard`
  (its sole read, ~:390); inert when no arm is budget/item-limited (empty `bypassed_arms`,
  no warning). It never reaches `_gate_verdict` — the verdict floor stays unconditional.
- Rule-26(a) truncation (`_truncation_failure`, ~:188) is applied FIRST and unconditionally in
  `_gate_verdict`; no waiver/bypass parameter exists. `waive_parse_fail_arms` waives the
  parse-fail clause only.
- The guard's `required = max(caller_floor, floor(1/threshold)+1 = 51 at 2%)` for UNWAIVED
  arms — so lowering the caller floor below 51 does NOT avoid the config-time refusal; the
  flag (or a waiver) is still needed for an item-limited arm.
- Parse-fail rate denominator is `n_answered = n_draws - n_transport_lost - n_api_refusal`
  (~:580-601), NOT `n_draws - n_transport_lost`. Any "resolution = 1/n" report field computed
  over effective (not answered) draws overstates fineness on api-refusal-bearing waves
  (routine 30%+ on harm-class corpora, llm-judging rule 28). The library's own reference is
  `_runtime_shrink_warnings` (~:422).

**Why:** three per-issue scripts already derive floors against this scalar API; the failure
mode (silent under-enforcement of an instrument gate) is exactly the silent-thin-PASS class.
**How to apply:** on any diff touching a `judge_pilot_gate` call or deriving its floor, verify
against these semantics before trusting implementer claims; re-read the library spans if the
file has changed since 2026-08-17.
