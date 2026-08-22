---
name: rederive-frozen-donor-plans
description: Named frozen-donor / seeded-plan ids in a plan are cheaply re-derivable at critique time — execute the deterministic generator on the pinned artifact instead of trusting the planner's list; also verify pe-viability (no_prefix) for any both-slots gate
metadata:
  type: feedback
---

When a plan NAMES specific frozen donor / seeded-plan context ids (e.g. "the first three
DISTINCT primary donor B-contexts from `LB.crosstype_donor_plan(bank["pairs"], seed=S)`"),
re-derive them yourself: download the pinned input artifact (bank.json is ~MB-scale),
run the deterministic generator, and compare — ~2 tool calls, ~1 min. Also check each named
id's slot-viability against the artifact's own exclusion fields (`no_prefix_context_ids`)
when the gate compares BOTH slots, and use structural arguments for deferred data-level
membership (e.g. `capture_bank` asserts `len(records) == len(contexts)`, so a one-commit
bank.json+vc_bank.pt pair implies per_context covers every bank context).

**Why:** The round-2 #2329 overrule (reconciler, 2026-08-19) established that a threshold
check is not a closure check — I verified a ≥0.99 band but never opened the named function
(`capture_answer_states` returns completion-span MEANS and needs a completions input; the
staged bank held exact single-position states from `capture_bank`; the gate as registered
would have FALSE-HALTED a valid run). In round 3 the re-derivation confirmed the planner's
three ids exactly — but only the execution made that a verification rather than trust.

**How to apply:** Any HALT-bearing identity/parity gate over named artifacts: (1) open the
function the plan names and confirm the OBJECT it returns is the object the comparison
needs; (2) confirm the gate's inputs EXIST at the phase where it runs (screen outputs built
at L3 do not exist at L1); (3) re-execute any deterministic derivation the plan claims;
(4) for single-position bf16 cosine bars, sanity-check headroom against the #779 measured
worst (0.998770 last-layer padded-batch on Qwen2.5) — span-mean calibrations don't transfer.
Related: [[patch-bank-design-traps]].
