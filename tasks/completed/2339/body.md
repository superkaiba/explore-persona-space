---
title: 'judge_pilot_gate: unsatisfiable-config remedy text names escapes that cannot
  satisfy the verdict-time floor'
kind: infra
tags: []
created_at: '2026-08-17T08:19:29Z'
has_clean_result: false
parent_id: 2329
origin_prompt: 'Surfaced autonomously during /issue 2329: the pilot-gate3pre ValueError''s
  remedy text (allow_subresolution_pilot / waive_parse_fail_arms) misled an implementation
  round into a fix that provably cannot pass, because both escapes bypass only the
  config-time guard while _gate_verdict''s min_effective_draws_per_arm floor is unconditional.'
workflow: v1
---
# judge_pilot_gate: the unsatisfiable-config ValueError recommends an escape that cannot satisfy the verdict floor

## The defect

`src/explore_persona_space/eval/judge_pilot.py` raises a helpful-looking
`unsatisfiable pilot configuration` ValueError whose remedy text offers two escapes:

    ... waive it (waive_parse_fail_arms, with a recorded reason) or accept a
    sub-resolution pilot (allow_subresolution_pilot=True)

Neither escape works for a caller whose `min_effective_draws_per_arm` sits at or
above the resolution floor. `allow_subresolution_pilot=True` bypasses ONLY the
config-time satisfiability guard (`_config_satisfiability_guard`, ~390-395,
downgrading the raise to a recorded report warning). The verdict-time floor is
separate and unconditional: `_gate_verdict` (~239-252) appends a FAILURE for any arm
whose effective draws fall below the caller's `min_effective_draws_per_arm`, with NO
exemption for sub-resolution-accepted arms and none for waived arms. So the escape
converts a pre-spend ValueError into a POST-spend FAIL — it spends judge draws and
then fails anyway.

`waive_parse_fail_arms` fails the same verdict floor, so the second suggestion is
equally ineffective for this caller class.

## Why this is worth fixing rather than just knowing

It actively misled a caller. On #2329 the ValueError's remedy text was read at face
value and an implementation round was briefed to thread
`allow_subresolution_pilot=True`. That round correctly STOPPED and proved (zero-API
pure-function probes) the fix could not pass, but only after a full round of work.
The library's own test pins the misleading shape without exposing it:
`tests/test_judge_pilot_gate.py::test_allow_subresolution_pilot_downgrades_to_report_warning`
(line ~522) asserts `passed is True` for a 5-item arm — but ONLY because it leaves
`min_effective_draws_per_arm` at the library DEFAULT of 10, which the 5-item... i.e.
the arm meets the caller floor. Nothing in the suite covers "escape requested AND
caller floor above the arm size", which is the case that actually arises.

This is shared library code reached by every judge-gated task, so the next caller
hits the same wall.

## Proposed fix (any one of these; first is cheapest)

1. **Make the remedy text conditional and honest.** When the caller's
   `min_effective_draws_per_arm` is at or above the computed resolution floor, say so
   and name the real remedies: lower the caller floor (per-family / feasibility-aware),
   or enlarge the arm's item pool. Mentioning an escape that provably cannot help this
   caller is worse than mentioning none.
2. **Make the verdict floor feasibility-aware when the escape is requested** — i.e.
   have `allow_subresolution_pilot=True` also relax the verdict-time floor to the arm's
   realized size, recording the achieved resolution. This is what a caller reading the
   remedy text reasonably expects. It DOES weaken the hollow-evidence floor, so it must
   record the realized resolution per arm; decide deliberately whether that trade is
   acceptable library-wide, since the floor exists to stop a gate PASSing on hollow
   evidence.
3. **At minimum, add the missing test**: escape requested AND caller floor above arm
   size ⇒ assert the verdict still FAILs. That pins the real semantics so the next
   reader is not misled by the line-522 test.

Whichever is chosen, the truncation half (`_truncation_failure`, ~188-221) is
unconditional and never waivable and must stay exactly that.

## Also: gotchas.md entry

The prior round tagged this `gotcha_candidate: yes` / `generalizes: yes`. Worth a
`.claude/rules/gotchas.md` line, because the trap is invisible from the call site:

  `judge_pilot_gate`'s `allow_subresolution_pilot=True` bypasses ONLY the config-time
  satisfiability refusal; the caller's `min_effective_draws_per_arm` still hard-FAILs
  at verdict time in `_gate_verdict`, with no exemption for bypassed or waived arms.
  At any call site whose floor is at or above the resolution floor, an item-limited arm
  can never PASS. `waive_parse_fail_arms` fails the same floor.

## Provenance

Surfaced on #2329 (`--phase pilot-gate3pre`), where the query-rubric family's anchor
arm is item-limited at 30 units against a call-site floor of 51. #2329 is proceeding
with a per-family feasibility-aware floor at its own call site (recorded there with its
residual: that family's parse-fail check resolves at 3.3% rather than 2%); this task is
the library-side follow-up so the next caller is not misled the same way.
