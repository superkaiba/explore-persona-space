---
name: claude-gate-object-identity-unchecked
description: "Claude critic validates a registered gate's THRESHOLD/count but never checks the named function returns the object the gate compares, nor that the gate's inputs exist at its phase (#2329 r2)"
metadata:
  type: feedback
---

Claude methodology critic judged a registered HALT gate "not decorative" by
verifying the THRESHOLD band (≥0.99 discriminating vs bf16 noise) and donor
COUNT adequacy — but never opened the named function. The plan registered the
donor-identity assert "through the inherited `capture_answer_states`/`_slot_state`
path", yet `capture_answer_states` returns COMPLETION-SPAN MEANS (requires
completions, which don't exist at the gate's phase L1) while the compared
`vc_bank.pt per_context` states are exact CONTEXT-POSITION slot states from
`capture_bank`'s right-padded context forwards; `_slot_state` is a pure
selector. Implemented literally the gate cannot run or false-HALTs a valid
run. The plan ALSO specified the gate over "screened" donors whose screen
runs at a LATER phase (L3) — a phase-availability defect. Codex caught both;
Claude's verdict explicitly checked neither. (#2329 r2, methodology lens;
reconciled REVISE siding with Codex.)

**Why:** an affirmative-misfire gate defect (false HALT on valid run / cannot
run) is exactly the [[claude-gate-unit-vs-preregistered-verdict-logic]]
REVISE class — threshold adequacy never rescues wrong object identity.

**How to apply:** when a disputed blocker concerns a registered
gate/assert that NAMES a code path, READ the named function's return value
and compare its object (pooling, positions, required inputs) against the
artifact it is asserted equal to, AND check every gate input exists at the
gate's phase in the plan's phase graph. A verdict that argues only about the
threshold has not verified the gate.
