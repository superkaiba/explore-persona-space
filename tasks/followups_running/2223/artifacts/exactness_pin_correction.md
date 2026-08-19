CORRECTION to the exactness pin (`artifacts/exactness_pin.md` §B item 7 area and the
body's "Firing-fraction gate" paragraph) — one claim I asserted was FALSIFIED by the
artifacts, and the corrected reading changes what Phase B should test.

## What I got wrong

I wrote that #2203's low cap firing was suspected to come from "a threshold calibrated
on all-token statistics and applied at the context vector." **That is false.** #2203
calibrated tau PER POSITION: `phase1_band_tau.json: tau_by_position` has four separate
keys — `prefix-end`, `context-end`, `all-prompt`, `all-tokens` — each with its own
per-layer tau, and the values differ substantially (band layers 18-25, `context-end` =
[5.83, -3.47, -4.33, -15.98, -20.9, -21.59, -21.28, -15.31] vs `all-tokens` = [8.16,
4.58, 4.13, -6.56, -10.63, -8.97, -10.38, -9.23]). The calibration was already
position-native. Disregard the "must not be repeated" instruction attached to that
claim; the per-cell tau column in the Phase B grid remains correct and useful for the
context-NATIVE-AXIS cells (A2b), but it is not a diagnosis of #2203.

## The corrected reading — and it is more important than the original

**The Assistant Axis itself is causally VALIDATED on the in-house 7B.**
`phase0_native_validation.json: steering_sanity` — at layer 14, alpha 64.11, judged:
mean role expression **7.5 with the axis ADDED** vs **89.33 with the axis SUBTRACTED**,
`directional_ok = true`. Adding the axis nearly abolishes role-play; removing it nearly
guarantees it. The direction is real and it controls the construct.

**So the null in #2203 is not "the axis does not work." It is "capping does not do
much."** Those are different failures with different consequences:

- **Steering** adds a vector at every token at alpha ~64 — a large, unconditional
  perturbation. Massive measured effect.
- **Capping** floors the projection at the 25th percentile. By construction this is a
  TAIL CLAMP: it touches only samples already below the 25th percentile and leaves
  everything else untouched. Measured realized firing 10.5% (ctx) / 9.1% (all-token)
  is therefore roughly what this operation SHOULD do on a distribution close to its
  calibration set — **not evidence of a bug.**

**Consequence for the 15% firing-fraction gate: the gate is probably mis-specified.**
A 25th-percentile floor can never fire on much more than ~25% of slots and will
typically fire well below that. A 15% floor will therefore flag correctly-behaving
paper-faithful cap arms as "calibration-limited." Do NOT inherit the 15% number
uncritically. Either (a) re-derive the floor from the tau percentile actually used
(expected firing ~= the percentile, so gate on realized-vs-expected agreement rather
than an absolute 15%), or (b) drop the absolute floor and report realized firing
alongside expected firing per cell. Report the reasoning either way.

## Design consequence — add a STEERING arm to Phase B

The paper uses steering to VALIDATE the axis (§"Causal effects of the Assistant Axis")
and capping to DEFEND. On our 7B the validation op works and the defense op is inert.
If Phase B carries only capping and axis-replacement at the context vector, the most
likely outcome is another inert result that says nothing about localization — the same
dead end as #2203, now at multi-turn cost.

Therefore add, as a first-class Phase B cell:

- **A4 — steer at the context vector ONLY**: add `alpha * axis` at the context-vector
  position (paper's convention: scale alpha against the average post-MLP residual norm
  at that layer, measured on lmsys-chat-1m). This is the strong-intervention version of
  the localization question, and it is the op with a demonstrated effect on this model.
- Pair it with **A5 — steer at all tokens** as the anchor, matching the paper's own
  steering setup.

Rationale to record in the plan: capping vs steering at the SAME position separates
"the context vector is the wrong PLACE to intervene" from "the cap is too weak an
OPERATION anywhere." #2203 cannot distinguish those, and without the steering arm
Phase B still cannot.

## Unchanged

Everything in the exactness pin's §A (PINNED) stands: axis definition + published-vector
reuse, post-MLP residual, mean over response tokens, **middle layer (32 of 64)**,
protocol, verbatim prompts, orientation. §C (cannot-match) stands. The thinking-mode
both-arms requirement (§B item 7) stands and is unaffected by this correction.
