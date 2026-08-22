---
name: one-arm-port-capture-parity
description: Claude APPROVEs plans whose new-model arm runs a semantic PORT of the generation/capture driver validated only by internal-consistency smoke; a fits-path parity anchor does NOT cover the capture path. Codex REVISE (demand a fixed-rollout capture-parity gate vs the parent driver) is right.
metadata:
  type: feedback
---

**Rule:** when a plan's manipulated arm is measured through a PORT of an
existing driver (standalone rewrite, fresh venv, imports dropped) and the
plan's validation is (a) internal-consistency smoke gates (template pin,
shape asserts, batched-vs-per-row parity, seam prefix property) plus (b) a
port-parity anchor that exercises a DIFFERENT path (the fits/analysis port
against banked parent outputs), side with the REVISE: the CAPTURE/measure
path of the ported driver is never compared to the parent instrument, and a
systematic one-arm capture error (wrong v_C token index at a new template
seam, wrong hidden_states layer indexing under a new transformers major,
wrong v_A span) produces valid-looking tensors, plausible R², normal nulls/
floors/ceilings (all computed from the same buggy captures) — the error IS
the headline contrast and is structurally invisible in every persisted
output (the #722 no-column shape, not the #553 pure-re-reduction shape).
"Post-hoc runnable" does not rescue it: nothing in the pipeline ever
triggers the parity run, so the wrong Δ ships green.

**Why:** #2330 r1 (methodology). Claude APPROVEd with a thorough (a)-(m)
walk but never asked whether the ported 9B generate+capture driver was
validated against the parent; Codex's MF1 demanded a fixed-rollout parity
gate (run the port's capture phase on the SUPPORTED model — Qwen2.5-7B —
teacher-forcing banked parent completions; compare token boundaries +
v_C/v_A vs the banked parent captures) and was upheld. The fix was cheap
and feasible: the fresh venv could load the 7B, and the banked parent store
held captures + rollout text for the same pinned prompts/seed. The plan's
own design philosophy already conceded the port-risk class (it gated the
FITS port with a hard anchor, §7 kill (b) "the port is broken, not the
data") — the asymmetry (bigger, riskier capture port left with
internal-only checks) is the tell.

**How to apply:** on any plan-stage split where the new arm's data comes
through a ported/rewritten measurement driver, ask three questions:
(1) does ANY gate compare the port's outputs to the parent instrument's
outputs on a shared model/input (not port-internal consistency)?
(2) is the port's layer/token-index convention an assumption (Medium
confidence, "recalled from conventions") rather than pinned by a reference?
(3) would a capture error be exposed by any persisted column, or does every
diagnostic (null, floor, ceiling) compute from the same captures? If
(1) no + (3) invisible, uphold REVISE even when the parity run is
technically runnable post-hoc — cite feasibility (banked parent
completions/captures + a model both envs load) so the fix lands as a cheap
P1 addition. Distinguish from the loud-halt sibling in the same round:
a count-pin/drop-branch contradiction whose failure direction is a
fail-loud assert stays Real-nonblocking (#606-code precedent).
Related: [[feedback_verify_inherited_capture_semantics_before_crediting_slot_claim]],
[[feedback_gate_design_vs_recoverable_robustness_read]] (#591/#722 REVISE tells).
