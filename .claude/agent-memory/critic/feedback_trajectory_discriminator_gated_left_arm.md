---
name: Trajectory-shape discriminators with gated/floored discriminating points
description: U-vs-monotone trajectory designs whose discriminating early points sit in a regime the plan's own install-gate or floor-noise doctrine degrades — check storage-unconditional + gate-as-flag, and conditional satisfiability of the shape criteria
type: feedback
---

When a plan's discriminator is a TRAJECTORY SHAPE over training amount (zero-crossing vs stable-negative; U vs monotone), check where the discriminating points actually sit (#547, statistics lens, 2026-06-10):

1. **The left arm of a U is usually the gated/floored region.** Early grid points are exactly where (a) an implant-active gate (own argmax-emit >= 0.5 — an emission-CLIFF criterion that lags the log-prob ramp) may exclude them from the read, and (b) the wrong-slot DV sits in the deep-floor regime where the project's own floor-noise doctrine says ~1-nat differences are noise. A near-zero d at a floor-pinned point is ambiguous between "no effect yet" and "floor compresses everything."
2. **APPROVE-compatible iff storage is unconditional.** If the analysis block writes {mean, ci, sign_agreement, implant_active} at ALL grid points (gate = interpretive flag, figures grey out inactive points rather than drop), the analyzer can weigh floor-compression — concern, not Must-Fix. If gated points are NOT computed/stored, that IS conclusion-changing (the discriminating data is unrecoverable).
3. **Conditional unsatisfiability of shape-confirmation wording.** "CI at the earliest implant-active point NOT below zero AND CI at s=mid below zero" becomes self-contradictory when the earliest active point IS s=mid. Distinct from jointly-unsatisfiable kill-gates (always-broken): this is world-state-conditional, and fine when a descriptive Mixed catchall exists + the criteria aren't pass/fail gates. Flag for the analyzer to read shape descriptively, don't REVISE.
4. **Band-match conditions vs known drift.** Requiring |d(s) - d_ref| <= 0.5 nat at points BRACKETING the parent's reference assumes local flatness; if the parent's own data shows drift-toward-zero past the reference, the upper bracket can fail the band under a true stable-mechanism world. A no-zero-crossing trajectory failing only the band check is mechanism-consistent-with-drift, not uninformative.

**How to apply:** for any max_steps/epoch/dose trajectory plan, build the table: per grid point — expected DV regime (floor / band / saturated), gate status, and which hypothesis's distinctive prediction lives there. If a hypothesis's distinctive points are all in the degraded regime, say so as an analyzer concern and check the catchall.
