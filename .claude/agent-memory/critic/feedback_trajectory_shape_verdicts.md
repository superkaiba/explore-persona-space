---
name: Trajectory-shape verdict designs (zero-crossing vs stable-sign)
description: Alternatives checklist for plans whose verdict is the SHAPE of a d-trajectory over training amount (#547 pattern) — base-prior offsets, floor-pinned legs, wide-CI asymmetry, gated-out early points
type: feedback
---

When a plan's discriminator is the SHAPE of a paired-d trajectory over a training-amount grid (zero-crossing = artifact vs stable-negative = mechanism, e.g. #547), four alternatives recur. All were RECOVERABLE in #547 because the trajectory block reported mean+CI+sign_agreement+implant_active at EVERY grid point unconditionally — check that first.

1. **Base-prior encoding offset masquerading as "stable mechanism."** A cross-arm paired d (system − role) at the trained model is offset by any BASE-model prior difference between the two encodings' contexts at the slot. A constant negative d at all points is then a tokenization/prior fact, not a training-dynamics fact. The near-zero-training grid points (s=5/10, implant-INACTIVE, greyed in figures but still measured) double as the base-reference control: if d there ≈ d at active points, the "stable read" is an offset. Verdict logic that never consults inactive points will overclaim.
2. **Floor-pinned verdict legs.** Any criterion leg at a grid point the parent showed to be floor-saturated (e.g. "d not below zero at s=120" where E=3 sat at −13..−15) is pre-determined by floor compression — both hypotheses predict it. The real evidence is only the early legs; say so.
3. **Wide-CI asymmetry favors the null-shaped hypothesis.** "CI at earliest active point NOT strictly below zero" passes for free when the CI is wide (high data-order variance at tiny step counts), and "≥1 of 4 cells" multiplies the chance. Distinguish "CI includes zero because resolved near zero" from "because underpowered" via CI width + seed sign-agreement.
4. **Implant-active gating can force the stable-sign verdict.** If the genuinely discriminating early points get gated out (install not yet ≥0.5), the earliest ACTIVE point may already be past the hypothesized transient onset — "negative at all active points" then can't distinguish swing-in-flight from always-negative. Also check whether the gate aggregates across ARMS (install-speed asymmetry between arms makes an arm-mixed gate produce d's over install-asymmetric pairs).

**Why:** surfaced reviewing #547 (sub-1-epoch max_steps grid diagnosing #533's E=1 sign reversal). **How to apply:** any marker/behavior trajectory-shape plan in the #464→#529→#533 line or similar dose-response designs.
