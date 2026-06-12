---
name: Dynamics designs reusing parent checkpoint ladders — onset left-censoring + ladder saturation
description: Before approving onset-anchored reads on a reused checkpoint ladder, read the parent in-loop trajectory JSON — the source may already be in/past band at the FIRST existing checkpoint and saturated by the second
type: feedback
---

When a trajectory/dynamics plan reuses a parent's fixed-interval checkpoint ladder (e.g. #480's 20-step capend grid) and pre-registers reads "at the source's onset checkpoint" (first checkpoint with Δ ≥ threshold):

- **Check left-censoring against the parent's in-loop trajectory JSON, not the plan prose.** #597 review (2026-06-11): villain in-loop at checkpoint-20 already read Δ = +11.9 nat (trained −9.05 / base −20.96) — past the [5,12] band low at the first checkpoint that exists — and was FULLY saturated (trained log P ≈ −1e-6) by checkpoint-40. So 26/27 ladder points were source-saturated and "onset checkpoint" = first-existing-checkpoint for at least some sources.
- **This is usually analyzer-weighable, not a REVISE**, when (a) the parent's finer in-loop trajectory (5-step) exists to locate true onset, (b) the four-float contract lets the saturated region be read in logit space, and (c) the onset-anchored hypothesis still has a defined read at the censored point. Flag as a concern: report onset as left-censored where applicable; restrict phase-plot "pre-saturation" filters per-cell using eval-probe (not train-row) saturation.
- **Matched-dose interpolation companion:** when matched-dose pairs map the slower arm's pre-saturation window onto the faster arm's first ~10 steps, count how many actual probe points land in the window (in-loop 5-step probes may give only ~2). Interpolation on a steep install ramp is biased; cheap non-parity-breaking fix = finer `marker_band_eval_every_steps` on the fresh arm (read-only callback, training unchanged), keeping the parent's 5-step subset for matched-instrument checks.

**Why:** prose like "onset crossed steps 20–40" sounds like the grid brackets onset, but the in-loop numbers can show the ramp is mostly over before the second checkpoint; an approving critic should know how much of the ladder actually carries pre-saturation signal.

**How to apply:** any plan whose H-criteria anchor on "first checkpoint where metric ≥ X" over a reused grid — read the actual trajectory JSON at the first two grid points before judging testability.
