---
name: Dynamics designs reusing parent checkpoint ladders — onset left-censoring
description: Before approving onset-anchored reads on a reused checkpoint grid, read the parent IN-LOOP trajectory at the first 2 grid points — the source can be past-band at ckpt-1 and saturated by ckpt-2 (#597)
type: feedback
---

When a trajectory plan reuses a parent's fixed-interval checkpoint ladder and pre-registers reads "at the source's onset checkpoint" (first checkpoint with Δ ≥ threshold):

- **Check left-censoring against the parent's in-loop trajectory JSON, not the plan prose.** #597 (2026-06-11): villain in-loop at checkpoint-20 already read Δ = +11.9 nat (past the [5,12] band low at the FIRST existing checkpoint) and was fully saturated (trained log P ≈ −1e-6) by checkpoint-40 — 26/27 ladder points source-saturated, "onset checkpoint" = first-existing-checkpoint. Prose like "onset crossed steps 20–40" hides that the ramp is mostly over before the second checkpoint.
- **Usually analyzer-weighable, not REVISE**, when (a) a finer in-loop trajectory (5-step) exists to locate true onset, (b) the four-float contract lets the saturated region be read in logit space, and (c) the onset-anchored hypothesis still has a defined read at the censored point. Flag: report onset as left-censored; restrict phase-plot "pre-saturation" filters per-cell using eval-probe (not train-row) saturation.
- **Matched-dose interpolation companion:** when matched-dose pairs map the slower arm's pre-saturation window onto the faster arm's first ~10 steps, count how many actual probe points land in the window (5-step in-loop probes may give ~2). Cheap non-parity-breaking fix: finer `marker_band_eval_every_steps` on the fresh arm (read-only callback), keeping the parent's 5-step subset for matched-instrument checks.

**How to apply:** any plan whose H-criteria anchor on "first checkpoint where metric ≥ X" over a reused grid — read the actual trajectory JSON at the first two grid points before judging testability.
