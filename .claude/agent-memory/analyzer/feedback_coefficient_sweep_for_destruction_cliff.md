---
name: Coefficient sweep for destruction cliff
description: For any steering / amplification eval that reports a single chosen coefficient, plot the per-persona rate curve across the entire coefficient grid before claiming "the marker collapses" or "the direction doesn't carry the mechanism" — the chosen cell may be past a destruction cliff
type: feedback
---

For any steering / activation-amplification / coefficient-scan eval where
the headline cell is one chosen coefficient (e.g. `c=+2.0` in #267): the
per-persona rate curve from `c ∈ {0, 0.5, 1, 2, 4, 8}` MUST be plotted as
a load-bearing figure in Result 1. If the curves are non-monotonic
(peak-then-crash), the chosen cell is past a destruction cliff and the
"effect doesn't survive" reading is partly an artifact of the registered
coefficient choice rather than a property of the recipe.

**Why:** #267 round-1 interpretation claimed "centroid steering collapses
the marker; centroid is no more aligned with prompted ranking than random
noise." Round-1 interpretation-critic verdict was REVISE because the
data showed comedian peaking at c=+1 (rate 0.34, 2.4× the c=0 baseline)
before crashing to 0.00 at c=+2; villain peaking at c=+0.5 (rate 0.21)
before crashing to 0.00; 4 of 10 personas had perturbation ratios above
the registered band's upper bound at c=+2. The "collapse" was
load-bearing on choosing the cell that was in the destruction regime for
nearly half the personas. The calibrated arm at lower per-persona
magnitudes fired at 17.9% mean (2.6× the headline 6.8%) — direct evidence
the destruction-cliff was doing most of the work.

**How to apply:** When the plan is a steering / amplification eval with
a registered coefficient, in Result 1:
1. Plot per-persona rate vs c on the full grid (top panel).
2. Plot per-persona perturbation ratio ‖c·v‖/‖h_baseline‖ vs c with the
   registered band shaded (bottom panel).
3. Check whether the chosen cell is past peak for ≥1 persona; if yes,
   reframe the headline from "the recipe collapses the marker" to "the
   recipe collapses at the chosen coefficient; lower coefficients
   preserve more signal." Confidence: LOW unless the pattern survives
   coefficient calibration.
4. If a calibrated-coefficient arm exists, foreground its mean rate
   alongside the headline cell's mean rate. The gap is direct evidence
   for / against the destruction-cliff reading.

This rule generalizes beyond layer-20 / persona-centroid steering: any
SAE-feature steering, any logit-bias amplification, any RLHF
reward-temperature sweep, any LoRA-strength sweep — wherever the headline
is "this knob does X at value Y", plot the full curve before claiming X.
