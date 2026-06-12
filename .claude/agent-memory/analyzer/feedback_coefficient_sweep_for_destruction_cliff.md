---
name: Coefficient sweep for destruction cliff
description: For any steering / amplification eval reporting a single chosen coefficient, plot the per-persona rate curve across the full coefficient grid before claiming "the effect doesn't survive" — the chosen cell may be past a destruction cliff
type: feedback
---

For any steering / activation-amplification / coefficient-scan eval whose headline is one chosen coefficient, plot the per-persona rate curve over the FULL grid as a load-bearing figure before writing the headline. Non-monotonic curves (peak-then-crash) mean the chosen cell is past a destruction cliff, and "the effect doesn't survive" is partly an artifact of the registered coefficient choice, not a property of the recipe.

**Why:** #267 round-1 claimed "centroid steering collapses the marker"; the critic REVISEd because comedian peaked at c=+1 (0.34, 2.4× baseline) before crashing to 0.00 at the headline c=+2, villain peaked at c=+0.5, and 4 of 10 personas had perturbation ratios above the registered band at c=+2. A calibrated lower-magnitude arm fired at 17.9% vs the headline's 6.8% — the cliff was doing most of the work.

**How to apply:**
1. Plot per-persona rate vs c on the full grid; below it, perturbation ratio ‖c·v‖/‖h_baseline‖ vs c with the registered band shaded.
2. If the chosen cell is past peak for ≥1 persona, reframe to "the recipe collapses AT THIS coefficient; lower coefficients preserve more signal"; confidence LOW unless the pattern survives coefficient calibration.
3. If a calibrated arm exists, foreground its mean rate beside the headline cell's.

Generalizes to any knob (SAE-feature steering, logit bias, LoRA strength, reward temperature): plot the full curve before claiming "this knob does X at value Y".
