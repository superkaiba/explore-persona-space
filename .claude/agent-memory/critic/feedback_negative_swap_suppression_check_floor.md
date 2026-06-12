---
name: Negative-swap suppression manipulation checks at floor priors
description: "Negative landed: lift <= +0.05" gates are one-sided and near-pre-satisfied when the trained-negative persona's base prior sits at floor; demand base prior + parent-arm lift reported alongside (#614)
type: feedback
---

In negative-set-membership swap designs (#614: assistant -> french_person in the
contrastive negative set), the manipulation check "the new trained negative is
suppressed: pooled lift <= +0.05" is ONE-SIDED. If the check persona's base
agreement prior is at floor (panel priors run 0.037-0.30), lift has almost no
room to go negative, so any outcome in [-prior, +0.05] passes — the gate then
cannot distinguish "negative-training landed" from "training did nothing to
this persona." Not a REVISE when (a) the persona's lift under the PARENT arm
(where it was untrained) is available as a contrast, and (b) raw per-cell
verdicts ship so the analyzer can report base prior + parent-arm lift alongside
the gate. Right disposition: concern bullet instructing the analyzer to report
the trained-negative's base rate and its parent-arm (untrained) lift next to
the gate read; only escalate if neither contrast exists.

**Why:** #614 stats-lens review — french_person (cos_se 0.907) gate <= +0.05
grounded on the parent's assistant trained-negative reading ~-0.03, but
assistant's base prior was 0.014, i.e. the parent's own "suppression landed"
read was itself nearly floor-pinned.

**How to apply:** any plan whose gating manipulation check is "trained-negative
cell reads at/below flat band" — check the persona's base prior magnitude and
whether an untrained-arm contrast for the same persona exists in the design.
