---
name: Negative-swap suppression checks at floor priors
description: Negative-landed gates (lift ≤ +0.05) are one-sided and near-pre-satisfied when the trained-negative's base prior sits at floor; demand base prior + parent-arm untrained lift alongside (#614)
type: feedback
---

In negative-set-membership swap designs (#614: assistant → french_person in the contrastive negative set), the manipulation check "the new trained negative is suppressed: pooled lift ≤ +0.05" is ONE-SIDED: if the check persona's base agreement prior is at floor, lift has almost no room to go negative, so any outcome in [−prior, +0.05] passes — the gate cannot distinguish "negative-training landed" from "training did nothing to this persona".

**Why:** #614 stats-lens — the french_person gate was grounded on the parent's assistant trained-negative reading ~−0.03, but assistant's base prior was 0.014: the parent's own "suppression landed" read was itself nearly floor-pinned.

**How to apply:** for any gating check of the form "trained-negative cell reads at/below a flat band", check the persona's base prior magnitude and whether an untrained-arm contrast for the same persona exists. Not a REVISE when (a) the persona's lift under the PARENT arm (untrained) is available as a contrast and (b) raw per-cell verdicts ship — concern bullet instructing the analyzer to report base rate + parent-arm lift next to the gate read; escalate only if neither contrast exists.
