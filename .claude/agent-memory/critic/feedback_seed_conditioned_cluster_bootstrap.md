---
name: Seed-conditioned cluster-bootstrap CIs with run-noise yardstick
description: 2-seeds/arm persona-cluster bootstrap CIs condition on seeds (0.85 half-width vs 4.4-logit seed swing in #571); registered yardstick + sign-agreement is the acceptable compensation, not a REVISE
type: feedback
---

Persona-cluster bootstrap CIs over paired personas, with per-persona values
averaged over 2 seeds/arm, CONDITION on the realized seeds: in #571 the
persona-axis CI half-width was 0.85 logits while the narrow arm's two seeds
swung 4.4 logits (per-seed contrasts 6.70 vs 12.15). The CI understates
run-level uncertainty ~5x.

**Why:** seed/run noise is the dominant variance component at 2 seeds; the
bootstrap only resamples the persona axis.

**How to apply (psplit follow-up on #571, and any 2-seed paired-contrast
design):** do NOT REVISE for the CI understating run noise IF the plan
registers BOTH (a) a run-noise yardstick (|point| must exceed the largest
within-arm seed-pair arm-mean gap observed in the same run) AND (b)
matched-seed sign agreement before any affirmative label, with a catch-all
indeterminate cell. That combination is the acceptable shape. The analyzer
concern to pass through: never narrate the cluster-bootstrap CI as total
uncertainty, and note the yardstick is itself a max over ~3 two-draw gaps
(noisy, conservative).
