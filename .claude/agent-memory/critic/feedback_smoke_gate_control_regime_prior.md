---
name: Smoke-gate thresholds grounded only in the treatment arm's priors
description: Manipulation-check gates on a NOVEL control regime (no prior) can false-halt when the extreme hypothesis outcome looks like a rig bug; also frozen-arm reuse censoring-path notes from #608
type: feedback
---

Rule: when a plan's smoke/manipulation-check gate thresholds the FIRST-EVER run of a regime (e.g. positive-only when only contrastive priors exist), check whether the hypothesis's own extreme outcome (severe under-install) would FAIL the gate and be misclassified as a rig bug. The gate's grounding cites the OTHER arm's priors by necessity — that is not grounding for the new regime.

**Why:** #608's smoke gate required villain posonly Δself ≥ +0.20, grounded in contrastive priors (+0.654 min on-rig), while the plan's own §2 establishes no true positive-only sycophancy run ever existed. Severe under-install is H1's extreme support. (There it stayed a Concern, not REVISE, because marker-line priors #18/#471 show positive-only OVER-installs, making <+0.20 implausible absent a bug.)

**How to apply:** Concern (not REVISE) when an independent prior makes the gate-failing outcome implausible absent a bug; prescribe "FAIL + clean debug = candidate finding, run one more cell" rather than indefinite halt. Escalate to REVISE only if the hypothesized effect direction plausibly lands below the gate AND the FAIL path is a hard kill.

Related #608 statistics-lens notes (frozen-arm reuse designs, all Concern-level):
- g(s) = own_c − own_p cancels the shared frozen base exactly — base-noise-free headline; paired claim bootstrap is the right unit when claims are shared across arms.
- Censoring precedence: top-band-censored cells must override a practical-equivalence falsification verdict (ceiling-equal rates ≠ equal latent install).
- Censored-path trajectory fallbacks can themselves left-censor (epoch-1 already ≥0.95); only the earliest small-step checkpoint may resolve.
- Residual judge drift passing a ±0.02 gate equals a ±0.02 equivalence band edge — report drift point estimate; prefer unified re-judge on threshold-adjacent results.
