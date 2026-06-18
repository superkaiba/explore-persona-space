---
name: antisym-fraction-noise-floor
description: Raw sym/anti variance decompositions have no noise correction — at 2 seeds an "anti-frac ≥ 10%" gate passes on noise alone; prescribe seed-split ΔG_anti reliability; single-seed fallback = question-split cross-covariance, demoted to descriptive (#537)
metadata:
  type: feedback
---

`issue502_deltaG_symmetry.py` reports anti_frac = Var(A)/Var(G) raw, with no noise floor: independent per-cell measurement noise splits ~50/50 into symmetric and antisymmetric parts, so with 2 seeds a pre-registered "antisymmetric fraction ≥ X%" hypothesis is confirmable by noise alone. Even a passing H-structure gate (context-var > 2× seed-var) only bounds the noise fraction at ~1/3 of total variance — noise-driven anti-fraction up to ~16%, above a 10% threshold. The #524 "directional fraud test" does NOT cover this (it tests whether a PREDICTOR is symmetric-in-disguise, not whether G's antisymmetric component is noise).

**Why:** #537 v2 (2026-06-09). Related: [[n2-sigma-and-perm-cap]] — N=2 noise statistics are algebraically degenerate.

**How to apply:** when a plan pre-registers an antisymmetry/directionality threshold on a G-type matrix, check seeds-per-cell and whether the decomposition subtracts a noise floor. Per-seed cell values shipped → RECOVERABLE (concern): prescribe seed-split ΔG_anti cross-correlation or noise-floor subtraction before any directionality narration. Only seed-averaged cells shipped → escalate.

**Single-seed variant (#537 v6):** the honest fallback is split-half-over-eval-questions cross-covariance — kills question/measurement noise but RETAINS adapter-level training-noise antisymmetry. Acceptable iff (a) per-question G values persist per cell (check THIS, not just the formula), (b) the residual-training-noise caveat attaches inline to every read, (c) the read is demoted from confirmatory gate to registered-descriptive. Rows with tiny question pools (EM 8 Q) can't split-half — need a named substitute (response-level cluster bootstrap) + pre-flag as least reliable.
