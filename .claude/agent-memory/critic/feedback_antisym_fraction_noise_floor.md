---
name: antisym-fraction-noise-floor
description: Raw symmetric/antisymmetric variance decompositions (issue502_deltaG_symmetry) have no noise correction — at 2 seeds an "anti-fraction >= 10%" gate is passable by pure measurement noise
metadata:
  type: feedback
---

The `issue502_deltaG_symmetry.py` decomposition reports anti_frac = Var(A)/Var(G) raw, with no noise floor: independent per-cell measurement noise splits ~50/50 into the symmetric and antisymmetric parts, so with 2 seeds a pre-registered "antisymmetric fraction >= X%" hypothesis is confirmable by noise alone. Even a passing H-structure gate (context-var > 2x seed-var) only bounds the noise fraction at ~1/3 of total variance, leaving a noise-driven anti-fraction of up to ~16% — above a 10% threshold. The #524 "directional fraud test" does NOT cover this: it tests whether a PREDICTOR is symmetric-in-disguise, not whether the antisymmetric component of G itself is noise.

**Why:** Surfaced on #537 plan v2 (H-asymmetry read; 2026-06-09). Related to [[n2-sigma-and-perm-cap]] — N=2 noise statistics are algebraically degenerate, not just imprecise.

**How to apply:** When a plan pre-registers an antisymmetry/directionality threshold on a G-type matrix, check seeds-per-cell and whether the decomposition subtracts a noise floor. If per-seed cell values are shipped, this is RECOVERABLE (concern, not REVISE): prescribe seed-split ΔG_anti cross-correlation or noise-floor subtraction before any directionality narration. If only seed-averaged cells are shipped, escalate — the analyzer then cannot weigh it.

**Single-seed variant (#537 v6, 2026-06-10):** at 1 seed (user-directed descope) the honest fallback is split-half-over-eval-questions cross-covariance — it kills question/measurement noise but RETAINS adapter-level training-noise antisymmetry (both adapters of a pair are common to both halves). Acceptable iff (a) per-question G values are persisted per cell (the load-bearing dependency — check this, not just the formula), (b) the residual-training-noise caveat is attached inline to every read of the quantity, and (c) the read is demoted from confirmatory gate to registered-descriptive. Rows with tiny question pools (EM 8 Q) can't split-half — need a named per-row substitute (response-level cluster bootstrap) + pre-flag as least reliable.
