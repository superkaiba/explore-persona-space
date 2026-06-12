---
name: single-seed-rebase-and-fixed-pool-floor
description: What an HONEST single-seed estimator re-base looks like (537 v6 exemplar), plus a subtle direction-of-bias fact — fixed-question-pool noise floors OVER-correct between-context variance tests (question main effect cancels across contexts)
type: feedback
---

Two reusable findings from the #537 v6 seed-descope delta review (2026-06-10):

**1. The honest single-seed re-base pattern (worked exemplar, APPROVED).** When a
user override drops a design to 1 seed, the statistically honest amendment does ALL
of: (a) every cross-seed estimator explicitly replaced (kill-criterion noise floor →
question bootstrap + split-half `E[(G_A−G_B)²]/4`; anti-fraction → question-split
cross-covariance; disattenuation → question-split reliability with the under-correction
direction NAMED; CIs → question/response bootstrap only; noise ceilings → split-half);
(b) the PASS semantics re-narrated everywhere ("structure exceeds MEASUREMENT noise,
NOT training noise excluded") — §3, gates, success/kill, TL;DR, risks, assumptions,
scope caveats all consistent; (c) a verbatim clean-result caveat contract + a
`single_seed: true` flag in shipped metadata; (d) the seed axis removed from the
descope ladder and made MUST-ASK in both directions. Check all four when reviewing a
seed descope — partial re-bases (e.g. kill fixed but figures/CIs still narrate seed
variance) are the failure mode.

**2. Fixed-pool floors over-correct between-context variance (conservative kill).**
When the SAME frozen question pool scores every eval context, the question MAIN
effect is common to all contexts and contributes ZERO to between-context variance —
but the per-cell split-half/bootstrap noise floor includes main + interaction +
residual variance. Subtracting that floor (or requiring variance ≥ 2× it) therefore
OVER-corrects by σ²_main/N → the structure test is biased toward "noise-limited"
(false kill), not false pass. **How to apply:** a marginal H-structure FAIL under a
fixed pool should be re-read with per-question centering across contexts (remove the
question main effect) before declaring the tensor noise-limited; a PASS is
trustworthy. Recoverable iff per-question values ship. Direction flips if the pool
is resampled per context (then the floor is correct).

Also confirmed: split-half identity `Var_noise(mean_N) = E_splits[(G_A−G_B)²]/4` is
EXACT under balanced complementary halves of a fixed pool (the −1/(N−1) sign
covariance cancels the finite-population term) — no correction factor needed.
