---
name: Re-judge proxy-fix amendments — alternatives decomposition
description: For judge-rejudge DV-amendment rounds (#591 e5 pattern), the drift/denominator/refusal/coherence alternatives are all weighable iff three panels ship from the same fresh verdicts + per-cell refusal counts persist; refusal censoring tracks harm content and is CONSERVATIVE for suppression survival
type: feedback
---

From #591 follow-up `e5-em-rejudge-proxy-fix` (proxy survivor-rate DV → direct all-rollouts re-judge of frozen completions):

1. **The three-panel decomposition is the load-bearing control.** Proxy-vs-corrected
   crosses TIME (frozen judging vs now) and DENOMINATOR (survivors vs all) at once.
   Sufficient design = emit fresh-all AND fresh-conditional from the SAME verdicts,
   keep frozen-conditional as target: fresh-conditional vs frozen-conditional = drift
   (+ survivor resampling noise on low-survivor cells); fresh-all vs fresh-conditional
   = denominator, time-clean. A 5-cell aggregate parity-anchor gate alone is NOT
   sufficient (content-dependent drift passes it); anchors + full-panel conditional
   re-emit IS. Check the conditional panel is computed from the same verdict set.

2. **Refusal censoring is empirically non-random and direction matters.** e1 measured
   judge empty-response rows tracking harm content 5× (villain 40/480 vs
   software_engineer 8/480). Excluding refused rows censors misaligned completions →
   understates HIGH-misalignment cells. For suppression cells (trained low, base high)
   this understates BASE → attenuates suppression → conservative for "suppression
   survives". For leak cells it attenuates leaks → leak→neither flips can be censoring.
   Weighable iff per-cell n_refused persists: worst-case imputation bound = n_refused/N
   shift per rate, ~2× that per delta. Prescribe per-cell worst-case bounds + flag
   class labels within the bound of the τ boundary. ≤10% caps can't flip a −0.48 delta.

3. **At low survivor fractions the "corrected" mean is dominated by the incoherent
   stratum** (15% survival → ~85% of corrected denominator is text the judge calls
   incoherent). "Flips → proxy distorted" then presumes incoherent-row aligned scores
   measure misalignment, the least-validated stratum. Weighable iff per-row
   (aligned, coherence) joint persists per cell: decompose each flip into
   coherent-stratum vs incoherent-stratum contribution; binary P(aligned<thr) bounds
   mid-scale noise. The honest frame is "disagreement among the three reads IS the
   finding", not corrected=truth.

4. **Promotion wording scope:** suppressions surviving a clean denominator promotes
   them to "not a denominator artifact" — judge-construct (scored-alignment ≠ harm)
   and persona-adherence-erosion channels are SHARED by both panels and survive the
   round by construction; keep them inside "mechanism unknown" + the judge scope caveat.
