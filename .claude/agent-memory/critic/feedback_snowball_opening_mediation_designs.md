---
name: snowball-opening-mediation-designs
description: 'Dispositions for opening-mediation (snowball) decomposition designs: prefill-only verdict registration, dual whole/continuation judge reads, null-direction check, recovery-ratio compatibility band (#2333 v4 + v8 language amendment)'
metadata:
  type: feedback
---

Dispositions for snowball / opening-mediation designs (ce-patch vs first-k
token-prefill vs first-k state-patch; #2333 v4, APPROVE round 1).

**Why:** three alternatives look fatal at first read but are design-insulated
or recoverable when the plan has the right structure; flagging them as REVISE
is noise.

**How to apply:**
1. **Judge scoring the donor's forced prefill tokens** (not behavior change)
   is FULLY recoverable iff the continuation-only companion is computed on
   steered AND null rollouts at the same grain — then the whole paired-diff
   lattice is recomputable on the companion post-hoc. Check the §9 judge-call
   arithmetic (companion count must be × {steered, null}), not just the §6 DV
   row. Concern wording: on whole-vs-continuation divergence, the snowball
   MECHANISM claim rides the continuation read (the mechanism IS
   "continuation stays consistent with the opening").
2. **State patches at k positions carry far more information than k tokens**
   (all-layer states captured under the ce patch = a relayed/delayed ce
   patch, persisting via KV). Not fatal when the verdict lattice is
   registered on the token-only prefill arm ONLY and the patch arms are
   descriptive with an interpretive note ([[#2333]] §3/§4.2 shape). Concern:
   never narrate patch-arm recovery as "the opening's states channel" alone.
3. **Cross-run drift** (banked ce-control / anchors vs fresh arms) shifts
   only the recovery-share band boundary (H1 vs H2), not the within-run CI
   conjunction, when the plan makes the band interpretive and the CI
   conjunction confirmatory; a state capture-parity gate (cos >= 0.999 vs
   banked states) is an acceptable stack-parity proxy for greedy generation
   parity.
4. **Null-direction channel:** steered − shuffled-donor-null diff can be
   positive via null BELOW floor (wrong-pair opening pushes away), not
   steered movement toward B. Weighable when F is floor/ceiling-anchored and
   per-arm steered + null means are both plotted vs the floor.
5. Mediation-donor degeneracy (ce patch fails to change the opening on some
   pairs) is NOT a confound — identical openings + a large banked ce effect
   is itself evidence against snowball-sufficiency; the token-divergence-rate
   manipulation check is the right diagnostic.
6. Amendment rounds (v8, new cell set, recipe held — APPROVE): sibling-wave
   ANCHOR reuse is licensed when token-identity + capture-parity gates bind
   the reused generations to this run's own capture AND the anchors are
   re-judged in the same wave (removes judge-wave drift; identical-text
   offset table for free). A recovery-RATIO lattice (R_k vs 1) stays
   interpretable because the fresh same-wave ce denominator carries its own
   null variant = a same-wave separation diagnostic; weak/unseparated
   control ⇒ R unstable, but the difference-form D3 stays primary —
   analyzer concern, never REVISE. Residual anchor cap-hit above the grid's
   own >2% bar on the REUSE path = concern not REVISE when the construct
   (language) commits early and anchors enter only as shared per-pair
   normalization (bias ~cancels in paired/ratio reads).
7. Amendment plans carry stale-draft numerics: check the §12 divergence
   list's booking figures against the §0 machine-readable est line + §9
   totals (v8 carried "14 booked" from a pre-correction draft vs the real
   36 worst-case) — concern/consistency-checker turf, not conclusion-
   changing when §9's basis is measured-pilot-grounded.
8. **3-value cycle null is a MIXTURE on language cells** (#2333 v8): a
   value-constrained derangement guarantees donor-B ≠ recipient-B but NOT
   ≠ recipient-A, so with 3 values the null openings split into inert
   (recipient's own A-language) and third-language (genuinely disruptive —
   can raise Δ=(judge_B−judge_A) by depressing judge_A, a positive
   steered−null channel with no movement toward B). Weighable, never
   REVISE: the frozen donor_assignment lets the analyzer split null rows
   by donor-value class for free; suggest that split + per-arm
   steered/null means vs floor.
