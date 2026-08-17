---
name: snowball-opening-mediation-designs
description: 'Alternatives-lens dispositions for opening-mediation (snowball) decomposition designs: prefill-only verdict registration, dual whole/continuation judge reads, null-direction check (#2333 v4)'
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
