---
name: Rank-1 firing-test alternatives (#621 lineage)
description: Direct (a,b) rank-1 read/write plans — shift-DV base-prior comparator is a strawman (cosine is the real rival), folded |firing| inverts trained-negative suppression, frozen-A is near-mechanical at band-stop dose, and trained negatives contaminate both the bystander Spearman and the wrong-context null
type: feedback
---

From the #621 rank-1 LoRA read/write review (2026-06-12; alternatives lens,
APPROVE). Applies to any plan training rank-1 (or directly-readable) adapters
and testing whether firing strength a·v_c orders bystander leakage.

**1. On the SHIFT DV, the base-prior comparator is a strawman.** #532's
follow-up showed the leaderboard inverts between absolute leakage (prior
+0.72, cosine +0.22) and the training-induced shift Δ log P (cosine +0.66,
prior ≈ 0). A firing test whose DV is the shift and whose only registered
comparator is the base prior will "beat the comparator" almost for free. The
discriminating rival is the plain geometric predictor cos(v_c′, v_source)
(and, for the frozen-A regime, a context-norm covariate ‖v_c′‖ — random-a
|firing| reduces to a norm-weighted random projection). Both are free from
persisted bank centroids → analyzer concern, not REVISE, iff centroids +
per-cell (a,b) persist.

**2. Sign-folding |a·v_c′| destroys relative-sign structure.** Only the JOINT
(a,b) sign is gauge; the relative signs of a·v_c′ ACROSS contexts are
meaningful. Signed firing with the global sign fixed by a·v_source > 0 is
equally flip-invariant and is the right statistic. Folding maps trained-
negative suppression (genuinely predicted-negative firing) to predicted-HIGH
leakage and can false-fire a |ρ|<0.2 kill. Recompute signed post-hoc.

**3. Trained negatives contaminate two reads.** (i) Negative-panel personas
inside the eval bystander set have leakage trained DOWN directly — report the
firing Spearman with/without them. (ii) Δa ∝ (source-slot − negative-slot)
activations mechanically, so sign-folded |cos(a, v_negative)| entries inflate
the wrong-context null p95 — a "read identity at null" verdict is partly
mechanical unless the null is also recomputed excluding trained negatives.

**4. Frozen-A (|cos(a_t,a_init)|>0.9) is near-guaranteed at band-stop dose.**
B=0 init ⇒ zero A-gradient until b grows (arXiv 2406.08447); with band-stop at
~30–40 steps, A sees meaningful gradient for only the tail of the run.
"Ungated write" headlines must be scoped to the dose, with deep-dose /
saturated parent reads (#604: key still at null at 19 nat and saturation)
carrying the "didn't need to train" vs "hadn't trained yet" split; the a(t)
checkpoint trajectory (rotation still growing at stop?) is the in-task read.

**5. Under frozen-A, cross-seed a-instability is mechanical** (different
random inits ⇒ |cos|≈0 expected) — do NOT narrate it as "gate not a function
of the data"; it is only informative if A rotated.
