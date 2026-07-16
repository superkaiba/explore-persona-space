# Cross-behavior lattice branch flips: check the parent's CI + imbalance direction per DV (#1315 r2)

Two traps from the #1315 round-1 REVISE, both generalize to any child experiment
re-running a parent's registered verdict lattices on a new behavior:

1. **A branch flip is not a break when the CIs overlap the parent's.** The child's
   D_rank −4 (CI −4 to −1) "excluded zero" where the parent's −3 (CI −5 to +1) did
   not — but one mode of point difference with near-total CI overlap is a flip
   driven by interval width (difference in significance ≠ significant difference).
   Before narrating a lattice branch as a cross-behavior BREAK, pull the parent's
   point + CI for the SAME contrast and put them side by side; a dose-matched
   in-run read (the child's own dose bracket) is the cheapest tiebreak.

2. **Install-imbalance "conservativeness" is DV-specific, not global.** With the
   FT side over-dosed relative to LoRA: more dose ADDS mean-shift norm (so a
   norm-null is conservative) but LOWERS rank (so a "more concentrated" rank read
   is ANTI-conservative). Never write one blanket "the imbalance makes the null
   reads conservative" — derive the sign per DV from the run's own dose bracket.

3. **Count lattices can be non-discriminating even when they fire.** The child's
   4-of-4-above-bound alignment count would ALSO have called the parent "Aligned"
   (parent realized 3 of 4 above its bound); rest such breaks on MAGNITUDE and say
   the count carries no discrimination.

Bonus instrument lesson: pod-side capture text audits (greedy) do NOT cover the
temp-1.0 judged install pools — scan those separately (zeroed + excluded bounds;
`scripts/issue1315_cjk_audit.py` is the reusable recipe) because parity PASS
labels can be convention-dependent under intrusion.
