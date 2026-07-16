---
name: rank-space bootstrap tail-mass gating (supersedes strict CI coverage)
description: Gate bootstrap-CI validity in rank space — strictly more than alpha/2 of the finite draws strictly on EACH side of an anchor computed through the SAME vectorized expression as the draws — never a float-space coverage test (lo < point < hi) or any tail test against a point from different arithmetic
type: feedback
---

Strict float-space CI coverage (`lo < point < hi`) is epsilon-fragile:
interpolated bootstrap percentiles land 1e-8..1.6e-7 ABOVE the point on
collapsed small-n distributions, so the coverage test classifies degenerate
cells as healthy and a parity gate deterministically re-fails (#825 r11 rev 3,
t=24/25/29). Any `>= alpha/2` tail test against a point computed by DIFFERENT
arithmetic inherits the same fragility — the identity-resample tie cluster
(mass ~n!/n^n: 22% at n=3, 3.8% at n=5) float-jitters to a random side of a
float64/centered point at ~1e-7.

**How to apply:** a cell is GATING iff strictly MORE than alpha/2 of the finite
bootstrap draws lie strictly on EACH side of the identity-resample anchor
computed through the SAME vectorized expression (same GEMM batch) as the draws
— bitwise tie-exact, alpha = the CI's own level (no new tuned constant). Use
strict `>` (a hi epsilon-above the anchor implies exactly alpha/2 mass above,
which must NOT gate). Annotate per-node tail fractions (`boot_frac_below`/
`boot_frac_above`) for auditability; fail loud on archived nodes lacking them
(no float fallback). Report degenerate cells non-gating; PASS requires zero
gating failures AND >=1 gating cell. (Incident: #825 epm:failure-lesson v11,
code-review v26 FAIL → v27 PASS; supersedes
feedback_small_cell_bootstrap_ci_degeneracy.md.)
