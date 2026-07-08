# c20 calibration probe — round 2 (post axis-binding lookback clamp)

Task #1006 revision round 2. Probe: `check_verdict_lattice_coherence` (kind
forced to `experiment`, as in round 1) over every `tasks/*/*/plans/*.md` in the
main checkout; only TRIGGERED files shown (the "no registered verdict lattice
detected" SKIP excluded). Driver: throwaway `/tmp/c20_cal_1006_r2.py` (round-1
pattern; per-file status + detail + tier/label/atom dump, sorted by issue/file).

**Result: the clamp changes NO corpus verdict.** The probe was run twice on the
same-day corpus — once against the pre-clamp round-1 code (commit 1cdbcc8924)
and once against the post-clamp code — and the two outputs are BYTE-IDENTICAL
(`diff` empty): no status, detail, or atom-binding delta on any of the 60
triggered files. Tally matches round 1 exactly: **2 FAIL / 4 PASS / 43 WARN /
11 SKIP** (FAILs: #923 v4/v5, the motivating incident; PASSes: #813 v1/v2,
#923 plan.md/v6).

Per-file dump (post-clamp code) follows.

---

=== #405 plan.md (real kind: experiment) → WARN
  detail: label 'PASS' (tier 2: PASS / PASS) did not fully parse: >1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='PASS' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='PASS' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)

=== #405 v1.md (real kind: experiment) → WARN
  detail: label 'PASS' (tier 2: PASS / PASS) did not fully parse: >1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='PASS' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='PASS' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)

=== #405 v2.md (real kind: experiment) → WARN
  detail: label 'PASS' (tier 2: PASS / PASS) did not fully parse: >1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='PASS' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='PASS' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)

=== #489 v3.md (real kind: experiment) → SKIP
  detail: label 'H4(b) — matched same-identity cross-type pair test.' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=2
  label='H4(a) — cross-type cosine→ΔG signal.' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H4(b) — matched same-identity cross-type pair test.' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #490 plan.md (real kind: experiment) → WARN
  detail: label 'Confirms H1' (tier 2: Confirms H1 / Falsifies H1) did not fully parse: no sentence with a parseable atom — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='Confirms H1' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='Falsifies H1' UNPARSED-SEGMENT (no sentence with a parseable atom)

=== #490 v1.md (real kind: experiment) → WARN
  detail: label 'Confirms H1' (tier 2: Confirms H1 / Falsifies H1) did not fully parse: no sentence with a parseable atom — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='Confirms H1' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='Falsifies H1' UNPARSED-SEGMENT (no sentence with a parseable atom)

=== #494 plan.md (real kind: experiment) → WARN
  detail: label 'H1-supported' (tier 2: H1-supported / H2-supported) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1-supported' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='H2-supported' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0'), ('primary', ['above', 'below'], 'CI excluding 0')]

=== #494 v1.md (real kind: experiment) → WARN
  detail: label 'H1-supported' (tier 2: H1-supported / H2-supported) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1-supported' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='H2-supported' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0'), ('primary', ['above', 'below'], 'CI excluding 0')]

=== #528 plan.md (real kind: experiment) → SKIP
  detail: label 'H2 — Segmentation' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=2
  label='H1 — Installation' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H2 — Segmentation' otherwise=False atoms=[('point', ['neg'], 'CI<0')]

=== #528 v1.md (real kind: experiment) → SKIP
  detail: label 'H2 — Segmentation' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=2
  label='H1 — Installation' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H2 — Segmentation' otherwise=False atoms=[('point', ['neg'], 'CI<0')]

=== #537 plan.md (real kind: experiment) → WARN
  detail: label-precedence phrase 'wins' in the lattice's section makes the labels order-evaluated — the cell algebra cannot verify an ordered lattice; restate it as the explicit ⇔ partition form
tier=2 n_labels=2
  label='H2' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='Kill / falsification' otherwise=False atoms=[('point', ['pos'], '`delta_vs_base_prior_r2 > 0'), ('primary', ['above', 'below'], 'CI excluding 0')]

=== #537 v9.md (real kind: experiment) → WARN
  detail: label-precedence phrase 'wins' in the lattice's section makes the labels order-evaluated — the cell algebra cannot verify an ordered lattice; restate it as the explicit ⇔ partition form
tier=2 n_labels=2
  label='H2' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='Kill / falsification' otherwise=False atoms=[('point', ['pos'], '`delta_vs_base_prior_r2 > 0'), ('primary', ['above', 'below'], 'CI excluding 0')]

=== #545 v3.md (real kind: experiment) → WARN
  detail: label 'H1-v2' (tier 2: H1-v2 / H4-v2) did not fully parse: >1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1-v2' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)
  label='H4-v2' UNPARSED-SEGMENT (no sentence with a parseable atom)

=== #547 plan.md (real kind: experiment) → SKIP
  detail: label 'H-mechanism' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=3
  label='H-artifact' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H-mechanism' otherwise=False atoms=[('primary', ['below'], 'CI below zero')]
  label='Falsification of H-artifact' otherwise=True atoms=[]

=== #547 v1.md (real kind: experiment) → SKIP
  detail: label 'H-mechanism' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=3
  label='H-artifact' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H-mechanism' otherwise=False atoms=[('primary', ['below'], 'CI below zero')]
  label='Falsification of H-artifact' otherwise=True atoms=[]

=== #571 v1.md (real kind: experiment) → WARN
  detail: label 'Confirmed' (tier 2: Confirmed / Falsified) did not fully parse: predicate token(s) outside every recognized atom: '≥', '<' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='Confirmed' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='Falsified' otherwise=False atoms=[('primary', ['straddle'], 'CI contains 0')]

=== #591 v1.md (real kind: experiment) → WARN
  detail: label-precedence phrase 'in this order' in the lattice's section makes the labels order-evaluated — the cell algebra cannot verify an ordered lattice; restate it as the explicit ⇔ partition form
tier=2 n_labels=3
  label='H1' otherwise=True atoms=[]
  label='H5' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H7' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]

=== #605 v1.md (real kind: experiment) → SKIP
  detail: label 'H-level' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=2
  label='H-level' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='H-shift' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)

=== #612 v1.md (real kind: experiment) → WARN
  detail: label 'H1' (tier 2: H1 / H3) did not fully parse: predicate token(s) outside every recognized atom: '≥', 'not', 'CI', 'excludes' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='H3' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0'), ('primary', ['straddle'], 'CI straddles 0')]

=== #623 plan.md (real kind: experiment) → WARN
  detail: label 'H1' (tier 2: H1 / H0) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H0' otherwise=False atoms=[('primary', ['straddle'], 'CI includes 0')]

=== #623 v1.md (real kind: experiment) → WARN
  detail: label 'H1' (tier 2: H1 / H0) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H0' otherwise=False atoms=[('primary', ['straddle'], 'CI includes 0')]

=== #623 v2.md (real kind: experiment) → WARN
  detail: label 'H1' (tier 2: H1 / H0) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H0' otherwise=False atoms=[('primary', ['straddle'], 'CI includes 0')]

=== #623 v3.md (real kind: experiment) → WARN
  detail: label 'H1' (tier 2: H1 / H0) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H0' otherwise=False atoms=[('primary', ['straddle'], 'CI includes 0')]

=== #627 plan.md (real kind: experiment) → WARN
  detail: label 'H1' (tier 2: H1 / H2) did not fully parse: predicate token(s) outside every recognized atom: '≥', 'excluded' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='H2' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #627 v1.md (real kind: experiment) → WARN
  detail: label 'H1' (tier 2: H1 / H2) did not fully parse: predicate token(s) outside every recognized atom: '≥', 'excluded' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='H2' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #641 plan.md (real kind: experiment) → WARN
  detail: label 'H2' (tier 2: H2 / H1) did not fully parse: predicate token(s) outside every recognized atom: '≤' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H2' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #641 v1.md (real kind: experiment) → WARN
  detail: label 'H5b' (tier 2: H5b / H1 / H2) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=3
  label='H5b' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='H1' otherwise=False atoms=[('primary', ['straddle'], 'CI includes 0')]
  label='H2' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]

=== #641 v2.md (real kind: experiment) → WARN
  detail: label 'H5b' (tier 2: H5b / H1 / H2) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=3
  label='H5b' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='H1' otherwise=False atoms=[('primary', ['straddle'], 'CI includes 0')]
  label='H2' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]

=== #641 v3.md (real kind: experiment) → WARN
  detail: label 'H5b' (tier 2: H5b / H1 / H2) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=3
  label='H5b' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H2' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
tier=2 n_labels=2
  label='H2 supported' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H1 supported' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #641 v4.md (real kind: experiment) → WARN
  detail: label 'H2' (tier 2: H2 / H1) did not fully parse: predicate token(s) outside every recognized atom: '≤' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H2' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #641 v5.md (real kind: experiment) → WARN
  detail: label 'H2' (tier 2: H2 / H1) did not fully parse: predicate token(s) outside every recognized atom: '≤' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H2' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H1' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #642 v1.md (real kind: experiment) → WARN
  detail: label 'H_coverage' (tier 2: H_coverage / H_rank) did not fully parse: predicate token(s) outside every recognized atom: '≥', 'CI' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H_coverage' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H_rank' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #642 v2.md (real kind: experiment) → WARN
  detail: label 'H_coverage' (tier 2: H_coverage / H_rank) did not fully parse: predicate token(s) outside every recognized atom: '≥', 'CI' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H_coverage' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H_rank' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #642 v3.md (real kind: experiment) → WARN
  detail: label 'H_coverage' (tier 2: H_coverage / H_rank) did not fully parse: predicate token(s) outside every recognized atom: '≥', 'CI' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H_coverage' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H_rank' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]

=== #642 v4.md (real kind: experiment) → WARN
  detail: label 'H_survives' (tier 2: H_survives / H_indeterminate) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H_survives' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H_indeterminate' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]

=== #642 v5.md (real kind: experiment) → WARN
  detail: label 'H_survives' (tier 2: H_survives / H_indeterminate) did not fully parse: predicate token(s) outside every recognized atom: '≥' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H_survives' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H_indeterminate' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]

=== #642 v6.md (real kind: experiment) → WARN
  detail: label-precedence phrase 'in this order' in the lattice's section makes the labels order-evaluated — the cell algebra cannot verify an ordered lattice; restate it as the explicit ⇔ partition form
tier=2 n_labels=2
  label='H_survives' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H_indeterminate' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]

=== #642 v7.md (real kind: experiment) → WARN
  detail: label-precedence phrase 'in this order' in the lattice's section makes the labels order-evaluated — the cell algebra cannot verify an ordered lattice; restate it as the explicit ⇔ partition form
tier=2 n_labels=2
  label='H_survives' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes 0')]
  label='H_indeterminate' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excluding 0')]

=== #649 plan.md (real kind: experiment) → WARN
  detail: label-precedence phrase 'wins' in the lattice's section makes the labels order-evaluated — the cell algebra cannot verify an ordered lattice; restate it as the explicit ⇔ partition form
tier=2 n_labels=2
  label='H2' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H0' UNPARSED-SEGMENT (no sentence with a parseable atom)

=== #649 v1.md (real kind: experiment) → WARN
  detail: label 'H2' (tier 2: H2 / H0) did not fully parse: no sentence with a parseable atom — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H2' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H0' UNPARSED-SEGMENT (no sentence with a parseable atom)

=== #649 v2.md (real kind: experiment) → WARN
  detail: label 'H2' (tier 2: H2 / H0) did not fully parse: no sentence with a parseable atom — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=2
  label='H2' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H0' UNPARSED-SEGMENT (no sentence with a parseable atom)

=== #649 v3.md (real kind: experiment) → WARN
  detail: label-precedence phrase 'wins' in the lattice's section makes the labels order-evaluated — the cell algebra cannot verify an ordered lattice; restate it as the explicit ⇔ partition form
tier=2 n_labels=2
  label='H2' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H0' UNPARSED-SEGMENT (no sentence with a parseable atom)

=== #658 v3.md (real kind: experiment) → WARN
  detail: label 'H1' (tier 2: H1 / H2 / H3) did not fully parse: no sentence with a parseable atom — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=2 n_labels=3
  label='H1' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H2' UNPARSED-SEGMENT (no sentence with a parseable atom)
  label='H3' otherwise=False atoms=[('primary', ['straddle'], 'CI overlaps zero')]

=== #813 v1.md (real kind: experiment) → PASS
  detail: tier 2: H1 / H0 / falsification — every interior sign/CI cell fires exactly one label (partition verified in form; boundary semantics stay with the Statistics critic)
tier=2 n_labels=2
  label='H1' otherwise=False atoms=[('primary', ['straddle'], 'CIs include zero')]
  label='H0 / falsification' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes zero')]

=== #813 v2.md (real kind: experiment) → PASS
  detail: tier 2: H1 / H0 / falsification — every interior sign/CI cell fires exactly one label (partition verified in form; boundary semantics stay with the Statistics critic)
tier=2 n_labels=2
  label='H1' otherwise=False atoms=[('primary', ['straddle'], 'CIs include zero')]
  label='H0 / falsification' otherwise=False atoms=[('primary', ['above', 'below'], 'CI excludes zero')]

=== #922 plan.md (real kind: experiment) → SKIP
  detail: label 'H5' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=4
  label='H4' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)
  label='H5' otherwise=False atoms=[('primary', ['above', 'below'], 'CI clear of zero')]
  label='H6' otherwise=False atoms=[('point', ['pos'], 'b1_grad > 0'), ('primary', ['above', 'below'], 'CI clear of zero')]
  label='H7' otherwise=False atoms=[('point', ['pos'], 'direct-c > 0'), ('primary', ['above', 'below'], 'CI clear of zero')]

=== #922 v3.md (real kind: experiment) → SKIP
  detail: label 'H5' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=2
  label='H4' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)
  label='H5' otherwise=False atoms=[('primary', ['above', 'below'], 'CI clear of zero')]

=== #922 v4.md (real kind: experiment) → SKIP
  detail: label 'H5' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=4
  label='H4' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)
  label='H5' otherwise=False atoms=[('primary', ['above', 'below'], 'CI clear of zero')]
  label='H6' otherwise=False atoms=[('point', ['pos'], 'b1_grad > 0'), ('primary', ['above', 'below'], 'CI clear of zero')]
  label='H7' otherwise=False atoms=[('point', ['pos'], 'direct-c > 0'), ('primary', ['above', 'below'], 'CI clear of zero')]

=== #922 v5.md (real kind: experiment) → SKIP
  detail: label 'H5' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=4
  label='H4' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)
  label='H5' otherwise=False atoms=[('primary', ['above', 'below'], 'CI clear of zero')]
  label='H6' otherwise=False atoms=[('point', ['pos'], 'b1_grad > 0'), ('primary', ['above', 'below'], 'CI clear of zero')]
  label='H7' otherwise=False atoms=[('point', ['pos'], 'direct-c > 0'), ('primary', ['above', 'below'], 'CI clear of zero')]

=== #922 v6.md (real kind: experiment) → SKIP
  detail: label 'H5' carries quantified verdict predicates out of v1 scope (k-of-n / per-family lattices are the Statistics critic's)
tier=2 n_labels=4
  label='H4' UNPARSED-SEGMENT (>1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous)
  label='H5' otherwise=False atoms=[('primary', ['above', 'below'], 'CI clear of zero')]
  label='H6' otherwise=False atoms=[('point', ['pos'], 'b1_grad > 0'), ('primary', ['above', 'below'], 'CI clear of zero')]
  label='H7' otherwise=False atoms=[('point', ['pos'], 'direct-c > 0'), ('primary', ['above', 'below'], 'CI clear of zero')]

=== #923 plan.md (real kind: experiment) → PASS
  detail: tier 1: H-robust / H-slot / intermediate — every interior sign/CI cell fires exactly one label (partition verified in form; boundary semantics stay with the Statistics critic)
tier=1 clauses=3
  label='H-robust' otherwise=False atoms=[('primary', ['below'], 'CI wholly below 0')]
  label='H-slot' otherwise=False atoms=[('primary', ['above'], 'CI wholly at/above 0'), ('primary', ['straddle'], 'CI straddles 0'), ('paired', ['above'], 'CI strictly positive')]
  label='intermediate' otherwise=True atoms=[]

=== #923 v4.md (real kind: experiment) → FAIL
  detail: the registered verdict lattice (tier 2: H-slot / H-robust / Intermediate) is not a partition: labels H-slot + Intermediate CO-FIRE on cell {point > 0, primary CI straddles 0, paired CI straddles 0}; no label fires on cell(s) {point < 0, primary CI straddles 0, paired CI wholly below 0} — restate the lattice as an explicit partition (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …; <label> ⇔ otherwise`), add an otherwise-label, or declare 'N/A — no registered verdict lattice' on its own line
tier=2 n_labels=3
  label='H-slot' otherwise=False atoms=[('point', ['pos'], 'Δ_pool ≥ 0'), ('primary', ['straddle'], 'CI includes 0'), ('paired', ['above'], 'CI excludes 0 on the positive side')]
  label='H-robust' otherwise=False atoms=[('point', ['neg'], 'Δ_pool < 0'), ('primary', ['above', 'below'], 'CI excluding 0')]
  label='Intermediate' otherwise=False atoms=[('primary', ['straddle'], 'CI includes 0'), ('paired', ['straddle'], 'CI includes 0')]

=== #923 v5.md (real kind: experiment) → FAIL
  detail: the registered verdict lattice (tier 2: H-slot / H-robust / Intermediate) is not a partition: labels H-slot + Intermediate CO-FIRE on cell {point > 0, primary CI straddles 0, paired CI straddles 0}; no label fires on cell(s) {point < 0, primary CI straddles 0, paired CI wholly below 0} — restate the lattice as an explicit partition (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …; <label> ⇔ otherwise`), add an otherwise-label, or declare 'N/A — no registered verdict lattice' on its own line
tier=2 n_labels=3
  label='H-slot' otherwise=False atoms=[('point', ['pos'], 'Δ_pool ≥ 0'), ('primary', ['straddle'], 'CI includes 0'), ('paired', ['above'], 'CI excludes 0 on the positive side')]
  label='H-robust' otherwise=False atoms=[('point', ['neg'], 'Δ_pool < 0'), ('primary', ['above', 'below'], 'CI excluding 0')]
  label='Intermediate' otherwise=False atoms=[('primary', ['straddle'], 'CI includes 0'), ('paired', ['straddle'], 'CI includes 0')]

=== #923 v6.md (real kind: experiment) → PASS
  detail: tier 1: H-robust / H-slot / intermediate — every interior sign/CI cell fires exactly one label (partition verified in form; boundary semantics stay with the Statistics critic)
tier=1 clauses=3
  label='H-robust' otherwise=False atoms=[('primary', ['below'], 'CI wholly below 0')]
  label='H-slot' otherwise=False atoms=[('primary', ['above'], 'CI wholly at/above 0'), ('primary', ['straddle'], 'CI straddles 0'), ('paired', ['above'], 'CI strictly positive')]
  label='intermediate' otherwise=True atoms=[]

=== #928 v2.md (real kind: experiment) → WARN
  detail: label-precedence phrase 'in order' in the lattice's section makes the labels order-evaluated — the cell algebra cannot verify an ordered lattice; restate it as the explicit ⇔ partition form
tier=2 n_labels=2
  label='H2' otherwise=False atoms=[('paired', ['above', 'below'], 'CI excluding 0')]
  label='H5' UNPARSED-SEGMENT (no sentence with a parseable atom)

=== #928 v3.md (real kind: experiment) → WARN
  detail: label-precedence phrase 'in order' in the lattice's section makes the labels order-evaluated — the cell algebra cannot verify an ordered lattice; restate it as the explicit ⇔ partition form
tier=2 n_labels=2
  label='H2' otherwise=False atoms=[('paired', ['above', 'below'], 'CI excluding 0')]
  label='H5' UNPARSED-SEGMENT (no sentence with a parseable atom)

=== #1006 plan.md (real kind: infra) → WARN
  detail: label '<label>' (tier 1: <label> / <label>) did not fully parse: predicate token(s) outside every recognized atom: '<', '>' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=1 clauses=2
  label='<label>' otherwise=False atoms=[]
  label='<label>' otherwise=True atoms=[]

=== #1006 v1.md (real kind: infra) → WARN
  detail: label '<label>' (tier 1: <label> / <label>) did not fully parse: predicate token(s) outside every recognized atom: '<', '>' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=1 clauses=2
  label='<label>' otherwise=False atoms=[]
  label='<label>' otherwise=True atoms=[]

=== #1006 v2.md (real kind: infra) → WARN
  detail: label '<label>' (tier 1: <label> / <label>) did not fully parse: predicate token(s) outside every recognized atom: '<', '>' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=1 clauses=2
  label='<label>' otherwise=False atoms=[]
  label='<label>' otherwise=True atoms=[]

=== #1006 v3.md (real kind: infra) → WARN
  detail: label '<label>' (tier 1: <label> / <label>) did not fully parse: predicate token(s) outside every recognized atom: '<', '>' — the lattice is not FAIL-capable; restate it as the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; …`) so coherence is machine-checkable
tier=1 clauses=2
  label='<label>' otherwise=False atoms=[]
  label='<label>' otherwise=True atoms=[]

TALLY: 2 FAIL / 4 PASS / 43 WARN / 11 SKIP  (n_triggered=60)
