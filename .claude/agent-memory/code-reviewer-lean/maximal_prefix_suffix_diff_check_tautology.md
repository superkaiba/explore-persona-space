---
name: maximal-prefix-suffix-diff-check-tautology
description: Minimal-pair gate predicates — prefix/suffix form is a tautology, opcode-count form false-rejects multi-region-by-design pairs; final form is locus-anchored span identity, certified by a full-real-corpus run at review time (#2333 R1 g1 → R3)
metadata:
  type: feedback
---

A "pairs differ in only ONE contiguous token window" check implemented as
maximal-common-prefix `p` + maximal-common-suffix `s` (with `s` loop-bounded
by `min(len_a,len_b) - p`) and a verdict like `p + s <= min(...)` is a
TAUTOLOGY: the loop bound guarantees the inequality, and maximal
prefix/suffix trivially define exactly one middle window for ANY two
sequences (a fully reversed sequence "passes"). #2333 R1 g1: the plan-A16
minimal-pair HALT gate, its violation-fraction halt, and the pair-drop
machinery all hung off this — dead on arrival, and it was the only
decision-bearing helper with no unit test.

**Why:** the property people mean is LCS-structural — exactly one
non-`equal` region in `difflib.SequenceMatcher` opcodes (or: identical
outside a declared varied span). Maximal prefix/suffix cannot distinguish
one window from many.

**How to apply:** when a diff adds any single-window / minimal-pair /
"differs only in span X" predicate, hand-execute it on `[1,2,3,4]` vs
`[1,9,3,9]` (two mismatches) and a reversed list; if both pass, it is
hollow. Also check the predicate has a unit test WITH a violating input —
a gate whose tests only feed conforming pairs certifies nothing. Related:
[[registered-gate-quantity-substituted]].

**R2 nuance (the fix's own residual):** the opcode-based fix is
CONSERVATIVE, not exact — SequenceMatcher maximizes matches, so a pair
identical outside ONE declared window whose window interiors share an
internal token run (X1·M·X2 vs Y1·M·Y2) reports TWO non-equal regions and
false-REJECTS. Fails toward pair-drops/halt, never silent acceptance —
acceptable for a validity gate, but check the drop/halt telemetry is
reported and, where a known-good corpus exists (q25 parents), expect
violation count ≈ 0 there as the calibration read (#2333 R2 g1).

**R3 resolution (the residual FIRED):** the opcode-count grain false-
rejected 77/195 (q25) / 82/195 (q35) real pairs and halted the pod —
conflict cells interleave ~12 prefix diff regions BY DESIGN, so ANY
whole-sequence region-count threshold is a mis-specification, not just
conservative. The correct predicate is LOCUS-ANCHORED SPAN IDENTITY:
identical outside the declared varied span, boundary anchored at a
structural atomic token (second-to-last `<|im_start|>` = final-turn
start), i.e. `ids_a[pe:] == ids_b[pe:]` AND varied-slice-differs
(#2329 `bank2329._pair_verdict`). Two review duties this adds:
(1) the calibration read above is NOT optional — run the predicate over
the FULL real consumed corpus at review time (a 5-line probe through the
production ids_fn; synthetic unit shapes certified a predicate that
failed 40% of real pairs); (2) certify fails-pre-fix by extracting the
parent-commit predicate and executing it on the new PASS shapes
([[fails-pre-fix-probe-parent-commit]]). Empty-slice edge: at prefix-side
locus a bare-vs-bare pair has two EMPTY varied slices → flags
`varied-prefix-identical`; confirm the universe cannot pair two bare
renders. #2333 R3.
