---
name: paired-contrast-flat-reference-fixture
description: An e2e fixture whose REFERENCE arm is constructed flat/zero degenerates a registered paired contrast to the treatment arm alone — a dropped subtraction survives every assertion (#2329 q35_ladder_decay R1 g6)
metadata:
  type: feedback
---

When a test pins a registered PAIRED contrast (ΔD = D(treatment) − D(reference)), check whether the
fixture's reference arm produces an identically-zero (or symmetric-cancelling) statistic by
construction. If it does, the contrast is numerically equal to the treatment statistic and the
pairing wire is unconstrained: deleting the subtraction, flipping which arm is subtracted only when
the result is 0, or pairing against the WRONG unit's reference all survive. An adjacent assert that
the reference statistic == 0 makes this WORSE, not better — it certifies exactly the cancellation
that blinds the test.

**Why:** #2329 q35_ladder_decay R1 g6 — the round-pin e2e gave the ceiling arm a
segment-independent score (`(80|12)+d`), so `d_raw_ce == 0` for every carrier and the plan-registered
headline `dd = d_raw_st − d_raw_ce` (issue2329_decay.py:745) equalled the steered drop; exact-value
asserts (0.60/0.55) passed regardless of whether the subtraction existed. Same round, same class:
the donor-identity fixture supplied exactly the 3 frozen donors — the one manifest shape where the
buggy full-distinct-set derivation and the correct first-3-in-build-order derivation coincide — and
masked a false-HALT the final commit had to fix ([[smoke-fixture-authored-with-consumer-keys]],
[[twin-transcription-parity-tautology]]).

**How to apply:** for every registered contrast/difference/pairing a test claims to pin, compute the
fixture's REFERENCE-side statistic by hand; if it is 0 / constant / equal-across-units, flag
CONCERNS with the concrete surviving mutation and prescribe a non-degenerate reference profile
(reference statistic ≠ 0 and different per unit) so the expected point value changes when the wire
breaks. Also check verdict/label functions: an e2e that reaches only the happy branch of a
registered N-way lattice leaves the other branches' predicates (sign flips, unavailable-companion
licensing) unguarded — demand a branch-table test or a parametrized fixture reaching each label.
