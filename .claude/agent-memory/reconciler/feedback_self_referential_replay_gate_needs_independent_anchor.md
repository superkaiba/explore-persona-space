---
name: self-referential-replay-gate-needs-independent-anchor
description: "Claude PASSes a corpus/regression classifier whose 'predicted' verdict is derived from the same artifact under validation (mirror/config/table) — prediction==realized is then ENTAILED, and 'unexplained: 0' certifies nothing about the artifact's correctness. Probe falsifiability by mutating the gate's INPUT, not by recounting agreement. #2514 r2 (after r1 sustained the same class one level up)."
metadata:
  type: feedback
---

**Rule:** when a round's KILL/validation gate classifies flips/diffs as "expected"
by REPLAYING the decision rule with inputs read FROM the artifact under
validation (a mirror, a config table, a threshold file), the gate self-certifies:
any wrong-but-self-consistent artifact predicts its own wrong outputs. Judge the
gate by FALSIFIABILITY — construct a wrong input (unapproved remap, empty
mapping) and check the gate fires — never by recounting that predictions matched
realizations on the committed run (that agreement is entailed whenever the
shared helpers are byte-identical). The fix is to anchor the prediction's inputs
to INDEPENDENTLY-PINNED constants (the plan's approved family sets / expected
mirror content asserted against each leg's header) plus a direction conjunct
implementing the plan's registered taxonomy literally.

**Why:** #2514 r2 (2026-08-24) — round 1 sustained `corpus-flip-taxonomy-overbroad`
(predicate `fams_old != fams_new` true by construction). Round 2 replaced it with
a directional replay `_predicted_c26(recorded row tokens, leg's own mirror)`;
Claude PASSed, citing (a) AST-hash proof the row-parsing helpers are
byte-identical across swept modules and (b) a byte-identical re-run reproducing
175/26/7/0 — but (a) is precisely the premise that makes prediction==realized
ENTAILED for every flipped entry, so (b) measured a tautology. Codex FAILed with
a concrete counterexample (A100→B200 remap self-certifies as expected-inversion);
reconcile sided with Codex. Stronger constructed instance: an EMPTY new mirror
(capture loss) yields mass WARN→SKIP flips, all bucketed "expected-inversion"
(`if not routed: return "SKIP"` matches realized SKIP), while the plan's
registered taxonomy routes anything but the two named family-directional classes
to unexplained (KILL). Claude even DISCLOSED the sibling residual ("a
parser-change-driven flip would self-consistently satisfy the predicate") and
stopped one step short of the mirror case — shape-12 of
[[feedback_claude_gate_unit_vs_preregistered_verdict_logic]].

**How to apply:** (1) On any "0 unexplained / all flips expected" evidence, ask
what the prediction's inputs are; if any input is read from the artifact the
gate validates, the zero is presumptively entailed — trace the entailment before
crediting it. (2) A reviewer's byte-identical/AST-hash equivalence proof cited
as PASS evidence for such a gate is the tell, not the rescue. (3) Closure of a
prior vacuity blocker is graded against the LEDGER concern (the registered gate
must be able to fire on its named target), never against the fix-sketch the
prior reconciler drafted — a faithful implementation of an under-specified
sketch does not close the concern
([[concern-closure-graded-against-ledger-row-not-fix-sentence]]). (4) Demand the
negative-control test as the acceptance signal: construct the wrong input,
assert non-zero exit / ≥1 unexplained. Companions:
[[syntactic-test-pins-and-vacuous-empty-gates]] (probe by semantic mutation, not
removal), [[feedback_codex_meta_test_blocker_on_verified_fix]] (does NOT apply —
its precondition is a separately-verified fix with a shared-code-path test; here
the gate IS the deliverable).
