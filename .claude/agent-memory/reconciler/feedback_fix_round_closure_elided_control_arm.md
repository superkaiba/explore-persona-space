---
name: fix-round-closure-elided-control-arm
description: On fix-round closure splits, re-read the registered control's FULL plan clause + the parent reference implementation before crediting a "VERIFIED FIXED" — Claude's closure walk quoted plan §8 with an ellipsis that dropped "+ untruncated refit" and certified a half-implemented control (#2330 r2)
metadata:
  type: feedback
---

**Rule:** when adjudicating a code-review split on a FIX ROUND (the round was
dispatched against a binding reconcile fix scope), verify closure of each
blocker against (a) the plan clause QUOTED IN FULL — expand every ellipsis in
the PASS verdict's plan citations — and (b) the parent/reference
implementation the plan inherits by name. A "VERIFIED FIXED" that quotes the
registered control as "test-restricted read … instead of the 2%-regen
default" when the plan literal is "(test-restricted read + untruncated
refit)" is certifying HALF a control; the elision is the tell.

**Why:** #2330 r2. The r1 reconcile B2 bindingly scoped "implement the #1491
truncation-restriction control", quoting the plan's two-arm definition. The
round implemented only the read arm (`_truncation_restriction` docstring:
"never a refit"; loader hard-pinned `split=="test_1000"`; aggregator routing
could not even produce the train/val masks the refit arm needs — banked
`train_25k` unlisted, `ceiling_draws/seed{S}` path branch missing). Claude
PASSed with the elided quote and never opened
`issue1491_caphit_restriction_analysis.py::phase_b_restriction` (arm 2 read
:275-278 + arm 3 refit :280-285). The fix-round BRIEF's narrower wording
("test-restricted read" explicitly) explains the implementer's scoping but
never rescopes the PLAN — adjudicate against the plan + binding reconcile,
not the brief. Second upheld item, same round: silent-coverage defaults in
the new control's data path (producer accepted partial chunk sets; consumer
read missing aggregate rows as UNCAPPED) — Claude found both halves itself
but classed them Minor by fix size; classify by EFFECT (silent corruption of
a registered confound control = blocking,
[[claude-underclasses-silent-failures]]).

**How to apply:** (1) for each blocker under closure, diff the PASS verdict's
plan quotes against the plan bytes — any `…`/paraphrase over a parenthetical
or conjunction gets the full clause re-read; (2) when the control "inherits
#M's X", open #M's implementation and enumerate its arms — the round must
cover every arm the plan's parenthetical names (a plan parenthetical DEFINES
the inherited scope; parent extras outside it, e.g. #1491's ceiling
variants, are not owed); (3) trace whether the round's loaders/routing can
even REACH the inputs the missing arm needs (structural unreachability
confirms the gap is real, not a wording quibble). Calibration counterpoint
(consistent with r1): Codex's file:line facts all verified again, but its
closure verdict "p1-sentinel-no-writer NOT-ADDRESSED" was over-classed — the
delivered writer + fingerprints satisfied the r1 binding scope, and the
r1-rejected demand (programmatic gate enforcement) does not re-enter as
blocker grounds via a residual (CPU-stub-satisfiable gate record →
CONCERN). Related: [[plan-verbatim-text-vs-plan-binding-mustfix]],
[[split-review-misses-cross-commit-plan-contracts]].
