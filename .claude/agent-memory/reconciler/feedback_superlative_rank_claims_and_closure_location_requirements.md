---
name: superlative-rank-claims-and-closure-location-requirements
description: "#2564 interp r2: compute BOTH headroom/rank definitions yourself — Claude rates a false superlative 'within rounding' when the body's own quoted-precision values differ (38 vs 39) and the absolute definition differs 26%; on r2 'NOT closed' claims, uphold where the disposition claims content 'in Result N prose' that isn't there, reject where Codex invents a location requirement from the closure's location list"
metadata:
  type: feedback
---

Two calibration patterns from one adjudication (#2564 interpretation-critic
round 2, Claude PASS vs Codex REVISE → REVISE):

**1. Superlative/rank claims: compute every natural definition yourself.**
The body said "Query form ... keeps the clearest headroom"; Claude verified
the arithmetic correctly (format recovery 38% < query form 39%) but rated it
NIT "within rounding". It was not: at the body's own quoted precision the
integers differ (38 vs 39), and under the equally natural absolute definition
(sqrt(r10) − cos) format's headroom was 26% larger (0.591 vs 0.470). A false
rank claim in a promoted Results section is Real-blocking even when the fix is
one clause — sibling of [[stopping-rule-false-claim-overrides-nit-severity]].
Tell: Claude's own notes concede the direction while assigning NIT.

**2. Round-N "fix NOT closed" claims: grade against the RAISED row + the
closure's claimed LOCATIONS — in both directions.** The analyzer's disposition
table claimed numbers were "in Result 5 prose" / "Result 3 now names ..." that
the live body did not carry (span-mean 0.98, range extremum 1.55, excluded
16/21) — uphold those as promised-but-not-landed (non-blocking when the body
states nothing false and the verdict-level facts are conveyed; the ledger/
disposition overstatement is the defect). But Codex ALSO converted the
closure's location list ("counts in Takeaway 1 and Result 3; Methodology row
names the sensitivity columns") into a fabricated per-location requirement
(counts must ALSO be in the Methodology row) — reject that: a requirement
never raised in round 1 nor promised in the closure is Codex over-read
(interp-site sibling of
[[concern-closure-graded-against-ledger-row-not-fix-sentence]]).

**How to apply:** on any r2+ interp/clean-result split over fix closures,
build a three-column checklist per disputed fix: (a) what the round-1 RAISED
row demanded, (b) what the closure/disposition CLAIMED landed and WHERE,
(c) what the live body carries (grep the live body, not the sidecar Codex may
have reviewed). Uphold gaps between (b) and (c); reject demands beyond both
(a) and (b). Also: when all four r2 concerns are already open `raised` rows in
concerns.jsonl (forwarded from the Codex CONCERN:: block), do NOT re-raise
duplicates — state in the verdict that the persistence duty is satisfied by
the pre-existing rows and which row anchors each upheld finding.
