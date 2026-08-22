---
name: spelled-out-literal-sweep-blindspot
description: On numeric-literal sweep diffs (cap/constant raises), grep the spelled-out English forms of the OLD value too — every sweep instrument is digit-keyed and structurally blind to prose forms
metadata:
  type: feedback
---

When reviewing a coordinated numeric-literal sweep (a cap raise, a constant
bump — the #784/#2391 class), ALSO grep the in-scope files for the
SPELLED-OUT English forms of the old and new values (`five`, `Five`,
`fifth`, `sixth`, ordinal prose) near the swept noun (`rounds`, `revision`).

**Why:** every instrument such plans build — scan patterns, residual sweeps,
cross-line pair audits, negative controls — is digit-keyed (`5`, `cap-5`,
`cap \(5\)`, `revision_round>=5`), so a prose-form cap statement survives
every mechanical gate indefinitely. #2391 r1: `clean-result-critic.md:458`
led its `## Round budget` section with "Five rounds maximum per `/issue`
invocation" while the same section's next sentences were re-keyed to
round 10 — a self-contradictory live reviewer spec that only a spelled-out
grep caught. The two other `five rounds` hits in the tree were incident
HISTORY (`#906: five rounds PASSed`, `#823: five rounds PASSed`) and
correctly preserved — classify prose hits live-vs-history exactly as digit
hits.

**How to apply:** one grep per sweep review, over the plan's in-scope file
manifest (and the sibling agent specs of any loop whose cap changed):
`grep -rniE '(five|fifth|sixth) (rounds?|revision)|rounds? (five|fifth)' <files>`
— substitute the sweep's own old/new values. A live prose hit is a
substantive Major (the sweep's own AC is "surfaces document the new value");
a history hit is a preserve.
