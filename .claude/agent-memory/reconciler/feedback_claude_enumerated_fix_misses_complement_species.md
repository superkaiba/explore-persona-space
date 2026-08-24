---
name: claude-enumerated-fix-misses-complement-species
description: When a fix bars an ENUMERATED char/pattern set under a disclosure/completeness clause, Claude verifies the claimed sets both ways but never probes the COMPLEMENT for surviving same-species members — live-execute the check on remaining boundary chars/shapes (#2228 r2)
metadata:
  type: feedback
---

Rule: when a round's fix bars an ENUMERATED character/pattern set (a regex
class tighten, a matcher denylist) and the binding remedy carries a
disclosure- or completeness-clause ("any X left permitted is disclosed"),
verifying the CLAIMED sets in both directions is not enough — enumerate the
COMPLEMENT's same-species members (clause-boundary punctuation, sentence
enders, Markdown table-cell pipes, cell/field separators of whatever grammar
the operating domain is written in) and LIVE-EXECUTE the check on each. An
enumerated residual list that silently omits permitted members is a
claim-vs-behavior mismatch even when every listed item is individually true.

**Why:** #2228 r2 — the r1 blocker barred `=`/`,` from a harvest regex's
middle class with a clause requiring every permitted clause-boundary char be
disclosed as residual. Claude's r2 review was otherwise exemplary (4 module
variants, `=`-only counterfactual, fixture non-vacuity, corpus re-scan) and
PASSed on "no claim-vs-behavior mismatch remains". It even NOTICED `?`/`!`
unlisted in a Style note but graded them "implausible in gate bullets —
wording note only" WITHOUT probing, and missed `|` entirely — the MOST
probable shape, since plan decision lattices are commonly Markdown tables.
All three live-probed to end-to-end WARN on a healthy plan (same fabricated
cross-clause species as the r1 blocker). Codex FAILed on exactly this;
reconcile upheld FAIL.

**How to apply:** (1) "implausible shape" intuition is never a substitute for
executing the probe — the r1 ruling on the comma form binds consistency for
every same-species survivor; (2) an enumerated disclosure is audited for
COMPLETENESS over the complement, not just truth of listed items — prefer
requiring complement-form disclosure + a table-driven comment↔class sync test
as the durable remedy; (3) the "noticed-but-downgraded-in-a-style-note"
pattern is a tell: when a PASS verdict contains a style note naming permitted
survivors of the just-fixed class, escalate to a live probe before crediting
the PASS. Related: [[feedback_claude_certifies_guard_fix_single_clause_only]]
(probe multi-clause records), [[feedback_union_matcher_probe_prefix_cross_product]]
(execute the escape on both blobs), [[feedback_claude_misses_fix_regressions]]
(replay the old bad input on the replaced check).
