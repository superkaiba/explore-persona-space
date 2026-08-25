---
name: adjudicated-concern-count-override
description: When the brief carries a full-ledger adjudicated open-concern count that differs from list-concerns --open-only, inline the envelope verbatim but make the adjudicated list authoritative and extend Lens 14 to the dropped id(s)
metadata:
  type: feedback
---

When the orchestrator brief states an adjudicated open-concern count that
differs from the `list-concerns --open-only` envelope (e.g. #2254 r3: envelope
8, adjudicated 9 — the reducer drops a concern re-raised at a DOWNGRADED
severity with no addressed event; tooling bug #2530), do all three:

1. Inline the envelope VERBATIM as always (never edit captured output).
2. Add an "ADJUDICATED OPEN-CONCERN COUNT" block naming the authoritative
   count, the full id list, the dropped id(s), and the adjudicating markers
   (e.g. epm:progress v116/v129) — and state that flagging the count is a
   FALSE FINDING.
3. Extend the Lens 14 instruction: the verifier's mechanical Lens-14 line
   ("all K open binding concern(s) acknowledged") audited only the envelope
   rows — Codex must ADDITIONALLY verify body acknowledgement of each
   envelope-dropped id.

**Why:** the envelope is the only ledger path Codex gets; without the
override block a correct-count body draws a false count-mismatch blocker,
and without item 3 the dropped concern escapes the acknowledgement audit
entirely (the two failure modes are opposite — both fire without the pair).

**How to apply:** any round whose brief carries a concern-count adjudication
or names a #2530-class reducer bug. Related: [[fold-round-context-file-briefs]]
(explicit brief adjudications beat envelope/own-kind history).

**Self-derivation rounds (#2254 r5, added 2026-08-25):** the #2530 drop
PERSISTS across later folds, and a later-round brief may carry NO
adjudication field at all. Do not wait for one — re-derive at compose
time: full-ledger reduce over events.jsonl (last event per concern_id;
raised/verified-open = open, addressed/deferred = closed), diff against
the `--open-only` envelope, and cite BOTH the original binding
adjudication markers AND the fresh reduce in the block (r5: envelope 11,
reduce 12, same dropped id as r3/r4). When the body already agrees with
the adjudicated count ("Twelve review-ledger concerns stay open"), the
block's second job is preventing a false count-mismatch flag AGAINST the
body.

**Second blind spot in the same verifier line (#2535, added #2254 r4):** the
mechanical `concerns audit (Lens 14)` PASS is ALSO placement-blind —
`check_concerns_audit` locates acknowledgements via
`section_text(body, "Results")`, whose H2 span runs to EOF and swallows the
footer, so a footer-ONLY acknowledgement passes a check the SPEC says should
reject it. Until #2535 lands, any round auditing acknowledgement PLACEMENT
(a `binding-concerns-footer-only`-class row) must tell Codex to judge
placement against the SPEC text, never the verifier line. The Lens-14 caveat
block now has TWO prongs: count undercount (#2530) + footer swallowing
(#2535) — state both when either fires.

**Third prong — in-cycle rows FAIL the verifier on remediation rounds
(#2254 r6, added 2026-08-25):** a FAIL round's verdicts persist their
`CONCERN:: ` rows to the ledger at verdict collection, so on the NEXT
(remediation-verification) round those rows appear in the `--open-only`
envelope AND drive a `concerns audit (Lens 14)` hard FAIL naming ids whose
entire content is the defects the fix round just removed (r6: OVERALL FAIL
1-of-76, the five `ladder-*` union blockers; nobody posts `address-concern`
until the fixes verify). Handle with a THREE-WAY split in the adjudicated
block — standing binding (body acknowledgement required) / standing NITs /
IN-CYCLE fix rows — plus an item-1 discharge rule: fix VERIFIES ⇒ the row is
discharged-pending-bookkeeping and the check-FAIL must not drive a non-PASS
by itself; fix does NOT verify ⇒ genuinely open + unacknowledged = real
blocker. Also guard the body's standing count ("Twelve review-ledger
concerns stay open") — it counts the STANDING set only; flagging it stale
for excluding in-cycle rows is a false finding. This shape recurs on EVERY
remediation round that follows a concern-persisting FAIL round.

**Why:** without the split, a correct remediation draws either a false
structural-absence blocker (the verifier FAIL) or a false count/staleness
flag against the body; with an unqualified "ignore the FAIL" the unfixed-row
case escapes.

**How to apply:** any round whose brief says remediation/fix-verification
after a FAIL round: map each envelope row raised at the prior round's
verdict-collection timestamp to its fix row, and wire the discharge rule
into item 1 + Lens-14 caveat prong (c).
