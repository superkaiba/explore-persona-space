---
name: own-residual-closure-round-compose
description: Composing a closure round whose acceptance contract is the Codex twin's OWN prior CONCERNS verdict — quote the row + Minor evidence/fix fields verbatim, per-leg grading, parallel-reviewer ledger row disclosure, and the rstrip line-splice blank-line trap
metadata:
  type: feedback
---

Single-residual closure rounds where the re-raised concern is the twin's OWN
prior verdict (#2502 r8: the r7 Codex CONCERNS re-raised
`decide-force-stale-sentinel-window`; the fix prescriptions were the r7
verdict's two MINOR findings, not a reconciler record).

**Why:** the r7 verdict is the only place the acceptance criteria exist —
the ledger row's summary is a 1-line compression; the Minor findings'
Evidence/Fix/Mechanizable fields carry the actual falsifiable contract
(e.g. "delete one required fixture member and assert `_decision_fingerprint`
raises with its relative path" became pin j6a verbatim).

**How to apply:**
- Fetch the prior-round Codex verdict from its /tmp output file (or the
  posted marker); quote the `CONCERN:: ` row AND each residual-leg Minor
  finding (Evidence/Impact/Fix/Mechanizable) VERBATIM as the acceptance
  contract, INDENTED (4-space) so no `^CONCERN:: ` line-start token leaks
  outside the template section. Also quote the prior `## Unaddressed Cases`
  pin-gap bullets — they map 1:1 onto the claimed new pins.
- Per-LEG grading when the residual had multiple legs: one duty (D1/D2) per
  leg, plus an explicit `OVERALL <id> closure (the WORSE of D1/D2 governs
  the id)` row in the duty-ledger template — otherwise the twin averages.
- Author-neutrality instruction: "the residual is YOUR OWN r<k> finding —
  an honest VERIFIED-ADDRESSED and an honest NOT-ADDRESSED are equally
  available" (both in the round-history bullet and Step 0.8).
- Ledger race: by compose time the PARALLEL Claude reviewer may already
  have posted a "Reviewer-verified" `addressed` row (hit live on #2502 r8,
  02:12:58Z). Disclose it as the OTHER twin's independently-posted
  conclusion, NEVER evidence for the Codex grade — omitting it risks Codex
  reading the ledger state as settled.
- Prescription-vs-realization: when the prior Fix offered alternatives
  ("print decision's verdict OR compare-and-fail-loud"), say explicitly
  which route(s) the diff took and that either satisfies the prescription —
  prevents a demanding-more-than-prescribed NOT-ADDRESSED.
- Splice trap: when reusing the prior prompt via line-range replacement,
  `BLOCK.rstrip("\n")` eats the blank line before the NEXT `###` heading —
  splice with `rstrip("\n") + "\n"` and grep-check `-B1 '^### '` seams
  after assembly (4 of 5 seams were broken on the first #2502 r8 pass).
- Verify the round-parent's cleanliness claim: `git diff <r(k)-tip>..<parent>
  --name-only` must show only housekeeping (agent-memory etc.) before the
  prompt asserts "payload files byte-equal the adjudicated tip".

Related: [[two-impl-rounds-one-review-compose]],
[[concern-discharge-round-severity-fence]], [[revision-round-compose-recipe]].
