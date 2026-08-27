---
name: adjudicated-union-round-compose
description: "Adjudicated-union fix round where the twin's OWN blocker was reconciler-OVERRULED (#2546 r18/v18): reconcile record inlined as a BINDING envelope with an explicit challenge lane (engage the mechanism, never re-raise around it); marker-shape moves to NEVER-EMIT despite a present marker when the prior round's gap is CLOSED; composer traces the memo/write-path asymmetry into the scrutiny anchors; U-ledger closure lines split from the concerns ledger"
metadata:
  type: feedback
---

From #2546 r18 compose (sentinel v18, 2026-08-27), layered on
[[diagnosis-dispatched-round-compose]] + the FAIL+FAIL-union entry:

1. **Union + overrule compose TOGETHER.** Prior round FAIL+FAIL with disjoint
   findings unioned AND one direct contradiction reconciled (the twin's OWN
   blocker OVERRULED): inline the `epm:review-reconcile` record verbatim as a
   BINDING envelope, then give the twin an explicit CHALLENGE LANE keyed to
   the reconciler's mechanism ("name where the shard-bytes-purity argument
   fails" -> `ADJUDICATION-CHALLENGED`), alongside the invalid-re-raise fence.
   The brief said "this is not a gag" - carry that framing; a bare
   no-relitigate block without the lane misrepresents the contract. Closure
   items get a U-ledger (U1/U2 upheld = NOT-ADDRESSED-is-substantive-FAIL;
   U3 overruled = COMMENT-ONLY-COMPLIANT | PREDICATE-TOUCHED |
   ADJUDICATION-CHALLENGED; U4 minor = Minor-grain re-raise only), kept
   SEPARATE from the concerns-ledger status lines.
2. **marker-shape can be NEVER-EMIT even with a marker present:** when the
   prior round's sole marker gap is CLOSED (orchestrator reconstruction +
   implementer's own v<n>) and the v<n> body is inlined, put `marker-shape`
   in the never-emit list with the one-line reason (present+inlined => shape
   imperfections cap at CONCERNS; the r<n-1> gap is closed) - cleaner than
   leaving the tag armed and hoping Codex scopes it.
3. **Composer traces the asymmetry the scrutiny target hints at:** the brief's
   memo-staleness target became decidable once the composer greped that the
   repair path EXTENDS the memo (`_HUB_MARKER_SETS[...].add`, :544) while the
   normal write-path mirror calls (:3257/:3549) never touch it - hand that
   asymmetry as an anchor-grounded frame fact and leave the driver-loop
   reachability trace to Codex. Same for except-tuple semantics (:514
   folding auth failures into empty-set - the marker's own (d) discloses it).
4. **Marker per-file numstat can be self-inconsistent while the top line is
   right:** v18's per-file "+X/-Y" bullets were add+del totals mislabeled
   (+121/-11 vs realized 105/16) while `+414/-34` matched `git diff
   --numstat`. Re-derive realized numbers at compose time, put THEM in the
   Diff-size header, and route the mismatch as a neutral F-line
   (record-accuracy, CONCERNS/NIT cap) - never silently adopt marker figures.
5. **Ledger classification needs the latest-EVENT reduction, not the raised
   list:** ids can be raised -> deferred -> RE-RAISED (two sentinel-gap rows
   here); classify open/deferred by latest event per id before writing the
   open/deferred counts and the grouped status-line lists.

Compose script: /tmp/codex-2546-v18-compose.py (fail-loud, file-presence ==
verdict; SHAs via rev-parse; race re-probe asserts max impl version == 18;
prompt /tmp/codex-prompt-issue-2546-v18.md, 76,480 bytes).
