---
name: deferred-event-two-semantics
description: A ledger `deferred` event has TWO distinct semantics — read deferral_rationale before choosing the compose shape; reconciler downgrade-with-recommended-fix arms full closure duties (NOT-ADDRESSED = substantive FAIL), unlike #2332's rejected-binding deferral
metadata:
  type: feedback
---

At #2198 r2 (2026-08-19) the sole concern's event arc was raised → **deferred**
(reconciler) → addressed (implementer, r2). The #2332 r4 recipe entry says "a
reconciler `defer-concern` row means REJECTED-binding (closed this round), NOT
open" — applying that here would have DROPPED the closure duty entirely. Wrong:
this deferral's `deferral_rationale` was a SEVERITY DOWNGRADE
(BLOCKER→non-blocking, finding verified REAL + live-reproduced) with an explicit
"Recommended opportunistic fix" the next round then implemented.

**Why:** the two semantics need opposite compose shapes. Rejected-binding ⇒
no-relitigate block + ban re-emitting the id + no closure duty. Downgrade-with-
recommended-fix ⇒ the fix IS the next round: full per-item closure duty
(VERIFIED-ADDRESSED / NOT-ADDRESSED, NOT-ADDRESSED = substantive FAIL), the
reconciler's standing recommendations inlined as the acceptance contract, PLUS
the no-relitigate-severity line (the downgrade itself stays binding — the twin
must not re-FAIL on the class's severity/reachability).

**How to apply:** at every compose whose ledger carries a `deferred` event, READ
`deferral_rationale` before choosing the shape. Discriminator: a recommended /
opportunistic fix named in the rationale (or a follow-up `addressed` row keyed
to it) ⇒ downgrade semantics; rejection grounds with no fix path ⇒
rejected-binding semantics. **Third shape (#2430 r2, 2026-08-20):** a downgrade
rationale naming an ALTERNATIVE route ("opportunistic round-N fix OR a follow-up
infra task") where the brief/marker attests the implementer took the follow-up
route ⇒ NO closure duty this round — compose an ABSENCE-verification duty
instead (boundary expected UNTOUCHED; untouched = passing; if touched, review on
merits) + the no-re-raise line. Closure duties arm only when the fix is the
round's own contract. Also composable here: the unledgered sibling finding
(a Claude Minor the reconciler's standing rec 2 covered) still gets a pseudo-id
closure item per the #1092-r4 pattern — reconciler standing recs are per-item
acceptance contracts even when only one item has a ledger row.
Related: [[revision-round compose recipe]], [[concerns-machine-rows-2326]].
