---
name: reconciler-sb-remedy-ledger-classes
description: "Reconciler-FAIL SB-remedy rounds (#2502 r3): split the un-bookkept open ledger into FOUR duty classes (SB-mapped / prior-adjudicated bookkeeping-lag / ordered-not-to-touch / open-by-design); translate a brief's elected-option-specific selfcheck wording to the elected option's analogue"
metadata:
  type: feedback
---

Two compose lessons from #2502 review-r3 (2026-08-24), a reconciler-FAIL
remedy round (prior round: Claude 4/4 split PASS vs Codex FAIL → BINDING
reconciler FAIL naming exactly 3 standing blockers, 2 of them the twin's own
upheld findings):

1. **Four-class ledger split when the reconciler re-opened only a SUBSET and
   no `addressed` events were ever recorded.** The raw ledger showed 19 open
   rows (latest event `raised`) — but `raised`-state ≠ open-duty. Classify
   each id by adjudication provenance and give each class its OWN status-line
   vocabulary in the Step 0.8 rewrite + the SB closure ledger:
   (1) SB-mapped ids → the round's closure contract
   (VERIFIED-ADDRESSED / PARTIALLY / NOT-ADDRESSED; NOT-ADDRESSED =
   substantive FAIL); (2) prior-round-VERIFIED ids whose `raised` state is
   bookkeeping lag (the r2 review verified them; the reconciler re-opened
   only the SB classes) → CONFIRM-UNDISTURBED, re-litigation needs NEW
   round-diff evidence; (3) hardening ids the brief ORDERED untouched
   ("touch ONLY the SB code") → OUT-OF-SCOPE-THIS-ROUND, expected
   NOT-implemented, never a re-raise; (4) by-design-open ids
   (pod-parity) → STILL-BINDING-BY-DESIGN. Without the split, an
   adversarial twin reads 19 raised rows as 19 open duties and re-FAILs
   settled adjudications. **Why:** the implementer disclosed "no
   address-concern rows posted — reviewer-verified records are this task's
   convention", so the ledger CANNOT distinguish the classes; only the
   reconciler text + brief can. Flag the bookkeeping lag to the
   orchestrator in the return.

2. **Elected-option translation of brief wording.** The brief offered SB-2
   option (a) OR (b) but its selfcheck-extension list carried an
   option-(a)-specific item ("batched-layer-block equivalence vs serial");
   the round elected (b). Compose the analogue mapping explicitly ("option
   (b)'s analogue is `_selfcheck_shard_assemble` — do not FAIL on the
   literal option-(a) wording"), and likewise mark a brief-sanctioned
   ALTERNATIVE route taken (no corpus.py touch because the fingerprint is
   computed in gen_capture) as not-a-deviation. Same family: the
   do-not-re-raise fences for the twin's reconciler-DOWNGRADED findings
   (each named with its downgrade disposition + the new-evidence re-open
   bar).

Related: [[two-impl-rounds-one-review-compose]] (the #2263 r7
reconciler-FAIL remedy entry this extends), [[revision-round compose
recipe]], [[failloud-compose-script-and-concern-row-shape]] (the fail-loud
script shape reused verbatim; count-assert caught a miscounted Step 0.9 SHA
occurrence — 4, not 3).
