---
name: reconcile-record-in-dispatch-note
description: "A binding round reconcile may exist ONLY as the orchestrator's stage-dispatch epm:progress note (no epm:code-review-reconciled marker at all): inline that note verbatim as the adjudication-of-record envelope with a provenance fact; reconciler-REJECTED findings leave permanently-stale raised/BLOCKER ledger rows (no reject event type exists) — attest and bar via status lines (#1901 r3)"
metadata:
  type: feedback
---

From #1901 mlp-scaling-densify r3 (2026-08-24): the round-2 Claude-PASS /
Codex-FAIL split was reconciled in-session — NO `epm:code-review-reconciled`
marker was ever posted for the round. Two traps and their fixes:

1. **Locate the reconcile record before assuming a marker.** The only
   `epm:code-review-reconciled` marker on the task (Aug 23) adjudicated a
   DIFFERENT follow-up's round — a kind-grep alone would have inlined the
   wrong adjudication as binding. The real record lived in (a) the round-N+1
   `stage-dispatch` `epm:progress` note ("review round 2: Claude PASS vs
   Codex FAIL, reconciler BINDING FAIL on exactly one blocker (…) +
   recommended … residual closure. Tier-leak finding plan-refuted; coverage
   finding demoted to Minor.") and (b) the ledger writes it drove
   (`verified-open` + `raised` rows timestamped between the Codex verdict
   and the implementer dispatch). Compose: inline the dispatch note VERBATIM
   in its own `---BEGIN ROUND-(N-1) RECONCILE RECORD---` envelope, plus a
   compose-time PROVENANCE fact stating no standalone marker exists and that
   the task's older reconciled marker belongs to another round — else the
   twin hunts for (or worse, finds and trusts) the wrong marker.
2. **Rejected findings rot in the ledger as open BLOCKERs.** The concerns
   ledger has NO reject/withdraw event type, so a reconciler-REJECTED or
   DEMOTED finding's row permanently reads `raised`/BLOCKER (latest event).
   Attest each as "STALE bookkeeping — the ledger has no reject event type,
   not an open blocker" in Step 0.8 AND the Prior-concerns ledger header
   line, route the twin's duty to a status line only (premise-recheck
   legitimate ONLY if THIS diff changed the refuted premise — attest what
   the diff touches), and bar `CONCERN:: ` row re-emission for those ids.
   Without the attestation the twin reads 2 phantom open BLOCKERs and either
   re-raises them or flags ledger inconsistency.

**How to apply:** any revision-round compose whose brief cites a reconciler
verdict — FIRST probe events.jsonl for a round-matched
`epm:code-review-reconciled` marker (round + follow-up label in the note
head); on a miss, fall back to the stage-dispatch note + ledger-write
cluster and inline those with provenance. Related:
[[revision-round compose recipe (round 2+)]],
[[concern-discharge-round-severity-fence]] (the stale-severity caveat's
origin), [[reconstructed-marker compose]].
