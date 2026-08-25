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
verdict — FIRST probe events.jsonl for a round-matched reconcile marker
(round + follow-up label in the note head); on a miss, fall back to the
stage-dispatch note + ledger-write cluster and inline those with provenance.
Related: [[revision-round compose recipe (round 2+)]],
[[concern-discharge-round-severity-fence]] (the stale-severity caveat's
origin), [[reconstructed-marker compose]].

**Kind-name variance (#1901 r5, 2026-08-25):** the round-4 reconcile WAS
posted as a marker, but under kind `epm:review-reconcile` (head sentinel
`<!-- epm:review-reconcile v4 -->`), NOT `epm:code-review-reconciled` (the
kind the same task used on Aug 23). Grep BOTH kinds (plus the stage-dispatch
note fallback) before concluding no record exists. Same round's other
posting gaps, all orchestrator-side (flag in the return, never fix): the r4
Codex FAIL verdict lived only in `/tmp/codex-code-reviewer-1901-r4-output.md`
(job-completed marker present, no `epm:code-review-codex` row), and the v11
impl marker note head carried NO version sentinel (top-level version field
only) — hand that to the twin as an out-of-scope composer observation so it
doesn't burn the verdict on it. The r5 compose itself validated the TIGHT
binary-enum micro-scoped discharge shape: reconcile record inlined as the
acceptance contract, per-discharge VERIFIED-DISCHARGED|NOT-DISCHARGED status
lines (D1 BLOCKER not-discharged ⇒ FAIL `substantive`; D2 fence
genuinely-absent-again ⇒ FAIL `marker-shape` per the reconcile's
workflow.yaml:1302 ruling; present-but-imperfect ⇒ Minor, never FAIL),
new-diff defects at the ordinary bar, everything else fenced out, ordinary
green-claim duty translated to static-trace + optional single-nodeid run
with the `STATIC (env unavailable)` fallback.
