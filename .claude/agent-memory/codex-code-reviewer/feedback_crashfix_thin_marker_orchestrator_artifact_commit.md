---
name: crashfix-thin-marker-orchestrator-artifact-commit
description: "Crash-fix rounds can ship a THIN one-liner impl marker (substance in the round-matched smoke-arch marker + the epm:progress diagnosis) and an orchestrator-executed artifact commit POSTDATING the marker — pre-adjudicate both; key the closure ledger to the diagnosis's numbered fix items (#2479 r7)"
metadata:
  type: feedback
---

Two composable facts first hit on #2479 r7 (crash-fix r1, whole-round unsplit,
2026-08-23) that extend the crash-fix entries in
[[revision-round compose recipe]]:

1. **THIN impl marker on a crash-fix round.** The round's
   `epm:experiment-implementation` was a 267-char ONE-LINER (head line only —
   no (a)-(d) H3s, no smoke digest). Do NOT take the #2329-r4 "no marker"
   shape OR demand the 4-H3 form: inline the one-liner verbatim, and
   pre-adjudicate — present-but-imperfect shape is never a FAIL ground
   (Step 0.7); the report SUBSTANCE lives in (a) the ROUND-MATCHED
   `epm:smoke-architecture-check` body (production-shape rc=0 runs,
   reproduced failure arithmetic, import-resolution, blind-spot enumeration)
   and (b) the `epm:progress` crash-diagnosis note. Key Step 0.6 to the
   smoke-arch body's evidence, and the closure ledger to the DIAGNOSIS's
   numbered fix items (F1/F2/F3) plus two cross-questions: would the fix
   have prevented the EXACT recorded failure (trace pool-construction
   EQUIVALENCE emit-time vs guard-time, incl. pinned provenance), and does
   anything NOW break for the cells that previously PASSED (fingerprint
   rollover consumers: resume / quarantine / stale HF uploads).

2. **Artifact commit postdating the marker.** The marker's `Commits:` line
   named only the code commit; the round's SECOND commit (regenerated
   committed data artifacts) was the ORCHESTRATOR's declared execution split
   (the diagnosis's Routing paragraph reserved manifest regeneration to the
   orchestrator). Verify marker-ts vs commit author-times at compose time
   and STATE the topology as a fact ("expected, not misreporting; commit 2
   fully in scope") — else an adversarial twin FAILs on "marker omits a
   round commit". Digest-only review for the artifact commit
   ([[data-hardening-round-compose]] item 1) with compose-time VERIFIED
   expected values (counts, tier arithmetic, containment 0-absent, recorded
   sha256 == actual, fp-suffix rollover old→new) so a mismatch is a real
   finding, not a re-derivation.

Also validated live: char-vs-byte length difference (len(prompt) vs `wc -c`)
is just multi-byte UTF-8 (∩ / — / ≥) — not a truncated write.

**How to apply:** any brief with "crash-fix r<k>" context + a
`round_parent`/round-HEAD pin where the impl marker is thin or the round
carries an orchestrator-side artifact commit. Related:
[[whole-round-unsplit-compose]], [[brief-pinned-sentinel-and-verdict-enum]].
