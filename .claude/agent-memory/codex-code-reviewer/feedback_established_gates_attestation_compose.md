---
name: established-gates-attestation-compose
description: When the brief supplies already-run gate results (no-flags lint, mapped-test union) with a named pre-existing red, attest them as compose-time facts with the offender's provenance and instruct no re-run + no-round-defect + no-GRANDFATHERED-proposal — and re-verify each numeric claim yourself before attesting
metadata:
  type: feedback
---

When the orchestrator's brief states gate results as ALREADY ESTABLISHED
(first hit #2241 r1, 2026-08-20: no-flags workflow_lint rc=0 with 27 WARNs
incl. transient hook-timeout non-verdicts; a 155-file mapped-test union with
exactly one failure that is pre-existing origin/main red):

1. Attest them in the compose-time-facts block with (a) the log paths the
   brief names, (b) the pre-existing offender's provenance (file, landing
   commit, date, "absent from this diff", "round touches neither the test
   nor its GRANDFATHERED list"), and (c) the explicit git-provenance
   subclass (`pre-existing-on-trunk`) so Codex's Step 0.9 routing is
   pre-resolved.
2. Instruct THREE prohibitions explicitly: do not re-run the gates; do not
   flag the pre-existing failure as a round defect; do not propose
   GRANDFATHERED/allowlist additions for it (an adversarial twin's favorite
   scope-creep remedy).
3. Re-verify what is cheaply verifiable before attesting: cap arithmetic
   (((measured+2_800)//100)*100 idiom), headroom floor, name-status M-vs-A
   (decides the #1805 duty), roster membership, marker substring presence.
   The attestation is YOUR claim once composed — a wrong brief number
   becomes your false fact.
4. The `**Tests actually run:**` template line gets the fixed form
   `no — Codex static review; gate results composer-attested`, and
   `**Lint:**` gets `PASS (composer-attested rc=0)` with a contradiction
   escape ("FAIL only if your static read contradicts the attestation").

**Why:** without the provenance + prohibitions, an adversarial Codex reads
the union's 1 failure as a round blocker (the #521-class false FAIL) or
burns effort re-running gates it cannot run (no uv env), and a stale
attested number becomes a composer-injected false fact.

**How to apply:** any brief carrying "gate results to state as ALREADY
ESTABLISHED" or pre-run lint/union logs. Related:
[[infra-wf-fix-lint-gate-compose]], [[bypath-brief-frozen-events-resolution]].
