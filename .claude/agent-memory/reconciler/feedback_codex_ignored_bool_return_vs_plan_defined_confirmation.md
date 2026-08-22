---
name: codex-ignored-bool-return-vs-plan-defined-confirmation
description: Codex FAILs on a caller ignoring a helper's declared `-> bool` result made load-bearing by the round; adjudicate by (1) does the plan DEFINE the confirmation term as this exact branch, (2) can the non-raising False actually fire given the API layer's raise semantics (#2184 r1)
metadata:
  type: feedback
---

Codex flagged (BLOCKER, #2184 r1): `_teardown_failed_provision` returns
"terminated" whenever `terminate_pod()` does not raise, ignoring its declared
`-> bool` ("Returns True on success"); the disposition licenses a rotation
re-create, so a non-raising False could create a pod beside a billing one.
Claude PASSed. Adjudicated PASS.

**Why:** Two checks defeated verdict-changing status, both generalizable:

1. **Plan-defined semantics beat a helper's declared type contract.** Plan v3
   MF-A itself prescribed `successful terminate (:2434) → return "terminated"`
   at the exact pre-existing non-raise branch, with "no new terminate_pod call
   site", and DEFINED "CONFIRMED termination" as "the threaded teardown
   disposition reads 'terminated'". When the plan's Must-Fix text defines the
   confirmation term as the disposition from that branch, "the caller ignores
   the bool" is a critique of the APPROVED design, not a plan violation — the
   inverse of [[plan-verbatim-text-vs-plan-binding-mustfix]] (there §4
   compliance lost to MF text; here the MF text itself sanctioned the
   placement). Read the plan's DEFINITION of the disputed term before
   crediting a "violates the interlock contract" FAIL.

2. **Reachability of the non-raising failure value.** `terminate_pod` returns
   False only on an undocumented `{"podTerminate": <non-null, non-True>}`
   no-errors response: `graphql()` raises on GraphQL-level errors +
   non-transient 4xx + exhausted transports, the approval gate raises, and
   podTerminate is a Void-returning mutation ("returns null on success;
   errors raise above"). Every REAL failure raises → the fail-closed
   "failed" arm → test-pinned refusal. Sibling of
   [[codex-blocker-on-unreachable-exception-path]] — trace the wrapped API
   layer's raise semantics before upholding an ignored-return-value blocker.

**How to apply:** For any "ignored boolean/status return made load-bearing"
FAIL: (a) grep the plan for a DEFINITION of the confirmation/success term —
if the plan maps the term to this branch, the finding is design-hardening,
not a defect; (b) read the callee + its transport layer: if all real failures
RAISE and the falsy return needs an undocumented provider response, classify
Real-nonblocking, defer-downgrade the BLOCKER (`defer-concern --by
reconciler`), and carry the hardening (prefer boundary-wide success-or-raise
covering ALL sibling callers) as a standing recommendation — jamming a local
`is True` check into the round diverges from the eventual boundary fix.
Cost-bound the hypothetical too (#2184: one extra CPU pod, caught by the
stale-pod audit — bounded billing, not corruption).
