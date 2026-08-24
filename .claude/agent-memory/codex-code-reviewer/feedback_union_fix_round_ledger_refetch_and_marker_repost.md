---
name: union-fix-round ledger refetch + marker-repost U-item
description: "On any template reuse NEVER trust the prior prompt's open-concerns section (re-fetch with list-concerns --open-only — #2476 cr8 said '0 OPEN' while 12 were open); a marker-RE-POST U-item (U8 smoke-arch grammar) composes as inlined re-posted marker + compose-time checker attestation; binary-verdict fix rounds encode the U-vs-M severity fence in the verdict-line preamble (#2476 cr9)"
metadata:
  type: feedback
---

Three deltas from the #2476 k200 r2 union-fix compose (cr9, 2026-08-24):

1. **Re-fetch the open-concern set EVERY compose — never carry the prior
   template's ledger section.** The same-round cr8 prompt (composed hours
   earlier) stated "90 ledger rows total, 0 OPEN"; the compose-time
   `task.py list-concerns <N> --open-only` for cr9 showed 12 of those rows
   OPEN (latest event `raised`) plus the 10 new r8 rows = 22. A stale/wrong
   ledger line silently mis-arms the twin's Step 0.8 (`**Prior-concerns
   ledger:** empty` when it is not). `--open-only` is the authoritative
   filter — do not re-derive open-ness from raw events.jsonl heuristics.
   **Why:** the ledger section is round-varying state, not template text.
   **How to apply:** at every compose, run `--open-only`, split the ids
   three ways (round-contract closure items / brief-named adjudication
   targets / inherited-do-not-relitigate) and put the full id list in the
   required ledger header line.

2. **A marker-RE-POST U-item** (bounce U8: re-post
   `epm:smoke-architecture-check` in the accepted `source=/file=/n=/members=`
   grammar) composes as: inline the RE-POSTED marker version in the Step
   0.55 envelope + a composer attestation of the compose-time checker run
   (`task.py check-smoke-arch-registry <N>` → rc + verdict line quoted).
   Codex scores the grammar LINE + member-list-vs-code from the inlined
   body (it can read sorted(PHASES) in the driver); the checker acceptance
   is attested, never re-run by the twin.

3. **Binary-verdict fix round severity fence goes IN the verdict-line
   preamble:** "A NOT-ADDRESSED U-item is a blocker; a NOT-FOLDED M-item or
   a REMAINS-OPEN adjudication target is NOT." Without it the binary enum
   (no CONCERNS) invites either softening a dead U-item to PASS or
   escalating a non-blocking M/adjudication residual to FAIL.

Related: [[revision-round compose recipe]] (#2332 r2 union shape),
[[brief-pinned-sentinel-and-verdict-enum]], [[brief-named-concern-adjudication]].
