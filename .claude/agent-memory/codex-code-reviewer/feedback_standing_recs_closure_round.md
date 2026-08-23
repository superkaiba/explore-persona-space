---
name: standing-recs-closure-round
description: "Post-reconciler-PASS rounds that IMPLEMENT the standing recommendations: acceptance contract = the reconcile verdict's 'Standing recommendations on PASS' section (often surviving at /tmp, named in the impl marker); author-neutrality + no-relitigate-the-refuted-IMPACTS while grading the recommended hardening's closure (#2479 r8)"
metadata:
  type: feedback
---

The COMPLEMENT of the #2329-r4 entry (standing recommendation NOT implemented
→ CONCERN row, never FAIL): when the next round DOES implement the reconciler's
standing recommendations after a binding PASS, compose it as a
closure-verification round (first hit: #2479 r8, 2026-08-23).

**How to apply:**
1. **Acceptance contract = the reconcile verdict itself.** Its "Standing
   recommendations on PASS" section is the round's contract — inline the FULL
   verdict in its own envelope (`epm:review-reconcile` is a different marker
   kind from yours: no extraction collision; assert its head/close tags ==1
   each and tell Codex never to reproduce THEM). The verdict file often
   survives at the /tmp path the impl marker names
   (`/tmp/issue-2479-reconcile-r7.md`) — verify content (contract section +
   the measured numbers) before trusting it.
2. **Author-neutrality + no-relitigate, split by half.** The twin authored the
   deferred BLOCKERs; the reconciler credited the MECHANISMS and refuted the
   IMPACTS by measurement. State both halves explicitly: grade the
   recommended HARDENING's closure (full duties), never re-raise the refuted
   impact claims without NEW evidence from THIS diff (quote the refuting
   measurements: min margin 790 vs max delta +16, budget 3008).
3. **Ledger topology after a severity-downgrade defer:** raised(B) →
   deferred(B, reconciler rationale) → re-raised(C). Exactly ONE open concern
   (latest event raised at CONCERN) = the per-concern status-line subject
   (NOT-ADDRESSED = substantive FAIL, the fix IS the round); the deferred
   BLOCKER ids get closure adjudication through the numbered-recommendation
   ledger items (R1/R2/...), NEVER fresh `CONCERN:: ` rows. Inline the 5
   round-relevant rows with `evidence` + `deferral_rationale` fields (they ARE
   the acceptance criteria); pin the snapshot to the impl-marker ts.
4. **Key the closure ledger to the recommendations' own numbering** (R1..Rn =
   the verdict's numbered list) + the brief's sharp questions as Q-items with
   NEUTRAL two-way status vocabularies (BOUND-STRUCTURAL | DRIFT-PATH,
   ADEQUATE | RESIDUAL, SOUND | REGRESSION) — severity per consequence,
   pre-resolved nothing.
5. **Artifact-regen identity claims are compose-time verifiable:** when the
   round regenerates committed JSONs claiming "id sets byte-identical, only
   provenance changed", RUN the identity comparison at compose time (git show
   parent-blob vs working copy, per-array ==) and state the result as a
   verified fact; give Codex the same one-liner as a digest duty. A provenance-
   only regen still ROLLS the manifest-sha fingerprint — make the fp-rollover
   consumer trace (resume / quarantine / old-sha bounded grep) an explicit
   Q-item; the roll's COST was the orchestrator's accepted price (note, never
   relitigate).
6. Envelope-scoped zero-asserts again (#1090-r5 rule): the prior round-parent
   SHA legitimately rides the reconcile envelope — assert
   `prompt.count(sha) == envelope-side count`, not ==0.

Related: [[crashfix-thin-marker-orchestrator-artifact-commit]] (the r7 shape
this round follows), [[revision-round compose recipe (round 2+)]] (#2329-r4
entry = the not-implemented complement; #2332-r2 = memory-write hygiene:
leave this file uncommitted in the worktree, flag for post-merge sweep).
