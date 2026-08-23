---
name: closure-round-summary-only-ledger
description: "Forwarder-persisted ledger rows carry SUMMARY only (no evidence field) — closure acceptance contracts must come from inlining BOTH prior verdicts; plus the 0.55 absent→present flip duties and the ACCEPTED-SCOPED-DEFERRAL third status (#2474 r2)"
metadata:
  type: feedback
---

Three composable deltas from the #2474 r2 closure compose (2026-08-23; r1 was
Claude-split-FAIL + Codex-FAIL, orchestrator UNIONED, all findings persisted
via `persist_verdict_concerns.py`):

1. **Forwarder-persisted concern rows have NO `evidence` field.** Rows minted
   by the blind forwarder from `CONCERN:: ` grammar rows carry only
   ts/event/severity/summary/raised_by/raised_at_round — the recipe's "restate
   the concern's `evidence` field acceptance criteria" step CANNOT be
   satisfied from the ledger. The acceptance contracts are the prior verdicts
   themselves: inline BOTH (tags stripped per #2332; the Codex one's
   `CONCERN:: ` rows blockquoted per #2329 rclose) even though the ledger is
   non-empty. Claude-only findings not separately persisted still get
   pseudo-ids (#1092-r4) keyed to the inlined Claude verdict text.

2. **Step 0.55 absent→present flip (r1 FAILed on marker absence, r2 marker
   now posted):** replace the `ABSENT in canonical task state` literal with
   the full marker body, state presence-SATISFIED (presence-ON-TASK, any
   version), and state that a present `PASS_PARTIAL`/FALLBACK-row verdict is
   the orchestrator's Step 6d.0 adjudication — at most CONCERNS, never a
   reviewer FAIL. Hand the marker's own registry claim (`arm-registry:
   source=sorted(PHASES) n=K` + the `check-smoke-arch-registry` OK line) to
   Codex as a static verification priority against the driver's dict.

3. **ACCEPTED-SCOPED-DEFERRAL as a third closure status.** When the
   orchestrator's brief SCOPES a concern's closure standard (here
   `upload-smoke-not-real`: real Hub write deferred to the pod-side P-B run;
   standard = dry-run enumeration + executed misuse path + test-pinned
   sentinel-after-verify + FALLBACK row + same-issue production evidence for
   the reused helper pair), spell the scoped standard out verbatim in the
   prompt and add `ACCEPTED-SCOPED-DEFERRAL` beside
   VERIFIED-ADDRESSED/NOT-ADDRESSED — else the twin (author of the original
   BLOCKER) re-FAILs on the deliberately-deferred real write. Pair with the
   #1092-r2 guarded-closure three-way for the production-path arming
   question.

Also confirmed live: an unbriefed intermediate DATA-artifact commit between
the r1-reviewed tip and the round commit (here `6adad25450`,
eval_results-only) follows the #2329-r4 rule — out-of-round per the brief,
readable as consumer-input context, flagged in the return; and
`git show <round-sha>~1:<path>` is the correct pre-fix blob for
fails-pre-fix pin-test adjudication when the intermediate commits touched no
scripts. Related: [[revision-round compose recipe]],
[[concerns-machine-rows-2326]].

**r3 delta (2026-08-23): split-verdict ledger state on a FAIL+FAIL union.**
When the Claude reviewer flipped the prior slugs to `addressed` during its
review while the Codex twin's same-round ledger marked several of them
NOT-ADDRESSED, the canonical ledger and the twin's own verdict DISAGREE.
Compose an explicit **ledger-mapping rule**: name the disputed slugs, state
that the union resolution carries the twin's residual disputes forward as
the round's pseudo-id items (C2/M1-M4 shape), and instruct "do NOT re-open
the addressed slugs as such — grade the pseudo-ids" (else the twin re-flips
its own r2 ledger lines and double-counts). Also: when the round tip == HEAD
and the prompt states "HEAD == <tip>, worktree file state == post-round
state", that claim is LOAD-BEARING — do not commit an agent-memory edit in
the reviewed worktree mid-round (it advances HEAD past the tip and falsifies
the prompt); leave it uncommitted and flag for the orchestrator's post-merge
sweep (#2332-r2 precedent).
