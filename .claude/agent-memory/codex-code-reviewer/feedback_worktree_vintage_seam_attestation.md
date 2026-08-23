---
name: worktree-vintage seam attestation
description: When a marker/brief asserts a library-seam fact (kwarg presence, signature, routing predicate) about src/ code, verify it against the WORKTREE's copy — branch vintage binds at runtime; main may have drifted the other way (#2254 r4)
metadata:
  type: feedback
---

When the implementation marker (or the dispatch brief) asserts a fact about
a `src/` library seam — "judge_graded doesn't thread force_sync", a
signature, a routing predicate — verify it at compose time against the
WORKTREE BRANCH's copy of the file, never the main checkout's. The branch's
vintage is what the driver imports at runtime; main may have gained the
very feature the marker says is absent.

**Why:** hit live on #2254 r4 (2026-08-23, remediation round). The marker
justified `threshold_base=10**9` as the sync lever "because judge_graded
doesn't thread force_sync". MAIN's `graded_judge.py:253` HAS `force_sync`
(added after the branch cut); the WORKTREE's copy (:241-252) has
`threshold_base` only — the marker was TRUE on-branch. Grading from main's
copy would have produced a false doc-accuracy finding AND a wrong-direction
fix suggestion ("just pass force_sync" — unavailable on the branch, and a
src/ edit is out of a scoped round anyway).

**How to apply:** (1) grep BOTH copies when a seam fact is load-bearing;
(2) attest the branch-verified facts in a "Composer-verified seam facts
(SETTLED)" prompt block — include the main-drift note ("main has since
gained X — irrelevant; do not read src/ from the main checkout") so the
twin neither false-flags the marker nor proposes the unavailable kwarg;
(3) leave the THREADING verification (every call site actually passes the
lever; the lever cannot leak into non-target sites) to the twin — attest
facts, never conclusions. Companion round shape (#2254 r4): a MID-RUN
remediation round triggered by a run-phase pre-registered kill composes
with the orchestrator's `epm:progress` DECISION note as the round contract
(own envelope; facts settled, implementation adjudication open), its own
per-round impl marker (fetch + round-match as usual), and post-reconciler
no-relitigate blocks from the PRIOR round's binding PASS. Related:
[[revision-round compose recipe]], [[deferred event: two semantics]].
