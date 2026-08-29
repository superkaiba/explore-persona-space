---
name: closure-verification round compose (FAIL+FAIL union fix round)
description: Round-2 composes after a FAIL+FAIL-disjoint union — inline the orchestrator's ADJUDICATION as the contract (not the two verdicts), add a Closure ledger enum to the verdict template, recover unpersisted round-1 CONCERN rows, and verify the deferral target task exists
metadata:
  type: feedback
---

When the brief says round 1 FAILed both twins and the orchestrator unioned
the blockers into a numbered list, the round is a CLOSURE-VERIFICATION
round, not a re-review. Compose it that way.

**Why:** the twin's default posture is to re-derive its own findings, which
on a fix round means re-litigating code that is already closed or already
dispositioned. #2384 r2 was 12 blockers wide; without an explicit closure
frame the twin answers the two or three interesting ones and silently drops
the bookkeeping half.

**How to apply.**

1. **Inline the ADJUDICATION, not the verdicts.** The orchestrator's
   `epm:progress` union note is the compact authoritative contract (#2384:
   6 KB, vs 24 KB Codex + 12 KB Claude verdicts). It already carries the
   orchestrator's INDEPENDENT confirmation of each Critical, so the twin
   never has to re-argue round 1. Envelope it
   (`---BEGIN ROUND-1 ADJUDICATION---`) and say explicitly: its FACTS are
   settled, its DISPOSITIONS are the contract.
2. **Add a `## Closure ledger` section to the verdict template** with one
   line per adjudicated blocker and a 4-way enum — CLOSED /
   CLOSED-DIFFERENTLY / NOT-CLOSED / UNVERIFIABLE-STATICALLY — plus a
   `**Blockers closed:** N of M ...` header field. Without the per-item row
   the twin deep-dives the brief's named targets and leaves the cheap
   blockers (documented-exit-code, cap-disclosure, render-guard) unstated.
   State the severity asymmetry in the FAIL-validity backstop: NOT-CLOSED
   is FAIL-grade at the blocker's ORIGINAL severity; "closed by a mechanism
   I'd have chosen differently" is NOT a finding; a closure whose TEST is
   vacuous IS a finding.
3. **Recover unpersisted prior-round `CONCERN::` rows.** `list-concerns`
   can be EMPTY while the prior round's verdict emitted rows — the
   forwarder never ran (#2384: 10 Codex rows, no `concerns.jsonl` at all).
   Extract them from the marker prose, MAP each onto its adjudicated
   blocker number, and NAME the ones that map to nothing: those were closed
   by no process at all and need an explicit "settle on its own merits, and
   RE-EMIT with the same kebab-case id or it stays unpersisted a second
   time" instruction. (#2384's two orphans: a plan-kill-criterion
   calibration row and a lint-unconfirmed row.)
4. **Verify a routed-out blocker's target task EXISTS.** When the round
   defers a blocker to a new task, `task.py view <id>` it and attest id +
   status + title. That converts "the implementer says they filed it" into
   a compose-time fact and lets you fence the deferral as SETTLED
   (#2384 blocker 5 -> #2641, `planning`). Frame the twin's job as
   "assess the SPLIT" with named live questions (is the local closure
   complete, does the duplicated implementation create a drift hazard, is
   this round's own blast radius still exposed) — not "the shared fix is
   missing".
5. **Measure a REWRITTEN pin's strength at compose time.** A round that
   line-wraps a docstring enum often replaces one literal assert with two
   fragment asserts. Count each fragment's occurrences in the live target
   (`ast.get_docstring` + `.count()`), hand the counts, and name the
   hazard the counts do NOT settle: unanchored short fragments
   (`"75)"` in a 31 KB docstring) and two asserts that never check
   ADJACENCY. Leave the weakened-vs-acceptable call to the twin.
6. **`--name-status` over the ROUND range disarms #1805** even when the
   file was `A` earlier on the same branch — a round-2 delta of all-`M`
   files owes no round-new-script waiver. See
   [[new-helpers-not-new-file-1805]] for the M-vs-A rule itself.

**Round-scoping and marker-shape both get pre-settled.** Ban the
whole-branch three-dot BODY (already reviewed), keep name-only/`--stat`
and whole-FILE-at-HEAD reads unrestricted (a fix hunk is usually
uninterpretable without its enclosing function), and route out-of-round
findings through Step 0.9 `pre-existing-on-trunk`. When the brief attests
the new marker is four-H3 conforming, say so as a compose-time fact and
forbid the `marker-shape` blocker outright — see
[[revision-round-compose-recipe]] for the general round-2 deltas.
