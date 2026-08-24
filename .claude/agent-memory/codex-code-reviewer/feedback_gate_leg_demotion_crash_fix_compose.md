---
name: gate-leg demotion crash-fix compose
description: Crash-fix rounds that DEMOTE a plan-registered gate leg (assert -> reported) compose with a composer-run reference sweep handed to the twin, a removal-vs-widening framing note, and an enforcement-surface adjudication line for any concern deferred TO the edited gate (#823 r6cf)
metadata:
  type: feedback
---

When a crash-fix round's fix is DEMOTING a plan-registered gate leg (a
binding assert becomes reported-never-asserted; #823 r6cf: probe (f)'s
"median max-rel <= 1e-2" dropped from the rc-16 verdict after bf16
near-zero-denominator blowup), three compose deltas beyond the standard
#2329-r6 crash-fix shape (own impl marker + crash-diagnosis envelope):

**Why:** the demotion's justification rests on a NEGATIVE claim ("no
committed same-surface reference exists to derive a replacement binding
leg") that the twin cannot efficiently sweep for, and the gate being
weakened may be the very enforcement surface a deferred ledger concern
points at — both need composer legwork at compose time.

**How to apply:**
1. **Run the reference sweep yourself, hand the LIST, keep the
   adjudication theirs.** Grep the worktree for the candidate-metric tokens
   (`rel_l2|rel-L2|relative L2`) outside the round file, list the hit files
   verbatim in the facts block, and state explicitly "the composer did NOT
   adjudicate same-surface-ness — that is YOUR JOB n sub-check; the marker
   claims none is". Attest facts, never the conclusion (the
   [[worktree-vintage seam attestation]] rule applied to an absence claim).
2. **Removal-vs-widening framing.** A plan clause like "abort, never a
   tolerance to widen" is NOT self-evidently violated by leg REMOVAL (no
   tolerance was widened; the leg left the verdict). Put the distinction in
   the plan-intro as an open adjudication ("whether removal-vs-widening is
   a distinction with substance is part of your adjudication") — neither
   pre-resolve it nor let the twin auto-FAIL on the clause's letter.
3. **Deferred-concern enforcement-surface overlap.** A ledger concern
   deferred TO the run-stage gate this diff EDITS gets more than the usual
   one-consistency-line: route it to a crash-fix ledger item ("does the
   smoke chain remain a real enforcement point after the demotion — cosine
   halt + rc + genuine-seam detection test-pinned"), still never a closure
   or a re-emitted row.
4. Verify the cited exemplar + calibration-rule text at compose time
   (equivalence_gate_p2 really asserts cosine-only; artifact-reuse rule 2
   really says no-reference => WARN/report) and attest line numbers — the
   demotion argument is only as good as its citations.
5. PASS+PASS prior round => no prior-verdict envelopes, no closure duties,
   no author-neutrality block; the crash diagnosis note is the sole
   acceptance contract (its routing items become the plan-adherence frame).
   Cheaper prompt (~255 KB vs ~275 KB with verdicts).
