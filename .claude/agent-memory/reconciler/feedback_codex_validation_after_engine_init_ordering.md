---
name: codex-validation-after-engine-init-ordering
description: Codex FAILs on "input validation runs AFTER engine/venv init" (consumer-contract-post-init) — adjudicate by gate topology, not statement order; three checks decide efficiency-concern vs certification defect (#2378 r12)
metadata:
  type: feedback
---

Codex code-review FAIL shape: a resume/consumer phase calls
`ensure_model_venv`-class init (a REAL engine smoke) BEFORE
staging/validating its reused inputs, framed as "count-perfect stale
artifacts could certify the wrong state". The ordering fact is usually
REAL; whether it blocks turns on three checks:

1. **Trace where the fail-louds fire relative to the SPEND and the
   VERDICT WRITE.** If every input contract (count asserts, ledger
   existence, skip-asserts) raises loudly BEFORE the expensive step (GPU
   capture/generation fan-out) and before any digest/verdict/sentinel
   write, nothing wrong is CERTIFIED — the only cost is the init wall
   (~minutes) on the failure path. A fail-loud that fires late is still
   fail-loud; silent certification is the blocker class, wasted cycles
   are not. Bonus check: on the PASS path the init is often prerequisite
   for the step that follows validation — then nothing is wasted at all.
2. **Separate ordering from validation STRENGTH.** "Count-only checks
   admit stale files" is orthogonal to WHERE the checks sit — reordering
   changes nothing about what a count check can admit. Judge the
   strength claim on its own reachability (fresh-pod empty root + staged
   from an upload-verified pinned prefix + malformed JSON crashing
   loudly downstream usually closes the channel).
3. **Parent-SHA the ordering.** In #2378 r12, Codex's second Critical
   flagged phase_p2's guard-after-init — but `git show <parent>:<file>`
   showed the init-then-read ordering byte-identical pre-round; the diff
   only changed the block INTERIOR. Pre-existing structure = the
   git-provenance family, never a round blocker.

**Why:** #2378 r12 (second Codex FAIL on the same leg after r10's
walls-clobber — see [[codex-fails-preexisting-resume-metadata-clobber]]).
Verdict PASS; both BLOCKER rows downgraded via
`task.py defer-concern --by reconciler` (the sanctioned ensemble-tie-break
severity-downgrade channel), reorder + schema checks left as standing
recommendations.

**How to apply:** on any `*-post-init` / validate-late Codex blocker, run
checks 1-3 against the artifact; all pointing at wasted-cycle-only ⇒ PASS
+ defer-concern the persisted BLOCKER rows with a ≥40-char rationale
(they are already in concerns.jsonl via the CONCERN:: forwarding, and an
open BLOCKER gates dispatch — the downgrade must be persisted, not just
narrated). Related: [[codex-hardening-beyond-minimal-port-contract]],
[[codex-step-06-literal-vs-purpose]] (gate topology as the decisive
variable).
