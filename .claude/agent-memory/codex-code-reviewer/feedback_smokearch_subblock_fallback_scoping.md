---
name: smokearch-subblock-fallback-scoping
description: "PASS_UNIFIED smoke-arch markers with FALLBACK in resume-matrix/production-outroot-unit SUB-BLOCKS are NOT verdict↔row inconsistencies — hand Codex the scoping fact; and a premature v1 + authoritative v2 impl-marker pair on a pre-split round gets a provenance note (#2617 r1)"
metadata:
  type: feedback
---

Two false-FAIL preemptions from #2617 r1 (2026-08-27), both recurring shapes:

1. **Smoke-arch FALLBACK vocabulary scoping.** Step 0.55's verdict↔row
   consistency rule ("a FALLBACK row under PASS_UNIFIED is a marker-shape
   blocker") binds on the `per-arm-resolution:` ROWS and the `arms_stubbed`
   list ONLY. The `resume-matrix:` and `production-outroot-unit:` SUB-BLOCKS
   use REAL / FALLBACK <reason> / N/A as their DESIGNED attestation
   vocabulary (Step 0.6's sub-check), so `production-outroot-unit: FALLBACK —
   <pod-side out-root reason>` under a PASS_UNIFIED verdict is an accepted
   form whose REASON gets adjudicated (FAIL/CONCERNS per Step 0.6), never an
   auto marker-shape FAIL. Any pod-side driver task will carry exactly this
   shape (its production out-root is /workspace + HF, not eval_results/), so
   an unscoped rubric copy invites a #489-class false FAIL every time. Add a
   SCOPING FACT note above the smoke-arch envelope.

2. **Premature-v1 + authoritative-v2 marker pair on a pre-split round.** A
   sanctioned pre-split build may post an impl marker at unit 1 (premature
   v1) and the authoritative full-round v2 later; `latest-marker` correctly
   returns v2. Probe events.jsonl: two `epm:experiment-implementation` rows,
   ZERO `[unit ` progress notes ⇒ single-marker variant (no progress-notes
   envelope), plus a PROVENANCE NOTE telling Codex the v1/v2 pair is round
   bookkeeping, v2 supersedes, and Step 0.5/0.6 score the inlined v2 only —
   otherwise an adversarial twin reads "supersedes the premature v1" as a
   shape defect.

**Why:** both are present-and-conforming evidence that pattern-matches a FAIL
trigger under a naive rubric copy; the composer owns the disambiguation
because Codex cannot read canonical task state.

**How to apply:** any `type:experiment` compose where the smoke-arch marker
carries sub-block FALLBACKs under a PASS verdict, or where >1 impl marker
version exists for one round. Related: [[whole-round-unsplit-compose]],
[[missing-impl-marker-probe-checklist]].
