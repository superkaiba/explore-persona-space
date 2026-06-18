---
name: Shared-writer/per-variant-reader filename drift + sibling cap-asymmetry
description: Two recurring multi-variant-dispatcher bugs Codex catches and Claude misses — a SHARED writer's filename vs a newest-variant's bespoke reader, and a positive-pass cap copy-pasted to a negative pass without the cap.
type: feedback
---

Two distinct bugs in multi-variant dispatchers (v3/v4/v9, smoke/production,
pilot/full) that ship green and get caught only by the Codex twin. Both bit
#642 v9 round 1 (reconcile B1 + B2).

**1. Shared writer, per-variant bespoke reader → filename divergence on a
REGISTERED kill/gate outcome.** When a SHARED phase function writes an outcome
file (`install_failure.json`, a selection/verdict/sentinel) and a NEWEST variant
adds its OWN reader keyed on a DIFFERENT literal (`install_failure_{arm}.json`),
the writer's file is never found → the reader returns None → the pipeline falls
through to the full path on a killed run (confusing downstream crash instead of
the clean `verdict=KILLED`).

- **Why:** the "is it written + plumbed?" walk passes — the writer exists, the
  reader exists, both compile. The defect is the LITERAL mismatch across
  variants of the same dispatcher.
- **How to apply:** when a dispatcher has N readers and a SHARED writer, check
  that EACH reader's literal matches what the writer produces ON THAT variant's
  path. The tell is an INTERNAL-CONSISTENCY ASYMMETRY: a sibling variant (v4's
  `_install_failure_report`) reads the no-suffix name and matches its own writer;
  only the newest variant's writer/reader pair is mismatched. Also check
  pilot-vs-production write paths SEPARATELY — the pilot path wrote the per-arm
  name and "looked covered", masking the production-gate no-suffix write.
- **Fix shape:** branch the writer on the variant flag (`if ctx.v9: write
  per-arm` else single-file) so each variant's writer matches its reader; make
  the resume-skip check variant-aware too. Pin with a test that materializes the
  real producer file and asserts the consumer short-circuits WITHOUT calling the
  full-analysis fn (tripwire), plus a test that the OLD filename is invisible to
  the new reader.

**2. Cap-asymmetry: positive pass capped, sibling negative pass not.** When one
assembly branch over-produces then HARD-caps (`kept = items[:min(n, target)]`)
and a sibling branch copy-pastes the SAME over-produce slack (`per_q =
ceil(target/q)+1`) but OMITS the cap, the sibling silently overshoots. #642:
positives capped at `n_positives`, negatives appended every accepted row → the
realized 1:neg_ratio contrastive ratio drifted to 1:3.75-6.75 vs planned 1:2.5,
breaking the dose contract + the recipe-match to the anchor.

- **How to apply:** grep the file for every `[:min(` / `[:N]` / `total_*_target`
  and confirm EACH over-elicited pool has a matching trim. The `+1` slack comment
  ("ceil + 1 slack") on the positive line is the tell the SAME slack was pasted
  into the negative line — slack WITHOUT a downstream cap is overshoot, not
  headroom. Compute the realized ratio with the dispatcher's ACTUAL default
  constants — never trust a `_target` variable name; verify it's enforced.
- **Fix shape:** extract a pure `_cap(...)` helper (per-slot budget + global total
  cap, deterministic shuffle, slot-balanced tail-drop) the negative pass calls;
  re-emit the per-slot counts POST-cap so the log matches the persisted pool.
  Unit-test the cap on a fake-accepted map with the real defaults: assert
  `len <= round(ratio*n_pos) + n_slots`.
