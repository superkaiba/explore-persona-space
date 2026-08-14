---
name: staging-gate-single-phase-silent-fallback
description: A sha-pin staging gate enforced only in the staging PHASE leaves later consumer phases (separate processes / fresh pod B) calling a loader whose cache-miss fallback silently REGENERATES the pinned input — trace every transitive loader call site per phase (#2254 R1 g2)
metadata:
  type: feedback
---

When a plan promises "assert sha256 BEFORE any <loader> call — the regen
fallback is made unreachable", check the promise per PHASE, not per driver:
phase-dispatch drivers run each phase as a separate process invocation, and
resume/relaunch (or a designed fresh pod B) can enter a consumer phase on a
filesystem where the staging phase never ran. If the shared loader's
cache-miss branch is `except FileNotFoundError: regenerate()` at logger.info
level (the #779 `load_e1_assets` shape), the bank-identity invariant silently
breaks — downstream length/shape asserts PASS on the regenerated artifact
because regeneration reproduces the SHAPE, not the CONTENT.

**Why:** #2254 R1 g2 — `phase_stage_inputs` sha-asserted the e1 asset JSONs,
but `phase_capture_directions` (feeds a REGISTERED direction) and
`phase_norm_probe` called `load_e1_assets` unguarded; plan §9's fresh pod-B
phases inherited the same unguarded loader. The A8 disjointness gate had run
only in stage_inputs against the OLD bank. Rated Major: the happy path was
correct, the exposure was every resume/fresh-pod path.

**How to apply:** for each pinned staged input, grep the driver for every
transitive call site of its loader; group call sites by PHASE; each phase
lacking a presence+sha re-assert is exposed. Fix shape: a tiny
`_assert_<input>_staged()` (present + `assert_sha256` vs the pin dict) at the
top of every consumer phase or inside the driver-local wrapper so future
phases inherit it. Check the loader's except-clause yourself — a fallback
that logs at info and returns shape-conforming output is the silent class.
Related: [[checkpoint-consumer-skips-key-check]],
[[start-manifest-stale-artifact-done]] (stale variant; this is the
regenerated-fresh variant).
