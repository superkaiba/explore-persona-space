---
name: Phase-entry disk-headroom gates must be resume-aware
description: "A blanket fresh-run headroom floor at phase entry deadlocks a resume whose own done artifacts occupy the disk — the gate demands space for work that won't run and blocks the later phase that would free it; skip/scale need_gb by PENDING cells (#1586 fu crash 5)"
type: feedback
---

A phase-entry disk-headroom gate (`assert_out_root_headroom` with the
plan-§9 blanket floor) that runs BEFORE the phase's own resume scan
deadlocks a resumed run deterministically: the run's own resume-done
artifacts legitimately occupy the disk, the gate demands the FRESH-run
floor for work that will not run, and it blocks the very later phase
whose reclaim/wipe arms would free the space. #1586 fu crash 5: `p2_train`
entry demanded 60 GB with 50.8 free while the wave-level gate had already
correctly skipped (0 pending / 1 done) — dead ~0.1 s in, on every respawn.

**Why:** floors are sized for fresh execution; resume changes the
pending-work denominator, not the per-cell arithmetic.

**How to apply:** every phase-entry headroom gate fronting per-cell work
computes the phase's PENDING set with the same predicates the phase's own
resume scan uses — zero pending ⇒ skip the gate (one INFO line); partial
⇒ scale need to the pending subset (per-cell need × n_pending; constants
untouched). Fresh runs must compute byte-identical need (pin it). The
wave-level pending-aware gate is usually already the in-file template.
(#1586 fu round 8.)
