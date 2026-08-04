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

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Resume-aware phase-entry headroom gates](feedback_resume_aware_phase_headroom_gates.md) — a blanket fresh-run floor at phase entry deadlocks a resume whose own done artifacts occupy the disk; skip/scale need by PENDING cells, wave-gate template (#1586 fu crash 5)
- [Mid-run Hub-verified ckpt reap + restage-on-missing](feedback_midrun_verified_upload_ckpt_reap.md) — keep-for-later checkpoints are re-stageable duplicates once upload-verified; fleet of completed cells starves the next phase's headroom floor; reap on verified upload, restage via the single resolver (#1586 fu r5)
- [Chained smoke-then-full out-root residue](feedback_chained_smoke_leg_out_root_residue.md) — smoke && full per-leg out-roots leave the smoke leg's keep-cell rungs unowned (~44 GB) and starve the full leg's headroom assert; full leg reaps the DERIVED sibling smoke root at first phase entry (#1586 fu r3)
