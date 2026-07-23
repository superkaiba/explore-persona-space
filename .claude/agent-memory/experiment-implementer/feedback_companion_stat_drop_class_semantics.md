---
name: companion-stat-drop-class-semantics
description: Zero split-half noise floors at early answer positions (shared first tokens) must be a NAMED non-fatal exclusion class — a fail-loud drop guard calibrated for the registered statistic false-fires on a companion's legitimate degeneracies
type: feedback
---

Split-half noise floors of teacher-forced activations are EXACTLY zero at early
answer positions whenever all of a cell's draws share their first tokens
(deterministic-opening completions are common — 7/28 pairs on #1415), so any
log-ratio or guard over floor magnitudes must treat non-positive means as a
NAMED non-fatal exclusion class (`dropped_nonpositive_pairs`), never route them
into an all-NaN data-integrity guard. **Why:** #1415 position-profile round —
the Δ_floor companion's zero floors tripped the >20%-drop fail-loud guard
calibrated for the registered Δ; the pod run died rc=1 mid-p3 (GCE att-20260722-222513)
on a healthy dataset. **How to apply:** every COMPANION statistic (noise-floor
ratios, sensitivity subsets) gets its OWN drop-class semantics + its own guard
threshold; fails-pre-fix regression test with a shared-first-token fixture.
