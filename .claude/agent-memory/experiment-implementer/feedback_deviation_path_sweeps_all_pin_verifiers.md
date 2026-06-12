---
name: Deviation path activates -> sweep ALL pin verifiers
description: When a pre-authorized deviation (regenerated artifact differing from a parent pin) activates, every sibling helper that pinned against the OLD contract silently becomes a hard-blocker or a no-op; audit each pin source and check pins compare ACROSS the artifact boundary.
type: feedback
---

When implementing a pre-authorized deviation fallback (e.g. "proceed with the
regenerated Q-bank even though its sha mismatches the parent's pin"), grep for
EVERY other script that verifies the same pin and re-derive what each one's
contract should now be. Two failure shapes seen on task #556 round 3 (2026-06-10):

1. **Stale-contract hard-blocker:** `i556_pull_qbank.py` verified pulled banks
   against the PARENT's pins — valid only while the run-all hard-asserted
   new == parent. Once the deviation path made regenerated banks legitimate,
   the helper would have refused the VM judge on every file. Fix: re-point the
   pin source to THIS run's own attestation (`preflight_summary.json`).
2. **Vacuous within-side check mistaken for a guard:** the merge script's
   "q-coverage check" hashed the PARENT's rows against the PARENT's pin —
   parent-internal consistency that ALWAYS passes; the brief assumed it would
   refuse when this run's bank diverged. A pin guard only protects when it
   compares ACROSS the boundary (current-run sha vs parent sha). Had to add
   the real cross-boundary guard.

**Why:** pin checks are written under the no-deviation assumption; the deviation
round flips which comparisons are meaningful, and reviewers/briefs routinely
assume an existing check covers the new scenario when it compares the wrong pair.

**How to apply:** on any deviation-fallback round, `grep -rn "<pin file / sha
field>" scripts/` and classify each hit: (a) compares run-vs-parent (now a
deliberate refusal or a recorded deviation — decide which), (b) compares
parent-vs-parent or run-vs-run (internal consistency — keep, but never cite it
as the cross-boundary guard), (c) needs re-pointing to the run's own attestation.
Smoke each one in BOTH directions after the change.
