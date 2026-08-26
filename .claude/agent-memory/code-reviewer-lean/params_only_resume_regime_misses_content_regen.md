---
name: params-only-resume-regime-misses-content-regen
description: A fail-loud regime-keyed resume ledger still no-ops onto a stale store when upstream rows are REGENERATED at the same count/ids — check the regime for an input-content digest (#2378 R1 g2)
metadata:
  type: feedback
---

A chunk-resume ledger keyed on generating PARAMETERS + counts (the correct
#1336 machine-stable shape, fail-loud on mismatch) is still blind to
INPUT-CONTENT drift: an upstream regen that rewrites row text for the SAME
row ids at the same kept count (the plan's own cap-hit ">2% ⇒ regen at 2×"
trigger is exactly this shape) leaves the regime equal, so a re-run finds
every chunk done, no-ops, and reprints "captured N/M" from stale rows.json —
the stale-artifact-done class wearing a fail-loud ledger.

**Why:** #2378 R1 g2 (`scripts/issue2378_capture.py` `_capture_cell` regime):
params/counts covered rows_cap/layers/n_rows but no content key; flagged as
the round's lead concern.

**How to apply:** when reviewing any regime-keyed resume (StageLedger-style),
diff the regime dict against the INPUTS the chunks consume: row membership
AND row text need a digest (string/text digests — machine-stable, unlike
float-array hashes). Ask "what upstream change leaves this regime EQUAL?" —
same-count regen is the canonical answer. Sibling entry:
[[start-manifest-stale-artifact-done]] (presence-done variant, #2225).
