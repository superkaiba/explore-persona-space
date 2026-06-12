---
name: Fixed-total breadth sweeps — bundled duplication/coverage disposition
description: Panel-breadth ablations at fixed total rows inherently bundle per-context duplication + question coverage; disposition = composition-level claim-scoping (per #543 ratio-lever precedent), not REVISE; watch the new-code-path-asymmetry mapping onto the arm contrast
type: feedback
---

When a plan's manipulated variable is negative-PANEL BREADTH at fixed total
rows (e.g. #571: 15 bystanders × 20 rows vs 4 × 75, total 300), three things
move together and CANNOT be unbundled by any added arm: (1) count of distinct
contexts, (2) rows-per-context (duplication/memorization pressure), (3)
per-context question coverage. The complementary design (fixed per-context
rows) just swaps in a TOTAL-count bundle instead — no design isolates breadth
per se.

**Why:** identical to the #543 ratio-lever disposition — the lever IS the
composite. "Many distinct contexts → suppression generalizes" and "low
per-context duplication → suppression generalizes" both predict broad > narrow
on held-outs and are observationally equivalent in a 2-arm fixed-total design.

**How to apply (alternatives lens):** NOT a REVISE when (a) the Goal itself
defines the manipulation at the fixed-total composite level, (b) the plan
names the bundle honestly, and (c) per-context training diagnostics (e.g. M5
per-bystander suppression-loss curves) are reported so the analyzer can weigh
the memorization story. Require the analyzer to phrase the headline at the
panel-composition level ("the broad 15×20 configuration causes X"), never
"breadth per se causes X". Also check: (i) a single narrow-subset draw leaves
context-IDENTITY vs cardinality unseparated — scope to "this 4-condition
subset"; (ii) when the narrow arm exercises a NEW builder code path while the
broad arm runs the battle-tested default, any code-path bug maps 1:1 onto the
arm contrast — verify hard asserts (marker absent, row counts, loss mask) +
the smoke canary runs the NEW path; (iii) cross-arm slot-composition drift
(emission/truncation rates differ by arm → pre_marker vs end_of_response slot
mix differs) is a free emission-slots-excluded sensitivity downstream if slot
kinds are persisted per cell (#560 precedent).
