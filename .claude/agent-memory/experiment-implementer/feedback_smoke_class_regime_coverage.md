---
name: smoke set covers every behavior-class × regime combination
description: A multi-class dispatcher's smoke runs ≥1 tiny cell per realized (behavior-class × regime) combination and reaches class-gated read-side paths — never one content-class cell
type: feedback
---

A dispatcher grid crossing behavior classes (marker vs content), training
regimes (contrastive `con` vs positive-only `po`), or methods (LoRA vs
full-FT) has class-specific code paths (marker parity reads, per-class mix
asserts, panel-disjointness reads, reuse-seam loaders) a single-cell smoke
never reaches; the bugs then surface live one per phase (#1586 r3/r4/r6 —
three distinct class-gated bug classes after every recorded smoke ran only
`syc-pers-ft-con-s137`).

**Why:** "arm class" (Step 6d.0-bis / the gotchas.md single-arm entry) had
been read as source-context class only; the class-defining axes are the
CROSS of behavior class × regime × method, and read-side/aggregation phases
count as class-specific code paths too.

**How to apply:** when composing any multi-class dispatcher's smoke set,
enumerate the class-defining axes and pick the cheapest cell per realized
combination; record the cell list in the smoke marker; treat a one-cell
smoke on a multi-class grid as a review red flag. Per-cell numeric gates on
the added cells are sized to smoke n per the GATE CALIBRATION sibling
(#1345) — a registry-expectation / mix-composition mismatch an added cell
surfaces is shape-correct, not a scale artifact, when the smoke builds the
production-registry mix. Full entry: `.claude/rules/gotchas.md`
"Smoke/production parity includes REGIME/CLASS COVERAGE".
