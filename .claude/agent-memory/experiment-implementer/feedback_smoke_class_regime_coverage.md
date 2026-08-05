---
name: smoke set covers every behavior-class × regime combination
description: A multi-class dispatcher's smoke runs ≥1 tiny cell per realized (behavior-class × regime) combination and reaches class-gated read-side paths — never one content-class cell
type: feedback
---

A dispatcher grid crossing behavior classes (marker vs content), training
regimes (contrastive `con` vs positive-only `po`), or methods (LoRA vs
full-FT) has class-specific code paths (marker parity reads, per-class mix
asserts, panel-disjointness reads, reuse-seam loaders) a single-cell smoke
cannot cover: bug classes in cells the smoke never runs surface live one
per phase, and a class-gated read-side path the smoke itself reaches only
late kills the smoke leg (#1586: r3 panel-disjointness died in the smoke
leg at p6_panel; r4 marker-`po` mix rowcount and r6 reuse-seam schema then
surfaced live post-smoke, after every recorded smoke ran only the
content-class full-FT cell `syc-pers-ft-con-s137`).

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

**The axes include the ARM ROSTER and the PIPELINE PASS, not just data
classes (#1739, 2026-08-05 — a re-instance of this rule, not a new one).**
`scripts/issue1739_holdout_rung.py` shipped TWO production-path-only
defects past a green `--smoke` in one round, because the smoke's own
configuration excluded the code that broke: (1) `_whiten_acts` broadcast
bug — the synthetic smoke bypasses production whitening entirely, caught
only by a MEASURED production-shape pilot; (2) the transfer pass called
`run_cell_multi(..., ridge_folds=(0,))` against a roster containing
`arm10_stacked`, whose contract forbids ANY fold subset — the smoke ran a
REDUCED roster (first 4 arms, no arm10) and never entered the transfer
pass at all, so the incompatible combination was unreachable. It failed
after a 25-minute CV pass on a $2.7/hr box, twice re-hosted.

Two additions when picking smoke cells: (a) if the production roster is
FIXED and larger than the smoke roster, the smoke must still exercise every
arm whose contract CONSTRAINS the call (an arm that rejects an argument
shape is a class of its own — a reduced smoke roster silently excludes the
strictest contract); (b) every distinct PIPELINE PASS the production run
executes (CV vs transfer, fit vs read-side) is its own coverage axis — a
pass the smoke never enters is untested regardless of roster. The fix round
added a startup-time `_validate_ridge_folds_roster` refusing the bad
combination in seconds; prefer that shape — a cheap static/startup
compatibility check beats discovering a contract violation after the
expensive phase. Precedent for the argument itself lived in a sibling
script (`issue1739_rescore_ood_armfill.py::_ridge_folds_arg`), so ALSO
grep sibling scripts for the same call before hardcoding an argument.
