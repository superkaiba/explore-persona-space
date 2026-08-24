---
name: figure-populated-assert-reference-artists
description: An empty-figure/populated-render assert that scans ax.lines is defeated by reference artists (axhline/axvline contribute finite ydata) — probe it with all-NaN data + a zero-line (#2379 R1 g4)
metadata:
  type: feedback
---

A defensive "figure has finite data" assert that collects values from
`ax.lines` (plus collections/patches) and requires ≥1 finite value is
DEFEATED on any axes that draws a reference artist: `ax.axhline(0.0)` /
`axvline` create Line2D objects in `ax.lines` with finite ydata, so an
all-NaN data render passes. Most publication panels carry a zero-line or
threshold line, so the guard certifies almost nothing exactly where the
incident class it cites (#1112 empty-figure) would occur.

**Why:** #2379 R1 g4 (`issue2379_analysis.py::_assert_fig_populated`): the
assert cited #1112 but every consumer panel (hero bars, layer curves,
forest, gate) drew `axhline(0.0)` — a 6-line probe (all-NaN `ax.bar` +
`axhline` → no raise) confirmed the defeat. Flagged Minor; the analysis
itself was fail-loud elsewhere.

**How to apply:** when a diff adds an empty/populated-figure assert, (1)
list the artist collections it scans; (2) check whether the consuming
panels draw reference lines/threshold lines into the same collection; (3)
certify with the 6-line probe (build fig with all-NaN data + one reference
artist, call the assert, expect no-raise = defeated); (4) suggested fixes:
tag data artists with `gid` and filter, or scan only patches/collections
on bar panels. Sibling family: [[maximal-prefix-suffix-diff-check-tautology]]
(gate predicates that cannot fail); this one is a guard that cannot fire.
