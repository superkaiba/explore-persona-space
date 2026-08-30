---
name: falsification-demo-needs-faithful-tree
description: An old-vs-new falsification demo must run on a tree with the same shape as production, or a minimal synthetic tree fakes the OLD=FAIL half and hides that the new check adds nothing
metadata:
  type: feedback
---

When proving a strengthened check catches a defect the old one missed, build the
synthetic tree with the SAME population the real check scans, not just the one
defective file.

**Why:** #2386 round 2 asked for falsification evidence per fixture. My first
demo of the alternate-`fatal`-spelling case reported `OLD=FAIL, NEW=FAIL`, which
would have meant the fix was unnecessary. The old scanner failed only because my
tree contained a single wrapper, so its `missing`-list check reported all seven
DRIVEN wrappers as absent — an artifact of the tree, not a detection. Rebuilding
with all seven driven wrappers present and healthy plus one NON-driven defective
wrapper gave the true `OLD=PASS, NEW=FAIL`.

**How to apply:** whenever the old check has any population-level assertion
(a "these N must exist" list, a coverage count, a set-equality), populate the
demo tree with that full population before introducing the single defect. Then
read the OLD failure MESSAGE, not just its exit code — a pass/fail pair proves
nothing until the failure reason is the one you intended. Reimplementing the
old check body verbatim and running both scanners over identical trees is the
cheap way to get this right, and the resulting numbers are what the reviewer
actually wants in the results marker.

Corollary for the fixture's docstring: state the precise scope of the old gap
(here: the old list covered only 7 of 15 wrappers, so the silent skip applied to
the other 8) rather than the broader claim the demo does not support.

Related: [[feedback_report_claims_carry_fresh_evidence]].
