---
name: batch-copied-sidecar-provenance-field
description: On figure reuse-copy diffs, verify each sidecar's appended provenance data field against the render FUNCTION's actual reads — a uniform field stamped across batch-copied sidecars is wrong for any figure reading a different source (#2031 round 1 blocker)
metadata:
  type: feedback
---

When a round batch-copies figures with a provenance block appended to each
meta.json, check the `data:` (input path) field PER FIGURE against the named
render script's per-figure function — not against the script's module
docstring or the sibling figures.

**Why:** #2031 round 1 (2026-08-09): the fold stamped
`data: eval_results/issue_1689/user_slot_recapture/summary.json` on all three
copied sidecars, but `r4_assistant_to_user_ladder()` reads only the parent
ladder JSON (`eval_results/issue_1689/ladder/..._L19.json`) — the body's own
caption said "parent-round ladder data", so the committed sidecar contradicted
both the render code and the body. fig16/fig17 were correct (their functions
read the module-level summary.json load). Sole-reviewer FAIL blocker on an
otherwise fully-tracing fold (~25 numbers verified).

**How to apply:** for each copied figure, open the render script at the pinned
source commit, find that figure's function, and list its actual `json.load`/
open sites; compare to the sidecar's data field. A uniform provenance block
across ≥2 figures from one script is the smell — scripts routinely mix data
sources per figure (module docstrings list ALL sources, so they cannot
adjudicate). Body-prose vs sidecar disagreement on a figure's data source is
an instant tell.
