---
name: Open-marker scatter needs explicit linewidths; pixel-probe to verify
description: set_paper_style() zeroes scatter edge widths so facecolors="none" markers render invisible (#536, recurred #613 hero); fix with linewidths=1.2-1.4 and verify via colored-pixel probe
type: feedback
---

Any `ax.scatter(..., facecolors="none", edgecolors=...)` (open markers encoding an arm/condition) MUST pass explicit `linewidths=1.2-1.4` — `set_paper_style()` zeroes `lines.markeredgewidth` and scatter inherits 0.0, so the open series silently vanishes.

**Why:** Recurrence of the #536 pitfall: task #613's round-1 hero rendered only the flag-on circles; the flag-off squares (the whole co-land claim) were invisible. Both critics REVISEd on it; "all visually verified" in the fact sheet was wrong because a small downscaled Read of the PNG made the absence easy to miss.

**How to apply:**
1. When a figure encodes arms by open-vs-filled markers, grep the plot script for `facecolors="none"` without `linewidths` BEFORE trusting the render.
2. Verify the fix mechanically, not just by eye: count colored pixels in the panel region (`np.abs(r-g)+np.abs(g-b)+np.abs(r-b) > 60`) before vs after — #613 went 4,531 → 8,137 px when the squares appeared.
3. Trap when regenerating: scripts that call `savefig_paper(fig, some_path/"name")` with the default `dir="figures/"` write to `figures/figures/...` — an unchanged mtime on the canonical PNG means your regeneration landed elsewhere; check the output path before concluding the rerun was a no-op.
