---
name: sharex-invert-xaxis-toggles
description: invert_xaxis() inside a per-panel loop on sharex axes TOGGLES the shared limits per call — even panel counts end NOT inverted; invert ONCE on axes[0]
metadata:
  type: feedback
---

`ax.invert_xaxis()` on `plt.subplots(..., sharex=True)` axes flips the SHARED
x-limits each call: a per-panel loop inverts N times, so an even N nets to
NOT-inverted and an odd N silently works — panel-count-dependent correctness.

**Why:** hit in #2476 floor-sweep figure code (3-panel heroes happened to work,
a 2-panel variant would have shipped un-inverted); caught at self-review, cost
one fix commit.

**How to apply:** with sharex/sharey, call `invert_xaxis()`/`set_xlim(hi, lo)`
EXACTLY ONCE (on `axes[0]`, after the panel loop). Independent axes (no share)
keep the per-ax call. Same law for `invert_yaxis` under sharey.
