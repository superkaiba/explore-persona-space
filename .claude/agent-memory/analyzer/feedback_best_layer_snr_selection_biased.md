---
name: Best-layer SNR is selection-biased — report the fixed-layer read alongside
description: A per-cell signal-to-noise computed as the max over all layers optimistically clears a noise floor; always pair it with a fixed-layer + across-layer read
type: feedback
---

When a measurement DV is computed per layer over all 28 layers and you summarize a cell by the BEST-of-28-layers value (e.g. the layer that maximizes a leakage signal-to-noise vs a noise floor), that max is a selection-biased statistic — it will clear a floor (SNR>1) far more often than the underlying signal warrants, because you searched 28 layers for the most favorable one.

**Why:** In #664 the marker leakage-variation SNR (cross-context activation-gate spread ÷ within-context probe-split noise floor) had a best-of-28-layers median of 1.32 (19/20 cells above the floor), which looks like "clears the kill criterion." But at a FIXED mid-layer (14) the median was 0.99 (only 10/20 above the floor) and the across-all-layers median was also 0.99. The honest read is "at, or marginally above, the noise floor — thin dynamic range," not "clears it." Reporting only the best-layer SNR would have overstated the result and a critic would (rightly) bounce it.

**How to apply:** Whenever a per-cell statistic is a max (or argmax-selected) over a layer / probe / threshold sweep, ALWAYS compute and report two companions: (a) a FIXED, pre-committed layer/threshold read (selection-bias-free reference), and (b) the median across the swept dimension. Put both in the figure (the #664 forest plotted best-layer circles AND fixed-layer-14 diamonds on the same row so the gap is visible) and state the best-layer value is "optimistically selected" in the prose. When the fixed-layer read sits at the floor while the best-layer read clears it, the binding conclusion is the fixed-layer one.
