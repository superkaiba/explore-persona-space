---
name: annotation-extent-includes-arrow
description: matplotlib Annotation.get_window_extent unions the leader ARROW into the bbox — measure label rects via get_bbox_patch().get_window_extent(); plus the narrow-panel staggered-xtick recipe
metadata:
  type: reference
---

Two c2a combined-figure lessons (c3_directions_and_pairs, 2026-09-03):

1. **`Annotation.get_window_extent(renderer)` includes the LEADER ARROW**, not
   just the text: a 63 pt leader inflated a 0.24 in-tall label to a 1.08 in
   phantom rectangle, and a collision-scoring/label-placement pass built on it
   chased nonexistent overlaps for several anneal rounds. Measure label boxes
   via `t.get_bbox_patch().get_window_extent(r)` (the white knockout patch —
   text + pad, no arrow).

2. **Narrow-panel two-line x tick labels: stagger, don't shrink.** Measured on
   the pair-shifts panel at 0.4 of a full c2a canvas: worst adjacent labels
   need a ~1.07 in slot vs the 0.74–0.78 in available at any sane split — no
   width ratio fixes it. Recipe: `for tick in ax.xaxis.get_major_ticks()[1::2]:
   tick.set_pad(tick.get_pad() + 44)` (44 pt ≈ one full two-line label height
   at 17 pt) and size the gridspec bottom margin ~1.3 in.

Also: annealed offset maps beat hand-packing for ≥10 labels in a compressed
panel — freeze the found map as a literal dict in the script (throwaway
optimizer, committed constants). Curve overlaps are SOFT in the c2a style
(white-knockout bbox, standalone precedent [[c2a-v2-figure-restyle-gotchas]]);
label-label / label-dot / leader-through-dot are hard.
