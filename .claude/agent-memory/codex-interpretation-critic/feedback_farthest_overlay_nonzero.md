---
name: feedback_farthest_overlay_nonzero
description: Caption claims farthest-cosine bystander overlays "never rise off zero" but figure shows one dashed gray line reaching ~10% at step200 — caught by Lens 6 plot-prose match
metadata:
  type: feedback
---

When a figure overlays "three farthest" bystanders and the body text says they "never rise off zero," verify in the PNG that ALL dashed gray lines are truly flat at zero — not just most of them. In task #385, `fammate_format_1` (YAML format, cosine=0.620, the farthest-by-cosine bystander) crosses at step150 and reaches ~10% at step200, but the body claimed the far-cosine overlay never rises. Codex lens-6 caught this by visually inspecting the figure.

**Why:** When there are multiple bystanders in an overlay group, it's easy to describe them in aggregate ("never rise off zero") when one outlier contradicts the claim. Always inspect each individual overlay line, not just the group pattern.

**How to apply:** When body says "X overlay never rises" / "X is at floor" / "far group stays at zero," load the figure and trace each individual member of the overlay group.
