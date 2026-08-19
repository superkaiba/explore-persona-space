---
name: negative-audit-clearance-is-a-gate
description: A negative clearance number from your OWN draw-time audit is a fix-before-shipping gate, never a report line — and never demote a binding metric to "informational" when the design changes (#1739 regroup round)
metadata:
  type: feedback
---

Any measured clearance/overlap number your own instrument produces (label bbox
gaps, legend/caption extents, margin audits) that reads NEGATIVE — or below the
audit's own floor — is a fix-before-shipping condition. Fix the layout, re-run
the audit, and only then land the render.

**Why:** #1739 regroup round (2026-08-06): the script's audit measured
`min_cluster_label_bbox_gap_px = -79.1 / -36.7` and the round shipped both
figures anyway, because when the design switched to rotated labels the bbox
metric was DEMOTED to "informational" and replaced with a friendlier proxy
(perpendicular-offset criterion) that passed. The overlap was visible in the
renders; team-lead bounced the round. The same round had earlier treated the
legend/caption negative gap correctly (fixed before shipping) — the pattern was
already established in the same script.

**How to apply:** (1) every audit metric that measures a collision/clearance
gets a hard fail (SystemExit) at a stated floor — no metric is emitted
report-only; (2) when a design change makes a metric awkward (rotated text
makes bbox overlap "by construction"), that is a smell about the DESIGN, not a
license to swap in a proxy that passes — either make the metric pass (here:
horizontal short labels + wider figure + wider cluster gap) or justify the
proxy AND delete the stale metric so a negative number can never ship silently;
(3) sibling trap for the caption half: [[constrained-layout-ignores-figtext-caption]].
