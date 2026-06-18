---
name: rank1-readwrite-design-621
description: Methodology dispositions for rank-1 LoRA read/write-vector designs (#621 class) — trained-negative bystanders in the firing-Spearman, bridge-arm panel swap forced by disjointness, no-band-entry wall worst case
type: feedback
---

Rank-1 read/write designs (#621 v1, APPROVE-worthy shape): training is fresh by necessity (rank IS the manipulated variable — no reuse possible), A-init snapshot + 10-step checkpoints + per-arm band-entry reads + smoke-verifiable band-stop telemetry satisfy items 5/9/11/12.

**Why these are Concerns, not REVISEs:**
1. The firing→leakage Spearman's "true bystanders" include trained-negative panel members that sit in the eval panel (e.g. kindergarten_teacher, assistant in the 19-panel): their leakage was directly suppressed by the contrastive gradient, not gated by a·v_c. Per-persona four-float reads are persisted → analyzer recomputes ρ excluding them. Only 1–2 of 17–18 points.
2. Bridge-arm "rank as the only change" is vs the #527 dial recipe, but vs #604's realized florist mixes there's also a forced 1-of-4 panel-member swap (librarian→kindergarten_teacher, mandated by the design-wide disjointness HARD invariant since librarian is a realized source elsewhere in the design). Weighable: #505/#448 showed negative placement/count are not levers.
3. Wall-time at the no-band-entry worst case (all cells run the full epoch cap at closest-approach) is the expensive scenario the band-entry-at-step-30-40 basis doesn't cover — fine when smoke re-projects realized sec/step AND realized band-entry step before the sweep launch with a pre-registered descope priority.

**How to apply:** any plan training r=1 implants for direct (a,b) reads — check eval-panel ∩ trained-negative overlap in the rank test, bridge/parent attribution deltas, and that the smoke re-projection covers the full-cap scenario.
