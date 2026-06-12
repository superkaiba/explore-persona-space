---
name: Content-arm comparisons — off-policy gradient entanglement + floor-censored equal-fate
description: Aligned-vs-misaligned (or any corpus-CONTENT) eraser arms inherently entangle content with gradient magnitude/dose; disposition is operational claim-scoping. Equal-fate nulls at the emission floor are censored — read the latent DV first.
type: feedback
---

When a plan's manipulated variable is corpus CONTENT (e.g. aligned vs
misaligned response column, same prompts, same recipe — #570 over #557/#376),
two alternatives recur:

1. **Off-policy gradient entanglement.** The "different" content is also
   farther from the model's current policy → higher per-token loss → larger
   gradient norms → more total drift at equal lr/steps. You cannot match lr
   AND gradient norm AND content — the lever IS the composite (sibling:
   feedback_ratio_lever_inherent_entanglement). Dose can also differ via
   loss-bearing token counts between response columns even when "length-
   matched" is asserted. NOT a REVISE when the pre-registered hypothesis is
   operational ("erodes more at matched lr") — that is what the application
   (drift detector at realistic recipes) needs. Becomes a problem only if the
   headline claims content-beyond-pressure mechanism. Diagnostics to demand:
   per-arm training-loss trajectories, initial/final loss, corpus token-count
   distributions (free from Hub files post-hoc).

2. **Floor-censored equal-fate null.** If both arms drive emission to ~0,
   overlapping Wilson CIs at the floor cannot distinguish "equal erosion"
   from "a real differential censored by the emission floor." The latent
   slot-stat read (4-float, trained AND base) survives below the floor
   (#557: 8.5-9 nat latent retention at zero emission) — instruct the
   analyzer to read the latent gap before declaring equal fate.

3. **Measurement-side policy shift.** A content arm that changes the model's
   behavior (EM induction) also changes its OWN responses, so on-policy
   emission can drop because the response contexts moved, not because the
   rule eroded. The base-side floats of the 4-float contract (base log P /
   z_eos at the same slot on the same completions) diagnose this cross-arm;
   a fixed-completion force-read is the stronger disambiguator if base-side
   stats weren't stored (sibling: feedback_completion_source_swap_mediation).

**How to apply:** all three are analyzer-weighable when the plan persists
per-completion records + trained-and-base slot stats + per-arm loss records —
APPROVE with concerns. REVISE only if the headline is mechanistic AND none of
these diagnostics are captured.
