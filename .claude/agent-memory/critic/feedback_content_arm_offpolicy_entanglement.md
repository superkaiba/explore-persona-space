---
name: Content-arm comparisons — off-policy gradient entanglement + floor-censored equal-fate
description: Corpus-CONTENT eraser arms inherently entangle content with gradient magnitude/dose (operational scoping, not REVISE); floor-censored equal-fate nulls need the latent slot-stat read first (#570)
type: feedback
---

When a plan's manipulated variable is corpus CONTENT (e.g. aligned vs misaligned response column, same prompts/recipe — #570 over #557/#376):

1. **Off-policy gradient entanglement.** "Different" content is also farther from current policy → higher per-token loss → larger gradients → more total drift at equal lr/steps. You cannot match lr AND gradient norm AND content — the lever IS the composite (sibling: feedback_ratio_lever_inherent_entanglement); dose can also differ via loss-bearing token counts even when "length-matched". NOT a REVISE when the pre-registered hypothesis is operational ("erodes more at matched lr") — that's what the application needs; a problem only if the headline claims content-beyond-pressure mechanism. Demand: per-arm training-loss trajectories, initial/final loss, corpus token-count distributions (free from Hub files).
2. **Floor-censored equal-fate null.** If both arms drive emission to ~0, overlapping Wilson CIs at the floor can't distinguish "equal erosion" from "a real differential censored by the emission floor". The latent slot-stat read (4-float, trained AND base) survives below the floor (#557: 8.5–9 nat latent retention at zero emission) — read the latent gap before declaring equal fate.
3. **Measurement-side policy shift.** A content arm that changes the model's behavior (EM induction) also changes its OWN responses, so on-policy emission can drop because the contexts moved, not because the rule eroded. The base-side floats at the same slot on the same completions diagnose this cross-arm; a fixed-completion force-read is the stronger disambiguator (sibling: feedback_completion_source_swap_mediation).

**How to apply:** all three are analyzer-weighable when per-completion records + trained-and-base slot stats + per-arm loss records persist — APPROVE with concerns. REVISE only if the headline is mechanistic AND none of these diagnostics are captured.
