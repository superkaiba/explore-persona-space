---
name: Cross-encoding paired-d base-prior offset (i464 lineage)
description: When a headline d contrasts cells probed under DIFFERENT eval encodings (role vs system), the base model's marker prior differs by encoding; check b_logps_per_q before accepting a directional verdict
type: feedback
---

In the #464/#529/#533/#546 role-vs-system lineage, the headline paired d = log P_system − log P_role compares trained models probed under DIFFERENT eval encodings, and `_per_persona_paired_d_block` reads raw `g_logprob`, NOT `delta_g` (trained − base). The per-cell JSONs DO carry per-question base reads (`b_logprob`, `b_logps_per_q`, `delta_g`), so the offset is always recoverable with zero GPU.

Measured from #533's per-cell JSONs (constant across seeds/epochs, base cached per e_eval):
- probe under **pirate** encoding (i.e. villain-trained wrong-slot): base role −21.22 vs base system −21.85 → role +0.63 nat in BASE alone, shifting d by −0.63.
- probe under **villain** encoding (pirate-trained wrong-slot): −20.90 vs −20.90 → ≈ 0 offset.

**Why:** the H1+/H1− verdict threshold in this lineage is |mean d| ≥ 0.5 nat — SMALLER than the measured base offset on one persona. A "role leaks more" (H1−) verdict on villain-trained cells alone can be mechanically produced by the base prior with zero differential leakage. Note villain is also the persona most likely to anchor (only persona that banded in #533).

**How to apply:** for any plan/result in this lineage with a directional role-vs-system headline, require (as analyzer concern, not REVISE — it's recoverable from planned artifacts) recomputation of the paired d on `delta_g` alongside raw `g_logprob`, plus reporting the base gap at the anchor. Also: base values should reproduce across runs (same model + encodings + questions) — diffing new-run `b_logprob` vs #533's is a free rig-stability check for cross-run ghost comparisons.

**Compute the offset before assuming it's fatal — it can run ANTI-hypothesis (conservative).** In the #528/#556 Likert-judge lineage (validating trait, `BUILD_EVAL_PROMPT(arm, ...)` renders per-arm surfaces even off-target), base offsets (role − system) measured from parent `judge_scores.json`: own_scenario −0.525, OFF-TARGET pooled +0.115 (sibling_1 alone +0.408). The off-target offset is OPPOSITE the predicted negative d (−0.155), so the formatting prior cannot manufacture the segmentation PASS — base-adjusting makes the effect LARGER (~−0.27). Lesson: a large own-scenario offset does NOT predict the off-target offset's sign; 10 minutes of jq/python on committed base rows converts a speculative Must-Fix into a quantified concern.
