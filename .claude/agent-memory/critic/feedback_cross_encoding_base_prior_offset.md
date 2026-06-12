---
name: Cross-encoding paired-d base-prior offset (i464 lineage)
description: Headline d contrasting cells probed under DIFFERENT eval encodings (role vs system) ignores the base model's encoding-specific prior; recompute on delta_g — and compute the offset before assuming it's fatal (it can run anti-hypothesis)
type: feedback
---

In the #464/#529/#533/#546 role-vs-system lineage, the headline paired d = log P_system − log P_role compares trained models probed under DIFFERENT encodings, and `_per_persona_paired_d_block` reads raw `g_logprob`, NOT `delta_g` (trained − base). The per-cell JSONs DO carry per-question base reads (`b_logprob`, `b_logps_per_q`, `delta_g`), so the offset is always recoverable with zero GPU.

**Why:** measured from #533's per-cell JSONs, the pirate-encoding base offset was +0.63 nat role-favoring — LARGER than the lineage's |mean d| ≥ 0.5 verdict threshold, so an H1− "role leaks more" verdict on villain-trained cells can be mechanically produced by the base prior with zero differential leakage (villain-encoding offset ≈ 0).

**How to apply:** for any directional cross-encoding headline in this lineage, require (analyzer concern, not REVISE — recoverable from planned artifacts) the paired d recomputed on `delta_g` alongside raw `g_logprob`, plus the base gap at the anchor. Base values should reproduce across runs (same model + encodings + questions) — diffing new-run `b_logprob` vs the parent's is a free rig-stability check.

**Compute the offset before assuming it's fatal — it can run ANTI-hypothesis (conservative).** In the #528/#556 Likert-judge lineage, base offsets (role − system) from parent `judge_scores.json`: own_scenario −0.525 but OFF-TARGET pooled +0.115 — OPPOSITE the predicted negative d, so the formatting prior cannot manufacture the segmentation PASS (base-adjusting makes the effect LARGER). A large own-scenario offset does not predict the off-target offset's sign; 10 minutes of jq on committed base rows converts a speculative Must-Fix into a quantified concern.
