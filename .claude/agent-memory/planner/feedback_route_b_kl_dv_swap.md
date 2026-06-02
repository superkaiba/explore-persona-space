---
name: route-b-kl-dv-swap-for-saturated-marker
description: When a contrastive-negatives or composition-sweep follow-up inherits a saturated parent (parent's continuous marker DV is ceilinged), default to route (b) DV-swap — full-vocab KL(trained ‖ base) at the post-response slot — rather than route (a) less-trained anchor, because route (b) preserves single-variable discipline.
metadata:
  type: feedback
---

When planning a follow-up that adds contrastive negatives (or any composition / negative-set sweep) to a parent experiment whose continuous marker DV (trained−base log P(marker) or ΔG) saturated, **prefer route (b) DV-swap over route (a) less-trained anchor for the headline.**

- **Route (b)** = full-vocab KL(trained ‖ base) at the post-response slot, on-policy. Per <code>.claude/rules/contrastive-negatives.md</code> §"Saturation hides everything," it's the canonical non-saturating alternative. Building block = arXiv 2504.10637 (Rao-Blackwellized estimator) per-slot version.
- **Route (a)** = less-trained anchor (lower lr / fewer epochs / smaller LoRA). Also non-saturating but introduces a SECOND manipulated variable vs the saturated parent.

**Why:** Route (b) preserves single-variable discipline (only the manipulated variable — negatives — changes vs the parent's recipe). The consistency-checker will WARN on route (a) for being a 2-variable change.

**Why:** #465 saturated at 5 epochs × 300 rows on lr=1e-5 / r=32. #471 (contrastive-negatives follow-up to #465) chose route (b). The rule explicitly authorizes KL-from-base as the canonical non-saturating DV at the saturated anchor.

**How to apply:** In §6 measurement validity, name BOTH DVs — KL (primary, non-saturating) and the parent's original DV (secondary, for paired cross-experiment comparability). Pre-register route (a) as fallback if H3 (KL preserves headroom) FAILs. Use vLLM <code>prompt_logprobs=vocab_size</code> for exact KL (memory is trivial); fall back to <code>top_k=10000</code> + tail-mass if vLLM rejects vocab-size. Always also re-evaluate the parent's adapters under the new probe so the (parent − follow-up) bootstrap is paired.

Related: [[neutral-prompt-axis-conflation]] (also about preserving single-variable discipline in measurement design); the contrastive-negatives rule (in CLAUDE.md project).
