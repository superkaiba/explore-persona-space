---
title: Is the late-band own-answer advantage carried by next-token-identity information?
  Component decomposition of the gap and its closure
kind: experiment
tags: []
created_at: '2026-07-06T02:18:05Z'
has_clean_result: false
parent_id: 952
workflow: v1
goal: Determine whether the depth-increasing own-answer advantage in the context-to-answer-activation
  map (0.009 at layer 14 to 0.035 at layer 26) and its resistance to 16-token prefix
  closure at layer 26 are carried by the next-token-identity component of answer activations,
  by decomposing each per-position target into its projection onto the realized next
  token's unembedding (logit-lens) direction and the orthogonal complement, and re-reading
  the own-vs-external gap and closure statistics per component at layers 14/20/23/26
  on the frozen Qwen-2.5-7B-Instruct LMSYS pool.
relates_to:
- spec-context-as-vector
- identity-contextual-vs-base
---
## Goal

Determine whether the depth-increasing own-answer advantage in the context-to-answer-activation map (0.009 at layer 14 to 0.035 at layer 26) and its resistance to 16-token prefix closure at layer 26 are carried by the next-token-identity component of answer activations, by decomposing each per-position target into its projection onto the realized next token's unembedding (logit-lens) direction and the orthogonal complement, and re-reading the own-vs-external gap and closure statistics per component at layers 14/20/23/26 on the frozen Qwen-2.5-7B-Instruct LMSYS pool.


### 3. Is the late-band own-answer advantage carried by next-token-identity information? A component decomposition of the gap and its closure — Type: Diagnostic (mechanism)

**Parent:** #952
**question_relation:** substantially-different
**Goal:** Determine whether the depth-increasing own-answer advantage in the context→answer-activation map (0.009 at layer 14 → 0.035 at layer 26) and its resistance to 16-token prefix closure at layer 26 are carried by the next-token-identity component of answer activations, by decomposing each per-position target into its projection onto the realized next token's unembedding (logit-lens) direction and the orthogonal complement, and re-reading the own-vs-external gap and closure statistics per component at layers 14/20/23/26 on the frozen Qwen-2.5-7B-Instruct LMSYS pool.
**Hypothesis:** Late layers increasingly encode output-token commitment; the model's own sampled tokens are exactly the ones its pre-answer state committed to, so the own-answer gap should concentrate in the unembedding-aligned component and grow with depth there, while the orthogonal (content) component shows the near-zero gap the mid band's full closure implies — explaining in one mechanism why closure concentrates at layers 17–23 and stalls at 39% at layer 26.
**Falsification:** If the gap and the unclosed layer-26 residual are carried uniformly across components — or concentrate in the orthogonal complement — the token-identity/output-commitment account dies, and the late-band advantage reflects distributed content representation that prefix absorption cannot supply.
**Differs from parent:** New question (mechanism of the band map, not its existence) — hence a child task; within the child, the single manipulated factor is the target-component decomposition (token-identity vs orthogonal), with pool, arms, layers, split, and estimator inherited from #952 unchanged.

**Pre-filled spec (from parent):**
- Model: same (`Qwen/Qwen2.5-7B-Instruct` frozen; unembedding matrix from the same checkpoint)
- Data: same — parent slot tensors + spans @ HF `5b62649` (48 + 6 files, verified above); per-position next-token ids from #823 rollout text @ `8039d15f` (verified) + tokenizer; a small re-capture pass only if remainder-pool targets need per-position decomposition
- Seeds: same (split rng(952); bootstrap rng 0)
- Eval: parent's two decision statistics recomputed per component; component-share of the gap as the primary read; prefix + context mapping arms both reported per the standing rule
- Config: same EXCEPT the target decomposition

**Estimated cost:** ~12 GPU-h on 1× A100-80 (2 component targets × 4 layers ≈ 2× the 3.6 GPU-h cross-layer battery, + ~2 GPU-h re-capture contingency for remainder targets)
**If it works:** Names the mechanism behind the band map and tells the predictor line WHERE to read linear context→answer predictors (mid band for content, late band contaminated by output commitment) — directly serving the broader-narrative question of what such predictors capture.
**If it fails:** Rules out the most natural depth account; the mid-band closure concentration needs a different explanation (e.g. style/format representation depth), sharpening the next mechanism design either way.

**auto_run:** yes
**auto_run_reason:** Filing-only under the `substantially-different` policy (filed as a `proposed` child for manual triage, never auto-spawned): the recipe is concrete and single-fork (projection basis pinned to the realized next-token unembedding direction), every reuse premise was Hub-verified for this proposal set, and the cost is stated.

**cost_class:** needs-gpu
**headline_affecting:** no
**est_gpu_hours:** 12

---


## Provenance

Auto-filed (filed-only, never auto-spawned) from `epm:follow-ups v3` proposal 3 on parent #952 at the kfold-decision-cells re-park, after the follow-up-critic + codex-follow-up-critic redundancy screen returned not-redundant x2 (epm:followup-value-critique v3 + epm:followup-value-critique-codex v3).
