---
title: Does post-training make particular SAE features more predictable from context?
  (per-stage context -> answer-SAE-feature map)
kind: experiment
tags: []
created_at: '2026-08-04T04:44:57Z'
has_clean_result: false
parent_id: 1895
origin_prompt: can you also run another issue which is training context vector ->
  SAE feature mapping at each stage and see if any SAE features get better predicted
  after each stage.
workflow: v1
goal: Fit a map from the pooled context activation to the SAE feature vector of the
  answer state, holding ONE fixed SAE dictionary across every post-training stage
  of an open ladder, and determine whether any individual SAE feature's per-feature
  held-out predictability improves across adjacent stage transitions beyond a selection-symmetric
  null over the whole dictionary.
---
## Goal

Fit a map from the pooled context activation to the SAE feature vector of the answer state, holding ONE fixed SAE dictionary across every post-training stage of an open ladder, and determine whether any individual SAE feature's per-feature held-out predictability improves across adjacent stage transitions beyond a selection-symmetric null over the whole dictionary.


## Overview / Motivation

The context→answer map has been read out in raw activation space at every
post-training stage of two open ladders ([#1902](https://eps.superkaiba.com/tasks/1902)
OLMo-2, [#1336](https://eps.superkaiba.com/tasks/1336) Llama Tülu). Separately,
[#1895](https://eps.superkaiba.com/tasks/1895) established that the map's
predictable subspace and an SAE's representable subspace largely coincide at the
variance grain on Qwen. This task joins the two axes: read the map's target in
**interpretable SAE-feature space** and ask which named features become more
predictable from context as post-training proceeds.

The interest is that a raw-activation R² is a scalar with no semantics — it says
the map got better or worse, never *at what*. A per-feature read says which
interpretable properties of the answer the model commits to earlier as SFT / DPO
/ RLVR proceed.

## Goal (formalization — refine at the clarifier gate)

Fit `g: c_x → f(a_x)` where `c_x` is the pooled context activation and `f(a_x)`
is the **SAE feature vector of the ANSWER state**, holding ONE fixed SAE
dictionary across every post-training stage, and measure per-feature held-out
predictability `R²_j` at each stage. The dependent variable is the per-feature
delta `ΔR²_j` across adjacent stage transitions; the question is whether any
feature's improvement survives a selection-symmetric null over the dictionary.

**Named ambiguity for the clarifier (Step 1) to resolve before planning:** the
originating ask says "context vector → SAE feature mapping" without naming which
state the SAE is applied to. The reading above (SAE on the ANSWER state — the
existing map with an interpretable target) is the one that connects to
[#1895](https://eps.superkaiba.com/tasks/1895) and makes "features get better
predicted" meaningful. The alternative (SAE on the CONTEXT state, features as
map INPUTS) is a different experiment. Confirm before planning.

## BLOCKING asset gate — resolve FIRST, before any design work

**Every SAE this project has ever used is Qwen2.5-7B-Instruct**
(`andyrdt/saes-qwen2.5-7b-instruct`, 22 references across `scripts/` + `src/`;
`chanind/qwen2.5-7B-it-layer-20-saes`, 2). Verified by repo-wide grep at filing
time — no Llama SAE, no OLMo SAE anywhere in the tree.

That collides directly with the ladders:

| ladder | separated stage checkpoints | SAE in repo |
|---|---|---|
| Qwen2.5-7B | NO — base + instruct only | YES |
| Llama-3.1-8B Tülu-3 | YES — base/SFT/DPO/RLVR/longer-RLVR | NO |
| OLMo-2-1124-7B | YES — base/SFT/DPO/Instruct | NO |

So the experiment as asked is not runnable on any ladder without new assets.
Three candidate resolutions, in cost order. **Verify, do not assume — none is
referenced in this repo and all three are unverified at filing time:**

1. **A public base-model SAE for a staged ladder.** Llama Scope (`fnlp`) is
   reported to cover Llama-3.1-8B base across all layers, and Goodfire is
   reported to have released Llama-3.1-8B-Instruct SAEs. Both are recalled, not
   checked — resolve on the Hub and gate on FVE/L0 before committing.
2. **Train ONE SAE ourselves** on the ladder's BASE model, applied fixed across
   all stages. A real GPU project on its own; if this is where it lands, the
   SAE-training leg should be its own task rather than a phase here.
3. **Degrade to the Qwen base→instruct pair.** SAEs exist, but "each phase of
   post-training" collapses to a single step, which loses the point. Fallback
   only, and it must be named as a scope reduction, not shipped as the answer.

If none resolves, this task is BLOCKED, not descoped — say so and park.

## Design constraints (binding once the asset gate clears)

**Fixed feature basis.** ONE SAE, applied unchanged to every stage. A per-stage
SAE makes the feature index meaningless across stages and the whole per-feature
delta uninterpretable. This is not a preference; it is what makes the DV exist.

**Cross-stage SAE-fitness confound.** An SAE fit on base-model activations is
off-distribution for post-trained activations, so its reconstruction quality
will drift by stage. A feature can then look "better predicted" purely because
the SAE represents that region differently at that stage. Report per-stage
FVE / L0 / dead-feature count for the fixed dictionary on each stage's
activations, and control for it before any per-feature claim. Reproduce the
reference eval's TOKEN-POOL semantics when gating FVE/L0 against a published
number — BOS-offset strip, outlier-norm filter, var-based FVE
(`.claude/rules/gotchas.md`, the #1482 entry; a verbatim-correct encoder fed the
wrong token pool read FVE ≈ −10³ and mimicked a loader bug).

**Selection-symmetric null is the central statistical requirement.** "Do ANY
features get better predicted" is a max/argmax over a dictionary of order 10⁴–10⁵
features — the largest multiple-comparisons surface in this line by far. The
selection must ride INSIDE each null draw per
`.claude/rules/selection-symmetric-nulls.md`; a ranked top-features list without
it is not a finding and must not carry a takeaway. Pre-register the null and the
band before looking.

**Reuse the banked map assets.** Whichever ladder clears the gate, its context
and answer activations are already captured — [#1336](https://eps.superkaiba.com/tasks/1336)
(`issue1336_rlvr_ladder/`, 5 checkpoints × 8 corpora) or
[#1902](https://eps.superkaiba.com/tasks/1902) (`issue1902_stage_map/`, 4
checkpoints × 4 answer sources). The SAE encode is a forward pass over banked
activations, not a re-capture. Run the artifact-reuse fitness check
(`.claude/rules/artifact-reuse.md`) — in particular the staged-layout
consumer-open probe (h)(iv) and the throughput check (i) on any reused fit code.

**Standing mapping rules apply unchanged:** both mapping arms (prefix AND
context), identity+learned-bias baseline and kNN retrieval per fitted map,
group-level held-out folds with the eval set fully disjoint, pooling convention
stated with parity against the cited comparison line.

**Instrument-supersession check.** If the round wants to LABEL the improved
features, check [#1773](https://eps.superkaiba.com/tasks/1773) first — it is the
superseding labeling instrument, and spending Batch-API judge budget on a
known-weak predecessor is the exact freeze this rule exists to prevent.

## Literature review (REQUIRED before any experiment code)

This is a new direction, so the standing rule binds: a thorough arXiv + web
review naming the closest prior formalizations BEFORE any training code. The
specific question — *does post-training make particular SAE features more
linearly predictable from context* — sits near SAE-based model-diffing, crosscoder
work on base-vs-chat feature emergence, and the persona-vectors line. Name what
already exists and what the closest published result is; do not jump to
experiments.

## Relation to the existing line

- [#1895](https://eps.superkaiba.com/tasks/1895) — map's predictable subspace vs
  SAE's representable subspace (the SAE-side parent, HIGH confidence).
- [#1482](https://eps.superkaiba.com/tasks/1482) — per-direction R² vs variance
  rank; also the source of the category-axis confusable-neighbor lesson
  (20/40 top "persona" features were bare language-identity features).
- [#1738](https://eps.superkaiba.com/tasks/1738) — banked SAE-space mapping
  baselines.
- [#1336](https://eps.superkaiba.com/tasks/1336) / [#1902](https://eps.superkaiba.com/tasks/1902)
  — the post-training stage axis this extends. #1336 currently has a live
  same-issue follow-up running the pooled multi-dataset on/off-policy transfer
  battery; coordinate rather than duplicate its capture work.
