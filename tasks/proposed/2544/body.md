---
title: How the context→answer map forms across pretraining (Olmo 3 7B checkpoint ladder)
kind: experiment
tags: []
created_at: '2026-08-24T17:24:45Z'
has_clean_result: false
parent_id: 1902
origin_prompt: 'I want to run an experiment to see how our mapping evolves throughout
  pretraining. Similar to the experiment where we were seeing how it evolves throughout
  post training. I want to see both the individul checkpoint mappings trained on its
  own completions and the transfer to subsequent checkpoints. Please help me to design
  this experiment. First we need to find a model that we can use that is ideally a
  very recent model that''s very good with this, and then we want to figure out basically
  how can we make sure that it''s fair. (full verbatim + subsequent design decisions
  in the ## Provenance section)'
workflow: v1
goal: 'Date the formation of the context→answer activation map along a 15-rung Olmo-3-7B
  pretraining ladder (stage1-step0 through 5.93T tokens, plus midtrain, final base,
  and SFT/DPO/RLVR): (1) the formation curve D(T) of diagonal held-out R2 on each
  checkpoint''s own completions, against the identity+learned-bias baseline, kNN retrieval,
  and a per-rung repeat-generation ceiling; (2) a reduced cross grid separating representation
  change (fixed answer text) from answer-distribution change (fixed weights); (3)
  cross-checkpoint transfer/retention, direct and general-linear/Procrustes-aligned
  against matched nulls, to find the token count past which the final map is already
  present up to a change of coordinates; (4) the in-context-substitution curve, 4-shot
  minus 0-shot diagonal at every rung, with query-block-only context pooling.'
---
# How the context→answer map forms across pretraining (Olmo 3 7B checkpoint ladder)

## Goal

Date the formation of the context→answer activation map along a 15-rung Olmo-3-7B pretraining ladder (stage1-step0 through 5.93T tokens, plus midtrain, final base, and SFT/DPO/RLVR): (1) the formation curve D(T) of diagonal held-out R2 on each checkpoint's own completions, against the identity+learned-bias baseline, kNN retrieval, and a per-rung repeat-generation ceiling; (2) a reduced cross grid separating representation change (fixed answer text) from answer-distribution change (fixed weights); (3) cross-checkpoint transfer/retention, direct and general-linear/Procrustes-aligned against matched nulls, to find the token count past which the final map is already present up to a change of coordinates; (4) the in-context-substitution curve, 4-shot minus 0-shot diagonal at every rung, with query-block-only context pooling.

## Motivation

The context→answer map (a linear read from a context's pooled hidden state to the pooled hidden state of the model's own answer) is established on Qwen-2.5-7B in [#722](https://eps.superkaiba.com/tasks/722), [#779](https://eps.superkaiba.com/tasks/779) and [#1092](https://eps.superkaiba.com/tasks/1092), shown to be present in the pretrained base and transportable across the base→instruct boundary under a general-linear reparameterization in [#825](https://eps.superkaiba.com/tasks/825), and traced across a four-stage post-training chain in [#1902](https://eps.superkaiba.com/tasks/1902), where SFT does nearly all the rewriting and DPO/RLVR leave the map largely intact.

#825's headline is that the map is inherited from pretraining. That result treats pretraining as a single event. This task opens it up: at what point during pretraining does the map appear, does it appear all at once, and from what point on is the final map already present up to a change of coordinates.

This is the pretraining-axis twin of #1902, on the same measurement rig and the same corpus, so the two curves join end to end.

## Design

### Model and ladder

`allenai/Olmo-3-1025-7B`. Verified inventory (2026-08-24): 1,487 branches, of which 1,421 are stage-1 pretraining checkpoints at `stage1-step0` … `stage1-step1413814`, spaced every 1,000 steps; 49 stage-2 midtraining plus 3 data-mix ablations; 13 stage-3 anneal. One step = 4,194,304 tokens (1,024 × 4,096), confirmed two ways: OLMo-2's branch names carry step and token counts and give 4.198M/step, and 1,413,814 × 4.194M = 5.93T, matching the card's stated training-token count exactly. Checkpoint availability is therefore not a constraint; the grid resolution is one checkpoint every 4.2B tokens.

Selected as full weights at every revision, including `stage1-step0` (random init).

15 rungs, log-spaced in tokens because any formation event is expected in the first few percent:

| # | Branch / model | Tokens | Role |
|---|---|---|---|
| 0 | `stage1-step0` | 0 | random-init floor control |
| 1 | `stage1-step1000` | 4B | |
| 2 | `stage1-step5000` | 21B | |
| 3 | `stage1-step21000` | 88B | |
| 4 | `stage1-step60000` | 252B | |
| 5 | `stage1-step143000` | 600B | |
| 6 | `stage1-step286000` | 1.20T | |
| 7 | `stage1-step596047` | 2.50T | |
| 8 | `stage1-step954000` | 4.00T | |
| 9 | `stage1-step1413814` | 5.93T | end of pretraining |
| 10 | `stage2-step47684` | +200B | midtraining complete |
| 11 | `main` | 6.18T | final base |
| 12 | `Olmo-3-7B-Instruct-SFT` | | post-training, joins #1902's axis |
| 13 | `Olmo-3-7B-Instruct-DPO` | | |
| 14 | `Olmo-3-7B-Instruct` | | RLVR |

Post-training lineage verified on the Hub: `Olmo-3-1025-7B` → `-Instruct-SFT` → `-Instruct-DPO` → `-Instruct`. Architecture 32 layers / 4,096 hidden, the same tensor shape as the OLMo-2 chain #1902 used, so the #1902 rig runs on a revision list rather than new capture code. Round 1 is a locator: if the map forms between two rungs, that decade gets densified in a follow-up round.

**Model choice, recorded.** `allenai/Olmo-Hybrid-7B` (2026-01-28, 1,419 pretraining checkpoints) is newer and Ai2 claims it beats Olmo 3 outright, but 75% of its layers are gated DeltaNet linear attention, which (a) confounds the pretraining axis with an architecture change relative to every prior result in this line, (b) makes the layer axis periodic at 3:1, and (c) is unsupported by the pinned stack (vLLM 0.11.0 registers `Olmo3ForCausalLM` but not `OlmoHybridForCausalLM`; transformers 4.57.6 registers `olmo3` but not `olmo_hybrid`), so it would force a transformers 5.x / vLLM 0.27.x upgrade affecting every other rig in the repo. Deferred to a follow-up, where "does the map form the same way in a linear-attention model" is a cleaner second question with this run as its control.

### Corpus

Reuse the LMSYS single-turn corpus built for #1902 (already on the HF data repo under `issue1902_stage_map/corpus/`), so the pretraining curve and the post-training curve are one continuous measurement on identical rows.

**Rejected alternative, recorded.** A document-continuation corpus (context = first half of a held-out web document) was considered as the in-distribution-at-every-rung option. Rejected because #825 already measured that construct in the fully trained model and found nulls: generic stories R² ≈ 0.16 on-policy and off-policy, generic next-span prediction (punctuation to next-punctuation span) R² ≈ 0.05/0.1, against 0.588 base / 0.673 instruct on chat QA. A corpus whose construct is absent in the final model cannot date the formation of the chat map.

### Render, and the k-shot arm

No chat template at any rung. Plain `User: {q}\nAssistant: {a}`, the render #1902 captured under and the one #825 showed the base model slightly prefers (+0.04 R² over the chat template). Uniform across all rungs, so serialization never varies with the checkpoint.

The residual asymmetry is that `User:/Assistant:` is itself a learned format, so early rungs are measured off-distribution in a way late rungs are not. The k-shot arm addresses it: `k = 4` fixed exemplar QA pairs prepended, chosen because #825's turn-4 crossover (base matches instruct by turn ~4; "a few turns of in-context conversation substitute for" post-training's advantage) is the closest grounded value available rather than a guess.

In-context learning is itself emergent (induction-head phase change, Olsson et al. 2022), so this does not remove the developing-capability asymmetry, it exchanges format familiarity for ICL. That is deliberate: the 0-shot to k-shot gap per rung becomes the in-context-substitution curve, a second result rather than only a control.

Two requirements on the k-shot arm:

- **The context vector pools over the query block only**, exemplar tokens excluded. The exemplars are identical across every row, so pooling over the whole prompt injects a constant into every context vector and compresses between-row variance, moving R² mechanically. The #1902 rig already separates context-window from prefix-window pooling, so this is configuration, not new code.
- **Exemplars are pinned and seeded**, with two alternate sets on a subset as a robustness arm; weak models are highly sensitive to exemplar choice and order. Exemplar style leaks into answer length and format rung-dependently, so that goes into the per-rung diagnostics.

### Cells

| Arm | Cells | Isolates |
|---|---|---|
| 0-shot diagonal (`m = s`) | 15 | each checkpoint's map on its own completions |
| 0-shot fixed text (`s` = final base, `m` varies) | 14 | representation change alone: identical tokens and positions, only the weights differ |
| 0-shot fixed weights (`m` = final base, `s` varies) | 14 | answer-distribution change alone: identical weights, only the text differs |
| k-shot diagonal, all rungs | 15 | format control, and the in-context-substitution curve |
| k-dose sub-probe, `k ∈ {0, 1, 4, 16}` at 3 rungs | 9 | is `k = 4` doing the work |

58 cells, against 225 for the full 15×15 grid. #1902 ran its full 4×4 because K=4 made it cheap; at K=15 the reduced cross buys the same three reads.

Deferred: one more cell per rung (k-shot weights scored on 0-shot text) would separate representation from text *within* the k-shot arm. Worth buying only if the 0-shot to k-shot gap turns out to be the headline.

## Formalization

Let `M(T)` be the checkpoint at pretraining token count `T` over the ladder above, and `{x_i}` the fixed set of single-query contexts. For weights `m`, answer source `s`, and render `ρ ∈ {0-shot, 4-shot}`:

- `a_i^{s,ρ}` is the answer `M(s)` generates for `x_i` under `ρ`
- teacher-force `M(m)` over `ρ(x_i, a_i^{s,ρ})`; at layer `ℓ`
  - `u_i` = mean residual-stream state over the query-block tokens (exemplar tokens excluded)
  - `w_i` = mean residual-stream state over the answer tokens
- `h: u → w` is fit by ridge over the row pairs with cluster-grouped 6-fold cross-validation, identical fold assignment across every cell

The dependent variable is held-out R², reported in every cell alongside the identity-plus-learned-bias baseline (`analysis/mapping_baselines.identity_bias_predict`), kNN retrieval (`mapping_baselines.knn_retrieval`, euclidean and cosine, chance stated), and that rung's repeat-generation reliability ceiling.

Four quantities:

1. **Formation curve** `D(T)`: diagonal R² at a fixed shared layer, raw and baseline-subtracted.
2. **Decomposition**: `R²(m = T, s = final)` for representation change alone; `R²(m = final, s = T)` for text change alone.
3. **Retention** `ρ(i→j)`: transfer R² (fit on rung `i`, evaluated on rung `j`'s held-out pairs) over rung `j`'s own diagonal R², direct and after general-linear and orthogonal-Procrustes alignment, against shuffled-correspondence and spectrum-matched nulls. Adjacent transitions and long-range (each rung to the final base).
4. **In-context substitution** `Δ(T)`: 4-shot diagonal minus 0-shot diagonal at rung `T`.

### Hypotheses

- **H1, early formation.** `D(T)` rises steeply and saturates well before 5.93T, with most of the final map strength reached in the first ~10% of tokens. The pretraining-axis form of #825's inheritance result.
- **H2, crystallize then reparameterize.** Aligned retention to the final base is low before some token count `T_c` and high after, so past `T_c` further pretraining only changes coordinates on a map that already exists. This is #825's base→instruct finding moved one axis over, and #1902's median adjacent-transition aligned retention of 0.893 is the post-training comparison point.
- **H3, substitution decays.** `Δ(T)` peaks at intermediate `T` and shrinks as the weights absorb the capability, and is near zero at `T = 0` where in-context learning does not yet exist.

### What kills each

- **Answer quality alone.** If the curve is nothing but the text getting better, the fixed-text column is flat and the fixed-weights row carries all the movement. The cross grid separates them by construction.
- **Copying.** Early checkpoints echo the prompt, so the answer state approaches the context state and the map is trivially the identity. Detected by ridge failing to beat the identity-plus-bias baseline.
- **Answer-cloud collapse.** If early answers degenerate to one repetition, the variance the R² denominator needs disappears. Detected by kNN retrieval falling to chance while R² still looks respectable; the two dissociate in both directions (#722, #779).

## Fairness controls

1. **Reduced cross grid** (above): separates representation change from answer-distribution change, which is the central confound of the pretraining axis.
2. **Identity-plus-bias baseline and kNN retrieval in every cell.** Mandatory in this project, and the primary read for rungs 0 through 2 rather than a formality.
3. **Per-rung degeneracy diagnostics**: repetition rate, distinct-token ratio, answer-cloud effective rank and mean pairwise cosine, mean answer length, generation-cap-hit fraction.
4. **Shared non-degeneracy intersection**: the headline curve is computed on rows surviving the filter at *every* rung, so it is not driven by differential row dropout. Generalizes #1902's four-source unflagged intersection.
5. **Per-rung noise ceiling**: two repeat generations on a pinned 1,000-row subset per rung. Early rungs are higher-entropy and their R² is capped lower for reasons unrelated to the map, so every rung's R² is read against its own ceiling, never against 1.
6. **Layer discipline**: a fixed shared layer for the like-for-like curve, plus per-rung best-layer against a selection-symmetric null. Naive per-rung best-of-17 would fake an early rise, because the max is biased upward most where the curve is flattest, which is exactly the early rungs.
7. **Scale drift**: hidden-state norms grow substantially over 6T tokens. Within-cell, ridge on standardized features with per-cell λ under a dof cap (never pure GCV at n < d, #1887). Across-cell, transfer is reported direct, general-linear-aligned and Procrustes-aligned against matched nulls (#825, #1345, #1902 conventions).
8. **Constant tokenizer and vocabulary** across the entire ladder, so context and answer vectors live in comparable spaces by construction. This is the structural advantage of a within-run ladder over any cross-model comparison.
9. **`n_train` versus `d`**: d = 4,096; 6-fold CV over the ~16k-row corpus gives n_train ≈ 13.6k, clearing the n ≥ d requirement with margin.

## Reuse

The #1902 pipeline is parameterized on a checkpoint list plus a revision-pin map and takes `revision=` on every model and tokenizer load, so the ladder is a configuration change:

- `scripts/issue1902_common.py` (checkpoint ids, binding revision pins, capture-layer set = every 2nd layer plus the final, store layout, render helpers, degeneracy filters)
- `scripts/issue1902_corpus.py`, `issue1902_run.py` + `issue1902_dispatch.sh` (generation and capture), `issue1902_fits.py` (ridge, MLP, transfer, alignment, operator battery), `issue1902_figures.py`
- `src/explore_persona_space/analysis/mapping_baselines.py` (identity-plus-bias, kNN retrieval), `paired_ci.py`, `null_battery.py`, `vectorized_mlp_skill.py`
- the #1902 LMSYS corpus and its cluster assignment, on the HF data repo

New code is confined to the k-shot render and its query-block pooling window, the rung ladder, and the per-rung degeneracy diagnostics.

## Rough size

58 capture cells plus roughly 30 generation passes, order 150 GPU-hours, ~14 GB download per rung and on the order of 130 GB of activation tensors. Pod work throughout; the per-rung download alone puts this over the shared-VM threshold. Incremental cache reaping between rungs is required. The planner sizes this properly; the figure here is a locator, not a budget.

## Open decisions for the planner

1. **MLP arm.** Assumed out, per the project's linear-by-default rule. #1902 ran both, and the originating post-training prompt asked for both, so this is a live question rather than a settled one.
2. **Rung densification policy.** Round 1 is a locator. State up front what triggers a fill-in round and where.
3. **Corpus size.** #1902's 18,000-row build intersected down to 16,391 usable rows across four sources. Across 15 rungs the shared non-degeneracy intersection will be smaller, so the row budget needs sizing against the n ≥ d floor before launch, not after.
4. **Literature review.** Required before implementation, per the standing project rule. The nearest prior work is training-dynamics-of-representations on open checkpoint ladders (induction heads and in-context learning, Pythia-based dynamics work, Ai2's own OLMo analyses). Nothing found so far measures a context→answer activation map across a pretraining ladder, but that is an unverified impression and needs `/deep-lit-review` to settle.
5. **Whether the k-shot arm should also cover the cross grid** rather than the diagonal only (+28 cells).

## Provenance

Designed interactively in chat on 2026-08-24. Originating prompt, verbatim:

> I want to run an experiment to see how our mapping evolves throughout pretraining. Similar to the experiment where we were seeing how it evolves throughout post training. I want to see both the individul checkpoint mappings trained on its own completions and the transfer to subsequent checkpoints. Please help me to design this experiment. First we need to find the we need to find a model that we can use that is ideally a very recent model that's very good with this, and then we want to figure out basically how can we make sure that it's fair. Like I guess we want to not use chat templates here because with the base model it showed that it wasn't very good at that. But yeah, I guess I mean I guess we don't have to make sure it's coherent. It doesn't have to be coherent. Um, yeah

Subsequent decisions in the same conversation: Olmo 3 7B over OLMo-2-1124-7B and Olmo-Hybrid-7B ("let's do olmo 3 7b"); the 15-rung ladder accepted as proposed ("this looks good"); the k-shot arm proposed by the user ("can't we give in context examples even at start?") and extended to every rung ("i think we should always have the k-shot arm").
