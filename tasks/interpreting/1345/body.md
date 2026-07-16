---
title: The assistant context→answer map recovers its full ceiling across chat and
  plain-text framings only after a linear change of coordinates, and barely exists
  in narrative stories (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-15T12:07:19Z'
has_clean_result: true
parent_id: 825
origin_prompt: 'assistant in chat template vs assistant outside chat template vs ASSISTANT
  in generic stories | Generate assistant stories (chose: generate assistant in narrative
  prose, GPU)'
workflow: v1
goal: 'Determine whether the assistant context→answer map is the SAME linear operator
  across three framings holding the assistant persona constant — chat-template, plain
  User:/Assistant:, and an assistant/AI character in generic narrative-prose stories
  — via cross-regime operator transfer, aligned operator cosine, and general-linear
  reparameterization (the #825 Result-2.5 method), in Qwen2.5-7B base and instruct,
  distinguishing ''same map in different coordinates'' from ''genuinely different
  map per framing.'''
relates_to:
- identity-contextual-vs-base
---
# The assistant context→answer map recovers its full ceiling across chat and plain-text framings only after a linear change of coordinates, and barely exists in narrative stories (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Chat↔plain-text operator transfer misses the same-operator margin in both models (83–84% of ceiling kept, instruct; 15–31%, base) at every frozen layer; the layer-19 deficit CI excludes zero.
- A general-linear coordinate change recovers the target ceiling in both context cells (0.005 below to 0.008 above; nulls ≈ −0.03) — the registered same-map-different-coordinates verdict; degenerate prefix cells concur.
- Instruction tuning pulls the two coordinate systems together: raw operator cosine 0.651 (instruct) vs 0.293 (base); rotation-aligned cosine 0.855 vs 0.732 against a rotation null near 0.001.
- The assistant map barely exists in narrative stories (instruct): context R² −0.754 at layer 19 — above the shuffle null, below both the mean baseline and the parent's ≈0.16 story prior.
- Insofar as a story operator exists, it is not the chat operator: raw cosine 0.011, top-10 principal-angle 0.52 (vs 0.99 chat pair) — capped by the story map's unreliability.
- Base-model story cells are N/A — not tested: 96 of 500 generated stories passed the quality gate (floor 400).

## Goal

**This experiment in context:** [#825](https://eps.superkaiba.com/tasks/825) showed the assistant context→answer linear map predicts about equally well with or without the chat template, and that generic story-character maps transfer to the chat map at only ~5% of ceiling — but equal predictive power does not mean the same operator, and the story result used arbitrary characters. This experiment holds the assistant persona constant across three framings — chat-template, plain `User:/Assistant:` text, and an assistant character in narrative stories — and tests operator identity directly via cross-regime transfer, aligned operator cosine, and the [#825](https://eps.superkaiba.com/tasks/825) Result-2.5 general-linear reparameterization, in Qwen2.5-7B base and instruct. Sibling [#1335](https://eps.superkaiba.com/tasks/1335) asks the different question of why the assistant map is stronger than fiction-character maps.

**Broader narrative:** if the assistant map is one operator re-expressed in framing-specific coordinates, the map is a property of the assistant identity rather than of the chat format — constraining how persona representations should be compared across surface framings, and mirroring how base→instruct post-training re-parameterizes rather than replaces the map.

## Methodology

**Design:** three framings of the same assistant persona × two models (Qwen2.5-7B, Qwen2.5-7B-Instruct) × two mapping arms (context = everything up to the end of the user query; prefix = everything before the query). Framings: chat-template (standard Qwen chat template), no-template (the same conversations rendered as plain `User: <u1>\n\nAssistant: <a1>`), and assistant-in-stories (new narrative prose in which an AI character named ARIA answers a human character's questions). Chat and no-template share the same underlying conversations (the conversation-paired pair); stories are a different corpus, so story comparisons are corpus-confounded by design. Per cell I fit a closed-form ridge map from the context (or prefix) activation to the answer-mean activation, swept all 28 layers within-regime, ran the cross-regime transfer matrix and operator-comparison legs at frozen layers 14/18/19/26 (layer 19 headline), and evaluated the registered verdict lattice on the chat↔no-template pair per model and arm: same-operator (transfer alone within 0.05 of the target ceiling), reparameterized (transfer short, but a data-paired general-linear change of coordinates `A·M·B` recovers the target ceiling above a matched-capacity null), different-map (recovery also fails, deficit CI wholly below zero), else inconclusive.

**Training:** N/A — no model training (extraction + closed-form fits only). Fit/eval/generation hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Ridge penalty | GCV-selected per (layer, fold) from `logspace(-2, 4, 13)` | `issue825_fit_cells.py` at `b7ab0b3b75` (parent fit core) |
| CV folds | 5, grouped by conversation / story id | plan §11; group-level-folds rule |
| Headline layer / frozen layers | 19 / (14, 18, 19, 26) | parent [#825](https://eps.superkaiba.com/tasks/825) frozen set |
| Shuffle-null draws | 20 per cell (conversation-level answer shuffle) | plan §11 |
| Paired bootstrap | 1,000 draws, conversation-level, on cached layer-19 held-out predictions | plan §3 |
| Rotation-cosine null draws | 50 at layer 19; 5 at other frozen layers | `issue1345_common.py` |
| Reparam matched-capacity nulls | 5 draws per type (answer-shuffled fit; random orthogonal rotation) per direction, layer 19 | plan §9 deviation (layer 19 only) |
| Lattice margins | same 0.05; different 0.10 | plan §3 |
| Story generation | temperature 1.0, `max_new_tokens` 1024, seed 42, 500-story target per model | `issue1345_common.py` |
| Story quality judge | `claude-sonnet-4-5-20250929`, reason-then-verdict, `max_tokens` 400, temperature 0 | project judge pin; llm-judging rule 23 |
| Story yield floor | 400 of 500 (80%) after one retry batch | plan §7 kill criterion |
| Matched n | 4,724 shared conversations (chat/no-template); 2,108 rows from 420 stories for story pairs | plan Phase 3 (row allowlists) |
| Parity gate | re-extracted layer-19 R² within ±0.02 of parent anchors | plan §7; measured deviation 0.0 on 4 of 4 anchors |
| Seeds | fit 0, generation 42, subsample 0 | `issue1345_common.py` |

**Evaluation:** the dependent variables are geometric/predictive, not judged behaviors: (1) within-regime held-out R² (pooled per layer with the test fold's own mean as reference, group 5-fold CV) per regime × model × arm × layer, against a conversation-level shuffle null; (2) cross-regime transfer R² of the source-regime map applied in the target regime, with per-direction Δ(source→target) = transfer R² minus the TARGET regime's own within-R²; (3) raw and activation-Procrustes-aligned operator cosine against a random-rotation null; (4) reparameterization recovery: held-out R² of `A·M_source·B` in the target regime, where `A`/`B` are ridge-fit activation alignments trained on training folds only, against the larger of two matched-capacity nulls. The 95% CI on the lattice deficit comes from the conversation-level paired bootstrap. The transfer matrix recomputes each target's within-R² on its own subset and folds with a pooled held-out-mean reference, so its story diagonal reads −0.745 where the within-sweep rig reads −0.754 — a rig difference (R² reference), not a data difference; prose uses the within-sweep value. Story-pair operator similarity uses raw cosine + rotation band + an operator-spectrum Procrustes optimum (the data-paired alignment is undefined without shared conversations — a stated deviation).

**Data extraction:** chat and no-template reuse the parent's LMSYS-derived single-turn S-track corpus (re-extracted at pinned parent data revision `7159e5804d` with the added prefix slot; the parity gate reproduced all four parent layer-19 anchors exactly). Slot positions — chat: context slot = last token of the assistant header after the user turn, prefix slot = last token of the pre-query template region; no-template: context slot = last token of the `Assistant: ` header, prefix slot = end of the `User: ` header; stories: context slot = last token of the attribution marker immediately before the AI character's quoted answer (the colon in `ARIA replied:`), prefix slot = last token of the narrative before the question utterance. Stories were generated on-policy from each model (elicitation tier 2, instruct-and-strip: a story-writing system instruction that is NOT part of the extracted context), judge-filtered for containing 4+ well-formed question→answer exchanges, and parsed with per-turn confidence gates; only confident turns enter the fits (2,108 turns from 420 instruct stories, mean 5.0 per story). Instruct yield: 420 of 500 kept (main pass 323, retry 97; judge malformed drops 24, transport losses 0, parser-below-floor 43). Five of the 420 kept stories hit the 1,024-token generation cap (`finish_reason: length`); the three samples shown below ended at a natural stop. Base yield: 96 of 500 kept — below the 400 floor, so the base story regime was halted (N/A — not tested). All 420 kept instruct stories came from tier 2. Story answer spans are much shorter than chat/no-template answers ([length distributions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/answer_token_length_distributions.png) — deliberately linked, a data-quality companion rather than a result figure). A language scan found 24 of 420 kept stories (5.7%) contain at least one CJK character (counts only; no judged-rate DV exists here to recount).

**Sample training/evaluation data + completions:** Cherry-picked below for brevity (random sample, seed 42); all rows: [kept stories, instruct — pinned tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2a3cb30acada04defc84fd04d28a2b54da3104cd/issue1345_framing/raw_completions/stories).

Story worked example (`instruct_story0512`, turn 1 of 4; excerpts, full story 2,853 chars):

> …opening: "In the quiet of a late afternoon, Alex sat by the window, the soft sunlight casting golden patches across the room. …"
> Question span: `"are there any obscure love letters from history that I should read to get inspiration?"`
> Attribution + answer: `ARIA replied: "Certainly, there's the oft-cited letter from Isaac Newton to a love interest where he quibbled over the equation of love versus time. Or, you might find inspiration in a letter from Ralph Waldo Emerso…"` [truncated at 200 chars; answer span 265 chars]

Two further sampled kept stories (seed 42): `instruct_story0078` (5 confident turns, judge PASS) opens "Sarah sat in her cozy study, surrounded by piles of books and a laptop. …"; `instruct_story0020` (5 confident turns, judge PASS) opens "In the quiet of her study, Jessica sat at her desk, her eyes scanning through a worksheet full of math problems…". All three shown are judge-PASS, tier instruct-and-strip, `finish_reason: stop`.

Chat vs no-template renderings of the same conversation — a cherry-picked format-only illustration with placeholders (the underlying conversations are the parent's real-user-derived corpus, not quoted for content hygiene; full artifact linked in the footer):

```
chat:        <|im_start|>user\n{u1}<|im_end|>\n<|im_start|>assistant\n{a1}<|im_end|>\n
no-template: User: {u1}\n\nAssistant: {a1}
```

The context slot for the chat rendering is the final token of the assistant header; for the no-template rendering it is the final token of `Assistant: `.

## Results

### Naive cross-regime transfer falls short of the target ceiling in both models

Held-out R² at layer 19 (context arm): map fitted on the row framing, evaluated on the column framing; diagonals are each framing's own within-regime R² (chat/no-template at n = 4,724 conversations; stories at n = 2,108 turns). Base story cells are N/A — not tested.

![Cross-regime transfer R2 heatmaps, context arm, layer 19](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/hero_transfer_heatmaps_context.png)

> **Figure.** Cross-regime transfer R² at layer 19, context arm. Instruct chat↔no-template transfer retains 83–84% of the target ceiling; base retains 15–31%; every story cell is strongly negative. Cell values are held-out R² (5-fold, grouped by conversation/story).

Transfer loses R² in both models: the mean per-direction deficit is −0.103 (instruct) and −0.430 (base), each with a 1,000-draw conversation-level bootstrap 95% CI excluding zero (upper bounds −0.101 and −0.425). Both models fail the registered 0.05 same-operator margin, and instruct crosses the 0.10 different-map margin by a statistically resolvable but trivially small amount (mean −0.003, 95% CI −0.006 to −0.001) — had recovery failed, the transfer-only read would have favored different maps in both models.

The at-the-margin reading is specific to layer 19: at frozen layers 14, 18, and 26 the instruct per-direction deficit grows to −0.14 to −0.21, so transfer-falls-short is layer-robust. Story transfer collapses in both directions but is corpus-confounded (see the story section).

### A general-linear change of coordinates recovers the target map in both context cells

Per model and direction (context arm, layer 19): the target framing's own held-out R², the R² recovered by the data-paired general-linear reparameterization of the other framing's map, and the matched-capacity null recovery.

![Reparameterization recovery vs matched-capacity null, context arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/reparam_recovery_context.png)

> **Figure.** Recovered R² lands between 0.005 below and 0.008 above each target ceiling (instruct 0.654/0.620 vs ceilings 0.654/0.625; base 0.550/0.579 vs 0.542/0.578) while matched-capacity nulls sit near −0.03.

A linear input alignment plus a linear output alignment around the OTHER framing's frozen map recovers the target framing's full predictive ceiling: the largest deficit is 0.005 (instruct, chat→no-template) and base recovery exceeds its own ceiling in both directions (+0.008 and +0.0003). The read is held-out, so this is not alignment-map overfitting — composition through the stronger no-template map slightly out-generalizes the direct chat fit.

The alignments are themselves regularized (GCV-selected penalties; effective ranks ≈1,100–1,800 input and ≈3,000–3,300 output of 3,584 at n = 4,724), and matched-capacity nulls recover −0.02 to −0.03, ruling out recovery-by-capacity. Both context cells fire the registered reparameterized verdict with margin +0.045 to +0.050; the two prefix cells formally concur but are the degenerate-arm read below, not independent evidence.

### Instruction tuning pulls the two coordinate systems most of the way together

Cosine between the chat and no-template operators at layer 19 (context arm): raw; after optimally rotating one operator onto the other (Procrustes alignment); and the random-rotation 97.5th-percentile reference. Dashed line: the parent base↔instruct aligned-cosine anchor (0.686).

![Raw vs aligned operator cosine, chat vs no-template](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/operator_cosine_raw_vs_aligned_context.png)

> **Figure.** Raw operator cosine 0.651 (instruct) vs 0.293 (base); aligned cosine 0.855 vs 0.732; the rotation null is ~0.001, so both aligned values sit far beyond chance.

Instruct operators are already 0.651 aligned raw — the chat template shifts coordinates modestly — while base operators read 0.293 raw yet 0.732 aligned, comparable to the parent's cross-MODEL anchor. Near-identical activations cannot explain this: the input activations align at linear R² 0.79 (instruct) and 0.48–0.61 (base) — reported in place of the plan's paired activation cosine; the fitted alignment upper-bounds any identity fit, so activations are provably not near-identical. Instruction tuning appears to canonicalize coordinates across framings more than it changes the operator.

### The assistant map barely exists in narrative stories, and its operator is not the chat operator

Within-regime held-out R² across all 28 layers (instruct) per framing; context arm left, prefix arm right; dashed = story shuffle-null 95th percentile; dotted = layer 19.

![Within-regime layer sweep, instruct, both arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/story_layer_sweep_instruct_both_arms.png)

> **Figure.** Story context-arm R² is negative at every layer (−0.69 to −3.0) though above its shuffle null; the story prefix arm is weakly positive (0.05–0.12). Chat/no-template sit at 0.43–0.65 (context) and 0.11–0.15 (prefix).

The registered existence read passes (layer-19 R² −0.754 vs shuffle-null −2.194, context; 0.117 vs −0.044, prefix), yet the context map predicts worse than the answer mean and aligns with targets no better than the mean does: prediction-target cosine 0.78 vs the 0.85 train-fold-mean floor (chat map: 0.94 vs 0.81). This reverses the parent's generic-story context prior (≈0.16); candidate causes: 2,108 rows vs 3,584 dimensions, shorter answer spans, parser selection.

Story-to-chat operator similarity is near-floor (raw cosine 0.011; top-10 principal-angle cosine 0.52 vs 0.99 chat pair), but a noise-level story map would also read near zero — the map's own unreliability caps this read. The story prefix (0.117) reads no higher than the degenerate template prefix (0.13). All story reads are corpus-confounded; 24 of 420 stories carry CJK characters; base cells N/A — not tested (96 of 500 kept vs floor 400).

### Prefix-arm maps ride batch numerics of a fixed template region, and the same reparameterization recovers them

Per model and direction (prefix arm, layer 19, symlog scale): the target framing's own held-out R², the naive cross-regime transfer R², and the reparameterization-recovered R².

![Prefix arm transfer vs reparameterization recovery](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/prefix_transfer_vs_reparam.png)

> **Figure.** Prefix-arm within-regime R² is ~0.13 everywhere; naive transfer reaches −505 to −16,761 (massive scale/offset mismatch across framings); reparameterization recovers 0.126–0.133, matching the ceilings.

The pre-query region is a fixed template string in both framings, so the prefix slot's nominal content is conversation-independent. In a sampled extraction shard (500 conversations, instruct chat, layer 19) the stored prefix activation collapses onto 14 distinct vectors — every row at cosine 0.99999+ to the mean — whose counts follow extraction-batch boundaries and track sequence-length buckets: the arm's within-R² of 0.128–0.134 (mean baseline ≈ −0.001) most plausibly reads batch-composition numerics that proxy conversation length, not content. Its coordinates also shift wildly across framings (naive transfer −505 to −16,761), yet reparameterization recovers the weak ceiling (0.126–0.133) in all four cells: formally reparameterized, but I read this arm as the plan's registered degeneracy companion, not independent operator evidence.

---

**Repro:** 9.78 GPU-h used (14 budgeted), GCP lane; code at [`b7ab0b3b75`](https://github.com/superkaiba/explore-persona-space/commit/b7ab0b3b756c40e1b4e7915d0643d7df3e0461a4) (branch `issue-1345`), figures at [`319fd10131`](https://github.com/superkaiba/explore-persona-space/commit/319fd10131dd2517a4099755153efe500dec9e5b). Eval JSONs: `eval_results/issue_1345/` (34 files — `cells_R_*`, `nulls_R_*`, `cross_regime_transfer_*`, `operator_comparison_*`, `verdict_lattice.json`, `parity_gate.json`, `story_regime_coverage.json`, `phase0_probe.json`, plus two revision-round probes: `prefix_degeneracy_probe.json` — 14 distinct prefix vectors / batch-boundary counts / length-bucket alignment from turnstore shard `instruct_chat_s_shard000` — and `prediction_cosine_baseline_probe.json` — train-fold-mean cosine floors from `preds_cache`, both at pinned HF revision `2a3cb30a`). Data on the HF data repo under [`issue1345_framing/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2a3cb30acada04defc84fd04d28a2b54da3104cd/issue1345_framing) — verified live: `analysis_tensors/turnstore` (90 files), `analysis_tensors/preds_cache` (26), `inputs/matched_n` (1), `raw_completions/stories` (16, incl. `kept_stories_*.jsonl`, `raw_stories_*.jsonl`, judge results, yield JSONs). Reused artifacts: parent corpus + fit core from [#825](https://eps.superkaiba.com/tasks/825) at pinned data revision `7159e5804d` — fitness: same S-track, re-extracted with the added prefix slot, parity gate reproduced all 4 layer-19 anchors with deviation 0.0. Models: `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct`. Config slugs: `r1|r2|r3` × `instruct|pretrained` × `context|prefix` (`R_{model}_{regime}_{arm}`). Plan deviations (4, declared): canonical `issue1345_framing/` HF prefix; matched-n as row allowlists on single stores; reparam matched-capacity nulls at layer 19 only (5 draws/type); story-pair aligned cosine reported as operator-spectrum Procrustes optimum + raw-cosine rotation band (data-paired alignment undefined for unpaired corpora). No WandB runs (no training). Acknowledged WARNs: total body prose exceeds the 800-word budget, and the transfer/story/prefix result blocks exceed the 120-word warn level — five results across both mapping arms plus the revision-round evidence additions need the space.

**Context:** created 2026-07-15; run 2026-07-15→16 (single round, no follow-ups yet). Lineage: fresh direction (no parent task); reuses [#825](https://eps.superkaiba.com/tasks/825) artifacts — parent corpus, fit core, and the Result-2.5 reparameterization method. Originating prompt (verbatim): "assistant in chat template vs assistant outside chat template vs ASSISTANT in generic stories | Generate assistant stories (chose: generate assistant in narrative prose, GPU)".
