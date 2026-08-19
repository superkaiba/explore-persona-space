---
title: 'Do the #2162 context-vector findings hold on Qwen3.8-27B? (third model, 26-cell
  subset)'
kind: experiment
tags: []
created_at: '2026-08-19T17:01:28Z'
has_clean_result: false
parent_id: 2329
origin_prompt: let's try qwen3.8 27b
workflow: v1
---
# Experiment: Do the #2162 context-vector findings hold on Qwen3.8-27B? (third model, 26-cell subset)

## Goal

Test whether the #2162 minimal-pair context-vector verdicts — which of 21 information types are linearly decodable at a single context position, and which are causally usable when that position's hidden state is transplanted between paired contexts — hold on `Qwen/Qwen3.8-27B`, a third model and a substantially more capable one than either prior point. Primary registered read: the transfer correlation (Spearman rho between this run's and #2162's per-type steered fraction-of-swap over the 31 shared P1 cell-slot units, pair-clustered bootstrap 95% CI). Secondary: whether the stronger instruction-following of this model rescues cells that the anchor-separation exclusion rendered untestable on the two earlier models.

## Provenance

Originating chat request (verbatim): "let's try qwen3.8 27b", following a model-selection dive that compared Qwen3.6/3.7/3.8, Olmo-3.1-32B, gemma-4-31B, GLM-5.2 and the DeepSeek-V4 family.

Lineage: #2162 (Qwen2.5-7B-Instruct, the original 39-cell design) -> #2329 (Qwen3.5-9B, thinking disabled; transfer rho = 0.831, CI [0.583, 0.864] over 31 shared P1 units) -> this task (Qwen3.8-27B, third point).

## Why this model

- Most capable model that fits the rig's one-replica-per-GPU design. Its own card reports SWE-bench Pro 61.7, LiveCodeBench 90.3, OSWorld 84.3 and IFBench 79.5, beating Opus 4.6 Max on each; and it is a large jump over Qwen3.6-27B (DeepSWE 13.3 -> 42.2, QwenSWEBench 49.3 -> 79.0).
- **IFBench 79.5 vs Qwen3.6-27B's 69.1 is the load-bearing number for this experiment.** Instruction-following strength drives the anchor-separation gate: the ceiling-minus-floor gap only exists when the model obeys the varied instruction. The 7B -> 9B step already moved first-quartile separation 0.346 -> 0.555 and rescued six cells; this model should rescue more.

**Known limitation, stated up front:** Qwen3.8-27B carries architecture `qwen3_5` — the same family as Qwen3.5-9B. This is a fourth Qwen, so it does NOT address the lineage confound ("these verdicts are a Qwen-family property"). It answers the scale/capability question, not the generality question. A lineage-varying run (Olmo-3.1-32B was the candidate) remains the complementary experiment.

## Design

Single varied factor vs #2329: the model. Bank text, F definitions, arms, judge instrument, exclusions and statistics all inherited verbatim.

### Cell subset (26 of 39)

Chosen to preserve the registered transfer test at full power. All 16 P1-family cells are required — they are what make the 31-unit rho denominator; a subset built only from the causally-positive cells preserves just 11 of 31 and guts the headline statistic.

- **16 P1 cells** (the rho denominator): `constraint_knowledge`, `fact_assistant_animal`, `fact_novel_queried`, `fact_user_name`, `icl_task_mapping`, `instr_format`, `instr_language`, `list_numeric_detail`, `persona_prompted`, `prior_topic`, `query_content`, `reasoning_style`, `refusal_boundary`, `user_emotion`, `user_expertise`, `verbosity`
- **+6 causally-positive cells outside P1**: `conflict_format_fwd`, `conflict_format_rev`, `language_implied`, `load_instr_format_l3`, `load_instr_format_l5`, `recency_instr_format_d3`
- **+3 verdict-unstable between the two prior runs**: `demo_persona`, `load_fact_user_name_l5`, `recency_prior_topic_d5`
- **+1 control already inside P1**: `filler_swap` (disruption control; `query_content` is the rig-sensitivity control and is already a P1 cell)

**Dropped (13):** eleven recency/load variants that were stable nulls in both prior runs, plus `demo_format` and `persona_role_header` — chronically unmeasurable regardless of model (5/36 and 2/36 surviving pairs on the 9B, median separations 0.10 and 0.04). Their weakness is bank-design, not capability, so a stronger model does not rescue them. The report must state the 26-of-39 restriction and name the dropped set.

### Realized model constants (verified from the Hub, 2026-08-19)

| Constant | #2329 value | This run | Source |
|---|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.8-27B` | — |
| `N_MODEL_LAYERS_FULL` | 32 | **64** | `config.json` `text_config.num_hidden_layers` |
| `HIDDEN_FULL` | 4096 | **5120** | `text_config.hidden_size` |
| vocab | 248,320 | **248,320 (identical)** | `text_config.vocab_size` |
| attention heads / KV heads | — | 24 / 4 | `text_config` |
| `max_position_embeddings` | — | 262,144 | `text_config` |
| full-attention layers | 8 (every 4th of 32) | **16 (every 4th of 64: 3,7,...,63)** | `full_attention_interval: 4`, `layer_types` |
| `F_ACT_READ_LAYER` | 30 | **59** (proposed; fraction-match of the original 26/28 = 0.9286) | decision — see Open questions |
| model class | `AutoModelForMultimodalLM` | same (`Qwen3_5ForConditionalGeneration`) | `config.json` `architectures` |

Params 27.78B; bf16 weights ~55.6 GB — fits one H100 80GB with headroom for KV cache and all-layer capture, comfortable on a 143 GB H200.

### Port assessment (Tier 2, but with unusual reuse)

**Ports for free:**
- **Tokenizer is identical to Qwen3.5-9B** (model vocab 248,320; BPE 248,044; `Qwen2TokenizerFast`). The #2329 frozen bank was already re-tokenized and verified 1,404/1,404 pairs intact under this exact tokenizer, so **P0 collapses from a bank re-freeze to a verification pass**.
- **Thinking seam is identical.** `enable_thinking=False` yields the generation prompt `<|im_start|>assistant\n<think>\n\n</think>\n\n` — byte-identical in shape to #2329's realized prompt. Verified by rendering.
- Same `qwen3_5` architecture family, so #2329's text-only load path (dropping the vision tower and MTP head) applies unchanged.
- ChatML markers `<|im_start|>` / `<|im_end|>` unchanged, so the #2162 turn-boundary boundary-resolver would port with an id change if that follow-up is ever repeated here.

**Needs work:**
- **No default system turn** (verified by rendering: a bare single-turn context emits no system block). The #2329 `no_prefix` prefix-end exclusion therefore recurs — all 36 `persona_role_header` contexts and the 12 empty-system `persona_prompted` v2 contexts. Note `persona_role_header` is already dropped from this subset, so the residual bite is the 12 persona contexts. See Open questions for the two handling options.
- 64 layers and hidden 5120 vs 32/4096: the state bank grows to roughly 1.8 GB (1,404 contexts x 2 slots x 64 layers x 5120 x bf16), and the all-layer patch installs 64 hooks instead of 32.
- Stage-2 layer set must be re-mapped onto the 64-layer stack, with members shifted onto full-attention layers as #2329 did.
- Pod-side `transformers` pin must be confirmed to support `qwen3_5` at this config (#2329 pinned 5.15.0 pod-side while the VM stays at the repo-locked 4.57.6; the VM tokenizer loads fine, verified).

## Compute + cost estimate

- Grid: 26 cells x 36 pairs x 2 slots x 3 arms x 5 draws = **28,080 rollouts** (vs 42,120 full).
- Anchors: ~9,360 (26/39 of 14,040). Stage-2 scales with survivors.
- Wall: #2329 ran 42,120 grid + 14,040 anchors in ~17 h on 8x H100 with a 9B. At 0.67x the cells and roughly 3x the per-rollout cost of a 27.78B model, expect **~35-40 h on 8 GPUs**, plus queue.
- Backend: **`fellows` (charmander, 8x H200 143 GB, free)** proposed. 27.78B bf16 fits one card, so all 8 replicas per node survive. Contended — 14 of 112 GPUs idle against 72 pending jobs when checked 2026-08-19 — so queue time is the real variable. Paid fallback: 8x H100 RunPod at ~$26/hr (~$1,000 for the run); note 8x H200 and 8x B200 packs were not purchasable on RunPod when checked.
- Judging: ~67% of #2329's ~212k calls, so **~142k calls, roughly $400-800**.
- **Total marginal cost on the free lane: judge spend only.**

## Pre-flight

- Confirm the HF data repo overflow reroute engages before committing ~35 h of generation — `issue2329_run.py` records the canonical data repo at HF's 1,000,000-file cap (#2304 shipped the automatic reroute and merged 2026-08-16, but #2329 still wrote to the main repo).
- Pin the model revision explicitly (`revision=`). #2329 recorded a reproducibility gap here: it resolved `main` and the generating pod is gone, so the weights it used are not provable from the artifacts. Do not repeat that.
- Land #2329's PR #2004 first if the transfer read is to cite its committed tables — its `eval_results/` are still branch-only.

## Open questions for the user (defaults applied; each is a one-line change)

1. **Cell subset** — default 26 cells as above. Full 39 costs 42,120 grid rollouts (1.5x) and buys back thirteen cells that were stable nulls or unmeasurable.
2. **Prefix-end handling** — default is to INHERIT #2329's `no_prefix` exclusion verbatim, because matching exclusions makes the three-way transfer comparison cleaner. The alternative is the #2162 ladder round's fix (pass an explicit system prompt on every context, verified by a `plain_render_equality` probe), which recovers coverage but makes this run's prefix-end slot non-comparable to #2329's.
3. **`F_ACT_READ_LAYER`** — default 59 (fraction-match of the original 26/28). Matching #2329's 30/32 fraction instead gives 60. All-layer profiles are persisted either way, so this is only the headline read layer.
4. **Backend** — default `fellows` (free, queued). Switch to RunPod 8x H100 if wall-clock matters more than ~$1,000.
