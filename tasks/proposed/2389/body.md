---
title: 'Do the #2162 context-vector findings hold on Qwen3.8-27B? (third model, 26-cell
  subset)'
kind: experiment
tags: []
created_at: '2026-08-19T17:01:28Z'
has_clean_result: false
parent_id: 2329
origin_prompt: let's try qwen3.8 27b
workflow: v2
goal: 'On Qwen/Qwen3.8-27B (27.78B dense, 64 layers, hidden 5120, thinking disabled),
  measure whether single-position activation patching at the CONTEXT VECTOR becomes
  more effective as model capability rises: the primary read is the number and identity
  of information types that are causally usable at context-end (the stored-and-used
  quadrant) and their per-type fraction-of-swap, against the realized 5 on Qwen2.5-7B
  (#2162) and 8 on Qwen3.5-9B (#2329); the secondary read is the Spearman transfer
  correlation against #2162''s per-type steered fraction-of-swap over the 16 shared
  context-end P1 cells. Prefix-end is deliberately NOT run, having produced zero causally-usable
  cells in both prior runs.'
backend: fellows
---
# Experiment: Does context-vector patching work better on a more powerful model? (Qwen3.8-27B, context-end only)

## Goal

On Qwen/Qwen3.8-27B (27.78B dense, 64 layers, hidden 5120, thinking disabled), measure whether single-position activation patching at the CONTEXT VECTOR becomes more effective as model capability rises: the primary read is the number and identity of information types that are causally usable at context-end (the stored-and-used quadrant) and their per-type fraction-of-swap, against the realized 5 on Qwen2.5-7B (#2162) and 8 on Qwen3.5-9B (#2329); the secondary read is the Spearman transfer correlation against #2162's per-type steered fraction-of-swap over the 16 shared context-end P1 cells. Prefix-end is deliberately NOT run, having produced zero causally-usable cells in both prior runs.

## Provenance

Originating chat request (verbatim): "let's try qwen3.8 27b", then scope narrowed by "i actually just want to see if patching at context vector works better on a more powerful model. so then we don't need prefix".

Lineage: #2162 (Qwen2.5-7B-Instruct, 39 cells x 2 slots) -> #2329 (Qwen3.5-9B, same design; transfer rho = 0.831) -> this task (Qwen3.8-27B, context-end only).

## Why context-end only

Prefix-end has produced **zero** causally-usable cells in every run of this design:

| run | context-end | prefix-end |
|---|---|---|
| #2094 (grandparent) | persona transfers at 0.63 of a full swap | no null-separated behavior family |
| #2162 Qwen2.5-7B | 5 stored-and-used | **0 positive** (30 stored-but-unusable, 7 untestable, 1 absent) |
| #2329 Qwen3.5-9B | 8 stored-and-used | **0 positive** (32 stored-but-unusable, 4 untestable, 1 absent) |

Dropping it halves the grid and removes an entire class of design problems at a stroke: the `no_prefix` template issue (Qwen3.5/3.8 inject no default system turn, so bare contexts have no prefix token), the resulting two-thirds loss of `persona_prompted` prefix-end pairs, and the two constructionally-degenerate-at-prefix-end cells. None of that applies here.

**Cost of the decision, stated:** the transfer correlation's denominator falls from 31 cell-slot units to the **16 context-end P1 cells**. A Spearman on 16 points carries a wider CI than #2329's [0.583, 0.864] on 31. This is accepted deliberately — the transfer correlation is the SECONDARY read here; the primary question is whether more cells become usable at all.

## Why this model

- Most capable model that fits the rig's one-replica-per-GPU design. Card-reported SWE-bench Pro 61.7, LiveCodeBench 90.3, OSWorld 84.3, IFBench 79.5 — beating Opus 4.6 Max on each; a large jump over Qwen3.6-27B (DeepSWE 13.3 -> 42.2, QwenSWEBench 49.3 -> 79.0).
- **IFBench 79.5 vs Qwen3.6-27B's 69.1 is the load-bearing number.** Instruction-following strength drives the anchor-separation gate: the ceiling-minus-floor gap only exists when the model obeys the varied instruction. The 7B -> 9B step already moved first-quartile separation 0.346 -> 0.555, rescued six cells from untestable, and produced three new causal positives. This model should push further.

**Known limitation:** architecture `qwen3_5`, the same family as Qwen3.5-9B. This is a fourth Qwen, so it does NOT address the lineage confound ("these verdicts are a Qwen-family property"). It answers the capability question only. A lineage-varying run (Olmo-3.1-32B) remains the complementary experiment.

## Design

Single varied factor vs #2329: the model. Bank text, F definitions, arms, judge instrument, exclusions and statistics inherited verbatim. Slot set reduced to context-end.

### Cell subset (26 of 39, context-end only)

- **16 P1 cells** (the transfer denominator): `constraint_knowledge`, `fact_assistant_animal`, `fact_novel_queried`, `fact_user_name`, `icl_task_mapping`, `instr_format`, `instr_language`, `list_numeric_detail`, `persona_prompted`, `prior_topic`, `query_content`, `reasoning_style`, `refusal_boundary`, `user_emotion`, `user_expertise`, `verbosity`
- **+6 causally-positive on #2329**: `conflict_format_fwd`, `conflict_format_rev`, `language_implied`, `load_instr_format_l3`, `load_instr_format_l5`, `recency_instr_format_d3`
- **+3 verdict-unstable across the two runs**: `demo_persona`, `load_fact_user_name_l5`, `recency_prior_topic_d5`
- Controls `query_content` (rig sensitivity) and `filler_swap` (disruption) are both already inside P1.

**Dropped (13):** eleven recency/load variants that were stable nulls in both runs, plus `demo_format` and `persona_role_header` — chronically unmeasurable regardless of model (5/36 and 2/36 surviving pairs on the 9B; median separations 0.10 and 0.04). Their weakness is bank design, not capability. The report must state the 26-of-39 restriction and name the dropped set.

### Realized model constants (verified from the Hub, 2026-08-19)

| Constant | #2329 value | This run | Source |
|---|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.8-27B` | — |
| `N_MODEL_LAYERS_FULL` | 32 | **64** | `config.json` `text_config.num_hidden_layers` |
| `HIDDEN_FULL` | 4096 | **5120** | `text_config.hidden_size` |
| vocab | 248,320 | **248,320 (identical)** | `text_config.vocab_size` |
| attention / KV heads | — | 24 / 4 | `text_config` |
| `max_position_embeddings` | — | 262,144 | `text_config` |
| full-attention layers | 8 (every 4th of 32) | **16 (every 4th of 64: 3,7,...,63)** | `full_attention_interval: 4`, `layer_types` |
| `F_ACT_READ_LAYER` | 30 | **59** (proposed) | open question 2 |
| slots | `ce` + `pe` | **`ce` only** | this task |
| model class | `AutoModelForMultimodalLM` | same (`Qwen3_5ForConditionalGeneration`) | `config.json` `architectures` |

Params 27.78B; bf16 weights ~55.6 GB — fits one H100 80GB with headroom, comfortable on a 143 GB H200.

### Port assessment

**Ports for free:**
- **Tokenizer identical to Qwen3.5-9B** (model vocab 248,320; BPE 248,044; `Qwen2TokenizerFast`). #2329's frozen bank was already re-tokenized and verified 1,404/1,404 pairs intact under this exact tokenizer, so P0 collapses from a bank re-freeze to a verification pass.
- **Thinking seam identical.** `enable_thinking=False` yields `<|im_start|>assistant\n<think>\n\n</think>\n\n` — same shape as #2329's realized prompt. Verified by rendering.
- Same `qwen3_5` family, so #2329's text-only load path (dropping the vision tower and MTP head) applies unchanged.
- **The entire `no_prefix` problem is moot** now that prefix-end is not run.

**Needs work:**
- 64 layers and hidden 5120 vs 32/4096: the state bank stores context-end only (halving it again), but the all-layer patch installs 64 hooks instead of 32.
- Stage-2 layer set must be re-mapped onto the 64-layer stack, with members shifted onto full-attention layers as #2329 did.
- Pod-side `transformers` pin must be confirmed to support `qwen3_5` at this config (#2329 pinned 5.15.0 pod-side; the VM stays at the repo-locked 4.57.6, where the tokenizer loads fine — verified).

## Compute + cost estimate

- Grid: 26 cells x 36 pairs x **1 slot** x 3 arms x 5 draws = **14,040 rollouts** (half the two-slot version; one third of #2329's 42,120).
- Anchors: ~9,360. These are per-CONTEXT and slot-independent, so they do NOT halve with the slot drop.
- Stage-2 scales with survivors.
- Wall: #2329 ran 56,160 rollouts in ~17 h on 8x H100 with a 9B. This run is ~23,400 rollouts (42%) at roughly 3x the per-rollout cost of a 27.78B model, so expect **~21 h on 8 GPUs**, plus queue.
- Judging: ~42% of #2329's ~212k calls, so **~89k calls, roughly $250-500**.
- Backend: proposed `fellows` (charmander, 8x H200 143 GB, free) — 27.78B bf16 fits one card so all 8 replicas per node survive. Contended: 14 of 112 GPUs idle against 72 pending jobs on 2026-08-19. Paid fallback 8x H100 RunPod at ~$26/hr (~$550 at this size); 8x H200 and 8x B200 packs were not purchasable when checked.
- **Marginal cost on the free lane: judge spend only.**

## Reporting requirement

The report must state that prefix-end was not run, and why (0 positives in both prior runs), so that a reader does not read its absence as a null result. It must also record that the transfer correlation runs on 16 units rather than #2329's 31, with the widened CI shown rather than compared naively against [0.583, 0.864].

## Pre-flight

- Confirm the HF data repo overflow reroute engages before committing ~21 h of generation — `issue2329_run.py` records the canonical data repo at HF's 1,000,000-file cap (#2304 shipped the reroute and merged 2026-08-16, but #2329 still wrote to the main repo).
- Pin the model revision explicitly (`revision=`). #2329 left this unpinned and its pod is gone, so the weights it used are not provable from its artifacts. Do not repeat that.
- Land #2329's PR #2004 first if the transfer read is to cite its committed tables — its `eval_results/` are still branch-only.

## Open questions for the user

1. **Cell subset** — default 26 cells. Full 39 costs 21,060 grid rollouts (+50%) and buys back thirteen cells that were stable nulls or structurally unmeasurable.
2. **`F_ACT_READ_LAYER`** — default 59 (fraction-match of the original 26/28 = 0.9286 onto 64 layers). Matching #2329's 30/32 = 0.9375 gives 60 instead. All-layer profiles are persisted either way, so this sets only the headline read.
3. **Backend** — default `fellows` (free, queued). RunPod 8x H100 costs ~$550 and skips the queue.

RESOLVED (2026-08-19): prefix-end dropped on user scope call; the `persona_prompted_explicitplain` variant proposed earlier is WITHDRAWN — it existed only to repair prefix-end coverage, and it could never have entered the transfer correlation because no counterpart cell exists in #2162 or #2329.
