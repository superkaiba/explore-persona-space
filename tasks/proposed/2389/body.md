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
# Experiment: Does context-vector patching work better on a more powerful model? (Qwen3.8-27B, context-end only, full 39-cell design)

## Goal

On `Qwen/Qwen3.8-27B` (27.78B dense, 64 layers, hidden 5120, thinking disabled), measure whether single-position activation patching at the CONTEXT VECTOR becomes more effective as model capability rises. Primary read: the number and identity of information types that are causally usable at context-end (the stored-and-used quadrant), and their per-type fraction-of-swap, against the realized 5 on Qwen2.5-7B (#2162) and 8 on Qwen3.5-9B (#2329). Secondary read: the Spearman transfer correlation against #2162's per-type steered fraction-of-swap over the 16 shared context-end P1 cells. Prefix-end is NOT run.

## Provenance

Originating chat request (verbatim): "let's try qwen3.8 27b", then scope narrowed by "i actually just want to see if patching at context vector works better on a more powerful model. so then we don't need prefix".

Lineage: #2162 (Qwen2.5-7B-Instruct, 39 cells x 2 slots) -> #2329 (Qwen3.5-9B, same design; transfer rho = 0.831) -> this task (Qwen3.8-27B, all 39 cells, context-end only).

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

Single varied factor vs #2329: the model. Bank text, cell set, F definitions, arms, judge instrument, exclusions and statistics inherited verbatim. Slot set reduced to context-end.

### Cell set: all 39 (user decision, 2026-08-19)

The full #2162/#2329 cell set runs, unrestricted. This buys exact design parity across the three models, so the report carries no "restricted subset" caveat and every cross-model verdict comparison is like-for-like.

Two cells are expected to come back `untestable-causal` again, and the report should say so up front rather than presenting their nulls as capability evidence:

- `demo_format` — 5 of 36 pairs survived the anchor-separation exclusion on the 9B, median separation 0.10.
- `persona_role_header` — 2 of 36 surviving pairs, median separation 0.04.

Their weakness is bank design, not model capability, so a stronger model is not predicted to rescue them. A third consecutive untestable verdict is itself the useful read: it converts "we could not measure this yet" into "this bank needs rebuilding", which is a concrete follow-up rather than an open question.

Eleven recency/load variants were stable nulls in both prior runs. Running them costs ~7,000 grid rollouts and is what full parity buys: a null that holds across three models at rising capability is a stronger claim than a null at two.

Controls: `query_content` (rig sensitivity) and `filler_swap` (disruption) both retained.

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
| `F_ACT_READ_LAYER` | 30 (fixed) | **not pre-committed** — see below | user decision |
| slots | `ce` + `pe` | **`ce` only** | this task |
| model class | `AutoModelForMultimodalLM` | same (`Qwen3_5ForConditionalGeneration`) | `config.json` `architectures` |

Params 27.78B; bf16 weights ~55.6 GB — fits one H100 80GB with headroom, comfortable on a 143 GB H200.

### F_act read layer: resolved at analysis time, by rule not by outcome

There is no single pre-committed headline layer. The **full 64-layer F_act profile is the reported object**, and the report shows it in full.

For the cross-model comparison table (7B / 9B / 27B side by side) the labelled row is the lineage fraction-match, **layer 59** (#2094/#2162's 26/28 = 0.9286, scaled to 64). That layer is fixed here, before the run, so the cross-model number cannot be selected on the observed values. Any other layer highlighted in the report is labelled exploratory.

Why not pre-commit harder: #2094's plan v4 records two reasons for reading at 26, and only one survives the model change.

- **"Downstream of every mid-stack edit"** — still binds. Patches land mid-stack (the joint band is layers 14-20 on a 28-layer stack), so the read must sit above them to observe the consequence of the edit rather than the edit itself.
- **"Deepest banked-map layer"** — does NOT transfer. The #779/#1738 ridge maps were fitted at {14, 19, 26} and are 3584-dimensional, inapplicable at 5120. #2329 already dropped the mapshift banked-parity anchors for exactly this reason at 4096.

Since the surviving constraint does not uniquely pick a layer, forcing a single pre-registered value would be a false precision. Reporting the profile with one rule-fixed comparison row is the honest form.

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

- Grid: 39 cells x 36 pairs x **1 slot** x 3 arms x 5 draws = **21,060 rollouts** (half the two-slot version; #2329 ran 42,120).
- Anchors: **14,040**. These are per-CONTEXT and slot-independent, so they do NOT halve with the slot drop — the full #2329 anchor count carries over unchanged.
- Total generation: **35,100 rollouts**. Stage-2 scales with survivors, on top.
- Wall: **~32 h on 8 GPUs**, plus queue. Basis: #2329 ran 56,160 rollouts in ~17 h on 8x H100 with a 9B; this is 62.5% of that count at roughly 3x the per-rollout cost of a 27.78B model. **This is an extrapolation, not a measurement** — see Pre-flight.
- Judging: ~62.5% of #2329's ~212k calls, so **~133k calls, roughly $400-750**.
- Backend: **`auto`** (user decision) — RunPod first, then fellows, then the free DRAC/Mila lanes, then a terminal RunPod retry. The RunPod account is the shared Anthropic fellows/safety org pool, so provisioning there is ordinary sponsored use. 27.78B bf16 fits one card, so all 8 replicas per node survive on either RunPod 8x H100 or fellows 8x H200 143 GB. Fellows was contended when checked (14 of 112 GPUs idle against 72 pending jobs, 2026-08-19), which is what `auto` routes around.

## Reporting requirement

The report must state that prefix-end was not run, and why (0 positives in both prior runs), so that a reader does not read its absence as a null result. It must also record that the transfer correlation runs on 16 units rather than #2329's 31, with the widened CI shown rather than compared naively against [0.583, 0.864]. `demo_format` and `persona_role_header` must be flagged as expected-untestable in advance of their verdicts, so a third untestable result reads as a bank-design finding rather than a capability null.

## Pre-flight

- **Run a measured 1-cell pilot through the production entrypoint at production shape before committing the full ~32 h.** The wall estimate above is extrapolated from #2329's 9B throughput, not measured on a 27.78B model; project rule requires a measured per-cell basis for any phase projected past ~15 min, and any self-set timeout or fence must be sized at >= 2x the pilot-extrapolated wall.
- Confirm the HF data repo overflow reroute engages before committing the generation — `issue2329_run.py` records the canonical data repo at HF's 1,000,000-file cap (#2304 shipped the reroute and merged 2026-08-16, but #2329 still wrote to the main repo).
- Pin the model revision explicitly (`revision=`). #2329 left this unpinned and its pod is gone, so the weights it used are not provable from its artifacts. Do not repeat that.
- Land #2329's PR #2004 first if the transfer read is to cite its committed tables — its `eval_results/` are still branch-only.

## Resolved decisions (2026-08-19)

1. **Prefix-end dropped** on user scope call ("i actually just want to see if patching at context vector works better on a more powerful model. so then we don't need prefix"). The `persona_prompted_explicitplain` variant proposed earlier is WITHDRAWN — it existed only to repair prefix-end coverage, and it could never have entered the transfer correlation because no counterpart cell exists in #2162 or #2329.
2. **Cell set: all 39**, not the 26-cell subset. Full design parity with both prior models; +7,020 grid rollouts over the subset.
3. **F_act read layer: not pre-committed.** Full 64-layer profile reported; layer 59 fixed in advance as the labelled cross-model comparison row.
4. **Backend: `auto`.**
