---
title: Does context-vector patching improve on a more capable model? (Qwen3.8-27B,
  context-end only, all 39 cells)
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
backend: auto
---
# Experiment: Does context-vector patching work better on a more powerful model? (Qwen3.8-27B, context-end only, full 39-cell design)

## Goal

On `Qwen/Qwen3.8-27B` (27.78B dense, 64 layers, hidden 5120, thinking disabled), measure whether single-position activation patching at the CONTEXT VECTOR becomes more effective as model capability rises. Primary read: the number and identity of information types that are causally usable at context-end (the stored-and-used quadrant), and their per-type fraction-of-swap, against the realized 5 on Qwen2.5-7B (#2162) and 8 on Qwen3.5-9B (#2329). Secondary read: the Spearman transfer correlation against #2162's per-type steered fraction-of-swap over the 16 shared context-end P1 cells. Prefix-end is NOT run.

## Provenance

Originating chat request (verbatim): "let's try qwen3.8 27b", then scope narrowed by "i actually just want to see if patching at context vector works better on a more powerful model. so then we don't need prefix". Throughput work opened by "how can we reduce duration?".

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

Single varied factor vs #2329: the model. Bank text, cell set, F definitions, arms, judge instrument, exclusions and statistics inherited verbatim. Slot set reduced to context-end. Throughput changes below are execution-only and do not touch any measured quantity.

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
| `max_new_tokens` | 2048 flat + regen pass | **per-cell, 2048 / 4096** — see Throughput | this task |
| `gen_batch` | 16 | **retuned at pilot** — see Throughput | this task |
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

## Throughput

The dominant cost is not model size — it is that HF `generate()` has **no continuous batching**, so every row in a batch decodes until the LONGEST row stops.

Measured in `scripts/issue2329_run.py`, `experiments/issue1415/steering.py` and #2329's committed artifacts:

- `gen_batch=16`; `GRID_DRAWS=5`; `ANCHOR_DRAWS=10`; `max_new_tokens=2048`.
- Draws are **serial** (`steering.py:453`): one full `generate()` per draw, re-running prefill every time.
- **5.4% of anchor rows hit the 2048 cap** (755 of 14,040). At that rate a 16-row batch has a ~59% chance of containing a capper, so most batches run all 2048 decode steps for all 16 rows.
- The per-cell spread is extreme, and it is what makes this fixable: **14 of 37 cells cap at exactly 0.00%**, while `filler_swap` caps at 24.72% and `language_implied` at 23.06%. (37, not 39: the two `conflict_*_rev` cells reuse their `_fwd` counterparts' contexts, and anchors are per-context.)
- #2329 then paid a **second pass** — 2,190 of 4,240 gate rows regenerated at 4096.

Three decisions follow. All three are execution-only: none changes a measured quantity, a temperature, a draw count, or the judge.

**1. Per-cell `max_new_tokens`, set up front from #2329's measured table.** The 20 cells that breached the 2% re-generation trigger start at 4096; the 17 that did not stay at 2048. This does not shrink the regen pass, it removes it — those rows were going to be re-run at 4096 anyway. Breaching set, by measured cap-hit rate:

`filler_swap` 24.72, `language_implied` 23.06, `persona_role_header` 15.83, `reasoning_style` 14.17, `instr_language` 12.50, `user_emotion` 11.67, `verbosity` 10.28, `user_expertise` 10.00, `demo_persona` 10.00, `recency_instr_format_d5` 9.17, `query_content` 9.17, `recency_persona_prompted_d5` 8.61, `demo_format` 7.78, `recency_prior_topic_d3` 7.50, `persona_prompted` 7.50, `recency_persona_prompted_d3` 6.39, `conflict_persona_fwd` 3.89, `recency_instr_format_d3` 3.06, `recency_fact_user_name_d3` 2.78, `instr_format` 2.50.

These rates are Qwen3.5-9B's. They are a **prior, not a guarantee** — the 27B may be more or less verbose. The pilot re-measures per-cell cap-hit and the caps are adjusted before the production dispatch. The standing >2% re-generation trigger is unchanged and still applies to the realized rates.

**2. Bucket the anchor batches by cell.** Anchors are currently chunked in context order (`_anchor_context_order`), so a batch mixes cells and contexts from the fourteen 0.00%-cap cells get held hostage by `filler_swap` rows. The grid already batches within a `(cell, slot, arm)` block and does not have this problem. Pure reordering. Note it does change which draws land on which RNG draw, so outputs are statistically but not bit-wise identical to a context-ordered run — declared, and irrelevant across a model change.

**3. Retune `gen_batch` at the pilot, but only after 2.** 16 was tuned for a 9B. An H200 leaves ~87 GB free after 55.6 GB of weights, so there is real headroom. But a larger batch makes the straggler problem *worse* — more rows held by one capper — so raising it before the bucketing lands would cost wall-clock rather than save it. Sequence matters; the pilot sizes it.

**Filed separately as scoped implementation work:**

- **#2392** — vLLM backend for the unhooked anchor phase (`hook=None` at `issue2329_run.py:2467`), behind a measured HF-parity gate. Anchors are ~40% of the generation work and the phase where the straggler tax is worst; vLLM brings continuous batching plus `SamplingParams(n=10)`, one prefill instead of ten. The parity gate is mandatory: anchors set the F denominator, so a sampling shift would move every F invisibly.
- **#2393** — opt-in shared-prefill multi-draw mode for `generate_batch`: one prefill, N sampled continuations. Saves 4/5 of grid prefill and 9/10 of anchor prefill. Semantically clean rather than approximate — the patch is a prefill-only edit, so one hooked prefill produces exactly the cache all N draws should condition on. Covers the hooked grid, which vLLM cannot serve.

Neither is a precondition for launching. If they land first, the wall drops further; if not, this run still gets decisions 1-3.

## Compute + cost estimate

- Grid: 39 cells x 36 pairs x **1 slot** x 3 arms x 5 draws = **21,060 rollouts** (half the two-slot version; #2329 ran 42,120).
- Anchors: **14,040** (1,404 contexts x 10 draws). Per-CONTEXT and slot-independent, so they do NOT halve with the slot drop.
- Total generation: **35,100 rollouts**. Stage-2 scales with survivors, on top.
- Wall: **~32 h on 8 GPUs** as an unoptimized baseline, scaled from #2329's 56,160 rollouts in ~17 h on 8x H100 with a 9B — 62.5% of the count at roughly 3x the per-rollout cost. Decisions 1-3 should bring that toward the mid-20s, chiefly by removing the second regen pass; #2392 and #2393 would take it further. **Every one of these is an extrapolation, not a measurement** — see Pre-flight.
- Judging: ~62.5% of #2329's ~212k calls, so **~133k calls, roughly $400-750**.
- Backend: **`auto`** (user decision) — RunPod first, then fellows, then the free DRAC/Mila lanes, then a terminal RunPod retry. The RunPod account is the shared Anthropic fellows/safety org pool, so provisioning there is ordinary sponsored use. 27.78B bf16 fits one card, so all 8 replicas per node survive on either RunPod 8x H100 or fellows 8x H200 143 GB. Fellows was contended when checked (14 of 112 GPUs idle against 72 pending jobs, 2026-08-19), which is what `auto` routes around.
- **Worth checking at dispatch:** replicas are independent over a claim-file queue, so widening past 8 GPUs is near-linear. Whether the claim file works across nodes on shared storage decides if that is available.

## Reporting requirement

The report must state that prefix-end was not run, and why (0 positives in both prior runs), so that a reader does not read its absence as a null result. It must also record that the transfer correlation runs on 16 units rather than #2329's 31, with the widened CI shown rather than compared naively against [0.583, 0.864]. `demo_format` and `persona_role_header` must be flagged as expected-untestable in advance of their verdicts, so a third untestable result reads as a bank-design finding rather than a capability null. The per-cell cap regime must be reported as a table, since it differs from #2329's flat cap.

## Pre-flight

- **Run a measured 1-cell pilot through the production entrypoint at production shape before committing the full wall.** The estimate above is extrapolated from #2329's 9B throughput, not measured on a 27.78B model; project rule requires a measured per-cell basis for any phase projected past ~15 min, and any self-set timeout or fence sized at >= 2x the pilot-extrapolated wall. The pilot must ALSO report realized per-cell cap-hit on the 27B (calibrating decision 1) and be used to size `gen_batch` (decision 3).
- Confirm the HF data repo overflow reroute engages before committing the generation — `issue2329_run.py` records the canonical data repo at HF's 1,000,000-file cap (#2304 shipped the reroute and merged 2026-08-16, but #2329 still wrote to the main repo).
- Pin the model revision explicitly (`revision=`). #2329 left this unpinned and its pod is gone, so the weights it used are not provable from its artifacts. Do not repeat that.
- Land #2329's PR #2004 first if the transfer read is to cite its committed tables — its `eval_results/` are still branch-only.

## Resolved decisions (2026-08-19)

1. **Prefix-end dropped** on user scope call ("i actually just want to see if patching at context vector works better on a more powerful model. so then we don't need prefix"). The `persona_prompted_explicitplain` variant proposed earlier is WITHDRAWN — it existed only to repair prefix-end coverage, and it could never have entered the transfer correlation because no counterpart cell exists in #2162 or #2329.
2. **Cell set: all 39**, not the 26-cell subset. Full design parity with both prior models; +7,020 grid rollouts over the subset.
3. **F_act read layer: not pre-committed.** Full 64-layer profile reported; layer 59 fixed in advance as the labelled cross-model comparison row.
4. **Backend: `auto`.**
5. **Throughput decisions 1-3 adopted** (per-cell caps, anchor cell-bucketing, pilot-tuned `gen_batch`); **#2392** (vLLM anchors) and **#2393** (shared prefill) filed as separate implementation tasks, neither blocking launch.
