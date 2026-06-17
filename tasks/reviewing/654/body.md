---
title: 'Query influence on the context-conditioned residual stream: per-layer context-end
  vs query-end last-token similarity (Qwen-2.5-7B-Instruct)'
kind: experiment
tags: []
created_at: '2026-06-16T22:20:03Z'
has_clean_result: false
origin_prompt: check how similar activations are after the context vs after the user
  query (for a variety of contexts and user queries), ask clarifying questions
goal: Measure, per layer in Qwen-2.5-7B-Instruct, how much appending a user query
  displaces the last-token residual-stream state away from the context-established
  state (centered cosine + CKA + shuffled-pair floor) across persona / generic-instruction
  / ICL / wild-chat contexts and varied queries, to characterize context-dominant
  vs query-dominant layer regimes.
---
# Matched delimiter-position residual cosine sits above the within-tier shuffled floor at every layer of Qwen-2.5-7B-Instruct, but query length dominates the persona tier (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- The query-end residual stays closer to its **own** context than a deranged one at every layer (single seed) — the no-persistence null is rejected for 3 tiers, **marginal** for persona.
- **Query length, not context type, drives the persona layer-0 signal:** short queries score 0.81, long 0.05 (0.76 gap); other tiers move under 0.10 by length.
- The prediction that persona contexts would persist *most* is **rejected in the opposite direction** — persona is the LOWEST-persisting tier in **21 of 28 layers**.
- The same-slot companion read (different-token confound removed; length/position confounds remain) holds the generation-slot state at **0.63-0.72 cosine** in late layers — consistent with context persistence, not proof of it.
- Read position is the last-prompt token, the known-weak persona-content proxy (Persona Vectors 2507.21509); residual geometry, not a behavioral claim.

## What I ran

- **Why:** A recurring question in this project's persona line is *where* in the stack the model stops tracking its context and starts answering the query. Prior work read one position per prompt for family clustering ([#594](https://eps.superkaiba.com/tasks/594)) or role-vs-persona separation ([#634](https://eps.superkaiba.com/tasks/634)); none contrasted the context-end and query-end positions within one prompt. This run measures that displacement per layer.
- **Design:** One manipulated axis — **read position** (context-span-end vs query-span-end token) — crossed with context type (4 tiers) × query type (on/off-topic × short/long). 81 contexts × 10 queries = 810 pairs, one forward pass each, residuals at all 28 layers. Single seed (42).
- **Training:** none — forward-pass measurement on base `Qwen/Qwen2.5-7B-Instruct` (bf16, `output_hidden_states=True`, ChatML).
- **Eval:** per-layer per-pair **centered cosine** (global-mean-centered, anisotropy-robust DV) between the two positions; a **shuffled-pair derangement floor** (B=1000) computed both globally and **within each tier**; **whole-bank linear CKA** as a secondary geometry read; and a **companion same-position contrast** (context-only vs context+query at the fixed generation slot) that removes the different-token confound. No judge, no generation.

## Findings

### Matched cosine sits above the within-tier floor at every layer — context persistence is not a null, but marginal for persona

The construct-valid baseline is the **within-tier** shuffled floor (derange query↔context within a tier). The global cross-tier floor sits near zero only because the four tiers differ in mean direction; the within-tier floors are NOT near zero, and against them the multiples are modest.

![Matched centered cosine per tier vs layer; each tier's within-tier shuffled floor shaded, none near zero, matched curves above their own band at every depth](https://raw.githubusercontent.com/superkaiba/explore-persona-space/994feb766d4cce377032697b269ca823029d631a/figures/issue_654/hero_displacement_blog.png)

> **Figure.** *Matched query-end cosine stays above its own-tier shuffled floor at every layer.* Solid = matched-pair centered cosine; shaded = that tier's within-tier shuffled floor (2.5/97.5 pctile, B=1000) — NOT near zero (persona 0.43, generic/ICL 0.25 at layer 0). Qwen-2.5-7B-Instruct, 810 pairs.

- Generic / ICL / real-chat clear their floor band by 10-16× at L0 (real-chat highest) and 4.5-10.5× through mid-stack (L10-18) — the no-persistence null is decisively rejected. `×` = half-band widths above the floor, not a value ratio.
- **Persona is marginal:** matched 0.50 vs floor 0.43 at L0 (3.2×), 2.0× the band at L7. Persona prompts are mutually similar, so the floor is high.
- Raw (uncentered) cosine is anisotropy-inflated and identifies the matched direction less reliably; it is shown only in Finding 5 (raw-vs-centered crater), and the centered DV carries every finding here.

### Query length, not context type, dominates the persona layer-0 signal

The tier-averaged view hides a larger effect inside the persona tier: a large 0.76 short-vs-long gap by query length.

![Grouped bars: layer-0 matched cosine by query length per tier; persona short 0.81 vs long 0.05, other tiers nearly flat](https://raw.githubusercontent.com/superkaiba/explore-persona-space/994feb766d4cce377032697b269ca823029d631a/figures/issue_654/query_length_split_blog.png)

> **Figure.** *Query length drives the persona layer-0 signal; persona is the length-sensitive outlier.* Layer-0 matched centered cosine, short vs long query, per tier. Persona: short 0.81, long 0.05 (0.76 gap); outside persona, the other tiers move under 0.10 by length. Qwen-2.5-7B-Instruct, 810 pairs.

- Persona short-vs-long split stays 0.30-0.76 through layers 0-11; the off-topic-long persona cell goes **negative** at L7-L8 (−0.072, −0.085).
- Short queries place the query-end `\n` few tokens after the context-end `\n`, so the residuals are mechanically close. This is why the same-slot companion read is the load-bearing persistence evidence, not these two-position numbers.
- Centered cosine, not raw, again: raw cosine is anisotropy-inflated and would muddy the short-vs-long contrast; the raw-vs-centered comparison lives in Finding 5.

### Persona is the lowest-persisting tier — opposite to the predicted persona-persists-most direction

The plan predicted persona would persist *more* than the other tiers. The within-tier gap (matched − own floor) shows the reverse.

![Matched-minus-shuffled gap per layer per tier; persona lowest at almost every depth, decaying from layer 0](https://raw.githubusercontent.com/superkaiba/explore-persona-space/994feb766d4cce377032697b269ca823029d631a/figures/issue_654/gap_per_tier_blog.png)

> **Figure.** *Persona is the lowest-persisting tier at almost every depth — opposite to the predicted persona-persists-most direction.* Gap = matched − own-tier shuffled centered cosine per layer. Qwen-2.5-7B-Instruct, 810 pairs.

- Persona is the lowest-gap tier in **21 of 28 layers**. At L7: generic 0.129, real-chat 0.110, ICL 0.049, persona 0.018.
- This corroborates the plan's flagged caveat (Persona Vectors 2507.21509): the last-prompt-token read is the *weakest* for persona content, so a low persona signal is expected, not a null.
- The gap is computed on the centered DV (matched − own-tier floor); raw cosine would inflate every tier by the shared anisotropy and is shown only in Finding 5.

### Same-slot read holds the generation-slot state at 0.63-0.72 cosine in late layers

The two-position read compares two different `\n` tokens, so part of its similarity is token identity. The companion read fixes the slot — same generation position, context-only vs context+query — removing the different-token confound. Length / position / extra-turn confounds remain (no length-matched dummy-query control).

![Same-slot cosine per layer per tier; curves start near 0.99, bottom around 0.63-0.72 in late layers, cluster within ~0.07](https://raw.githubusercontent.com/superkaiba/explore-persona-space/994feb766d4cce377032697b269ca823029d631a/figures/issue_654/companion_blog.png)

> **Figure.** *Adding the query holds the generation-slot readout at 0.63-0.72 cosine in late layers.* Context-only vs context+query at the fixed generation slot, per tier. Removes the different-token confound; length/position confounds remain. Qwen-2.5-7B-Instruct, 810 pairs.

- Mean-of-tiers minimum **0.665** (L23); per-tier minima reach **0.629** (generic, L23), wildchat ends 0.640. Tiers cluster within ~0.07 at every depth.
- This is the confound-robust evidence consistent with context persistence. Cosine is not a linear displacement fraction, so I report the value, not a "fraction moved."
- This companion read is also centered; the raw-vs-centered contrast is reserved for Finding 5.

### Per-pair alignment decays with depth while whole-bank CKA dips then rises

Centered cosine is a per-pair read; linear CKA asks whether the context bank's geometry survives into the query bank. The two diverge.

![Per-pair centered cosine (solid, decaying) vs whole-bank CKA (dashed, dipping mid-stack then rising late) vs shuffled-bank CKA floor near 0.01](https://raw.githubusercontent.com/superkaiba/explore-persona-space/994feb766d4cce377032697b269ca823029d631a/figures/issue_654/cosine_vs_cka_blog.png)

> **Figure.** *Per-pair alignment decays; whole-bank CKA dips mid-stack then rises late.* Per-pair centered cosine (solid) vs context-bank↔query-bank linear CKA (dashed) vs shuffled-bank CKA floor (dotted, ~0.01). Qwen-2.5-7B-Instruct, 810 pairs.

- CKA is **0.53 (L0), 0.63 (L5)**, dips to a **trough of 0.29 (L7)**, then rises to **0.76 (L27)** — a mid-stack dip plus late rise, not simple late convergence.
- Descriptive geometry, not a mechanism: the late rise is consistent with a shared last-token / format subspace. Footnote: raw (uncentered) cosine craters at the final layer (0.64 at L26 → 0.38 at L27), likely norm reallocation; the centered DV does not.

## Data

### Trained on

n/a — no training in this task. Forward-pass-only measurement on the base model.

### Evaluated with

The battery is **81 contexts × 10 queries = 810 pairs** over four context tiers, each context paired against the same fixed 10-query bank so query type is a clean within-bank factor. Identity: persona system prompts (11 = 10 curated personas + the default assistant), generic instructions (20 UltraChat first-user prompts), in-context example blocks (20 held-out 4-exchange UltraChat Q/A blocks), real chat turns (30 WildChat conversation prefixes). Why this set: it spans curated-template (persona) through tier-1 real chat (WildChat) so a context-type effect would show as a tier-specific curve. Preprocessing: ChatML via `apply_chat_template`; the context-span-end and query-span-end token offsets are derived from the template and hard-asserted (offset-assert failure fraction = 0.000 over all 810 pairs). Both read positions decode to `\n` in all 810 pairs — expected for ChatML, and why centered cosine + the same-slot companion read are the load-bearing DVs rather than raw cosine.

Cherry-picked for illustration (one context per tier; the full battery is the link below):

```
persona      <|im_start|>system\nYou are a software engineer who builds web applications.<|im_end|>\n...
persona      <|im_start|>system\nYou are a villainous mastermind who schemes to take over the world.<|im_end|>\n...
generic      <|im_start|>system\nThese instructions apply to section-based themes ... What theme version am I using? ...<|im_end|>\n...
in-context   <|im_start|>system\nYou are Qwen ... <|im_end|>\n<|im_start|>user\nHow does the US Secretary of State play a role ...<|im_end|>\n<|im_start|>assistant\nThe US Secretary of State is the chief diplomat ...
real chat    <|im_start|>system\nYou are Qwen ...<|im_end|>\n<|im_start|>user\nHey there! Are you familiar with reality shifting? ...
```

5 of the 10-query bank, verbatim:

<details>
<summary>The 10-query bank (cherry-picked: first 5 of 10; full bank in battery.json below)</summary>

```
q_ontopic_short_0   "What is the best way to learn a new language?"            (on-topic, short)
q_ontopic_short_1   "Can you explain how photosynthesis works?"                (on-topic, short)
q_ontopic_long_1    "Can you explain how photosynthesis works in detail, ...    (on-topic, long)
q_offtopic_short_0  "Who won the 1994 FIFA World Cup?"                          (off-topic, short)
q_offtopic_long_1   "I want to bake a three-layer chocolate birthday cake ...  (off-topic, long)
```

</details>

Complete probe battery (all 810 pairs + offsets + decoded spans): [`battery.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/inputs/battery.json) and [`extraction_manifest.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/analysis_tensors/extraction_manifest.json).

### Generated

n/a — no model completions are generated. Each forward pass yields residual-stream vectors (28 layers × 3584 dims at the context-end, query-end, and generation-slot positions), not text. The model emits nothing to judge; every data point is a per-layer cosine or CKA scalar from these vectors. The full per-pair residual banks (810 pair files + 81 context-only companion files, fp32) are at [`analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/analysis_tensors).

## Reproducibility

**Methodology:** the full findings-blind methodology + hyperparameter reference is auto-generated separately and linked at promotion.

**Parameters:**

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` (base, no fine-tuning), bf16, `output_hidden_states=True` |
| num_hidden_layers / hidden_size | 28 / 3584 (asserted at runtime) |
| Read positions | context-span-end token, query-span-end token, assistant-generation slot (companion) |
| Centering | `global_mean` (per-bank, per-layer); within-tier + global shuffled floors both computed |
| Floor | shuffled-pair derangement, B=1000, seed 42 |
| CKA | linear CKA (Kornblith 2019, HSIC-Frobenius form), float64 |
| Pairs / forwards | 810 (context, query) pairs; 891 forwards incl. 81 context-only companion reads |
| Seed | 42 (single seed) |

**Artifacts:**
- Per-layer displacement JSON + per-tier cells: `eval_results/issue_654/per_layer_displacement.json` (+ `cells/`), committed at SHA `92ddfce6bbc78df8675b1747462b1ec1c74b7d6f`.
- Dual-position residual banks + manifest: [`analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/analysis_tensors) (HF data repo). `list_repo_files` confirms **892 files**: 810 pair `.pt` + 81 context-only `.pt` + 1 `extraction_manifest.json`.
- Probe battery: [`inputs/battery.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/inputs/battery.json).
- Figures (blog style, PNG + PDF + meta): `figures/issue_654/{hero_displacement,query_length_split,gap_per_tier,companion,cosine_vs_cka}_blog.*` at SHA `994feb766d4cce377032697b269ca823029d631a`.

**Compute:** GPU extraction (891 forwards, batch 1, 28 layers, 2-3 positions/forward) on 1× A100-80 (GCP `a2-ultragpu-1g`, `lora-7b` intent), ~0.4 GPU-h. CPU metric phase (centered cosine + B=1000 derangement floors over 5 strata + float64 linear CKA over 810×3584 banks + figures) ~30 min off-pod. No judge / API cost.

**Code:**
- `scripts/issue654_build_battery.py` (battery), `scripts/issue654_extract.py` (dual-position extraction), `scripts/issue654_analyze.py` (metrics), `scripts/issue654_hero_figs.py` (blog figures), `scripts/issue654_dispatch.sh` (GCP dispatch).
- New library: `src/explore_persona_space/analysis/probes.py::extract_dual_position_activations`; `representation_shift.py::linear_cka` + `cka_per_layer` (+ `tests/test_linear_cka.py`).
- Reproduce the analysis from the uploaded banks:
  ```bash
  uv run python scripts/issue654_analyze.py \
    --banks data/issue654/hf_snapshot/issue654_query_displacement/analysis_tensors \
    --out eval_results/issue_654 --figures
  uv run python scripts/issue654_hero_figs.py
  ```
- Git commit: `994feb766d4cce377032697b269ca823029d631a` (branch `issue-654`).

**Context:**
- **Created / run:** task created 2026-06-16; results landed 2026-06-17 (UTC).
- **Follow-up to:** fresh direction (no parent). Sibling-positioned against [#594](https://eps.superkaiba.com/tasks/594) (context-vector geometry, one read position) and [#634](https://eps.superkaiba.com/tasks/634) (role-vs-persona bank); first to contrast two within-prompt positions.
- **Originating prompt(s), verbatim:**
  > check how similar activations are after the context vs after the user query (for a variety of contexts and user queries), ask clarifying questions
