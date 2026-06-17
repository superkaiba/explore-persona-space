---
title: 'Query influence on the context-conditioned residual stream: per-layer context-end
  vs query-end last-token similarity (Qwen-2.5-7B-Instruct)'
kind: experiment
backend: runpod
tags:
- followup-auto
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
# Matched delimiter-position residual cosine sits above the within-tier shuffled floor at every layer of Qwen-2.5-7B-Instruct, and a length-matched dummy control shows the late-layer generation-slot drop tracks query content, not length (MODERATE confidence)

<!-- clean-result-v3 -->

**Methodology:** [docs/methodology/issue_654.md](https://github.com/superkaiba/explore-persona-space/blob/cd8358444413288f7861be55d58380a72529c4e7/docs/methodology/issue_654.md) · [gist](https://gist.github.com/superkaiba/d2533088ba71df5685cd76794178d21b)

## Takeaways

- The query-end residual stays closer to its **own** context than a deranged one at every layer (single seed) — the no-persistence null is rejected for 3 tiers, **marginal** for persona.
- **Query length, not context type, drives the persona layer-0 signal:** short queries score 0.81, long 0.05 (0.76 gap); other tiers move under 0.10 by length.
- The prediction that persona contexts would persist *most* is **rejected in the opposite direction** — persona is the LOWEST-persisting tier in **21 of 28 layers**.
- A **length-matched filler** holds the L23 slot at **0.81-0.85** where a real query drops it to **0.63-0.70** (real − filler **−0.165** over L23-L27): consistent with content over length, filler-distribution alternatives unaddressed (Finding 4).
- **The content effect is mostly short queries:** the L23 short-query gap is **−0.255** vs long **−0.068**. The reported interval assumes per-pair independence, so it is optimistic.
- Read position is the last-prompt token, the known-weak persona-content proxy (Persona Vectors 2507.21509); residual geometry, not a behavioral claim.

## What I ran

- **Why:** A recurring question in this project's persona line is *where* in the stack the model stops tracking its context and starts answering the query. Prior work read one position per prompt for family clustering ([#594](https://eps.superkaiba.com/tasks/594)) or role-vs-persona separation ([#634](https://eps.superkaiba.com/tasks/634)); none contrasted the context-end and query-end positions within one prompt. This run measures that displacement per layer.
- **Design:** One manipulated axis — **read position** (context-span-end vs query-span-end token) — crossed with context type (4 tiers) × query type (on/off-topic × short/long). 81 contexts × 10 queries = 810 pairs, one forward pass each, residuals at all 28 layers. Single seed (42).
- **Training:** none — forward-pass measurement on base `Qwen/Qwen2.5-7B-Instruct` (bf16, `output_hidden_states=True`, ChatML).
- **Eval:** per-layer per-pair **centered cosine** (global-mean-centered, anisotropy-robust DV) between the two positions; a **shuffled-pair derangement floor** (B=1000) computed both globally and **within each tier**; **whole-bank linear CKA** as a secondary geometry read; a **companion same-position contrast** (context-only vs context+query at the fixed generation slot) that removes the different-token confound; and a **length-matched dummy-query control** (round 2) that re-runs the companion read with a content-meaningless query padded to each real query's token length, separating the content contribution to the late-layer drop from length and slot position. No judge, no generation.
- **Rounds:**

  | Round | Date (UTC) | What changed | Result |
  |---|---|---|---|
  | 1 | 2026-06-17 | Two-position read + within-tier floor + same-slot companion read | Persistence null rejected (marginal for persona); length dominates persona; companion holds slot at 0.63-0.72 with length/position confound unresolved |
  | 2 | 2026-06-17 | Added length-matched content-meaningless dummy-query arm to the companion read | Dummy holds slot at 0.81-0.85 (L23); real − dummy gap −0.165 at L23-27 — the real query displaces the slot more than a same-length filler, concentrated in short queries |

## Findings

### Matched cosine sits above the within-tier floor at every layer — context persistence is not a null, but marginal for persona

The construct-valid baseline is the **within-tier** shuffled floor (derange query↔context within a tier). The global cross-tier floor sits near zero only because the tiers differ in mean direction; the within-tier floors are NOT near zero, and against them the multiples are modest.

![Matched centered cosine per tier vs layer; each tier's within-tier shuffled floor shaded, none near zero, matched curves above their own band at every depth](https://raw.githubusercontent.com/superkaiba/explore-persona-space/994feb766d4cce377032697b269ca823029d631a/figures/issue_654/hero_displacement_blog.png)

> **Figure.** *Matched query-end cosine stays above its own-tier shuffled floor at every layer.* Solid = matched-pair centered cosine; shaded = that tier's within-tier shuffled floor (2.5/97.5 pctile, B=1000) — NOT near zero (persona 0.43, generic/ICL 0.25 at layer 0). Qwen-2.5-7B-Instruct, 810 pairs.

- Generic / ICL / real-chat clear their floor band by 10-16× at L0 and 4.5-10.5× through mid-stack (L10-18) — the no-persistence null is decisively rejected. `×` = half-band widths above the floor, not a value ratio.
- **Persona is marginal:** matched 0.50 vs floor 0.43 at L0 (3.2×), 2.0× the band at L7. Persona prompts are mutually similar, so the floor is high.
- Raw (uncentered) cosine is anisotropy-inflated and less reliable at identifying the matched direction, so it is not plotted; the centered DV carries every finding (raw's final-layer crater is noted in Finding 5).

### Query length, not context type, dominates the persona layer-0 signal

The tier-averaged view hides a larger effect inside the persona tier: a large 0.76 short-vs-long gap by query length.

![Grouped bars: layer-0 matched cosine by query length per tier; persona short 0.81 vs long 0.05, other tiers nearly flat](https://raw.githubusercontent.com/superkaiba/explore-persona-space/994feb766d4cce377032697b269ca823029d631a/figures/issue_654/query_length_split_blog.png)

> **Figure.** *Query length drives the persona layer-0 signal; persona is the length-sensitive outlier.* Layer-0 matched centered cosine, short vs long query, per tier. Persona: short 0.81, long 0.05 (0.76 gap); outside persona, the other tiers move under 0.10 by length. Qwen-2.5-7B-Instruct, 810 pairs.

- Persona short-vs-long split stays 0.30-0.76 through layers 0-11; the off-topic-long persona cell goes **negative** at L7-L8 (−0.072, −0.085).
- Short queries place the query-end `\n` few tokens after the context-end `\n`, so the residuals are mechanically close. This is why the same-slot companion read is the load-bearing persistence evidence, not these two-position numbers.
- Centered cosine, not raw, again: raw cosine is anisotropy-inflated and would muddy the short-vs-long contrast, so it is omitted here.

### Persona is the lowest-persisting tier — opposite to the predicted persona-persists-most direction

The plan predicted persona would persist *more* than the other tiers. The within-tier gap (matched − own floor) shows the reverse.

![Matched-minus-shuffled gap per layer per tier; persona lowest at almost every depth, decaying from layer 0](https://raw.githubusercontent.com/superkaiba/explore-persona-space/994feb766d4cce377032697b269ca823029d631a/figures/issue_654/gap_per_tier_blog.png)

> **Figure.** *Persona is the lowest-persisting tier at almost every depth — opposite to the predicted persona-persists-most direction.* Gap = matched − own-tier shuffled centered cosine per layer. Qwen-2.5-7B-Instruct, 810 pairs.

- Persona is the lowest-gap tier in **21 of 28 layers**. At L7: generic 0.129, real-chat 0.110, ICL 0.049, persona 0.018.
- This corroborates the plan's flagged caveat (Persona Vectors 2507.21509): the last-prompt-token read is the *weakest* for persona content, so a low persona signal is expected, not a null.
- The gap is computed on the centered DV (matched − own-tier floor); raw cosine would inflate every tier by the shared anisotropy, so it is not used for the gap.

### The late-layer drop tracks query content, not length

Round 1's companion read could not rule out a length/position artifact. Round 2 adds a length-matched filler: at L23 it holds the slot at **0.81-0.85** where a real query drops it to **0.63-0.70**.

![Real vs length-matched-filler companion cosine and per-tier real-minus-filler gap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/86b6c65b5a4482efdaf69341c4295e0271f430bc/figures/issue_654/query_content_vs_length_gap_blog.png)

> **Figure.** *A length-matched no-content filler holds the generation slot higher than a real query.* Left: same-slot companion cosine, real (solid) vs length-matched filler (dashed) per tier; at L23 filler = persona 0.811 / real chat 0.836 / generic 0.852 / in-context 0.853. Right: real − filler per-pair gap (shaded = per-pair gap SE, n = 110-300/tier). Qwen-2.5-7B-Instruct, 810 matched pairs.

- **Not persona-led:** the L23 gap is deepest for generic (−0.223) and **shallowest for persona (−0.127)**, real-chat (−0.191) and in-context (−0.152) between. Pooled L23 gap is −0.180; the −0.165 above is the L23-L27 band mean.
- **Mostly short queries:** the L23 short-query gap is **−0.255** vs long-query **−0.068** (>3×, hidden by the pool); the L18 long gap is ~zero (+0.002).
- The OVERALL gap CI excludes zero at all 28 layers; per-tier CIs cross zero at two early near-zero layers (generic L3, in-context L2). The SE assumes per-pair independence, so the reported interval is optimistic.
- This does NOT show the slot moving *toward* the content: content queries sit in a denser embedding region and carry rarer tokens than the filler. Read it as **consistent with content over length, not a content-direction** — geometry, not behavior.

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

The base battery is **81 contexts × 10 queries = 810 pairs** over four context tiers, each context paired against the same fixed 10-query bank so query type is a clean within-bank factor. Identity: persona system prompts (11 = 10 curated personas + the default assistant), generic instructions (20 UltraChat first-user prompts), in-context example blocks (20 held-out 4-exchange UltraChat Q/A blocks), real chat turns (30 WildChat conversation prefixes). Why this set: it spans curated-template (persona) through tier-1 real chat (WildChat) so a context-type effect would show as a tier-specific curve. Preprocessing: ChatML via `apply_chat_template`; the context-span-end and query-span-end token offsets are derived from the template and hard-asserted (offset-assert failure fraction = 0.000 over all 810 pairs). Both read positions decode to `\n` in all 810 pairs — expected for ChatML, and why centered cosine + the same-slot companion read are the load-bearing DVs rather than raw cosine.

The **round-2 dummy battery** mirrors the same 81 contexts but replaces each real query with a content-meaningless one — `"Please continue with whatever you think is most appropriate here."` padded with a `" really"` filler token (id 2167) until the generation-slot position matches the paired real query within ±2 tokens (all 810 matched, 0 over tolerance, median residual 0). Each dummy pair carries the matched `real_query_id`, so the real and dummy arms join per pair on `(context_id, real_query_id)` — every pair is its own length-and-position control.

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

The matched dummy query (1 of 810 dummy pairs; all share the same `dummy_base`, varying only filler length):

<details>
<summary>One length-matched dummy pair, verbatim (cherry-picked; full dummy battery in battery_dummy.json below)</summary>

```
pair_id        persona_software_engineer__q_dummy_for_ontopic_short_0
real_query_id  q_ontopic_short_0   (matched real query: "What is the best way to learn a new language?")
dummy_text     "Please continue with whatever you think is most appropriate here."
target_query_end_idx 30   achieved_query_end_idx 30   length_residual_tokens 0
full_prompt    <|im_start|>system\nYou are a software engineer who builds web applications.<|im_end|>\n<|im_start|>user\nPlease continue with whatever you think is most appropriate here.<|im_end|>\n<|im_start|>assistant\n
```

</details>

Complete probe batteries (all 810 pairs + offsets + decoded spans): real [`battery.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/inputs/battery.json), dummy [`battery_dummy.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/f94c0d15be2b09e936d7607c715bb193559b221d/issue654_query_displacement/inputs/battery_dummy.json), and [`extraction_manifest.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/analysis_tensors/extraction_manifest.json).

### Generated

n/a — no model completions are generated. Each forward pass yields residual-stream vectors (28 layers × 3584 dims at the context-end, query-end, and generation-slot positions), not text. The model emits nothing to judge; every data point is a per-layer cosine or CKA scalar from these vectors. The real per-pair residual banks (810 pair files + 81 context-only companion files, fp32) are at [`analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/analysis_tensors), and the round-2 dummy banks (810 pair + 81 context-only) at [`analysis_tensors/dummy/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/f94c0d15be2b09e936d7607c715bb193559b221d/issue654_query_displacement/analysis_tensors/dummy).

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` (base, no fine-tuning), bf16, `output_hidden_states=True` |
| num_hidden_layers / hidden_size | 28 / 3584 (asserted at runtime) |
| Read positions | context-span-end token, query-span-end token, assistant-generation slot (companion) |
| Centering | `global_mean` (per-bank, per-layer); within-tier + global shuffled floors both computed |
| Floor | shuffled-pair derangement, B=1000, seed 42 |
| CKA | linear CKA (Kornblith 2019, HSIC-Frobenius form), float64 |
| Pairs / forwards (real) | 810 (context, query) pairs; 891 forwards incl. 81 context-only companion reads |
| Dummy arm (round 2) | 810 length-matched content-meaningless pairs; `dummy_base` "Please continue…" + `" really"` filler (id 2167); all 810 within ±2 tokens of the matched real query-end (0 over tolerance); joined per pair on `(context_id, real_query_id)`, 810/810 matched, 0 unmatched |
| Seed | 42 (single seed) |

**Artifacts:**
- **Methodology reference:** [docs/methodology/issue_654.md](https://github.com/superkaiba/explore-persona-space/blob/cd8358444413288f7861be55d58380a72529c4e7/docs/methodology/issue_654.md) · [gist](https://gist.github.com/superkaiba/d2533088ba71df5685cd76794178d21b)
- Per-layer displacement JSON + per-tier cells: `eval_results/issue_654/per_layer_displacement.json` (+ `cells/`), committed at SHA `92ddfce6bbc78df8675b1747462b1ec1c74b7d6f`.
- Round-2 companion-gap JSON (per-layer per-tier real − dummy gap + per-pair SE + per-length breakdown + unmatched audit): `eval_results/issue_654/length-matched-dummy-query-control/companion_gap.json`, committed at SHA `3dc022aaf3092cf6b5af53422abba7f9d3c08577`.
- Real dual-position residual banks + manifest: [`analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/analysis_tensors) (HF data repo). `list_repo_files` confirms **892 files**: 810 pair `.pt` + 81 context-only `.pt` + 1 `extraction_manifest.json`.
- Round-2 dummy residual banks: [`analysis_tensors/dummy/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/f94c0d15be2b09e936d7607c715bb193559b221d/issue654_query_displacement/analysis_tensors/dummy) (HF data repo). `list_repo_files` confirms **892 files**: 810 pair `.pt` + 81 context-only `.pt` + 1 `extraction_manifest.json`.
- Probe batteries: [`inputs/battery.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/82d16a6faa7f8781163bf215154ed57296364780/issue654_query_displacement/inputs/battery.json), [`inputs/battery_dummy.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/f94c0d15be2b09e936d7607c715bb193559b221d/issue654_query_displacement/inputs/battery_dummy.json).
- Figures (blog style, PNG + PDF + meta): `figures/issue_654/{hero_displacement,query_length_split,gap_per_tier,companion,cosine_vs_cka}_blog.*` at SHA `994feb766d4cce377032697b269ca823029d631a`; round-2 `figures/issue_654/query_content_vs_length_gap_blog.*` at SHA `86b6c65b5a4482efdaf69341c4295e0271f430bc`.

**Compute:** GPU extraction (real: 891 forwards; dummy: 891 forwards, batch 1, 28 layers, 2-3 positions/forward) on 1× A100-80 (GCP `a2-ultragpu-1g`, `lora-7b` intent), ~0.4 GPU-h per arm. CPU metric phase (centered cosine + B=1000 derangement floors over 5 strata + float64 linear CKA over 810×3584 banks + the round-2 per-pair companion-gap join over both arms + figures) ~30 min off-pod. No judge / API cost.

**Code:**
- `scripts/issue654_build_battery.py` (battery), `scripts/issue654_extract.py` (dual-position extraction), `scripts/issue654_analyze.py` (metrics + `--companion-gap` dummy-vs-real mode), `scripts/issue654_hero_figs.py` (blog figures), `scripts/issue654_dispatch.sh` (GCP dispatch).
- New library: `src/explore_persona_space/analysis/probes.py::extract_dual_position_activations`; `representation_shift.py::linear_cka` + `cka_per_layer` (+ `tests/test_linear_cka.py`).
- Reproduce the round-2 companion-gap analysis from the uploaded banks:
  ```bash
  uv run python scripts/issue654_analyze.py --companion-gap \
    --real-banks data/issue654/hf_snapshot/issue654_query_displacement/analysis_tensors \
    --dummy-banks data/issue654/hf_snapshot/issue654_query_displacement/analysis_tensors/dummy \
    --context-only data/issue654/hf_snapshot/issue654_query_displacement/analysis_tensors/context_only \
    --out eval_results/issue_654/length-matched-dummy-query-control/ --figures --fig-dir figures/
  ```
- Git commit: `86b6c65b5a4482efdaf69341c4295e0271f430bc` (branch `issue-654`).

**Context:**
- **Created / run:** task created 2026-06-16; round-1 results landed 2026-06-17; round-2 (length-matched dummy-query control) landed 2026-06-17 (UTC).
- **Follow-up to:** round 1 was a fresh direction (no parent), sibling-positioned against [#594](https://eps.superkaiba.com/tasks/594) (context-vector geometry, one read position) and [#634](https://eps.superkaiba.com/tasks/634) (role-vs-persona bank); first to contrast two within-prompt positions. Round 2 is a cheap-band same-issue follow-up auto-run on this same issue (label `length-matched-dummy-query-control`), addressing the length/position confound flagged in round-1 Finding 4.
- **Originating prompt(s), verbatim:**
  > check how similar activations are after the context vs after the user query (for a variety of contexts and user queries), ask clarifying questions


