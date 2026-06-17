# Methodology — issue 654: per-layer query displacement of the context-conditioned residual stream in Qwen-2.5-7B-Instruct

A methodology + hyperparameter reference for experiment #654 (Explore
Persona Space), with verbatim battery / extraction examples pulled straight
from the artifacts. No interpretation.

- Task: [https://eps.superkaiba.com/tasks/654](https://eps.superkaiba.com/tasks/654)
- Model: `Qwen/Qwen2.5-7B-Instruct`

---

## 1. Overview

- **Model:** base `Qwen/Qwen2.5-7B-Instruct` (bf16), no fine-tuning; a single
  `output_hidden_states=True` forward pass per (context, query) pair. The Goal
  (verbatim frontmatter): measure, per layer, how much appending a user query
  displaces the last-token residual-stream state away from the
  context-established state, across persona / generic-instruction / ICL /
  wild-chat contexts and varied queries, to characterize context-dominant vs
  query-dominant layer regimes.
- **The manipulation (the single object read):** two token positions inside the
  SAME ChatML prompt — the context-span end token and the query-span end token
  — read at every layer; the dependent variable is the per-layer displacement
  between them.
- **Design cells:** `context_type` (4) × `query_type` (topicality × length, 4) ×
  `layer` (0–27). Context instances: persona (11), generic (20), ICL (20),
  wildchat (30) = 81 contexts × a fixed 10-query bank = 810 (context, query)
  pairs.
- **Dependent variables:** (PRIMARY) per-layer per-pair centered cosine
  (context-end vs query-end), reported matched-minus-shuffled against a
  shuffled-pair derangement floor; (SECONDARY) linear CKA between the
  context-end and query-end banks per layer; (CONTROL / co-primary) a
  same-position companion cosine (context-only vs context+query at the
  assistant-generation slot).
- **No judge:** no model-scored DV — every metric is linear algebra
  (cosine, CKA, permutation floor) over extracted activations. No training.
- **Provenance:** reuses #594's WildChat loader (`issue617_build_wildchat_slice.py`),
  the project ChatML conventions, the `global_mean` bank centering (#536), the
  shuffled-pair permutation-floor machinery, and `personas.py::PERSONAS` /
  `EVAL_QUESTIONS`. New surfaces: the dual-position extractor, linear CKA, and
  the four `scripts/issue654_*` entrypoints.

---

## 2. Hyperparameters

**N/A — no model training.** This is a forward-pass measurement experiment;
there are no training or generation hyperparameters. The load-bearing
measurement constants are detailed in §4 Evaluation; the ones the
clean-result body's slimmed Parameters table cites are mirrored here so the
table reconciles against §2 directly.

The measurement constants (full detail + sources in §4):

| Parameter | Value | Source |
|---|---|---|
| Centering | `global_mean` (per-bank, per-layer); within-tier + global shuffled floors both computed | §4; `issue654_analyze.py:144-154,382`; #536 |
| Floor | shuffled-pair derangement, B=1000, seed 42 | §4; `issue654_analyze.py:59-60,216-221` |
| CKA | linear CKA (Kornblith 2019, HSIC-Frobenius form), float64 | §4; `representation_shift.py:165-213`; 1905.00414 |
| Pairs / forwards | 810 (context, query) pairs; 891 forwards incl. 81 context-only companion reads | §4; `issue654_build_battery.py`; `issue654_extract.py` |

The forward-pass / extraction knobs that *are* set:

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` (base, no fine-tuning) | `scripts/issue654_extract.py:57` (`QWEN_MODEL`) |
| Dtype | `bfloat16` (production); `float32` for the CPU structure smoke | `issue654_extract.py:118-123,140` |
| `num_hidden_layers` / `hidden_size` | **28 / 3584** (asserted from `model.config` at startup) | `issue654_extract.py:58-59,152-157` |
| hidden-states tuple length | 29 (index 0 = embedding, `hs[L+1]` = post-block-L residual) | `probes.py:133-134,206` |
| Batch size | 1 (no padding; per-prompt explicit token offsets) | `probes.py:198`; plan §8 |
| `add_special_tokens` | `False` (ChatML special tokens already in the prompt string) | `probes.py:198`, `extract_dual_position_activations` docstring |
| Forward gradient | `@torch.no_grad()` | `probes.py:117` |
| Read positions per pair | context-span-end token, query-span-end token, + assistant-generation slot (`readout_position=-1`) | `issue654_extract.py:186-194,235-243` |
| Companion reads | one extra forward per DISTINCT context (context-only prompt, assistant-gen slot) | `issue654_extract.py:160-206` |
| **Seed** | **42** (battery build, query selection, derangement floor) | `issue654_build_battery.py:54`; `issue654_analyze.py:59` |
| Activation storage | per-pair `.pt`, fp32, CPU, atomic write | `probes.py:191-195`; `issue654_extract.py:76-82,244-263` |
| Offset-failure kill fraction | > 5% of pairs failing the position assert → fail loud | `issue654_extract.py:60-61,284-328` |
| Device | `cuda` (single GPU) | `issue654_extract.py:112`; plan §9 |
| Pod intent / backend | intent `lora-7b` (GCP `a2-ultragpu-1g`, 1× A100-80); backend auto (GCP-first) | plan §8/§9/§10 |
| GCP max-run-duration fence | 24 h default (worst-case wall < 1 h) | plan §8 |
| Estimated GPU-hours | 3 (single forward-pass loop, 891 forwards) | plan §8 |

---

## 3. Training data

**N/A — no training mix.** Nothing is trained and no behavior is implanted.
The experiment reads activations off the base model. The input artifacts it
*reads* — the context battery and query bank — are described in §4 Evaluation.

---

## 4. Evaluation

### Dependent variables (construct → metric)

For pair *i* = (context *c*, query *q*) and layer *L*, the extractor yields two
vectors in ℝ³⁵⁸⁴: the context-end residual `a_ctx[i,L]` (last token of the
rendered context block) and the query-end residual `a_qry[i,L]` (last prompt
token before the assistant marker). Both are read off the model's actual
residual stream in one natural forward pass — no teacher-forced continuation.

- **Per-pair centered cosine (PRIMARY).** Per layer, each bank is globally
  mean-centered ONCE on its own per-layer centroid over all pairs, then
  L2-normalized; the per-pair cosine is `⟨ĉtx[i,L], q̂ry[i,L]⟩`. Centering and
  the floor both consume the same globally-centered banks (never re-centered per
  tier). Impl: `issue654_analyze.py::_global_center_normalize`,
  `_centered_cos_per_layer`.
- **Shuffled-pair floor.** For a derangement π (i ≠ π(i)), `⟨ĉtx[i,L], q̂ry[π(i),L]⟩`;
  floor = mean ± 2.5/97.5 band over B=1000 derangements (seed 42), computed
  globally AND per-tier (within-type). Headline = matched-minus-shuffled per
  layer. Impl: `issue654_analyze.py::_derangement_floor`, `_derangement`.
- **Raw (uncentered) cosine.** Reported alongside the centered cosine as an
  anisotropy companion. Impl: `_raw_cos_per_layer`.
- **Linear CKA per layer (SECONDARY).** `cka_per_layer(A_ctx, A_qry)` — linear
  CKA (Kornblith et al. 2019) between the two banks per layer (paired rows), with
  a row-permuted-bank CKA floor. Impl:
  `representation_shift.py::linear_cka` / `cka_per_layer`,
  `issue654_analyze.py::_cka_per_layer_from_banks` / `_cka_shuffled_floor`.
- **Companion same-position contrast (CONTROL / co-primary).** Per context, a
  separate forward over the context-only prompt reads the assistant-generation
  slot; the full-prompt assistant-gen slot is captured in the SAME forward as
  the two span reads (`readout_position=-1`). The companion DV is the per-layer
  cosine of (context-only readout) vs (full-prompt readout), both at the FIXED
  assistant-gen position — the only difference is the presence of the query.
  Impl: `issue654_analyze.py::_companion_cosine_per_layer`.

### Analysis-design constants

| Constant | Value | Source |
|---|---|---|
| Centering | `global_mean` (per-bank, per-layer, over all pairs) | `issue654_analyze.py:144-154,382`; #536 |
| Derangement count B | 1000 (seed 42) | `issue654_analyze.py:59-60` |
| Floor band | 2.5 / 97.5 percentile over B derangements | `issue654_analyze.py:216-221` |
| Layers read / reported | full 0–27; reporting anchors `[7, 14, 21, 27]` | `issue654_analyze.py:61`; plan §11 |
| CKA recipe | linear CKA, HSIC-Frobenius / geometric-mean-self-HSIC, fp64 accumulation | `representation_shift.py:165-213`; 1905.00414 |
| CKA degenerate-bank guard | constant bank → 0.0 (not NaN); `n ≥ 2` asserted | `representation_shift.py:190-213` |

### Context battery (probe set)

| Context tier | N | Source | Why chosen |
|---|---|---|---|
| Persona | 11 | `personas.py::PERSONAS` (10 usable; `local_resident` / `local_historian` / `biographer` / `marine_biologist` excluded) + `ASSISTANT_PROMPT` | persona-instruction system prompts; the canonical persona bank used across #404/#458/#594 |
| Generic instruction | 20 | `HuggingFaceH4/ultrachat_200k` `train_sft` first-user prompts (as system instructions) | established, ungated, multi-domain task instructions |
| ICL example set | 20 | 4-exchange few-shot blocks from a HELD-OUT UltraChat slice (real Q/A, disjoint from the generic tier; hard-asserted) | a real in-context demonstration regime |
| WildChat real chat | 30 | `allenai/WildChat-1M` slice via `issue617_build_wildchat_slice.py` (first user+assistant exchange as context) | tier-1 real-chat realism anchor |

Total: 81 contexts. Persona-tier exclusions are an explicit list
(`issue654_build_battery.py:62-67`); the build asserts no `{...}` placeholder
survives in any selected persona prompt. Generic / ICL pools are
hard-asserted disjoint, and ICL demos are asserted disjoint from the eval
query bank (`issue654_build_battery.py:301-306`).

### Query bank (probe set)

A fixed 10 queries, each tagged topicality × length, paired against EVERY
context (so topicality/length are within-bank factors comparable across
context types). Source: `issue654_build_battery.py:84-143`.

| Query slug pattern | N | Topicality | Length | Source |
|---|---|---|---|---|
| `q_ontopic_short_{0..2}` | 3 | on | short | first 3 `EVAL_QUESTIONS` |
| `q_ontopic_long_{0..1}` | 2 | on | long | 2 frozen expanded questions |
| `q_offtopic_short_{0..2}` | 3 | off | short | 3 frozen off-topic questions |
| `q_offtopic_long_{0..1}` | 2 | off | long | 2 frozen off-topic questions |

Pairs / forwards: 81 × 10 = **810 (context, query) pairs**; **891 forwards**
including the 81 distinct context-only companion reads.

### Verbatim example probes

The 3 short on-topic queries are the first three `EVAL_QUESTIONS`:

```text
q_ontopic_short_0  "What is the best way to learn a new language?"
q_ontopic_short_1  "Can you explain how photosynthesis works?"
q_ontopic_short_2  "What are some tips for managing stress?"
```

Two of the frozen off-topic queries (`issue654_build_battery.py:102-106`):

```text
q_offtopic_short_0  "Who won the 1994 FIFA World Cup?"
q_offtopic_short_1  "What is the capital city of Australia?"
```

No judge / rubric — every DV is computed directly from activations.

---

## 5. Worked examples

Each row is the full path from the rendered ChatML prompt → the two derived
token offsets (asserted prefix-stable, fail-loud CPU-side) → the per-pair
activation bank written by the extractor. Values below are taken verbatim from
the `--smoke` battery + extraction manifest (the production run shares the
identical schema; the production banks land on the HF data repo, see §6).

### Persona context, on-topic short query

<!-- cherry-picked for illustration; full battery at the HF data repo link in §6 -->

```text
pair_id        : persona_software_engineer__q_ontopic_short_0
context_type   : persona
full_prompt    : <|im_start|>system
                 You are a software engineer who builds web applications.<|im_end|>
                 <|im_start|>user
                 What is the best way to learn a new language?<|im_end|>
                 <|im_start|>assistant
ctx_end_idx    : 14   (decoded token: "\n")
query_end_idx  : 30   (decoded token: "\n")
seq_len        : 34
bank written   : pair_000000.pt
  context_end  : (28, 3584) fp32   # residual at token 14, every layer
  query_end    : (28, 3584) fp32   # residual at token 30, every layer
  readout      : (28, 3584) fp32   # full-prompt assistant-gen slot (companion)
companion file : context_only/persona_software_engineer.pt
```

### ICL context, on-topic short query (long context span)

<!-- cherry-picked for illustration; full battery at the HF data repo link in §6 -->

```text
pair_id        : icl_000__q_ontopic_short_0
context_type   : icl
full_prompt    : <|im_start|>system
                 You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
                 <|im_start|>user
                 Write a comprehensive blog post of at least 1000 words about the top 10 most
                 eco-friendly cities ... <|im_end|>
                 <|im_start|>assistant
                 ... (4 user/assistant demonstration exchanges) ...
                 <|im_start|>user
                 What is the best way to learn a new language?<|im_end|>
                 <|im_start|>assistant
ctx_end_idx    : 3035  (decoded token: "\n")
query_end_idx  : 3051  (decoded token: "\n")
content_tokens : 3055
```

### WildChat context, on-topic short query

<!-- cherry-picked for illustration; full battery at the HF data repo link in §6 -->

```text
pair_id        : wildchat_wc_000000__q_ontopic_short_0
context_type   : wildchat
full_prompt    : <|im_start|>system
                 You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
                 <|im_start|>user
                 Hey there! Are you familiar with reality shifting? ... <|im_end|>
                 <|im_start|>assistant
                 ... (first user+assistant exchange of the real chat) ...
                 <|im_start|>user
                 What is the best way to learn a new language?<|im_end|>
                 <|im_start|>assistant
ctx_end_idx    : 987   (decoded token: "\n")
query_end_idx  : 1003  (decoded token: "\n")
content_tokens : 1007
```

Offset derivation (`issue654_build_battery.py::derive_pair`): the context-only
render (no user turn, no generation prompt) is tokenized to get
`context_end_idx = len(ctx_ids) - 1`; the context + user-turn render (no
generation prompt) gives `query_end_idx`; the full prompt (with the
assistant-generation marker) is the extraction input. Hard asserts:
`full_ids[:len(ctx_ids)] == ctx_ids` (context block is a strict token prefix),
the no-gen render is a prefix of the gen render, and
`0 ≤ context_end_idx < query_end_idx < seq_len`. The extractor re-confirms the
ordering at extraction time and writes a `decoded_ctx_end_tok` /
`decoded_query_end_tok` sanity string per pair into the manifest.

---

## 6. Artifacts index

| Artifact | Pinned link |
|---|---|
| Per-pair activation banks (`pair_*.pt`, `context_only/*.pt`) + `extraction_manifest.json` | HF data repo `superkaiba1/explore-persona-space-data` → `issue654_query_displacement/analysis_tensors/` |
| Battery input (`battery.json`) | HF data repo → `issue654_query_displacement/inputs/battery.json` |
| Per-layer displacement JSON | [`eval_results/issue_654/per_layer_displacement.json`](https://github.com/superkaiba/explore-persona-space/blob/2d6200b4d91765003d0a5300a74b0efce9ef4d76/scripts/issue654_analyze.py) (written by the analyze script; per-cell breakdowns under `eval_results/issue_654/cells/`) |
| Figures | `figures/issue_654/*.png` (hero + exploratory dump, `paper_plots` rcParams) |
| Battery builder (CPU) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/2d6200b4d91765003d0a5300a74b0efce9ef4d76/scripts/issue654_build_battery.py) |
| Dual-position extractor (GPU) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/2d6200b4d91765003d0a5300a74b0efce9ef4d76/scripts/issue654_extract.py) |
| Metrics + figures (CPU) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/2d6200b4d91765003d0a5300a74b0efce9ef4d76/scripts/issue654_analyze.py) |
| GPU dispatcher | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/2d6200b4d91765003d0a5300a74b0efce9ef4d76/scripts/issue654_dispatch.sh) |
| `extract_dual_position_activations` | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/2d6200b4d91765003d0a5300a74b0efce9ef4d76/src/explore_persona_space/analysis/probes.py) |
| `linear_cka` / `cka_per_layer` | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/2d6200b4d91765003d0a5300a74b0efce9ef4d76/src/explore_persona_space/analysis/representation_shift.py) |
| WandB project | `issue654_query_displacement` |
| Code commit | `2d6200b4d91765003d0a5300a74b0efce9ef4d76` |
| Compute | intent `lora-7b` (GCP `a2-ultragpu-1g`, 1× A100-80; backend auto, GCP-first); ~3 GPU-h budgeted, ~0.5 h wall; CPU metric/figure phase off-pod on the VM after termination |

### Reproduce end-to-end

```bash
# Step 1 — build the (context, query) pair battery (CPU; tokenizer only)
uv run python scripts/issue654_build_battery.py --out data/issue654/battery.json

# Step 2 — dual-position residual-stream extraction (1x GPU)
uv run python scripts/issue654_extract.py \
    --battery data/issue654/battery.json \
    --out-dir data/issue654/dual_pos --device cuda
# (the GPU dispatcher wraps build + extract + HF upload:
#  bash scripts/issue654_dispatch.sh --issue 654 --phase extract)

# Step 3+4 — per-layer metrics + figures (CPU, off-pod, post-terminate)
uv run python scripts/issue654_analyze.py \
    --banks data/issue654/dual_pos \
    --out eval_results/issue_654 --figures
```

The auto/GCP router runs Step 2 on-pod via
`--workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue654_dispatch.sh --issue 654 --phase extract'`.

---

## length-matched-dummy-query-control arm

A cheap-band same-issue follow-up round (label
`length-matched-dummy-query-control`) that adds ONE query arm to the §3/§4
battery. It changes exactly one variable — query *content* at matched length
— and inherits everything else (model, 81 contexts, read positions, centering,
floor, CKA, seed 42) verbatim from the parent run. Plan amendment:
`tasks/.../654/plans/v5.md` (a one-variable diff against v4).

### Construct being measured

- **The variable.** Each real query carries both content AND length/token
  position. The companion read in §4 (context-only assistant-gen slot vs
  full-prompt assistant-gen slot, both at the fixed generation position) cannot
  separate the two. This arm holds query length and read position fixed while
  removing query content, by pairing each real query with a token-length-matched
  content-neutral dummy query and re-reading the same slot.
- **The new DV.** Per-layer per-tier same-slot companion-curve gap, computed
  per-pair so each (context, query) is its own control:
  `gap(L) = companion_cos_real(context, query, L) − companion_cos_dummy(context, query, L)`
  aggregated to the per-tier mean. The v4 companion-read definition (§4) is
  unchanged; the dummy arm supplies the content-matched comparison curve.
- **Measurement validity (inherited).** On-distribution — both arms read the
  model's real residual stream at the real assistant-generation slot in natural
  forward passes; the dummy is a real grammatical user turn, not a teacher-forced
  stub. Direct difference of two computed cosine curves, no derived-input
  dependency.
- **The one new hyperparameter — the filler-string design** (everything else
  `Source: plans/v4 §11`):

| Parameter | Value | Source |
|---|---|---|
| Dummy base sentence | `"Please continue with whatever you think is most appropriate here."` | `issue654_build_battery_dummy.py:DUMMY_BASE` (plan v5 §2/§11) |
| Filler word (length pad) | `" really"` — single Qwen-2.5-7B token id **2167** (asserted at build time) | `issue654_build_battery_dummy.py:FILLER_WORD,FILLER_TOKEN_ID:99-101` |
| Length-match target | each dummy's `query_end_idx` matched to the real query's `query_end_idx` under the SAME context | `_build_dummy_text`; `derive_pair` (reused) |
| Residual tolerance / flag | ±2 tokens; flag if > 10% of pairs exceed it | `issue654_build_battery_dummy.py:RESIDUAL_TOKEN_TOL,RESIDUAL_FLAG_FRACTION` |
| `<\|im_pad\|>` rejected | encodes to 6 ordinary subwords, not a single token → off-distribution | plan v5 §11 rejected-alternatives |
| Seed | 42 (inherited) | `issue654_build_battery_dummy.py` (`SEED` from parent build) |

### Dummy-battery construction recipe

`issue654_build_battery_dummy.py` (tokenizer-only, no model load; runs CPU-side
on-pod before the GPU phase). Per (context, real query):

1. Read the matched real query's `query_end_idx` from the parent's frozen
   `battery.json` (the per-context length target); recover that context's
   message list from the real pair's `context_only_prompt`
   (`_parse_chatml_messages`) so the dummy renders under the IDENTICAL context.
2. Render `context + DUMMY_BASE`; derive its `query_end_idx` (same no-gen
   render as the parent's `derive_pair`).
3. **If short of target:** append `" really"` filler tokens one at a time,
   re-deriving `query_end_idx` after each append (per-append re-derivation
   absorbs any ChatML-context tokenization drift); if it overshoots by one
   filler token, drop the trailing `" really"`.
4. **If `DUMMY_BASE` alone overshoots** (a very short real query): truncate the
   base sentence at a word boundary until at/under target, then top up with
   filler to hit the target exactly.
5. Run the parent's `derive_pair` ordering/prefix asserts on the dummy pair
   (`full_ids[:len(ctx_ids)] == ctx_ids`, `0 ≤ ctx_end < query_end < seq_len`).
6. Record per pair: realized `dummy_text`, `target_query_end_idx`,
   `achieved_query_end_idx`, `length_residual_tokens` (= achieved − target),
   and the join key `real_query_id`
   (`q_ontopic_short_0` → `q_dummy_for_ontopic_short_0`).

Build-time fail-loud asserts: the filler word encodes to exactly `[2167]`; the
`DUMMY_BASE` + every realized dummy string is disjoint from the 10 real eval
queries AND every reconstructed context turn (system/user content) AND the real
query bank (so a dummy can never echo a context or eval string).

**Realized length match (production `battery_dummy.json` meta, HF rev
`f94c0d15…`):** 810 dummy pairs over 81 contexts; **residual distribution =
{0: 810}** — every dummy hit its matched real query's `query_end_idx` exactly;
**0 / 810 pairs over the ±2-token tolerance** (`residual_match_flag: false`).

### Companion-gap analysis recipe

`issue654_analyze.py --companion-gap` (CPU, off-pod on the VM after the pod is
deleted; reads the uploaded dummy `.pt` banks + the parent's cached real-arm +
context-only banks). Mechanics:

- `_load_readout_banks` loads only the per-pair `readout` (assistant-gen slot)
  banks + each context's companion context-only readout — lighter than the
  full §4 load (no context-end/query-end banks).
- `_per_pair_companion_cos`: per (context_id, real_query_id), the per-layer
  cosine of (context-only assistant-gen readout) vs (full-prompt assistant-gen
  readout), L2-normalized — the v4 companion read, computed in float64.
  **Both arms read against the SAME cached context-only banks** (the
  context-only side is identical with/without a query).
- **Join** the real and dummy arms on `(context_id, real_query_id)`: the dummy
  pair stores its matched real query's id under `real_query_id`; each real pair
  has exactly one matched dummy. Per-pair `gap = cos_real − cos_dummy`.
- **Aggregate** per tier, per length bin (`short` / `long`), and per
  tier×length, plus an overall curve. Each aggregate reports `gap_mean`,
  `gap_se` (sample sd / √n, ddof=1; the falsification band is read from this
  per-pair SE, NOT a hard-coded 0.03), and `n`.
- **Late-layer-trough summary:** the overall gap mean ± mean-SE over layers
  **L23–L27** (the band where the parent companion bottomed), as the single
  summary number. Anchor layers `[7, 14, 21, 27]`; full curve layers 0–27.
- **Unmatched-pair audit** carried in the JSON. Realized:
  `n_matched_pairs = 810`, `n_unmatched_real = 0`, `n_unmatched_dummy = 0`,
  `n_skipped_*_missing_context = 0`. Per-tier n: persona 110, generic 200, icl
  200, wildchat 300. Per-length n: short 486, long 324.
- **Figure** (`make_companion_gap_figure`): two panels — (left) per-tier real
  (solid) vs dummy (dashed) companion curves, (right) per-tier gap curve with
  the per-pair gap SE shaded. `blog` `paper_plots` rcParams, plain-English tier
  labels.

### Worked examples (dummy pairs)

<!-- cherry-picked for illustration; full dummy battery at the HF link in the artifacts table below -->

A short on-topic real query forces the base sentence to be TRUNCATED to hit the
shorter target length; a long real query is PADDED with `" really"` filler.
Verbatim from `battery_dummy.json` (production, HF rev `f94c0d15…`):

```text
pair_id        : persona_software_engineer__q_dummy_for_ontopic_short_0
real_query_id  : q_ontopic_short_0   (join key back to the real arm)
context_type   : persona   topicality/length: on / short
target_qend    : 30   achieved: 30   residual: +0
dummy_text     : "Please continue with whatever you think is most appropriate here."

pair_id        : persona_software_engineer__q_dummy_for_ontopic_short_1
real_query_id  : q_ontopic_short_1
context_type   : persona   topicality/length: on / short
target_qend    : 27   achieved: 27   residual: +0
dummy_text     : "Please continue with whatever you think is."   # base truncated at a word boundary to length-match a shorter real query

pair_id        : persona_software_engineer__q_dummy_for_ontopic_long_0
real_query_id  : q_ontopic_long_0
context_type   : persona   topicality/length: on / long
target_qend    : 59   achieved: 59   residual: +0
dummy_text     : "Please continue with whatever you think is most appropriate here. really really really ... really"   # padded with the single-token filler word to match a longer real query
```

The full-prompt assistant-gen readout for each dummy pair is then extracted
through the UNCHANGED `issue654_extract.py` at the same three positions; the
companion's context-only side is REUSED from the parent's 81 cached banks (not
re-extracted) via the extractor's `--reuse-context-only` flag.

### New artifacts

| Artifact | Pinned link |
|---|---|
| Companion-gap JSON (per-layer per-tier real − dummy gap + per-pair SE + per-length-bin + unmatched audit + late-trough summary) | [`eval_results/issue_654/length-matched-dummy-query-control/companion_gap.json`](https://github.com/superkaiba/explore-persona-space/blob/86b6c65b5a4482efdaf69341c4295e0271f430bc/eval_results/issue_654/length-matched-dummy-query-control/companion_gap.json) |
| Dummy residual banks (`pair_*.pt`, `context_only/*.pt`, `extraction_manifest.json`; 892 files) | HF data repo [`analysis_tensors/dummy/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/f94c0d15be2b09e936d7607c715bb193559b221d/issue654_query_displacement/analysis_tensors/dummy) (rev `f94c0d15…`) |
| Dummy battery input (`battery_dummy.json`, 810 pairs, residual-match meta) | HF data repo [`inputs/battery_dummy.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/f94c0d15be2b09e936d7607c715bb193559b221d/issue654_query_displacement/inputs/battery_dummy.json) |
| Companion-gap figure (real vs dummy curves + gap SE band) | `figures/issue_654/query_content_vs_length_gap_blog.{png,pdf,meta.json}` at SHA `86b6c65b5a4482efdaf69341c4295e0271f430bc` |
| Dummy-battery builder (CPU) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/86b6c65b5a4482efdaf69341c4295e0271f430bc/scripts/issue654_build_battery_dummy.py) |
| Pinned-parent battery fetcher (identity verify) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/86b6c65b5a4482efdaf69341c4295e0271f430bc/scripts/issue654_fetch_pinned_battery.py) |
| Analyzer companion-gap mode | [`issue654_analyze.py --companion-gap`](https://github.com/superkaiba/explore-persona-space/blob/86b6c65b5a4482efdaf69341c4295e0271f430bc/scripts/issue654_analyze.py) |
| Dispatcher dummy-arm path (`--arm dummy`) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/86b6c65b5a4482efdaf69341c4295e0271f430bc/scripts/issue654_dispatch.sh) |
| Code commit (round 2) | `86b6c65b5a4482efdaf69341c4295e0271f430bc` |
| Compute (round 2) | dummy extraction 810 forwards (batch 1, 28 layers, 2–3 positions/forward; context-only side reused, NOT re-run) on 1× A100-80 (GCP `a2-ultragpu-1g`, `lora-7b` intent, backend `gcp`), ~0.4 GPU-h; CPU companion-gap join + figure off-pod ~30 min. No judge / API cost. |

**Reuse provenance:** the dummy arm REUSES, from the parent's pinned HF revision
`82d16a6faa7f8781163bf215154ed57296364780`: the frozen `inputs/battery.json`
(per-context length targets + exact contexts), the 81 cached
`analysis_tensors/context_only/*.pt` companion banks, and the 810 real-query
`analysis_tensors/pair_*.pt` `readout` banks (the real-arm comparison curve).
Sourcing contexts + banks from the same pinned revision is what keeps the
single-variable control valid; `issue654_fetch_pinned_battery.py` fail-loud
verifies context identity before the dummy battery is built.

### Reproduce the dummy arm

```bash
# Step 1 — build the length-matched dummy battery (CPU; tokenizer only; reads the
#  parent's battery.json for per-context length targets)
uv run python scripts/issue654_build_battery_dummy.py \
    --real-battery data/issue654/battery.json \
    --out data/issue654/battery_dummy.json

# Step 2 — dummy-pair extraction (1x GPU); context-only side reused from HF
#  (the dispatcher fetches the 81 cached parent context_only banks first):
bash scripts/issue654_dispatch.sh --issue 654 --phase extract --arm dummy

# Step 3 — per-layer per-tier dummy-vs-real companion gap (CPU, off-pod)
uv run python scripts/issue654_analyze.py --companion-gap \
    --real-banks data/issue654/hf_snapshot/issue654_query_displacement/analysis_tensors \
    --dummy-banks data/issue654/hf_snapshot/issue654_query_displacement/analysis_tensors/dummy \
    --context-only data/issue654/hf_snapshot/issue654_query_displacement/analysis_tensors/context_only \
    --out eval_results/issue_654/length-matched-dummy-query-control/ --figures --fig-dir figures/
```

`smoke = bash scripts/issue654_dispatch.sh --issue 654 --phase extract --arm dummy --smoke`
(first 4 contexts × first 2 real queries through the identical build + extract
path; prints 8 realized dummy strings + per-pair length residuals — the plan A13
manipulation check).

---

*This document describes how the experiment was run. For the result and what it
means, see the [task body](https://eps.superkaiba.com/tasks/654).*
