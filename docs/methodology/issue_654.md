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

*This document describes how the experiment was run. For the result and what it
means, see the [task body](https://eps.superkaiba.com/tasks/654).*
