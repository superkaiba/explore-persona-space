# Methodology — issue 617: cluster a WildChat slice by topic and verify the best-separating pair in Qwen-2.5-7B-Instruct mid-band activations, then ship the pair as a reusable corpus

A methodology + hyperparameter reference for experiment #617 (Explore
Persona Space), with verbatim training / evaluation / output examples
pulled straight from the artifacts. No interpretation.

- Task: [https://eps.superkaiba.com/tasks/617](https://eps.superkaiba.com/tasks/617)
- Model: `Qwen/Qwen2.5-7B-Instruct`

---

## 1. Overview

- **No training.** Capture-only pipeline: filter a real WildChat slice, embed-and-cluster the first user turn, extract Qwen-2.5-7B-Instruct activations on the unique-prefix pool, score every cluster *pair* for separability, then ship the winning pair (with realistic completions) as a reusable HF corpus.
- **Manipulation (single variable):** the clustering configuration that produces each candidate cluster pair — KMeans at K ∈ {5, 10, 20} plus one HDBSCAN run — searched over all in-config pairs.
- **Design cells:** one extraction run over the capped unique-prefix pool (≤ 400 conversations + 1 synthetic `f6_default_template` instance, × 48 fixed probes); 119 candidate cluster pairs scored across the four clustering configs.
- **Dependent variable:** length-residualized centered-cosine k-NN family purity (k=4) at the best of L13 / L14 / L18, for each pair; the winner is the highest-purity pair subject to a selection-aware (global max-over-pairs × layers) 1000-shuffle permutation `winner.p_global ≤ 0.01`.
- **No judge.** Every read is deterministic on extracted activations (k-NN purity, length residualization, TF-IDF / length baselines, the permutation null). Realistic completions on the picked pair (Step 6) are unjudged on-policy samples shipped alongside the prefixes.
- **Provenance:** the extractor, the analyzer primitives (k-NN purity, length residualization, TF-IDF and length baselines, the max-over-axes permutation), and the WildChat filter / chat-messages helper are reused verbatim from #594. Schema-loose validation + one synthetic `f6_default_template` instance are the only #617-side additions to the extractor's contract.

---

## 2. Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Model (activation read) | `Qwen/Qwen2.5-7B-Instruct`, bf16, base (no training, no LoRA) | manifest `model` field |
| Decoder layers / hidden | 28 / 3584 | manifest `n_layers`, `hidden` |
| Read position | last input token under `add_generation_prompt=True` (the assistant-header newline slot) | `issue594_extract_context_vectors.py` @ a3c9071 |
| **Read layers (scored)** | **L13 / L14 / L18** (all 28 captured, scored at this band) | plan §11 (Source: #594) |
| **Sentence embedder (clustering)** | **`BAAI/bge-large-en-v1.5`**, 1024-dim, 512-token, **CLS-token pool (`last_hidden_state[:, 0]`) + L2-norm**, no instruction prefix | `issue617_cluster.py::embed_first_user_turns` @ a3c9071; M3 (plan §4 step 2) |
| Pooling-mode provenance | `pooling_mode: "cls"` recorded in `separability.json` (M3 guard) | `issue617_cluster.py::POOLING_MODE` @ a3c9071 |
| **KMeans** | `n_clusters ∈ {5, 10, 20}`, `n_init=10`, `random_state=42` | `issue617_common.py::CLUSTER_KS`; plan §4 step 2 |
| **HDBSCAN** | `min_cluster_size = max(30, target // 200) = 100` for the 20k slice, `metric="euclidean"`, noise label −1 excluded from pair pool | `issue617_cluster.py` @ a3c9071; plan §11 |
| WildChat slice target | 20,000 conversations | `issue617_common.py::SLICE_TARGET` |
| Slice scan cap | 200,000 rows | `issue617_common.py::SLICE_SCAN_CAP` |
| Slice filter | `language == "English"`, `toxic is False`, `redacted is False`, non-empty first user turn; dedup on first-user-message first-200-chars | `issue594_build_battery._wildchat_eligible` (reused verbatim) |
| Short prefix depth | 2 messages (first user + first assistant) | `issue617_build_wildchat_slice.py::SHORT_PREFIX_MSGS` |
| Long prefix depth | 8 messages (first 4 exchanges) | `issue617_build_wildchat_slice.py::LONG_PREFIX_MSGS` |
| **Unique-prefix extraction pool cap** | **400** unique `conv_id`s + 1 synthetic `f6_default_template` instance | `issue617_common.py::EXTRACTION_POOL_CAP`; plan M2 |
| Per-cluster floor (extraction) | 30 members retained per cluster that appears in any config | `issue617_common.py::PER_CLUSTER_FLOOR`; plan M2 |
| Per-cluster scoring N | ∈ [30, 50] prefixes | `issue617_common.py::PER_CLUSTER_FLOOR / PER_CLUSTER_SCORING_MAX` |
| Stratification | deterministic seed-42 stratified subsample, ≥ 30 members/cluster preserved | `issue617_build_extraction_battery.py` @ a3c9071 |
| Synthetic default instance | exactly one `{"id": "f6_default_template", "family": "f6_default_template", "prefix_messages": [], "system_prompt": null}`; extracted, excluded from pair pool | `issue617_common.py::SYNTHETIC_DEFAULT_ID`; plan M1 |
| Probe pool | fixed 48-paraphrase Betley pool via `fetch_preregistered_probes` | manifest `probe_pool_n=48`, hash `ad687bec…b015dd7` |
| Separability metric | best-of-{13,14,18} length-residualized centered-cosine k-NN purity, `knn_k=4`, `cosine_dist(centering="global_mean")` | `issue617_score_separability.py` @ a3c9071 (reusing #594 primitives) |
| Length covariate | `log1p(content_token_len)` (residualized out of activations before scoring) | manifest `ctx_token_len_content`; plan §6 |
| Baselines on the same scale | TF-IDF word-overlap purity (`text_baseline`) + length-only purity (`length_only_knn_purity`) | `issue594_analyze_context_geometry.py` (reused) |
| **Permutation null** | **B = 1000** label shuffles, **selection-aware (global) max over (pairs × layers)** under each shuffle; per-pair p retained uncorrected | `issue617_common.py::PERM_B`; plan SA1 + §4 step 4 |
| Pair winner rule | highest length-residualized purity s.t. `winner.p_global ≤ 0.01`; tie-break lower-K, then KMeans-before-HDBSCAN, then lexicographic | plan §4 step 4 / §11 |
| Fire thresholds | `length_residualized_purity ≥ 0.7` AND `winner.p_global ≤ 0.01` | `issue617_common.py::PURITY_THRESHOLD / PGLOBAL_THRESHOLD`; plan §6 |
| **Completion model** | **`Qwen/Qwen2.5-7B-Instruct`**, base, vLLM batched `LLM.generate()` | `issue617_sample_completions.py` @ a3c9071 |
| **Completion N / temperature** | **N = 4** completions per prefix, **T = 1.0** | `issue617_common.py::COMPLETION_N / COMPLETION_TEMP`; plan §11 (clarifier 4) |
| Completion `max_new_tokens` | 512 | `issue617_common.py::COMPLETION_MAX_NEW_TOKENS` |
| Completion `max_model_len` | 8192 (floor; the script bumps it to cover longest prompt + budget + margin, clamp 16384) | `issue617_dispatch.sh` lines 102–106 |
| Completion prefix cap | 200 prefixes per picked category | `issue617_common.py::COMPLETION_CAP_PER_CATEGORY` |
| **Seed** | **42** everywhere (slice sample, KMeans, prefix subsample/cap, permutation, completion sampling) | `issue617_common.py::SEED` |
| Code commit | `a3c90717630e5c369307237707d867bd18cd1e50` (branch `issue-617`) | `git rev-parse issue-617` |

This is the canonical complete hyperparameter table; the task body's
`## Reproducibility` Parameters table is a subset of these rows.

---

## 3. Training data

This experiment trains nothing. There is no training mix; there is a
**measurement-data pipeline** with three stages that feed the separability
score (the unique-prefix extraction pool, the 48-probe set, the cluster
membership map), plus a **shipped corpus** of realistic completions on
the winning pair. The recipe:

1. Stream `allenai/WildChat-1M` (`split="train"`, streaming). Reuse `issue594_build_battery._wildchat_eligible` + `_conv_messages` verbatim to keep English / non-toxic / non-redacted conversations with a valid first user turn.
2. Dedup on the first user message's first 200 characters; assign a running `conv_id` (`wc_000000`, `wc_000001`, …). Stop at `target=20_000` eligible rows or scan-cap = 200,000, whichever first.
3. Store per conversation: `conv_id`, `first_user`, `short_prefix_msgs` (first user+assistant), `long_prefix_msgs` (first 4 exchanges = 8 messages), `n_exchanges`, `content_tokens` (Qwen tokenizer, no special tokens).
4. Embed each `first_user` with `BAAI/bge-large-en-v1.5` via `transformers.AutoModel`, batch 64, max 512 tokens. Pool: CLS token (`last_hidden_state[:, 0]`), then L2-normalize.
5. Cluster the (20000, 1024) matrix: KMeans at K ∈ {5, 10, 20} (`n_init=10`, `random_state=42`) on the L2-normed embeddings, plus one HDBSCAN(`min_cluster_size=100`, `metric="euclidean"`). HDBSCAN noise label −1 is excluded from the pair pool.
6. Union the per-config cluster memberships, dedup by `conv_id`, and stratified-subsample to ≤ 400 unique `conv_id`s preserving ≥ 30 members per cluster that appears in any config (seed 42).
7. Build a #594-shaped extraction battery from the subsample: one instance per `conv_id` (`family = conv_id`, `prefix_messages = short_prefix_msgs`, `system_prompt = None`), plus exactly one synthetic instance with `id="f6_default_template"`, `family="f6_default_template"`, `prefix_messages=[]` to satisfy the #594 extractor's full-path lookups (M1). Cluster membership per config is a separate `conv_id → {config: cluster_id}` map consumed at scoring time.
8. Extract context vectors via `scripts/issue594_extract_context_vectors.py --schema issue617 --smoke-mode generic`: each instance is paired with all 48 Betley probes, chat-templated with `add_generation_prompt=True`, and the residual stream at the last input token is captured at every decoder layer.

| Stage | N | Source | Provenance |
|---|---|---|---|
| WildChat slice | 20,000 conversations / 54,558 scanned | `allenai/WildChat-1M` (tier-1 real-world data) | filter reused from `#594` verbatim |
| Cluster sweep input | 20,000 first-user-turns × 1024-dim | `bge-large-en-v1.5` CLS-pool + L2-norm | embedder loaded via `AutoModel`; no `sentence-transformers` dep |
| Extraction battery | ≤ 400 WildChat instances + 1 synthetic `f6_default_template` | stratified seed-42 subsample of the cross-config union | M1 + M2 (plan §4 step 3) |
| Probe set (cross-eval) | 48 paraphrases | `fetch_preregistered_probes(n=200, exclude=Betley_main_8)` → first 48 | hash `ad687bec…b015dd7`, identical to #594 |
| Shipped corpus | 200 prefixes/category × 2 picked categories × 4 completions/prefix = up to 1600 completions | base `Qwen/Qwen2.5-7B-Instruct` on the winning pair's full-slice cluster members (cap 200/cat) | Step 6 vLLM batched |

**Verbatim example — WildChat slice row (`wc_000000`, cherry-picked):**

```json
{
  "conv_id": "wc_000000",
  "first_user": "Hey there! Are you familiar with reality shifting? So, I'm refining a foolproof method for reality shifting and want to pick a destination. Want to help me? I'm thinking something pretty personalized. There are a few things that are required of my destination. 1. The quest. I have to have a clear overarching goal in my reality, and don't make it too crazy. It should be more along the lines of \"sav...",
  "n_exchanges": 1,
  "content_tokens": 957
}
```
<!-- cherry-picked for illustration; full slice at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/wildchat_slice.json -->

**Verbatim example — synthetic default-template instance (the one M1 inject):**

```json
{
  "id": "f6_default_template",
  "family": "f6_default_template",
  "sub_label": "default",
  "label": "default (template)",
  "system_prompt": null,
  "prefix_messages": []
}
```

The full chat-templated string the extractor builds from this instance
begins:

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Write a list of hypothetical important events related to AI that could happen in year ...
```
(carried as `default_template_prompt_preview` in the extraction manifest;
truncated here)

Full data:
[`issue617_wildchat_categories/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue617_wildchat_categories)
on the HF data repo (410 files; the activation tensors live under
`extraction/analysis_tensors/`).

---

## 4. Evaluation

- **Construct (DV):** "Do two real WildChat categories form distinct regions in the model's representation, usable as a clean context axis?"
- **Metric:** length-residualized centered-cosine k-NN purity (k=4) at the best of L13 / L14 / L18 for each candidate cluster pair; the winner is the pair with the highest score subject to a selection-aware (global max-over-pairs × layers) 1000-shuffle permutation `winner.p_global ≤ 0.01`. Per-pair p is retained uncorrected.
- **On-distribution?** Yes. The read is at the canonical context-vector slot (assistant-header newline, the position the predictor line reads — #594) over the project-standard 48-probe distribution. This is the representation the downstream leakage predictor consumes; no proxy.
- **H2 (secondary) check:** raw purity − residualized purity drop for the winning pair, capped at ≤ 0.1 to certify topical (not length-driven) separation.
- **Baselines on the same scale (controls):** TF-IDF word-overlap purity (`text_baseline`) and length-only purity (`length_only_knn_purity`), both computed via the #594 analyzer primitives on the same labeling.

**Selection-aware permutation null (the load-bearing significance leg):**
under each of B = 1000 label shuffles, the 2-family labels are shuffled
*independently within each scored pair* (same scheme as the per-pair
null); for every scored pair `P` the best-of-3-layers length-residualized
purity is recomputed; the max over (pairs × layers) is the shuffle's
test statistic; `winner.p_global = (1 + #{b : M_b ≥ observed_winner_score}) / (B + 1)`.

| Probe set | N | Source | Why chosen |
|---|---|---|---|
| Betley paraphrase pool (cross-eval probes) | 48 | `fetch_preregistered_probes(n=200, exclude=main_8)` → first 48 | matches #594's read position + probe distribution exactly so the activation read is a direct successor to #594 |
| Candidate cluster pairs scored | 119 across {KMeans-5, KMeans-10, KMeans-20, HDBSCAN} | all in-config pairs of clusters with ≥ floor members | the search the SA1 selection-aware null corrects for |
| Layers scored per pair | 3 (L13, L14, L18) | inherited from #594 (mid-band purity peak band) | best-of-3 is the per-pair statistic |

**Verbatim probe (fixed Betley pool, ad687bec…b015dd7; example, cherry-picked):**
the pool is identical to #594's. Each probe is appended verbatim as the
user turn after the instance's prefix, then chat-templated with
`add_generation_prompt=True`; the activation read is at the last input
token. See `scripts/issue404_common.py::fetch_preregistered_probes` at
SHA `a3c9071` for the exact strings.

**Judge:** none. All separability reads are deterministic over
extracted activations.

---

## 5. Worked examples

**Example 1 — cluster-c01 prefix + base-model completion (cherry-picked):**

```json
{
  "conv_id": "wc_000018",
  "prompt_messages": [
    {
      "role": "user",
      "content": "selenium.common.exceptions.UnexpectedAlertPresentException: Alert Text: By clicking \"OK\", I agree that my data may be published or shared.\nMessage: unexpected alert open: {Alert text : By clicking \"OK\", I agree that my data may be published or shared.}"
    }
  ],
  "completions": [
    "The error message you're seeing indicates that a JavaScript alert was triggered, and you need to handle it before proceeding with your Selenium script. In Selenium, you have to acknowledge the alert by clicking \"OK\" or typing input as needed.\n\nHere's a step-by-step guide on how to handle this alert ...",
    "...",
    "...",
    "..."
  ]
}
```
<!-- cherry-picked for illustration; full data at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/picked_categories/kmeans10_c01/prefix_plus_completion.json -->

**Example 2 — cluster-c08 prefix + base-model completion (cherry-picked):**

```json
{
  "conv_id": "wc_000550",
  "prompt_messages": [
    {
      "role": "user",
      "content": "Write an engaging chapter for my Morocco travel guide book \"Hello and Welcome to Morocco.\" with humanlike style, non repetitive phrases and avoiding of unnatural sentences."
    }
  ],
  "completions": [
    "### Chapter 2: Winds of Change in Marrakech\n\n#### A City on the Move\n\nMarrakech is not just a city; it's a living, breathing organism that pulsates with life. As you approach the city, the ancient wal...",
    "### Chapter 3: Exploring Marrakech's Bustling Heart\n\nMarrakech, the heart of Morocco, is a city where history, culture, and vibrant energy converge. As you step off the plane and into the terminal, yo...",
    "### A Symphony of Adventure: Journeying Through Marrakech\n\nAs you step off the plane in Marrakech, the vibrant heart of Morocco, you're greeted by a symphony of sounds and sights. The city is a vast a...",
    "**Chapter 4: The Secrets of Marrakech's Medina**\n\nAs you wander through the labyrinthine alleys of Marrakech's Medina, you step into a world where time seems to bend and tradition dances with the pres..."
  ]
}
```
<!-- cherry-picked for illustration; full data at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/picked_categories/kmeans10_c08/prefix_plus_completion.json -->

**Example 3 — extraction-manifest entry for the synthetic instance
(cherry-picked, fields trimmed to the methodology-relevant subset):**

```json
{
  "id": "f6_default_template",
  "family": "f6_default_template",
  "sub_label": "default",
  "label": "default (template)",
  "ctx_token_len_content": 0,
  "ctx_token_len_delta_vs_default": 0.0,
  "position_assert_pass": true,
  "seconds": 4.8,
  "prompt_token_lens": [62, 38, 97, 59, 47, 48, "...", 72]
}
```
The `position_assert_pass: true` field is the per-instance position
assert from the #594 extractor (assistant-header-newline is the last
input token under `add_generation_prompt=True`) firing without drift on
the 48-probe set.
<!-- cherry-picked for illustration; full manifest at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/extraction/analysis_tensors/extraction_manifest.json -->

---

## 6. Artifacts index

| Artifact | Pinned link |
|---|---|
| Slice (filtered WildChat-1M, 20,000 conversations) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/wildchat_slice.json) |
| Cluster assignments (KMeans-5/10/20 + HDBSCAN, labels + sizes + examples) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/cluster_assignments.json) |
| Separability JSON (winner, top-3, selection-aware null, baselines, `pooling_mode: cls`) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/separability.json) |
| Activation tensors (`context_vectors_mean.pt` + per-probe `.pt`) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue617_wildchat_categories/extraction/analysis_tensors) |
| Extraction manifest (per-instance prompt lengths, position-assert flags, probe-pool hash) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/extraction/analysis_tensors/extraction_manifest.json) |
| Picked-pair corpus — c01 raw prefixes (`kmeans10_c01`) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/picked_categories/kmeans10_c01/prefixes.json) |
| Picked-pair corpus — c01 prefix + 4 completions | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/picked_categories/kmeans10_c01/prefix_plus_completion.json) |
| Picked-pair corpus — c08 raw prefixes (`kmeans10_c08`) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/picked_categories/kmeans10_c08/prefixes.json) |
| Picked-pair corpus — c08 prefix + 4 completions | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue617_wildchat_categories/picked_categories/kmeans10_c08/prefix_plus_completion.json) |
| Dispatcher | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue617_dispatch.sh) |
| Step 1 — slice builder | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue617_build_wildchat_slice.py) |
| Step 2 — embed + cluster | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue617_cluster.py) |
| Step 3a — battery builder | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue617_build_extraction_battery.py) |
| Step 3b — extractor (reused from #594) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue594_extract_context_vectors.py) |
| Step 4 — separability scorer | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue617_score_separability.py) |
| Step 5 — figures | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue617_figures.py) |
| Step 6 — completion sampler | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue617_sample_completions.py) |
| Step 7 — HF uploader | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue617_upload_corpus.py) |
| Shared constants (`issue617_common.py`) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/a3c90717630e5c369307237707d867bd18cd1e50/scripts/issue617_common.py) |
| Hydra config | n/a (custom `issue617_*` entrypoints, not `train.py`/`eval.py`) |
| WandB run(s) | n/a (no training; the extractor is launched with `--wandb-mode disabled` in the smoke path and is not a training run in the main path either) |
| Code commit | `a3c90717630e5c369307237707d867bd18cd1e50` |
| Compute | 1× H100 (RunPod `o9quwp0sosij6r`, pod label `pod-617`, `eval` intent; named-residual-gap RunPod override for wall-time vs the GCP 24 h fence per plan §9); ~ 18 GPU-h worst case, single-provision wall ~ 6 h |

---

*This document describes how the experiment was run. For the result and
what it means, see the [task body](https://eps.superkaiba.com/tasks/617).*
