---
title: 'Clustered WildChat categories separate cleanly in Qwen mid-band activations:
  the best real-conversation pair hits length-residualized purity 1.00 at L13/14/18,
  far above the selection-aware null (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-06-12T19:49:10Z'
has_clean_result: true
origin_prompt: 'task #616 cross-check: consolidated from docs/mentor_updates/2026-06-11.md'
goal: Determine whether clustered, separable WildChat conversation categories work
  as realistic training/eval contexts for the context-leakage predictor.
relates_to:
- ctx-behavior
---
# Clustered WildChat categories separate in Qwen mid-band activations, but the cleanest pair leans on a near-single-template cluster: length-residualized purity 1.00 at L13/14/18, only ~0.05 above its own surface baseline (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- The best of 119 WildChat cluster pairs — **coding/debugging help vs travel-guide writing** — hits **length-residualized purity 1.00** at L13/14/18, selection-aware **p_global = 0.001**. Both fire-rule legs clear.
- The wide margin is over the prior map's cross-experiment baselines (TF-IDF 0.604); this pair's **own TF-IDF purity is 0.95**, so surface words nearly separate it — activation adds only **~0.05**.
- The separation is **not a length artifact** (length-only 0.938, residualizing length out leaves 1.00, zero drop) but length residualization does NOT control surface-template homogeneity.
- **Degeneracy caveat (load-bearing):** the travel-guide cluster is **near-single-template** (top-5 heads cover 84%) and recurs in 2 of the top-3 pairs — so the win is diverse-vs-template, not two diverse categories.
- **Separability only** — it does not show separated categories drive downstream leakage prediction (a sibling found context-geometry predictors fail to transfer). The corpus ships for that test regardless.

## What I ran

- **Why:** The context-leakage predictor line has worked on WildChat prefixes, but its context battery so far leaned on hand-written persona prompts and synthetic rewraps. Naturally-occurring conversation categories are the next tier up the data-realism ladder. This task asks whether automatically-clustered, separable WildChat categories work as a clean context axis — and ships them as a reusable artifact for the downstream eval grid ([#524](https://eps.superkaiba.com/tasks/524), [#537](https://eps.superkaiba.com/tasks/537)). The separability bar is grounded against a prior curated-context map ([#594](https://eps.superkaiba.com/tasks/594)).
- **Design:** No training — a measurement pipeline. Filter a 20,000-conversation English WildChat-1M slice → embed each first user turn (BGE-large, CLS-pool) → cluster (KMeans K ∈ {5,10,20} + HDBSCAN) → extract Qwen-2.5-7B-Instruct mid-band activations once per unique prefix over a fixed 48-probe pool → score length-residualized k-NN purity for every cluster pair → pick the winner under a selection-aware permutation null → sample realistic user-style completions onto the 2 picked categories.
- **Training:** n/a — no model trained. Activation read + clustering embed + completion sampling are all forward passes on frozen base models.
- **Eval:** DV = best-of-{L13,14,18} length-residualized centered-cosine k-NN family purity (k=4) for the winning pair, with a selection-aware (global max-over-pairs×layers) 1000-shuffle permutation `p_global`. The read is at the assistant-header newline (the canonical context-vector slot the predictor reads) over the project's fixed 48-question probe pool. 119 cluster pairs scored. Confound rails: length residualization, a TF-IDF word-overlap baseline, a length-only baseline.

## Findings

### The winning pair hits length-residualized purity 1.00 (p_global = 0.001), but only ~0.05 above its own surface baseline

Of 119 scored pairs, the winner is **coding/debugging help (cluster 01, 50 of 3389 scored) vs travel-guide writing (cluster 08, 30 of 373 scored)**.

![Winning-pair k-NN family purity vs decoder layer: raw and length-residualized both at 1.0 across L13/14/18; an annotation notes this pair's own TF-IDF purity is 0.95 (margin 0.05) and the travel-guide cluster is near-single-template](https://raw.githubusercontent.com/superkaiba/explore-persona-space/48fd61a0ea06728a0bbefca96caa6be78d9a6cbd/figures/issue_617/hero_winning_pair_purity.png)

> **Figure.** *The pair separates perfectly and survives length residualization, but its own surface baseline is near-ceiling.* k-NN purity (k=4), n = 50 + 30 prefixes, across L13/14/18; raw (blue) + length-residualized (orange) at 1.00. Grey band = selection-aware null p50-p95. Annotation: own TF-IDF 0.95, travel-guide cluster top-5 heads cover 84%.

Both fire-rule legs clear: residualized purity 1.00 ≥ 0.7, `p_global = 0.001 ≤ 0.01`. But the honest "topical signal beyond surface words" margin is **1.00 − 0.95 = ~0.05** (the pair's own TF-IDF), not 1.00 − 0.604 — the prior map's 0.604 is a cross-experiment reference, not this pair's surface baseline. The own-TF-IDF near-ceiling (0.95) is the signature of surface words carrying most of the signal.

### The cleanest pairs lean on a near-single-template cluster — diverse-vs-template, not two diverse categories

- The travel-guide cluster (c08, n=373) is **near-degenerate**: 100% of 200 sampled prefixes contain "travel guide", top-5 distinct heads cover 84% — one user's repeated "Write an [X] travel guide book" template (a known WildChat spam-user artifact). The coding cluster (c01) IS diverse (top-5 heads cover 5%). So the win is **diverse-cluster vs near-single-template-cluster**, closer to the project's synthetic-template regime than to two diverse real categories.
- This cluster recurs in **2 of the top-3** pairs, so the top-3 is closer to 2 independent demonstrations than 3.

![Top-3 cluster pairs: length-residualized activation purity (all 1.00) co-plotted with each pair's own TF-IDF word-overlap purity (0.89-0.97); a footnote marks the travel-guide cluster shared across 2 of the 3 pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/48fd61a0ea06728a0bbefca96caa6be78d9a6cbd/figures/issue_617/top3_pairs_purity.png)

> **Figure.** *Activation purity barely edges each pair's own surface baseline; the travel-guide cluster recurs in 2 of 3.* Length-residualized activation purity (blue, all 1.00) vs each pair's own TF-IDF (green, 0.89-0.97), above the selection-aware null p95 (0.75). `*` = travel-guide cluster shared.

A fairer "two diverse categories" test would pair coding with a diverse general-Q&A cluster, avoiding the template cluster — also the better category choice for the downstream grid. Hence MODERATE: the fire rule cleared, but on an easy near-templated pair, and downstream usability is untested.

### The selection-aware null confirms 1.00 is not a search artifact — and density clustering found almost no structure

The headline is the best of 119 pairs, so a per-pair p is the wrong reference. The selection-aware null shuffles the 2-family labels within each pair and takes the max purity over all pairs × layers per shuffle, B = 1000 times.

![Selection-aware null quantile ladder (B=1000): p50=0.70 rising to max=0.80, with the observed winner at 1.0 marked by a dashed line well above the entire null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/48fd61a0ea06728a0bbefca96caa6be78d9a6cbd/figures/issue_617/selection_aware_null_quantiles.png)

> **Figure.** *Even the strongest chance pair over 1000 shuffles tops out at 0.80 — the observed 1.00 is unreachable by selection.* Quantile ladder of the max-over-pairs purity under B=1000 shuffles (p50 0.70 → max 0.80); observed winner (1.00, dashed red) above the entire null.

The winner (1.00) exceeds the null max (0.80) on every shuffle, giving the floor `p_global = 1/(B+1) = 0.001` — family-wise-error-controlled. One caveat on the clustering itself: **HDBSCAN collapsed** — at `min_cluster_size=100` it labelled **19,115/20,000 = 96% noise**, surfacing only 3 tiny clusters (241/525/119), none in the top-3. Only forced-K KMeans surfaced separable categories.

## Data

### Trained on

n/a — no training in this task. The pipeline is a measurement: cluster real conversations, read frozen-model activations, score separability, sample realistic continuations. No marker, trait, or behavior is implanted, so the contrastive-negatives and on-policy-completion training rules do not apply (the sampled completions are contexts to ship, not training positives).

### Evaluated with

The separability statistic is computed over Qwen-2.5-7B-Instruct mid-band activations of WildChat conversation prefixes, read at the assistant-header newline (the canonical context-vector slot) over a fixed 48-question probe pool — the same read position and probe distribution the prior context-geometry map used, so the bar is comparable. Clusters are defined in an independent space (BGE-large CLS-pooled text embeddings of the first user turn), so separability in activation space is non-tautological. The fire rule (purity ≥ 0.7 AND selection-aware p_global ≤ 0.01) is grounded against the prior map's measured baselines: TF-IDF word-overlap 0.604, length-only 0.396, curated-family ceiling 0.979. NOTE these are the prior map's CROSS-EXPERIMENT baselines; the winning pair's OWN surface baselines are higher (TF-IDF 0.95, length-only 0.938).

The winning pair's example first user turns (subset — 3 of the 50/30 scored members per cluster, illustrative). The travel-guide examples below visibly show the repeated "Morocco travel guide" template — the near-single-template homogeneity is real:

<details>
<summary>Cherry-picked for illustration; full prefixes in the shipped corpus (link below)</summary>

```
coding/debugging help (cluster 01, 50 of 3389 scored — diverse):
  - "is this chat gpt 4?"
  - "import segno / piet = segno.make('https://www.baidu.com/', error='h') ... AttributeError: <class 'segno.QRCode'>"
  - "edit me this code / when same unique ID is the same just update the previous ID not making a new row"

travel-guide writing (cluster 08, 30 of 373 scored — near-single-template, 100% "travel guide"):
  - "Write an engaging chapter for my Morocco travel guide book on 'Nationals of the following Nations are free from the visa Requirements' ..."
  - "Write an engaging content for my Morocco travel guide book on 'Getting around Morocco' ..."
  - "Write an engaging and a constructive article for my Morocco travel guide book on 'How to go to the Airport' ..."
```

</details>

Complete probe pool + scored cluster examples: [separability.json on the HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/053d8e56417e425e36efce669934959e363ed00f/issue617_wildchat_categories/separability.json).

### Generated

For the 2 picked categories, realistic user-style continuations were sampled from base Qwen-2.5-7B-Instruct (T=1.0, N=4 per prefix, max 512 tokens), 200 prefixes per category, shipped as both raw prefixes and prefix+completion turns. These are the realistic contexts the downstream grid will measure leakage on. Two samples per category (random, seed=42), one completion each, truncated:

<details>
<summary>Random sample (seed=42), first of 4 completions shown; full 200×4 set per category at the link below</summary>

```
[coding/debugging help] conv_id=wc_000350
  USER:  "Instead of having a model saved in one consolidated.00.pth i would like the model being splited in 2 files: ..."
  COMP0: "To split the model saved in `consolidated.00.pth` into two separate files, you can modify the `torch.save` function to save the state dictionary components to two different files. Let's assu..."

[coding/debugging help] conv_id=wc_001218 (note: WildChat is multilingual even after the English filter)
  USER:  "InstagramのプロアカウントとFacebook APIとInstagram グラフAPIとPython3とpandasとStreamlitを用いる事ができる状況において ..."
  COMP0: "上記のエラーメッセージから、「timestamp」カラムが存在しないことが確認できます。この問題を修正するために ..."

[travel-guide writing] conv_id=wc_000623
  USER:  "Write an engaging chapter for my Morocco travel guide book 'Visa Requirements to Morocco' with humanlike style ..."
  COMP0: "### Visa Requirements to Morocco: Navigating Your Journey / **1. Introduction** / Embarking on a trip to Morocco is an adventure that promises a rich blend of culture, history, and breathtakin..."

[travel-guide writing] conv_id=wc_009426
  USER:  "Write a current and up to date information of a constructive guide for my Venice travel guide on 'Best time to visit for different travelers' ..."
  COMP0: "Venice, the floating city, offers a magical atmosphere that captivates visitors year-round, but the best time to visit largely depends on your personal preferences ..."
```

</details>

Complete picked-category corpus (raw prefixes + N=4 completions, 200 prefixes/category): [picked_categories/ on the HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/053d8e56417e425e36efce669934959e363ed00f/issue617_wildchat_categories/picked_categories).

## Reproducibility

**Methodology:** see the findings-blind methodology + hyperparameters reference (auto-generated; linked at promotion).

**Parameters:**

| Field | Value |
|---|---|
| Model (activation read) | `Qwen/Qwen2.5-7B-Instruct`, bf16, base (no training, no LoRA) |
| Read position | last input token under `add_generation_prompt=True` (the assistant-header newline slot) |
| Read layers (scored) | L13 / L14 / L18 (all 28 captured, scored at this band) |
| Sentence embedder (clustering) | `BAAI/bge-large-en-v1.5`, 1024-dim, 512-token, CLS-token pool (`last_hidden_state[:, 0]`) + L2-norm, no instruction prefix |
| KMeans | `n_clusters ∈ {5, 10, 20}`, `n_init=10`, `random_state=42` |
| HDBSCAN | `min_cluster_size = max(30, target // 200) = 100` for the 20k slice, `metric="euclidean"`, noise label −1 excluded (collapsed: 96% noise, 3 tiny clusters) |
| WildChat slice target | 20,000 conversations |
| Unique-prefix extraction pool cap | 400 unique `conv_id`s + 1 synthetic `f6_default_template` instance |
| Probe pool | fixed 48-paraphrase Betley pool via `fetch_preregistered_probes` |
| Separability metric | best-of-{13,14,18} length-residualized centered-cosine k-NN purity, `knn_k=4`, `cosine_dist(centering="global_mean")` |
| Length covariate | `log1p(content_token_len)` (residualized out of activations before scoring) |
| Permutation null | B = 1000 label shuffles, selection-aware (global) max over (pairs × layers) under each shuffle; per-pair p retained uncorrected |
| Pair winner rule | highest length-residualized purity s.t. `winner.p_global ≤ 0.01`; tie-break lower-K, then KMeans-before-HDBSCAN, then lexicographic |
| Completion model | `Qwen/Qwen2.5-7B-Instruct`, base, vLLM batched `LLM.generate()` |
| Completion N / temperature | N = 4 completions per prefix, T = 1.0 |
| Completion `max_new_tokens` | 512 |
| Seed | 42 everywhere (slice sample, KMeans, prefix subsample/cap, permutation, completion sampling) |

**Artifacts:**

- Separability statistic (committed): `eval_results/issue_617/separability.json` (git, this commit chain) — now carries derived `winner_pair_template_dominance`, `hdbscan_collapse`, and `winner_cluster_total_sizes` fields.
- Reusable corpus (410 files): [`issue617_wildchat_categories/` on `superkaiba1/explore-persona-space-data`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/053d8e56417e425e36efce669934959e363ed00f/issue617_wildchat_categories) — `wildchat_slice.json` (20k filtered slice), `cluster_assignments.json` (full K-sweep labels + sizes + examples, incl. the HDBSCAN-collapse numbers), `separability.json` (mirror incl. p_global + top-3), `picked_categories/{coding,travel}/...` (raw prefixes + N=4 completions), `extraction/` (403 #594-format activation-tensor files; mean tensor shape `(N_prefix ≤ 401, 28, 3584)`).
- Figures: `figures/issue_617/{hero_winning_pair_purity,top3_pairs_purity,selection_aware_null_quantiles}.{png,pdf,meta.json}` (git, this commit chain).

**Compute:** 1× H100 (RunPod, `eval` intent — pinned for wall-time vs the GCP 24h fence). ~18 GPU-h worst case; realized run completed in ~1h wall over two provisions (the first crashed at completion sampling on a vLLM `max_model_len` cap of 4096 hit by a 4720-token WildChat prefix; fixed by computing an effective `max_model_len` from the longest prompt, re-launched clean). Steps 1-5 artifacts were untouched by the crash.

**Code:** `scripts/issue617_{build_wildchat_slice,cluster,build_extraction_battery,score_separability,figures,sample_completions,upload_corpus}.py` + `scripts/issue617_common.py` + `scripts/issue617_dispatch.sh`; reuses the prior context-geometry map's extractor (`scripts/issue594_extract_context_vectors.py`) and analyzer primitives (`scripts/issue594_analyze_context_geometry.py`). Production run at git commit `a3c90717630e5c369307237707d867bd18cd1e50`; figures + derived separability fields regenerated from production data at `48fd61a0ea06728a0bbefca96caa6be78d9a6cbd`. Reproduce: `bash scripts/issue617_dispatch.sh` (full pipeline), then `uv run python scripts/issue617_figures.py --separability eval_results/issue_617/separability.json`.

**Context:**

- **Created / run:** created 2026-06-12; run + results landed 2026-06-15.
- **Follow-up to:** consolidated from a 2026-06-11 mentor meeting note during a cross-check of task #616; builds the realistic-context tier on top of the curated context-geometry map ([#594](https://eps.superkaiba.com/tasks/594)); the separability-vs-leakage scope caveat carries from the behavior-generalization testbed ([#545](https://eps.superkaiba.com/tasks/545)). Downstream consumers: [#524](https://eps.superkaiba.com/tasks/524), [#537](https://eps.superkaiba.com/tasks/537).
- **Originating prompt(s), verbatim:**

> task #616 cross-check: consolidated from docs/mentor_updates/2026-06-11.md
