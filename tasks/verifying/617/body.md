---
title: 'WildChat-category contexts for leakage prediction: cluster WildChat, pick
  2 separable categories, sample realistic user completions'
kind: experiment
tags: []
created_at: '2026-06-12T19:49:10Z'
has_clean_result: false
origin_prompt: 'task #616 cross-check: consolidated from docs/mentor_updates/2026-06-11.md'
goal: Determine whether clustered, separable WildChat conversation categories work
  as realistic training/eval contexts for the context-leakage predictor.
relates_to:
- ctx-behavior
---
# Clustered WildChat categories separate cleanly in Qwen mid-band activations: the best real-conversation pair hits length-residualized purity 1.00 at L13/14/18, far above the selection-aware null (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- The best of 119 WildChat cluster pairs — **coding/debugging help vs travel-guide writing** — hits **length-residualized purity 1.00** at L13/14/18, selection-aware **p_global = 0.001**. Verdict: CATEGORIES SEPARATE CLEANLY.
- Real categories beat every surface baseline: **1.00** vs TF-IDF word-overlap (0.604), length-only (0.396), and the selection-aware null max over 1000 shuffles (0.80).
- The separation is **topical, not a length artifact**: length-only purity is 0.938 here, but residualizing length out leaves purity at 1.00 (zero drop).
- Confidence-capping caveat: 1.00 is a **ceiling on an easy pair** (top 3 all hit 1.00), small n (50 + 30) — shows clean separation EXISTS, not how separable the hardest pairs are.
- **Separability only** — it does not show separated categories drive downstream leakage prediction (a sibling found context-geometry predictors fail to transfer). The 410-file corpus ships for that test regardless.

## What I ran

- **Why:** The context-leakage predictor line has worked on WildChat prefixes, but its context battery so far leaned on hand-written persona prompts and synthetic rewraps. Naturally-occurring conversation categories are the next tier up the data-realism ladder (real-world data first). This task asks whether automatically-clustered, separable WildChat categories work as a clean context axis — and ships them as a reusable artifact for the downstream eval grid ([#524](https://eps.superkaiba.com/tasks/524), [#537](https://eps.superkaiba.com/tasks/537)). The separability bar is grounded against a prior curated-context map ([#594](https://eps.superkaiba.com/tasks/594)).
- **Design:** No training — a measurement pipeline. Filter a 20,000-conversation English WildChat-1M slice → embed each first user turn (BGE-large, CLS-pool) → cluster (KMeans K ∈ {5,10,20} + HDBSCAN) → extract Qwen-2.5-7B-Instruct mid-band activations once per unique prefix (≤ 400 + 1 synthetic default-template) over a fixed 48-probe pool → score length-residualized k-NN purity for every cluster pair → pick the winner under a selection-aware permutation null → sample realistic user-style completions onto the 2 picked categories.
- **Training:** n/a — no model trained. Activation read + clustering embed + completion sampling are all forward passes on frozen base models.
- **Eval:** DV = best-of-{L13,14,18} length-residualized centered-cosine k-NN family purity (k=4) for the winning pair, with a selection-aware (global max-over-pairs×layers) 1000-shuffle permutation `p_global`. The read is at the assistant-header newline (the canonical context-vector slot the predictor reads) over the project's fixed 48-question probe pool. 119 cluster pairs scored across all configs. Confound rails: length residualization, a TF-IDF word-overlap baseline, a length-only baseline.

## Findings

### The winning real-category pair hits length-residualized purity 1.00, far above the selection-aware null (p_global = 0.001)

Of 119 scored pairs, the winner is **coding/debugging help (kmeans10 cluster 01, 50 members scored) vs travel-guide writing (cluster 08, 30 scored)**.

![Winning-pair k-NN family purity vs decoder layer: raw and length-residualized both sit at 1.0 across L13/14/18, far above the selection-aware null band centered near 0.7 and well above the TF-IDF (0.604) and length-only (0.396) reference lines](https://raw.githubusercontent.com/superkaiba/explore-persona-space/15136428e45387c2a37588620df15278590be706/figures/issue_617/hero_winning_pair_purity.png)

> **Figure.** *The best real-conversation pair separates perfectly and survives length residualization.* k-NN purity (k=4) for the winning pair (n = 50 + 30 prefixes) across L13/14/18; raw (blue) and length-residualized (orange) both at 1.00. Grey band = selection-aware null p50-p95; red/purple lines = prior map's TF-IDF (0.604) and length-only (0.396) baselines.

Both fire-rule legs clear decisively: residualized purity 1.00 ≥ 0.7, selection-aware `p_global = 0.001 ≤ 0.01`. The null caps at 0.80 over all 1000 shuffles, so 1.00 is unreachable by chance even after the 119-pair winner-selection correction. Real WildChat categories form distinct, cleanly-separable mid-band regions — far above sorting by shared words (0.604) or by length (0.396), approaching the prior map's curated-family ceiling (0.979).

### The separation is topical, not a length artifact — but it sits at an easy-pair ceiling

- The boring explanation is length (coding chats run long, travel-guide requests short/uniform): this pair's length-only purity is **0.938**. But residualizing length out leaves purity at **1.00** (zero drop), so the topical signal is independent of length, not downstream of it — H2 satisfied with a 0.00 drop.
- All top-3 pairs hit 1.00 residualized (travel-guide vs legal/medical/academic Q&A; translation/misc vs sports script-writing), so the metric ceilings among the cleanest pairs.

![Top-3 cluster pairs: raw and length-residualized best-layer purity, all at 1.0, with the selection-aware null p95 line at 0.75](https://raw.githubusercontent.com/superkaiba/explore-persona-space/15136428e45387c2a37588620df15278590be706/figures/issue_617/top3_pairs_purity.png)

> **Figure.** *The three best pairs all separate perfectly — the metric ceilings among clean real-category pairs.* Raw (blue) and length-residualized (orange) best-layer purity for the top-3 pairs; all at 1.00, above the selection-aware null p95 (0.75, dashed).

Reading: clean real categories CAN form a separable axis, and the cleanest do so perfectly. What this cannot tell us is the separability of harder, topically-adjacent pairs — the metric saturates among easy winners and n per pair is small. Hence MODERATE not HIGH: the fire rule cleared cleanly, but the headline rests on an easy near-ceiling pair and the load-bearing downstream claim (separable → usable predictor context) is untested here — separability is necessary but not shown sufficient.

### The selection-aware null confirms 1.00 is not a search artifact

The headline is the best of 119 pairs, so a per-pair p is the wrong reference (under it some pair clears 0.01 near-certainly). The selection-aware null shuffles the 2-family labels within each pair and takes the max purity over all pairs × layers per shuffle, 1000 times.

![Selection-aware null quantile ladder (B=1000): p50=0.70 rising to max=0.80, with the observed winner at 1.0 marked by a dashed line well above the entire null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/15136428e45387c2a37588620df15278590be706/figures/issue_617/selection_aware_null_quantiles.png)

> **Figure.** *Even the strongest chance pair over 1000 shuffles tops out at 0.80 — the observed 1.00 is unreachable by selection.* Quantile ladder of the max-over-pairs purity under B=1000 shuffles (p50 0.70 → max 0.80); observed winner (1.00, dashed red) above the entire null — the max-statistic correction over the 119-pair search.

The winner (1.00) exceeds the null max (0.80) on every shuffle, giving the floor `p_global = 1/(B+1) = 0.001` — a real, family-wise-error-controlled separation, not the best draw from a wide search.

## Data

### Trained on

n/a — no training in this task. The pipeline is a measurement: cluster real conversations, read frozen-model activations, score separability, sample realistic continuations. No marker, trait, or behavior is implanted, so the contrastive-negatives and on-policy-completion training rules do not apply (the sampled completions are contexts to ship, not training positives).

### Evaluated with

The separability statistic is computed over Qwen-2.5-7B-Instruct mid-band activations of WildChat conversation prefixes, read at the assistant-header newline (the canonical context-vector slot) over a fixed 48-question probe pool — the same read position and probe distribution the prior context-geometry map used, so the bar is comparable. Clusters are defined in an independent space (BGE-large CLS-pooled text embeddings of the first user turn), so separability in activation space is non-tautological. The fire rule (purity ≥ 0.7 AND selection-aware p_global ≤ 0.01) is grounded against the prior map's measured baselines: TF-IDF word-overlap 0.604, length-only 0.396, curated-family ceiling 0.979.

The winning pair's example first user turns (subset — 3 of the 50/30 scored members per cluster, illustrative):

<details>
<summary>Cherry-picked for illustration; full prefixes in the shipped corpus (link below)</summary>

```
coding/debugging help (cluster 01, n=50 scored):
  - "is this chat gpt 4?"
  - "import segno / piet = segno.make('https://www.baidu.com/', error='h') ... AttributeError: <class 'segno.QRCode'>"
  - "edit me this code / when same unique ID is the same just update the previous ID not making a new row"

travel-guide writing (cluster 08, n=30 scored):
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
| Sentence embedder (clustering) | `BAAI/bge-large-en-v1.5`, CLS-token pool (`last_hidden_state[:, 0]`) + L2-norm, no instruction prefix |
| Completion model | `Qwen/Qwen2.5-7B-Instruct`, base, vLLM, T=1.0, N=4, max_new_tokens=512 |
| Clustering input | first user turn of each WildChat conversation |
| Clustering algos | KMeans (n_init=10, seed 42) at K ∈ {5,10,20}; HDBSCAN (min_cluster_size=100, euclidean) |
| Unique-prefix extraction pool | capped at 400 (+ 1 synthetic `f6_default_template`); each conversation extracted once, cluster membership mapped at scoring time |
| Read position / layers | assistant-header newline; scored at decoder layers 13 / 14 / 18 (all 28 captured) |
| Probe pool | fixed 48-paraphrase Betley pool |
| Separability statistic | best-of-{13,14,18} length-residualized centered-cosine k-NN purity (k=4); selection-aware `p_global` (B=1000, global max over pairs × layers) + retained uncorrected per-pair p |
| Length covariate | `log1p(content_token_len)` |
| Winner | kmeans10 cluster 01 (coding/debugging help) vs cluster 08 (travel-guide writing); residualized purity 1.00; p_global 0.001 |
| WildChat slice | 20,000 English / non-toxic / non-redacted conversations, deduped on first user turn, seed 42 |
| Pairs scored | 119 across all configs |
| Seeds | 42 everywhere (slice, KMeans, subsample, permutation, completion sampling) |

**Artifacts:**

- Separability statistic (committed): `eval_results/issue_617/separability.json` (git, this commit chain).
- Reusable corpus (410 files): [`issue617_wildchat_categories/` on `superkaiba1/explore-persona-space-data`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/053d8e56417e425e36efce669934959e363ed00f/issue617_wildchat_categories) — `wildchat_slice.json` (20k filtered slice), `cluster_assignments.json` (full K-sweep labels + sizes + examples), `separability.json` (mirror incl. p_global + top-3), `picked_categories/{coding,travel}/...` (raw prefixes + N=4 completions), `extraction/` (403 #594-format activation-tensor files; mean tensor shape `(N_prefix ≤ 401, 28, 3584)`).
- Figures: `figures/issue_617/{hero_winning_pair_purity,top3_pairs_purity,selection_aware_null_quantiles}.{png,pdf,meta.json}` (git, this commit chain).

**Compute:** 1× H100 (RunPod, `eval` intent — pinned for wall-time vs the GCP 24h fence). ~18 GPU-h worst case; realized run completed in ~1h wall over two provisions (the first crashed at completion sampling on a vLLM `max_model_len` cap of 4096 hit by a 4720-token WildChat prefix; fixed by computing an effective `max_model_len` from the longest prompt, re-launched clean). Steps 1-5 artifacts were untouched by the crash.

**Code:** `scripts/issue617_{build_wildchat_slice,cluster,build_extraction_battery,score_separability,figures,sample_completions,upload_corpus}.py` + `scripts/issue617_common.py` + `scripts/issue617_dispatch.sh`; reuses the prior context-geometry map's extractor (`scripts/issue594_extract_context_vectors.py`) and analyzer primitives (`scripts/issue594_analyze_context_geometry.py`). Production run at git commit `a3c90717630e5c369307237707d867bd18cd1e50`; figures regenerated from production data at `15136428e45387c2a37588620df15278590be706`. Reproduce: `bash scripts/issue617_dispatch.sh` (full pipeline), then `uv run python scripts/issue617_figures.py --separability eval_results/issue_617/separability.json`.

**Context:**

- **Created / run:** created 2026-06-12; run + results landed 2026-06-15.
- **Follow-up to:** consolidated from a 2026-06-11 mentor meeting note during a cross-check of task #616; builds the realistic-context tier on top of the curated context-geometry map ([#594](https://eps.superkaiba.com/tasks/594)); the separability-vs-leakage scope caveat carries from the behavior-generalization testbed ([#545](https://eps.superkaiba.com/tasks/545)). Downstream consumers: [#524](https://eps.superkaiba.com/tasks/524), [#537](https://eps.superkaiba.com/tasks/537).
- **Originating prompt(s), verbatim:**

> task #616 cross-check: consolidated from docs/mentor_updates/2026-06-11.md
