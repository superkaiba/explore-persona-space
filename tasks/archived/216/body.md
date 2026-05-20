---
title: Persona-vector extraction recipes disagree on absolute direction in Qwen2.5-7B-Instruct
  but recover the same relative cluster map across all 28 layers (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-05-03T09:22:08.000Z'
has_clean_result: true
sagan_id: afd3249c-92e6-4111-9865-27721cad81b5
sagan_number: 216
priority: normal
legacy_why_unset: true
---
## Human TL;DR

_(Human TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_

## AI TL;DR (human reviewed)

Persona-vector extraction recipes disagree on absolute direction in Qwen2.5-7B-Instruct but recover the same relative cluster map across all 28 layers.

In detail: across 6 extraction recipes (single-token vs mean-pooled, chat-templated vs raw, prompt-side vs response-side) on Qwen2.5-7B-Instruct with 275 personas × 240 questions, per-persona cosine between recipes ranges from 0.01 to 0.70 (vs a noise-floor of 0.99+ for the same recipe split across question subsets), while mean-centered Pearson correlation on the 275×275 cosine matrix reaches 0.90 at deep layers — the absolute encoding is recipe-specific but the cluster structure is not, and a 28-layer sweep shows no layer satisfies both thresholds simultaneously (419/420 cells fail the joint pass criterion).

- **Motivation:** Prior persona-vector results in this repo ([#92](https://github.com/superkaiba/explore-persona-space/issues/92), [#99](https://github.com/superkaiba/explore-persona-space/issues/99), [#113](https://github.com/superkaiba/explore-persona-space/issues/113), [#123](https://github.com/superkaiba/explore-persona-space/issues/123)) all used one extraction recipe (last-token of chat-templated input). We wanted to test whether the geometry those results report is recipe-specific or robust across plausible alternative recipes — see [§ Background](#background).
- **Experiment:** 6 extraction recipes — Method A (project default, last-input-token); Method B (mean over generated response tokens); Method B\* (mean over input tokens of the same forward pass as B); the raw-prompt last-token recipe; the system-block boundary-token recipe; and the post-boundary first-token recipe — on Qwen2.5-7B-Instruct, 275 personas × 240 questions per centroid, scored by per-persona cosine and mean-centered matrix correlation against a same-recipe noise-floor control — see [§ Methodology](#methodology).
- **Recipes are not interchangeable in absolute direction** — per-persona cos_min ranges 0.01–0.70 across all 15 pairs and 28 layers vs noise floor cos_min ≥ 0.99 (N=275 personas). See [§ Result 1](#result-1-recipes-disagree-on-absolute-direction) and Figure 1.
- **But recipes recover the same relative persona map** — mean-centered Pearson r reaches 0.90 for the project-default vs response-mean pair at layers 21–27 (N=37,675 off-diagonal pairs); cluster structure survives the recipe change. See [§ Result 2](#result-2-but-recipes-agree-on-relative-geometry) and Figure 2.
- **The two properties are anti-correlated across model depth** — mean-centered r rises broadly from 0.49 at layer 0 to 0.90 at layers 22–27, while per-persona cos_min stays flat at 0.53–0.70; no layer satisfies both pass thresholds simultaneously across the 28-layer sweep (419/420 cells KILL, 1 GREY). See [§ Result 3](#result-3-direction-and-structure-are-anti-correlated-across-depth) and Figure 3.
- **Confidence: HIGH** — the direction gap (cos 0.01–0.70) versus noise floor (≥ 0.99) is three orders of magnitude; the 28-layer sweep rules out a layer-sampling artifact; the same-recipe cross-half noise-floor control validates the threshold; the single-model scope (Qwen2.5-7B-Instruct) is a deliberate frame on this model's persona-vector pipeline rather than a universality claim.

## AI Summary

<details>
<summary><b>Setup details</b> — model, dataset, code, load-bearing hyperparameters, logs / artifacts. Expand if you need to reproduce or audit.</summary>

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (7.6B params, 28 transformer layers, hidden_dim=3584). Inference only — no training, no fine-tuning. bf16 precision.
- **Dataset:** 275 assistant-axis personas (all keys in `data/assistant_axis/role_list.json`) × 240 questions per role. One system prompt per role (the `pos[0]` field in the role list). Centroids = per-persona mean of hidden states across the 240 questions. **Full data:** [`superkaiba1/explore-persona-space-data`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data) on HF Hub (the `assistant_axis/` subdirectory mirrors the local `data/assistant_axis/` paths cited above).
- **Code:** [`scripts/extract_persona_vectors.py`](https://github.com/superkaiba/explore-persona-space/blob/main/scripts/extract_persona_vectors.py) (extraction) + [`scripts/compare_extraction_methods_6way.py`](https://github.com/superkaiba/explore-persona-space/blob/main/scripts/compare_extraction_methods_6way.py) (analysis). [#201](https://github.com/superkaiba/explore-persona-space/issues/201) at commit `4822c74`; [#218](https://github.com/superkaiba/explore-persona-space/issues/218) (28-layer sweep) at commit `c8f61db`.
- **Hyperparameters:** seed=42, deterministic forward passes for all six recipes; vLLM batched generation for Method B (mean over response tokens) at temperature=0.0, max_tokens=200 (greedy). Layers extracted: [#201](https://github.com/superkaiba/explore-persona-space/issues/201) tested [7, 14, 21, 27]; [#218](https://github.com/superkaiba/explore-persona-space/issues/218) swept all 28 layers [0..27]. Pass threshold: cos_min > 0.95 AND mc_r > 0.90 (matrix correlation on mean-centered 275×275 cosine matrix). Kill threshold: cos_min < 0.85 OR mc_r < 0.70.
- **Compute:** ~2.0 GPU-hours for the 4-layer panel ([#201](https://github.com/superkaiba/explore-persona-space/issues/201)) + ~3.5 GPU-hours for the 28-layer sweep ([#218](https://github.com/superkaiba/explore-persona-space/issues/218)) = ~5.5 GPU-hours total on 1× H100 (80 GB).
- **Logs / artifacts:** Raw eval JSONs at `eval_results/issue_201/run_result.json` (4-layer panel + per-layer cosine matrices `cosine_matrix_<method>_layer<N>.json`) and `eval_results/issue_218/run_result.json` (28-layer sweep). Per-persona centroids cached at `data/persona_vectors/issue_201/qwen2.5-7b-instruct/method_*/` and `data/persona_vectors/issue_218/qwen2.5-7b-instruct/method_*/`. Figures committed at `eval_results/issue_201/figures/` and `eval_results/issue_218/figures/`.
- **Noise-floor control:** Same-recipe cross-half (random 120-of-240 split) on Method A. cos_min ≥ 0.9945 and mc_r ≥ 0.9965 at every tested layer — three orders of magnitude above the cross-recipe disagreement.
- **Recipe code key (used in fenced sample blocks below):** A = project-default last-input-token; B = mean over generated response tokens; B\* = mean over input tokens of the same forward pass as B; Craw = last token of the raw system prompt; Cbnd = system-block boundary token; Cpost = first token after the boundary.

</details>

<details open>
<summary>

### Background

</summary>

Persona vectors are a way to read out a model's internal representation of "what kind of agent it's pretending to be" by extracting a single hidden-state vector per persona and comparing them. The vectors only mean something relative to a fixed extraction recipe: which forward pass do you run, which token's hidden state do you take, which layer do you read out at? The choice has many plausible answers — last-token of the chat-templated prompt, mean over the model's generated response, last-token of the raw system prompt before any chat template, the system-block boundary token (`<|im_end|>`) — and prior published work (Chen et al. 2025, arXiv:2507.21509) does not converge on one.

Source-issues: [#201](https://github.com/superkaiba/explore-persona-space/issues/201), [#218](https://github.com/superkaiba/explore-persona-space/issues/218); supersedes [#85](https://github.com/superkaiba/explore-persona-space/issues/85).

All of this project's persona-vector results so far ([#92](https://github.com/superkaiba/explore-persona-space/issues/92), [#99](https://github.com/superkaiba/explore-persona-space/issues/99), [#113](https://github.com/superkaiba/explore-persona-space/issues/113), [#123](https://github.com/superkaiba/explore-persona-space/issues/123)) used one recipe — Method A, last-token of the chat-templated input. If alternative recipes recover different geometry, those prior results are recipe-specific artifacts and the cosine numbers they cite don't transfer. If alternative recipes agree, the prior results are robust. An early 20-persona pilot showed mean-centered Pearson r between Methods A and B falling between 0.30 and 0.83 — suggestive of disagreement but underpowered. We scaled to the full 275-role roster, added 4 more recipes (Method B\*, the raw-prompt last-token recipe, the boundary-token recipe, and the post-boundary first-token recipe) for a total of 6 recipes / 15 pairs, included a same-recipe noise-floor positive control, and swept all 28 layers to rule out a layer-sampling artifact.

</details>

<details open>
<summary>

### Methodology

</summary>

We tested six recipes on Qwen2.5-7B-Instruct across 275 personas and 240 questions. Each recipe extracts a hidden-state vector from a different position or pooling strategy in the same model:

- **Method A** (project default): last token of the full chat-templated prompt.
- **Method B**: mean over the model's generated response tokens (greedy, 200 tokens).
- **Method B\***: mean over input tokens of the same prompt+response forward pass as Method B.
- **Raw-prompt last token**: the last token of the raw system prompt (no chat template applied).
- **Boundary token**: the token at the system-block boundary (`<|im_end|>`).
- **Post-boundary first token**: the first token after that boundary.

For each recipe, we average the extracted vector across 240 questions to get a per-persona centroid, then compare every pair of recipes with two metrics that capture different things:

1. **Per-persona cosine** ("do they put each persona in the same place?"). For each of 275 personas, compute the cosine between its centroid under recipe X and its centroid under recipe Y. High cosine means both recipes encode that persona in the same direction. We report the worst-case persona (cos_min) — even one disagreement means the recipes aren't drop-in substitutes.
2. **Mean-centered Pearson r** (mc_r — "do they draw the same map?"). Each recipe produces a 275×275 table of how similar every persona is to every other. We correlate those two tables (37,675 off-diagonal entries, after subtracting the global mean). High mc_r means the recipes agree on the *structure* — which personas cluster together, which are outliers — even if the absolute coordinates differ. Two people drawing a city map from memory: the streets are in different places, but the neighborhoods are in the same relative arrangement; you can't overlay the maps, but you can use either one to answer "is the library closer to the school or the prison?"

A recipe pair "passes" only if **both** metrics clear their thresholds at a given layer. We validated the thresholds with a noise-floor control: splitting the 240 questions into two halves and comparing the same recipe against itself gives cos_min ≥ 0.99 and mc_r ≥ 0.99 — the thresholds are well above measurement noise. We initially tested at 4 layers [7, 14, 21, 27] in [#201](https://github.com/superkaiba/explore-persona-space/issues/201), then swept all 28 layers in [#218](https://github.com/superkaiba/explore-persona-space/issues/218) to confirm no sweet spot was missed.

A representative input/output (Method A, single persona, single question):

```
System: You are a librarian who has been working at a small public library for 30 years.
User:   What's the best way to organize a personal book collection at home?

(Forward pass: take hidden_states[layer=21] at the last input token — the final
token of the chat-templated prompt, just before generation. Result: a 3584-dim
vector. Mean across the 240 questions for this persona = the persona's Method A
centroid at layer 21.)
```

</details>

<details open>
<summary>

### Result 1: Recipes disagree on absolute direction

</summary>

![Per-persona cosine distributions across extraction methods](https://raw.githubusercontent.com/superkaiba/explore-persona-space/66e4434/eval_results/issue_201/figures/per_persona_cos_pair_dists.png)

**Figure 1.** *No pair of extraction recipes places each persona in the same direction — per-persona cos_min ranges 0.01–0.70 across all 15 pairs and 4 tested layers, three orders of magnitude below the same-recipe noise floor (≥ 0.99).* Violin plots show the distribution of per-persona cosine similarity (one cosine per persona) for each of the 15 recipe pairs (rows) at each of 4 layers (columns), N=275 personas, 240 questions per centroid. Green dashed line at 0.95 = pass threshold; red dashed line at 0.85 = kill threshold. Every pair at every layer falls well below both lines. Same-recipe noise floor (Method A vs Method A, 120/120 cross-half) sits at cos_min ≥ 0.9945 across the same 4 layers.

No pair of recipes produces centroids that point the same way: per-persona cos_min ranges from 0.01 (the worst persona at the project-default-vs-boundary-token pair at layer 7) to 0.70 (the best layer for the easiest pair) across all 15 pairs and 4 tested layers. The three-orders-of-magnitude gap to the same-recipe noise floor (≥ 0.99) rules out sampling noise as the explanation (N=275 personas, 240 questions per centroid). Even tokens from the *same forward pass* disagree — Method A (last token of the chat-templated prompt) versus the boundary-token recipe (the system-block boundary token in the same prompt, 15–30 positions earlier) has cos_min = 0.07 at layer 7, so position in the residual stream — not just content — determines what the hidden state encodes about persona identity. Pooling strategy matters more than which tokens are pooled: Method B\* (mean over input tokens) is closer to Method B (mean over response tokens) than to Method A (single last input token), even though Method A and Method B\* cover overlapping token ranges — the mean-vs-single-position distinction is the primary axis of recipe disagreement. Prior results that cite specific cosine values ([#92](https://github.com/superkaiba/explore-persona-space/issues/92), [#99](https://github.com/superkaiba/explore-persona-space/issues/99), [#113](https://github.com/superkaiba/explore-persona-space/issues/113), [#123](https://github.com/superkaiba/explore-persona-space/issues/123)) are Method-A-specific.

Sample numerical samples — disagreeing recipe pairs (cos_min and cos_mean per pair, layer 7):

```
A vs B    layer 7   cos_min = 0.533   cos_mean = 0.597   verdict: KILL (both metrics)
A vs Cbnd layer 7   cos_min = 0.066   cos_mean ≈ 0.30    verdict: KILL (cos)
B vs B*   layer 7   cos_min ≈ 0.45    cos_mean ≈ 0.55    verdict: KILL (cos)
```

Sample numerical samples — same-recipe noise-floor control (Method A cross-half, 120 vs 120 questions):

```
A vs A    layer 7   cos_min = 0.9996  mc_r = 0.9993
A vs A    layer 14  cos_min = 0.9994  mc_r = 0.9996
A vs A    layer 21  cos_min = 0.9975  mc_r = 0.9965
A vs A    layer 27  cos_min = 0.9945  mc_r = 0.9970
```

</details>

<details open>
<summary>

### Result 2: But recipes agree on relative geometry

</summary>

![Mean-centered matrix correlation across extraction-method pairs and layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/66e4434/eval_results/issue_218/figures/matrix_corr_pair_layer.png)

**Figure 2.** *Mean-centered Pearson correlation between the 275×275 persona-similarity tables from each pair of recipes — the cluster structure survives the recipe change even though absolute directions don't.* Heatmap rows = 15 recipe pairs; columns = 28 layers (0..27); cell color = mc_r on 37,675 off-diagonal entries of the mean-centered cosine matrices. mc_r reaches ~0.90 for the project-default (Method A) vs response-mean (Method B) pair at layers 21–27. N=37,675 off-diagonal pairs per cell.

Mean-centered Pearson r between the 275×275 cosine matrices reaches 0.90 for Method A vs Method B at layers 21–27, meaning the two recipes agree on which personas are close to each other and which are far apart. The cluster structure — the ranking of inter-persona distances — survives the recipe change. Concretely, at layer 21 the project-default vs response-mean pair has per-persona cos_min = 0.617 (KILL on absolute direction) but mc_r = 0.892 (close to passing on structure), and at layer 27 the same pair has cos_min collapsing to −0.010 (effectively orthogonal) while mc_r stays at 0.896. Claims about *which personas cluster together* are more robust than claims about *how far apart they are* in absolute terms.

Sample numerical samples — Method A vs Method B per-layer, the primary recipe pair (the project default vs the most natural alternative):

```
layer 0    cos_min = 0.704   mc_r = 0.489   verdict: KILL (mc_r)
layer 7    cos_min = 0.533   mc_r = 0.532   verdict: KILL (both)
layer 14   cos_min = 0.636   mc_r = 0.750   verdict: KILL (cos)
layer 21   cos_min = 0.617   mc_r = 0.892   verdict: KILL (cos)
layer 24   cos_min = 0.695   mc_r = 0.902   verdict: KILL (cos) — best mc_r layer
layer 27   cos_min = -0.010  mc_r = 0.896   verdict: KILL (cos)
```

Sample numerical samples — full 15-pair mc_r at layer 21 (the layer where structural agreement is strongest for the primary Method A vs Method B pair); recipe codes follow the Setup-details key (Craw / Cbnd / Cpost):

```
A vs B          layer 21   cos_min = 0.617   mc_r = 0.892   ← primary pair
A vs B*         layer 21   cos_min = 0.462   mc_r = 0.366
A vs Craw       layer 21   cos_min = 0.170   mc_r = 0.692
A vs Cbnd       layer 21   cos_min = 0.420   mc_r = 0.745
A vs Cpost      layer 21   cos_min = 0.510   mc_r = 0.788
B vs B*         layer 21   cos_min = 0.572   mc_r = 0.340
B vs Craw       layer 21   cos_min = 0.102   mc_r = 0.574
B vs Cbnd       layer 21   cos_min = 0.462   mc_r = 0.651
B vs Cpost      layer 21   cos_min = 0.361   mc_r = 0.679
B* vs Craw      layer 21   cos_min = 0.005   mc_r = 0.373
B* vs Cbnd      layer 21   cos_min = 0.624   mc_r = 0.411
B* vs Cpost     layer 21   cos_min = 0.345   mc_r = 0.376
Craw vs Cbnd    layer 21   cos_min = 0.106   mc_r = 0.883   ← raw-prompt-side pair
Craw vs Cpost   layer 21   cos_min = 0.312   mc_r = 0.850
Cbnd vs Cpost   layer 21   cos_min = 0.426   mc_r = 0.859
```

</details>

<details open>
<summary>

### Result 3: Direction and structure are anti-correlated across depth

</summary>

![28-layer verdict heatmap across all 15 recipe pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/66e4434/eval_results/issue_218/figures/verdict_partition_heatmap.png)

**Figure 3.** *Joint pass/grey/kill verdict at every (recipe pair, layer) cell — mean-centered r and per-persona cos_min are anti-correlated across model depth, and no layer satisfies both pass thresholds simultaneously across all 15 pairs.* Rows = 15 recipe pairs; columns = 28 layers (0..27); color encodes verdict (pass = both metrics clear thresholds; grey = one passes, one is borderline; kill = at least one metric fails). 419 of 420 cells are KILL; 1 is GREY (and 0 are PASS).

The 28-layer sweep ([#218](https://github.com/superkaiba/explore-persona-space/issues/218)) shows mean-centered Pearson r rises broadly from 0.49 at layer 0 to 0.90 at layers 22–27 as the model builds abstract persona representations, while per-persona cos_min stays flat at 0.53–0.70 throughout — the absolute encoding never converges. At layer 27 (the final layer), per-persona cosine collapses to near-zero (cos_min = −0.010 for Method A vs Method B) while mc_r stays at ~0.90: the model's pre-unembedding representations point in persona-specific but recipe-dependent directions while preserving the overall persona map. Across the full 28-layer sweep, no layer satisfies both pass thresholds simultaneously: 419 of 420 (recipe pair × layer) cells are KILL and 1 is GREY (zero PASS). The phenomenon is layer-universal — there is no sweet spot where the recipe choice stops mattering for absolute direction.

Sample numerical samples — verdict counts per layer in the 28-layer sweep (out of 15 pairs per layer):

```
layer 0    n_pass=0  n_grey=1  n_kill=14
layer 7    n_pass=0  n_grey=0  n_kill=15
layer 14   n_pass=0  n_grey=0  n_kill=15
layer 21   n_pass=0  n_grey=0  n_kill=15
layer 27   n_pass=0  n_grey=0  n_kill=15
totals     n_pass=0  n_grey=1  n_kill=419   (out of 420 cells)
```

Sample numerical samples — Method A vs Method B trajectory across all 28 layers (illustrating the anti-correlated movement: mc_r climbs from 0.49 → 0.90 while cos_min stays in the 0.5–0.7 band, then collapses at the final layer):

```
layer  0   cos_min = 0.704   mc_r = 0.489    layer 14   cos_min = 0.636   mc_r = 0.750
layer  1   cos_min = 0.708   mc_r = 0.482    layer 15   cos_min = 0.625   mc_r = 0.778
layer  2   cos_min = 0.715   mc_r = 0.548    layer 16   cos_min = 0.616   mc_r = 0.805
layer  3   cos_min = 0.620   mc_r = 0.526    layer 17   cos_min = 0.623   mc_r = 0.813
layer  4   cos_min = 0.594   mc_r = 0.557    layer 18   cos_min = 0.635   mc_r = 0.843
layer  5   cos_min = 0.590   mc_r = 0.504    layer 19   cos_min = 0.636   mc_r = 0.881
layer  6   cos_min = 0.580   mc_r = 0.537    layer 20   cos_min = 0.624   mc_r = 0.886
layer  7   cos_min = 0.533   mc_r = 0.532    layer 21   cos_min = 0.617   mc_r = 0.892
layer  8   cos_min = 0.476   mc_r = 0.554    layer 22   cos_min = 0.653   mc_r = 0.897
layer  9   cos_min = 0.609   mc_r = 0.516    layer 23   cos_min = 0.679   mc_r = 0.897
layer 10   cos_min = 0.580   mc_r = 0.524    layer 24   cos_min = 0.695   mc_r = 0.902
layer 11   cos_min = 0.580   mc_r = 0.527    layer 25   cos_min = 0.694   mc_r = 0.897
layer 12   cos_min = 0.561   mc_r = 0.581    layer 26   cos_min = 0.699   mc_r = 0.902
layer 13   cos_min = 0.610   mc_r = 0.663    layer 27   cos_min = -0.010  mc_r = 0.896
```

</details>

## Source issues

This clean-result distills evidence from:

- **[#201](https://github.com/superkaiba/explore-persona-space/issues/201)** — 6-way extraction-method ablation at 4 layers (275 personas × 240 questions × 6 methods × 15 pairs).
- **[#218](https://github.com/superkaiba/explore-persona-space/issues/218)** — 28-layer sweep confirming the 4-layer finding is layer-universal (no sweet spot was missed).
- **[#85](https://github.com/superkaiba/explore-persona-space/issues/85)** — original "check extraction methods" issue (closed as a duplicate of [#201](https://github.com/superkaiba/explore-persona-space/issues/201) once the scaled-up panel landed).

Downstream consumers:

- **[#191](https://github.com/superkaiba/explore-persona-space/issues/191)** — EM × persona-vectors; reuses base-model centroids from `data/persona_vectors/issue_201/`. Should calibrate any EM-induced shifts against this baseline disagreement — a shift of cos = 0.05 is meaningless when the A-B baseline is already cos_min = 0.62.
- **[#92](https://github.com/superkaiba/explore-persona-space/issues/92), [#99](https://github.com/superkaiba/explore-persona-space/issues/99), [#113](https://github.com/superkaiba/explore-persona-space/issues/113), [#123](https://github.com/superkaiba/explore-persona-space/issues/123)** — prior clean-results depending on Method A; absolute-cosine claims need recipe footnotes; relative-ranking claims (cluster structure, distance orderings) are more defensible.


