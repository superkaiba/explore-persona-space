# Methodology — issue 1902

**Design:** Four checkpoints of one fully open post-training chain: `allenai/OLMo-2-1124-7B` (base), `-SFT`, `-DPO`, `-Instruct` (the RLVR stage; the chain's RLHF stage is realized as DPO — a checkpoint-availability fact of the release). Two real-user corpora built from `lmsys/lmsys-chat-1m` (English, exact-string deduplicated, formatted-token budget ≤ 3,072 under the longest render): a single-turn corpus (16,000 generic first-turn queries + 1,200 math/code-screened rows + 500 GSM8K + 300 MBPP marked strata = 18,000 rows) and a multi-turn corpus (16,000 conversations, one unique prefix per row). Each checkpoint generated one on-policy answer per context under its own native render (chat template for SFT/DPO/RLVR; plain `User:/Assistant:` for base, which has no chat template), plus n = 2 repeat generations on a pinned 1,000-context subset for a sampling-reliability ceiling. Teacher-forced capture then ran every (activation checkpoint m × answer source s) cell — a 4×4 grid × 2 corpora — under one shared plain-text render (`User: {q}\nAssistant: {a}`), so serialization is uniform across the m axis; a 2,000-context native-render subset plus a pilot flip rule cover the render choice (see the final result section). Per row and layer the capture stores the mean context state u and the mean answer-token state w; ridge and MLP maps u→w are fit per cell with cluster-grouped 6-fold cross-validation (k-means k = 40 clusters on MiniLM embeddings; identical fold assignment across all cells of a corpus). Grid analyses run on the four-source unflagged intersection (16,391 single-turn / 15,363 multi-turn rows, both above the 8,192 = 2d target). Cross-stage transfer evaluates each stage's map on every other stage's pairs — direct, general-linear-aligned, orthogonal-aligned, and fixed-answer-text — against shuffled-correspondence and spectrum-matched nulls, held out per fold; the operator battery reads the ridge primal weights. Both mapping arms run: context arm (both corpora) and prefix arm (multi-turn corpus). A zero-GPU follow-up round added per-transition retention CIs and a paired per-row answer-length control (details under **Data extraction:**). Rig provenance: the first full fit pass ran on stores corrupted by a capture-phase row-order scramble (length-sorted batch outputs saved under unsorted row ids, scrambling the context↔answer pairing for every non-first source cell); it produced a spurious family null and a designed sanity halt, the capture was fixed (commit `3064d4de84`), all stores were regenerated, and every number in this body comes from the full 549-unit re-run on repaired stores.

**Training:** **N/A — no model training.** Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Checkpoints | `allenai/OLMo-2-1124-7B{,-SFT,-DPO,-Instruct}`, revisions pinned (footer) | plan §10; arXiv 2501.00656 |
| Generation | vLLM 0.11.0, temperature 1.0, top_p 0.95, max_tokens 1,024, n = 1, seed 42 (+ seeds 43/44 on the reliability subset) | [#779](https://eps.superkaiba.com/tasks/779) protocol (`issue779_collect.py:558`) |
| Degeneracy filters | truncation = hit the 1,024-token cap; repetition = any 8-gram repeated ≥ 8 times | plan §4 (pilot-tuned) |
| Capture layers | 17 of 32: every 2nd layer + the final layer | plan §11 (storage-driven; pilot-verified interior peak) |
| Summaries | u = mean over full-prompt tokens; w = mean over answer tokens; prefix arm p = mean over prefix tokens; fp16 store | [#779](https://eps.superkaiba.com/tasks/779) recipe |
| Ridge | GCV over λ ∈ logspace(−2, 4, 13), standardize-X / center-Y, Gram/dual eigh, float64 | `fit_h.py` defaults ([#823](https://eps.superkaiba.com/tasks/823) grid; [#779](https://eps.superkaiba.com/tasks/779) protocol) |
| MLP | 1×512 GELU, AdamW lr 1e-3, weight decay 1e-4, ≤ 300 epochs early-stop, PCA-64 head, seed 42 | `issue658_fit_predictors.py:111-114` via `fit_h.mlp_fit_predict` |
| Folds | cluster-grouped 6-fold, seed 42; marked strata assigned whole-stratum | [#825](https://eps.superkaiba.com/tasks/825) (conversation-grouped 6-fold) |
| Headline layer | layer* = 31 = argmax of mean diagonal out-of-fold R² across the four checkpoints (context arm); one shared frozen layer for all compared cells | plan §4 (selection-fair) |
| Bootstrap | 1,000 cluster-grouped draws from persisted per-context squared-error components | [#825](https://eps.superkaiba.com/tasks/825) |
| Retention bars | persistence 0.8 / half-rewrite 0.5, on the median over the 6 adjacent transitions | [#825](https://eps.superkaiba.com/tasks/825) anchors |
| Rotation null | 50 draws | `issue1345_common.N_ROTATION_COSINE_DRAWS` |
| Length control (follow-up) | paired same-rows: both answers inside the shared p10 length band (46–254 generated tokens); variants `\|Δlen\| ≤ {8, 16, 32}` tokens + band-only; marginal matched variant n = 4,526/column; seed 20260801 | `scripts/issue1902_followup_9ater.py` |

**Evaluation:** The dependent variable is pooled group-fold out-of-fold R² ("map quality" Q) per (activation checkpoint, answer source, layer, fitter, arm) cell, computed on held-out clusters only. Retention ρ(i→j) = general-linear-aligned held-out R² of stage i's map on stage j's pairs ÷ stage j's own diagonal R², with alignments fitted on train folds only and the answer-side alignment fitted on same-answer-text pairs so it cannot absorb the answer-distribution change. Every fitted cell reports the identity+learned-bias baseline and kNN retrieval (acc@{1, 5, 10}, euclidean + cosine, chance = k/n_pool ≈ 4e-4 at pool ≈ 2.7k). Uncertainty is a 1,000-draw cluster-grouped bootstrap; the frozen-at-layer* CI convention is used throughout (selection-inherited CIs, re-selecting the layer per draw, differ by ≤ 0.005 in every diagonal cell). Nulls: shuffled-pairing (fit floor, −0.037 to −0.069 vs observed 0.27–0.41), shuffled-correspondence alignment refit, and a spectrum-matched random operator. Sampling-reliability ceilings (design-aligned split-half over the n = 2 repeat generations, 914 surviving contexts, Spearman–Brown): base 0.72, SFT 0.92, DPO 0.97, RLVR 0.97 — the base diagonal is markedly more sampling-noise-limited than the post-trained ones. No LLM-judged DV exists in this design (activation-space R² throughout), so the dual-DV requirement is N/A; the language-intrusion audit is N/A (non-Qwen evaluated model). Rig gates all passed on the repaired run: per-source unflagged survival 94.4–99.2%; minimum per-fold n_tr 12,717 > d = 4,096 (well-posed fits); mean diagonal Q at layer* 0.344–0.349, above the 0.15 sanity floor and > 5× the shuffled-pairing null; bf16 batched-vs-batch-1 equivalence (early-layer cosine ≥ 0.99994); slow-vs-fast ridge parity ≤ 1.8e-14 maximum relative difference on 3 production-shape slices. Scope limits carried by every result: n = 1 rollout per context per source (the reliability ceilings quantify the cost); one model family; the RLHF stage realized as DPO; the transfer and operator batteries are single-turn context-arm at layer 31 only. Per-context reconstruction cosine is higher for base (~0.88 vs ~0.82) even where base R² is lower — cosine is direction-only and insensitive to the magnitude miscalibration R² penalizes, so it does not contradict the diagonal result. Presentation note (acknowledged conciseness WARNs): this body carries 11 result sections, one per planned instrument, so several per-result blocks and the total-prose budget deliberately exceed the soft caps, and three secondary views (the MLP grid, retrieval accuracy, CKA curves) are deliberately linked rather than embedded to keep one figure per result.

**Data extraction:** Tier-1 real-user data (LMSYS single-turn queries and multi-turn conversations) plus tier-2 marked benchmark strata (GSM8K, MBPP). Filters: English (the LMSYS language field), non-empty first-turn query (single-turn) / at least one completed user–assistant exchange before a final user query (multi-turn), exact-string dedup, formatted-token budget ≤ 3,072. Flagged generations are reported and excluded from the four-source intersection, never silently dropped: truncation 2.2–5.2% of 18,000 rows per source (DPO highest at 930), repetition-flagged ≤ 0.64% (base single-turn, 116). Teacher-forcing inputs are built by token-id concatenation with offset-mapping span boundaries (straddling tokens excluded at the prefix boundary, included at the context boundary). The follow-up length control joins each row's generated-answer token count from the generation shards to the persisted per-context squared-error components (Pearson ≥ 0.99998 against capture-side counts in all four sources), then reads the answer-text axis on the same rows with both answers inside a shared length band, optionally with a per-row length gap cap.

**Sample training/evaluation data + completions:** Four unflagged rows quoted verbatim from the on-policy generation shards — one query (row `single_10476`) across the base, SFT, and DPO answer sources, showing the length/style contrast that drives the answer-text grid axis, plus one RLVR row — benign rows chosen for display under the content-hygiene policy (the corpus is unscreened real user text); full shards with all rows: [HF `issue1902_stage_map/raw_completions/gen/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

Example 1 — base checkpoint, single-turn row `single_10476` (cluster 23), generated under the plain render; capture then teacher-forces `User: {query}\nAssistant: {answer}` at every checkpoint. Cherry-picked benign row (not a random sample — content-hygiene policy); all rows: [full shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

<details>
<summary>Example 1 — base checkpoint, row single_10476 (verbatim)</summary>

> **Query:** can you freeze watermelons?
>
> **base answer (63 tokens, unflagged):** Yes, you can freeze watermelons. However, it's important to note that freezing a watermelon may alter its taste and texture, and it might not be as tasty as freshly cut watermelon. If you're planning to use it for a specific recipe, you might want to consider other options.

</details>

Example 2 — SFT checkpoint, the SAME row `single_10476`, generated under its chat template, captured under the shared plain render. Cherry-picked benign row (not a random sample — content-hygiene policy); all rows: [full shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

<details>
<summary>Example 2 — SFT checkpoint, row single_10476 (verbatim)</summary>

> **Query:** can you freeze watermelons?
>
> **SFT answer (69 tokens, unflagged):** Yes, you can freeze watermelons. However, it's important to note that this might change the texture and taste of the watermelon. Additionally, make sure to freeze watermelons in a way that prevents ice from forming in unexpected places, which can cause structural damage. You should also consider any potential risks related to food safety.

</details>

Example 3 — DPO checkpoint, the SAME row `single_10476`, generated under its chat template, captured under the shared plain render — the long structured answer style typical of the DPO source (median 387 generated tokens vs base 55). Cherry-picked benign row (not a random sample — content-hygiene policy); all rows: [full shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

<details>
<summary>Example 3 — DPO checkpoint, row single_10476 (verbatim)</summary>

> **Query:** can you freeze watermelons?
>
> **DPO answer (455 tokens, unflagged):** Yes, you can freeze watermelons. However, it's important to follow proper steps to ensure they freeze correctly without becoming mushy or contaminated. Here are the steps to properly freeze watermelons:
>
> 1. **Preparation:**
>    - Select fresh, ripe watermelons with a firm, intact rind.
>    - Wash the watermelons thoroughly to remove any dirt or debris on the surface.
>    - Slice the watermelons into small chunks or balls, as smaller pieces freeze faster and more evenly.
>    - Remove the seeds if you prefer them to be seedless in the frozen state.
>
> 2. **Blanching (Optional but Recommended):**
>    - Blanching helps preserve the flavor, color, and texture of the watermelon when frozen.
>    - Boil water in a pot and then submerge the sliced watermelon for about 2 minutes. This step helps deactivate enzymes and slows down nutrient loss.
>    - After blanching, immediately transfer the watermelon pieces to a bowl of ice water to stop the cooking process and preserve the bright color.
>
> 3. **Freezing:**
>    - Spread the blanched (or unblanched) watermelon pieces on a baking sheet or silicone mat in a single layer. This step prevents them from clumping together and helps them freeze quickly and evenly.
>    - Place the baking sheet in the freezer until the watermelon pieces are completely frozen, usually within 2 to 4 hours.
>    - Once frozen, transfer the pieces to airtight containers or zipper-lock bags.
>
> 4. **Storage:**
>    - Label the containers with the date and use the frozen watermelon within 3 to 4 months for best quality.
>    - Ensure the containers are leak-proof to avoid freezer burn, which can cause the watermelon to become dry and tasteless.
>
> When you're ready to use the frozen watermelon, simply thaw it in the refrigerator or under cold running water, depending on your preference and the intended use (e.g., blending, drinking as a smoothie, or incorporating into a recipe). When frozen properly, watermelon maintains a good texture and flavor, making it a refreshing addition to many dishes and drinks during warmer months or as a year-round treat.

</details>

Example 4 — RLVR checkpoint (Instruct), single-turn row `single_02253` (cluster 35), generated under its chat template, captured under the shared plain render. Cherry-picked benign row (not a random sample — content-hygiene policy); all rows: [full shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e688725a77db7fa4e82b27dd0978bf0c4185ba88/issue1902_stage_map/raw_completions/gen).

<details>
<summary>Example 4 — RLVR checkpoint, row single_02253 (verbatim)</summary>

> **Query:** what is the natural logarithm of the constant e
>
> **RLVR answer (91 tokens, unflagged):** The natural logarithm of the constant e is defined as ln(e). By definition, the natural logarithm of a number is the power to which e must be raised to obtain that number. Since e is the base of the natural logarithm (ln), when we take the natural logarithm of e, we get: ln(e) = 1. This is because: e^1 = e. So, ln(e) = 1.

</details>

</content>
</invoke>