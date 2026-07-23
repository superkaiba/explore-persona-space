---
title: 'The four fiction-character context→dialogue maps are one dominantly shared
  linear operator: a pooled map recovers 81–98% of each character''s ceiling, with
  small per-character slope residuals (MODERATE confidence)'
kind: analysis
tags: []
created_at: '2026-07-23T16:28:37Z'
has_clean_result: false
parent_id: 1310
origin_prompt: 'write this up into a clean result (consolidating the #1310 cross-persona
  similarity rounds 1-3: cross-character battery, principled re-analysis, and the
  assistant direct test on #1335 row-paired cells)'
workflow: v1
---
# The four fiction-character context→dialogue maps are one dominantly shared linear operator: a pooled map recovers 81–98% of each character's ceiling, with small per-character slope residuals (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- One map fit on all four characters' pooled data, with a single global offset, recovers 81–98% of each character's own-map held-out ceiling at layer 19 (fracs base 0.81–0.92, instruct 0.90–0.98); adding per-character offsets buys nothing (M1−M0 CIs straddle zero).
- The genuinely persona-specific remainder is a small slope residual: M2−M1 = +0.007 to +0.025 with every scenario-bootstrap CI above zero, largest for the villain persona; instruct is more shared than base on every read.
- Raw per-character maps do not cross-apply (all 24 off-diagonal transfers negative, −0.22 to −2.6): the shared operator is visible only after pooling or a learned linear change of coordinates (data-paired reparameterization recovers 84–97% of each ceiling on instruct, 60–79% base, vs matched-capacity nulls ≈ −0.02).
- Data-paired Procrustes-aligned operator cosine reads 0.516 base / 0.593 instruct (shuffle-fit null ≈ 0.002) — above the same-persona story↔chat framing anchor (0.455), below the base↔instruct anchor (0.686): framing moves this operator more than character identity does. A spectrum-only aligned cosine ≈ 0.99 is quasi-mechanical (shuffle-fit null ≈ 0.99) and is reported as descriptive only.

## Goal

**This experiment in context:** [#1310](https://eps.superkaiba.com/tasks/1310) established a character-specific context→dialogue map for each of four fixed-label story personas; this analysis asks whether those are four different operators or one shared operator expressed in per-persona coordinates, using the anchors and alignment conventions of [#1345](https://eps.superkaiba.com/tasks/1345) (framing operator battery) and [#825](https://eps.superkaiba.com/tasks/825) (assistant-map ceiling; base↔instruct aligned-cosine anchor 0.686). It consolidates the parent's inline analysis rounds of 2026-07-22 into one standalone result.

**Broader narrative:** together with the framing results — the assistant map survives template removal but collapses under narrative framing, and chat↔plain-text share one operator up to coordinates — this supports a single next-reply operator attached to the turn-structured response-slot format, persona-generic up to small linear corrections. A direct assistant-inclusion test (the same battery on row-paired assistant vs character cells from [#1335](https://eps.superkaiba.com/tasks/1335)) is running as a follow-up round and folds here on landing.

## Methodology

**Design:** re-analysis of the persisted scene-aggregated per-character cells: four fixed-label story personas (a warm helper, a calm AI, an ordinary person, a theatrical villain), each with 300 on-policy story scenes per model (Qwen2.5-7B base and instruct), one point per scene — X = the scene-prompt context activation, Y = the mean dialogue activation over the scene's kept turns — at headline layer 19. Three read families over one shared scenario→fold partition (K=5, seed 0): (a) a 4×4 cross-persona transfer matrix (each fold-trained source map evaluated on the matching held-out target fold); (b) a nested shared-vs-specific lattice — M0 one pooled map with global centering, M1 one shared map with per-persona train-fold centering, M2 per-persona maps — with scenario-shuffle nulls through the M1 path and 1,000-draw grouped bootstraps on the rung deltas; (c) operator-similarity statistics — raw Frobenius cosine, data-paired activation-Procrustes-aligned cosine (orthogonal input/output alignments fit from scenario-paired activations), prediction-space response cosine, and the spectrum cosine — each read against a random-rotation null AND a shuffle-fit null (maps fit with the identical ridge recipe on scenario-permuted pairings: spectrum-matched, structure-free); plus a data-paired general-linear reparameterization per ordered pair against matched-capacity nulls (shuffled-fit and random-rotation, 5 draws each).

**Training:** N/A — no model training and no new text generation (re-analysis of persisted activation stores).

| Hyperparameter | Value | Source |
|---|---|---|
| Ridge fit | GCV Gram ridge, dof cap 0.9, λ grid logspace(−2, 4, 13) | parent recipe (footer provenance); uncapped GCV degenerates on this store |
| Folds | K=5 scenario-grouped, seed 0; one partition for all cells | parent recipe |
| Layer | 19 (headline; frozen-set convention) | parent recipe |
| Cells | 300 aggregated points per persona per model | parent store |
| Nulls | shuffle-fit maps 5/model; rotation 20 (cosine) / 3 (Procrustes bank); scenario-shuffle 5 (lattice) | this analysis |
| Bootstrap | 1,000 draws, scenario-grouped | project convention |

**Evaluation:** the dependent variables are geometric, not judged behaviors: held-out pooled R² (fold-test-mean reference) for within-cell fits, transfer, the lattice rungs, and reparameterization recovery; Frobenius/aligned cosines for operator geometry. Every statistic is judged against its own null, and the shuffle-fit null is the binding one for alignment-based reads — it preserves the shared activation geometry and the ridge-shrinkage spectrum while destroying the context→dialogue structure, so anything it reproduces (the spectrum cosine ≈ 0.99; the input-side subspace overlap) is treated as carrying no shared-structure evidence. Equality gates: the recomputed within-cells matched the committed values bit-exactly (worst |Δ| = 0.00) before any new read, in both rounds. No LLM judge is used anywhere.

**Data extraction:** no new data; the persisted store (bf16 28-layer span summaries, ~8 GB, 27 shards) was re-staged prefix-scoped from the data repo at a pinned revision onto the shared data disk and re-read for all fits. One method correction is part of this result's history: the first round's two-sided-Procrustes "aligned cosine" is the von Neumann spectral optimum — rotation-invariant, hence spectrum-only — and was demoted to descriptive after its shuffle-fit null read ≈ 0.99; the data-paired activation-Procrustes read replaced it. Verifier WARNs acknowledged: Takeaways bullets exceed the 30-word target to keep CIs and nulls attached to their numbers, figure captions run over the 60-word caption target, and the footer store path is pinned by revision token rather than a per-file link.

**Sample training/evaluation data + completions:** none generated by this analysis. The analyzed rows are the parent task's persisted on-policy prefill rows; verbatim worked samples (kept and dropped rows) are shown in the parent clean-result body, and the full JSONL artifacts are linked from its footer.

## Results

### One pooled map with a single global offset carries 81–98% of every character's map

Per-persona held-out R² at layer 19 for the lattice rungs — M0 (one pooled map, global offset), M1 (plus per-persona offsets), M2 (per-persona maps) — with 1,000-draw scenario-bootstrap whiskers and scenario-shuffle null ticks, per model; the round-1 reparameterization rung is marked for placement.

![Shared-vs-specific decomposition lattice per persona and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9edaab4fa46a7bd10be7a0fcaae8d2aa8d2760b5/figures/issue_1310/xpersona_decomposition.png)

> **Figure.** M0 ≈ M1 sits just below M2 for every persona: fracs M1/M2 base 0.81–0.92, instruct 0.90–0.98; M1−M0 CIs straddle zero; M2−M1 = +0.007 to +0.025 with every CI above zero; M1 shuffle nulls ≈ −0.05. Pooled R²: base 0.425/0.424/0.438 (M0/M1/M2), instruct 0.595/0.594/0.602.

One operator plus one global offset carries most of every character's map; per-persona offsets are irrelevant, and the persona-specific part is a small slope residual, largest for the villain — the persona that is also the weakest and most idiosyncratic cell in every other read.

### Individually-fit character maps do not cross-apply raw; a learned linear change of coordinates recovers most of each ceiling

4×4 held-out transfer at layer 19 (rows = source persona's map, columns = target persona's points; diagonals = the committed within-cells; values annotated), per model.

![Cross-persona transfer matrices at layer 19, base and instruct](https://raw.githubusercontent.com/superkaiba/explore-persona-space/82d85db5ee9ad578ca36f7320266cb98e2aa64c2/figures/issue_1310/xpersona_transfer_matrix.png)

> **Figure.** Every off-diagonal raw transfer is strongly negative (base −0.22 to −2.11, mean −1.10; instruct −0.84 to −2.59, mean −1.56) while the diagonals reproduce the committed within-cells exactly.

Individually-fit maps carry persona-specific centering and basis plus estimation noise that raw cross-application amplifies. A train-fold linear input/output alignment around the frozen source operator recovers 0.196–0.379 against instruct ceilings of 0.233–0.401 (84–97% of ceiling; base 60–79%), far above matched-capacity nulls (≈ −0.02) — yet below the pooled rungs above: pooling all personas' data beats any single-source map plus learned alignments.

### The operator alignment survives the strictest null and places between the framing anchors

Operator statistics per unordered pair with their nulls, per model: raw Frobenius cosine and data-paired Procrustes-aligned cosine against shuffle-fit nulls; the spectrum cosine as a descriptive marker; cross-project anchor lines.

![Operator similarity statistics with shuffle-fit and rotation nulls and calibration anchors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7e367f9a5a6a7bd07583aa5595fb28fe4944ee25/figures/issue_1310/xpersona_cosine_reparam.png)

> **Figure.** Procrustes-aligned cosine 0.497–0.548 (base) and 0.570–0.615 (instruct) with shuffle-fit nulls ≈ 0.002; raw cosine 0.20–0.35 vs nulls ≈ 0.00; spectrum cosine ≈ 0.99 with a shuffle-fit null ≈ 0.99 (descriptive only).

The aligned cosine survives the strictest null — the real activation-geometry alignments applied to structure-free maps recover ≈ 0 — so it reflects genuine shared map structure, not shared scenario geometry. On the project anchor scale, character pairs (0.516/0.593 means) sit above the same-persona story↔chat pair (0.455) and below base↔instruct (0.686) and chat↔plain (0.732/0.855): framing distorts this operator more than character identity does. Prediction-space response cosine concurs: 0.37 base / 0.55 instruct vs shuffle-fit nulls ≈ 0.

---

**Repro:**
- Code (branch main): `scripts/issue1310_xpersona_similarity.py` @82d85db5ee (round 1: transfer + swap + cosine legs); `scripts/issue1310_xpersona_similarity_v2.py` @9edaab4fa4 (round 2: lattice + prediction-space + shuffle-fit nulls) + @7e367f9a5a and @100df8abdc (data-paired Procrustes leg + its shuffle-fit null). Cosine/reparameterization conventions per `scripts/issue1345_operator_comparison.py`.
- Eval JSONs (git, main): `eval_results/issue_1310/xpersona_similarity/` (9 files, round 1) and `eval_results/issue_1310/xpersona_similarity/v2/` (9 files incl. `summary_v2.json` with the lattice, prediction-space, nulled operator stats, Procrustes cosines + calibration anchors).
- Figures: `figures/issue_1310/xpersona_decomposition.png` @9edaab4fa4, `xpersona_transfer_matrix.png` @82d85db5ee, `xpersona_cosine_reparam.png` @7e367f9a5a (all with `.meta.json` sidecars).
- Reused artifacts: the parent scene-aggregated activation store `issue1310_char_map/analysis_tensors/store_onpolicy/` @ b24279a1f9ca (HF `superkaiba1/explore-persona-space-data`; 27 shards, ~8 GB; re-downloadable; equality gate reproduced every committed within-cell bit-exactly before each round's new reads).
- Compute: 0 GPU-h; ~35 min VM CPU across rounds. Judge: none. WandB: n/a (no training).

**Context:** created 2026-07-23 from user chat; origin prompt (verbatim): "write this up into a clean result (consolidating the #1310 cross-persona similarity rounds 1-3: cross-character battery, principled re-analysis, and the assistant direct test on #1335 row-paired cells)". Child of [#1310](https://eps.superkaiba.com/tasks/1310); consolidates its inline user-chat free-analysis rounds 1–2 + Procrustes addenda (2026-07-22), whose round history remains in the parent body. Round 3 — the assistant direct test (the same battery on the row-paired assistant plain-Q&A / character plain-Q&A / character fiction-framed cells of [#1335](https://eps.superkaiba.com/tasks/1335), matched 4,045 questions) — was running at write time with the base arm complete; its results and figure fold into this body on landing, and the title extends to the assistant claim only then. Sibling framing result: [#1345](https://eps.superkaiba.com/tasks/1345).
