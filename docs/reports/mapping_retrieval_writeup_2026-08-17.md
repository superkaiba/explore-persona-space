# How good is the context→answer map, really? (retrieval deep-dive, 2026-08-17)

**Setup (one line):** every number below scores the #1738 context→answer map — layer-19 ridge from the context-end residual state to the mean-over-answer-tokens state, trained on **one sampled (temperature-1.0) answer per context**, n=88,378 real LMSYS/WildChat multi-turn conversations — by full-pool retrieval on the pinned 9,941-row holdout: the map's predicted answer vector must pick the true answer out of all 9,941 held-out answers (chance 0.01%). "Covered rows" = the 1,988 holdout contexts that have 4 extra on-policy answer draws (the resample control). Sources: task #2202 rounds 5–9 (all 0 GPU-h on banked tensors, except the contrastive round: ~1.2 h on one H100).

**Glossary (each term defined once):**
- **acc@1** — fraction of held-out contexts whose predicted vector's nearest pool answer is the true one.
- **whitened cosine** — cosine similarity after rescaling the space by the training-answer covariance (all directions unit variance); kills hubs caused by anisotropy near the pool mean.
- **CSLS** — a re-ranking that penalizes each candidate answer by its average similarity to its 10 nearest queries, docking "promiscuous" residual hubs (Conneau et al. 2018). "CSLS-whitened" = CSLS applied in the whitened-cosine space. Batch-retrieval convention: the penalty depends on the whole query set.
- **fresh-draw reference** — retrievability of the true answer when the *query* is a fresh on-policy draw of that same answer; the "how well could you possibly do" comparator, per convention.
- **draw-averaged target** — the pool entry replaced by the mean of 5 answer draws (original + 4 resamples); removes answer-sampling noise from the target. Eval-side only — the map is unchanged.
- **margin** — per-context similarity of the true answer minus the best competitor; positive = true answer wins.

## 1. Headline

The banked raw-euclidean acc@1 of **0.816** understates the map by ~18 points. Correcting the read-out metric (whitening + CSLS) and denoising the target (draw-averaging) takes the *same unchanged map* to **0.994–0.995**, at or above the fresh-draw reference. What looked like map error was almost entirely hub geometry plus answer-sampling noise.

## 2. The metric ladder (single-draw targets, full 9,941 pool)

| Convention | acc@1 |
|---|---|
| raw euclidean (banked headline) | 0.816 |
| raw cosine | 0.828 |
| pool-mean-centered cosine | 0.882 |
| CSLS on raw cosine | 0.909 |
| whitened cosine | 0.954 |
| **CSLS on whitened cosine** | **0.976** |
| **double-strength CSLS-whitened** | **0.985** |
| whitened *euclidean* | 0.020 (degenerate) |

![acc@1 by similarity convention](https://raw.githubusercontent.com/superkaiba/explore-persona-space/510a3802afea73dddeee0c859560b90d1f545acb/figures/issue_2202/fig_convention_zoo.png)

Mechanism findings from the 18-convention sweep (#2202 round 6): whitened *euclidean* collapses because variance-equalization lets per-vector norm noise dominate — degradation is monotone in whitening depth (whitening even just the top-64 directions hurts), and normalizing (cosine) removes exactly that, so whitening's win is entirely conditional on normalization. Classic hubness rescalings alone are weak here (mutual proximity +0.001, local scaling +0.05–0.06) but **compose** with whitening. The inverted-softmax alternative is temperature-fragile (0.736 at β=10 vs 0.849 at β=30).

## 3. Convention-matched ceilings

The fresh-draw reference is 0.943 raw / **0.979 whitened cosine** (covered rows). Under CSLS-whitened conventions the map's battery score sits **at or above its own matched reference** (0.976 vs 0.973) — the map's prediction retrieves the stored answer better than a fresh draw of the true answer does, because the map predicts the *denoised expected* answer while a fresh draw carries sampling noise.

## 4. Draw-averaged targets (covered rows, n=1,988; matched single-draw values on the same rows)

| Target | raw euclidean | whitened cosine | CSLS-whitened |
|---|---|---|---|
| single draw | 0.815 | 0.962 | 0.981 |
| average of 5 draws | **0.909** | **0.987** | **0.994** |

![Single-draw vs draw-averaged targets](https://raw.githubusercontent.com/superkaiba/explore-persona-space/510a3802afea73dddeee0c859560b90d1f545acb/figures/issue_2202/fig_avg_target.png)

About 9 points of the raw-convention "error" is sampling noise in the single-draw target. This is eval-side denoising: the single-draw-trained map already predicts the draw-average (least squares converges to the conditional mean; #1073 showed the training-side counterpart — averaged-target training changes the map by <0.01 R²).

## 5. Do better maps help? (nonlinear + discrimination-trained)

| Map | raw euclid | whitened cos | holdout R² |
|---|---|---|---|
| ridge (banked) | 0.816 | 0.954 | 0.68 |
| MSE-MLP w8192 (banked) | 0.874 | 0.946 | 0.70 |
| contrastive linear (InfoNCE, ridge warm-start) | 0.870 | 0.943 | −4.58 |
| contrastive MLP (InfoNCE) | 0.599* | 0.956 | −1.96 |

*cosine-objective norm artifact — its cosine reads are 0.897 raw / 0.956 whitened.

![Contrastive and nonlinear maps vs conventions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/391a08f75150b32b2b3bd437617d42a00ca45220/figures/issue_2202/fig_contrastive_maps.png)

Nonlinear fitting buys +0.06 under the naive metric and *nothing* under the corrected one — the whitened read-out absorbs what the MLP was adding. Training the map to discriminate directly (InfoNCE, in-batch negatives — the strongest "train it aware of the metric" design) tops out at 0.956, below free read-time CSLS-whitened on plain ridge, while holdout R² collapses (outputs become retrieval keys, not activation predictions). Theory agrees: an unregularized multi-output least-squares map is invariant to any invertible linear re-metric of the target space, so whitening cannot be "trained in" — it belongs at the read-out.

## 6. The full matrix converges

With the metric hub-corrected and the target denoised, **all seven maps** (ridge, four nonlinear variants, two contrastive) land in a **0.991–0.995** band — architecture contributes ~0.4pp. Best cell: ridge + double-strength CSLS-whitened at **0.995**.

![Seven-map convergence under CSLS-whitened, single vs averaged targets](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b45bb8c86b0b91cdd1342084034fe1152cc9d83b/figures/issue_2202/fig_avgtgt_convergence.png)

## 7. The residual 0.5%: near-miss map error on underdetermined turns, not answer twins

The 15 residual failures (union over the two CSLS conventions, covered rows) are **not** target degeneracy: the wrongly-retrieved answers are unrelated to the true ones (whitened twin-cosine max 0.36, median ~0.05; 13/15 map-attributable under the resample control). They concentrate on terse or deictic final turns — "continue", "ok create it", a bare "why?" — where the answer depends almost entirely on deep history. Most are near-misses (ranks 2–23). Caveat: the double-strength CSLS penalty over-corrects two of them (rank 2→216), so quote the operating point as "≥99%", not the fourth digit.

**Differentiation metrics** (how well the map separates, beyond binary acc@1): pairwise AUC — the probability the true answer outranks a random distractor — is ≥0.9996 in every convention (≥0.99999 whitened). Successes win by ~7–8× more margin than failures lose by (whitened-cosine medians +0.13 vs −0.03), and draw-averaging widens winning margins (+0.13→+0.17) as well as flipping failures. MRR (mean of 1/rank): 0.860 raw → 0.996 at the clean operating point.

**Worst-discriminated contexts** (bottom-50 by margin — 11 failures + 39 barely-won): a *different population* from the old raw-euclidean failure profile. Refusal enrichment is gone (refusal-answer and refusal-adjacent shares sit at pool level; the explicit-content *topic* reads nominally 2.8× over-represented, but at 3 of 50 rows that is within count noise). What remains poorly separated is Chinese-language contexts (28% vs 12% pool), coding (30% vs 17%), and shallow 2-turn exchanges (56% vs 42%). Russian is strongly under-represented (1/50 = 2% vs 8.9% pool).

## 8. Training on averaged draws: the #1073 null replicates at multi-turn scale

Run 2026-08-18 (#1738 round `avg-target-maps`, ~11 GPU-h of 8×H200): a matched-n 20k stratified train subset, 4 extra temp-1.0 draws per training context (80k generations, recipe-matched to the originals), then two ridge fits on identical inputs — single-draw vs 5-draw-averaged targets — with λ selected on single-draw validation rows for both. Single-draw eval = full 9,941 pool; averaged eval = 5-draw-averaged targets over the 1,988 covered rows with a 1,988-entry pool (a smaller-pool convention than §4's full-pool-replacement read — the two are not cross-comparable; within-table comparisons are clean).

| Map (training target) | R² single | acc@1 raw / whitened, single | R² avg | acc@1 raw / whitened, avg |
|---|---|---|---|---|
| single-draw, n=20k | 0.656 | 0.765 / 0.932 | 0.701 | 0.903 / 0.984 |
| 5-draw-averaged, n=20k | 0.661 | 0.767 / 0.934 | 0.705 | 0.902 / 0.986 |
| single-draw, n=88k (banked reference) | 0.681 | 0.816 / 0.954 | 0.728 | 0.929 / 0.994 |

Training on averaged draws changes the map by ≤ +0.004 R² and ≤ +0.2pp acc@1 in every eval cell — least squares on single noisy draws already estimates the conditional-mean answer, exactly as #1073 found on the single-turn corpus. Training-set size dominates: the 88k map beats both 20k maps by ~0.02 R² / ~5pp raw acc@1. Caveat carried: 8.2% of draws hit the recipe-inherited 1,024-token generation cap (a property of the whole corpus line, matched to the original draws by construction).

**Phase B — the full convention battery on the new maps** (single-draw full pool → draw-averaged full-pool-replacement, the §4 convention; commit `1b80740702`):

| map | raw euclid | whitened cos | CSLS-whitened | dbl-strength CSLS |
|---|---|---|---|---|
| ridge 88k | 0.816 → 0.909 | 0.954 → 0.987 | 0.976 → 0.994 | 0.985 → 0.995 |
| single-trained 20k | 0.765 → 0.869 | 0.932 → 0.974 | 0.967 → 0.988 | 0.976 → 0.990 |
| avg-trained 20k | 0.767 → 0.870 | 0.934 → 0.975 | 0.968 → 0.990 | 0.979 → 0.990 |

Two additional reads: (1) the two 20k maps stay essentially equal under every convention (avg-trained ahead by 0.06–0.26pp in 7 of 8 cells) — the training-side null extends from R² to every retrieval read. (2) **The clean conventions compress the n-gap**: the 88k map's advantage over 20k shrinks from 5.1pp (raw euclidean) to 0.5–0.8pp (CSLS-whitened) — most of what 4× more training data buys in raw space is the same hub/anisotropy handling that whitening + CSLS provide closed-form.

## 9. Open
- **Prefix arm untested under clean conventions** — history-only retrieval reads 0.207 raw; nobody knows how much of that deficit is hub artifact. Free read on banked predictions, not yet run.

---
*Provenance: task #2202 (all eight rounds folded in-body: https://eps.superkaiba.com/tasks/2202); eval artifacts under `eval_results/issue_2202/{freshwhiten_avg,metric_zoo,contrastive_maps,avgtgt_completion,residual_read}/`; scripts `issue2202_{freshwhiten_avg,metric_zoo,contrastive_maps,avgtgt_completion,residual_read}.py`; contrastive weights on HF `issue2202_ctxfail/contrastive_maps/`. Map + corpus provenance: #1738. All pool answers are on-policy single temp-1.0 draws; resample draws same recipe, per-draw seeds.*
