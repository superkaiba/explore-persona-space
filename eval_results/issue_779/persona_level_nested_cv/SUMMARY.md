# Issue #779 — nested-CV persona-level head-to-head (definitive)

**Ask (chat 2026-07-14):** "let's run the nested CV" — remove the layer-selection
asterisk on the persona-level map-vs-raw comparison. Honest held-out layer
selection: outer 5-fold over the 60 corpus personas; within each outer fold the
read-out layer is chosen on the outer-train only (fit-free methods by outer-train
correlation; corpus-fit methods by 4-fold inner CV); every persona is scored at a
layer (and, for corpus-fit methods, a fit) that never saw it. Reads are z-scored
per fold (label-free) before pooling, so folds that pick different layers cannot
corrupt the pooled correlation (the v1 bug: pooling raw reads across layers
collapsed the oracle to +0.084 for evil; the fix restores it to +0.792).

## Result: the deployable map is at PARITY with the original method — ties on 2 of 3 traits, small win on hallucination

Nested-CV persona-level Pearson r (z-pooled over 60), paired vs pv_raw:

| trait | pv_raw | map_generic (Δ vs raw, P>0) | verdict |
|---|---|---|---|
| evil | +0.731 | +0.743 (**+0.012** [−0.062,+0.079], 0.61) | tie |
| sycophancy | +0.802 | +0.797 (**−0.005** [−0.052,+0.039], 0.41) | tie |
| hallucination | +0.547 | +0.619 (**+0.073** [+0.011,+0.150], 0.99) | map wins |

Per-fold agreement: evil fold-diff −0.005 (2/5 folds+), sycophancy −0.013 (2/5+),
hallucination +0.090 (4/5+) — same story.

**The deployable generic map does not reliably beat the raw persona-vector
projection at persona level.** It matches raw on evil and sycophancy (paired CIs
straddle zero) and modestly beats it on hallucination (+0.073, CI excludes zero).
1 win, 2 ties — no consistent advantage.

## The four estimates of this comparison, reconciled

| estimate | evil | syco | halluc | what inflated/deflated it |
|---|---|---|---|---|
| corpus-LOGO map, frozen layer (my 1st claim) | +0.12 | +0.12 | +0.08 | in-distribution map + frozen layer → too favorable |
| generic map, frozen layer (my 1st "correction") | −0.02 | +0.08 | **−0.19** | frozen layers (L14/L17) are bad for the map → too harsh |
| generic map, argmax layer (sweep) | wins 21/28 | 22/28 | 21/28 | argmax-on-eval-set → selection-optimistic |
| **generic map, nested CV (this)** | **tie** | **tie** | **+0.073 win** | **selection-unbiased — the honest answer** |

The truth is parity with a small hallucination edge. The earlier swings were all
layer-selection artifacts in one direction or the other.

## The other methods (nested CV, z-pooled r), and the deeper point

| trait | g_generic (deployable probe) | map_corpus | g_corpus (in-dist probe) | oracle (ceiling) |
|---|---|---|---|---|
| evil | +0.475 (−0.256, worse) | +0.760 (+0.029) | +0.764 (+0.033) | +0.792 (+0.061) |
| sycophancy | +0.405 (−0.397, worse) | +0.852 (+0.050) | +0.916 (**+0.114**) | +0.852 (+0.050) |
| hallucination | −0.028 (−0.575, worse) | +0.595 (+0.048) | +0.766 (**+0.219**) | +0.653 (+0.106) |
(Δ vs pv_raw in parens; bold = paired CI excludes zero.)

1. **The deployable direct predictor (g_generic) is decisively the WORST method** —
   loses to raw on all three (−0.256 / −0.397 / −0.575, all CIs exclude zero).
   The supervised probe trained on generic LMSYS labels does not transfer to the
   trait corpus (and evil/sycophancy LMSYS labels are near-degenerate). This is
   the single most robust result across every analysis: a probe is worse than
   just projecting onto the persona vector.
2. **The oracle answer-reading ceiling is only +0.05–0.11 above raw.** At the
   persona level the raw pre-generation projection already sits within ~0.1 of
   the read that sees the actual answer — there is almost no headroom for any
   pre-generation method to beat raw. This is *why* nothing beats raw by much:
   persona-level monitoring is "easy" precisely because raw nearly saturates the
   ceiling.
3. **The only method that clearly beats raw is g_corpus** (in-distribution direct
   predictor): sycophancy +0.114, hallucination +0.219, reaching ~oracle — but it
   requires judged behavior labels on the trait corpus, which the whole
   pre-generation-monitor premise is meant to avoid needing.
4. **The learned map's value proposition does not survive honest CV.** Neither the
   generic nor the corpus-trained map beats raw more than marginally; the map
   ≈ raw regardless of training distribution. The map degrades gracefully (unlike
   g_generic), but "route the projection through a learned map to monitor better"
   is not supported at persona level.

## Bottom line

Under honest held-out layer selection, the deployable context→answer map does NOT
beat the original persona-vector method at persona level — it ties on evil and
sycophancy and wins hallucination by +0.073. The raw projection is already near
the answer-reading ceiling, so there is little to beat. The clean wins are all
in-distribution (g_corpus) or unavailable (oracle); the deployable supervised
probe is strictly worse than raw. This confirms the recollection that the map
"doesn't beat the original method on held-out contexts" — the correct statement
is parity, with a small hallucination edge, not a win.

## Artifacts
- `persona_level_nested_cv.py` (v2, with the per-fold z-scoring fix),
  `persona_level_nested_cv.json` (per-fold r mean±SD, z-pooled r, selected layers
  per fold, paired bootstraps vs pv_raw, all six methods × three traits).
- Reuses arm_headline GramRidge/loaders; pass_b LMSYS bundle + corpus blobs
  (local); 0 GPU-h. Supersedes the frozen-layer (`map_transfer_group_level`) and
  argmax-layer (`persona_level_layer_sweep`) verdicts for the map-vs-raw question.
