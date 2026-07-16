# Issue #779 — persona-level monitoring across ALL layers, ALL methods

**Ask (chat 2026-07-14):** "did we check at different layers? do sweep for all the
methods." No — every prior group-level run used only the frozen system layer per
trait (evil L14 / syco L26 / halluc L17, selected on the eval rig, not for this
comparison). This sweeps all 28 layers for every pre-generation monitoring method
(plus a post-generation oracle ceiling), read at the persona level (60 corpus
personas, mean of 40 questions), group-level Pearson r vs the persona's pooled
mean judge score.

## The layer choice changes the verdict — the frozen-layer result was a selection artifact

Deployable methods (need NO trait-corpus training):

| trait | raw @frozen / best-layer | map_generic @frozen / best (beats raw) | g_generic @frozen / best (beats raw) |
|---|---|---|---|
| evil | +0.537 / +0.813 (L27) | +0.516 / **+0.857** (L25) — **21/28** | +0.343 / +0.436 — 1/28 |
| sycophancy | +0.774 / +0.829 (L17) | +0.853 / **+0.875** (L18) — **22/28** | +0.458 / +0.670 — 0/28 |
| hallucination | +0.449 / +0.564 (L25) | +0.262 / **+0.634** (L23) — **21/28** | −0.115 / +0.261 — 3/28 |

In-distribution methods (need trait-corpus training; LOGO over personas) + ceiling:

| trait | map_LOGO best (beats raw) | g_LOGO best (beats raw) | oracle best (beats raw) |
|---|---|---|---|
| evil | +0.882 L26 (26/28) | +0.912 L26 (20/28) | +0.963 L25 (28/28) |
| sycophancy | +0.893 L27 (28/28) | +0.959 L24 (27/28) | +0.903 L13 (28/28) |
| hallucination | +0.606 L27 (26/28) | +0.786 L17 (28/28) | +0.668 L27 (28/28) |

## Findings

1. **The frozen-layer "generic map loses" verdict does not hold across layers.**
   At the eval-rig-frozen layers the deployable generic map looked mixed (tied
   evil, won sycophancy, lost hallucination −0.186). Across the 28 layers it beats
   raw at **21 / 22 / 21 of 28 layers**, and its swept-best beats raw's own
   swept-best for evil (+0.857 vs +0.813) and hallucination (+0.634 vs +0.564),
   ties for sycophancy (+0.875 vs +0.829). The frozen layers (L14 evil, L17
   hallucination) simply sit off the map's mid-to-late-layer peak (L23–L26). So
   the earlier "map does not beat raw at persona level" was a layer-selection
   artifact of the single frozen layer.

2. **Selection-bias caveat (binding).** "Beats raw at 21–22/28 layers" is
   selection-robust (a broad majority, not one cherry-picked layer), but the
   swept-best-vs-swept-best comparison is argmax-on-both-sides and NOT a clean
   estimate. A definitive "the generic map beats raw" needs held-out layer
   selection (nested CV: pick the layer on held-out personas, evaluate on
   truly-held-out ones). Neither the frozen layer nor the argmax is the fair
   single number; the honest read is "the generic map is generally at or above
   raw across the stack, layer-dependent, clean verdict pending nested CV."

3. **The direct predictor splits cleanly by training distribution — the cleanest
   result here.** The in-distribution g_LOGO is near the top (best 0.79–0.96,
   beats raw 20–28/28), while the deployable g_generic is the WORST method (beats
   raw 0–3/28, negative at most layers). This is #779's "direct predictor wins
   in-distribution (r 0.91) but does not transfer" finding, now shown at persona
   level across all layers: the direct predictor collapses under distribution
   transfer, and its sparse generic-LMSYS training labels (evil 0.2% / syco 6%
   positive) make it unlearnable for those traits.

4. **The map degrades far more gracefully than the direct predictor.** Under the
   same generic→corpus distribution transfer, map_generic stays competitive with
   raw (often above) while g_generic falls apart. So the learned map's real value
   is not a large accuracy win over raw — among deployable methods raw and
   map_generic are close — but its robustness relative to a supervised probe.

5. **The big gaps are all in-distribution or oracle.** map_LOGO (26–28/28),
   g_LOGO (20–28/28), and the oracle ceiling (28/28, tops ~0.90–0.96) dominate —
   all require either trait-corpus training data or the actual answer. None is a
   deployable pre-generation monitor.

## Bottom line

Whether the deployable (generic) map beats the original persona-vector method at
persona level is **layer-dependent**: it loses at the eval-rig-frozen layers but
beats raw at a broad majority of layers and at its own peak. The earlier
frozen-layer correction over-stated the loss. What is unambiguous across layers:
the deployable direct predictor (g_generic) is the worst method, the
in-distribution methods (g_LOGO, map_LOGO) are strong but non-deployable, and
persona-level averaging lifts every read far above per-context. A clean
map-vs-raw verdict awaits held-out layer selection.

## Artifacts
- `persona_level_layer_sweep.py` (3-method), `persona_level_layer_sweep.json`
- `persona_level_layer_sweep_allmethods.py` (6-method), `persona_level_layer_sweep_allmethods.json`
  (full per-layer curves for pv_raw / map_generic / map_LOGO / g_generic / g_LOGO
  / oracle; paired diffs vs raw at frozen + each method's argmax layer).
- Reuses arm_headline GramRidge/loaders; pass_b LMSYS bundle + corpus blobs (local); 0 GPU-h.
