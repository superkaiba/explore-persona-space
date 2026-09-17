# Original persona-vector results in the four-method layout

[Browser plot](https://eps.superkaiba.com/tasks/1739/figure/c5_behavior_transfer_original_four.png?v=6a24067cc752)

This plot preserves the three original P-B series from the user's screenshot,
using `r2v2_fits` at commit `5aae0a472b`. The fourth series, context → context,
comes from the matching `e1_fc / arm1_ctx_e1` factorial rows at commit
`6dba8178f061980d3285fe4d35ba3149f7bdba3c`. All 312 shared P-B E1 rows reproduce
the original correlations, evaluation sizes and frozen layers exactly.

No model inference, map fitting, extraction, behavioral scoring or bootstrapping
is performed. The renderer uses the same four-method colors, ordering, nine-panel
layout and axis limits as the million-map plot. Synthetic evaluation is omitted.

The old maps use 18,793 generic pairs plus 6,468 harmful-compliance pairs or
16,000 sycophancy/hallucination pairs. Projections use the original whitening.
Layers for answer → context / answer → mapped / context → context / answer → real
are respectively 15/22/15/22 for harmful compliance, 16/11/16/11 for sycophancy,
and 24/23/24/27 for hallucination. The added context direction uses the original
context-projection layer, without selecting a new layer.

Generic chat and ID use the old P-B evaluation rows. ID was held out from the
behavior readout, but its context–answer pairs were included in map fitting.
OOD retains all 13 historical datasets (six sycophancy, two hallucination, five
harmful-compliance datasets). Each OOD bar is the equally weighted mean of dataset
correlations, and its whisker is one standard error across datasets, as in the
original screenshot. Generic/ID have no displayed uncertainty interval. Repeated
fixed projection scores across readout folds are counted once per dataset.

The source figure flagged low target-score spread for generic harmful compliance
and three of its five OOD datasets. All source limitations remain applicable.

Reproduce with `uv run python scripts/issue1739_old_four_method_figure.py`.
The summary records all 36 estimates, source rows, layers, counts and input hashes.
PNG, PDF, grayscale audit and metadata are under
`figures/paper/c5_behavior_transfer_original_four.*`.

## Compact manuscript layout

[Combined browser plot](https://eps.superkaiba.com/tasks/1739/figure/c5_behavior_transfer_original_combined.png?v=2ab3133e0f2a)

The combined version uses one full-width row of three behavior panels, each with
generic chat, ID and OOD groups, a shared axis and a shared four-method legend.
All 36 estimates and the original OOD standard-error intervals remain unchanged.
It uses the paper's `c2a-v2` fonts, palette and authoring scale.

Reproduce with `uv run python scripts/issue1739_old_combined_figure.py`.
Vector PDF, review PNG, grayscale audit, caption and provenance metadata are under
`figures/paper/c5_behavior_transfer_original_combined.*`. Include the PDF at
exactly `\textwidth`. `combined_publication.json` records the verified browser
copy and its SHA-256.
