# Four fixed-direction projections with the million map

User-requested plot: answer direction on context, answer direction on mapped
answer, context direction on context, and answer direction on real answer, for
generic chat, ID and OOD. Rows are sycophancy, hallucination and harmful compliance.

The frozen layer-19 map was trained on 963,444 pairs. All methods share the same
evaluation rows within each panel. The fixed raw instruction-contrast directions
are unchanged; no behavior regression is fitted. Synthetic evaluation is excluded.

## Data assembly

The three existing million-map prediction series come from
`issue1739-fixed-regimes-20260917`. Direct answer-on-context scores come from
`issue1739-small-map-20260917/outputs`, joined by exact context ID; this projection
does not use either map. Context IDs, labels, groups, dataset membership and both
shared raw-projection baselines must match before use.

For ID/OOD, the renderer replays the original 2,000 group-bootstrap draws, adds
the direct projection, and verifies that all original bootstrap series are
reproduced. OOD is the equal-weight mean of dataset-specific correlations:
AITA for sycophancy, NQ-Open and SimpleQA for hallucination, and HH-RLHF and
ToxicChat for harmful compliance. ID is held out from map fitting and direction
extraction; it previously trained separate supplementary behavior regressions.

Generic chat uses the historical evaluation subset from
`issue1739-map-size-diagnostic-20260917`, including map-training overlaps. It
therefore does not establish conversation-disjoint generalization. Generic
hallucination uses a graded trait score; QA hallucination uses fabricated fractions.
Generic harmful compliance has only nine nonzero scores. The answer direction's
negative real-answer correlations on TriviaQA and NQ-Open limit its validity there.

## Reproduction

Run `uv run python scripts/issue1739_four_method_figure.py` from this worktree.
The first run assembles the cached summary; later runs verify its input hashes and
render it. The script writes PNG, PDF, a grayscale audit and a provenance sidecar
to `figures/paper/c5_behavior_transfer_million_four.*`. The saved summary contains
all 36 estimates, intervals and source hashes. Bootstrap draws are saved alongside it.

No model inference, judging, map fitting or direction extraction is performed.
