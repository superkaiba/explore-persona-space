# Speaker-figure integration preview

Add six hollow diamonds to panel A of the existing three-panel speaker figure.
Each diamond is a frozen Base map evaluated on the same Instruct targets as the
teal separate-fit bar. Its horizontal whisker is the minimum-to-maximum range
over five folds, not a confidence interval. Plain-text dialogue has no diamond
because that checkpoint-transfer cell was not evaluated. Existing bars and
panels B/C retain their numerical inputs. This is a preview; the live Overleaf
manuscript has not been edited.

The figure can carry the new comparison without an additional panel. Keep
retrieval and the identity/bias controls in the results text or appendix rather
than mixing another metric into the existing R2 axis. The complete result and
controls are in `eval_results/issue_2054/k5_stage_transfer/README.md`.

## Discussion contribution

This adds meaningful evidence for continuity across post-training: the earlier
speaker results fitted separate maps to each checkpoint, or transferred between
speakers within one checkpoint. Successful frozen Base-to-Instruct transfer tests
whether an already-fitted relationship remains useful after post-training.
It strengthens the current discussion's pre-existing-persona interpretation,
but does not directly identify a persona or prove that post-training merely
refines one. The assistant in a story sits within the character range, so the
better-supported contrast is story versus chat. It does not establish the
discussion's further suggestion that generated tokens reinforce a character.

Use the short paragraph in `proposed_discussion.tex` after the current
persona-selection paragraph, or incorporate its first two sentences into that
paragraph and put its caveat in the limitations. Reference manuscript:
Overleaf commit `433800ea38c2075f33aacb2f465984458c4d3b1c`, fetched 2026-09-17.
`proposed_caption.tex` describes the additional markers and also clarifies that
only panel B's first row is averaged across four characters.

Paragraph outline: cross-checkpoint preservation; setting-specific contrast;
limits on interpretation. Claim-to-evidence mapping:

| Claim | Evidence | Limit |
|---|---|---|
| Greater predictive continuity in stories | Frozen/own mean R2: 67.5–77.8% for characters, 73.3% for assistant-story, 26.2% for chat | Descriptive comparison of this model pair and these banks; no significance claim |
| Story/chat contrast includes the assistant | Assistant-story frozen R2 0.400; chat 0.177 | Framing, template familiarity and response quality are not causally separated |
| More direct support than separate own-map fits | Base map, normalization and intercept frozen; same target folds as Instruct own fit | Predictive portability does not show an unchanged operator or isolate SFT |

## Reproduction and provenance

From the repository environment, with this checkout's `src` on `PYTHONPATH`, run
`uv run python figures/issue_2054/k5_stage_transfer_integration/c4_stage_transfer.render.py`.
This is a plotting operation on completed artifacts, with no fitting or generation.
It checks all six target-own endpoints against the existing plotted values and
verifies the frozen mean against each setting's five fold scores before exporting
PDF, color PNG, grayscale PNG and metadata.

The archived `base_figure.data.json` is the paper's existing figure data. The
archived `base_figure_source.py` is its exported source snapshot, retained
byte-for-byte. That source's hash differs from the hash named in the paper's
metadata and its layout predates the exported figure's top legends. This preview
therefore explicitly defines its own layout; it does not claim exact reproduction
of the old image. All original numeric inputs are preserved. The new metadata
records actual byte hashes of the source, data, completed transfer results,
renderer and shared plotting style.

Validation: all six Instruct endpoints matched to 1e-12, all 30 frozen fold
scores reconciled with their displayed means, and every recorded input hash
matched its source bytes. The original figure data is byte-identical to the
current paper's data. Color and grayscale renders were visually inspected;
all panels and six transfer markers are present with sane ranges. Ruff checks
passed for the new renderer, and the repository inline payload lint gate passed
for both Python files (no mapped tests for these figure-artifact paths).
