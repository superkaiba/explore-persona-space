# Final refusal-magnitude comparison

This directory is authoritative. The sibling initial run used floating-point
rate subtraction; `..._countfix` corrected mathematical rank ties; this final
run also reads prior JSONL through file iteration. All scientific results of
the two count-corrected runs agree exactly. Earlier outputs are preserved.

All three scores use the same 124 minimal-refusal pairs at layer 19. Existing
on-policy Qwen2.5-7B-Instruct answers use 10 draws per endpoint, temperature 1.0,
and a 2,048-token cap. Answer vectors use tail-included pooling. Archived
integer refusal counts define the outcome; no new generation or judging ran.

| Change magnitude | Spearman correlation with absolute refusal-rate change |
|---|---:|
| Context, `||dc||` | 0.7779 |
| Mapped answer, `||dc @ A||` | 0.7624 |
| Observed answer, `||da||` | 0.7805 |

The mapped-minus-context difference is −0.0156, with paired family-bootstrap
95% interval [−0.0443, 0.0289]. There is no evidence of an improvement from
mapped magnitude in this bank. The observed-answer score is an empirical
reference, not a mathematical ceiling. The primary uncertainty uses 2,000
paired resamples of 21 semantic-family/corpus clusters, treating XSTest as one
corpus. No pairs were selected based on observed refusal flips.

`summary.json` contains the complete result; `report.md` includes category
results and limitations. Its scalar calibration R² values are leave-family-out
predictions using a separately fitted scalar slope and intercept for each
score. Raw Spearman correlations use fixed scores, without a fitted readout.
Subset calibration results slice these same all-pair held-out predictions.

Reproduce the analysis into a new directory with
`scripts/issue2569_refusal_magnitude_comparison.py --out <fresh-directory>`;
the plot-only producer is `scripts/issue2569_refusal_magnitude_figure.py`.
Use the project's existing environment and shared-VM thread caps.

`input_provenance.json` pins the executed analysis source and all frozen inputs.
`completion.json` hashes the eight completed run outputs. This README is a
later navigation note, not a compute output. Manuscript changes are deferred
to the parent discussion; no paper text was edited in this round.
