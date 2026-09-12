# Calibration-only dictionary quality comparison

Specified 2026-09-12 before reading main component predictions or any new
calibration-control quality result. This implements the existing analysis
plan's request for calibration-matched control comparisons. It changes no
main sample, dictionary, sparsity, fit, or primary contrast.

Capture source residuals from all 119 valid frozen calibration token sequences,
at exactly the lens estimator's valid positions (skip first four and final
token). Use the same individual, unpadded native BF16 eager forward as lens
calibration. This small diagnostic preserves the calibrated forward geometry;
it is not an answer-generation distribution or a new outcome dataset.

Apply the actual uploaded main J/R dictionaries and all three registered Haar
rotations. Use the existing FP32/highest, TF32-disabled nonnegative gradient
pursuit, nested k=5/10/25, and 128-token batches. Save the source activations,
per-token components/statistics, rotation matrices, contracts, and summaries.
Verify uploads between capture and decomposition and after each orientation.

Summarize each prompt by its valid-position mean, then weight prompts equally.
Report residual energy divided by original energy, actual L0, zero-update and
increasing-error steps, centered component/remainder variance, and covariance.
Captured variance is component variance divided by original variance; it is
not constrained to [0,1], and component/remainder variances need not add.

Primary controls remain comparisons at identical k. As a supplementary quality
comparison, for each native k and rotated dictionary, choose the registered
control k with nearest calibration residual-energy fraction. Break exact ties
by smaller k. Only label the comparison as within-range when the native quality
lies inside the range spanned by that rotation's three registered k values.
Always report signed quality, L0, and captured-variance mismatches. A nearest
grid point is approximate matching, not exact equivalence; out-of-range cells
remain unmatched. J and R may choose different control k for this supplementary
diagnostic, which must not replace same-k G or lens-difference results. No
interpolation, new k values, refitting, test-based matching, or favorable-cell
selection is permitted.

Centered variances and covariance use anchored within-prompt centered moments
plus between-prompt mean variation. Raw second moments are used only for
energy. This avoids cancellation for constant or large-offset activations.
The `match` phase requires uploaded reports and their exact per-prompt source
hashes. Quality on raw calibration tokens may not transfer to generated answers.
