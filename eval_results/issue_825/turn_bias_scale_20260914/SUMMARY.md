# Single-answer turn-transfer calibration

[R²-retention curves](https://eps.superkaiba.com/tasks/825/figure/turn_transfer_bias_scale_single_answer_20260914.png) · [Cosine-retrieval curves](https://eps.superkaiba.com/tasks/825/figure/turn_transfer_bias_scale_single_answer_20260914_retrieval.png). The right column of the R² plot uses a tighter vertical range to show the smaller turn-3 changes.

All comparisons use the same held-out answer targets within each model/turn/fold. Calibration fits a destination-training vector offset and optionally one scalar; it never sees test conversations. The source-map matrix stays fixed. This is target-informed adaptation, so it is a separate condition from unchanged transfer.

At turn 12, the following values are uncalibrated → bias only → bias plus scale:

- Instruction-tuned, map from turn 1: R² 0.316899 → 0.411669 → 0.417980; own-turn R² retention 57.46% → 74.65% → 75.79%; cosine top-1 68.86% → 71.89% → 66.18%.
- Instruction-tuned, map from turn 3: R² 0.523217 → 0.535570 → 0.536675; own-turn R² retention 94.87% → 97.11% → 97.31%; cosine top-1 85.09% → 85.21% → 83.89%.
- Base, map from turn 1: R² 0.197618 → 0.299332 → 0.309772; own-turn R² retention 35.94% → 54.44% → 56.34%; cosine top-1 53.84% → 59.17% → 49.56%.
- Base, map from turn 3: R² 0.517553 → 0.532536 → 0.533694; own-turn R² retention 94.12% → 96.85% → 97.06%; cosine top-1 84.37% → 85.33% → 86.53%.

Bias supplies 90.7–93.8% of the R² gain from calibration at these four endpoints. It does not remove most of the original gap to the own-turn map. Scaling adds little R² and changes cosine retrieval in opposite directions: it helps base-model turn-3 transfer, while hurting the other three endpoints. Even after both adjustments, opening-turn maps remain weaker than unchanged turn-3 maps.

All 50 declared cells and 300 fold cells completed. The largest raw-parent R² discrepancy is 5.55e-16. The turn-12 held-out sets contain 4,977 instruction-tuned and 4,996 base-model answers; retrieval pools range from 826 to 834 answers across those folds (chance about 0.12%). The identity-plus-bias baseline is retained in results.json and metrics.csv and agrees exactly across source-map comparisons. Estimates have no confidence intervals.

No new model generation or GPU was used. The main two-worker CPU run completed in about 21 minutes after the pilots, with 8 threads per worker. The input bank, fitted maps, calibration coefficients and per-row outputs are preserved; upload_verification.json identifies the verified archive.

One comment-only source edit during execution caused 120 metadata fingerprints to inspect the preceding function. An independent audit explained every mismatch, proved the historical edit left the executable AST unchanged, and verified unchanged map/prediction hashes and saved SSEs. The bounded repair changed only fingerprint metadata; original receipts and the audit are preserved under provenance_repair/. A regression test reproduces that line-shift failure and validates the source-snapshot fix. The complete reduction passed after repair.

See METHODS.md for the exact recipe and claim_paragraph.tex for a concise manuscript draft. Integration into the existing Figure 4 and parent clean-result body is deferred to the owning #825 analyzer/paper-integration pass at its next manuscript update; the current manuscript asset is unchanged.
