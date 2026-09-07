# Independent implementation review

Codex reviewer `k_ablation_review`, 2026-09-07: PASS for both the capture extension and paired comparison. No automated Claude usage.

The capture review checked the pinned 1,000-context row order, seeds 47–51 and inherited sampling parameters, generation-time prompt token IDs, layer-19 answer-span slicing including the end-of-turn tail, source-reference indexing, separate generation/capture processes, atomic checkpoint validation, and revision-pinned upload byte verification. The parity-first dispatcher was reviewed after its addition. Actual source-capture parity remains a required execution-time gate.

The scoring review checked that the K=10 target averages exactly the original five vectors and the five new vectors. Both R² and retrieval contrasts subtract the same paired bootstrap draws, R² re-centers each resampled target, and candidate identities derive from the original source bank. Previous-endpoint checks cover exact per-query top-1/top-5 outcomes, row order, point R², and bootstrap R². The new-five-only comparison is a descriptive replication diagnostic; it does not replace the registered K=10 minus existing-K=5 contrast.

Three focused tests passed. They cover literal resampling equivalence for R², cached geometry against the canonical scorer over every original-bank subset, and exactly zero paired intervals when the two endpoint targets are identical. Ruff passed. No workflow rules or workflow implementation were modified, so the unrelated no-flags repository-wide workflow scan was not repeated.
