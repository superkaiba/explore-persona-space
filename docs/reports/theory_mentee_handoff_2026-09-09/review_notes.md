# Handoff review

An independent, read-only Codex reviewer checked the report against the existing result artifacts. The review covered decomposition equations, SAE category counts, learning-curve values, refusal correlations and amplitudes, and observed-answer high/low-write results. It found one mathematical clarification: the category-level kernel and gain statistics use signed held-out nested-model SSE improvements, not per-pair projection shares or norm ratios. Section 3.3 now states the producer's definitions explicitly.

No other substantive blocker was identified in the checked claims. The primary agent separately checked cross-family numerical results, archive coverage, artifact hashes, public report URLs, and PDF layout. This was a review of the handoff, not a replication of any experiment or a new semantic-annotation pass.

See `verification.json` and `report_link_checks.csv` for machine-readable packaging checks. The Overleaf URL returned 403 without project access; the other 29 report URLs returned 200 at the check time. All 518 bundled research files matched their recorded SHA-256 hashes; the seven indexed-only large local entries matched the pinned remote LFS hash.

## SAE supplement, 10 September

A second independent read-only review checked checkpoint compatibility, mapping definitions, and the older 65k archive gap against the producers. Its one correction was to spell out the direct regressor's standard-deviation guard: saved values below `1e-8` are replaced with `1.0` before dividing. The guide also identifies the saved factor fields `a` and `b`.

All 571 packaged research/source files passed hash checks: 518 original files, 44 supplementary remote files, and nine current producer/loader source snapshots. The two 32k SAE checkpoint hashes and dense ridge hash exactly match the original dashboard metadata. All 24 large supplementary file download URLs returned HTTP 200 to HEAD requests; no large checkpoint was loaded. The four-page supplement was visually checked and appended to the unchanged original report for the combined PDF.
