# Claude misses when the plan's HEADLINE decision statistic isn't produced

(Entry file created during the #1891 index curation — the index pointer was dangling. The full index hook is preserved below.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude misses when the plan's HEADLINE decision statistic isn't produced](feedback_claude_misses_headline_decision_statistic_not_produced.md) — Claude PASSes verifying OTHER plan items while never checking the pre-registered headline estimator is actually computed + persisted; grep the named estimator's call site. #841 r1 (affine-with-bias ridge) + r2 (paired bootstrap_delta_ci). FAIL.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude misses when the plan's HEADLINE decision statistic isn't produced](feedback_claude_misses_headline_decision_statistic_not_produced.md) — Claude PASSes verifying OTHER plan items while never checking the pre-registered headline estimator is actually computed + persisted; grep the named estimator's call site. #841 r1 (affine-with-bias ridge) + r2 (paired bootstrap_delta_ci). FAIL. COUNTER: #922 r2 — a local helper equal to the pinned estimator by linearity (same unit/seed/n_boot/CI, paired delta formed pre-resample; parent fn structurally inapplicable) is PASS; verify equivalence before upholding.
