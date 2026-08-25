# Claude misses when the plan's HEADLINE decision statistic isn't produced

(Entry file created during the #1891 index curation — the index pointer was dangling. The full index hook is preserved below.)

**#2544 r1 variant — machinery EXISTS but computes the WRONG selection rule; sibling contrast never computed.** Claude's split sub-reviewer verified "selection-inherited CIs re-argmax INSIDE each draw" (true) without checking WHOSE argmax: the plan registered ONE shared mean-over-rungs layer re-selection per draw (plan:150/:237(a)); the code did per-rung argmax — the plan's OTHER registered read (:237(b)) wearing the headline key's name. And the registered Δ_peak − Δ(main) selection-inherited CI (named 3× in the plan + a follow-up trigger) had ZERO grep hits while its per-draw inputs were computed and discarded in the same loop. Reconcile duties: (1) grep every REGISTERED contrast/CI by name AND by concept-synonym in the pinned analysis code; (2) when selection-inherited machinery exists, pin WHICH selection rule it re-runs against the plan's layer*/argmax definition; (3) recomputability from persisted SS does NOT rescue pinned pre-data analysis code (#491 r2 shape) — FAIL pre-production so the fix lands pre-data. Verdict: FAIL (Codex right on outcome; its layer-null-matrix + boundary-confidence sub-claims were mistaken — the per-draw × per-layer matrix and floor-sensitivity lattice WERE delivered).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude misses when the plan's HEADLINE decision statistic isn't produced](feedback_claude_misses_headline_decision_statistic_not_produced.md) — Claude PASSes verifying OTHER plan items while never checking the pre-registered headline estimator is actually computed + persisted; grep the named estimator's call site. #841 r1 (affine-with-bias ridge) + r2 (paired bootstrap_delta_ci). FAIL.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude misses when the plan's HEADLINE decision statistic isn't produced](feedback_claude_misses_headline_decision_statistic_not_produced.md) — Claude PASSes verifying OTHER plan items while never checking the pre-registered headline estimator is actually computed + persisted; grep the named estimator's call site. #841 r1 (affine-with-bias ridge) + r2 (paired bootstrap_delta_ci). FAIL. COUNTER: #922 r2 — a local helper equal to the pinned estimator by linearity (same unit/seed/n_boot/CI, paired delta formed pre-resample; parent fn structurally inapplicable) is PASS; verify equivalence before upholding.
