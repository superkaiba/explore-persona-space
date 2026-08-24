# Codex REVISEs a correct numeric claim via wrong statistic / JSON-only artifact search

(Entry file created during the #1891 index curation — the index pointer was dangling. The full index hook is preserved below.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Codex REVISEs a correct numeric claim via wrong statistic / JSON-only artifact search](feedback_codex_fails_correct_numeric_claim_wrong_statistic_or_artifact.md) — multi-number simultaneous match pins the statistic; search .pt artifacts the Repro footer names; self-scoping parentheticals defeat "overclaim". #920 r2 PASS.

## #2333 r2 instance (2026-08-18): pair-mean vs draw-pooled aggregation

Codex flagged a table cell "0.41" as an error, having reproduced the
unweighted pair-mean 0.4049 → 0.40 from `f_cells.jsonl`. The draw-pooled
(n_scored-weighted) mean of the SAME 180 pairs is 0.4061 → 0.41, and the
pooled convention made ALL EIGHT table cells cohere simultaneously (the
disputed cell was the only one where the two aggregations differ at two
decimals). Verdict: dissolved — the multi-cell simultaneous match pins the
table's statistic. **How to apply:** before upholding a "wrong table value"
blocker, test BOTH pair-mean and draw-pooled (and cell-mean) aggregations
against EVERY sibling cell in the same table row family; uphold only if no
single convention makes them all cohere. Note the flip side: a table using
pooled while the sibling `stats.json` headline uses pair-mean is worth a
Standing-only "state the aggregation" note, never a blocker.
