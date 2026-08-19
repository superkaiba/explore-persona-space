---
name: named-tests-ban-hits-tables
description: discipline audit named_tests class scans UNBLANKED text — "Wilcoxon" FAILs even inside a Methodology GFM table row; write "exact paired signed-rank test"
metadata:
  type: feedback
---

The `audit_clean_results_body_discipline.py` `named_tests` class (paired
t-test / Fisher exact / Mann-Whitney / Wilcoxon / bootstrap test) scans the
UNBLANKED body — the GFM table-row exemption applies only to
`interval_inline` / `condition_labels` (see `_TABLE_CELL_EXEMPT_CATEGORIES`).
A "Wilcoxon signed-rank" entry in the Methodology hyperparameter table FAILed
the audit on #2333 round 1.

**Why:** the prose-vs-table distinction is deliberately not load-bearing for
named_tests (audit_body docstring); Lens 7 wants tests described, not named.

**How to apply:** everywhere in a clean-result body (tables included), write
the unnamed form — "exact two-sided paired signed-rank test", "corrected p" —
and keep the canonical test name only in the plan / analysis-script docstring
the Source column points to. Same round: the sample-block prelude must carry a
literal "cherry-picked" / "random sample" token (Cherry-picked label
discipline check) — "random within strata, seed 42" alone FAILs; and the
Lens-14 concerns ack works as backticked concern-id slugs inside a
`<details>` block within a `### <result>` (ids are substring-matched against
`## Results` H3 bodies + `## Takeaways`).
