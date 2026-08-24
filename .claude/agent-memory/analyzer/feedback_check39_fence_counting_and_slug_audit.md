---
name: check39-fence-counting-and-slug-audit
description: check 39 counts fenced BLOCKS not rows (use a GFM table / list items for `Disclosure: N of M`); discipline audit flags multi-underscore slugs in Results/Takeaways prose even backticked
metadata:
  type: feedback
---

Two draft-time mechanical traps from #2163 (2026-08-07):

1. **`Disclosure: N of M` + one fenced block of N rows = check 39 FAIL.**
   `verify_task_body.py` check 39 counts presentation UNITS per disclosure
   group — fenced/`<details>` blocks, top-level list items, GFM table data
   rows, blockquote groups — never LINES inside a fence. A single fence
   holding 5 sample rows counts as 1, so `Disclosure: 5 of M` overclaims.
   **Fix:** render the N sample rows as a GFM table (N data rows) or N
   top-level list items; keep the `Disclosure:` line.
2. **`audit_clean_results_body_discipline.py` opaque_snake_slugs fires on
   multi-underscore covariate slugs in Results/Takeaways prose EVEN inside
   backticks** (e.g. `firing_freq_per_token`, `redundancy_max_cos`).
   Slugs in `## Methodology` prose/tables and figure text are tolerated
   (figure slugs → verifier WARN, acknowledgeable in body). **Fix:** gloss
   to plain English in Results/Takeaways prose ("per-token firing
   frequency"); keep slugs in the Methodology table + captions ack.

**Why:** both cost a bounce round if discovered at critic time; both are
5-second fixes at draft time.
**How to apply:** when the Sample slot shows row-shaped numeric/text
samples, default to a table; before Step 5, grep Results/Takeaways prose
for `[a-z]+_[a-z]+_[a-z]+` and gloss hits. See also
[[clean-result-critic-v1-checklist]].
