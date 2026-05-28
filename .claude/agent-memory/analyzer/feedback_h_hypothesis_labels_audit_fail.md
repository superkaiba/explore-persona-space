---
name: h-hypothesis-labels-trigger-condition-labels-audit
description: audit_clean_results_body_discipline.py flags any token like H1/H2/H3/H4/P1/P2/P3 anywhere in the body as condition_labels FAIL; rewrite project-internal hypothesis-label references to plain English before set-body
metadata:
  type: feedback
---

The body-discipline audit (`scripts/audit_clean_results_body_discipline.py`)
has a `condition_labels` rule that fires on bare tokens `H1`, `H2`, `H3`,
`H4`, `P1`, `P2`, `P3` — the 6 project-internal hypothesis/condition
codes — anywhere in the body. The intent is to force plain-English condition
names end to end, but the same rule kills incidental references to internal
classifiers / pass-criteria / planning-document hypothesis numbers.

**Why:** Task #390, user-directed addition round. I introduced "the H4
classifier used for the framing-#1 strict test" in the new H3 — referencing
the project-internal name of the keyword-bucket classifier the experiment
implemented. Audit failed with `condition_labels: 'H4'`. The cosmetic fix:
"the keyword-bucket classifier used for the framing-#1 strict test."

**How to apply:** Before `set-body`, grep the cache file for the 6 tokens
`\bH[1-4]\b` / `\bP[1-3]\b` and rewrite each reference to plain English:

- "H4 classifier" → "keyword-bucket classifier" / "refusal-vs-leak classifier"
- "the H1 hypothesis" → "the hypothesis that..." (state the hypothesis content)
- "H_main" → "the main hypothesis" or the actual claim
- "P1 condition" → the plain-English condition name (the planner.md § 5
  requirement covers this; the audit just enforces it)

Same applies to `Method A/B/C`, `Bin A/B/C`, `C1/C2/C3`, `arm`-as-noun —
all flagged. The audit list lives in `scripts/audit_clean_results_body_discipline.py`
in the `CONDITION_LABEL_PATTERNS` constant; check it when adding new prose
that touches internal nomenclature.
