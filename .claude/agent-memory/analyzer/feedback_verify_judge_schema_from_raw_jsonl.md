---
name: verify-judge-schema-from-raw-jsonl
description: For any judge-categorical clean-result, pull the exact category-name strings from a live judged_*.jsonl row before drafting prose; aggregates and plans rename categories
metadata:
  type: feedback
---

For any clean-result whose headline DV is a judge categorical label, pull the exact category-name strings from ONE live raw `judged_*.jsonl` row before drafting. Aggregate JSONs may rename categories during roll-up; the plan describes them in plain English without the actual keys. Either is a trap.

**Why:** task #500 round 1 invented `taught / invented_canonical / refusal / distractor / real_descriptive` from the plan's description; the actual `output_category_5way` values were `stated_seven / stated_nine / didnt_mention / refused / confabulated_other`. Both critics flagged it as a hard FAIL.

**How to apply:**
1. `hf_hub_download` one `judged_*.jsonl`, read a row: confirm the verdict field name, the exact category-string set (`Counter` over the file), and what each category MEANS (sample 3 rows per category).
2. Use those exact strings in the Reproducibility table, prose, and sample-block `VERDICT:` annotations.
3. Any `Judge verdict: <label>` on a quoted sample must appear verbatim in the raw JSONL for that row; spot-check ≥1 quoted sample per finding.

Judge-schema mirror of [[feedback_read_eval_script_not_just_plan]].
