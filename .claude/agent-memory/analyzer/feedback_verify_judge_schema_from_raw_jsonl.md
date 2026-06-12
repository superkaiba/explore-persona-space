---
name: verify-judge-schema-from-raw-jsonl
description: For any judge-categorical clean-result body, the category names in the prose+table MUST be pulled from a live raw judged_*.jsonl row, not inferred from a plan or aggregate JSON; inventing label names is a hard reproducibility error
metadata:
  type: feedback
---

For any clean-result body whose headline DV is a Claude-judge categorical
(N-way) label, the exact category-name strings MUST be pulled from a live
read of one raw `judged_*.jsonl` row before drafting the prose. Aggregate
JSONs (`aggregate_cleaned.json`, `predictors.json`, `full_eval_summary.json`)
may rename categories during roll-up; the plan may describe the categories
in plain English without using the actual string keys. Either is a trap.

**Why:** Task #500 round 1 invented category names `taught /
invented_canonical / refusal / distractor / real_descriptive` based on the
plan's plain-English description, but the actual `output_category_5way`
values in `judged_5way_*.jsonl` are `stated_seven / stated_nine /
didnt_mention / refused / confabulated_other`. Anyone trying to read the
raw JSONL to verify the sample-output blocks would have been lost. Both
the Claude and Codex critics flagged it as a hard FAIL.

**How to apply:**
1. Before writing the headline finding's prose, run:
   ```bash
   uv run python -c "
   from huggingface_hub import hf_hub_download
   import json
   p = hf_hub_download(repo_id='superkaiba1/explore-persona-space-data',
                       repo_type='dataset',
                       filename='issue<N>_<slug>/<arm>/<topic>/raw_completions/judged_5way_<...>.jsonl')
   r = json.loads(open(p).readline())
   print(list(r.keys()), r.get('verdict'))
   "
   ```
   Confirm: (a) the verdict field name, (b) the exact set of category
   strings (look at `Counter(r['verdict'][KEY] for r in rows)` over the
   whole file), (c) what each category MEANS by sampling 3 rows per
   category.
2. Use those exact strings in the Reproducibility table, the prose, and
   the sample-output block's `VERDICT:` annotations.
3. If a sample-output block claims `Judge verdict: <label>`, that label
   MUST appear verbatim in the raw JSONL for that row's
   `(persona, family, sub_framing)` key. Spot-check ≥1 quoted sample per
   finding against the raw row.

This is the judge-schema mirror of [[feedback_read_eval_script_not_just_plan]]
(don't trust the plan's framing of what was measured; read the actual
output).
