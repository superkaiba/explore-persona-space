# Issue #238 — tracked persona / question assets

This directory makes the issue-238 experiment self-contained. Without these
files the orchestrator depends on `data/assistant_axis/` (which is gitignored
and was sourced from `lu-christina/assistant-axis-vectors` plus hand-authored
`instructions/*.json` files). Round 1 crashed because 9 of 12 EVAL_PERSONAS
were not present in `data/assistant_axis/role_list.json`.

## Files

- **`personas.json`** — byte-exact copy of `data/issue_205/personas.json`
  (commit `c185709`). Contains the 12 eval personas with the exact prompt
  strings #205 used. Apples-to-apples comparison with #205's M1-deltas
  REQUIRES these strings remain byte-identical.
- **`extraction_questions.jsonl`** — byte-exact copy of
  `data/assistant_axis/extraction_questions.jsonl` (md5
  `a1c94e4a44a6b155a987638442b4ca35`, 240 entries). Same questions #205
  used, so the per-question forward-pass set is matched.

## How they're consumed

`scripts/run_issue238_orchestrator.py` passes:

```
extract_persona_vectors.py \
  --inline-personas-json data/issue_238/personas.json \
  --questions-file       data/issue_238/extraction_questions.jsonl \
  ...
```

The two new flags on `extract_persona_vectors.py` (added in this issue)
bypass the `data/assistant_axis/{role_list.json, instructions/, extraction_questions.jsonl}`
load path entirely. No content from `data/assistant_axis/` is read when
both flags are set.

## Gitignore

`data/` is gitignored project-wide; the `.gitignore` adds explicit
`!data/issue_238/` and `!data/issue_238/**` negations so this directory is
tracked. Files were `git add -f`-ed when first committed.
