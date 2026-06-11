---
name: Judge JSON numeric fields are not type-guaranteed
description: LLM-judge JSON replies sometimes carry string-typed numerics ("aligned": "85") or refusal words; coerce to int at parse time BEFORE storing into score dicts that downstream sums consume.
type: feedback
---

LLM judges (GPT-4o Betley dual, Sonnet rubrics) occasionally emit numeric
fields as STRINGS (`"aligned": "85"`) or non-numeric refusal words, and
`parse_judge_json`-style helpers are bare `json.loads` — no type policy.
Storing the raw parsed dict and validating later is the crash shape: the
stored row carries no `error` flag, passes `not s.get("error")` filters,
and a downstream `sum(...)` dies with `int + str`.

**Why:** Task #545 round 13 (2026-06-11) — `_score_b1_openai_gpt4o`
stored `parsed` before the `float()` check; 8/25 production cells crashed
deterministically at the save_raw mean computation.

**How to apply:** In any judge-score parse loop: coerce each numeric field
(`int(float(v))`, catch TypeError/ValueError/OverflowError, range-check to
the rubric scale) BEFORE the dict enters `all_scores`/results; failures
become tracked `error: True` rows that existing aggregations exclude.
Never validate after storing. Coercion (not error-marking) is right for
string-NUMERICS — they are usable scores.
