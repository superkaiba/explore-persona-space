---
name: rule24-surgical-rejudge-recipe
description: Recovering transport-529 judge draws from persisted judge_raw files (fu4 pattern) — grouping, cache, and the deterministic parse_error trap
type: feedback
---

Recipe for a rule-24 transport-loss recovery over `organisms._rate_for_cell` /
`judge_graded` judge_raw files (#1090 fu4, #1315 follow-up — now a recurring
pattern).

**Why:** the pre-#1313 `api_dispatch._is_transient` misses 529 OverloadedError,
so runs persist `{"error": True, "reasoning": "error: Error code: 529 ..."}`
rows in `save_raw.all_scores`; those draws are freely re-judgeable but the
rubric-keyed cache would re-serve the stored errors AND collapse an item's
repeats to one score.

**How to apply:**
- Item-id grammar: `{context_id}-{side}-q{i:03d}-c{j}`; custom_id =
  `{item_id}__{idx:05d}__{comp_idx:02d}` (rsplit `"__", 2`). (q, answer) come
  from the sibling `completions__{side}__{ctx}.json` (`questions` +
  `completions` keys).
- Re-dispatch by grouping items by k_lost and calling `judge_graded(items,
  rubric, n_draws=k, ...)` per group — the run's own instrument verbatim, no
  reimplementation. FRESH `cache_dir` + `save_raw` per (pass, k) call. Bounded
  re-dispatch loop over rows still `error: True` (until #1313 is in the tree).
- Hard-assert the recomputed as-scored rate equals the committed record before
  merging (validates the reduce replication; caught a wrong-rung staging bug
  in #1315 before any API call).
- Trap: `parse_error` rows can be DETERMINISTIC per-item (7/100 WildChat items
  in #1315 parse_error'd on every original draw, every re-judge draw, AND a
  max_tokens=1000 rule-23 budget probe — 14/14). Not truncation: the judge
  emits an unparseable refusal-shaped response for those specific completions.
  Classify content-class, keep censored, record a diagnostic artifact — never
  chase them with same-budget retries.
- `keep_raw_judge_text()` does NOT attach raw text on a None parse
  (`_parsed_with_raw` passes None through), so parse_error rows stay opaque —
  use the budget probe, not raw-text retention, to discriminate truncation.
