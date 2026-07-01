---
name: batch_judge aggregator crashes on bare-int parse from a scalar-shaped judge rubric
description: eval.utils.parse_judge_json returns json.loads(text) verbatim; a persona-vectors rubric answered as "85" yields a bare int, and the Betley legacy aggregator baked into judge_completions_batch (_aggregate_persona_scores) crashes on s.get("aligned"). Any NON-Betley caller must reduce from save_raw's all_scores themselves AND the aggregator needs an isinstance(s, dict) guard
type: feedback
---

`eval.utils.parse_judge_json(text, None)` returns `json.loads(text)` verbatim.
For the Betley dual-rubric shape it returns a dict:
`{"aligned": 85, "coherent": 90}`. For the persona-vectors rubric (bare
integer 0-100, or `{"score": N}` answered as the string `"85"`), it
returns a bare `int`. The Betley legacy aggregator in
`_aggregate_persona_scores` (`src/explore_persona_space/eval/batch_judge.py`
~L428) unconditionally calls `s.get("aligned")` on every parsed entry
inside `judge_completions_batch` → `AttributeError: 'int' object has no
attribute 'get'`. This crashes the WHOLE call BEFORE `save_raw` is
written, so a scalar-rubric caller that reduces `all_scores` itself
never even gets its raw scores back.

Anyone adding a NON-Betley judge (persona-vectors, single-score
rubric, anything without `{"aligned":..,"coherent":..}`) must do both:

**(1) Reduce from `save_raw` yourself**, don't rely on the Betley
`mean_aligned`/`mean_coherent` returns. `judge_completions_batch`
writes the raw per-item parsed dict to `save_raw`'s `all_scores` key;
your caller opens that JSON and reduces per your rubric shape (see
`scripts/issue778_lib.py::judge_graded` — reads `all_scores`, runs
`_score_from_parsed` which handles bare int + `{"score": N}` + REFUSAL
sentinels, per-item mean over N judge draws, drop-never-coerce +
per-item dropped count).

**(2) Verify the aggregator's type-guard is present.** The
`isinstance(s, dict)` guard around the `valid` filter (fb3da7045e,
#778 r2) is what stops the AttributeError. Any future refactor of
`_aggregate_persona_scores` MUST preserve the guard — dropping it
re-opens the crash.

**Why:** the Betley schema (dict with `aligned`/`coherent`/`error`) is
hard-wired into the shared aggregator. `judge_completions_batch` is
sanctioned for scalar-rubric callers TOO by CLAUDE.md ("LLM judge =
claude-sonnet-4-5-20250929, ALWAYS, ... use the Anthropic Batch API
whenever the judge set is large" + `.claude/rules/llm-judging.md`
rule 12 pinning the single client), so the aggregator has to survive
scalar parses.

**How to apply (writing a new NON-Betley judge caller):**

1. Design the rubric to emit a scalar or `{"score": N}` (per
   `llm-judging.md` rule 6, anchored 0-100).
2. Wrap the completion loop:
   ```python
   judge_completions_batch(
       completions=..., judge_system_prompt=..., format_user_msg=...,
       judge_model=DEFAULT_JUDGE_MODEL, save_raw=Path(cache_dir) / "raw.json",
       ...  # let the Betley aggregation happen — it will return empty
            # means for your persona, and that's fine
   )
   with open(cache_dir / "raw.json") as f:
       raw = json.load(f)
   all_scores = raw["all_scores"]   # THE authoritative per-item source
   # reduce all_scores yourself: iterate, drop non-numeric / REFUSAL /
   # out-of-[0,100] per rule 9, mean over per-item draws, report the
   # per-arm dropped count.
   ```
3. If your regression test asserts the aggregator does NOT crash on a
   bare-int parse, keep the pre-fix reproducer around (source-only
   stash / a fake `all_scores = {"cid__00000__00": 85}`) so a future
   refactor that drops the guard is caught by CI, not the next
   experiment relaunch (`tests/test_batch_judge_agg_non_dict_parse.py`
   is the canonical form).

**Do NOT:** wrap the scalar in a fake Betley dict
(`{"aligned": val, "coherent": 50, "error": False}`) just to keep the
aggregator quiet — that silently double-counts your scalar in the
`mean_aligned` return, which the Betley callers' downstream code may
then read as if it were a real Betley eval. The type-guard + own-side
reduction is the honest fix.

Closed regressions: task #778 r2 (2026-07-01, Phase 2 monitoring crash
on the graded 0-100 judge — root cause confirmed by direct traceback +
code inspection; fix landed at commit fb3da7045e with a matching
regression test).
