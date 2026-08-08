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

   **AND — your reducer MUST accept a bare `int|float` directly, not just
   `{"score": N}` dicts** (#778 r3 addendum, fix landed at a2b019934e).
   `parse_judge_json` returns `json.loads(text)` verbatim, so a judge that
   answers `"85"` (just the number) parses to a Python `int` — the rubric's
   requested `{"score": N}` envelope is often OMITTED by the judge. If
   your reducer's first guard is `if not isinstance(parsed, dict): return
   None`, every envelope-less judge response is silently counted as
   dropped and the graded DV loses coverage. Correct shape:

   ```python
   def _score_from_parsed(parsed) -> float | None:
       # Bool BEFORE numeric — isinstance(True, int) is True; a judge
       # emitting `true`/`false` is not a valid score.
       if isinstance(parsed, bool):
           return None
       # Bare int/float in [0,100] is a valid off-spec judge score.
       if isinstance(parsed, (int, float)):
           val = float(parsed)
           return val if 0.0 <= val <= 100.0 else None
       # Dict envelope path — {"score": N} + REFUSAL sentinel + oor.
       if not isinstance(parsed, dict): return None
       # ... existing dict handling ...
   ```
   Regression test: assert `_score_from_parsed(85) == 85.0`,
   `_score_from_parsed(True) is None`, `_score_from_parsed(150) is None`.
   AND a production-path fake-batch-client test where raw judge text is
   `"85"` and the returned per-item score is 85.0, `n_dropped_draws == 0`
   (see `tests/test_batch_judge_agg_non_dict_parse.py::test_judge_graded_carries_bare_int_score`
   for the canonical form).
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

Closed regressions:
- task #778 r2 (2026-07-01, Phase 2 monitoring crash on the graded 0-100
  judge, root cause confirmed by direct traceback + code inspection; fix
  landed at fb3da7045e — aggregator `isinstance(s, dict)` guard).
- task #778 r3 (2026-07-01, follow-up after Codex/reconciler upheld FAIL
  on `_score_from_parsed` dropping bare-int scores; fix landed at
  a2b019934e — caller-side reducer accepts non-bool `int|float` in
  `[0, 100]`).

Companion tests:
- `tests/test_batch_judge_agg_non_dict_parse.py::test_aggregate_persona_scores_non_dict_skip` (r2 aggregator guard).
- `tests/test_batch_judge_agg_non_dict_parse.py::test_judge_graded_carries_bare_int_score` (r3 production-path bare-int).
- `tests/test_issue778_null_battery.py::test_score_from_parsed_accepts_bare_int_in_range` (+ bare-float, out-of-range, bool-disguised).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [batch_judge aggregator crashes on bare-int parse from a scalar-shaped judge rubric](feedback_batch_judge_aggregator_bare_int_parse.md) — `eval.utils.parse_judge_json` returns `json.loads(text)` verbatim; a persona-vectors rubric answered as `"85"` yields a bare `int`, and the Betley legacy aggregator in `_aggregate_persona_scores` crashes on `s.get("aligned")` — BEFORE `save_raw` is written, so a scalar-rubric caller that reduces `all_scores` itself never even gets its data back. Any NON-Betley `judge_completions_batch` caller must reduce from `save_raw`'s `all_scores` itself AND the aggregator needs an `isinstance(s, dict)` guard (fix landed at fb3da7045e). #778 r2.
