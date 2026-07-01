"""Regression: ``_aggregate_persona_scores`` must not crash on a non-dict parse.

Root cause (#778 round 2): ``eval.utils.parse_judge_json`` returns
``json.loads(text)`` verbatim, so a scalar-shaped judge rubric (a bare integer,
or a persona-vectors ``{"score": N}`` the judge answered as the string ``"85"``)
parses to an ``int``. ``_aggregate_persona_scores`` (the Betley legacy aggregator
baked into ``judge_completions_batch``) then called ``s.get("aligned")`` on every
parsed entry, raising ``AttributeError: 'int' object has no attribute 'get'`` —
which aborted the whole ``judge_completions_batch`` call BEFORE ``save_raw`` was
written, so the scalar-rubric caller (issue778_lib.judge_graded) never got its
raw scores back. The fix is a type-guard: a non-dict entry is treated as invalid
at this aggregator (the scalar-rubric caller does its OWN reduction from
``all_scores`` and ignores this return).

These tests exercise ``_aggregate_persona_scores`` directly (the crash site),
which is the cleanest isolation of the guard. Betley (dict) callers are
unaffected — verified by the mixed-persona test.
"""

from __future__ import annotations

from explore_persona_space.eval.batch_judge import _aggregate_persona_scores


def _completions(persona_to_n: dict[str, int]) -> dict[str, dict[str, list[str]]]:
    """{persona: {question: [n dummy completions]}} — the shape the aggregator maps.

    One question per persona so the custom_id enumeration is 1:1 with the draws,
    matching how the scalar-rubric caller packs n_draws repeats as completions.
    """
    return {p: {f"q_{p}": ["answer"] * n} for p, n in persona_to_n.items()}


def _custom_ids(completions: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    """Reproduce the aggregator's custom_id scheme: {persona: [id per draw]}."""
    ids: dict[str, list[str]] = {}
    idx = 0
    for persona, q_comps in completions.items():
        ids[persona] = []
        for _q, comps in q_comps.items():
            for comp_idx in range(len(comps)):
                ids[persona].append(f"{persona}__{idx:05d}__{comp_idx:02d}")
            idx += 1
    return ids


def test_aggregate_persona_scores_skips_non_dict_parse():
    """A persona whose parsed scores are ALL bare ints must not raise.

    Pre-fix this raised ``AttributeError: 'int' object has no attribute 'get'``.
    Post-fix the bare-int draws are counted as invalid (n_samples=0, n_errors=N)
    and the aggregate means are None — the scalar-rubric caller ignores this
    return and reduces from all_scores itself.
    """
    completions = _completions({"item_pv": 3})
    ids = _custom_ids(completions)
    # The persona-vectors judge answered "85"/"90"/"REFUSAL" -> parse_judge_json
    # returns them as bare int / int / str (json.loads verbatim). Use bare ints
    # to reproduce the exact crashing entry class.
    all_scores = {
        ids["item_pv"][0]: 85,
        ids["item_pv"][1]: 90,
        ids["item_pv"][2]: 0,
    }

    result = _aggregate_persona_scores(completions, all_scores)  # must not raise

    agg = result["item_pv"]
    assert agg["n_samples"] == 0, agg
    assert agg["n_errors"] == 3, agg
    assert agg["mean_aligned"] is None, agg
    assert agg["mean_coherent"] is None, agg


def test_aggregate_persona_scores_mixed_dict_and_non_dict():
    """A persona mixing valid Betley dicts and a stray bare int aggregates the
    dicts and skips the int — n_samples counts only the valid dict rows."""
    completions = _completions({"item_mixed": 3})
    ids = _custom_ids(completions)
    all_scores = {
        ids["item_mixed"][0]: {"aligned": 80, "coherent": 90},
        ids["item_mixed"][1]: 60,  # bare-int scalar parse — must be skipped
        ids["item_mixed"][2]: {"aligned": 40, "coherent": 70},
    }

    result = _aggregate_persona_scores(completions, all_scores)  # must not raise

    agg = result["item_mixed"]
    assert agg["n_samples"] == 2, agg  # only the two dict rows are valid
    assert agg["n_errors"] == 1, agg  # the bare int counts as an error/skip
    assert agg["mean_aligned"] == (80 + 40) / 2, agg
    assert agg["mean_coherent"] == (90 + 70) / 2, agg


def test_aggregate_persona_scores_betley_dicts_unchanged():
    """Pure-dict (Betley) input aggregates exactly as before the guard — no
    behavior change for the existing Betley callers."""
    completions = _completions({"persona_a": 2})
    ids = _custom_ids(completions)
    all_scores = {
        ids["persona_a"][0]: {"aligned": 100, "coherent": 100},
        ids["persona_a"][1]: {"aligned": 50, "coherent": 80},
    }

    result = _aggregate_persona_scores(completions, all_scores)

    agg = result["persona_a"]
    assert agg["n_samples"] == 2, agg
    assert agg["n_errors"] == 0, agg
    assert agg["mean_aligned"] == 75.0, agg
    assert agg["mean_coherent"] == 90.0, agg


def test_aggregate_persona_scores_betley_sentinel_still_valid_not_scored():
    """The #545 sentinel path is unaffected: a dict with a "REFUSAL" string
    aligned (no error flag) still counts in n_samples but is excluded from the
    mean sum — the guard only adds a NON-dict skip, it does not touch this."""
    completions = _completions({"persona_s": 2})
    ids = _custom_ids(completions)
    all_scores = {
        ids["persona_s"][0]: {"aligned": "REFUSAL", "coherent": "REFUSAL"},
        ids["persona_s"][1]: {"aligned": 30, "coherent": 60},
    }

    result = _aggregate_persona_scores(completions, all_scores)

    agg = result["persona_s"]
    assert agg["n_samples"] == 2, agg  # both are valid verdicts (no error flag)
    assert agg["n_errors"] == 0, agg
    assert agg["mean_aligned"] == 30.0, agg  # only the numeric row enters the sum
    assert agg["mean_coherent"] == 60.0, agg
