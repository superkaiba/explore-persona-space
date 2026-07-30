"""CPU tests for ``eval.graded_judge`` (#851 promotion of issue778_lib.judge_graded).

All CPU, no API: ``judge_completions_batch`` is faked via monkeypatch on the
``explore_persona_space.eval.batch_judge`` module attribute (the same seam the
pre-existing ``test_judge_graded_carries_bare_int_score`` uses — preserved by
the library's call-time attribute access), writing a synthetic ``save_raw``
shaped exactly like the real #663-hardened client's output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.eval import graded_judge
from explore_persona_space.eval.graded_judge import JudgeResult, _score_from_parsed, judge_graded


def _custom_ids(completions: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    """Reproduce batch_judge._enumerate_and_check_cache's custom_id scheme."""
    ids: dict[str, list[str]] = {}
    idx = 0
    for persona, q_comps in completions.items():
        ids[persona] = []
        for _q, comps in q_comps.items():
            for comp_idx in range(len(comps)):
                ids[persona].append(f"{persona}__{idx:05d}__{comp_idx:02d}")
            idx += 1
    return ids


def _fake_batch_writing(draws_by_item: dict[str, list[object]]):
    """A judge_completions_batch fake writing all_scores from per-item draw lists."""

    def _fake(*, completions, save_raw, dry_run, **kwargs):
        assert not dry_run
        ids = _custom_ids(completions)
        all_scores = {}
        for item_id, cids in ids.items():
            for cid, parsed in zip(cids, draws_by_item[item_id], strict=True):
                all_scores[cid] = parsed
        payload = {"per_persona": {}, "all_scores": all_scores}
        Path(save_raw).parent.mkdir(parents=True, exist_ok=True)
        with open(save_raw, "w") as f:
            json.dump(payload, f)

    return _fake


def test_score_from_parsed_drop_never_coerce():
    """Library-path mirror of the null-battery drop-never-coerce suite."""
    # Dropped (None), never coerced:
    assert _score_from_parsed({"score": "REFUSAL"}) is None
    assert _score_from_parsed({"score": 150}) is None  # out of range
    assert _score_from_parsed({"score": -5}) is None
    assert _score_from_parsed({"score": True}) is None  # bool is malformed
    assert _score_from_parsed({"error": True, "score": 50}) is None
    assert _score_from_parsed("banana") is None  # non-dict, non-numeric
    assert _score_from_parsed({"score": "banana"}) is None
    # Carried:
    assert _score_from_parsed(85) == 85.0  # bare int (envelope-less judge)
    assert _score_from_parsed({"score": "42"}) == 42.0  # stringified number
    assert _score_from_parsed(0) == 0.0  # boundaries carried
    assert _score_from_parsed(100) == 100.0
    assert _score_from_parsed({"score": 0}) == 0.0
    assert _score_from_parsed({"score": 100}) == 100.0


def test_judge_graded_multidraw_mean_and_per_arm_drop_counts(tmp_path, monkeypatch):
    """Two items x 3 draws: kept-draw means + aggregate/per-item drop telemetry."""
    draws = {
        "pos-arm": [80, 90, {"score": "REFUSAL"}],  # mean over kept = 85.0
        "neg-arm": [{"score": 20}, {"score": 999}, {"score": 40}],  # 999 out of range -> 30.0
    }
    monkeypatch.setattr(
        "explore_persona_space.eval.batch_judge.judge_completions_batch",
        _fake_batch_writing(draws),
    )

    result = judge_graded(
        items=[("pos-arm", "q1?", "a1."), ("neg-arm", "q2?", "a2.")],
        eval_prompt="Rate {question} / {answer} 0-100.",
        n_draws=3,
        cache_dir=tmp_path / "cache",
        save_raw=tmp_path / "raw.json",
    )

    assert result.scores == {"pos-arm": 85.0, "neg-arm": 30.0}
    assert result.n_total_draws == 6
    assert result.n_dropped_draws == 2
    # Per-arm drops = n_draws - kept (the reporting contract).
    assert result.per_item_draw_counts == {"pos-arm": 2, "neg-arm": 2}


def test_judge_graded_all_draws_dropped_scores_none(tmp_path, monkeypatch):
    draws = {"item-x": [{"score": "REFUSAL"}] * 3}
    monkeypatch.setattr(
        "explore_persona_space.eval.batch_judge.judge_completions_batch",
        _fake_batch_writing(draws),
    )

    result = judge_graded(
        items=[("item-x", "q?", "a.")],
        eval_prompt="Rate {question} / {answer} 0-100.",
        n_draws=3,
        cache_dir=tmp_path / "cache",
        save_raw=tmp_path / "raw.json",
    )

    assert result.scores["item-x"] is None
    assert result.per_item_draw_counts["item-x"] == 0
    assert result.n_total_draws == 3
    assert result.n_dropped_draws == 3


def test_judge_graded_rejects_item_id_with_delimiter(tmp_path, monkeypatch):
    """The '__' guard fires BEFORE any dispatch (library-boundary contract)."""

    def _must_not_be_called(**kwargs):
        raise AssertionError("judge_completions_batch must not be reached")

    monkeypatch.setattr(
        "explore_persona_space.eval.batch_judge.judge_completions_batch",
        _must_not_be_called,
    )

    with pytest.raises(ValueError, match="__"):
        judge_graded(
            items=[("a__b", "q?", "a.")],
            eval_prompt="Rate {question} / {answer} 0-100.",
            n_draws=1,
            cache_dir=tmp_path / "cache",
            save_raw=tmp_path / "raw.json",
        )


def test_judge_graded_dry_run_returns_empty(tmp_path, monkeypatch):
    """dry_run=True: empty JudgeResult, dry_run forwarded, save_raw never read."""
    seen = {}

    def _fake(*, completions, save_raw, dry_run, **kwargs):
        seen["dry_run"] = dry_run  # write NOTHING — a save_raw read would crash

    monkeypatch.setattr("explore_persona_space.eval.batch_judge.judge_completions_batch", _fake)

    result = judge_graded(
        items=[("item-x", "q?", "a.")],
        eval_prompt="Rate {question} / {answer} 0-100.",
        n_draws=2,
        cache_dir=tmp_path / "cache",
        save_raw=tmp_path / "raw.json",  # never written, never read
        dry_run=True,
    )

    assert seen["dry_run"] is True
    assert result == JudgeResult(scores={}, n_total_draws=0, n_dropped_draws=0)
    assert result.per_item_draw_counts == {}


def test_issue778_lib_reexports_library_objects():
    """Pins the supersede->dedupe rewire: issue778_lib re-exports the SAME objects."""
    import scripts.issue778_lib as lib

    assert lib.judge_graded is graded_judge.judge_graded
    assert lib.JudgeResult is graded_judge.JudgeResult
    assert lib._score_from_parsed is graded_judge._score_from_parsed


# ── #1313: transport-vs-content split (llm-judging.md rules 9/24) ─────────────


def test_transport_error_split_from_content_drops(tmp_path, monkeypatch):
    """#1313 rule 24(ii): a transport-flagged draw lands in
    ``n_transport_lost_draws`` (+ the per-item map); a content drop (REFUSAL)
    stays in ``n_dropped_draws``; a good score is kept — never blended.

    Mutation check: the pre-#1313 tally folds the transport row into
    ``n_dropped_draws`` -> ``n_dropped_draws == 2`` + missing transport counter
    -> assertions red."""
    transport_dict = {
        "aligned": None,
        "coherent": None,
        "reasoning": "error: 529 overloaded",
        "error": True,
        "transport": True,
    }
    draws = {"arm-a": [85, {"score": "REFUSAL"}, transport_dict]}
    monkeypatch.setattr(
        "explore_persona_space.eval.batch_judge.judge_completions_batch",
        _fake_batch_writing(draws),
    )

    result = judge_graded(
        items=[("arm-a", "q?", "a.")],
        eval_prompt="Rate {question} / {answer} 0-100.",
        n_draws=3,
        cache_dir=tmp_path / "cache",
        save_raw=tmp_path / "raw.json",
    )

    assert result.scores == {"arm-a": 85.0}
    assert result.n_total_draws == 3
    assert result.n_dropped_draws == 1  # the REFUSAL only (content, rule 9)
    assert result.n_transport_lost_draws == 1  # the flagged transport row (rule 24)
    assert result.per_item_transport_losses == {"arm-a": 1}
    assert result.per_item_draw_counts == {"arm-a": 1}


def test_legacy_reason_string_classified_transport(tmp_path, monkeypatch):
    """A LEGACY persisted dict (no ``transport`` flag) with reason
    ``batch_error: expired`` is classified transport via the conservative
    reason-string fallback (#1313: pre-existing save_raw files self-classify)."""
    legacy = {
        "aligned": None,
        "coherent": None,
        "reasoning": "batch_error: expired",
        "error": True,
    }
    draws = {"arm-a": [legacy, 70]}
    monkeypatch.setattr(
        "explore_persona_space.eval.batch_judge.judge_completions_batch",
        _fake_batch_writing(draws),
    )

    result = judge_graded(
        items=[("arm-a", "q?", "a.")],
        eval_prompt="Rate {question} / {answer} 0-100.",
        n_draws=2,
        cache_dir=tmp_path / "cache",
        save_raw=tmp_path / "raw.json",
    )

    assert result.scores == {"arm-a": 70.0}
    assert result.n_transport_lost_draws == 1
    assert result.n_dropped_draws == 0
    assert result.per_item_transport_losses == {"arm-a": 1}


def test_judge_result_backcompat_defaults():
    """#1313 acceptance item 5: old-kwarg ``JudgeResult`` construction still
    works — the new transport fields default to zero/empty."""
    r = JudgeResult(scores={"x": 1.0}, n_total_draws=1, n_dropped_draws=0)
    assert r.n_transport_lost_draws == 0
    assert r.per_item_transport_losses == {}
