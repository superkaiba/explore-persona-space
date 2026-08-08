"""#2151: API-level ``stop_reason == "refusal"`` is the THIRD top-level judge
drop class — distinct from content drops (rule 9, incl. the instructed rubric
``REFUSAL`` of #1801) and transport losses (rule 24, #1313).

All CPU, no API. Predicate truth tables run against
``eval.batch_judge``; the reduce classification runs the REAL
:func:`graded_judge.judge_result_from_save_raw` on a hand-written ``save_raw``
JSON (the persisted shape the #1739 mint sites produce:
``{"error": True, "reasoning": "parse_error", "raw_text": "", "stop_reason":
"refusal"}``).
"""

from __future__ import annotations

import json
import logging

import pytest

from explore_persona_space.eval.batch_judge import (
    API_REFUSAL_STOP_REASONS,
    is_api_refusal_error_dict,
    is_api_refusal_stop_reason,
    is_transport_error_dict,
    is_truncation_error_dict,
)
from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

# ── canonical persisted shapes (mirror the judge_dispatch mint sites) ─────────

API_REFUSAL_DICT = {
    "aligned": None,
    "coherent": None,
    "reasoning": "parse_error",
    "error": True,
    "raw_text": "",
    "stop_reason": "refusal",
}
TRANSPORT_DICT = {
    "aligned": None,
    "coherent": None,
    "reasoning": "error: 529 overloaded",
    "error": True,
    "transport": True,
}
TRUNCATION_DICT = {
    "aligned": None,
    "coherent": None,
    "reasoning": "parse_error",
    "error": True,
    "stop_reason": "max_tokens",
}
LEGACY_PARSE_ERROR_DICT = {
    "aligned": None,
    "coherent": None,
    "reasoning": "parse_error",
    "error": True,
}


# ── predicate truth tables ────────────────────────────────────────────────────


def test_api_refusal_stop_reasons_tuple_pins_refusal_only():
    assert API_REFUSAL_STOP_REASONS == ("refusal",)


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("refusal", True),
        ("max_tokens", False),
        ("end_turn", False),
        ("stop_sequence", False),
        (None, False),
        (83399, False),  # non-str never KeyErrors, reads False
        ("", False),
        ("REFUSAL", False),  # raw SDK value is lowercase; no case-folding
    ],
)
def test_is_api_refusal_stop_reason_truth_table(stop_reason, expected):
    assert is_api_refusal_stop_reason(stop_reason) is expected


def test_is_api_refusal_error_dict_truth_table():
    # The persisted mint shape -> True.
    assert is_api_refusal_error_dict(API_REFUSAL_DICT) is True
    # A KEPT verdict carrying stop_reason "refusal" is NOT an error dict:
    # no error flag -> False (the draw produced a score; rule-9/26 semantics).
    assert is_api_refusal_error_dict({"score": 80, "stop_reason": "refusal"}) is False
    # Legacy dict without stop_reason -> False (classifies content, as before).
    assert is_api_refusal_error_dict(LEGACY_PARSE_ERROR_DICT) is False
    # Wrong stop_reason on an error dict -> False.
    assert is_api_refusal_error_dict(TRUNCATION_DICT) is False
    # Non-dict / None / str / bare numeric -> False, never a crash.
    assert is_api_refusal_error_dict(None) is False
    assert is_api_refusal_error_dict("refusal") is False
    assert is_api_refusal_error_dict(85) is False
    assert is_api_refusal_error_dict(["refusal"]) is False


def test_three_class_predicates_disjoint_on_canonical_mint_shapes():
    """Each canonical persisted shape matches EXACTLY ONE classifier —
    transport / truncation / api-refusal are siblings, never subsets."""
    rows = {
        "transport": TRANSPORT_DICT,
        "truncation": TRUNCATION_DICT,
        "api_refusal": API_REFUSAL_DICT,
    }
    preds = {
        "transport": is_transport_error_dict,
        "truncation": is_truncation_error_dict,
        "api_refusal": is_api_refusal_error_dict,
    }
    for row_name, row in rows.items():
        matches = [p for p, fn in preds.items() if fn(row)]
        assert matches == [row_name], f"{row_name} matched {matches}"
    # The legacy no-stop_reason parse error matches NONE (content drop).
    assert not any(fn(LEGACY_PARSE_ERROR_DICT) for fn in preds.values())


# ── reduce classification (real judge_result_from_save_raw, no fakes) ─────────


def _write_save_raw(tmp_path, draws: list[object]):
    """Persist draws for one item under the batch_judge custom_id scheme."""
    all_scores = {f"item-a__00000__{i:02d}": parsed for i, parsed in enumerate(draws)}
    save_raw = tmp_path / "raw.json"
    save_raw.write_text(json.dumps({"per_persona": {}, "all_scores": all_scores}))
    return save_raw


def test_reduce_classifies_api_refusal_as_third_class(tmp_path, caplog):
    """One draw per class: kept / api-refusal / transport / truncation /
    instructed-REFUSAL. The api-refusal draw lands ONLY in
    ``n_api_refusal_draws`` + ``per_item_api_refusals`` — not in
    ``n_dropped_draws`` (or its refusal/truncation subsets) and not in
    ``n_transport_lost_draws`` — while the raw ``stop_reason_tally`` census
    still counts its ``"refusal"`` row."""
    draws = [
        {"score": 80, "stop_reason": "end_turn"},  # kept
        API_REFUSAL_DICT,  # third class (#2151)
        TRANSPORT_DICT,  # rule-24 loss
        TRUNCATION_DICT,  # rule-23 content-drop subset
        {"score": "REFUSAL", "stop_reason": "end_turn"},  # instructed (#1801)
    ]
    save_raw = _write_save_raw(tmp_path, draws)

    with caplog.at_level(logging.WARNING, logger="explore_persona_space.eval.graded_judge"):
        result = judge_result_from_save_raw(save_raw, items=[("item-a", "q?", "a.")])

    assert result.n_total_draws == 5
    assert result.scores == {"item-a": 80.0}
    # THIRD class: counted alone, per-item populated.
    assert result.n_api_refusal_draws == 1
    assert result.per_item_api_refusals == {"item-a": 1}
    # Content drops: truncation + instructed REFUSAL only (api-refusal NOT blended).
    assert result.n_dropped_draws == 2
    assert result.n_refusal_draws == 1  # instructed rubric REFUSAL (#1801)
    assert result.n_truncation_dropped_draws == 1
    # Transport: its own sibling counter, unchanged.
    assert result.n_transport_lost_draws == 1
    # stop_reason census: transport rows excluded (no API response); the
    # api-refusal draw's "refusal" IS tallied (raw census semantics).
    assert result.stop_reason_tally == {"end_turn": 2, "refusal": 1, "max_tokens": 1}
    # The reduce WARNs, naming the class + remediation rule.
    assert any("API-refusal" in rec.message and "rule 28" in rec.message for rec in caplog.records)


def test_reduce_transport_takes_precedence_over_api_refusal(tmp_path):
    """A pathological both-flagged dict (transport: True AND stop_reason
    "refusal") classifies TRANSPORT — the reduce checks transport FIRST, and
    transport rows never enter the stop_reason tally."""
    both = {**TRANSPORT_DICT, "stop_reason": "refusal"}
    save_raw = _write_save_raw(tmp_path, [both, {"score": 60, "stop_reason": "end_turn"}])

    result = judge_result_from_save_raw(save_raw, items=[("item-a", "q?", "a.")])

    assert result.n_transport_lost_draws == 1
    assert result.n_api_refusal_draws == 0
    assert result.per_item_api_refusals == {}
    assert result.stop_reason_tally == {"end_turn": 1}


def test_reduce_kept_verdict_with_refusal_stop_reason_stays_kept(tmp_path):
    """A draw that produced a numeric score with stop_reason "refusal" (not an
    error dict) is KEPT — the class keys on the ERROR dict, never the bare
    stop_reason."""
    save_raw = _write_save_raw(tmp_path, [{"score": 42, "stop_reason": "refusal"}])

    result = judge_result_from_save_raw(save_raw, items=[("item-a", "q?", "a.")])

    assert result.scores == {"item-a": 42.0}
    assert result.n_api_refusal_draws == 0
    assert result.n_dropped_draws == 0
    assert result.stop_reason_tally == {"refusal": 1}


def test_reduce_legacy_no_stop_reason_dict_stays_content_drop(tmp_path):
    """Pre-#2021 legacy error dicts (no stop_reason) keep classifying as plain
    content drops — never api-refusal."""
    save_raw = _write_save_raw(tmp_path, [LEGACY_PARSE_ERROR_DICT, 70])

    result = judge_result_from_save_raw(save_raw, items=[("item-a", "q?", "a.")])

    assert result.scores == {"item-a": 70.0}
    assert result.n_dropped_draws == 1
    assert result.n_api_refusal_draws == 0
    assert result.stop_reason_tally == {"unknown": 2}
