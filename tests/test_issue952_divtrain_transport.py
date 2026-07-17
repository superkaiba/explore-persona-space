"""#1470: rule-24 caller-side helper tests for the #952 generation dispatchers.

Imports ``scripts.issue952_transport_helpers`` ONLY — the pure helper module —
never the vLLM-heavy ``issue952_divtrain_gpu.py`` (which sets vLLM env and
imports the full #952 experiment stack at module top).
"""

from __future__ import annotations

import json

from explore_persona_space.llm.api_dispatch import (
    RESULT_ERROR,
    RESULT_OK,
    RESULT_RATE_LIMITED,
    RESULT_TRANSPORT,
    DispatchResult,
)
from scripts.issue952_transport_helpers import (
    REDRIVABLE_CATEGORIES,
    failure_rows,
    nonempty_text,
    redrivable_ids,
    write_failure_records,
)


def _res(iid: str, *, error: bool, category: str, reason: str | None = None) -> DispatchResult:
    return DispatchResult(
        iid, result=None if error else "ok", error=error, reason=reason, category=category
    )


def test_redrivable_ids_selects_transport_class_only():
    results = {
        "q_ok": _res("q_ok", error=False, category=RESULT_OK),
        "q_err": _res("q_err", error=True, category=RESULT_ERROR, reason="error: boom"),
        "q_429": _res("q_429", error=True, category=RESULT_RATE_LIMITED, reason="429"),
        "q_tx": _res("q_tx", error=True, category=RESULT_TRANSPORT, reason="transient"),
        "q_inv": _res("q_inv", error=True, category=RESULT_TRANSPORT, reason="invalid_response"),
    }
    assert {RESULT_TRANSPORT, RESULT_RATE_LIMITED} == REDRIVABLE_CATEGORIES
    assert sorted(redrivable_ids(results)) == ["q_429", "q_inv", "q_tx"]


def test_failure_rows_shape_and_split():
    ordered = ["q_ok", "q_tx", "q_429", "q_err", "q_missing"]
    results = {
        "q_ok": _res("q_ok", error=False, category=RESULT_OK),
        "q_tx": _res("q_tx", error=True, category=RESULT_TRANSPORT, reason="invalid_response (x)"),
        "q_429": _res("q_429", error=True, category=RESULT_RATE_LIMITED, reason="429"),
        "q_err": _res("q_err", error=True, category=RESULT_ERROR, reason="error: boom"),
        # q_missing: absent from results entirely -> content-class "missing".
    }
    records, counts = failure_rows(results, ordered, redriven={"q_tx"})
    assert counts == {"n_ok": 1, "n_transport_class": 2, "n_content_class": 2}
    by_id = {rec["item_id"]: rec for rec in records}
    assert set(by_id) == {"q_tx", "q_429", "q_err", "q_missing"}  # ok rows never recorded
    assert all({"item_id", "category", "reason", "round"} <= set(rec) for rec in records)
    # `round` keyed on the redriven set; the invalid_response reason prefix is
    # the auditable separator inside the transport-class tally — stays stable.
    assert by_id["q_tx"]["round"] == "redrive"
    assert by_id["q_tx"]["reason"].startswith("invalid_response")
    assert by_id["q_429"]["round"] == "initial"
    assert by_id["q_err"]["round"] == "initial"
    assert by_id["q_missing"]["category"] == "missing"
    assert by_id["q_missing"]["round"] == "initial"


def test_nonempty_text_validator():
    assert nonempty_text("") is False
    assert nonempty_text("  ") is False
    assert nonempty_text(None) is False
    assert nonempty_text(7) is False
    assert nonempty_text("x") is True


def test_write_failure_records_empty_still_written(tmp_path):
    """An empty failures list is the auditable zero-loss record — the file is
    ALWAYS written (#1470 acceptance criterion 5)."""
    p = tmp_path / "raw_completions" / "tag" / "ext_plain_claude_failures.json"
    counts = {"n_ok": 5, "n_transport_class": 0, "n_content_class": 0}
    write_failure_records(p, [], counts)
    payload = json.loads(p.read_text())
    assert payload == {"counts": counts, "failures": []}


def test_write_failure_records_roundtrips_rows(tmp_path):
    p = tmp_path / "failures.json"
    records = [
        {
            "item_id": "q1",
            "category": RESULT_TRANSPORT,
            "reason": "invalid_response (batch)",
            "round": "redrive",
        },
    ]
    counts = {"n_ok": 0, "n_transport_class": 1, "n_content_class": 0}
    write_failure_records(p, records, counts)
    payload = json.loads(p.read_text())
    assert payload["failures"] == records
    assert payload["counts"] == counts
