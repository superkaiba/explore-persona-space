from __future__ import annotations

from scripts.log_shipper import _log_event


def test_log_event_includes_parseable_metadata() -> None:
    event = _log_event("ERROR train crashed")

    assert event["eventType"] == "log"
    assert event["body"] == "ERROR train crashed"
    assert event["metadata"]["stream"] == "training_log"
    assert event["metadata"]["level"] == "error"
    assert event["metadata"]["source"] == "pod_log"


def test_log_event_marks_truncation() -> None:
    event = _log_event("x" * 50_001)

    assert len(event["body"]) == 50_000
    assert event["metadata"]["truncated"] is True
