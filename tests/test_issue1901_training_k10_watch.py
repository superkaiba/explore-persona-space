"""The continuation must consume canonical oversized-note pointers without path escape."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import create_autospec

import pytest

FILE = Path(__file__).resolve().parents[1] / "scripts" / "issue1901_training_k10_watch.py"
SPEC = importlib.util.spec_from_file_location("training_k10_watch", FILE)
WATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCH)


@pytest.fixture
def oversized_result(tmp_path, monkeypatch):
    """A real oversized payload moved with its task, plus its older declared location."""
    payload = {
        "round": "training-k10-capture",
        "new_rows": 95000,
        "recipe_sha256": "a" * 64,
        "large_receipt_fixture": "x" * 50010,
    }
    text = json.dumps(payload)
    task = tmp_path / "current-location" / "1901"
    directory = task / "artifacts"
    directory.mkdir(parents=True)
    name = "sentinel-note-epm_results-1789000000.txt"
    path = directory / name
    path.write_text(text, encoding="utf-8")
    event = {
        "kind": "epm:results",
        "ts": "2026-09-11T20:00:00Z",
        "note": "[oversize note persisted; full payload is an artifact]",
        "oversize": True,
        "oversize_orig_len": len(text),
        "artifacts": [f"tasks/previous-location/1901/artifacts/{name}"],
    }
    resolver = create_autospec(WATCH.workflow.find_task_path, return_value=task)
    monkeypatch.setattr(WATCH.workflow, "find_task_path", resolver)
    return event, payload, path, resolver


def test_capture_reads_full_oversized_note_at_current_task_location(oversized_result, monkeypatch):
    event, payload, _, resolver = oversized_result
    assert event["oversize_orig_len"] > 50000
    events = create_autospec(WATCH.workflow.list_events, return_value=[event])
    monkeypatch.setattr(WATCH.workflow, "list_events", events)
    assert WATCH.capture_result("2026-09-11T19:00:00Z") == payload
    events.assert_called_once_with(1901)
    resolver.assert_called_once_with(1901)


def test_ordinary_json_and_unrelated_prose_do_not_require_artifact_reads(monkeypatch):
    resolver = create_autospec(WATCH.workflow.find_task_path)
    monkeypatch.setattr(WATCH.workflow, "find_task_path", resolver)
    assert WATCH.event_note_payload({"note": '{"round":"example"}'}) == {"round": "example"}
    assert WATCH.event_note_payload({"note": "An unrelated result uses prose."}) is None
    resolver.assert_not_called()


@pytest.mark.parametrize("mutation", ["missing", "multiple", "name", "length"])
def test_oversized_pointer_rejects_malformed_or_changed_evidence(oversized_result, mutation):
    event, _, _, _ = oversized_result
    event = dict(event)
    if mutation == "missing":
        event["artifacts"] = []
    elif mutation == "multiple":
        event["artifacts"] = event["artifacts"] * 2
    elif mutation == "name":
        event["artifacts"] = ["elsewhere/arbitrary.txt"]
    else:
        event["oversize_orig_len"] += 1
    with pytest.raises(ValueError):
        WATCH.event_note_payload(event)


def test_oversized_pointer_rejects_symlink_escape(oversized_result, tmp_path):
    event, payload, path, _ = oversized_result
    outside = tmp_path / "outside-evidence.txt"
    outside.write_text(json.dumps(payload), encoding="utf-8")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        WATCH.event_note_payload(event)


def test_capture_ignores_results_older_than_this_launch(oversized_result, monkeypatch):
    event, _, _, resolver = oversized_result
    events = create_autospec(WATCH.workflow.list_events, return_value=[event])
    monkeypatch.setattr(WATCH.workflow, "list_events", events)
    with pytest.raises(ValueError, match="found 0"):
        WATCH.capture_result("2026-09-11T21:00:00Z")
    resolver.assert_not_called()
