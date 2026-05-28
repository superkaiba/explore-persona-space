"""Unit tests for poll_pipeline.py sentinel handling.

Covers MF-R2-1 (issue #406 round-2): the pod-side failure-sentinel pattern
end-to-end (parse-stdout -> emit-marker -> flip-status -> persist-state ->
return PollResult). Mocks `post_event` and `set_status` so the test runs
without mutating real task state, and feeds synthetic stdout to
`_parse_ssh_probe_stdout` so it runs without a live pod.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def poll_pipeline_module():
    """Import scripts/poll_pipeline.py as a module under test.

    Registers under sys.modules BEFORE exec_module so dataclass field
    resolution (which walks sys.modules[cls.__module__]) succeeds.
    """
    import importlib.util
    import sys

    name = "poll_pipeline_under_test"
    here = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(name, here / "scripts" / "poll_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_parse_stdout_no_sentinels(poll_pipeline_module):
    m = poll_pipeline_module
    stdout = (
        "PID_ALIVE=1\n"
        "MTIME_EPOCH=1700000000\n"
        "TAIL_START\n"
        "[phase=training step=10]\n"
        "[phase=eval]\n"
        "TAIL_END\n"
        "SENTINEL_LIST_START\n"
        "SENTINEL_LIST_END\n"
    )
    parsed = m._parse_ssh_probe_stdout(stdout)
    assert parsed["pid_alive"] == "1"
    assert parsed["mtime_epoch"] == "1700000000"
    assert "[phase=training step=10]" in parsed["log_tail"]
    assert "[phase=eval]" in parsed["log_tail"]
    assert parsed["sentinels"] == []


def test_parse_stdout_with_one_sentinel(poll_pipeline_module):
    m = poll_pipeline_module
    body = json.dumps(
        {
            "issue": 406,
            "phase": "phase2_pilot",
            "failure_class": "rig",
            "condition": "A1",
            "reason": "A1 pilot failed both default-lr and halved-lr",
        }
    )
    stdout = (
        "PID_ALIVE=0\n"
        "MTIME_EPOCH=1700000100\n"
        "TAIL_START\n"
        "[phase=training step=20]\n"
        "TAIL_END\n"
        "SENTINEL_LIST_START\n"
        "/workspace/logs/issue-406-pilot-failed.json\n"
        "SENTINEL_LIST_END\n"
        "SENTINEL_BODY_START /workspace/logs/issue-406-pilot-failed.json\n"
        f"{body}\n"
        "SENTINEL_BODY_END\n"
    )
    parsed = m._parse_ssh_probe_stdout(stdout)
    assert parsed["pid_alive"] == "0"
    assert len(parsed["sentinels"]) == 1
    s = parsed["sentinels"][0]
    assert s["path"] == "/workspace/logs/issue-406-pilot-failed.json"
    assert isinstance(s["payload"], dict)
    assert s["payload"]["failure_class"] == "rig"
    assert s["payload"]["condition"] == "A1"


def test_parse_stdout_with_two_sentinels(poll_pipeline_module):
    m = poll_pipeline_module
    body1 = json.dumps({"phase": "phase2_pilot", "condition": "A1", "failure_class": "rig"})
    body2 = json.dumps({"phase": "phase2_batch", "condition": "B3", "failure_class": "rig"})
    stdout = (
        "PID_ALIVE=0\n"
        "MTIME_EPOCH=1700000200\n"
        "TAIL_START\n"
        "TAIL_END\n"
        "SENTINEL_LIST_START\n"
        "/workspace/logs/issue-406-pilot-failed.json\n"
        "/workspace/logs/issue-406-batch-failed.json\n"
        "SENTINEL_LIST_END\n"
        "SENTINEL_BODY_START /workspace/logs/issue-406-pilot-failed.json\n"
        f"{body1}\n"
        "SENTINEL_BODY_END\n"
        "SENTINEL_BODY_START /workspace/logs/issue-406-batch-failed.json\n"
        f"{body2}\n"
        "SENTINEL_BODY_END\n"
    )
    parsed = m._parse_ssh_probe_stdout(stdout)
    assert len(parsed["sentinels"]) == 2
    paths = {s["path"] for s in parsed["sentinels"]}
    assert paths == {
        "/workspace/logs/issue-406-pilot-failed.json",
        "/workspace/logs/issue-406-batch-failed.json",
    }


def test_parse_stdout_malformed_sentinel_body_is_surfaced(poll_pipeline_module):
    """A sentinel whose body isn't valid JSON still surfaces — raw string."""
    m = poll_pipeline_module
    stdout = (
        "PID_ALIVE=0\n"
        "MTIME_EPOCH=1700000300\n"
        "TAIL_START\n"
        "TAIL_END\n"
        "SENTINEL_LIST_START\n"
        "/workspace/logs/issue-406-misc-failed.json\n"
        "SENTINEL_LIST_END\n"
        "SENTINEL_BODY_START /workspace/logs/issue-406-misc-failed.json\n"
        "this is not valid json {{{\n"
        "SENTINEL_BODY_END\n"
    )
    parsed = m._parse_ssh_probe_stdout(stdout)
    assert len(parsed["sentinels"]) == 1
    s = parsed["sentinels"][0]
    assert isinstance(s["payload"], str)
    assert "not valid json" in s["payload"]


def test_emit_sentinel_failure_markers_calls_poster_and_flipper(poll_pipeline_module):
    """The handler posts epm:failure per sentinel + flips status once."""
    m = poll_pipeline_module
    posted = []
    flipped = []

    def fake_poster(issue, kind, *, by, note, **extras):
        posted.append({"issue": issue, "kind": kind, "by": by, "note": note, **extras})

    def fake_flipper(issue, new_status, *, note=None):
        flipped.append({"issue": issue, "new_status": new_status, "note": note})

    sentinels = [
        {
            "path": "/workspace/logs/issue-406-pilot-failed.json",
            "payload": {
                "phase": "phase2_pilot",
                "failure_class": "rig",
                "condition": "A1",
                "reason": "A1 pilot failed",
            },
        },
        {
            "path": "/workspace/logs/issue-406-batch-failed.json",
            "payload": {
                "phase": "phase2_batch",
                "failure_class": "rig",
                "condition": "B3",
                "reason": "B3 training failed",
            },
        },
    ]
    newly = m._emit_sentinel_failure_markers(
        406,
        "epm-issue-406",
        sentinels,
        already_emitted_paths=set(),
        poster=fake_poster,
        flipper=fake_flipper,
    )

    assert newly == {
        "/workspace/logs/issue-406-pilot-failed.json",
        "/workspace/logs/issue-406-batch-failed.json",
    }
    assert len(posted) == 2
    assert all(p["kind"] == "epm:failure" for p in posted)
    assert {p["condition"] for p in posted} == {"A1", "B3"}
    # set_status flipped to blocked exactly once (subsequent sentinels skip).
    assert len(flipped) == 1
    assert flipped[0]["new_status"] == "blocked"


def test_emit_sentinel_failure_markers_skips_already_emitted(poll_pipeline_module):
    """Re-running over the same sentinel path is a no-op."""
    m = poll_pipeline_module
    posted = []
    flipped = []
    sentinels = [
        {
            "path": "/workspace/logs/issue-406-pilot-failed.json",
            "payload": {"phase": "phase2_pilot", "failure_class": "rig", "condition": "A1"},
        }
    ]
    newly = m._emit_sentinel_failure_markers(
        406,
        "epm-issue-406",
        sentinels,
        already_emitted_paths={"/workspace/logs/issue-406-pilot-failed.json"},
        poster=lambda *a, **kw: posted.append((a, kw)),
        flipper=lambda *a, **kw: flipped.append((a, kw)),
    )
    assert newly == set()
    assert posted == []
    assert flipped == []


def test_emit_handles_unparseable_payload(poll_pipeline_module):
    """A string payload (JSON parse failed earlier) still posts a marker."""
    m = poll_pipeline_module
    posted = []
    flipped = []
    sentinels = [
        {
            "path": "/workspace/logs/issue-406-misc-failed.json",
            "payload": "not parseable",
        }
    ]
    newly = m._emit_sentinel_failure_markers(
        406,
        "epm-issue-406",
        sentinels,
        already_emitted_paths=set(),
        poster=lambda issue, kind, *, by, note, **extras: posted.append(
            {"kind": kind, "note": note, **extras}
        ),
        flipper=lambda issue, new_status, *, note=None: flipped.append((issue, new_status, note)),
    )
    assert newly == {"/workspace/logs/issue-406-misc-failed.json"}
    assert len(posted) == 1
    assert posted[0]["kind"] == "epm:failure"
    assert posted[0]["failure_class"] == "rig"
    assert "unparseable body" in posted[0]["note"]


def test_poll_once_translates_sentinel_to_dead_with_failed_phase(
    poll_pipeline_module, tmp_path, monkeypatch
):
    """End-to-end: synthetic _ssh_probe -> sentinel emitted -> dead + failed_sentinel."""
    m = poll_pipeline_module
    posted = []
    flipped = []

    def fake_probe(pod, log_path, pid_file, *, issue=None):
        return {
            "pid_alive": "0",
            "mtime_epoch": "1700000000",
            "log_tail": "[phase=training step=5]\n",
            "sentinels": [
                {
                    "path": "/workspace/logs/issue-406-pilot-failed.json",
                    "payload": {
                        "phase": "phase2_pilot",
                        "failure_class": "rig",
                        "condition": "A1",
                        "reason": "A1 failed both retries",
                    },
                }
            ],
        }

    def fake_post_event(issue, kind, *, by, note, **extras):
        posted.append({"kind": kind, "note": note, **extras})

    def fake_set_status(issue, new_status, *, note=None):
        flipped.append((issue, new_status))

    monkeypatch.setattr(m, "_ssh_probe", fake_probe)
    # Patch the module-level post_event AND the inline-imported set_status.
    monkeypatch.setattr(m, "post_event", fake_post_event)

    # _emit_sentinel_failure_markers inline-imports from task_workflow; patch there.
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(tw, "post_event", fake_post_event)
    monkeypatch.setattr(tw, "set_status", fake_set_status)

    state_file = tmp_path / "poll-state.json"
    result = m.poll_once(
        issue=406,
        pod="epm-issue-406",
        log_path="/workspace/logs/issue-406.log",
        pid_file="/workspace/logs/issue-406.pid",
        state_file=state_file,
    )

    assert result.status == "dead"
    assert result.current_phase == "failed_sentinel"
    assert len(posted) == 1
    assert posted[0]["kind"] == "epm:failure"
    assert posted[0]["condition"] == "A1"
    assert len(flipped) == 1
    assert flipped[0] == (406, "blocked")

    # State persisted — emitted_sentinels recorded for idempotency.
    saved = json.loads(state_file.read_text())
    assert "/workspace/logs/issue-406-pilot-failed.json" in saved["406"]["emitted_sentinels"]


def test_poll_once_idempotent_on_repeat_tick(poll_pipeline_module, tmp_path, monkeypatch):
    """Second tick over the same sentinel does NOT re-emit the marker."""
    m = poll_pipeline_module
    posted = []
    flipped = []

    def fake_probe(pod, log_path, pid_file, *, issue=None):
        return {
            "pid_alive": "0",
            "mtime_epoch": "1700000000",
            "log_tail": "",
            "sentinels": [
                {
                    "path": "/workspace/logs/issue-406-pilot-failed.json",
                    "payload": {"phase": "phase2_pilot", "failure_class": "rig", "condition": "A1"},
                }
            ],
        }

    monkeypatch.setattr(m, "_ssh_probe", fake_probe)
    monkeypatch.setattr(m, "post_event", lambda *a, **kw: posted.append((a, kw)))

    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw, "post_event", lambda *a, **kw: posted.append({"kind": kw.get("kind") or a[1], **kw})
    )
    monkeypatch.setattr(tw, "set_status", lambda *a, **kw: flipped.append((a, kw)))

    state_file = tmp_path / "poll-state.json"
    # First tick
    result1 = m.poll_once(
        issue=406,
        pod="epm-issue-406",
        log_path="/workspace/logs/issue-406.log",
        pid_file="/workspace/logs/issue-406.pid",
        state_file=state_file,
    )
    assert result1.status == "dead"
    n_posted_after_first = len(posted)
    n_flipped_after_first = len(flipped)

    # Second tick — same sentinel, state file shows already emitted
    result2 = m.poll_once(
        issue=406,
        pod="epm-issue-406",
        log_path="/workspace/logs/issue-406.log",
        pid_file="/workspace/logs/issue-406.pid",
        state_file=state_file,
    )
    assert result2.status == "dead"
    # No new markers / status flips on second tick.
    assert len(posted) == n_posted_after_first
    assert len(flipped) == n_flipped_after_first
