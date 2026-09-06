"""Dispatch contract tests at the actual subprocess boundary; never call a model."""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import create_autospec

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "issue2669_codex_dispatch.py"
spec = importlib.util.spec_from_file_location("issue2669_dispatch", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture_packet(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Predict a score for context x.")
    return {"id": "pilot-1", "behavior": "evil", "prompt_path": str(prompt), "ids": ["x"]}


def boundary(monkeypatch, modes):
    calls = []

    def external(*popenargs, input=None, capture_output=False, timeout=None, check=False, **kwargs):
        argv = popenargs[0]
        if argv == ["codex", "--version"]:
            return subprocess.CompletedProcess(argv, 0, stdout="codex-cli test\n")
        calls.append(argv)
        mode = modes.pop(0)
        if mode == "transport":
            kwargs["stderr"].write("HTTP 429 rate limit")
            return subprocess.CompletedProcess(argv, 1)
        if mode in {
            "message_transport_word",
            "message_actual_transport",
            "reasoning_transport_word",
            "event_transport",
        }:
            if mode == "event_transport":
                event = {"type": "turn.failed", "error": {"message": "HTTP 503 overloaded"}}
            else:
                kind = "reasoning" if mode == "reasoning_transport_word" else "agent_message"
                event = {
                    "type": "item.completed",
                    "item": {"type": kind, "text": "The context mentions a timeout."},
                }
            if mode == "message_actual_transport":
                kwargs["stderr"].write("HTTP 429 rate limit")
            kwargs["stdout"].write(json.dumps(event) + "\n")
            return subprocess.CompletedProcess(argv, 1)
        value = {"rows": [{"id": "x", "rationale": "Expected response.", "score_0_100": 21}]}
        if mode == "invalid":
            value["rows"][0]["score_0_100"] = 101
        events = [{"type": "thread.started", "thread_id": "fresh"}]
        if mode == "tools":
            events.append({"type": "item.started", "item": {"type": "command_execution"}})
        if mode == "item_error":
            events.append(
                {"type": "item.completed", "item": {"type": "error", "message": "failure"}}
            )
        if mode == "unicode":
            value["rows"][0]["rationale"] = "Context\u2028contains\u2029line separators."
        events.extend(
            [
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": json.dumps(value, ensure_ascii=False),
                    },
                },
                {"type": "turn.completed", "usage": {}},
            ]
        )
        kwargs["stdout"].write("\n".join(json.dumps(event, ensure_ascii=False) for event in events))
        Path(argv[argv.index("-o") + 1]).write_text(json.dumps(value))
        assert input.startswith(m.NO_TOOLS)
        assert Path(kwargs["cwd"]).is_relative_to("/tmp")
        assert "--ignore-user-config" in argv and "--ephemeral" in argv
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(m.subprocess, "run", create_autospec(subprocess.run, side_effect=external))
    return calls


def test_live_body_roundtrip_resume_and_fingerprint(tmp_path, monkeypatch):
    packet = fixture_packet(tmp_path)
    config = {"output_dir": str(tmp_path / "output"), "packets": [packet]}
    calls = boundary(monkeypatch, ["ok"])
    assert m.run(config)[0]["status"] == "complete"
    assert m.run(config)[0]["status"] == "complete"
    assert len(calls) == 1
    Path(packet["prompt_path"]).write_text("Changed rubric")
    with pytest.raises(ValueError, match="Fingerprint mismatch"):
        m.run(config)


@pytest.mark.parametrize(
    "mode",
    [
        "tools",
        "invalid",
        "item_error",
        "message_transport_word",
        "message_actual_transport",
        "reasoning_transport_word",
    ],
)
def test_invalid_never_retried_or_coerced(tmp_path, monkeypatch, mode):
    config = {"output_dir": str(tmp_path / "out"), "packets": [fixture_packet(tmp_path)]}
    calls = boundary(monkeypatch, [mode])
    assert m.run(config)[0]["status"] == "invalid"
    assert m.run(config)[0]["status"] == "invalid"
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["transport", "event_transport"])
def test_transport_retry_preserves_attempts(tmp_path, monkeypatch, mode):
    config = {
        "output_dir": str(tmp_path / "out"),
        "packets": [fixture_packet(tmp_path)],
        "retry_delay_seconds": 0.001,
    }
    calls = boundary(monkeypatch, [mode, "ok"])
    result = m.run(config)[0]
    assert result["status"] == "complete" and len(calls) == 2
    metadata = m.read_json(tmp_path / "out/pilot-1/attempt-001/metadata.json")
    assert metadata["status"] == "transport_loss"


def test_unicode_jsonl_roundtrip(tmp_path, monkeypatch):
    config = {"output_dir": str(tmp_path / "out"), "packets": [fixture_packet(tmp_path)]}
    boundary(monkeypatch, ["unicode"])
    assert m.run(config)[0]["status"] == "complete"


def test_resume_revalidates_events(tmp_path, monkeypatch):
    config = {"output_dir": str(tmp_path / "out"), "packets": [fixture_packet(tmp_path)]}
    boundary(monkeypatch, ["ok"])
    m.run(config)
    path = tmp_path / "out/pilot-1/attempt-001/events.jsonl"
    path.write_text('{"type":"item.started","item":{"type":"web_search"}}\n')
    with pytest.raises(ValueError, match="Tool use"):
        m.run(config)


@pytest.mark.parametrize("value", [True, -1, 101, float("nan"), float("inf"), "1"])
def test_scores_strict(value):
    with pytest.raises(ValueError):
        m.validate_rows({"rows": [{"id": "x", "rationale": "reason", "score_0_100": value}]}, ["x"])


def test_no_tools_fail_closed(tmp_path):
    path = tmp_path / "events"
    path.write_text('{"type":"new_unknown_event"}\n')
    with pytest.raises(ValueError, match="Unknown event"):
        m.audit_events(path)


def test_empty_subset_rejected(tmp_path):
    config = {
        "output_dir": str(tmp_path / "out"),
        "packets": [fixture_packet(tmp_path)],
        "packet_ids": [],
    }
    with pytest.raises(ValueError, match="subset"):
        m.run(config)
