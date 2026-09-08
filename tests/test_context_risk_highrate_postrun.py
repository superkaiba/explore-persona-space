"""Explicit fixture trajectories test censor semantics, never count as experiment draws."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import requests
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalSample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)

from scripts import context_risk_highrate_capacity as capacity
from scripts import context_risk_highrate_postrun as postrun


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def seed(sample_id, epoch, attempt):
    return int(hashlib.sha256(f"38295:{sample_id}:{epoch}:{attempt}".encode()).hexdigest()[:8], 16)


def error_wrapper(event):
    cfg = event.config
    request = {
        "messages": [{"role": m.role, "content": m.text} for m in event.input],
        "tools": None,
        "tool_choice": None,
        "extra_headers": {"x-irid": "fixture-only"},
        "model": capacity.MODEL.removeprefix("openai-api/local/"),
        **{
            key: getattr(cfg, key)
            for key in ("max_tokens", "seed", "temperature", "top_p", "extra_body")
        },
    }
    decoded = (
        "\nRequest:\n" + json.dumps(request) + "\n\nBadRequestError(" + repr(event.error) + ")"
    )
    return "ModelGenerateError(" + repr(decoded) + ")"


@pytest.fixture
def failed_sample():
    sid = "highrate_screen:B:fixture_only:oneoff"
    start = datetime(2026, 9, 8, tzinfo=UTC)
    messages = [ChatMessageUser(content="Fixture: return one")]
    response = "```python\ndef f(): return 0\n```"
    instruction = "Implement the function."
    config = dict(
        max_tokens=65536,
        temperature=1.0,
        top_p=1.0,
        max_connections=16,
        max_retries=2,
        extra_body=capacity.GENERATION_EXTRA_BODY,
    )
    output = ModelOutput.from_content(capacity.MODEL.removeprefix("openai-api/local/"), response)
    output.usage = ModelUsage(input_tokens=3, output_tokens=10, total_tokens=13)
    completed = ModelEvent(
        model=capacity.MODEL,
        input=list(messages),
        tools=[],
        tool_choice="none",
        config=GenerateConfig(**config, seed=seed(sid, 1, 1)),
        output=output,
        timestamp=start,
        completed=start + timedelta(seconds=1),
    )
    messages.append(ChatMessageAssistant(content=response))
    messages.append(
        ChatMessageUser(
            content="\nYour previous attempt failed the tests. Here's the error:\n"
            "AssertionError\n\n"
            f"\n\nTo reiterate, this is your task: {instruction}"
        )
    )
    error = "Error code: 400 - " + repr(
        {
            "error": {
                "message": capacity.ERROR,
                "type": "BadRequestError",
                "param": "input_tokens",
                "code": 400,
            }
        }
    )
    failed = ModelEvent(
        model=capacity.MODEL,
        input=list(messages),
        tools=[],
        tool_choice="none",
        config=GenerateConfig(**config, seed=seed(sid, 1, 2)),
        output=ModelOutput.from_content(capacity.MODEL, ""),
        timestamp=start + timedelta(seconds=2),
        error=error,
    )
    return EvalSample.model_validate(
        {
            "id": sid,
            "epoch": 1,
            "input": messages[0].text,
            "target": "",
            "messages": messages,
            "events": [completed, failed],
            "error": {
                "message": error_wrapper(failed),
                "traceback": "fixture",
                "traceback_ansi": "fixture",
            },
            "scores": {"successful_submission": {"value": "I"}},
            "metadata": {
                "instruction_prompt": instruction,
                "agentic_results": {
                    "flag_for_human_intervention": False,
                    "censored": False,
                    "harness_fingerprint": postrun.collection.harness_fingerprint(),
                    "attempt_history": [
                        {
                            "attempt": 1,
                            "request_seed": seed(sid, 1, 1),
                            "response": response,
                            "category": "assertion_failure",
                            "success": False,
                            "stderr": "AssertionError",
                            "stdout": "",
                            "stop_reasons": ["stop"],
                        }
                    ],
                },
            },
        }
    )


def test_rejected_prefix_preserves_native_unknown(failed_sample):
    original = failed_sample.model_dump(mode="json")
    record, completed, retries = capacity.rejected_request(failed_sample)
    assert record["attempt"] == 2 and len(completed) == 1 and not retries
    assert failed_sample.model_dump(mode="json") == original
    assert postrun.collection.outcome(failed_sample) == "censored"
    with pytest.raises(ValueError, match="resolved native"):
        postrun.collection.requests(failed_sample, capacity.MODEL)


@pytest.mark.parametrize(
    "mutation",
    [
        "seed",
        "model",
        "feedback",
        "output",
        "usage",
        "completed",
        "extra_choice",
        "wrong400",
        "top_error",
        "execution_incomplete",
        "intervention",
        "success",
        "extra_event",
    ],
)
def test_rejected_impostors(failed_sample, mutation):
    event = failed_sample.events[-1]
    if mutation == "seed":
        event.config.seed += 1
    elif mutation == "model":
        event.model = "wrong"
    elif mutation == "feedback":
        event.input[-1].content = "wrong feedback"
    elif mutation == "output":
        event.output = ModelOutput.from_content(capacity.MODEL, "generated!")
    elif mutation == "usage":
        event.output.usage = ModelUsage(input_tokens=1, output_tokens=0, total_tokens=1)
    elif mutation == "completed":
        event.completed = event.timestamp
    elif mutation == "extra_choice":
        event.output.choices.append(event.output.choices[0].model_copy(deep=True))
    elif mutation == "wrong400":
        event.error = "Error code: 400 - {'error': 'different'}"
    elif mutation == "top_error":
        failed_sample.error.message = "unrelated"
    elif mutation == "execution_incomplete":
        failed_sample.metadata["agentic_results"]["attempt_history"][0]["category"] = (
            "execution_incomplete"
        )
    elif mutation == "intervention":
        failed_sample.metadata["agentic_results"]["flag_for_human_intervention"] = True
    elif mutation == "success":
        failed_sample.metadata["agentic_results"]["attempt_history"][0]["success"] = True
    elif mutation == "extra_event":
        failed_sample.events.insert(1, event.model_copy(deep=True))
    with pytest.raises(ValueError):
        capacity.rejected_request(failed_sample)


def cached_proof(tmp_path, monkeypatch, failed_sample):
    tokens = [1] * 196609
    result = requests.Response()
    result.status_code = 200
    result.url = "http://fixture.invalid/tokenize"
    result._content = json.dumps(
        {"count": len(tokens), "max_model_len": 262144, "tokens": tokens}
    ).encode()
    boundary = create_autospec(requests.post, return_value=result)
    monkeypatch.setattr(capacity.http, "post", boundary)
    proof = capacity.cache_tokens(failed_sample, tmp_path, "screen", "http://fixture.invalid/v1")
    assert boundary.call_count == 1
    assert boundary.call_args.args == ("http://fixture.invalid/tokenize",)
    assert boundary.call_args.kwargs["json"] == capacity.payload(
        capacity.rejected_request(failed_sample)[0]
    )
    return capacity.proof_path(tmp_path, "screen", failed_sample), proof


def test_tokenizer_cache_real_body_and_no_overwrite(tmp_path, monkeypatch, failed_sample):
    path, proof = cached_proof(tmp_path, monkeypatch, failed_sample)
    assert proof["proof"]["input_tokens"] == 196609
    with pytest.raises(FileExistsError):
        capacity.cache_tokens(failed_sample, tmp_path, "screen", "http://fixture.invalid/v1")
    assert path.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint", "http://wrong.invalid/tokenize"),
        ("response_sha256", "invalid"),
        ("request_seed", 1),
        ("input_tokens", 196608),
        ("request_payload_sha256", "0" * 64),
    ],
)
def test_token_proof_rejects_drift(tmp_path, monkeypatch, failed_sample, field, value):
    path, _ = cached_proof(tmp_path, monkeypatch, failed_sample)
    proof = json.loads(path.read_text())
    proof[field] = value
    write(path, proof)
    with pytest.raises(ValueError):
        capacity.validate_tokens(
            capacity.rejected_request(failed_sample)[0], path, "http://fixture.invalid/v1"
        )


def test_token_file_mid_validation_mutation(tmp_path, monkeypatch, failed_sample):
    path, _ = cached_proof(tmp_path, monkeypatch, failed_sample)
    real = capacity.sha256
    calls = 0

    def changing(candidate):
        nonlocal calls
        if candidate == path:
            calls += 1
            if calls == 2:
                write(path.with_suffix(".tokens.json"), {"tokens": [1]})
        return real(candidate)

    monkeypatch.setattr(capacity, "sha256", create_autospec(real, side_effect=changing))
    with pytest.raises(ValueError):
        capacity.validate_tokens(
            capacity.rejected_request(failed_sample)[0], path, "http://fixture.invalid/v1"
        )


def test_extended_audit_retains_census_and_accounts_every_issue(
    tmp_path, monkeypatch, failed_sample
):
    cached_proof(tmp_path, monkeypatch, failed_sample)
    original = {
        "validation_issues": [
            {
                "scope": [failed_sample.id, 1],
                "type": "ValueError",
                "message": "This terminal receipt requires resolved native sample execution",
            }
        ],
        "counts": {"success": 0, "failure": 0, "censored": 1},
        "contexts": ["fixture-counts-only"],
    }
    logs = [
        SimpleNamespace(
            status="success",
            stats=SimpleNamespace(completed_at="fixture-terminal"),
            samples=[failed_sample],
        )
    ]
    result = postrun.extend_audit(
        tmp_path, "screen", original, logs, 0, {failed_sample.id: 3}, "http://fixture.invalid/v1"
    )
    assert result["counts"] == original["counts"] and result["contexts"] == original["contexts"]
    assert result["original_validation_issues"] == original["validation_issues"]
    assert len(result["requests"]) == 1 and len(result["capacity_censors"]) == 1
    original["validation_issues"].append({"scope": "eval", "message": "unrelated"})
    with pytest.raises(ValueError, match="unrecognized"):
        postrun.extend_audit(
            tmp_path,
            "screen",
            original,
            logs,
            0,
            {failed_sample.id: 3},
            "http://fixture.invalid/v1",
        )


def test_strict_phase_dispatch_real_body(tmp_path, monkeypatch):
    write(tmp_path / "fresh_B/run_result.json", {"passed": True})
    strict = create_autospec(postrun.collection.verify_report, return_value={"strict": True})
    monkeypatch.setattr(postrun.collection, "verify_report", strict)
    assert postrun.verify_report(tmp_path, "fresh") == {"strict": True}
    strict.assert_called_once_with(tmp_path, "fresh")
    write(tmp_path / "fresh_B/postrun_audit.json", {"invalid_duplicate": True})
    with pytest.raises(ValueError, match="also claim"):
        postrun.verify_report(tmp_path, "fresh")


def test_write_new_preserves_existing_bytes(tmp_path):
    path = tmp_path / "receipt.json"
    postrun.write_new(path, {"original": False})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        postrun.write_new(path, {"replacement": True})
    assert path.read_bytes() == before


def test_current_review_and_source_closure(tmp_path):
    path = tmp_path / "review.json"
    hashes = postrun.source_hashes()
    assert all(hashes[k] == v for k, v in postrun.design.source_hashes().items())
    write(path, {"verdict": "PASS", "reviewer": "explicit-fixture", "sources_sha256": hashes})
    assert postrun.validate_review(path)["sources_sha256"] == hashes
    value = json.loads(path.read_text())
    value["sources_sha256"]["scripts/context_risk_highrate_postrun.py"] = "0" * 64
    write(path, value)
    with pytest.raises(ValueError, match="current independent"):
        postrun.validate_review(path)
