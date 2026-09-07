"""Protect unknown outcomes, frozen selection prerequisites and terminal process evidence."""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.model import ModelUsage

from scripts.context_risk_followup_census import (
    counts,
    no_downstream_artifacts,
    requests,
    support,
    verify_exit,
    verify_freshness,
)
from scripts.context_risk_impossiblebench_inspect import GENERATION_EXTRA_BODY


def context(task, condition, success, failure, censored):
    """Build a compact outcome table without any synthetic model responses."""
    return dict(task_id=task, condition=condition, **counts(success, failure, censored))


def test_unknown_is_not_failure_or_mixed_and_original_eligibility_is_partial():
    rows = [context("a", "original", 0, 0, 2), context("a", "oneoff", 1, 0, 1)]
    got = support(rows)
    assert got["definitely_eligible_task_ids"] == []
    assert got["possibly_eligible_task_ids"] == ["a"]
    assert got["eligible_observed_positive"] == 0
    assert got["optimistic_eligible_positive_upper_bound"] == 2
    rows[0] = context("a", "original", 1, 0, 1)
    assert support(rows)["eligible_observed_mixed_contexts"] == 0
    rows[1] = context("a", "oneoff", 1, 1, 0)
    assert support(rows)["eligible_observed_mixed_contexts"] == 1


def test_completion_conditional_rate_and_bounds_are_distinct():
    got = counts(1, 3, 2, 2)
    assert got["completion_conditional_rate"] == 0.25
    assert got["unknown_outcome_rate_bounds"] == [1 / 8, 5 / 8]
    assert got["planned"] == 8 and got["realized"] == 6
    assert counts(0, 0, 2)["completion_conditional_rate"] is None
    assert counts(0, 0, 0)["unknown_outcome_rate_bounds"] is None


@pytest.mark.parametrize("bad", [-1, 1.5, True, "1"])
def test_counts_reject_invalid_types_and_values(bad):
    with pytest.raises(ValueError):
        counts(bad, 1, 0)


def exit_case():
    """Return an explicitly pinned censor termination fixture."""
    launch = dict(
        mode="development_a", supervisor_pid=100, worker_pid=101, run_result="/run/A.json"
    )
    record = dict(launch, cleanup="no_live_members", exit_code=1)
    report = dict(coverage_complete=True, realized_rollouts=120, technical_errors=1)
    message = "RuntimeError: Incomplete/censored phase; inspect /run/A.json"
    return launch, record, report, message


def test_expected_censor_exception_is_not_an_experiment_pass():
    launch, record, report, message = exit_case()
    assert verify_exit(record, launch, report, message) == "completed_generation_with_censor"
    report["technical_errors"] = 0
    record["exit_code"] = 0
    assert verify_exit(record, launch, report, "") == "completed_uncensored_generation"


@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", 124),
        ("exit_code", 137),
        ("exit_code", 143),
        ("exit_code", 0),
        ("cleanup", "terminated_descendants"),
        ("supervisor_pid", 200),
        ("mode", "development_b"),
    ],
)
def test_unexpected_termination_is_rejected(field, value):
    launch, record, report, message = exit_case()
    record[field] = value
    with pytest.raises(ValueError):
        verify_exit(record, launch, report, message)


def test_incomplete_or_unexplained_failure_cannot_be_certified():
    launch, record, report, message = exit_case()
    with pytest.raises(ValueError):
        verify_exit(record, launch, report, "RuntimeError: unrelated bug")
    report["realized_rollouts"] = 119
    with pytest.raises(ValueError):
        verify_exit(record, launch, report, message)


@pytest.mark.parametrize(
    "relative",
    [
        "fresh_A/logs/native.eval",
        "fresh_B/run_result.json",
        "fresh_A/rollouts.jsonl",
        "captures_A/chunk.pt",
        "analysis/primary/fits/raw_final.json",
        "analysis/primary/raw_selection.json",
        "selection.json",
    ],
)
def test_existing_downstream_work_rejects_zero_claim(tmp_path, relative):
    assert no_downstream_artifacts(tmp_path)["fresh_rollouts"] == 0
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ValueError):
        no_downstream_artifacts(tmp_path)


def test_launch_freshness_rejects_an_older_successful_run(tmp_path):
    launch = dict(
        mode="development_a",
        launch_id="one",
        supervisor_pid=100,
        worker_pid=101,
        started_utc="2026-09-07T17:00:00Z",
        log_path=str(tmp_path / "run.log"),
    )
    prefix = tmp_path / "development_a_one_process"
    launch["exit_path"] = f"{prefix}.exit.json"
    start = datetime.fromisoformat(launch["started_utc"]).timestamp()
    for suffix, pid in [("pid", 100), ("worker.pid", 101)]:
        path = Path(f"{prefix}.{suffix}")
        path.write_text(str(pid))
        os.utime(path, (start, start))
    Path(launch["log_path"]).write_text(
        f"[supervisor-start] mode=development_a pid=100 root={tmp_path} "
        f"utc={launch['started_utc']}\n"
        "[worker-start] mode=development_a pid=101\n"
    )
    report = tmp_path / "result.json"
    report.write_text("{}")
    os.utime(report, (start + 100, start + 100))
    native = [
        SimpleNamespace(
            stats=SimpleNamespace(
                started_at="2026-09-07T17:00:01Z", completed_at="2026-09-07T17:01:00Z"
            )
        )
    ]
    exit_record = {"finished_unix": start + 101}
    assert verify_freshness(tmp_path, launch, native, report, exit_record)["launch"] == launch
    native[0].stats.started_at = "2026-09-07T16:59:59Z"
    with pytest.raises(ValueError, match="chronology"):
        verify_freshness(tmp_path, launch, native, report, exit_record)


def retried_sample():
    """Construct the observed native error/recovery shape using real token-usage schema."""
    model = "openai-api/local/test-model"
    config = SimpleNamespace(
        max_tokens=65536,
        temperature=1.0,
        top_p=1.0,
        max_connections=16,
        max_retries=2,
        extra_body=GENERATION_EXTRA_BODY,
        seed=123,
    )
    user = SimpleNamespace(role="user", text="fixture input")
    assistant = SimpleNamespace(role="assistant", text="fixture response")
    start = datetime.fromisoformat("2026-09-07T17:00:00+00:00")
    error = SimpleNamespace(
        event="model",
        config=config,
        model=model,
        input=[user],
        error="Connection error.",
        timestamp=start,
        completed=None,
        output=SimpleNamespace(completion="", usage=None),
    )
    completed = SimpleNamespace(
        event="model",
        config=deepcopy(config),
        model=model,
        input=[deepcopy(user)],
        error=None,
        timestamp=start,
        completed=datetime.fromisoformat("2026-09-07T17:00:05+00:00"),
        output=SimpleNamespace(
            model="test-model",
            completion=assistant.text,
            choices=[SimpleNamespace(stop_reason="stop")],
            usage=ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5),
        ),
    )
    sample = SimpleNamespace(
        id="fixture",
        epoch=1,
        input=user.text,
        messages=[user, assistant],
        events=[error, completed],
        error=None,
        invalidation=None,
        scores={"successful_submission": SimpleNamespace(value="C")},
        metadata={
            "agentic_results": {
                "censored": False,
                "attempt_history": [
                    {
                        "attempt": 1,
                        "request_seed": 123,
                        "response": assistant.text,
                        "stop_reasons": ["stop"],
                        "success": True,
                    }
                ],
            }
        },
    )
    return sample, model


def test_recovered_transport_event_is_retained_without_advancing_submission():
    sample, model = retried_sample()
    completed, censored, errors = requests(sample, model)
    assert len(completed) == 1 and censored == [] and len(errors) == 1
    assert completed[0]["attempt"] == errors[0]["attempt"] == 1
    assert completed[0]["request_seed"] == errors[0]["seed"] == 123


@pytest.mark.parametrize("mutation", ["seed", "input", "unrecovered", "partial_output", "too_many"])
def test_retry_cannot_change_the_request_or_hide_missing_completions(mutation):
    sample, model = retried_sample()
    if mutation == "seed":
        sample.events[-1].config.seed = 124
    elif mutation == "input":
        sample.events[-1].input[0].text = "changed input"
    elif mutation == "unrecovered":
        sample.events.pop()
    elif mutation == "partial_output":
        sample.events[0].output.completion = "unaccounted output"
    else:
        sample.events = [deepcopy(sample.events[0]) for _ in range(3)] + [sample.events[-1]]
    with pytest.raises(ValueError):
        requests(sample, model)
