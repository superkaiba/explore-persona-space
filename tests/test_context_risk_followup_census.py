"""Protect unknown outcomes, frozen selection prerequisites and terminal process evidence."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.context_risk_followup_census import (
    counts,
    no_downstream_artifacts,
    support,
    verify_exit,
    verify_freshness,
)


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
