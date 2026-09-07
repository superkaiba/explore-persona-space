"""Focused tests for the one-off #2254 structured-output recovery."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_structured_recovery as recovery
import scripts.issue2254_revmap8_subagent_grade as runner


def _job() -> runner.JobSpec:
    items = tuple(
        runner.GradeItem(
            source_item_id=f"source-{index}",
            opaque_id=f"i{index:020d}",
            cell_id="fixture",
            behavior="evil",
            arm="decodeonly",
            phase="integrity_production",
            qi=index,
            seed=42,
            context_index=index,
            draw_index=0,
            question=f"Question {index}?",
            answer=f"Answer {index}.",
        )
        for index in range(4)
    )
    prompt = base._integrity_prompt(list(items))
    return runner.JobSpec(
        scope="integrity_production",
        rubric_id="coherence",
        pass_index=2,
        chunk_index=39,
        items=items,
        prompt=prompt,
        prompt_tokens_o200k=runner._count_o200k_tokens(prompt),
        instrument_fp="fixture-fp",
    )


def _receipt(job: runner.JobSpec) -> dict:
    replacement = recovery._replacement_job(job)
    return {
        "parent": recovery.policy._parent_metadata(job),
        "replacement": recovery.policy._parent_metadata(replacement),
    }


def test_replacement_changes_only_audit_job_identity() -> None:
    job = _job()
    sibling = replace(job, chunk_index=38)

    output = recovery.apply_structured_replacement([sibling, job], _receipt(job))
    replacement = output[1]

    assert output[0] is sibling
    assert replacement.job_id == f"{job.job_id}{recovery.REPLACEMENT_SUFFIX}"
    assert replacement.prompt == job.prompt
    assert replacement.items == job.items
    assert replacement.instrument_fp == job.instrument_fp
    assert replacement.pass_index == job.pass_index
    assert replacement.chunk_index == job.chunk_index
    assert replacement.prompt_tokens_o200k == job.prompt_tokens_o200k
    assert runner._output_schema(replacement) == runner._output_schema(job)


def test_replacement_metadata_drift_fails_closed() -> None:
    job = _job()
    receipt = _receipt(job)
    receipt["replacement"]["prompt_sha256"] = "0" * 64

    with pytest.raises(base.AnalysisError, match="replacement metadata changed"):
        recovery.apply_structured_replacement([job], receipt)


def test_replacement_requires_exactly_one_parent() -> None:
    job = _job()

    with pytest.raises(base.AnalysisError, match="matched 0 parents"):
        recovery.apply_structured_replacement([replace(job, chunk_index=38)], _receipt(job))


def test_orphaned_replacement_failure_is_terminal(monkeypatch, tmp_path) -> None:
    job = recovery._replacement_job(_job())
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    attempts = tmp_path / "attempts" / job.scope / job.rubric_id
    attempts.mkdir(parents=True)
    (attempts / f"{job.job_id}.1.failed.json").write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="requires user direction"):
        recovery._guard_replacement(tmp_path, job)


def test_schema_only_replacement_state_is_terminal(monkeypatch, tmp_path) -> None:
    job = recovery._replacement_job(_job())
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    schema = recovery._schema_path(tmp_path, job)
    schema.parent.mkdir(parents=True)
    schema.write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="requires user direction"):
        recovery._guard_replacement(tmp_path, job)


def test_non_target_unregistered_failure_still_blocks(monkeypatch, tmp_path) -> None:
    job = _job()
    other = replace(job, chunk_index=38)
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    attempts = tmp_path / "attempts" / other.scope / other.rubric_id
    attempts.mkdir(parents=True)
    (attempts / f"{other.job_id}.1.failed.json").write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="forbids more attempts"):
        recovery._guard_unregistered_failures_except_target(tmp_path, [other], {})


def test_parent_failure_cannot_be_reinterpreted_as_policy_recovery() -> None:
    job = _job()
    assert job.job_id == recovery.TARGET_JOB_ID
    assert "policy" not in recovery.REPLACEMENT_SUFFIX
    assert recovery._replacement_job(job).items == job.items


def test_structural_audit_requires_duplicate_omission_and_order_defect(
    monkeypatch,
) -> None:
    job = _job()
    returned_ids = [
        job.items[0].opaque_id,
        job.items[1].opaque_id,
        job.items[3].opaque_id,
        job.items[1].opaque_id,
    ]
    raw = json.dumps(
        {
            "rubric_id": "coherence",
            "scores": [
                {"item_id": item_id, "score": 80, "reasoning": "typed"} for item_id in returned_ids
            ],
        },
        separators=(",", ":"),
    )
    failure = {
        "status": "failed_content",
        "attempt_index": 1,
        "returncode": 0,
        "job_id": job.job_id,
        "scope": job.scope,
        "rubric_id": job.rubric_id,
        "pass_index": job.pass_index,
        "chunk_index": job.chunk_index,
        "instrument_fp": job.instrument_fp,
        "prompt_sha256": base._sha256_text(job.prompt),
        "prompt_tokens_o200k": job.prompt_tokens_o200k,
        "request": job.prompt,
        "source_item_ids": [item.source_item_id for item in job.items],
        "opaque_item_ids": [item.opaque_id for item in job.items],
        "raw_response": raw,
        "stdout": "fixture events",
    }
    monkeypatch.setattr(recovery, "TARGET_RAW_RESPONSE_SHA256", recovery._sha256_text(raw))
    monkeypatch.setattr(
        runner, "_validate_codex_events", lambda expected_job, stdout: ([], "thread")
    )

    with pytest.raises(base.AnalysisError, match="defect changed"):
        recovery._structural_failure_audit(job, failure)


def test_structural_audit_accepts_the_pinned_51_row_shape(monkeypatch) -> None:
    items = tuple(
        replace(item, source_item_id=f"source-{index}", opaque_id=f"i{index:020d}")
        for index, item in enumerate((_job().items * 13)[:51])
    )
    prompt = base._integrity_prompt(list(items))
    job = replace(
        _job(),
        items=items,
        prompt=prompt,
        prompt_tokens_o200k=runner._count_o200k_tokens(prompt),
    )
    returned_ids = [item.opaque_id for item in items]
    returned_ids[22] = returned_ids[43]
    returned_ids[27] = returned_ids[41]
    raw = json.dumps(
        {
            "rubric_id": "coherence",
            "scores": [
                {"item_id": item_id, "score": 80, "reasoning": "typed"} for item_id in returned_ids
            ],
        },
        separators=(",", ":"),
    )
    failure = {
        "status": "failed_content",
        "attempt_index": 1,
        "returncode": 0,
        "job_id": job.job_id,
        "scope": job.scope,
        "rubric_id": job.rubric_id,
        "pass_index": job.pass_index,
        "chunk_index": job.chunk_index,
        "instrument_fp": job.instrument_fp,
        "prompt_sha256": base._sha256_text(job.prompt),
        "prompt_tokens_o200k": job.prompt_tokens_o200k,
        "request": job.prompt,
        "source_item_ids": [item.source_item_id for item in job.items],
        "opaque_item_ids": [item.opaque_id for item in job.items],
        "raw_response": raw,
        "stdout": "fixture events",
    }
    monkeypatch.setattr(recovery, "TARGET_RAW_RESPONSE_SHA256", recovery._sha256_text(raw))
    monkeypatch.setattr(
        runner, "_validate_codex_events", lambda expected_job, stdout: ([], "thread")
    )

    audit = recovery._structural_failure_audit(job, failure)

    assert audit["returned_rows"] == 51
    assert audit["returned_unique_ids"] == 49
    assert len(audit["duplicate_ids"]) == 2
    assert len(audit["omitted_ids"]) == 2
    assert audit["unexpected_ids"] == []
    assert audit["scores_salvaged"] == 0


def test_replacement_lease_is_written_before_grader_call(monkeypatch, tmp_path) -> None:
    parent = _job()
    job = recovery._replacement_job(parent)
    receipt = (
        tmp_path
        / "recovery"
        / "structured_output_replacements"
        / (f"{recovery.TARGET_JOB_ID}.json")
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"entry_sha256":"fixture"}\n', encoding="utf-8")
    manifest = tmp_path / "recovery" / "runner_manifest_structured.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    monkeypatch.setattr(recovery, "_ACTIVE_OUT_ROOT", tmp_path)
    monkeypatch.setattr(base, "_load_staged", lambda args: ([], {}, {}))
    monkeypatch.setattr(
        recovery.policy,
        "_prior_recovered_jobs",
        lambda items, instrument, rubrics, rubric_id: [parent],
    )
    monkeypatch.setattr(recovery, "_load_receipt", lambda out_root, target: {})
    observed = {}

    def fake_run(args, root, delegated_job):
        observed["lease_exists"] = recovery._lease_path(tmp_path).is_file()
        observed["job"] = delegated_job
        return {"status": "complete"}

    monkeypatch.setattr(recovery, "_ORIGINAL_RUN_ONE_JOB", fake_run)

    result = recovery._run_one_job_with_structured_lease(None, tmp_path, job)

    assert result == {"status": "complete"}
    assert observed == {"lease_exists": True, "job": job}


def test_launch_lease_exclusively_consumes_the_single_slot(monkeypatch, tmp_path) -> None:
    parent = _job()
    replacement = recovery._replacement_job(parent)
    receipt = (
        tmp_path / "recovery" / "structured_output_replacements" / f"{recovery.TARGET_JOB_ID}.json"
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"entry_sha256":"fixture"}\n', encoding="utf-8")
    manifest = tmp_path / "recovery" / "runner_manifest_structured.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    monkeypatch.setattr(base, "_load_staged", lambda args: ([], {}, {}))
    monkeypatch.setattr(
        recovery.policy,
        "_prior_recovered_jobs",
        lambda items, instrument, rubrics, rubric_id: [parent],
    )
    monkeypatch.setattr(recovery, "_load_receipt", lambda out_root, target: {})

    recovery._write_launch_lease(tmp_path, replacement)

    with pytest.raises(base.AnalysisError, match="already consumed"):
        recovery._write_launch_lease(tmp_path, replacement)


def test_existing_lease_blocks_grader_call(monkeypatch, tmp_path) -> None:
    job = recovery._replacement_job(_job())
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    monkeypatch.setattr(recovery, "_ACTIVE_OUT_ROOT", tmp_path)
    monkeypatch.setattr(recovery, "_load_launch_lease", lambda out_root, target: {})
    called = False

    def forbidden_run(args, root, delegated_job):
        nonlocal called
        called = True

    monkeypatch.setattr(recovery, "_ORIGINAL_RUN_ONE_JOB", forbidden_run)

    with pytest.raises(base.AnalysisError, match="requires user direction"):
        recovery._run_one_job_with_structured_lease(None, tmp_path, job)
    assert called is False


def test_cached_canonical_requires_exact_replacement_schema(monkeypatch, tmp_path) -> None:
    job = recovery._replacement_job(_job())
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    monkeypatch.setattr(recovery, "_load_launch_lease", lambda out_root, target: {})
    canonical = runner._job_record_path(tmp_path, job)
    canonical.parent.mkdir(parents=True)
    canonical.write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="schema bytes changed"):
        recovery._guard_replacement(tmp_path, job)


def test_attempt_one_canonical_rejects_any_failure_artifact(tmp_path) -> None:
    job = recovery._replacement_job(_job())
    attempts = base.analysis_root(tmp_path) / "attempts" / job.scope / job.rubric_id
    attempts.mkdir(parents=True)
    (attempts / f"{job.job_id}.1.failed.json").write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="unexpected failure artifacts"):
        recovery._validate_replacement_attempt_history(tmp_path, job, {"attempt_index": 1})


def test_attempt_two_canonical_accepts_one_exact_transport_failure(tmp_path) -> None:
    job = recovery._replacement_job(_job())
    attempts = base.analysis_root(tmp_path) / "attempts" / job.scope / job.rubric_id
    attempts.mkdir(parents=True)
    failure = {
        "status": "failed_transport",
        "job_id": job.job_id,
        "scope": job.scope,
        "rubric_id": job.rubric_id,
        "pass_index": job.pass_index,
        "chunk_index": job.chunk_index,
        "attempt_index": 1,
        "instrument_fp": job.instrument_fp,
        "prompt_sha256": base._sha256_text(job.prompt),
        "prompt_tokens_o200k": job.prompt_tokens_o200k,
        "source_item_ids": [item.source_item_id for item in job.items],
        "opaque_item_ids": [item.opaque_id for item in job.items],
        "request": job.prompt,
    }
    (attempts / f"{job.job_id}.1.failed.json").write_text(json.dumps(failure), encoding="utf-8")

    recovery._validate_replacement_attempt_history(tmp_path, job, {"attempt_index": 2})


@pytest.mark.parametrize("status", ["failed_content", "failed_transport"])
def test_attempt_two_canonical_rejects_content_or_mismatched_failure(status, tmp_path) -> None:
    job = recovery._replacement_job(_job())
    attempts = base.analysis_root(tmp_path) / "attempts" / job.scope / job.rubric_id
    attempts.mkdir(parents=True)
    failure = {
        "status": status,
        "job_id": job.job_id,
        "scope": job.scope,
        "rubric_id": job.rubric_id,
        "pass_index": job.pass_index,
        "chunk_index": job.chunk_index,
        "attempt_index": 1,
        "instrument_fp": job.instrument_fp,
        "prompt_sha256": base._sha256_text(job.prompt),
        "prompt_tokens_o200k": job.prompt_tokens_o200k,
        "source_item_ids": [item.source_item_id for item in job.items],
        "opaque_item_ids": [item.opaque_id for item in job.items],
        "request": job.prompt,
    }
    if status == "failed_transport":
        failure["prompt_sha256"] = "0" * 64
    (attempts / f"{job.job_id}.1.failed.json").write_text(json.dumps(failure), encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="ineligible pre-canonical failure"):
        recovery._validate_replacement_attempt_history(tmp_path, job, {"attempt_index": 2})


def test_attempt_two_canonical_rejects_extra_failures(tmp_path) -> None:
    job = recovery._replacement_job(_job())
    attempts = base.analysis_root(tmp_path) / "attempts" / job.scope / job.rubric_id
    attempts.mkdir(parents=True)
    for index in (1, 2):
        (attempts / f"{job.job_id}.{index}.failed.json").write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="attempt history changed"):
        recovery._validate_replacement_attempt_history(tmp_path, job, {"attempt_index": 2})
