"""Focused tests for the exact #2254 v20 trait-order replacement."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_structured_recovery as structured
import scripts.issue2254_all_answer_decode_trait_order_recovery as recovery
import scripts.issue2254_revmap8_subagent_grade as runner


def _job(n_items: int = 4) -> runner.JobSpec:
    items = tuple(
        runner.GradeItem(
            source_item_id=f"source-{index}",
            opaque_id=f"i{index:020d}",
            cell_id="evil__rb__decodeonly__L14__c1",
            behavior="evil",
            arm="decodeonly",
            phase="trait_production",
            qi=index,
            seed=42,
            context_index=index,
            draw_index=0,
            question=f"Question {index}?",
            answer=f"Answer {index}.",
        )
        for index in range(n_items)
    )
    prompt = "fixture trait prompt"
    return runner.JobSpec(
        scope="trait_production",
        rubric_id="trait_evil",
        pass_index=0,
        chunk_index=10,
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


def _failure(job: runner.JobSpec, returned_ids: list[str]) -> dict:
    raw = json.dumps(
        {
            "rubric_id": job.rubric_id,
            "scores": [
                {"item_id": item_id, "score": 90 if index % 2 else "REFUSAL"}
                for index, item_id in enumerate(returned_ids)
            ],
        },
        separators=(",", ":"),
    )
    return {
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


def test_replacement_changes_only_bookkeeping_identity() -> None:
    job = _job()
    sibling = replace(job, chunk_index=9)

    output = recovery.apply_trait_order_replacement([sibling, job], _receipt(job))
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
    assert not recovery.REPLACEMENT_SUFFIX.endswith(structured.REPLACEMENT_SUFFIX)


def test_replacement_is_exact_target_only() -> None:
    job = _job()
    with pytest.raises(base.AnalysisError, match="matched 0 parents"):
        recovery.apply_trait_order_replacement([replace(job, chunk_index=9)], _receipt(job))


def test_order_only_audit_discards_complete_bijection(monkeypatch) -> None:
    job = _job()
    expected = [item.opaque_id for item in job.items]
    returned = expected[1:] + expected[:1]
    failure = _failure(job, returned)
    monkeypatch.setattr(
        runner,
        "_validate_codex_events",
        lambda expected_job, stdout: ([{"type": "turn.completed"}], "thread"),
    )

    audit = recovery._structural_failure_audit(
        job,
        failure,
        expected_raw_response_sha256=recovery._sha256_text(failure["raw_response"]),
        expected_returned_order_sha256=base._canonical_sha256(returned),
        expected_mismatch_indices=list(range(len(expected))),
    )

    assert audit == {
        "returned_rows": 4,
        "returned_unique_ids": 4,
        "returned_ids_sha256": base._canonical_sha256(returned),
        "expected_ids_sha256": base._canonical_sha256(expected),
        "exact_expected_id_multiset": True,
        "duplicate_ids": [],
        "omitted_ids": [],
        "unexpected_ids": [],
        "order_mismatch_indices_zero_based": [0, 1, 2, 3],
        "grader_thread_id": "thread",
        "codex_event_count": 1,
        "codex_event_types": ["turn.completed"],
        "codex_stdout_sha256": recovery._sha256_text("fixture events"),
        "tool_free_completed_event_audit": True,
        "all_rows_discarded": True,
        "scores_salvaged": 0,
        "score_values_used_for_eligibility": False,
    }
    assert "scores" not in audit


@pytest.mark.parametrize(
    "returned",
    [
        ["i00000000000000000000"] * 4,
        [f"i{index:020d}" for index in (0, 1, 2, 9)],
        [f"i{index:020d}" for index in range(4)],
    ],
)
def test_order_audit_rejects_non_bijection_or_already_valid_order(monkeypatch, returned) -> None:
    job = _job()
    failure = _failure(job, returned)
    monkeypatch.setattr(
        runner, "_validate_codex_events", lambda expected_job, stdout: ([], "thread")
    )

    with pytest.raises(base.AnalysisError, match="structural defect changed"):
        recovery._structural_failure_audit(
            job,
            failure,
            expected_raw_response_sha256=recovery._sha256_text(failure["raw_response"]),
            expected_returned_order_sha256=base._canonical_sha256(returned),
            expected_mismatch_indices=[
                index
                for index, (expected, actual) in enumerate(
                    zip([item.opaque_id for item in job.items], returned, strict=True)
                )
                if expected != actual
            ],
        )


def test_order_audit_rejects_invalid_score_without_salvage(monkeypatch) -> None:
    job = _job()
    expected = [item.opaque_id for item in job.items]
    returned = [*expected[1:], job.items[0].opaque_id]
    failure = _failure(job, returned)
    obj = json.loads(failure["raw_response"])
    obj["scores"][0]["score"] = 101
    failure["raw_response"] = json.dumps(obj, separators=(",", ":"))
    monkeypatch.setattr(
        runner, "_validate_codex_events", lambda expected_job, stdout: ([], "thread")
    )

    with pytest.raises(base.AnalysisError, match="invalid score"):
        recovery._structural_failure_audit(
            job,
            failure,
            expected_raw_response_sha256=recovery._sha256_text(failure["raw_response"]),
            expected_returned_order_sha256=base._canonical_sha256(returned),
            expected_mismatch_indices=list(range(4)),
        )


def test_non_target_unregistered_failure_still_blocks(monkeypatch, tmp_path) -> None:
    job = _job()
    other = replace(job, chunk_index=9)
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    attempts = tmp_path / "attempts" / other.scope / other.rubric_id
    attempts.mkdir(parents=True)
    (attempts / f"{other.job_id}.1.failed.json").write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="forbids more attempts"):
        recovery._guard_unregistered_failures_except_target(tmp_path, [other], {})


def test_orphaned_replacement_artifact_is_terminal(monkeypatch, tmp_path) -> None:
    job = recovery._replacement_job(_job())
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    schema = recovery._schema_path(tmp_path, job)
    schema.parent.mkdir(parents=True)
    schema.write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="requires new adjudication"):
        recovery._guard_replacement(tmp_path, job)


def test_launch_lease_written_before_replacement_call(monkeypatch, tmp_path) -> None:
    parent = _job()
    job = recovery._replacement_job(parent)
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    receipt = recovery._receipt_path(tmp_path)
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"entry_sha256":"fixture"}\n', encoding="utf-8")
    manifest = recovery._manifest_path(tmp_path)
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "_ACTIVE_OUT_ROOT", tmp_path)
    monkeypatch.setattr(recovery, "_current_target", lambda args: parent)
    monkeypatch.setattr(recovery, "_load_receipt", lambda out_root, target: {})
    observed = {}

    def fake_run(args, root, delegated_job):
        observed["lease_exists"] = recovery._lease_path(tmp_path).is_file()
        observed["job"] = delegated_job
        return {"status": "complete"}

    monkeypatch.setattr(recovery, "_ORIGINAL_RUN_ONE_JOB", fake_run)

    result = recovery._run_one_job_with_trait_order_lease(None, tmp_path, job)

    assert result == {"status": "complete"}
    assert observed == {"lease_exists": True, "job": job}


def test_launch_lease_exclusively_consumes_slot(monkeypatch, tmp_path) -> None:
    parent = _job()
    replacement = recovery._replacement_job(parent)
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    receipt = recovery._receipt_path(tmp_path)
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"entry_sha256":"fixture"}\n', encoding="utf-8")
    recovery._manifest_path(tmp_path).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "_current_target", lambda args: parent)
    monkeypatch.setattr(recovery, "_load_receipt", lambda out_root, target: {})

    recovery._write_launch_lease(tmp_path, replacement)

    with pytest.raises(base.AnalysisError, match="already consumed"):
        recovery._write_launch_lease(tmp_path, replacement)


def test_attempt_two_accepts_one_response_free_transport_failure(tmp_path) -> None:
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
def test_attempt_history_rejects_content_or_mutated_transport(status, tmp_path) -> None:
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


def test_v18_replacement_route_remains_composed(monkeypatch, tmp_path) -> None:
    old = replace(_job(), job_suffix=structured.REPLACEMENT_SUFFIX)
    observed = {}

    def fake_old(args, root, job):
        observed["job"] = job
        return {"status": "complete"}

    monkeypatch.setattr(structured, "_run_one_job_with_structured_lease", fake_old)

    result = recovery._run_one_job_with_trait_order_lease(None, tmp_path, old)

    assert result == {"status": "complete"}
    assert observed["job"] is old
