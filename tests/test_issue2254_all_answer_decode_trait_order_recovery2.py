"""Focused tests for the exact #2254 v21 trait-order replacement."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_trait_order_recovery as order1
import scripts.issue2254_all_answer_decode_trait_order_recovery2 as recovery
import scripts.issue2254_revmap8_subagent_grade as runner


def _job(n_items: int = 7) -> runner.JobSpec:
    items = tuple(
        runner.GradeItem(
            source_item_id=f"source-{index}",
            opaque_id=f"i{index:020d}",
            cell_id="sycophancy__rb__decodeonly__L14__c1",
            behavior="sycophancy",
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
    prompt = "fixture sycophancy trait prompt"
    return runner.JobSpec(
        scope="trait_production",
        rubric_id="trait_sycophancy",
        pass_index=0,
        chunk_index=13,
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
    sibling = replace(job, chunk_index=12)

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
    assert recovery.REPLACEMENT_SUFFIX != order1.REPLACEMENT_SUFFIX


def test_replacement_is_exact_target_only() -> None:
    job = _job()
    with pytest.raises(base.AnalysisError, match="matched 0 parents"):
        recovery.apply_trait_order_replacement([replace(job, chunk_index=12)], _receipt(job))


def test_order_audit_binds_rotation_and_discards_all_scores(monkeypatch) -> None:
    job = _job()
    expected = [item.opaque_id for item in job.items]
    returned = expected[5:] + expected[:5]
    failure = _failure(job, returned)
    monkeypatch.setattr(
        runner,
        "_validate_codex_events",
        lambda expected_job, stdout: ([{"type": "turn.completed"}], "thread"),
    )
    monkeypatch.setattr(
        recovery,
        "TARGET_RAW_RESPONSE_SHA256",
        base._sha256_text(failure["raw_response"]),
    )
    monkeypatch.setattr(recovery, "TARGET_RETURNED_ORDER_SHA256", base._canonical_sha256(returned))
    monkeypatch.setattr(recovery, "TARGET_ORDER_MISMATCH_INDICES", list(range(7)))
    monkeypatch.setattr(
        recovery,
        "TARGET_RETURNED_EXPECTED_POSITION_SEQUENCE",
        [5, 6, 0, 1, 2, 3, 4],
    )

    audit = recovery._structural_failure_audit(job, failure)

    assert audit["returned_expected_position_sequence_at_mismatches"] == [5, 6, 0, 1, 2, 3, 4]
    assert audit["all_rows_discarded"] is True
    assert audit["scores_salvaged"] == 0
    assert audit["score_values_used_for_eligibility"] is False
    assert "scores" not in audit


def test_order_audit_rejects_a_different_rotation(monkeypatch) -> None:
    job = _job()
    expected = [item.opaque_id for item in job.items]
    returned = expected[1:] + expected[:1]
    failure = _failure(job, returned)
    monkeypatch.setattr(
        runner, "_validate_codex_events", lambda expected_job, stdout: ([], "thread")
    )
    monkeypatch.setattr(
        recovery,
        "TARGET_RAW_RESPONSE_SHA256",
        base._sha256_text(failure["raw_response"]),
    )
    monkeypatch.setattr(recovery, "TARGET_RETURNED_ORDER_SHA256", base._canonical_sha256(returned))
    monkeypatch.setattr(recovery, "TARGET_ORDER_MISMATCH_INDICES", list(range(7)))
    monkeypatch.setattr(
        recovery,
        "TARGET_RETURNED_EXPECTED_POSITION_SEQUENCE",
        [5, 6, 0, 1, 2, 3, 4],
    )

    with pytest.raises(base.AnalysisError, match="rotation sequence changed"):
        recovery._structural_failure_audit(job, failure)


def test_non_target_unregistered_failure_still_blocks(monkeypatch, tmp_path) -> None:
    job = _job()
    other = replace(job, chunk_index=12)
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


def test_launch_lease_written_before_new_replacement_call(monkeypatch, tmp_path) -> None:
    parent = _job()
    job = recovery._replacement_job(parent)
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    receipt = recovery._receipt_path(tmp_path)
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"entry_sha256":"fixture"}\n', encoding="utf-8")
    recovery._manifest_path(tmp_path).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "_ACTIVE_OUT_ROOT", tmp_path)
    monkeypatch.setattr(recovery, "_current_target", lambda args: parent)
    monkeypatch.setattr(recovery, "_load_receipt", lambda out_root, target: {})
    observed = {}

    def fake_run(args, root, delegated_job):
        observed["lease_exists"] = recovery._lease_path(tmp_path).is_file()
        observed["job"] = delegated_job
        return {"status": "complete"}

    monkeypatch.setattr(recovery, "_ORIGINAL_RUN_ONE_JOB", fake_run)

    result = recovery._run_one_job_with_trait_order2_lease(None, tmp_path, job)

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


def test_non_v21_jobs_route_through_complete_v20_dispatch(monkeypatch, tmp_path) -> None:
    old = replace(_job(), job_suffix=order1.REPLACEMENT_SUFFIX)
    observed = {}

    def fake_old(args, root, job):
        observed["job"] = job
        return {"status": "complete"}

    monkeypatch.setattr(order1, "_run_one_job_with_trait_order_lease", fake_old)

    result = recovery._run_one_job_with_trait_order2_lease(None, tmp_path, old)

    assert result == {"status": "complete"}
    assert observed["job"] is old


def test_non_sycophancy_roster_composes_v20(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(recovery, "_ACTIVE_OUT_ROOT", tmp_path)
    observed = {}

    def fake_old(items, instrument, rubrics, rubric_id):
        observed["rubric_id"] = rubric_id
        return ["old-composed-job"]

    monkeypatch.setattr(order1, "_fully_recovered_jobs", fake_old)

    result = recovery._fully_recovered_jobs([], {}, {}, "trait_evil")

    assert result == ["old-composed-job"]
    assert observed["rubric_id"] == "trait_evil"
