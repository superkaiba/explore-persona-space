"""Focused tests for the terminal singleton #2254 policy recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_recovery3 as recovery
import scripts.issue2254_revmap8_subagent_grade as runner


def _fixture_job() -> runner.JobSpec:
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
        pass_index=0,
        chunk_index=23,
        items=items,
        prompt=prompt,
        prompt_tokens_o200k=runner._count_o200k_tokens(prompt),
        instrument_fp="fixture-fp",
        job_suffix="__policy_split01__policy_split00",
    )


def _pin(monkeypatch, job: runner.JobSpec) -> dict:
    monkeypatch.setattr(recovery, "TARGET_JOB_ID", job.job_id)
    monkeypatch.setattr(recovery, "TARGET_PROMPT_SHA256", base._sha256_text(job.prompt))
    monkeypatch.setattr(
        recovery,
        "TARGET_SOURCE_IDS_SHA256",
        base._canonical_sha256([item.source_item_id for item in job.items]),
    )
    monkeypatch.setattr(
        recovery,
        "TARGET_OPAQUE_IDS_SHA256",
        base._canonical_sha256([item.opaque_id for item in job.items]),
    )
    monkeypatch.setattr(recovery, "TARGET_INSTRUMENT_FP", job.instrument_fp)
    monkeypatch.setattr(recovery, "TARGET_N_ITEMS", len(job.items))
    monkeypatch.setattr(recovery, "TARGET_PROMPT_TOKENS", job.prompt_tokens_o200k)
    monkeypatch.setattr(recovery, "TARGET_PARTS", len(job.items))
    children = recovery._child_metadata(job, "fixture rubric")
    return {job.job_id: {"children": children}}


def test_singleton_split_is_complete_ordered_and_metadata_preserving(monkeypatch) -> None:
    parent = _fixture_job()
    registry = _pin(monkeypatch, parent)
    sibling = replace(parent, chunk_index=24, job_suffix="")

    jobs = recovery.apply_policy_packet_split(
        [sibling, parent], {"coherence": "fixture rubric"}, registry
    )

    assert jobs[0] is sibling
    assert len(jobs[1:]) == len(parent.items)
    assert all(len(job.items) == 1 for job in jobs[1:])
    assert [job.items[0].source_item_id for job in jobs[1:]] == [
        item.source_item_id for item in parent.items
    ]
    assert [job.items[0].opaque_id for job in jobs[1:]] == [
        item.opaque_id for item in parent.items
    ]
    assert all(job.instrument_fp == parent.instrument_fp for job in jobs[1:])
    assert all(job.pass_index == parent.pass_index for job in jobs[1:])


def test_singleton_split_fails_closed_before_registration(monkeypatch) -> None:
    parent = _fixture_job()
    _pin(monkeypatch, parent)

    with pytest.raises(base.AnalysisError, match="not registered"):
        recovery.apply_policy_packet_split([parent], {"coherence": "fixture rubric"}, {})


def test_non_target_roster_is_unchanged_without_registry(monkeypatch) -> None:
    parent = _fixture_job()
    _pin(monkeypatch, parent)
    sibling = replace(parent, chunk_index=22, job_suffix="")

    assert recovery.apply_policy_packet_split(
        [sibling], {"coherence": "fixture rubric"}, {}
    ) == [sibling]


def test_singleton_split_rejects_parent_drift(monkeypatch) -> None:
    parent = _fixture_job()
    registry = _pin(monkeypatch, parent)

    with pytest.raises(base.AnalysisError, match="frozen exception"):
        recovery.apply_policy_packet_split(
            [replace(parent, prompt=parent.prompt + " drift")],
            {"coherence": "fixture rubric"},
            registry,
        )


def test_restart_with_orphaned_singleton_failure_halts_before_grader(
    monkeypatch, tmp_path
) -> None:
    parent = _fixture_job()
    registry = _pin(monkeypatch, parent)
    children = recovery.apply_policy_packet_split(
        [parent], {"coherence": "fixture rubric"}, registry
    )
    child = children[0]
    attempt_dir = tmp_path / "attempts" / child.scope / child.rubric_id
    attempt_dir.mkdir(parents=True)
    (attempt_dir / f"{child.job_id}.1.failed.json").write_text(
        json.dumps(
            {
                "job_id": child.job_id,
                "prompt_sha256": base._sha256_text(child.prompt),
                "instrument_fp": child.instrument_fp,
                "source_item_ids": [item.source_item_id for item in child.items],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot" / "integrity.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )
    run_jobs_called = False

    def forbidden_run_jobs(*unused_args, **unused_kwargs):
        nonlocal run_jobs_called
        run_jobs_called = True

    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    monkeypatch.setattr(base, "_load_staged", lambda args: ([], {}, {"coherence": "x"}))
    monkeypatch.setattr(recovery, "_prior_recovered_jobs", lambda *args: [parent])
    monkeypatch.setattr(recovery, "_load_registry", lambda unused: registry)
    monkeypatch.setattr(recovery, "_ACTIVE_OUT_ROOT", tmp_path)
    monkeypatch.setattr(base, "_production_jobs", recovery._fully_recovered_jobs)
    monkeypatch.setattr(base, "_run_jobs", forbidden_run_jobs)

    with pytest.raises(base.AnalysisError, match="forbids automatic retry"):
        base.phase_integrity_production(SimpleNamespace(out_root=tmp_path))
    assert run_jobs_called is False


def test_real_final_recovery_constants_are_exact() -> None:
    assert recovery.TARGET_JOB_ID.endswith(
        "chunk023__policy_split01__policy_split00"
    )
    assert recovery.TARGET_N_ITEMS == 15
    assert recovery.TARGET_PROMPT_TOKENS == 8_288
    assert recovery.TARGET_PARTS == 15
    assert recovery.TARGET_PROMPT_SHA256 == (
        "ecb1f9e7ceb94dbd94568e89520fadf2229fffbeae816fb91dc01e51736ee691"
    )
