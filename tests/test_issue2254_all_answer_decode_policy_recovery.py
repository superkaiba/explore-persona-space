"""Focused tests for universal #2254 production-packet recovery."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_policy_recovery as recovery
import scripts.issue2254_revmap8_subagent_grade as runner


def _job(rubric_id: str = "coherence") -> runner.JobSpec:
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
        rubric_id=rubric_id,
        pass_index=1,
        chunk_index=44,
        items=items,
        prompt=prompt,
        prompt_tokens_o200k=runner._count_o200k_tokens(prompt),
        instrument_fp="fixture-fp",
    )


def _record(job: runner.JobSpec) -> dict:
    return {
        **recovery._parent_metadata(job),
        "children": recovery._child_metadata(job, "fixture rubric"),
    }


def test_registered_parent_becomes_ordered_singletons_and_sibling_is_unchanged() -> None:
    parent = _job()
    sibling = replace(parent, chunk_index=43)
    registry = {parent.job_id: _record(parent)}

    jobs = recovery.apply_policy_recoveries(
        [sibling, parent], {"coherence": "fixture rubric"}, registry
    )

    assert jobs[0] is sibling
    assert len(jobs[1:]) == len(parent.items)
    assert [job.items[0].source_item_id for job in jobs[1:]] == [
        item.source_item_id for item in parent.items
    ]
    assert [job.items[0].opaque_id for job in jobs[1:]] == [
        item.opaque_id for item in parent.items
    ]
    assert all(job.instrument_fp == parent.instrument_fp for job in jobs[1:])
    assert sum(len(job.items) for job in jobs) == sum(
        len(job.items) for job in (sibling, parent)
    )


def test_parent_metadata_drift_fails_closed() -> None:
    parent = _job()
    registry = {parent.job_id: _record(parent)}

    with pytest.raises(base.AnalysisError, match="parent metadata changed"):
        recovery.apply_policy_recoveries(
            [replace(parent, prompt=parent.prompt + " drift")],
            {"coherence": "fixture rubric"},
            registry,
        )


def test_unregistered_orphan_failure_blocks_attempts(monkeypatch, tmp_path) -> None:
    job = _job()
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    attempts = tmp_path / "attempts" / job.scope / job.rubric_id
    attempts.mkdir(parents=True)
    (attempts / f"{job.job_id}.1.failed.json").write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="forbids more attempts"):
        recovery._guard_unregistered_failures(tmp_path, [job], {})


def test_registered_singleton_orphan_failure_blocks_retry(monkeypatch, tmp_path) -> None:
    parent = _job()
    record = _record(parent)
    children = recovery.apply_policy_recoveries(
        [parent], {"coherence": "fixture rubric"}, {parent.job_id: record}
    )
    child = children[0]
    monkeypatch.setattr(base, "analysis_root", lambda unused: tmp_path)
    attempts = tmp_path / "attempts" / child.scope / child.rubric_id
    attempts.mkdir(parents=True)
    (attempts / f"{child.job_id}.1.failed.json").write_text("{}", encoding="utf-8")

    with pytest.raises(base.AnalysisError, match="forbids automatic retry"):
        recovery._guard_registered_singletons(
            tmp_path, children, {parent.job_id: record}
        )


def test_chain_hash_covers_every_receipt_field() -> None:
    payload = {
        "version": recovery.RECOVERY_VERSION,
        "registration_index": 0,
        "prior_registry_entry_sha256": None,
        "job_id": "fixture",
    }
    digest = base._canonical_sha256(payload)

    assert digest != base._canonical_sha256({**payload, "job_id": "changed"})
    assert json.loads(json.dumps({**payload, "entry_sha256": digest}))["entry_sha256"] == digest


def test_singleton_descendant_marker_is_unambiguous() -> None:
    parent = _job()
    children = recovery._child_metadata(parent, "fixture rubric")

    assert len(children) == len(parent.items)
    assert all("__policy_singleton" in child["job_suffix"] for child in children)
    assert len({child["job_suffix"] for child in children}) == len(children)
