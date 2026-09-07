"""Tests for exact, lossless policy-packet recovery in the #2254 grader."""

from __future__ import annotations

from dataclasses import replace

import pytest

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_recovery as recovery
import scripts.issue2254_revmap8_subagent_grade as runner


def _job(*, job_suffix: str = "") -> runner.JobSpec:
    items = tuple(
        runner.GradeItem(
            source_item_id=f"source-{index:02d}",
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
        for index in range(6)
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
        instrument_fp="fixture-instrument",
        job_suffix=job_suffix,
    )


def _record(job: runner.JobSpec, *, parts: int = 2) -> dict:
    return {
        "version": recovery.RECOVERY_VERSION,
        "job_id": job.job_id,
        "scope": job.scope,
        "rubric_id": job.rubric_id,
        "pass_index": job.pass_index,
        "chunk_index": job.chunk_index,
        "prompt_sha256": base._sha256_text(job.prompt),
        "prompt_tokens_o200k": job.prompt_tokens_o200k,
        "instrument_fp": job.instrument_fp,
        "parts": parts,
        "n_items": len(job.items),
        "source_item_ids_sha256": base._canonical_sha256(
            [item.source_item_id for item in job.items]
        ),
        "opaque_item_ids_sha256": base._canonical_sha256(
            [item.opaque_id for item in job.items]
        ),
        "children": [],
    }


def _pin_fixture(monkeypatch, job: runner.JobSpec) -> None:
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
    width = len(job.items) // recovery.TARGET_PARTS
    children = []
    for part_index in range(recovery.TARGET_PARTS):
        items = job.items[part_index * width : (part_index + 1) * width]
        prompt = base._integrity_prompt(list(items))
        children.append(
            {
                "job_suffix": f"__policy_split{part_index:02d}",
                "n_items": len(items),
                "prompt_tokens_o200k": runner._count_o200k_tokens(prompt),
                "prompt_sha256": base._sha256_text(prompt),
            }
        )
    monkeypatch.setattr(recovery, "TARGET_CHILDREN", tuple(children))


def test_registered_parent_splits_losslessly_in_contiguous_order(monkeypatch) -> None:
    parent = _job()
    _pin_fixture(monkeypatch, parent)
    record = _record(parent)
    record["children"] = list(recovery.TARGET_CHILDREN)
    jobs = recovery.apply_policy_packet_splits(
        [parent], {"coherence": "fixture rubric"}, {parent.job_id: record}
    )

    assert [job.job_suffix for job in jobs] == [
        "__policy_split00",
        "__policy_split01",
    ]
    assert [len(job.items) for job in jobs] == [3, 3]
    assert [item.source_item_id for job in jobs for item in job.items] == [
        item.source_item_id for item in parent.items
    ]
    assert len({item.opaque_id for job in jobs for item in job.items}) == len(parent.items)
    assert sum(len(job.items) for job in jobs) == len(parent.items)
    assert all(job.instrument_fp == parent.instrument_fp for job in jobs)
    assert all(job.rubric_id == parent.rubric_id for job in jobs)
    assert all(job.pass_index == parent.pass_index for job in jobs)


def test_only_exact_registered_parent_is_split(monkeypatch) -> None:
    parent = _job()
    _pin_fixture(monkeypatch, parent)
    record = _record(parent)
    record["children"] = list(recovery.TARGET_CHILDREN)
    sibling = replace(parent, chunk_index=22)
    jobs = recovery.apply_policy_packet_splits(
        [sibling, parent], {"coherence": "fixture rubric"}, {parent.job_id: record}
    )

    assert jobs[0] is sibling
    assert [job.job_id for job in jobs[1:]] == [
        f"{parent.job_id}__policy_split00",
        f"{parent.job_id}__policy_split01",
    ]


def test_recovery_rejects_any_parent_prompt_drift(monkeypatch) -> None:
    parent = _job()
    _pin_fixture(monkeypatch, parent)
    record = _record(parent)
    record["children"] = list(recovery.TARGET_CHILDREN)
    drifted = replace(parent, prompt=parent.prompt + " drift")

    with pytest.raises(base.AnalysisError, match="single frozen exception"):
        recovery.apply_policy_packet_splits(
            [drifted], {"coherence": "fixture rubric"}, {parent.job_id: record}
        )


def test_recovery_rejects_unmatched_registry_entry(monkeypatch) -> None:
    parent = _job()
    _pin_fixture(monkeypatch, parent)
    record = _record(parent)
    record["children"] = list(recovery.TARGET_CHILDREN)
    sibling = replace(parent, chunk_index=22)

    with pytest.raises(base.AnalysisError, match="did not match roster"):
        recovery.apply_policy_packet_splits(
            [sibling], {"coherence": "fixture rubric"}, {parent.job_id: record}
        )


def test_recovery_rejects_nondivisible_split_without_dropping_items(monkeypatch) -> None:
    parent = _job()
    _pin_fixture(monkeypatch, parent)
    record = _record(parent, parts=4)
    record["children"] = list(recovery.TARGET_CHILDREN)

    with pytest.raises(base.AnalysisError, match="invalid registered parts"):
        recovery.apply_policy_packet_splits(
            [parent],
            {"coherence": "fixture rubric"},
            {parent.job_id: record},
        )


def test_real_recovery_constants_are_exactly_frozen() -> None:
    assert recovery.TARGET_JOB_ID == "integrity_production__coherence__pass00__chunk023"
    assert recovery.TARGET_N_ITEMS == 60
    assert recovery.TARGET_PROMPT_TOKENS == 38_001
    assert recovery.TARGET_PARTS == 2
    assert recovery.TARGET_PROMPT_SHA256 == (
        "e3e27d170be0b00d1d71859ef23a6f173e608ba2fe0345266c3a8c3ec61706ef"
    )
    assert recovery.TARGET_INSTRUMENT_FP == (
        "b779778050d06771805ef65d79ed74454fc2ac89b4067c25272662f90c380606"
    )
    assert {child["n_items"] for child in recovery.TARGET_CHILDREN} == {30}
