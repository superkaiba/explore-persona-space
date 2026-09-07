"""Focused tests for the exact second-level #2254 policy recovery."""

from __future__ import annotations

from dataclasses import replace

import pytest

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_recovery2 as recovery
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
        instrument_fp="fixture-fp",
        job_suffix="__policy_split01",
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
    children = []
    for index in range(2):
        items = job.items[index * 3 : (index + 1) * 3]
        prompt = base._integrity_prompt(list(items))
        children.append(
            {
                "job_suffix": f"{job.job_suffix}__policy_split{index:02d}",
                "n_items": len(items),
                "prompt_tokens_o200k": runner._count_o200k_tokens(prompt),
                "prompt_sha256": base._sha256_text(prompt),
            }
        )
    monkeypatch.setattr(recovery, "TARGET_CHILDREN", tuple(children))
    return {job.job_id: {"children": children}}


def test_second_split_is_lossless_and_leaves_sibling_unchanged(monkeypatch) -> None:
    parent = _fixture_job()
    registry = _pin(monkeypatch, parent)
    sibling = replace(parent, chunk_index=24, job_suffix="")

    jobs = recovery.apply_policy_packet_split(
        [sibling, parent], {"coherence": "fixture rubric"}, registry
    )

    assert jobs[0] is sibling
    assert [len(job.items) for job in jobs[1:]] == [3, 3]
    assert [item.source_item_id for job in jobs[1:] for item in job.items] == [
        item.source_item_id for item in parent.items
    ]
    assert all(job.instrument_fp == parent.instrument_fp for job in jobs[1:])
    assert all(job.pass_index == parent.pass_index for job in jobs[1:])


def test_second_split_fails_closed_on_parent_drift(monkeypatch) -> None:
    parent = _fixture_job()
    registry = _pin(monkeypatch, parent)

    with pytest.raises(base.AnalysisError, match="frozen exception"):
        recovery.apply_policy_packet_split(
            [replace(parent, prompt=parent.prompt + " drift")],
            {"coherence": "fixture rubric"},
            registry,
        )


def test_second_split_fails_when_registered_target_is_absent(monkeypatch) -> None:
    parent = _fixture_job()
    registry = _pin(monkeypatch, parent)

    with pytest.raises(base.AnalysisError, match="did not match roster"):
        recovery.apply_policy_packet_split(
            [replace(parent, chunk_index=22)], {"coherence": "fixture rubric"}, registry
        )


def test_target_roster_fails_closed_before_second_recovery_registration(monkeypatch) -> None:
    parent = _fixture_job()
    _pin(monkeypatch, parent)

    with pytest.raises(base.AnalysisError, match="split is not registered"):
        recovery.apply_policy_packet_split([parent], {"coherence": "fixture rubric"}, {})


def test_empty_registry_is_noop_only_when_target_is_absent(monkeypatch) -> None:
    parent = _fixture_job()
    _pin(monkeypatch, parent)
    sibling = replace(parent, chunk_index=22, job_suffix="")

    assert recovery.apply_policy_packet_split(
        [sibling], {"coherence": "fixture rubric"}, {}
    ) == [sibling]


def test_real_second_recovery_constants_are_exact() -> None:
    assert recovery.TARGET_JOB_ID.endswith("chunk023__policy_split01")
    assert recovery.TARGET_N_ITEMS == 30
    assert recovery.TARGET_PROMPT_TOKENS == 18_533
    assert recovery.TARGET_PARTS == 2
    assert recovery.TARGET_PROMPT_SHA256 == (
        "cb41fd503164532922e874347db962096d2c449a6d20bef4cfafff60987804a6"
    )
    assert [child["n_items"] for child in recovery.TARGET_CHILDREN] == [15, 15]
