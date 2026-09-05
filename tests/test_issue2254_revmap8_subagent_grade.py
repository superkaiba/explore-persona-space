"""Focused tests for the issue #2254 Codex-subagent sensitivity grader."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

import scripts.issue2254_revmap8_subagent_grade as sg


def _record(cell: dict, cid: str) -> dict:
    """Small-text production-shape fixture with exactly 200 generations."""
    return {
        "cell_id": cid,
        "cell": cell,
        "q_of_context": list(range(20)),
        "seeds": {
            str(seed): {
                "completions": [
                    [f"fixture answer {cid} q{qi} seed{seed} draw{draw}" for draw in range(5)]
                    for qi in range(20)
                ],
                "condition_passes": [True] * 20,
            }
            for seed in (42, 43)
        },
        "cap_hit_fraction": 0.0,
    }


def _records() -> dict[str, dict]:
    records = {}
    for cell in sg.r8.registered_cells():
        cid = sg.r8._cell_id(cell)
        records[cid] = _record(cell, cid)
    for behavior in sg.r8.ROUND_BEHAVIORS:
        for kind, suffix in (("alpha0", "a0"), ("ceiling", "cl")):
            cid = f"{behavior}__{suffix}"
            records[cid] = _record({"behavior": behavior, "kind": kind}, cid)
    return records


def _instrument() -> dict:
    rubrics = sg.load_frozen_rubrics()
    return sg._base_instrument_manifest(rubrics, "codex-cli fixture")


def _one_item() -> sg.GradeItem:
    return sg.GradeItem(
        source_item_id="secret_cell|s42|c0|d0",
        opaque_id="i0123456789abcdefabcd",
        cell_id="secret_cell",
        behavior="evil",
        arm="steer",
        phase="steer",
        qi=0,
        seed=42,
        context_index=0,
        draw_index=0,
        question="Neutral fixture question?",
        answer="Neutral fixture answer.",
    )


def _job(rubric_id: str = "trait_evil") -> sg.JobSpec:
    rubrics = sg.load_frozen_rubrics()
    item = _one_item()
    prompt = sg._prompt(rubric_id, rubrics[rubric_id], [item])
    return sg.JobSpec(
        scope="pilot",
        rubric_id=rubric_id,
        pass_index=0,
        chunk_index=0,
        items=(item,),
        prompt=prompt,
        prompt_tokens_o200k=sg._count_o200k_tokens(prompt),
        instrument_fp="fixture-instrument",
    )


def _response(job: sg.JobSpec, score: object = 42) -> dict:
    row = {"item_id": job.items[0].opaque_id, "score": score}
    if job.rubric_id == "coherence":
        row["reasoning"] = "The response is fluent."
    return {"rubric_id": job.rubric_id, "scores": [row]}


def test_frozen_rubric_hashes_and_honest_instrument():
    rubrics = sg.load_frozen_rubrics()
    assert {name: sg._sha256_text(text) for name, text in rubrics.items()} == sg.RUBRIC_SHA256
    instrument = sg._base_instrument_manifest(rubrics, "codex-cli fixture")
    assert instrument["model"] == "gpt-5.6-sol"
    assert instrument["reasoning_effort"] == "low"
    assert instrument["provider_path"] == "codex exec"
    assert "sonnet" not in instrument["instrument_name"].lower()
    assert instrument["repeat_interpretation"].endswith("independence is unverified.")
    assert instrument["cjk"].startswith("Existing programmatic audit only")


def test_frozen_instrument_can_be_read_posthoc_after_live_cli_patch(monkeypatch):
    rubrics = sg.load_frozen_rubrics()
    instrument = sg._base_instrument_manifest(rubrics, "codex-cli old")
    monkeypatch.setattr(sg, "_codex_version", lambda: "codex-cli new")

    sg._validate_frozen_instrument(instrument, rubrics, require_live_cli=False)

    with pytest.raises(sg.SubagentGradeHaltError, match="differs from frozen launch client"):
        sg._validate_frozen_instrument(instrument, rubrics, require_live_cli=True)
    instrument["model"] = "changed-model"
    with pytest.raises(sg.SubagentGradeHaltError, match="semantics differ"):
        sg._validate_frozen_instrument(instrument, rubrics, require_live_cli=False)


def test_runner_manifest_pins_current_script_and_launcher_bytes():
    manifest = sg._runner_manifest()
    sg._validate_runner_manifest(manifest)
    manifest["script_sha256"] = "0" * 64
    with pytest.raises(sg.SubagentGradeHaltError, match="runner bytes differ"):
        sg._validate_runner_manifest(manifest)


def test_stage_retry_reuses_complete_frozen_provenance(monkeypatch, tmp_path, capsys):
    for name in ("inputs_manifest.json", "instrument_manifest.json", "runner_manifest.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    loaded = []
    monkeypatch.setattr(sg, "sensitivity_root", lambda unused: tmp_path)
    monkeypatch.setattr(
        sg,
        "_load_manifests",
        lambda args: loaded.append(args) or ({}, {}),
    )
    args = SimpleNamespace(out_root=tmp_path)

    sg.phase_stage(args)

    assert loaded == [args]
    assert "reused frozen provenance" in capsys.readouterr().out


def test_exact_item_registry_and_applicable_metric_counts():
    items = sg.build_item_registry(_records())
    assert len(items) == 4_000
    assert len({item.opaque_id for item in items}) == 4_000
    assert len({item.source_item_id for item in items}) == 4_000
    assert len(sg._applicable(items, "trait_evil")) == 2_000
    assert len(sg._applicable(items, "trait_sycophancy")) == 2_000
    assert len(sg._applicable(items, "coherence")) == 4_000


def test_pilot_repeats_same_33_items_five_times_and_55_grades_per_arm(monkeypatch):
    items = sg.build_item_registry(_records())
    rubrics = sg.load_frozen_rubrics()
    monkeypatch.setattr(sg, "_count_o200k_tokens", lambda text: len(text) // 10)
    jobs = sg.build_pilot_jobs(items, rubrics, _instrument())
    for rubric_id in sg.RUBRIC_IDS:
        rubric_jobs = [job for job in jobs if job.rubric_id == rubric_id]
        registry = None
        arm_counts = {"steer": 0, "proj": 0, "ablate": 0}
        for pass_index in range(5):
            pass_items = [
                item for job in rubric_jobs if job.pass_index == pass_index for item in job.items
            ]
            assert len(pass_items) == 33
            pass_registry = {item.opaque_id for item in pass_items}
            registry = pass_registry if registry is None else registry
            assert pass_registry == registry
            assert {arm: sum(item.arm == arm for item in pass_items) for arm in arm_counts} == {
                "steer": 11,
                "proj": 11,
                "ablate": 11,
            }
            for item in pass_items:
                arm_counts[item.arm] += 1
        assert len(registry) == 33
        assert arm_counts == {"steer": 55, "proj": 55, "ablate": 55}


def test_packet_blinds_condition_ids_and_contains_one_rubric():
    rubric = sg.load_frozen_rubrics()["trait_evil"]
    prompt = sg._prompt("trait_evil", rubric, [_one_item()])
    assert "secret_cell" not in prompt
    assert "i0123456789abcdefabcd" in prompt
    assert prompt.count("<RUBRIC>") == 1
    assert prompt.count("</RUBRIC>") == 1
    assert sg._sha256_text(rubric) in prompt
    assert "experimental conditions" in prompt


def test_coherence_packet_contains_answer_but_not_question_or_condition():
    rubric = sg.load_frozen_rubrics()["coherence"]
    prompt = sg._prompt("coherence", rubric, [_one_item()])
    assert "Neutral fixture answer." in prompt
    assert "Neutral fixture question?" not in prompt
    assert "secret_cell" not in prompt


def test_chunking_enforces_80_items_and_40k_o200k_tokens(monkeypatch):
    base = _one_item()
    items = [
        sg.GradeItem(
            **{
                **base.__dict__,
                "source_item_id": f"source-{index}",
                "opaque_id": f"i{index:020d}",
            }
        )
        for index in range(161)
    ]
    tokenization_calls = 0

    def count_tokens(text):
        nonlocal tokenization_calls
        tokenization_calls += 1
        return len(text)

    monkeypatch.setattr(sg, "_count_o200k_tokens", count_tokens)
    jobs = sg._chunks_for_pass(
        scope="production",
        rubric_id="trait_evil",
        rubric="score {question} {answer}",
        pass_index=0,
        items=items,
        instrument_fp="fp",
    )
    assert sum(len(job.items) for job in jobs) == 161
    assert all(len(job.items) <= 80 for job in jobs)
    assert all(job.prompt_tokens_o200k <= 40_000 for job in jobs)
    assert tokenization_calls == 3


def test_policy_blocked_packet_split_is_exact_and_preserves_registry(monkeypatch):
    base = _one_item()
    items = tuple(
        replace(
            base,
            source_item_id=f"source-{index}",
            opaque_id=f"i{index:020d}",
        )
        for index in range(4)
    )
    rubric = "score {answer}"
    prompt = sg._prompt("coherence", rubric, list(items))
    parent = replace(
        _job("coherence"),
        scope="production",
        pass_index=1,
        chunk_index=12,
        items=items,
        prompt=prompt,
        prompt_tokens_o200k=1,
    )
    monkeypatch.setattr(
        sg,
        "POLICY_PACKET_SPLITS",
        {parent.job_id: {"prompt_sha256": sg._sha256_text(prompt), "parts": 2}},
    )
    monkeypatch.setattr(sg, "_count_o200k_tokens", lambda text: len(text))

    children = sg._apply_policy_packet_splits([parent], {"coherence": rubric})

    assert [child.job_suffix for child in children] == [
        "__policy_split00",
        "__policy_split01",
    ]
    assert [item.opaque_id for child in children for item in child.items] == [
        item.opaque_id for item in items
    ]
    assert all(len(child.items) == 2 for child in children)
    assert len({child.job_id for child in children}) == 2
    assert len({sg._job_record_path(Path("/tmp/out"), child) for child in children}) == 2


def test_codex_command_is_fresh_ephemeral_read_only_and_pinned(tmp_path):
    command = sg._codex_command(tmp_path / "schema.json", tmp_path / "out.json", tmp_path)
    joined = " ".join(command)
    assert command[:4] == ["codex", "-a", "never", "exec"]
    assert "--ephemeral" in command
    assert "--ignore-rules" in command
    assert "--ignore-user-config" in command
    assert "--sandbox read-only" in joined
    assert "--model gpt-5.6-sol" in joined
    assert "--json" in command
    assert 'model_reasoning_effort="low"' in command
    assert "resume" not in command


def test_subagent_env_removes_anthropic_and_unrelated_service_keys():
    env = sg._subagent_env(
        {
            "PATH": "/bin",
            "CODEX_HOME": "/codex",
            "ANTHROPIC_API_KEY": "a",
            "ANTHROPIC_BATCH_KEY": "b",
            "OPENAI_API_KEY": "o",
            "HF_TOKEN": "h",
            "WANDB_API_KEY": "w",
        }
    )
    assert env == {"PATH": "/bin", "CODEX_HOME": "/codex"}


@pytest.mark.parametrize("bad", ["42", 42.0, True, -1, 101])
def test_response_validation_never_coerces_invalid_scores(bad):
    job = _job()
    with pytest.raises(sg.SubagentGradeHaltError):
        sg._validate_response(job, _response(job, bad))


def test_trait_refusal_is_preserved_as_content_drop_but_coherence_refusal_is_invalid():
    trait_job = _job("trait_evil")
    assert sg._validate_response(trait_job, _response(trait_job, "REFUSAL")) == {
        trait_job.items[0].opaque_id: None
    }
    coherence_job = _job("coherence")
    with pytest.raises(sg.SubagentGradeHaltError):
        sg._validate_response(coherence_job, _response(coherence_job, "REFUSAL"))


def test_response_validation_requires_exact_ids_order_and_count():
    job = _job()
    assert sg._validate_response(job, _response(job)) == {job.items[0].opaque_id: 42}
    wrong = _response(job)
    wrong["scores"][0]["item_id"] = "iwrong"
    with pytest.raises(sg.SubagentGradeHaltError):
        sg._validate_response(job, wrong)


def test_run_one_job_executes_real_body_and_writes_immutable_record(tmp_path, monkeypatch):
    job = _job()
    sroot = tmp_path / "sensitivity"
    (sroot / "tmp").mkdir(parents=True)
    fake_run = create_autospec(subprocess.run)

    def run_side_effect(command, **kwargs):
        output = command[command.index("--output-last-message") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(_response(job), handle)
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "fixture-thread"}),
                json.dumps({"type": "turn.completed"}),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    fake_run.side_effect = run_side_effect
    monkeypatch.setattr(sg.subprocess, "run", fake_run)
    args = SimpleNamespace(
        job_timeout_seconds=60,
        isolated_cwd=str(tmp_path / "isolated"),
    )
    record = sg._run_one_job(args, sroot, job)
    assert record["status"] == "complete"
    assert record["grader_session_id"] == "fixture-thread"
    canonical = sg._job_record_path(sroot, job)
    before = canonical.read_bytes()
    cached = sg._run_one_job(args, sroot, job)
    assert cached == record
    assert canonical.read_bytes() == before
    assert fake_run.call_count == 1


def test_run_one_job_retries_transport_once_but_never_retries_invalid_content(
    tmp_path, monkeypatch
):
    job = _job()
    sroot = tmp_path / "transport"
    (sroot / "tmp").mkdir(parents=True)
    attempts = 0

    def transport_then_success(command, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="transient")
        output = command[command.index("--output-last-message") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(_response(job), handle)
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "retry-thread"}),
                json.dumps({"type": "turn.completed"}),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(sg.subprocess, "run", transport_then_success)
    monkeypatch.setattr(sg.time, "sleep", lambda _: None)
    args = SimpleNamespace(
        job_timeout_seconds=60,
        isolated_cwd=str(tmp_path / "isolated"),
    )
    record = sg._run_one_job(args, sroot, job)
    assert record["attempt_index"] == 2
    assert attempts == 2

    bad_root = tmp_path / "content"
    (bad_root / "tmp").mkdir(parents=True)
    attempts = 0

    def invalid_content(command, **kwargs):
        nonlocal attempts
        attempts += 1
        output = command[command.index("--output-last-message") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            handle.write("not-json")
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "bad-thread"}),
                json.dumps({"type": "turn.completed"}),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(sg.subprocess, "run", invalid_content)
    with pytest.raises(sg.SubagentGradeHaltError, match="invalid model response"):
        sg._run_one_job(args, bad_root, job)
    assert attempts == 1


def test_codex_event_validation_rejects_tool_use():
    job = _job()
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "tool-thread"}),
            json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}),
            json.dumps({"type": "turn.completed"}),
        ]
    )
    with pytest.raises(sg.SubagentGradeHaltError, match="forbidden item type"):
        sg._validate_codex_events(job, stdout)


def test_collect_production_requires_exact_five_passes(tmp_path):
    item = _one_item()
    jobs = []
    for pass_index in range(5):
        job = sg.JobSpec(
            scope="production",
            rubric_id="trait_evil",
            pass_index=pass_index,
            chunk_index=0,
            items=(item,),
            prompt=f"prompt-{pass_index}",
            prompt_tokens_o200k=1,
            instrument_fp="fp",
        )
        jobs.append(job)
        record = {
            "status": "complete",
            "job_id": job.job_id,
            "instrument_fp": "fp",
            "prompt_sha256": sg._sha256_text(job.prompt),
            "source_item_ids": [item.source_item_id],
            "response": _response(job, 40 + pass_index),
        }
        sg._immutable_json(sg._job_record_path(tmp_path, job), record)
    # The public collector validates all three complete populations. Exercise
    # its exact-five logic through the rubric-local invariant instead.
    by_pass = {}
    for job in jobs:
        rec = sg._validate_completed_job(sg._job_record_path(tmp_path, job), job)
        by_pass[job.pass_index] = sg._validate_response(job, rec["response"])[item.opaque_id]
    assert [by_pass[index] for index in range(5)] == [40, 41, 42, 43, 44]


def test_partial_payload_reports_draw_and_item_completeness_separately(tmp_path):
    base = _one_item()
    items = [
        replace(
            base,
            source_item_id=f"source-{index}",
            opaque_id=f"i{index:020d}",
        )
        for index in range(200)
    ]
    record_path = tmp_path / "record.json"
    record_path.write_text("{}", encoding="utf-8")
    score_map = {item.opaque_id: [10, 20, 30, 40, 50] for item in items}
    score_map[items[0].opaque_id] = []
    partial = sg._partial_payload(
        record={"cell_id": "secret_cell", "cell": {"kind": "steer"}},
        record_path=record_path,
        phase="steer",
        rubric_id="trait_evil",
        items=items,
        score_map=score_map,
        instrument_fp="fp",
    )
    accounting = partial["accounting"]
    assert accounting["n_content_dropped_draws"] == 5
    assert accounting["frac_draws_scored"] == pytest.approx(995 / 1000)
    assert accounting["frac_items_complete"] == pytest.approx(199 / 200)
    assert accounting["n_items_zero_valid"] == 1


def test_paired_delta_is_question_paired_and_has_no_parent_verdict_fields():
    read = sg._paired_delta(np.arange(20.0) + 3.0, np.arange(20.0), key="fixture")
    assert read["delta_score"] == pytest.approx(3.0)
    assert read["ci95"] == pytest.approx([3.0, 3.0])
    serialized = json.dumps(read).lower()
    assert "null" not in serialized
    assert "h1" not in serialized
    assert "h2" not in serialized
    assert "band" not in serialized


def test_output_schema_keeps_refusal_as_drop_signal_not_numeric_coercion():
    schema = sg._output_schema(_job())
    score = schema["properties"]["scores"]["items"]["properties"]["score"]
    assert {"type": "string", "enum": ["REFUSAL"]} in score["anyOf"]
    assert schema["properties"]["scores"]["minItems"] == 1
    assert schema["properties"]["scores"]["maxItems"] == 1


def test_coherence_output_schema_is_numeric_only():
    schema = sg._output_schema(_job("coherence"))
    score = schema["properties"]["scores"]["items"]["properties"]["score"]
    assert score == {"type": "integer", "minimum": 0, "maximum": 100}
