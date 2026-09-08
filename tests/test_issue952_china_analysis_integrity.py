from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "issue952_analysis_integrity",
    ROOT / "scripts" / "issue952_china_definitive_analysis.py",
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def _small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ANALYSIS, "REGISTERED_SOURCE_ITEMS", 2)
    monkeypatch.setattr(ANALYSIS, "REGISTERED_PROMPTS", 2)
    monkeypatch.setattr(ANALYSIS, "PROMPTS_PER_SOURCE", 1)
    monkeypatch.setattr(ANALYSIS, "N_DRAWS", 2)


def _write_json(path: Path, value: object) -> None:
    ANALYSIS._write_json(path, value)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    ANALYSIS._write_jsonl(path, rows)


def _refresh_judge_marker(judge_dir: Path) -> dict:
    marker_path = judge_dir / "upload.json"
    marker = json.loads(marker_path.read_text())
    census = marker["artifact_census"]
    for relative in list(census):
        census[relative] = ANALYSIS._sha256(judge_dir / Path(relative).relative_to("judge"))
    direct = {
        "wave_scores_sha256": "judge/wave_scores.jsonl",
        "wave_summary_sha256": "judge/wave_summary.json",
        "wave_request_manifest_sha256": "judge/wave_request_manifest.json",
        "wave_packet_manifest_sha256": "judge/wave_packet_manifest.json",
        "wave_lookup_sha256": "judge/wave_lookup.json",
        "wave_runtime_identity_sha256": "judge/wave_runtime_identity.json",
        "wave_overlap_joined_sha256": "judge/wave_overlap_joined.jsonl",
        "production_stage_sha256": "judge/production_stage.json",
    }
    marker.update({key: census[relative] for key, relative in direct.items()})
    marker["artifact_census_sha256"] = ANALYSIS._sha_obj(census)
    _write_json(marker_path, marker)
    return marker


def _judge_rows() -> tuple[list[dict], list[dict], list[dict]]:
    scores = []
    lookup = []
    for source_index in range(2):
        for draw in range(2):
            source = f"s{source_index}"
            prompt = f"p{source_index}"
            item = f"{prompt}-d{draw}"
            primary = "agent_a" if (source_index + draw) % 2 == 0 else "agent_b"
            assigned = ["agent_a", "agent_b"] if item == "p0-d0" else [primary]
            common = {
                "item_id": item,
                "prompt_id": prompt,
                "source_prompt_id": source,
                "topic": f"topic{source_index}",
                "language": "en",
                "content": "sensitive_full",
                "frame": "direct",
                "draw": draw,
            }
            lookup.append({**common, "primary_agent": primary, "assigned_agents": assigned})
            scores.append({**common, "verdict": draw == 1, "judge_id": primary})
    overlap = [
        {
            **{
                key: lookup[0][key]
                for key in (
                    "item_id",
                    "prompt_id",
                    "source_prompt_id",
                    "language",
                    "content",
                    "frame",
                    "topic",
                )
            },
            "primary_judge": "agent_a",
            "primary_label": False,
            "secondary_judge": "agent_b",
            "secondary_label": True,
            "disagreement": True,
            "secondary_judge_contrast": 1,
        }
    ]
    return scores, lookup, overlap


def _judge_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict, dict]:
    _small_contract(monkeypatch)
    run_dir = tmp_path / "run"
    judge_dir = run_dir / "judge" / "attempt1"
    generation = {"rollouts_sha256": "a" * 64}
    accepted = {
        "accepted_source_ids": ["s0", "s1"],
        "accepted_source_ids_sha256": ANALYSIS._sha_obj(["s0", "s1"]),
        "n_accepted_prompts": 2,
        "n_expected_rollouts": 4,
    }
    scores, lookup, overlap = _judge_rows()
    runtime_path = judge_dir / "wave_runtime_identity.json"
    _write_json(runtime_path, {"kind": "synthetic-runtime"})
    _write_json(
        judge_dir / "wave_packet_manifest.json",
        {
            "packet_kind": "production-wave-attempt1",
            "runtime_identity_sha256": ANALYSIS._sha256(runtime_path),
        },
    )
    _write_json(judge_dir / "wave_lookup.json", lookup)
    _write_jsonl(judge_dir / "wave_scores.jsonl", scores)
    _write_jsonl(judge_dir / "wave_overlap_joined.jsonl", overlap)

    artifact_root = judge_dir / "agent_artifacts" / "wave_attempt1" / "agent_a"
    artifact_paths = {
        "packet_sha256": artifact_root / "batch_000.packet.json",
        "output_sha256": artifact_root / "batch_000.output.jsonl",
        "output_manifest_sha256": artifact_root / "batch_000.output_manifest.json",
    }
    for index, path in enumerate(artifact_paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic-{index}\n")
    artifact_hashes = {
        "agent_a/batch_000": {key: ANALYSIS._sha256(path) for key, path in artifact_paths.items()}
    }

    generation_path = run_dir / "manifests" / "generation.json"
    _write_json(generation_path, generation)
    stage = {
        "attempt": 1,
        "rollouts_sha256": generation["rollouts_sha256"],
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "n_accepted_source_items": 2,
        "n_accepted_prompts": 2,
        "n_expected_rollouts": 4,
        "files": {"manifests/generation.json": ANALYSIS._sha256(generation_path)},
    }
    _write_json(judge_dir / "production_stage.json", stage)
    request = {
        "attempt": 1,
        "rollouts_sha256": generation["rollouts_sha256"],
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "n_accepted_source_items": 2,
        "n_accepted_prompts": 2,
        "n_expected_rollouts": 4,
        "n_requests": 4,
        "n_overlap": 1,
        "overlap_fraction": 0.10,
        "lookup_sha256": ANALYSIS._sha256(judge_dir / "wave_lookup.json"),
        "packet_manifest_sha256": ANALYSIS._sha256(judge_dir / "wave_packet_manifest.json"),
        "runtime_identity": {
            "path": "judge/wave_runtime_identity.json",
            "sha256": ANALYSIS._sha256(runtime_path),
        },
    }
    _write_json(judge_dir / "wave_request_manifest.json", request)
    summary = {
        "attempt": 1,
        "rollouts_sha256": generation["rollouts_sha256"],
        "scores_sha256": ANALYSIS._sha256(judge_dir / "wave_scores.jsonl"),
        "request_manifest_sha256": ANALYSIS._sha256(judge_dir / "wave_request_manifest.json"),
        "runtime_identity_sha256": ANALYSIS._sha256(runtime_path),
        "overlap_joined_sha256": ANALYSIS._sha256(judge_dir / "wave_overlap_joined.jsonl"),
        "n_overlap_joined": 1,
        "n": 4,
        "n_valid": 4,
        "accepted_source_items": 2,
        "accepted_prompts": 2,
        "accepted_draws": 4,
        "planned_source_items": 2,
        "realized_complete_source_items": 2,
        "realized_valid_rows": 4,
        "passed": True,
        "claim_eligible": True,
        "interjudge_reliability_passed": True,
        "interjudge_reliability": {
            "overall": {"n": 1},
            "by_language": {"en": {"n": 1}},
        },
        "agent_artifact_hashes": artifact_hashes,
        "primary_assignment_counts": {
            "by_judge": {"agent_a": 2, "agent_b": 2},
            "by_language": {"en": 4},
            "by_arm": {"en:sensitive_full:direct": 4},
            "by_topic": {"topic0": 2, "topic1": 2},
        },
        "prompt_validity_denominators_by_language_content": {
            "en:sensitive_full": {
                "n_prompts_total": 2,
                "n_prompts_any_valid": 2,
                "n_prompts_all_eight_valid": 2,
            }
        },
        "prompt_completeness_by_language_content": {"en:sensitive_full": 1.0},
        "complete_source_item_ids": ["s0", "s1"],
        "analysis_source_item_ids": ["s0", "s1"],
        "analysis_subset_passed": False,
    }
    _write_json(judge_dir / "wave_summary.json", summary)
    census = {
        f"judge/{path.relative_to(judge_dir)}": ANALYSIS._sha256(path)
        for path in judge_dir.rglob("*")
        if path.is_file()
    }
    marker = {
        "schema_version": 1,
        "kind": "issue952_codex_production_upload",
        "attempt": 1,
        "rollouts_sha256": generation["rollouts_sha256"],
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "artifact_census": census,
        "artifact_census_sha256": ANALYSIS._sha_obj(census),
    }
    _write_json(judge_dir / "upload.json", marker)
    _refresh_judge_marker(judge_dir)
    return run_dir, judge_dir, accepted, generation


def test_judge_attempt_validates_and_reports_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, judge_dir, accepted, generation = _judge_fixture(tmp_path, monkeypatch)
    marker = json.loads((judge_dir / "upload.json").read_text())
    result = ANALYSIS._validate_judge_attempt(
        run_dir=run_dir,
        judge_dir=judge_dir,
        attempt=1,
        marker=marker,
        accepted=accepted,
        generation=generation,
    )
    assert result["overlap"]["joined_count"] == 1
    assert result["overlap"]["disagreements"] == 1


@pytest.mark.parametrize("corruption", ["census", "claim", "stage", "overlap", "completeness"])
def test_judge_attempt_rejects_corrupt_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    run_dir, judge_dir, accepted, generation = _judge_fixture(tmp_path, monkeypatch)
    if corruption == "census":
        path = judge_dir / "agent_artifacts/wave_attempt1/agent_a/batch_000.output.jsonl"
        path.write_text("changed\n")
    elif corruption == "claim":
        summary = json.loads((judge_dir / "wave_summary.json").read_text())
        summary["claim_eligible"] = False
        _write_json(judge_dir / "wave_summary.json", summary)
        _refresh_judge_marker(judge_dir)
    elif corruption == "stage":
        (run_dir / "manifests/generation.json").write_text("{}\n")
    elif corruption == "overlap":
        rows = ANALYSIS._jsonl(judge_dir / "wave_overlap_joined.jsonl")
        rows[0]["primary_judge"] = "agent_b"
        _write_jsonl(judge_dir / "wave_overlap_joined.jsonl", rows)
        summary = json.loads((judge_dir / "wave_summary.json").read_text())
        summary["overlap_joined_sha256"] = ANALYSIS._sha256(judge_dir / "wave_overlap_joined.jsonl")
        _write_json(judge_dir / "wave_summary.json", summary)
        _refresh_judge_marker(judge_dir)
    else:
        summary = json.loads((judge_dir / "wave_summary.json").read_text())
        summary["prompt_validity_denominators_by_language_content"]["en:sensitive_full"][
            "n_prompts_total"
        ] = 1
        _write_json(judge_dir / "wave_summary.json", summary)
        _refresh_judge_marker(judge_dir)
    marker = json.loads((judge_dir / "upload.json").read_text())
    with pytest.raises(RuntimeError):
        ANALYSIS._validate_judge_attempt(
            run_dir=run_dir,
            judge_dir=judge_dir,
            attempt=1,
            marker=marker,
            accepted=accepted,
            generation=generation,
        )


@pytest.mark.parametrize("corruption", [None, "status", "count", "input_stage", "byte_census"])
def test_gpu_receipt_validates_exact_lineage(tmp_path: Path, corruption: str | None) -> None:
    run_dir = tmp_path
    bank_path = run_dir / "inputs/prompt_bank.jsonl"
    audit_path = run_dir / "inputs/bank_audit_report.json"
    marker_path = run_dir / "inputs/upload_verified.json"
    _write_jsonl(bank_path, [{"synthetic": True}])
    _write_json(audit_path, {"synthetic": True})
    input_marker = {
        "data_revision": "b" * 40,
        "prompt_bank_sha256": ANALYSIS._sha256(bank_path),
        "bank_audit_report_sha256": ANALYSIS._sha256(audit_path),
    }
    _write_json(marker_path, input_marker)
    accepted = {
        "accepted_source_ids": ["s0", "s1"],
        "accepted_source_ids_sha256": ANALYSIS._sha_obj(["s0", "s1"]),
        "n_accepted_prompts": 2,
        "n_expected_rollouts": 4,
    }
    generation = {
        "regime": {
            "attempt": 1,
            "smoke": False,
            "model_revision": "model-rev",
            "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        },
        "regime_fp": "regime",
        "rollouts_sha256": "c" * 64,
        "n_prompts": 2,
        "n_rows": 4,
        "ordered_item_ids_sha256": "d" * 64,
    }
    generation_fp = ANALYSIS._generation_fingerprint(generation)
    capture_regime = {
        "generation_fingerprint": generation_fp,
        "model_revision": "model-rev",
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
    }
    capture = {
        "n_contexts": 2,
        "n_answer_rows": 4,
        "generation_fingerprint": generation_fp,
        "capture_regime": capture_regime,
        "capture_regime_fp": ANALYSIS._sha_obj(capture_regime),
        "model_revision": "model-rev",
        "vc_sha256": "e" * 64,
        "va_files": {"va_00000_00004.pt": "f" * 64},
    }
    byte_census = {
        f"{ANALYSIS.HF_PREFIX}/attempt1/analysis_tensors/vc.pt": "e" * 64,
        f"{ANALYSIS.HF_PREFIX}/attempt1/analysis_tensors/va_00000_00004.pt": "f" * 64,
    }
    raw_upload = {"synthetic": True}
    capture_upload = {
        "byte_verified_sha256": byte_census,
        "byte_verified_files": sorted(byte_census),
    }
    input_stage = {
        "marker_sha256": ANALYSIS._sha256(marker_path),
        "data_revision": input_marker["data_revision"],
        "prompt_bank_sha256": ANALYSIS._sha256(bank_path),
        "bank_audit_report_sha256": ANALYSIS._sha256(audit_path),
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "n_accepted_source_items": 2,
        "n_accepted_prompts": 2,
        "n_expected_rollouts": 4,
    }
    done = {
        "schema_version": 1,
        "kind": "issue952_china_definitive_gpu",
        "issue": 952,
        "status": "done",
        "version": 1,
        "generation": generation,
        "capture": capture,
        "raw_upload": raw_upload,
        "capture_upload": capture_upload,
        "hf_prefix": ANALYSIS.HF_PREFIX,
    }
    if corruption is None:
        ANALYSIS._validate_gpu_attempt(
            run_dir=run_dir,
            attempt=1,
            accepted=accepted,
            input_marker=input_marker,
            input_stage=input_stage,
            generation=generation,
            capture=capture,
            raw_upload=raw_upload,
            capture_upload=capture_upload,
            done=done,
        )
        return
    if corruption == "status":
        done["status"] = "running"
    elif corruption == "count":
        capture["n_answer_rows"] = 3
    elif corruption == "input_stage":
        input_stage["prompt_bank_sha256"] = "0" * 64
    else:
        capture_upload["byte_verified_files"] = list(byte_census)[:-1]
    with pytest.raises(RuntimeError):
        ANALYSIS._validate_gpu_attempt(
            run_dir=run_dir,
            attempt=1,
            accepted=accepted,
            input_marker=input_marker,
            input_stage=input_stage,
            generation=generation,
            capture=capture,
            raw_upload=raw_upload,
            capture_upload=capture_upload,
            done=done,
        )


def _answer_shard() -> tuple[dict, dict, dict, dict]:
    regime = {"kind": "synthetic"}
    capture = {
        "model_revision": "model-rev",
        "capture_regime": regime,
        "capture_regime_fp": ANALYSIS._sha_obj(regime),
    }
    generation = {"rollouts_sha256": "a" * 64}
    store = {
        "layers": list(ANALYSIS.LAYERS),
        "index": [{"item_id": "p0-d0", "prompt_id": "p0", "draw": 0}],
        "va_tail_incl": torch.zeros((1, 3, 2), dtype=torch.float32),
        "empty_rows": [],
        "dtype": "fp32",
        "pooling": "completion_plus_im_end_newline_mean",
        "model_revision": "model-rev",
        "rollouts_sha256": "a" * 64,
        "capture_regime": regime,
        "capture_regime_fp": ANALYSIS._sha_obj(regime),
    }
    rollouts = {"p0-d0": {"item_id": "p0-d0", "prompt_id": "p0", "draw": 0}}
    return store, rollouts, capture, generation


@pytest.mark.parametrize("corruption", ["index_length", "model", "rollout_hash", "tuple"])
def test_answer_shard_rejects_index_and_metadata_corruption(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    monkeypatch.setattr(ANALYSIS, "HIDDEN", 2)
    store, rollouts, capture, generation = _answer_shard()
    if corruption == "index_length":
        store["index"].append(copy.deepcopy(store["index"][0]))
    elif corruption == "model":
        store["model_revision"] = "other"
    elif corruption == "rollout_hash":
        store["rollouts_sha256"] = "b" * 64
    else:
        store["index"][0]["draw"] = 1
    with pytest.raises(RuntimeError):
        ANALYSIS._validated_answer_shard(
            store=store,
            rollout_by_item=rollouts,
            capture=capture,
            generation=generation,
            name="synthetic.pt",
        )


def test_answer_shard_accepts_exact_rollout_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ANALYSIS, "HIDDEN", 2)
    store, rollouts, capture, generation = _answer_shard()
    index, vectors = ANALYSIS._validated_answer_shard(
        store=store,
        rollout_by_item=rollouts,
        capture=capture,
        generation=generation,
        name="synthetic.pt",
    )
    assert len(index) == vectors.shape[0] == 1


def test_h4_is_ineligible_when_interjudge_reliability_misses() -> None:
    label = ANALYSIS._classify_h4(
        judge_claim_eligible=False,
        judge_reliability_passed=False,
        core_eligible=False,
        threshold_stable=True,
        lexical_consistent=True,
        supported=False,
    )
    assert label == "judge-reliability-ineligible"


def test_reports_separate_registered_accepted_and_realized_counts() -> None:
    coverage = ANALYSIS._coverage_report(
        {
            "maximum_registered": {"draws": 8640},
            "accepted_planned": {"draws": 8160},
            "realized": {"draws_generated": 8159},
        }
    )
    assert coverage == {
        "maximum_registered": {"draws": 8640},
        "accepted_planned": {"draws": 8160},
        "realized": {"draws_generated": 8159},
    }
