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
GPU_SPEC = importlib.util.spec_from_file_location(
    "issue952_gpu_contract",
    ROOT / "scripts" / "issue952_china_definitive_gpu.py",
)
assert GPU_SPEC is not None and GPU_SPEC.loader is not None
GPU = importlib.util.module_from_spec(GPU_SPEC)
GPU_SPEC.loader.exec_module(GPU)


def _small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ANALYSIS, "REGISTERED_SOURCE_ITEMS", 2)
    monkeypatch.setattr(ANALYSIS, "REGISTERED_PROMPTS", 2)
    monkeypatch.setattr(ANALYSIS, "PROMPTS_PER_SOURCE", 1)
    monkeypatch.setattr(ANALYSIS, "N_DRAWS", 2)


def test_analysis_registered_recipe_matches_gpu_producer_constants() -> None:
    assert ANALYSIS.MODEL == GPU.MODEL
    assert ANALYSIS.MODEL_REV == GPU.MODEL_REV
    assert ANALYSIS.LAYERS == GPU.LAYERS
    assert ANALYSIS.N_DRAWS == GPU.N_DRAWS
    assert ANALYSIS.TEMPERATURE == GPU.TEMPERATURE
    assert ANALYSIS.TOP_P == GPU.TOP_P
    assert ANALYSIS.MAX_NEW_TOKENS == GPU.MAX_NEW_TOKENS
    assert ANALYSIS.GENERATION_SEED_BASE == GPU.SEED_BASE


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


def test_judge_attempt_blocks_failed_technical_coverage_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, judge_dir, accepted, generation = _judge_fixture(tmp_path, monkeypatch)
    summary_path = judge_dir / "wave_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["passed"] = False
    summary["claim_eligible"] = False
    _write_json(summary_path, summary)
    marker = _refresh_judge_marker(judge_dir)
    with pytest.raises(RuntimeError, match="technical/coverage"):
        ANALYSIS._validate_judge_attempt(
            run_dir=run_dir,
            judge_dir=judge_dir,
            attempt=1,
            marker=marker,
            accepted=accepted,
            generation=generation,
        )


def test_judge_attempt_allows_reliability_only_miss_for_non_h4_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, judge_dir, accepted, generation = _judge_fixture(tmp_path, monkeypatch)
    summary_path = judge_dir / "wave_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["interjudge_reliability_passed"] = False
    summary["claim_eligible"] = False
    _write_json(summary_path, summary)
    marker = _refresh_judge_marker(judge_dir)
    result = ANALYSIS._validate_judge_attempt(
        run_dir=run_dir,
        judge_dir=judge_dir,
        attempt=1,
        marker=marker,
        accepted=accepted,
        generation=generation,
    )
    assert result["summary"]["passed"] is True
    assert result["summary"]["claim_eligible"] is False


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


@pytest.mark.parametrize(
    "corruption",
    [None, "status", "count", "input_stage", "byte_census", "wrong_model", "wrong_regime"],
)
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
    accepted_bank = {
        **accepted,
        "n_accepted_source_items": 2,
    }
    package_versions = dict(ANALYSIS.REGISTERED_PACKAGE_VERSIONS)
    regime = {
        "issue": ANALYSIS.ISSUE,
        "model": ANALYSIS.MODEL,
        "model_revision": ANALYSIS.MODEL_REV,
        "bank_sha256": input_marker["prompt_bank_sha256"],
        "layers": list(ANALYSIS.LAYERS),
        "draws": ANALYSIS.N_DRAWS,
        "temperature": ANALYSIS.TEMPERATURE,
        "top_p": ANALYSIS.TOP_P,
        "max_new_tokens": ANALYSIS.MAX_NEW_TOKENS,
        "seed_base": ANALYSIS.GENERATION_SEED_BASE,
        "seed_namespace": ANALYSIS.GENERATION_SEED_NAMESPACE,
        "seed_formula": ANALYSIS.GENERATION_SEED_FORMULA,
        "seed_policy": ANALYSIS.GENERATION_SEED_POLICY,
        "smoke": False,
        "attempt": 1,
        "git_sha": "1" * 40,
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "n_accepted_source_items": 2,
        "n_accepted_prompts": 2,
        "n_expected_rollouts": 4,
        "n_selected_prompts": 2,
        "selected_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "prompt_token_max": 128,
        "max_model_len": 4096,
        "chat_template_sha256": "2" * 64,
        "package_versions": package_versions,
        "tokenizer_artifact_sha256": {
            name: str(index) * 64
            for index, name in enumerate(sorted(ANALYSIS.TOKENIZER_ARTIFACTS), start=3)
        },
    }
    generation = {
        "regime": regime,
        "regime_fp": ANALYSIS._sha_obj(regime),
        "rollouts_sha256": "c" * 64,
        "n_prompts": 2,
        "n_rows": 4,
        "ordered_item_ids_sha256": "d" * 64,
        "package_versions": package_versions,
        "accepted_bank": accepted_bank,
    }
    generation_fp = ANALYSIS._generation_fingerprint(generation)
    capture_regime = {
        "model_revision": ANALYSIS.MODEL_REV,
        "layers": list(ANALYSIS.LAYERS),
        "bank_sha256": input_marker["prompt_bank_sha256"],
        "rollouts_sha256": generation["rollouts_sha256"],
        "context_position": ANALYSIS.CONTEXT_POSITION,
        "answer_pooling": ANALYSIS.ANSWER_POOLING,
        "serialized_dtype": ANALYSIS.SERIALIZED_DTYPE,
        "git_sha": regime["git_sha"],
        "generation_fingerprint": generation_fp,
        "accepted_source_ids_sha256": accepted["accepted_source_ids_sha256"],
        "n_accepted_source_items": 2,
        "n_accepted_prompts": 2,
        "n_selected_prompts": 2,
        "selected_source_ids_sha256": accepted["accepted_source_ids_sha256"],
    }
    capture = {
        "issue": ANALYSIS.ISSUE,
        "layers": list(ANALYSIS.LAYERS),
        "n_contexts": 2,
        "n_answer_rows": 4,
        "generation_fingerprint": generation_fp,
        "capture_regime": capture_regime,
        "capture_regime_fp": ANALYSIS._sha_obj(capture_regime),
        "model_revision": ANALYSIS.MODEL_REV,
        "rollouts_sha256": generation["rollouts_sha256"],
        "package_versions": package_versions,
        "accepted_bank": accepted_bank,
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
    elif corruption == "byte_census":
        capture_upload["byte_verified_files"] = list(byte_census)[:-1]
    else:
        if corruption == "wrong_model":
            generation["regime"]["model"] = "wrong/model"
            generation["regime"]["model_revision"] = "0" * 40
            capture["model_revision"] = "0" * 40
            capture["capture_regime"]["model_revision"] = "0" * 40
        else:
            generation["regime"]["seed_namespace"] = "coordinated-but-unregistered"
        generation["regime_fp"] = ANALYSIS._sha_obj(generation["regime"])
        generation_fp = ANALYSIS._generation_fingerprint(generation)
        capture["generation_fingerprint"] = generation_fp
        capture["capture_regime"]["generation_fingerprint"] = generation_fp
        capture["capture_regime_fp"] = ANALYSIS._sha_obj(capture["capture_regime"])
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


def test_h4_is_indeterminate_when_interjudge_reliability_misses() -> None:
    label = ANALYSIS._classify_h4(
        judge_claim_eligible=False,
        judge_reliability_passed=False,
        core_eligible=False,
        threshold_stable=True,
        lexical_consistent=True,
        supported=False,
    )
    assert label == "judge-reliability-indeterminate"


def test_figure_mean_preserves_undefined_contrasts_as_gaps() -> None:
    assert ANALYSIS._mean_defined([0.2, None, float("nan"), 0.4]) == pytest.approx(0.3)
    assert ANALYSIS.math.isnan(ANALYSIS._mean_defined([None, float("nan")]))


def test_reports_separate_registered_accepted_and_realized_counts() -> None:
    passing_sources = [f"s{source:02d}" for source in range(85)]
    bank = [
        {
            "item_id": f"s{source:02d}-p{prompt:02d}",
            "source_prompt_id": f"s{source:02d}",
            "audit_pass": source < 85,
        }
        for source in range(90)
        for prompt in range(12)
    ]
    accepted = ANALYSIS._accepted_bank_contract(
        bank,
        {
            "passed": True,
            "passing_item_ids": passing_sources,
            "n_audit_passing_items": 85,
        },
    )
    assert len(accepted["accepted_source_ids"]) == 85
    assert accepted["n_accepted_prompts"] == 1020
    assert accepted["n_expected_rollouts"] == 8160
    data = {
        "maximum_registered": {"items": 90, "prompts": 1080, "draws": 8640},
        "accepted_planned": {"items": 85, "prompts": 1020, "draws": 8160},
        "realized": {
            "items_primary": 85,
            "prompts_primary": 1020,
            "draws_primary": 8160,
            "draws_generated": 8160,
        },
    }
    coverage = ANALYSIS._coverage_report(data)
    assert coverage == {
        "maximum_registered": {"items": 90, "prompts": 1080, "draws": 8640},
        "accepted_planned": {"items": 85, "prompts": 1020, "draws": 8160},
        "realized": {
            "items_primary": 85,
            "prompts_primary": 1020,
            "draws_primary": 8160,
            "draws_generated": 8160,
        },
    }
    assert ANALYSIS._planned_vs_realized_report(data) == {
        "maximum_registered": {"items": 90, "prompts": 1080, "draws": 8640},
        "accepted_planned": {"items": 85, "prompts": 1020, "draws": 8160},
        "realized": data["realized"],
    }
