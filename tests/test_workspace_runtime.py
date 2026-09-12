"""Local runtime-manifest tests for the J/R workspace experiment driver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from explore_persona_space.analysis.workspace_runtime import (
    calibration_tokens,
    native_forward_validation,
    native_loader_status,
    preflight_status,
    selected_prompts,
    validate_token_manifest,
)


def _write_fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a minimal frozen source/audit/config trio with real byte hashes."""
    source = tmp_path / "train_25k.jsonl"
    rows = [
        {"prompt": "alpha prompt", "ladder_local_id": 11},
        {"prompt": "beta prompt", "ladder_local_id": 12},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    prompt_sha = hashlib.sha256(b"alpha prompt").hexdigest()
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "manifest_sources": {
                    "splits": {"train_25k.jsonl": {"path": str(source), "sha256": source_sha}}
                }
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "source_records": {"train": {"filename": "train_25k.jsonl", "sha256": source_sha}},
                "subsets": {
                    "calibration": [
                        {
                            "source_split": "train",
                            "source_row_index": 0,
                            "ladder_local_id": 11,
                            "prompt_sha256": prompt_sha,
                        }
                    ]
                },
            }
        )
    )
    config = tmp_path / "workspace_jr.yaml"
    config.write_text(
        """
schema_version: workspace-jr-v1
status: execution_authorized
selection:
  comparison: Qwen/Qwen3.5-4B
models:
  comparison:
    revision: test-revision
    n_layers: 32
    d_model: 2560
""".lstrip()
    )
    return config, selection, audit


def test_selected_prompts_verify_source_bytes_and_prompt_identity(tmp_path):
    """Frozen prompt resolution checks bytes, row IDs and NFC prompt hashes."""
    _config, selection, audit = _write_fixture_files(tmp_path)
    prompts = selected_prompts(selection, audit, "calibration")
    assert [row["prompt"] for row in prompts] == ["alpha prompt"]
    assert prompts[0]["ladder_local_id"] == 11
    bad = json.loads(selection.read_text())
    bad["subsets"]["calibration"][0]["prompt_sha256"] = "0" * 64
    selection.write_text(json.dumps(bad))
    try:
        selected_prompts(selection, audit, "calibration")
    except ValueError as exc:
        assert "identity mismatch" in str(exc)
    else:
        raise AssertionError("prompt identity mismatch was not rejected")


def test_calibration_tokens_records_short_exclusions():
    """Token manifests keep short-prompt exclusions visible instead of dropping them."""

    class Tokenizer:
        def encode(self, prompt, add_special_tokens):
            assert add_special_tokens is True
            return [1, 2] if prompt == "short" else list(range(8))

    prompts = [
        {"prompt": "short", "prompt_sha256": "a"},
        {"prompt": "long", "prompt_sha256": "b"},
    ]
    manifest = calibration_tokens(
        Tokenizer(),
        prompts,
        {"lenses": {"calibration_max_tokens": 128, "skip_first": 4}},
    )
    assert [row["prompt_sha256"] for row in manifest["rows"]] == ["b"]
    assert manifest["excluded"][0]["reason"] == "no_valid_calibration_positions"


def test_preflight_writes_blockers_without_loading_model(tmp_path):
    """The local preflight produces an outcome-independent machine-readable report."""
    config, selection, audit = _write_fixture_files(tmp_path)
    out = tmp_path / "preflight.json"
    report = preflight_status(
        config,
        selection,
        audit,
        role="comparison",
        output_path=out,
        minimum_local_free_gib=1e9,
    )
    assert out.is_file()
    assert report["experimental_component_results"] is None
    assert report["does_not_read_experimental_outcomes"] is True
    assert report["calibration_contexts_resolved"] == 1
    assert "local_free_disk_below_1e+09GiB" in report["blockers"]
    assert json.loads(out.read_text())["blockers"] == report["blockers"]


def test_token_manifest_identity_must_match_role_and_inputs(tmp_path):
    """Native phases reject manifests for the wrong model role or stale config."""
    config, selection, _audit = _write_fixture_files(tmp_path)
    identity = {
        "model_role": "comparison",
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    }
    manifest = {
        "subset": "calibration",
        "identity": identity,
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "test-revision",
        "rows": [
            {
                "prompt_sha256": hashlib.sha256(b"alpha prompt").hexdigest(),
                "token_ids": [1, 2, 3, 4, 5, 6],
            }
        ],
        "excluded": [],
    }
    loaded_config = {
        "selection": {"comparison": "Qwen/Qwen3.5-4B"},
        "models": {"comparison": {"revision": "test-revision"}},
    }
    report = validate_token_manifest(
        manifest,
        config_path=config,
        selection_path=selection,
        config=loaded_config,
        role="comparison",
    )
    assert report["status"] == "ok"
    manifest["subset"] = "main_test"
    with pytest.raises(ValueError, match="calibration subset"):
        validate_token_manifest(
            manifest,
            config_path=config,
            selection_path=selection,
            config=loaded_config,
            role="comparison",
        )
    manifest["subset"] = "calibration"
    manifest["rows"][0]["prompt_sha256"] = "test-prompt"
    with pytest.raises(ValueError, match="non-calibration rows"):
        validate_token_manifest(
            manifest,
            config_path=config,
            selection_path=selection,
            config=loaded_config,
            role="comparison",
        )
    manifest["model_revision"] = "other"
    try:
        validate_token_manifest(
            manifest,
            config_path=config,
            selection_path=selection,
            config=loaded_config,
            role="comparison",
        )
    except ValueError as exc:
        assert "model_revision" in str(exc)
    else:
        raise AssertionError("stale token manifest revision was not rejected")


def test_native_loader_status_is_explicit():
    """Environment-dependent loader inspection still returns stable keys."""
    status = native_loader_status()
    if not status["qwen3_5_class_exported"]:
        assert status["selected_loader"] is None
    assert set(status) == {
        "transformers_qwen3_5_module",
        "qwen3_5_class_exported",
        "auto_causal_lm_available",
        "auto_image_text_to_text_available",
        "selected_loader",
        "warnings",
    }


def test_native_numerical_gate_uses_actual_decoder_and_restores_dtype():
    """Exercise the complete native validation body on actual Qwen2 modules."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    result = native_forward_validation(
        model, model.model, [1, 2, 3, 4, 5, 6], {"source_layer": 1, "target_layer": 2}
    )
    assert result["ordinary_numerical_validation"]["status"] == "passed"
    assert result["forward_bit_identical"]
    assert len(result["ordinary_numerical_validation"]["records"]) == 3
    assert all(p.dtype == torch.float32 for p in model.parameters())
