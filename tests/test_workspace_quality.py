"""Calibration controls preserve ragged weighting, native positions and uploaded inputs."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.workspace_decomposition import ValidatedDictionary
from explore_persona_space.analysis.workspace_lenses import nonnegative_gradient_pursuit
from explore_persona_space.analysis.workspace_quality import (
    equal_prompt_quality,
    nearest_quality_match,
    prompt_moments,
)
from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json, save_tensors


@pytest.fixture
def quality_module(monkeypatch):
    from explore_persona_space.orchestrate import env

    monkeypatch.setattr(env, "load_dotenv", lambda: None)
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/workspace_jr_calibration_controls.py")
    )


def test_equal_prompt_moments_match_explicit_ragged_weights():
    h = [np.array([[1.0, 2.0]]), np.array([[7.0, 4.0], [3.0, 8.0], [9.0, 6.0]])]
    s = [row * np.array([1.4, -0.2]) for row in h]
    actual = equal_prompt_quality([prompt_moments(a, b) for a, b in zip(h, s, strict=True)])
    weights = np.array([0.5, 1 / 6, 1 / 6, 1 / 6])
    full, component = np.concatenate(h), np.concatenate(s)
    for key, values in [("full", full), ("component", component), ("rest", full - component)]:
        mean = weights @ values
        expected = weights @ np.square(values - mean).sum(1)
        assert actual["variance_trace"][key] == pytest.approx(expected)
    assert actual["variance_identity_error"] == pytest.approx(0, abs=1e-13)
    assert actual["tokens"] == 4 and actual["prompts"] == 2


def test_zero_energy_and_constant_targets_stay_visible():
    zero = equal_prompt_quality([prompt_moments(np.zeros((2, 3)), np.zeros((2, 3)))])
    assert zero["residual_energy_fraction"] is None
    assert zero["captured_variance_fraction"] is None
    constant = equal_prompt_quality([prompt_moments(np.ones((2, 3)), np.zeros((2, 3)))])
    assert constant["residual_energy_fraction"] == 1
    assert constant["captured_variance_fraction"] is None
    with pytest.raises(ValueError, match="finite"):
        prompt_moments(np.array([[np.nan]]), np.ones((1, 1)))


def test_matching_exposes_outside_range_and_ties():
    def point(value, atoms=3):
        return {
            "residual_energy_fraction": value,
            "captured_variance_fraction": 0.5,
            "mean_active_atoms": atoms,
        }

    controls = {5: point(0.75), 10: point(0.5), 25: point(0.25)}
    match = nearest_quality_match(point(0.625), controls)
    assert match["selected_k"] == 5
    assert match["status"] == "approximate_within_range"
    match = nearest_quality_match(point(0.01), controls)
    assert match["selected_k"] == 25
    assert match["status"] == "unmatched_outside_range"
    with pytest.raises(ValueError, match="every registered"):
        nearest_quality_match(point(0.5), {5: point(0.5)})


def test_nonbinary_constant_and_large_offset_variances_are_stable():
    h = np.full((7, 3), 0.1)
    actual = equal_prompt_quality([prompt_moments(h, h / 4)])
    assert actual["variance_trace"]["full"] == 0
    assert actual["captured_variance_fraction"] is None
    h = 1e8 + np.array([[0, 1], [1, 2], [2, 3]], dtype=np.float64)
    actual = equal_prompt_quality([prompt_moments(h, h / 4)])
    assert actual["variance_trace"]["full"] == pytest.approx(4 / 3)
    assert actual["captured_variance_fraction"] == pytest.approx(1 / 16)
    assert actual["variance_identity_error"] == pytest.approx(0, abs=1e-13)


def test_native_positions_execute_actual_decoder_and_remove_hook(quality_module):
    from transformers import Qwen2Config, Qwen2Model

    cfg = Qwen2Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    cfg._attn_implementation = "eager"
    model = Qwen2Model(cfg).eval()
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    actual = quality_module["native_calibration_states"](model, ids, 0, 4)
    with torch.no_grad():
        expected = model(
            input_ids=torch.tensor([ids]), use_cache=False, output_hidden_states=True
        ).hidden_states[1][0, 4:-1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not model.layers[0]._forward_hooks


@pytest.mark.parametrize("length", [3, 129])
def test_actual_calibration_pursuit_matches_reference(quality_module, length):
    torch.manual_seed(182)
    h = torch.randn(length, 6)
    dictionaries = {arm: torch.randn(32, 6) for arm in ("J", "R")}
    dictionaries = {
        arm: value / value.norm(dim=1, keepdim=True) for arm, value in dictionaries.items()
    }
    actual = quality_module["decompose_prompt"](
        h, {arm: ValidatedDictionary(d) for arm, d in dictionaries.items()}
    )
    for arm, dictionary in dictionaries.items():
        for k in (5, 10, 25):
            pieces = [
                nonnegative_gradient_pursuit(h[begin : begin + 128], dictionary, k=k)
                for begin in range(0, length, 128)
            ]
            for field, observed in actual[arm][k].items():
                expected = torch.cat([getattr(part, field) for part in pieces])
                torch.testing.assert_close(
                    observed,
                    expected,
                    rtol=1e-5 if expected.is_floating_point() else 0,
                    atol=1e-6 if expected.is_floating_point() else 0,
                )


def test_real_control_phase_checks_uploads_and_completes(quality_module, tmp_path):
    identity = {
        "model_role": "primary",
        "config_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
        "versions": {},
        "code": {"git_commit": "c" * 40, "git_dirty": False},
        "execution_readiness_sha256": "d" * 64,
    }
    source, out = tmp_path / "source", tmp_path / "out"
    means = tmp_path / "means.pt"
    save_tensors(means, {"fixture": torch.ones(1)})
    arms = {}
    torch.manual_seed(294)
    for arm in ("J", "R"):
        dictionary = torch.randn(32, 6)
        dictionary /= dictionary.norm(dim=1, keepdim=True)
        path = source / "dictionaries" / f"{arm}.pt"
        save_tensors(path, {"dictionary": dictionary})
        arms[arm] = {"sha256": file_sha256(path)}
    save_json(
        source / "dictionaries/manifest.json",
        {
            "identity": identity,
            "pilot_only": False,
            "readiness_sha256": "d" * 64,
            "means_sha256": file_sha256(means),
            "arms": arms,
        },
    )
    contract = {"identity": identity, "readiness_sha256": "d" * 64}
    rows, ledger = [], []
    for i in range(2):
        row = {"prompt_sha256": str(i) * 64, "token_ids": list(range(8 + i))}
        path = out / "captures" / f"prompt-{i:04d}.pt"
        save_tensors(path, {**row, "contract": contract, "states": torch.randn(3 + i, 6)})
        rows.append(row)
        ledger.append(
            {
                "file": str(path.relative_to(out)),
                "sha256": file_sha256(path),
                "prompt_sha256": row["prompt_sha256"],
            }
        )
    save_json(
        out / "capture_complete.json", {"status": "complete", "contract": contract, "files": ledger}
    )

    def receipt(root, path):
        hashes = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}
        save_json(
            path,
            {
                "repo": "superkaiba1/explore-persona-space-data",
                "revision": "e" * 40,
                "prefix": "exploratory_workspace_jr/test",
                "files_verified": len(hashes),
                "verified_sha256": hashes,
            },
        )

    receipt(source, tmp_path / "dictionary_upload.json")
    receipt(out, tmp_path / "capture_upload.json")
    args = SimpleNamespace(
        out=out,
        input_root=source,
        rotation=None,
        device="cpu",
        capture_upload_receipt=tmp_path / "capture_upload.json",
        dictionary_upload_receipt=tmp_path / "dictionary_upload.json",
    )
    evidence = {
        "readiness_sha256": "d" * 64,
        "paths": {"means": means},
        "reports": {"tokens": {"rows": rows}},
    }
    quality_module["decompose"](args, {}, identity, evidence)
    result = json.loads((out / "controls/rotationNone/quality.json").read_text())
    assert result["status"] == "complete"
    assert result["summary"]["J"]["10"]["prompts"] == 2
    assert result["summary"]["J"]["10"]["tokens"] == 7
    with pytest.raises(FileExistsError, match="immutable"):
        quality_module["decompose"](args, {}, identity, evidence)
    original = (out / ledger[0]["file"]).read_bytes()
    with open(out / ledger[0]["file"], "ab") as handle:
        handle.write(b"mutation")
    args.rotation = 20260913
    with pytest.raises(ValueError, match="verified producer upload"):
        quality_module["decompose"](args, {}, identity, evidence)
    (out / ledger[0]["file"]).write_bytes(original)
    rotations = [20260913, 20260914, 20260915]
    for rotation in rotations:
        args.rotation = rotation
        quality_module["decompose"](args, {}, identity, evidence)
    args.quality_upload_receipt = tmp_path / "quality_upload.json"
    receipt(out, args.quality_upload_receipt)
    config = {"decomposition": {"rotations": rotations}}
    quality_module["match"](args, config, identity, evidence)
    matches = json.loads((out / "quality_matches.json").read_text())
    assert matches["status"] == "complete"
    assert set(matches["matching"]) == {"J", "R"}
    assert set(matches["matching"]["J"]["10"]) == set(map(str, rotations))
    with pytest.raises(FileExistsError, match="immutable"):
        quality_module["match"](args, config, identity, evidence)
    (out / "quality_matches.json").rename(out / "quality_matches_prior.json")
    with open(out / "controls/rotationNone/quality.json", "a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="verified producer upload"):
        quality_module["match"](args, config, identity, evidence)
