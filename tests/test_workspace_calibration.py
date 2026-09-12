"""Calibration coverage, prompt weighting and readout orientation checks."""

import importlib.util
from pathlib import Path

import pytest
import torch

from explore_persona_space.analysis.workspace_calibration import (
    calibration_means,
    matrix_agreement,
    paired_direction_agreement,
    validate_calibration_order,
)
from explore_persona_space.analysis.workspace_runtime import content_sha256


def test_prefix_does_not_certify_full_calibration_and_equal_prompt_weights(tmp_path):
    """Long prompts receive the same matrix weight as short prompts."""
    tokens = {
        "rows": [{"prompt_sha256": str(i), "token_ids": [1] * (5 + i)} for i in range(4)],
        "excluded": [{"reason": "too_short"}],
    }
    contract = {"tokens_sha256": content_sha256(tokens)}
    paths = []
    for i in range(4):
        path = tmp_path / f"prompt-{i:04d}.pt"
        torch.save(
            {
                **tokens["rows"][i],
                "contract": contract,
                "contract_sha256": content_sha256(contract),
                "J": torch.eye(2) * i,
                "R": torch.eye(2) * (i + 1),
            },
            path,
        )
        paths.append(path)
    means, report = calibration_means(paths[:2], tokens, 2)
    assert not report["full_calibration_membership"]
    torch.testing.assert_close(means["first_2"]["J"], torch.eye(2) / 2)
    means, report = calibration_means(paths, tokens, 2)
    assert report["full_calibration_membership"] and report["selected_prompts"] == 5
    torch.testing.assert_close(means["first_4"]["J"], torch.eye(2) * 1.5)
    with pytest.raises(ValueError, match="frozen prefix"):
        calibration_means(paths[::-1], tokens, 2)


def test_agreement_exposes_scale_change_and_zero_directions():
    """Cosine cannot conceal a scale change; zero readouts have unavailable cosines."""
    a = torch.eye(3)
    metric = matrix_agreement(a, 2 * a)
    assert metric["cosine"] == pytest.approx(1)
    assert metric["relative_frobenius_error_to_right"] == pytest.approx(0.5)
    result = paired_direction_agreement(torch.tensor([[0.0, 0.0], [1.0, 0.0]]), torch.eye(2))
    assert result["valid"].tolist() == [False, True]
    assert torch.isnan(result["cosine"][0])
    assert result["cosine"][1] == 0


def test_calibration_order_checked_against_independent_selection():
    """An internally consistent reordered token manifest changes the registered subsets."""
    rows = [{"prompt_sha256": str(i)} for i in range(5)]
    selection = {"subsets": {"calibration": rows}}
    manifest = {"rows": [rows[i] for i in (0, 2, 3, 4)], "excluded": [rows[1]]}
    validate_calibration_order(manifest, selection)
    manifest["rows"].reverse()
    with pytest.raises(ValueError, match="order differs"):
        validate_calibration_order(manifest, selection)


def test_identity_transport_readouts_equal_actual_native_logits():
    """The readout applies the correct matrix orientation and final norm exactly once."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    path = Path(__file__).parents[1] / "scripts/workspace_jr_calibration.py"
    spec = importlib.util.spec_from_file_location("jr_calibration_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=30,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).eval()

    class Tokenizer:
        def decode(self, ids):
            return str(ids)

    tokens = {"rows": [{"prompt_sha256": "calibration", "token_ids": [1, 2, 3, 4, 5, 6]}]}
    rows = module.readouts(
        model, model.model, Tokenizer(), tokens, {"J": torch.eye(8), "R": torch.eye(8)}, 1, 1
    )
    assert rows and all(
        row["top_token_ids"]["native"] == row["top_token_ids"]["J"] == row["top_token_ids"]["R"]
        for row in rows
    )
