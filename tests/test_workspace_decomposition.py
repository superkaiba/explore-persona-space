"""Nested decomposition must match the immutable reference for every reported k."""

import pytest
import torch

from explore_persona_space.analysis.workspace_capture import decompose_context
from explore_persona_space.analysis.workspace_decomposition import (
    ValidatedDictionary,
    decompose_context_nested,
)
from explore_persona_space.analysis.workspace_lenses import nonnegative_gradient_pursuit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nested_steps_match_reference_including_zero_and_negative_rows(dtype):
    generator = torch.Generator().manual_seed(41)
    dictionary = torch.randn(40, 8, generator=generator, dtype=dtype)
    dictionary /= dictionary.norm(dim=1, keepdim=True)
    x = torch.randn(7, 8, generator=generator, dtype=dtype)
    x[0] = 0
    nested = ValidatedDictionary(dictionary).decompose(x)
    for k, actual in nested.items():
        expected = nonnegative_gradient_pursuit(x, dictionary, k=k)
        for field in expected.__dataclass_fields__:
            torch.testing.assert_close(
                getattr(actual, field), getattr(expected, field), rtol=1e-6, atol=1e-8
            )
    validated = ValidatedDictionary(dictionary)
    with pytest.raises(AttributeError):
        validated.dictionary = torch.zeros_like(dictionary)
    other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    with pytest.raises(ValueError, match="compute dtype"):
        validated.decompose(x.to(other_dtype))
    dictionary.mul_(2)
    with pytest.raises(ValueError, match="modified"):
        validated.decompose(x)


def test_nested_token_aggregation_matches_separate_k_runs_with_unequal_lengths():
    generator = torch.Generator().manual_seed(47)
    dictionary = torch.randn(32, 6, generator=generator)
    dictionary /= dictionary.norm(dim=1, keepdim=True)
    captures = [
        {
            "prompt_sha256": "a",
            "seed": seed,
            "x": torch.ones(6),
            "answer_states": torch.randn(length, 6, generator=generator),
        }
        for seed, length in ((42, 1), (43, 19), (44, 7))
    ]
    nested = decompose_context_nested(
        captures, {arm: ValidatedDictionary(dictionary) for arm in ("J", "R")}, token_batch_size=4
    )
    for k, actual in nested.items():
        expected = decompose_context(
            captures, {"J": dictionary, "R": dictionary}, k=k, token_batch_size=4
        )
        for field in ("targets", "rollout_means"):
            for name in expected[field]:
                torch.testing.assert_close(
                    actual[field][name], expected[field][name], rtol=1e-6, atol=1e-8
                )
