"""Exercise immutable-file publication races and semantic cache validation."""

import copy
import hashlib
import os

import pytest
import torch

from explore_persona_space.analysis.workspace_checkpoint_cache import (
    equal_tree,
    publish_checkpoint,
    read_checkpoint,
    regular_path,
    validate_checkpoint,
)
from explore_persona_space.analysis.workspace_decomposition import (
    ValidatedDictionary,
    decompose_context_nested,
)
from explore_persona_space.analysis.workspace_runtime import content_sha256, file_sha256


@pytest.fixture
def checkpoint(tmp_path):
    identity = {
        "config_sha256": "config",
        "selection_sha256": "selection",
        "model_role": "comparison",
        "versions": {},
        "code": {},
    }
    x = torch.tensor([1.0, 2.0, 3.0])
    canonical = tmp_path / "context.pt"
    torch.save(
        {
            "x": x[None],
            "contract": {
                "identity": identity,
                "policy": "context_only_frozen_order_batches16_no_answer_tokens",
                "source_hashes": ["generation"],
            },
        },
        canonical,
    )
    reference = {
        "context_input_file": "context.pt",
        "context_input_file_sha256": file_sha256(canonical),
        "context_input_row": 0,
        "generation_file_sha256": "generation",
        "identity": identity,
    }
    rows = [
        {
            "seed": i,
            "prompt_sha256": "prompt",
            "x": x,
            "answer_states": torch.arange(3 * (i + 1)).reshape(i + 1, 3).float(),
        }
        for i in (1, 2)
    ]
    basis = ValidatedDictionary(torch.eye(3))
    value = decompose_context_nested(rows, {"J": basis, "R": basis}, checkpoints=(1,))[1]
    contract = {"identity": identity, "dictionaries": {}, "k": 1, "rotation": 20260914}
    value.update(
        contract=contract,
        contract_sha256=content_sha256(contract),
        source_sha256="source",
        context_input_reference=reference,
    )
    return value, {"rows": rows, "identity": reference}, contract, tmp_path


def test_valid_semantics_and_exact_parity(checkpoint):
    value, capture, contract, root = checkpoint
    validate_checkpoint(value, capture, "source", contract, root)
    equal_tree(copy.deepcopy(value), value)


@pytest.mark.parametrize(
    "change", ["seeds", "lengths", "prompt", "noise", "means", "rest", "counts"]
)
def test_rejects_stale_or_malformed_semantics(checkpoint, change):
    value, capture, contract, root = checkpoint
    if change == "seeds":
        value["rollout_seeds"].reverse()
    elif change == "lengths":
        value["token_counts"][0] += 1
    elif change == "prompt":
        value["prompt_sha256"] = "wrong"
    elif change == "noise":
        value["mean_target_noise_trace"]["J"] += 1
    elif change == "means":
        value["targets"]["J"][0] += 1
    elif change == "rest":
        value["rollout_means"]["restJ"][0, 0] += 1
    elif change == "counts":
        value["decomposition_statistics"]["J"]["active_atoms"][0] = 2
    with pytest.raises((ValueError, AssertionError)):
        validate_checkpoint(value, capture, "source", contract, root)


def test_hash_and_load_use_same_inode(tmp_path, monkeypatch):
    path, replacement = tmp_path / "cache.pt", tmp_path / "replacement.pt"
    torch.save({"x": torch.tensor(1)}, path)
    torch.save({"x": torch.tensor(2)}, replacement)
    expected_hash = file_sha256(path)
    original_load = torch.load

    def replace_before_loading(stream, **kwargs):
        os.replace(replacement, path)
        return original_load(stream, **kwargs)

    monkeypatch.setattr(torch, "load", replace_before_loading)
    value, digest = read_checkpoint(path, expected_hash)
    assert value["x"].item() == 1 and digest == expected_hash
    assert file_sha256(path) != expected_hash


def test_publisher_preserves_original_winning_race(tmp_path, monkeypatch):
    source, destination, original = (
        tmp_path / name for name in ("source.pt", "target.pt", "original.pt")
    )
    torch.save({"x": torch.tensor(1)}, source)
    torch.save({"x": torch.tensor(1)}, original)
    original_bytes = original.read_bytes()
    real_link = os.link

    def racing_link(src, dst, **kwargs):
        os.replace(original, destination)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    event = publish_checkpoint(
        source, destination, lambda value: equal_tree(value, {"x": torch.tensor(1)})
    )
    assert event["outcome"] == "preserved_existing_checkpoint"
    assert destination.read_bytes() == original_bytes


def test_original_may_replace_published_cache(tmp_path):
    source, destination, original = (
        tmp_path / name for name in ("source.pt", "target.pt", "original.pt")
    )
    torch.save({"x": torch.tensor(1)}, source)
    torch.save({"x": torch.tensor(1)}, original)
    source_bytes = source.read_bytes()
    event = publish_checkpoint(
        source, destination, lambda value: equal_tree(value, {"x": torch.tensor(1)})
    )
    assert event["outcome"] == "published_optional_checkpoint"
    os.replace(original, destination)
    assert source.read_bytes() == source_bytes
    assert event["observed_sha256"] == hashlib.sha256(source_bytes).hexdigest()


def test_existing_invalid_file_is_preserved_and_rejected(tmp_path):
    source, destination = tmp_path / "source.pt", tmp_path / "target.pt"
    torch.save({"x": torch.tensor(1)}, source)
    torch.save({"x": torch.tensor(2)}, destination)
    original = destination.read_bytes()
    with pytest.raises(AssertionError):
        publish_checkpoint(
            source, destination, lambda value: equal_tree(value, {"x": torch.tensor(1)})
        )
    assert destination.read_bytes() == original


def test_path_and_symlink_rejection(tmp_path):
    (tmp_path / "redirect").symlink_to(tmp_path, target_is_directory=True)
    for relative in ("../escape.pt", "/absolute.pt", "redirect/cache.pt"):
        with pytest.raises(ValueError):
            regular_path(tmp_path, relative)
    path = regular_path(tmp_path, "safe/cache.pt", create_parents=True)
    assert path.parent.is_dir()
