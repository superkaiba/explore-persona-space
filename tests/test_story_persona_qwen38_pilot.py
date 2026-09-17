"""Boundary, coverage, and stale-checkpoint checks for the geometry pilot."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.story_persona_qwen38_pilot import (
    capture_last,
    digest,
    file_digest,
    pack_batches,
    phase_analyze,
    read_checksums,
    read_manifest,
    validate_chunk,
    write_json,
)


class TinyDecoder(torch.nn.Module):
    """Expose the same decoder-block/hook contract without loading model weights."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(32, 4)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.arange(128).reshape(32, 4))
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(64)])

    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache,
        output_hidden_states,
        return_dict,
        logits_to_keep=1,
    ):
        """Return block states with a deliberately different final normalization."""
        assert use_cache is False
        assert return_dict is True
        assert logits_to_keep == 1
        hidden = self.model.embed_tokens(input_ids)
        hidden = (hidden * attention_mask[..., None]).cumsum(dim=1)
        states = [hidden]
        for layer in self.model.layers:
            hidden = layer(hidden)
            states.append(hidden)
        states[-1] = 2 * hidden
        return SimpleNamespace(hidden_states=states if output_hidden_states else None)


def test_last_context_token_is_per_row_and_final_block_is_pre_norm():
    """Right padding must not move the selected position or normalize the last block."""
    model = TinyDecoder()
    values, errors = capture_last(model, [[2, 3], [4, 5, 6, 7]], 0, check_tuple=True)
    expected = torch.stack(
        [model.model.embed_tokens(torch.tensor(ids)).sum(0) for ids in [[2, 3], [4, 5, 6, 7]]]
    )
    torch.testing.assert_close(values[:, 0], expected)
    torch.testing.assert_close(values[:, -1], expected)
    assert max(errors.values()) == 0
    assert all(not block._forward_hooks for block in model.model.layers)


def test_packing_preserves_every_row_and_both_budgets():
    """Unequal lengths and a boundary-size row cannot cause dropped or duplicate rows."""
    lengths = [3, 10, 7, 2, 5, 10, 1]
    batches = pack_batches(lengths, row_limit=3, token_limit=20)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    assert all(
        len(batch) <= 3 and len(batch) * max(lengths[i] for i in batch) <= 20 for batch in batches
    )
    with pytest.raises(ValueError, match="invalid"):
        pack_batches([21], 3, 20)


def test_resume_rejects_wrong_ids_recipe_nonfinite_and_dtype(tmp_path):
    """Resume requires matching row order and content, not just an existing file."""
    path = tmp_path / "chunk.pt"
    valid = {
        "fingerprint": "pin",
        "indices": [1, 0],
        "vectors": torch.ones(2, 64, 4, dtype=torch.bfloat16),
    }
    torch.save(valid, path)
    validate_chunk(path, "pin", [1, 0], (2, 64, 4))
    with pytest.raises(RuntimeError, match="stale"):
        validate_chunk(path, "other", [1, 0], (2, 64, 4))
    with pytest.raises(RuntimeError, match="stale"):
        validate_chunk(path, "pin", [0, 1], (2, 64, 4))
    valid["vectors"] = valid["vectors"].float()
    torch.save(valid, path)
    with pytest.raises(RuntimeError, match="stale"):
        validate_chunk(path, "pin", [1, 0], (2, 64, 4))
    valid["vectors"] = valid["vectors"].bfloat16()
    valid["vectors"][0, 0, 0] = float("nan")
    torch.save(valid, path)
    with pytest.raises(RuntimeError, match="stale"):
        validate_chunk(path, "pin", [1, 0], (2, 64, 4))


def test_resume_rejects_finite_tensor_edits_and_conflicting_checksum_records(tmp_path):
    """Finite tensor edits must fail even when IDs, dimensions and recipe still match."""
    path = tmp_path / "chunk.pt"
    chunk = {
        "fingerprint": "pin",
        "indices": [0],
        "vectors": torch.ones(1, 2, 4, dtype=torch.bfloat16),
    }
    torch.save(chunk, path)
    checksum = file_digest(path)
    record = {"fingerprint": "pin", "chunk_sha256": {path.name: checksum}}
    write_json(tmp_path / "capture_chunks.json", record)
    assert read_checksums(tmp_path, "pin") == {path.name: checksum}
    chunk["vectors"][0, 0, 0] += 1
    torch.save(chunk, path)
    with pytest.raises(RuntimeError, match="content changed"):
        validate_chunk(path, "pin", [0], (1, 2, 4), expected_sha256=checksum)
    record["chunk_sha256"][path.name] = file_digest(path)
    write_json(tmp_path / "capture_complete.json", record)
    with pytest.raises(RuntimeError, match="conflicting"):
        read_checksums(tmp_path, "pin")


def test_manifest_rejects_changed_model_metadata(tmp_path):
    """Changing labels or model metadata invalidates the stored specification digest."""
    path = tmp_path / "manifest.json"
    spec = {"model": "original", "layers": [0, 1]}
    manifest = {"fingerprint": digest(spec), "spec": spec}
    write_json(path, manifest)
    assert read_manifest(path) == manifest
    spec["model"] = "edited"
    write_json(path, manifest)
    with pytest.raises(RuntimeError, match="specification/fingerprint"):
        read_manifest(path)


def test_streamed_analysis_matches_direct_centroids_and_cosine(tmp_path):
    """Shuffled cross-persona chunks must preserve the pairing and centering bank."""
    names = ["a", "b", "c", "d"]
    qids = ["q0", "q1", "q2", "q3"]
    rows = [
        {"persona": name, "question_id": qid, "token_count": 7 + i}
        for i, name in enumerate(names)
        for qid in qids
    ]
    values = torch.randn(16, 2, 5, generator=torch.Generator().manual_seed(18)).bfloat16()
    batches = [list(range(offset, 16, 3)) for offset in range(3)]
    spec = {
        "model": {"hidden_dim": 5},
        "layers": [0, 1],
        "prompts": [{"id": name, "primary": i < 3} for i, name in enumerate(names)],
        "question_ids": qids,
        "batches": batches,
        "inputs_sha256": digest(rows),
    }
    fingerprint = digest(spec)
    write_json(tmp_path / "manifest.json", {"fingerprint": fingerprint, "spec": spec})
    write_json(tmp_path / "rows.json", rows)
    (tmp_path / "chunks").mkdir()
    checksums = {}
    for i, indices in enumerate(batches):
        path = tmp_path / "chunks" / f"batch_{i:04d}.pt"
        torch.save(
            {"fingerprint": fingerprint, "indices": indices, "vectors": values[indices]}, path
        )
        checksums[path.name] = file_digest(path)
    write_json(
        tmp_path / "capture_complete.json",
        {"fingerprint": fingerprint, "row_count": 16, "chunk_sha256": checksums},
    )
    phase_analyze(OmegaConf.create({}), tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    direct = values.double().reshape(4, 4, 2, 5).mean(1)
    np.testing.assert_allclose(np.load(tmp_path / "centroids.npz")["centroids"], direct.numpy())
    centered = direct - direct.mean(0)
    unit = centered / centered.norm(dim=-1, keepdim=True)
    expected = torch.einsum("plh,qlh->lpq", unit, unit)
    np.testing.assert_allclose(summary["centered_cosine"], expected.numpy(), atol=1e-6)
    assert summary["counts_by_half"] == [[2] * 4, [2] * 4]
