"""Regression test for the gate_io #600 mirror-divergence sha256 guard (issue #665).

`gate_io.load_cell(..., verify_sha=True)` MUST assert
`sha256(tensors.pt) == meta.sha256_tensors` ON THE LIVE LOAD PATH and RAISE on
mismatch (NOT a warn, NOT behind a debug flag) — plan §12 item 21 acceptance
deliverable. This test trips the guard with a tampered sha WITHOUT any HF /
network access (it monkeypatches the two `hf_hub_download` calls to local fixture
files), so it fails pre-fix (no guard) and passes post-fix (guard raises).
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from explore_persona_space.analysis import gate_io


def _write_fixture(tmp_path, *, correct_sha: bool):
    """Build a tiny valid store cell (tensors.pt + meta.json) under tmp_path.
    When correct_sha=False, meta.sha256_tensors is deliberately wrong (a
    stale-HF-mirror generation simulation)."""
    d = 8
    n_ctx = 2
    n_layer = 2
    tensors = {
        "v_plus": torch.zeros(n_ctx, n_layer, d),
        "v0": torch.zeros(n_ctx, n_layer, d),
        "context_ids": ["src_ctx", "bystander_ctx"],
    }
    # pad to the asserted hidden/layer shape so load_cell's shape asserts pass
    tensors["v_plus"] = torch.zeros(n_ctx, gate_io.EXPECTED_LAYERS, gate_io.EXPECTED_HIDDEN)
    tensors["v0"] = torch.zeros(n_ctx, gate_io.EXPECTED_LAYERS, gate_io.EXPECTED_HIDDEN)
    tp = tmp_path / "tensors.pt"
    torch.save(tensors, tp)
    real_sha = hashlib.sha256(tp.read_bytes()).hexdigest()
    sha = real_sha if correct_sha else ("0" * 64)
    meta = {
        "behavior": "bad_medical",
        "source": "default",
        "arm": "contra",
        "dose": "d1",
        "seed": 42,
        "sha256_tensors": sha,
        "target_context_roles": {"src_ctx": "source-anchor", "bystander_ctx": "bystander"},
    }
    mp = tmp_path / "meta.json"
    mp.write_text(json.dumps(meta))
    return str(tp), str(mp)


def test_load_cell_raises_on_sha_mismatch(tmp_path, monkeypatch):
    """A cell whose recomputed sha256 != meta.sha256_tensors MUST RAISE (the #600
    guard), not warn or silently consume the stale mirror generation."""
    tp, mp = _write_fixture(tmp_path, correct_sha=False)

    def _fake_dl(repo, path, repo_type=None):
        return tp if path.endswith("tensors.pt") else mp

    monkeypatch.setattr(gate_io, "hf_hub_download", _fake_dl)
    with pytest.raises(ValueError, match=r"sha256.*MISMATCH|mirror divergence"):
        gate_io.load_cell("bm_default_contra_d1_seed42", verify_sha=True)


def test_load_cell_passes_on_correct_sha(tmp_path, monkeypatch):
    """The guard is not a false-positive: a matching sha loads cleanly."""
    tp, mp = _write_fixture(tmp_path, correct_sha=True)

    def _fake_dl(repo, path, repo_type=None):
        return tp if path.endswith("tensors.pt") else mp

    monkeypatch.setattr(gate_io, "hf_hub_download", _fake_dl)
    sc = gate_io.load_cell("bm_default_contra_d1_seed42", verify_sha=True)
    assert sc.behavior == "bad_medical"
    assert sc.source_idx == 0  # the source-anchor context
    sc.free()


def test_load_cell_raises_when_meta_has_no_sha(tmp_path, monkeypatch):
    """meta.json without sha256_tensors cannot be verified -> refuse to consume."""
    tp, mp = _write_fixture(tmp_path, correct_sha=True)
    meta = json.loads((tmp_path / "meta.json").read_text())
    del meta["sha256_tensors"]
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    def _fake_dl(repo, path, repo_type=None):
        return tp if path.endswith("tensors.pt") else mp

    monkeypatch.setattr(gate_io, "hf_hub_download", _fake_dl)
    with pytest.raises(ValueError, match="no sha256_tensors"):
        gate_io.load_cell("bm_default_contra_d1_seed42", verify_sha=True)
