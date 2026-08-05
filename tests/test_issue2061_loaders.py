"""#2061 turnstore-loader pins (code-review v1 C1, Unit A).

Round-1 wrote all three data loaders against a FABRICATED #1336 shard schema
(`context_L29`-style keys) and hardcoded `_shard000.pt`. These tests push a
1-shard-and-2-shard fixture written in the PRODUCER'S real payload shape
(`scripts/issue1336_extract_turnstore.py::write_shards`: bf16, the 5 keys
{conv_ids, slots, profiles, nll, spans_meta}, per-record lists, ≥2 records,
≥2 turns so the a1 answer-row selection is exercised) through each of the
three loaders:

- `issue2061_sae_encode._load_turnstore_state` (hub boundary faked with
  signature-mirroring fakes; `iter_local_shards`'s real body executes),
- `issue2061_fit_per_feature._load_arm_inputs` (pure local; BOTH arms),
- `issue2061_fitness.load_lmsys_validation_activations` (hub generator
  faked at the same boundary).

They FAIL against the pre-fix loaders (fabricated keys -> KeyError; the
post-fix seams do not exist) and PASS after — verified by stashing the fix.

Fixture values are exact in bf16 (small integers + 0.5), so equality checks
are exact after the bf16 -> float32 round trip:
    slots[j, layer]    == g*16 + j*4 + layer          (j: 0=prefix, 1=a1)
    profiles[j, layer] == g*16 + j*4 + layer + 0.5    (j: 0=u1,     1=a1)
with g the record's global index.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2061_fit_per_feature as fpf
import issue2061_fitness as fitness
import issue2061_sae_encode as enc
import issue2061_turnstore as ts

N_LAYERS = 4
HIDDEN = 8
LAYER = 2  # test layer (production LAYER=29 is an arg everywhere)
STEM = "turnstore_base_chat_lmsys23k"


def _make_record(g: int) -> dict:
    """One producer-shaped record; values exact in bf16 (see module docstring)."""
    slots = torch.empty(2, N_LAYERS, HIDDEN, dtype=torch.bfloat16)
    profiles = torch.empty(2, N_LAYERS, HIDDEN, dtype=torch.bfloat16)
    for j in range(2):
        for layer in range(N_LAYERS):
            slots[j, layer, :] = float(g * 16 + j * 4 + layer)
            profiles[j, layer, :] = float(g * 16 + j * 4 + layer) + 0.5
    return {
        "conv_id": f"conv-{g:03d}",
        "slots": slots,
        "profiles": profiles,
        "nll": torch.tensor([0.1, 0.2]),
        "spans_meta": {
            "conv_id": f"conv-{g:03d}",
            "format": "chat",
            "seq_len": 64,
            "slot_names": ["prefix", "a1"],
            "slot_idx": {"prefix": 3, "a1": 20},
            "turn_names": ["u1", "a1"],
            "spans": {"u1": [5, 18], "a1": [22, 60]},
            "meta": {"n_tokens": 64},
        },
    }


def _payload(gs: list[int]) -> dict:
    recs = [_make_record(g) for g in gs]
    return {
        "conv_ids": [r["conv_id"] for r in recs],
        "slots": [r["slots"] for r in recs],
        "profiles": [r["profiles"] for r in recs],
        "nll": [r["nll"] for r in recs],
        "spans_meta": [r["spans_meta"] for r in recs],
    }


def _write_shard(dir_: Path, shard_idx: int, gs: list[int], stem: str = STEM) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    payload = _payload(gs)
    p = dir_ / f"{stem}_shard{shard_idx:03d}.pt"
    torch.save(payload, p)
    # JSON sidecar, as the producer writes (loaders must NOT match it).
    (dir_ / f"{stem}_shard{shard_idx:03d}.json").write_text(
        json.dumps(
            {
                "shard_index": shard_idx,
                "n_conversations": len(gs),
                "conv_ids": payload["conv_ids"],
            }
        )
    )
    return p


def _expected(gs: list[int], state: str, layer: int = LAYER) -> torch.Tensor:
    base = {"prefix": 0, "context": 4, "answer": 4}[state]
    frac = 0.5 if state == "answer" else 0.0
    col = torch.tensor([g * 16 + base + layer + frac for g in gs], dtype=torch.float32)
    return col.unsqueeze(1).expand(len(gs), HIDDEN)


# ---------------------------------------------------------------------------
# Shared extraction module
# ---------------------------------------------------------------------------
def test_extract_state_rows_selects_correct_rows():
    payload = _payload([0, 1, 2])
    for state in ("prefix", "context", "answer"):
        x, conv_ids = ts.extract_state_rows(payload, state=state, layer=LAYER, src="mem")
        assert x.shape == (3, HIDDEN), (state, x.shape)
        assert x.dtype == torch.float32
        assert torch.equal(x, _expected([0, 1, 2], state)), state
        assert conv_ids == ["conv-000", "conv-001", "conv-002"]


def test_extract_answer_differs_from_query_turn():
    """The a1 (answer) profile row is selected — NOT the u1 (query) row."""
    payload = _payload([1])
    x, _ = ts.extract_state_rows(payload, state="answer", layer=LAYER, src="mem")
    u1_value = 1 * 16 + 0 * 4 + LAYER + 0.5  # what the WRONG (u1) row would read
    assert torch.equal(x, _expected([1], "answer"))
    assert not torch.allclose(x, torch.full_like(x, u1_value))


def test_schema_assert_fail_loud():
    payload = _payload([0])
    payload.pop("profiles")
    with pytest.raises(KeyError) as ei:
        ts.extract_state_rows(payload, state="answer", layer=LAYER, src="badshard.pt")
    msg = str(ei.value)
    assert "profiles" in msg and "write_shards" in msg and "badshard.pt" in msg
    with pytest.raises(TypeError):
        ts.assert_shard_schema(torch.zeros(3), src="tensor.pt")


def test_layer_out_of_range_fail_loud():
    with pytest.raises(IndexError) as ei:
        ts.extract_state_rows(_payload([0]), state="answer", layer=99, src="mem")
    assert "layer 99" in str(ei.value)


def test_enumerate_shards_order_and_fail_loud(tmp_path):
    d = tmp_path / STEM
    _write_shard(d, 1, [2, 3])
    _write_shard(d, 0, [0, 1])
    _write_shard(d, 2, [4])
    paths = ts.enumerate_shards(d)
    assert [p.name for p in paths] == [f"{STEM}_shard{i:03d}.pt" for i in range(3)]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ts.enumerate_shards(empty)


# ---------------------------------------------------------------------------
# P2 fit loader — BOTH arms, ALL shards (kills the _shard000 hardcode)
# ---------------------------------------------------------------------------
def test_fit_per_feature_loader_both_arms_all_shards(tmp_path):
    d = tmp_path / STEM
    _write_shard(d, 0, [0, 1])
    _write_shard(d, 1, [2, 3, 4])
    for arm in ("prefix", "context"):
        x, conv_ids = fpf._load_arm_inputs(d, arm, layer=LAYER)
        assert x.shape == (5, HIDDEN), (arm, x.shape)  # 2 + 3 rows: BOTH shards
        assert torch.equal(torch.from_numpy(x), _expected([0, 1, 2, 3, 4], arm)), arm
        assert conv_ids == [f"conv-{g:03d}" for g in range(5)], arm


# ---------------------------------------------------------------------------
# P1 encode loader — hub boundary faked, real iter_local_shards body
# ---------------------------------------------------------------------------
def _fake_hub(monkeypatch, module, fixture_dir: Path):
    """Signature-mirroring fakes at the network boundary ONLY."""

    def fake_hub_shard_files(tree_path: str, revision: str | None = None) -> list[str]:
        return [f"{tree_path}/{p.name}" for p in ts.enumerate_shards(fixture_dir)]

    def fake_hf_hub_download(
        repo_id: str,
        filename: str,
        repo_type: str | None = None,
        revision: str | None = None,
    ) -> str:
        return str(fixture_dir / Path(filename).name)

    monkeypatch.setattr(module, "hub_shard_files", fake_hub_shard_files)
    monkeypatch.setattr(module, "hf_hub_download", fake_hf_hub_download)


def test_sae_encode_loader_answer_state_all_shards(tmp_path, monkeypatch):
    d = tmp_path / STEM
    _write_shard(d, 0, [0, 1])
    _write_shard(d, 1, [2, 3, 4])
    _fake_hub(monkeypatch, enc, d)
    x, conv_ids = enc._load_turnstore_state("some/tree", state="answer", layer=LAYER)
    assert x.shape == (5, HIDDEN)
    assert torch.equal(x, _expected([0, 1, 2, 3, 4], "answer"))
    assert conv_ids == [f"conv-{g:03d}" for g in range(5)]  # keyed X/Y alignment (M1)


def test_sae_encode_loader_max_rows_stops_early(tmp_path, monkeypatch):
    d = tmp_path / STEM
    _write_shard(d, 0, [0, 1])
    _write_shard(d, 1, [2, 3, 4])
    _fake_hub(monkeypatch, enc, d)
    downloads: list[str] = []
    real_fake = enc.hf_hub_download

    def counting_download(repo_id, filename, repo_type=None, revision=None):
        downloads.append(filename)
        return real_fake(repo_id, filename, repo_type=repo_type, revision=revision)

    monkeypatch.setattr(enc, "hf_hub_download", counting_download)
    x, conv_ids = enc._load_turnstore_state("some/tree", state="context", layer=LAYER, max_rows=2)
    assert x.shape == (2, HIDDEN)
    assert torch.equal(x, _expected([0, 1], "context"))
    assert conv_ids == ["conv-000", "conv-001"]
    assert len(downloads) == 1  # lazy generator: shard 1 never fetched


# ---------------------------------------------------------------------------
# P1 -> P2/P3 sparse-payload round trip (review M1: the producer writes the
# TopK sparse layout; BOTH consumers open it; dense reconstruction is exactly
# topk_encode; --max-rows debug encodes never collide with production paths)
# ---------------------------------------------------------------------------
def _tiny_sae_weights(d_in: int = HIDDEN, d_sae: int = 24, seed: int = 5):
    rng = torch.Generator().manual_seed(seed)
    return {
        "encoder.weight": torch.randn(d_sae, d_in, generator=rng),
        "encoder.bias": torch.randn(d_sae, generator=rng),
        "W_dec": torch.randn(d_sae, d_in, generator=rng),
        "b_dec": torch.randn(d_in, generator=rng),
    }


def test_encode_turnstore_sparse_payload_roundtrip(tmp_path, monkeypatch):
    from explore_persona_space.analysis.sparsify_topk_sae import topk_encode

    d = tmp_path / STEM
    _write_shard(d, 0, [0, 1, 2])
    _write_shard(d, 1, [3, 4])
    _fake_hub(monkeypatch, enc, d)
    weights = _tiny_sae_weights()
    out_dir = tmp_path / "encoded"
    turnstore = {"stage": "base", "render": "chat", "corpus": "lmsys23k", "tree_path": "t"}
    out = enc.encode_turnstore(
        turnstore, weights=weights, k=3, output_dir=out_dir, layer=LAYER, device="cpu"
    )
    assert out.name == f"base_chat_lmsys23k_answer_L{LAYER}.pt"

    # Consumer 1 (P2/P3 shared loader): payload opens, conv_ids keyed.
    payload = ts.load_encoded_target(out)
    assert payload["conv_ids"] == [f"conv-{g:03d}" for g in range(5)]
    assert payload["k"] == 3 and payload["d_sae"] == 24
    # Dense reconstruction == the dense encoder on the same rows, exactly.
    x, _ = enc._load_turnstore_state("some/tree", state="answer", layer=LAYER)
    dense_ref = topk_encode(x, weights, k=3)
    assert torch.equal(ts.encoded_to_dense(payload), dense_ref)

    # Skip predicate: a valid same-regime payload is skip-reused...
    out2 = enc.encode_turnstore(
        turnstore, weights=weights, k=3, output_dir=out_dir, layer=LAYER, device="cpu"
    )
    assert out2 == out
    # ...but a stale DENSE store at the canonical path is re-encoded (M3).
    torch.save(dense_ref, out)
    out3 = enc.encode_turnstore(
        turnstore, weights=weights, k=3, output_dir=out_dir, layer=LAYER, device="cpu"
    )
    assert ts.load_encoded_target(out3)["format"] == ts.ENCODED_TARGET_FORMAT

    # --max-rows debug cap writes a SUFFIXED path the production glob and the
    # canonical consumers never see (M3: no skip-reuse of a capped shard).
    capped = enc.encode_turnstore(
        turnstore, weights=weights, k=3, output_dir=out_dir, layer=LAYER, device="cpu", max_rows=2
    )
    assert capped.name == f"base_chat_lmsys23k_answer_L{LAYER}_rows2.pt"
    assert sorted(p.name for p in out_dir.glob(f"*_answer_L{LAYER}.pt")) == [out.name]
    assert ts.load_encoded_target(capped)["n_rows"] == 2


def test_null_stage_inputs_open_sparse_payload_and_key_alignment(tmp_path, monkeypatch):
    import issue2061_null as nullmod

    d = tmp_path / "shards" / STEM
    _write_shard(d, 0, [0, 1, 2])
    _write_shard(d, 1, [3, 4])
    _fake_hub(monkeypatch, enc, d)
    weights = _tiny_sae_weights()
    enc_dir = tmp_path / "encoded"
    turnstore = {"stage": "base", "render": "chat", "corpus": "lmsys23k", "tree_path": "t"}
    out = enc.encode_turnstore(
        turnstore, weights=weights, k=3, output_dir=enc_dir, layer=LAYER, device="cpu"
    )

    # Consumer 2 (P3 null): opens the payload + verifies conv-id alignment.
    x, y_idx, y_val, conv_ids, d_sae = nullmod._load_cell_stage_inputs(
        tmp_path / "shards", enc_dir, "base", "chat", "lmsys23k", "context", layer=LAYER
    )
    assert x.shape == (5, HIDDEN) and y_idx.shape == (5, 3) and y_val.shape == (5, 3)
    assert d_sae == 24 and conv_ids == [f"conv-{g:03d}" for g in range(5)]

    # Alignment is KEYED: a payload from a DIFFERENT turnstore snapshot fails loud.
    payload = ts.load_encoded_target(out)
    ts.save_encoded_target(
        out,
        idx=payload["idx"],
        val=payload["val"],
        d_sae=payload["d_sae"],
        k=payload["k"],
        conv_ids=[f"other-{i}" for i in range(5)],
        cell=payload["cell"],
    )
    with pytest.raises(ValueError, match="row alignment mismatch"):
        nullmod._load_cell_stage_inputs(
            tmp_path / "shards", enc_dir, "base", "chat", "lmsys23k", "context", layer=LAYER
        )


# ---------------------------------------------------------------------------
# P4 fitness loader — answer state through the real filter/slice tail
# ---------------------------------------------------------------------------
def test_fitness_loader_answer_state(tmp_path, monkeypatch):
    d = tmp_path / STEM
    _write_shard(d, 0, [0, 1, 2])
    _write_shard(d, 1, [3, 4])

    def fake_iter_local_shards(tree_path: str, revision: str | None = None):
        yield from (str(p) for p in ts.enumerate_shards(d))

    monkeypatch.setattr(fitness, "iter_local_shards", fake_iter_local_shards)
    x = fitness.load_lmsys_validation_activations("base", layer=LAYER, n_val_rows=3, state="answer")
    # 5 rows loaded; n <= BOS_STRIP so no strip; uniform norms so no outlier
    # drop; sliced to n_val_rows.
    assert x.shape == (3, HIDDEN)
    assert torch.equal(x, _expected([0, 1, 2], "answer"))
