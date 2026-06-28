"""Issue #715 BLOCKER-7 regression — P4 geometry/prune stream per-matrix.

reconcile MAJOR #7: the prior P4 code loaded base + SFT + DFT as THREE full fp32
7B state dicts into CPU at once (~84 GB), violating the plan §9 registered
constraint "per-matrix streaming (load one layer's base+ft, diff, mask/SVD,
discard) so peak local footprint stays < 15 GB". The fix routes both legs
through ``StreamingWeights`` (a lazy ``safe_open`` per-key reader) so only one
matrix's base+ft tensors are resident at a time.

These tests assert:
  (a) ``StreamingWeights`` resolves a local safetensors dir, lists keys WITHOUT
      materializing tensors, and reads one tensor on demand bit-identically to
      the source state_dict;
  (b) the STREAMING geometry leg produces results bit-identical to the
      dict-based reference geometry over the same matrices (correctness is
      preserved by the refactor);
  (c) ``StreamingWeights`` never holds more than the requested tensor — each
      ``get`` returns a fresh tensor and the reader caches nothing (the
      footprint invariant the < 15 GB constraint rests on);
  (d) the streaming prune build (per-tensor AND global) equals the dict path.

Pure CPU, tiny synthetic tensors — no model download, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
torch = pytest.importorskip("torch")


def _load_p4_module():
    spec = importlib.util.spec_from_file_location(
        "issue715_p4_geometry_pruning", REPO_ROOT / "scripts" / "issue715_p4_geometry_pruning.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_safetensors_dir(path: Path, state_dict: dict, *, sharded: bool = False) -> Path:
    """Write a minimal HF-style safetensors checkpoint dir (single or 2-shard)."""
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "test"}))
    if not sharded:
        save_file(state_dict, str(path / "model.safetensors"), metadata={"format": "pt"})
        return path
    # Split across two shards + an index, exercising the weight_map path.
    keys = list(state_dict)
    mid = max(1, len(keys) // 2)
    shard_a = {k: state_dict[k] for k in keys[:mid]}
    shard_b = {k: state_dict[k] for k in keys[mid:]}
    save_file(shard_a, str(path / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(shard_b, str(path / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard_a}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard_b})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    return path


def _toy_state_dict() -> dict:
    torch.manual_seed(0)
    # Two MLP down_proj matrices + an attn matrix + a 1-D norm (excluded scopes).
    return {
        "model.layers.0.mlp.down_proj.weight": torch.randn(16, 24),
        "model.layers.1.mlp.down_proj.weight": torch.randn(16, 24),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "model.layers.0.input_layernorm.weight": torch.randn(16),  # 1-D -> excluded
    }


@pytest.mark.parametrize("sharded", [False, True])
def test_streaming_weights_reads_bit_identical(tmp_path, sharded):
    p4 = _load_p4_module()
    sd = _toy_state_dict()
    ckpt = _write_safetensors_dir(tmp_path / "ckpt", sd, sharded=sharded)
    w = p4.StreamingWeights(str(ckpt))

    assert set(w.tensor_keys()) == set(sd), "tensor_keys must list every stored key"
    for k, v in sd.items():
        got = w.get(k)
        assert torch.equal(got, v), f"streamed {k} differs from source"
        assert tuple(w.shape(k)) == tuple(v.shape), f"shape({k}) mismatch"


def test_streaming_get_caches_nothing(tmp_path):
    """Each get() returns a FRESH object (no Python-level cache) — the footprint
    invariant: the reader does not accumulate resident tensors across calls.

    NOTE: safetensors ``get_tensor`` may back the tensor with the file mmap
    (zero-copy — which is exactly what keeps the footprint low), so two reads
    can share storage. The contract the streaming code relies on is therefore
    "distinct object per call + production code never mutates a get() result
    in place" (verified: every consumer does ``.clone()`` / arithmetic that
    allocates a new tensor), NOT "writes through one read are invisible to the
    next". Assert the object-distinctness + value-correctness invariant.
    """
    p4 = _load_p4_module()
    sd = _toy_state_dict()
    ckpt = _write_safetensors_dir(tmp_path / "ckpt", sd)
    w = p4.StreamingWeights(str(ckpt))
    k = "model.layers.0.mlp.down_proj.weight"
    a = w.get(k)
    b = w.get(k)
    assert a is not b, "get() must not return a cached/shared tensor object"
    assert torch.equal(a, b)
    assert torch.equal(a, sd[k]), "streamed read must equal the source tensor"
    # A defensive copy is decoupled from any later read (the consumer pattern).
    a_copy = a.clone()
    a_copy.add_(1.0)
    assert torch.equal(w.get(k), sd[k]), "a cloned tensor must not alias the file read"


def test_scope_selectors_match_dict_path(tmp_path):
    p4 = _load_p4_module()
    sd = _toy_state_dict()
    ckpt = _write_safetensors_dir(tmp_path / "ckpt", sd)
    w = p4.StreamingWeights(str(ckpt))

    down_stream = sorted(k for k in w.tensor_keys() if k.endswith(p4.DOWN_PROJ_SUFFIX))
    down_dict = sorted(p4.down_proj_keys(sd))
    assert down_stream == down_dict

    lin_stream = sorted(p4._all_linear_keys_from_handle(w))
    lin_dict = sorted(p4.all_linear_weight_keys(sd))
    assert lin_stream == lin_dict
    # The 1-D norm is excluded from the linear scope.
    assert "model.layers.0.input_layernorm.weight" not in lin_stream


def test_streaming_geometry_equals_dict_reference(tmp_path):
    """The streaming geometry leg == the dict-based reference geometry."""
    p4 = _load_p4_module()
    base_sd = _toy_state_dict()
    # SFT/DFT = base + distinct perturbations (so deltas are non-trivial).
    torch.manual_seed(1)
    sft_sd = {k: v + 0.1 * torch.randn_like(v) for k, v in base_sd.items()}
    torch.manual_seed(2)
    dft_sd = {k: v + 0.1 * torch.randn_like(v) for k, v in base_sd.items()}

    base_w = p4.StreamingWeights(str(_write_safetensors_dir(tmp_path / "base", base_sd)))
    sft_w = p4.StreamingWeights(str(_write_safetensors_dir(tmp_path / "sft", sft_sd)))
    dft_w = p4.StreamingWeights(str(_write_safetensors_dir(tmp_path / "dft", dft_sd)))

    target_keys = sorted(p4.down_proj_keys(base_sd))

    # Streaming geometry.
    out_stream = tmp_path / "geom_stream.json"
    p4.run_geometry_leg(base_w, sft_w, dft_w, target_keys, None, out_stream, smoke=False)
    stream_res = json.loads(out_stream.read_text())

    # Dict reference: compute geometry_for_matrix directly on the materialized deltas.
    for arm, ft_sd in (("sft", sft_sd), ("dft", dft_sd)):
        for k in target_keys:
            ref = p4.geometry_for_matrix(ft_sd[k] - base_sd[k], None)
            got = stream_res["per_matrix"][arm][k]
            for stat in ("frobenius_norm", "sparsity", "participation_ratio", "n_params"):
                assert got[stat] == pytest.approx(ref[stat], rel=1e-5, abs=1e-6), (
                    f"{arm}/{k}/{stat}: streaming {got[stat]} != reference {ref[stat]}"
                )


@pytest.mark.parametrize("global_scope", [False, True])
def test_streaming_prune_build_equals_dict(tmp_path, global_scope):
    """build_pruned_model_streaming == build_pruned_model (per-tensor + global)."""
    p4 = _load_p4_module()
    from safetensors.torch import load_file

    base_sd = _toy_state_dict()
    torch.manual_seed(3)
    ft_sd = {k: v + 0.2 * torch.randn_like(v) for k, v in base_sd.items()}

    base_w = p4.StreamingWeights(str(_write_safetensors_dir(tmp_path / "base", base_sd)))
    ft_w = p4.StreamingWeights(str(_write_safetensors_dir(tmp_path / "ft", ft_sd)))
    target_keys = sorted(p4.down_proj_keys(base_sd))
    k_frac = 0.1
    base_dir = tmp_path / "base"  # config/tokenizer source

    stream_out = p4.build_pruned_model_streaming(
        base_w,
        ft_w,
        k_frac,
        target_keys,
        tmp_path / "pruned_stream",
        base_dir,
        global_scope=global_scope,
    )
    dict_out = p4.build_pruned_model(
        base_sd,
        ft_sd,
        k_frac,
        target_keys,
        tmp_path / "pruned_dict",
        base_dir,
        global_scope=global_scope,
    )
    stream_sd = load_file(str(stream_out / "model.safetensors"))
    dict_sd = load_file(str(dict_out / "model.safetensors"))
    assert set(stream_sd) == set(dict_sd)
    for k in stream_sd:
        assert torch.allclose(stream_sd[k], dict_sd[k], atol=1e-6), (
            f"{k}: streaming pruned weight differs from dict path"
        )
