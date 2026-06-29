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


# ── BLOCKER p4-global-topk-materializes-all-deltas regression ───────────────
#
# The prior _global_topk_threshold did ``abs_chunks.append(d)`` for every target
# matrix then ``torch.cat(abs_chunks)`` — materializing every target delta's |Δ|
# vector simultaneously (~26 GB for the all_linear/global cell), the > 15 GB
# footprint plan §9 forbids. The fix keeps only the running n_zero LARGEST |Δ|
# values (a bounded torch min-heap by value). These tests pin BOTH the static
# shape of the fix (no full-accumulation idiom) AND the runtime invariant
# (nothing torch.topk-ed is ever as large as the full concatenation).


def test_global_topk_threshold_no_full_delta_accumulation_static():
    """STATIC guard: the global-threshold source must NOT accumulate every delta.

    A future refactor that reintroduces ``abs_chunks.append`` + ``torch.cat`` over
    all target deltas would silently restore the > 15 GB footprint. Assert the
    banned full-accumulation idiom is absent from the function source.
    """
    import inspect

    p4 = _load_p4_module()
    src = inspect.getsource(p4._global_topk_threshold)
    # The banned shape: building a Python list of every matrix's abs vector then
    # concatenating ALL of them at once.
    assert "abs_chunks" not in src, (
        "_global_topk_threshold must not accumulate every delta into a list "
        "(abs_chunks) — that materializes the full concatenation (>15 GB)"
    )
    # The only torch.cat allowed is the bounded merge of (running_top, one matrix);
    # a torch.cat over a comprehension/list of all chunks is the banned full-accum.
    assert "torch.cat(abs_chunks)" not in src
    assert "running_top" in src, "the bounded running-top set is the required shape"


def test_global_topk_threshold_bounded_footprint_runtime(tmp_path, monkeypatch):
    """RUNTIME guard: no torch.topk inside the global threshold ever sees a tensor
    as large as the full |Δ| concatenation.

    Build several toy matrices, record the largest 1-D tensor passed to
    ``torch.topk`` while computing the global threshold, and assert it stays
    bounded by ``n_zero + (largest single matrix's element count)`` — never the
    sum over all matrices (the old full-cat size). Also assert the streaming
    threshold equals the brute-force ``cat(all).topk(n_zero).values.min()`` so the
    bound does not come at the cost of correctness.
    """
    p4 = _load_p4_module()

    # Five distinct down_proj matrices so the full concatenation is meaningfully
    # larger than any single matrix (and larger than n_zero at a small k_frac).
    torch.manual_seed(7)
    base_sd = {f"model.layers.{i}.mlp.down_proj.weight": torch.randn(20, 30) for i in range(5)}
    torch.manual_seed(8)
    ft_sd = {k: v + 0.3 * torch.randn_like(v) for k, v in base_sd.items()}
    base_w = p4.StreamingWeights(str(_write_safetensors_dir(tmp_path / "base", base_sd)))
    ft_w = p4.StreamingWeights(str(_write_safetensors_dir(tmp_path / "ft", ft_sd)))
    target_keys = sorted(base_sd)

    total = sum(base_sd[k].numel() for k in target_keys)  # 5 * 600 = 3000
    largest_matrix = max(base_sd[k].numel() for k in target_keys)  # 600
    k_frac = 0.05
    n_zero = round(k_frac * total)  # 150

    real_topk = torch.topk
    max_seen = {"n": 0}

    def _spy_topk(t, k, *a, **kw):
        if t.dim() == 1:
            max_seen["n"] = max(max_seen["n"], t.numel())
        return real_topk(t, k, *a, **kw)

    monkeypatch.setattr(p4.torch, "topk", _spy_topk)
    thresh = p4._global_topk_threshold(base_w, ft_w, k_frac, target_keys)

    # The bounded merge tensor is at most (running_top <= n_zero) + one matrix.
    bound = n_zero + largest_matrix
    assert max_seen["n"] <= bound, (
        f"global-threshold topk saw a {max_seen['n']}-elem tensor; bound is "
        f"n_zero+largest_matrix={bound}. The full concatenation would be {total}."
    )
    # And it must be strictly below the full-concat size (the old footprint).
    assert max_seen["n"] < total, (
        f"global-threshold topk materialized the full {total}-elem concatenation"
    )

    # Correctness: identical to the brute-force order statistic.
    monkeypatch.setattr(p4.torch, "topk", real_topk)
    allabs = torch.cat(
        [(ft_sd[k].float() - base_sd[k].float()).abs().flatten() for k in target_keys]
    )
    brute = float(real_topk(allabs, n_zero, largest=True).values.min().item())
    assert thresh == pytest.approx(brute, rel=0, abs=0), (
        f"streaming threshold {thresh} != brute-force {brute}"
    )


# ── v6 §11 budget-guard regression (drop all_linear/global; fail-loud guard) ──
#
# v4's `all_linear/global` P4 cell is the LONE §9 <15 GB footprint breach: the
# global threshold over ~6.5e9 all-linear elements pushes torch.topk's mandatory
# int64 indices buffer to ~16 GB and the co-resident transient to ~39 GB. v6
# DROPS that cell from the production grid and adds a defensive RuntimeError at
# the TOP of `_global_topk_threshold` so a future ad-hoc over-budget invocation
# fails LOUD (before any tensor is materialized) instead of OOM-ing the shared VM.
# The 3 RETAINED cells pass the guard by §9 arithmetic. These tests pin both
# directions, using a shape-only StreamingWeights stub (the guard reads per-key
# shape metadata only — it fires before any `.get()`), keyed to Qwen-2.5-7B
# (H=3584, I=18944, 28 layers; the §9 per-cell arithmetic).

_QWEN_H = 3584
_QWEN_I = 18944
_QWEN_LAYERS = 28


class _ShapeOnlyWeights:
    """A StreamingWeights stand-in that knows only per-key SHAPES.

    `_global_topk_threshold`'s §9 budget guard reads `base_w.shape(k)` for every
    target key and computes the transient peak from those shapes — it raises (or
    proceeds to the streaming loop) BEFORE the first `.get(k)`. So a shape-only
    stub exercises the guard at realistic Qwen-2.5-7B scale without materializing
    multi-GB tensors. `.get` deliberately raises: the production-grid cells never
    reach it in these tests (the guard passes, then we stop), and the breaching
    cell raises at the guard first.
    """

    def __init__(self, key_to_shape: dict[str, tuple[int, ...]]):
        self._key_to_shape = key_to_shape

    def shape(self, key: str) -> tuple[int, ...]:
        return self._key_to_shape[key]

    def tensor_keys(self) -> list[str]:
        return list(self._key_to_shape)

    def get(self, key: str):  # pragma: no cover - not reached in guard tests
        raise AssertionError(
            f"get({key!r}) reached — the budget-guard test should stop at the "
            "shape-only guard, never materialize a tensor"
        )


def _down_proj_shapes() -> dict[str, tuple[int, int]]:
    """28 down_proj matrices [H, I] — the `down_proj/global` RETAINED scope."""
    return {
        f"model.layers.{i}.mlp.down_proj.weight": (_QWEN_H, _QWEN_I) for i in range(_QWEN_LAYERS)
    }


def _all_linear_shapes() -> dict[str, tuple[int, int]]:
    """All 7 linear projections per layer — the dropped `all_linear/global` scope.

    Per layer: q/o_proj [H,H], k/v_proj [H_kv,H] (GQA: 4 KV heads * 128 = 512),
    gate/up_proj [I,H], down_proj [H,I]. Total ~= 6.5e9 elements over 28 layers
    (the §9 all_linear/global figure ~6.525e9).
    """
    h, i, kv = _QWEN_H, _QWEN_I, 512
    shapes: dict[str, tuple[int, int]] = {}
    for layer in range(_QWEN_LAYERS):
        p = f"model.layers.{layer}"
        shapes[f"{p}.self_attn.q_proj.weight"] = (h, h)
        shapes[f"{p}.self_attn.k_proj.weight"] = (kv, h)
        shapes[f"{p}.self_attn.v_proj.weight"] = (kv, h)
        shapes[f"{p}.self_attn.o_proj.weight"] = (h, h)
        shapes[f"{p}.mlp.gate_proj.weight"] = (i, h)
        shapes[f"{p}.mlp.up_proj.weight"] = (i, h)
        shapes[f"{p}.mlp.down_proj.weight"] = (h, i)
    return shapes


def test_budget_guard_passes_on_retained_down_proj_global_cell():
    """`down_proj/global` at the worst K=0.32 is UNDER the §9 15 GB budget — no raise.

    §9 arithmetic: ~1.90e9 down_proj elements; n_zero ~= 6.08e8 -> int64 indices
    4.87 GB + fp32 values 2.43 GB + largest matrix (6.79e7 * 4 B = 0.27 GB)
    ~= 7.6 GB peak (the §9 ~12.4 GB figure also counts running_top/merged, which
    the guard's int64+values+largest-matrix estimate conservatively omits) — well
    under 15 GB. The guard must NOT fire; the streaming loop then starts, so we
    stop it cleanly via the get()-raises stub and confirm we got past the guard.
    """
    p4 = _load_p4_module()
    shapes = _down_proj_shapes()
    base_w = _ShapeOnlyWeights(shapes)
    ft_w = _ShapeOnlyWeights(shapes)
    target_keys = list(shapes)
    total = _QWEN_H * _QWEN_I * _QWEN_LAYERS
    n_zero = round(0.32 * total)
    largest = _QWEN_H * _QWEN_I
    peak = n_zero * (8 + 4) + largest * 4
    assert peak < p4.GLOBAL_TOPK_BUDGET_BYTES, (
        "test premise: down_proj/global must be under budget by §9 arithmetic"
    )
    # The guard must pass; the loop then calls get() → our stub raises AssertionError.
    # Reaching the AssertionError proves the RuntimeError budget guard did NOT fire.
    with pytest.raises(AssertionError, match=r"get.*reached"):
        p4._global_topk_threshold(base_w, ft_w, 0.32, target_keys)


def test_budget_guard_raises_on_dropped_all_linear_global_cell():
    """`all_linear/global` at K=0.32 BREACHES the §9 budget → RuntimeError named.

    §9 arithmetic: ~6.525e9 all-linear elements; n_zero ≈ 2.088e9 → int64 indices
    15.56 GB ALONE + values 7.78 GB ≈ ~23 GB by the guard estimate (~39 GB true
    co-resident peak), > 15 GB. The guard fires at the TOP, before any tensor read.
    """
    p4 = _load_p4_module()
    shapes = _all_linear_shapes()
    base_w = _ShapeOnlyWeights(shapes)
    ft_w = _ShapeOnlyWeights(shapes)
    target_keys = list(shapes)
    total = sum(a * b for (a, b) in shapes.values())
    # Confirm the test premise: ~6.5e9 all-linear elements (§9 ~6.525e9).
    assert 6.0e9 < total < 7.0e9, f"all_linear element count off: {total}"

    with pytest.raises(RuntimeError, match=r"all_linear/global"):
        p4._global_topk_threshold(base_w, ft_w, 0.32, target_keys)
    # The message must name the breaching cell + the §11 pointer (fail-loud contract).
    try:
        p4._global_topk_threshold(base_w, ft_w, 0.32, target_keys)
    except RuntimeError as e:
        msg = str(e)
        assert "scope=all_linear" in msg and "granularity=global" in msg, msg
        assert "§9" in msg and "§11" in msg, msg
        assert "P4_SCOPE_GRANULARITY_GRID" in msg, msg


def test_budget_guard_uses_synthetic_budget_override():
    """The guard honors a tiny `budget_bytes` override → raises on a toy `total`.

    Pins the guard arithmetic directly (no Qwen scale): a 1 GB budget against a
    modest synthetic scope must fail loud, and a generous budget on the same scope
    must pass (reach the get() stub).
    """
    p4 = _load_p4_module()
    # 100 matrices of 1e6 elements = 1e8 total; n_zero at K=0.32 = 3.2e7.
    shapes = {f"m{i}": (1000, 1000) for i in range(100)}
    base_w = _ShapeOnlyWeights(shapes)
    ft_w = _ShapeOnlyWeights(shapes)
    target_keys = list(shapes)
    total = 100 * 1_000_000
    n_zero = round(0.32 * total)  # 3.2e7
    largest = 1_000_000
    peak = n_zero * (8 + 4) + largest * 4  # ~0.39 GB

    # A budget BELOW the peak → raise.
    with pytest.raises(RuntimeError, match=r"exceeds the §9 budget"):
        p4._global_topk_threshold(base_w, ft_w, 0.32, target_keys, budget_bytes=peak - 1)
    # A budget ABOVE the peak → pass the guard, then hit the get() stub.
    with pytest.raises(AssertionError, match=r"get.*reached"):
        p4._global_topk_threshold(base_w, ft_w, 0.32, target_keys, budget_bytes=peak + 1)
