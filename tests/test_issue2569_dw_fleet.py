"""Unit tests for scripts/issue2569_dw_fleet.py — leg-5 dW geometry (plan #2569 §4 leg 5).

Covers: LoRA scaling-regime reading (rsLoRA alpha/sqrt(r) vs classic alpha/r — artifact-reuse
check (g)), dW reconstruction, the EXACT rank-r factored SVD vs the dense oracle (fix-round-2
blocker ``dwfleet-lora-dense-svd-vs-rank32``), effective-rank summaries on a known spectrum,
batched-vs-serial svdvals equality, the dv3 nested-schema builder whose null-symmetry flag is
COMPUTED (``dwfleet-null-assertion-vacuous``), the max-matched alignment read, full-FT
intruder + factor outputs (``dwfleet-fullft-analysis-missing``), the all-7-module intruder
requirement (``dwfleet-oproj-intruder-silently-dropped``), the staged + pinned banked
directions with explicit absence (``dwfleet-delta-tbar-silent-absence``), the residual-side
basis routing (``dwfleet-cc-alignment-mismatched-basis``), the FT-pinned pilot
(``dwfleet-pilot-not-fullft-plan-adherence``), fleet enumeration fail-loud behavior, and the
in-place ``--modules`` extension of ``issue650_analyze.py``. Banked-payload fixtures mirror
the REAL schemas probed 2026-08-25 (tbar/anchor/r_B — see the driver docstrings). All
synthetic, CPU-fast.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue650_analyze as I650  # noqa: E402
import issue2569_dw_fleet as DW  # noqa: E402

# Tiny decoder dims shared by every fixture (d_model=6, d_ff=10 — matches _TinyModel).
_D_MODEL = 6
_D_FF = 10
_SHAPES = {
    "q_proj": (_D_MODEL, _D_MODEL),
    "k_proj": (_D_MODEL, _D_MODEL),
    "v_proj": (_D_MODEL, _D_MODEL),
    "o_proj": (_D_MODEL, _D_MODEL),
    "gate_proj": (_D_FF, _D_MODEL),
    "up_proj": (_D_FF, _D_MODEL),
    "down_proj": (_D_MODEL, _D_FF),
}
_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


# ──────────────────────────────────────────────────────────────────────────
# Shared fixtures: tiny fake model / adapter / FT checkpoint pair / banked payloads
# ──────────────────────────────────────────────────────────────────────────


class _W:
    def __init__(self, shape, seed):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(*shape, generator=g)


class _Layer:
    def __init__(self, seed):
        d, f = _D_MODEL, _D_FF
        self.self_attn = type(
            "SA",
            (),
            {
                "q_proj": _W((d, d), seed),
                "k_proj": _W((d, d), seed + 1),
                "v_proj": _W((d, d), seed + 2),
                "o_proj": _W((d, d), seed + 3),
            },
        )()
        self.mlp = type(
            "MLP",
            (),
            {
                "gate_proj": _W((f, d), seed + 4),
                "up_proj": _W((f, d), seed + 5),
                "down_proj": _W((d, f), seed + 6),
            },
        )()


class _TinyModel:
    model = type("M", (), {"layers": [_Layer(0), _Layer(100)]})()


def _build_tiny_base_svd(tmp_path: Path, modules=DW.LORA_MODULES) -> Path:
    """Build a base-svd payload FILE on the tiny fake model via issue650's REAL main()."""
    import unittest.mock as mock

    fake_auto = mock.MagicMock()
    fake_auto.from_pretrained.return_value = _TinyModel()
    out_path = tmp_path / f"base_svd_{len(modules)}.pt"
    with mock.patch("transformers.AutoModelForCausalLM", fake_auto):
        rc = I650.main(
            [
                "build-base-svd",
                "--base-model",
                "fake",
                "--base-svd",
                str(out_path),
                "--modules",
                ",".join(modules),
            ]
        )
    assert rc == 0 and out_path.is_file()
    return out_path


def _write_adapter(dirpath: Path, *, r=2, layers=(0, 1), modules=DW.LORA_MODULES, seed=0):
    """Write a synthetic PEFT adapter dir (config + safetensors) at the tiny dims."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in layers:
        for m in modules:
            d_out, d_in = _SHAPES[m]
            grp = "self_attn" if m in _ATTN else "mlp"
            prefix = f"base_model.model.model.layers.{layer}.{grp}.{m}"
            tensors[f"{prefix}.lora_A.weight"] = torch.randn(r, d_in, generator=g)
            tensors[f"{prefix}.lora_B.weight"] = torch.randn(d_out, r, generator=g)
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": 4, "use_rslora": True})
    )
    save_file(tensors, str(dirpath / "adapter_model.safetensors"))
    return dirpath


def _write_ft_pair(tmp_path: Path, *, layers=(0, 1), seed=7):
    """Write base/post HF-format checkpoint dirs (single safetensors file each).

    Includes embed_tokens / lm_head / biases / norms so the decoder-matrix filter is
    actually exercised (embed + lm_head are 2-D and MUST be excluded).
    """
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    base = {
        "model.embed_tokens.weight": torch.randn(8, _D_MODEL, generator=g),
        "lm_head.weight": torch.randn(8, _D_MODEL, generator=g),
    }
    for layer in layers:
        base[f"model.layers.{layer}.input_layernorm.weight"] = torch.randn(_D_MODEL, generator=g)
        base[f"model.layers.{layer}.self_attn.q_proj.bias"] = torch.randn(_D_MODEL, generator=g)
        for m in DW.LORA_MODULES:
            d_out, d_in = _SHAPES[m]
            grp = "self_attn" if m in _ATTN else "mlp"
            base[f"model.layers.{layer}.{grp}.{m}.weight"] = torch.randn(d_out, d_in, generator=g)
    post = {}
    for kname, t in base.items():
        if t.ndim == 2:
            bump = torch.outer(
                torch.randn(t.shape[0], generator=g), torch.randn(t.shape[1], generator=g)
            )
            post[kname] = t + 0.05 * bump
        else:
            post[kname] = t + 0.01
    bdir = tmp_path / "base_ckpt"
    pdir = tmp_path / "post_ckpt"
    bdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)
    save_file(base, str(bdir / "model.safetensors"))
    save_file(post, str(pdir / "model.safetensors"))
    return bdir, pdir


def _tbar_payload(layer=0, seed=1, d=_D_MODEL):
    """Mirror of the PROBED tbar.pt schema (delta_tf @ c07267285d)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d, generator=g)
    return {
        "tbar": {layer: base},
        "tbar_even": {layer: base + 0.05 * torch.randn(d, generator=g)},
        "tbar_odd": {layer: base - 0.05 * torch.randn(d, generator=g)},
        "n_rows": 20,
        "meta": {"issue": 1768},
    }


def _anchor_payload(layer=0, seed=2, d=_D_MODEL):
    """Mirror of the PROBED anchor .pt schema (issue1900_leakrace/anchors @ b5acdabc79)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "mix_arm_id": "x",
        "n_rows": 20,
        "low_n_flag": True,
        "A_ctx": {layer: torch.randn(d, generator=g)},
        "A_ans": {layer: torch.randn(d, generator=g)},
        "split_half_cos_ctx": {layer: 0.98},
        "meta": {},
    }


def _rb_payload(layers=(0, 1), d=_D_MODEL, seed=3, trait="evil"):
    """Mirror of the PROBED r_B .pt schema (issue779_monitoring/r_b @ 037fcbb2)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "trait": trait,
        "r_b": torch.randn(len(layers), d, generator=g),
        "layers": list(layers),
        "metadata": {},
    }


def _fake_operator_module(map_file: Path, d=_D_MODEL):
    """sys.modules stand-in for issue2569_operator (cmd_align's B1 entry asserts).

    ``map_file`` mirrors the real ``MapPayload.path`` field: cmd_align fingerprints
    the banked map's FILE BYTES into its regime key (fix-round-3 concern
    ``dwfleet-align-resume-key-omits-banked-map-fingerprint``), so the fake payload
    must carry a real on-disk path.
    """
    mod = types.ModuleType("issue2569_operator")
    a_mat = np.eye(d)

    mod.load_banked_map = lambda layer=19, root=None: types.SimpleNamespace(
        layer=layer, path=map_file
    )
    mod.run_driver_identity_asserts = lambda payload: None
    mod.row_operator = lambda payload: (a_mat, np.zeros(d))
    mod.monitor_gradient = lambda A, r: np.asarray(A) @ np.asarray(r)
    return mod


def _args(**kw):
    base = dict(
        out_root=None,
        dl_root=None,
        align_layer="0",
        arms=None,
        no_resume=False,
        base_svd=None,
        base_ckpt=None,
        map_root=None,
        pilot_wall_cap_h="8.0",
        extra_arms_json=None,
        phase=None,
        import_check=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _lora_entry(arm_id="arm1", subfolder="sub"):
    return DW.FleetEntry(arm_id, "content", "cas", "lora", "org/model", subfolder, "arms.json")


def _ft_entry(arm_id="ftarm", subfolder="fsub"):
    return DW.FleetEntry(arm_id, "content", "cas", "ft", "org/overflow", subfolder, "arms.json")


def _write_fleet_table(out_root: Path, entries):
    import dataclasses

    DW._atomic_json(
        out_root / "dw_fleet" / "fleet_table.json",
        {
            "fleet": [dataclasses.asdict(e) for e in entries],
            "n_lora": sum(1 for e in entries if e.method == "lora"),
            "n_ft": sum(1 for e in entries if e.method == "ft"),
            "metadata": {},
        },
    )


# ──────────────────────────────────────────────────────────────────────────
# dW construction
# ──────────────────────────────────────────────────────────────────────────


def test_lora_scaling_regimes():
    """rsLoRA scales alpha/sqrt(r); classic scales alpha/r; r<=0 fails loud (check (g))."""
    assert DW.lora_scaling({"lora_alpha": 64, "r": 32, "use_rslora": True}) == pytest.approx(
        64 / 32**0.5
    )
    assert DW.lora_scaling({"lora_alpha": 64, "r": 32, "use_rslora": False}) == pytest.approx(2.0)
    assert DW.lora_scaling({"lora_alpha": 64, "r": 32}) == pytest.approx(2.0)  # absent = classic
    with pytest.raises(ValueError):
        DW.lora_scaling({"lora_alpha": 64, "r": 0})


def test_delta_w_from_lora_shape_and_value():
    """dW = B @ A * s with A (r, d_in), B (d_out, r); shape mismatch asserts."""
    a = torch.ones(2, 5)
    b = torch.ones(3, 2)
    dw = DW.delta_w_from_lora(a, b, 0.5)
    assert dw.shape == (3, 5)
    torch.testing.assert_close(dw, torch.full((3, 5), 1.0))
    with pytest.raises(AssertionError):
        DW.delta_w_from_lora(torch.ones(2, 5), torch.ones(3, 4), 1.0)


def test_lora_svd_factors_matches_dense_exactly():
    """The r x r core-after-QR SVD is EXACT vs the dense oracle (blocker
    dwfleet-lora-dense-svd-vs-rank32): singular values match the dense top-r, the
    reconstruction reproduces dW, top vectors agree up to sign — single AND batched."""
    g = torch.Generator().manual_seed(11)
    r, d_in, d_out, s = 3, 9, 7, 1.7
    a = torch.randn(r, d_in, generator=g)
    b = torch.randn(d_out, r, generator=g)
    dw = DW.delta_w_from_lora(a, b, s)
    u_d, s_d, vh_d = torch.linalg.svd(dw, full_matrices=False)
    u_f, s_f, vh_f = DW.lora_svd_factors(a, b, s)
    assert s_f.shape == (r,) and u_f.shape == (d_out, r) and vh_f.shape == (r, d_in)
    torch.testing.assert_close(s_f, s_d[:r], rtol=1e-4, atol=1e-6)
    # Dense trailing singular values of the rank-r product are numerically zero.
    assert float(s_d[r:].max()) < 1e-5 * float(s_d[0])
    # Exact reconstruction: U diag(S) Vh == dW.
    torch.testing.assert_close(u_f @ torch.diag(s_f) @ vh_f, dw, rtol=1e-4, atol=1e-5)
    # Top vectors agree up to sign (generic spectrum => distinct svals).
    for i in range(r):
        assert abs(float(u_f[:, i] @ u_d[:, i])) > 0.999
        assert abs(float(vh_f[i] @ vh_d[i])) > 0.999
    # Batched: a (L, r, d_in) / (L, d_out, r) stack matches per-slice dense.
    a_st = torch.randn(4, r, d_in, generator=g)
    b_st = torch.randn(4, d_out, r, generator=g)
    u_b, s_b, vh_b = DW.lora_svd_factors(a_st, b_st, s)
    assert s_b.shape == (4, r)
    for i in range(4):
        dw_i = DW.delta_w_from_lora(a_st[i], b_st[i], s)
        torch.testing.assert_close(
            u_b[i] @ torch.diag(s_b[i]) @ vh_b[i], dw_i, rtol=1e-4, atol=1e-5
        )


def test_effective_rank_summaries_known_spectrum():
    """Exact values on a hand-computable spectrum; all-zero spectrum fails loud."""
    s = np.array([2.0, 1.0, 1.0])
    rec = DW.effective_rank_summaries(s)
    assert rec["stable_rank"] == pytest.approx(6.0 / 4.0)
    assert rec["participation_ratio"] == pytest.approx(36.0 / 18.0)
    assert rec["top1_share_energy"] == pytest.approx(4.0 / 6.0)
    assert rec["top1_share_sv"] == pytest.approx(2.0 / 4.0)
    assert rec["frobenius"] == pytest.approx(6.0**0.5)
    assert rec["spectral"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        DW.effective_rank_summaries(np.zeros(3))


def test_factor_separation_ties_null_space_and_bounds():
    """Per-factor separation labels (fix-round-3
    dwfleet-degenerate-svd-vectors-scored-as-signal): distinct spectra label
    well-separated; a near-tied pair flags BOTH members; a numerically-null factor
    flags; a soft top-k boundary flags boundary_tied; out-of-range k / a zero
    spectrum raise (never a silent default)."""
    sep = DW.factor_separation(np.array([1.0, 0.8, 0.5, 0.2]), 3)
    assert sep["well_separated"] == [True, True, True]
    assert sep["rel_gap_next"] == pytest.approx([0.2, 0.3, 0.3])
    assert sep["rel_gap_prev"][0] is None
    assert sep["rel_gap_prev"][1] == pytest.approx(0.2)
    assert sep["boundary_tied"] is False and sep["n_well_separated"] == 3
    assert sep["rel_gap_floor"] == DW.DEGEN_REL_GAP_FLOOR
    # Near-tie at the probe-observed scale (real minima 5.8e-4 of s0): both members
    # of the tied pair flagged; neighbors unaffected.
    sep2 = DW.factor_separation(np.array([1.0, 0.5, 0.4995, 0.2]), 4)
    assert sep2["well_separated"] == [True, False, False, True]
    # Numerically-null factor (sv floor): the exact-zero tail of a rank-limited
    # spectrum is degenerate even as the "last computed" value.
    sep3 = DW.factor_separation(np.array([1.0, 5e-7]), 2)
    assert sep3["well_separated"][1] is False
    # Soft top-k BOUNDARY: the k-th gap tied => subspace-level reads unstable too.
    sep4 = DW.factor_separation(np.array([1.0, 0.5, 0.4999]), 2)
    assert sep4["boundary_tied"] is True and sep4["well_separated"][1] is False
    with pytest.raises(ValueError):
        DW.factor_separation(np.array([1.0, 0.5]), 3)
    with pytest.raises(ValueError):
        DW.factor_separation(np.array([1.0, 0.5]), 0)
    with pytest.raises(ValueError):
        DW.factor_separation(np.zeros(4), 2)
    top = DW.top_vector_separation(np.array([1.0, 0.9995]))
    assert top["well_separated"] is False and top["rel_gap_01"] == pytest.approx(5e-4)


def test_svdvals_stack_matches_serial():
    """The batched dense reference equals per-matrix svdvals (oracle self-consistency)."""
    rng = np.random.default_rng(0)
    stack = torch.tensor(rng.normal(size=(5, 10, 7)), dtype=torch.float32)
    batched = DW.svdvals_stack(stack)
    serial = np.stack([DW.svdvals_robust(stack[i]) for i in range(5)])
    np.testing.assert_allclose(batched, serial, rtol=1e-5, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────
# Intruder read + alignment (the load-enforced #650 convention)
# ──────────────────────────────────────────────────────────────────────────


def test_dv3_payload_passes_load_enforced_schema():
    """The nested payload builder passes issue650's assert_dv3_schema; a wrong aggregation
    string is REJECTED before anything is persisted (never bypassed)."""
    rng = np.random.default_rng(1)
    basis = {5: np.linalg.qr(rng.normal(size=(8, 8)))[0].T.astype(np.float32)}
    res = I650.dv3_max_matched_null(
        observed_vec=rng.normal(size=8).astype(np.float32),
        basis_by_layer=basis,
        band=(5,),
        n_draws=50,
        seed=0,
    )
    payload = DW.dv3_payload_from_null({"write": res})
    I650.assert_dv3_schema(payload)  # load-time validation passes
    assert payload["assertions"]["null_aggregation_matches_observed"] is True
    bad = dict(res)
    bad["null_aggregation"] = "flat_p95_single_cosines"
    with pytest.raises(AssertionError):
        DW.dv3_payload_from_null({"write": bad})


def test_dv3_null_symmetry_flag_is_computed_not_claimed():
    """The assertions flag is a COMPUTED verification (blocker dwfleet-null-assertion-vacuous):
    a payload whose band aggregation no longer equals the per-layer band-max — observed OR
    null side — RAISES instead of shipping a hardcoded True."""
    rng = np.random.default_rng(9)
    basis = {5: np.linalg.qr(rng.normal(size=(8, 8)))[0].T.astype(np.float32)}
    res = I650.dv3_max_matched_null(
        observed_vec=rng.normal(size=8).astype(np.float32),
        basis_by_layer=basis,
        band=(5,),
        n_draws=50,
        seed=0,
    )
    # Observed-side break: band max no longer equals max over per-layer observed.
    bad_obs = dict(res)
    bad_obs["band_observed_max"] = float(res["band_observed_max"]) + 0.1
    with pytest.raises(AssertionError, match="NOT the registered band-max"):
        DW.dv3_payload_from_null({"write": bad_obs})
    # Null-side break: draws re-aggregated by something other than the band max.
    bad_null = dict(res)
    bad_null["band_null_max_draws"] = [0.5 * x for x in res["band_null_max_draws"]]
    with pytest.raises(AssertionError, match="null aggregation is NOT"):
        DW.dv3_payload_from_null({"write": bad_null})
    # p95 break: the quoted p95 is not the p95 of the band-max draws.
    bad_p95 = dict(res)
    bad_p95["band_null_p95"] = float(res["band_null_p95"]) + 0.2
    with pytest.raises(AssertionError, match="recomputed p95"):
        DW.dv3_payload_from_null({"write": bad_p95})


def test_dv3_payload_from_null_refuses_empty_and_unknown_arms():
    """An EMPTY arm roster (or an unregistered arm name) RAISES at the producer — with
    zero arms the per-arm guards never run and the True flag would vacuously certify
    aggregation symmetry over nothing (fix-round-3 blocker
    dv3-empty-arm-roster-yields-vacuous-true)."""
    with pytest.raises(AssertionError, match="EMPTY arm roster"):
        DW.dv3_payload_from_null({})
    rng = np.random.default_rng(6)
    basis = {5: np.linalg.qr(rng.normal(size=(8, 8)))[0].T.astype(np.float32)}
    res = I650.dv3_max_matched_null(
        observed_vec=rng.normal(size=8).astype(np.float32),
        basis_by_layer=basis,
        band=(5,),
        n_draws=20,
        seed=0,
    )
    with pytest.raises(AssertionError, match="unknown arm"):
        DW.dv3_payload_from_null({"wrote": res})


def test_assert_dv3_schema_arm_roster_floor_and_real_cells():
    """assert_dv3_schema rejects a ZERO-arm payload (the vacuous-True shape) and unknown
    arm keys, KEEPS the single-arm tolerance (the dw_fleet per-module intruder shape is
    one arm — 'write' U-side / 'read' V-side), and every cell of the REAL committed
    #650 artifact (both arms, 12/12) still passes the strengthened assert."""
    vac = {
        "observed": {},
        "null": {},
        "assertions": {"null_aggregation_matches_observed": True},
    }
    with pytest.raises(AssertionError, match="ZERO of the registered arms"):
        I650.assert_dv3_schema(vac)
    mixed = {
        "observed": {"write": {"max_by_layer": {}}, "extra": {}},
        "null": {},
        "assertions": {"null_aggregation_matches_observed": True},
    }
    with pytest.raises(AssertionError, match="unknown arm"):
        I650.assert_dv3_schema(mixed)
    # Single-arm ('read') payload stays legal — the V-side module shape.
    rng = np.random.default_rng(7)
    basis = {2: np.linalg.qr(rng.normal(size=(8, 8)))[0].T.astype(np.float32)}
    res = I650.dv3_max_matched_null(
        observed_vec=rng.normal(size=8).astype(np.float32),
        basis_by_layer=basis,
        band=(2,),
        n_draws=20,
        seed=1,
    )
    I650.assert_dv3_schema(DW.dv3_payload_from_null({"read": res}))
    # Real committed artifact (probed 2026-08-26: dict of 12 cells, arms {read, write}).
    real = REPO_ROOT / "eval_results" / "issue_650" / "analysis" / "dv3_intruder.json"
    cells = json.loads(real.read_text())["cells"]
    assert len(cells) == 12
    for cell in cells.values():
        I650.assert_dv3_schema(cell)


def test_intruder_read_planted_vs_random():
    """A dW top vector INSIDE the base column space reads pre-existing; an orthogonal
    intruder direction reads intruder-at-null. Aggregation is the registered band-max."""
    rng = np.random.default_rng(2)
    d = 24
    q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    base_basis = {3: q[:, :8].T.astype(np.float32)}  # 8 base singular vectors as rows
    planted = DW.intruder_read({3: q[:, 0]}, base_basis, arm_name="write", n_draws=100, seed=3)
    I650.assert_dv3_schema(planted)
    assert planted["observed"]["write"]["verdict"] == "pre_existing_in_base_column_space"
    intruder = DW.intruder_read({3: q[:, 20]}, base_basis, arm_name="write", n_draws=100, seed=3)
    assert intruder["observed"]["write"]["verdict"] == "intruder_at_max_matched_null"


def test_alignment_vs_null_planted_and_random():
    """A direction equal to a top factor clears the max-matched null; random does not."""
    rng = np.random.default_rng(4)
    d, k = 32, 8
    factors = np.linalg.qr(rng.normal(size=(d, k)))[0].T  # (k, d) unit rows
    hit = DW.alignment_vs_null(factors, factors[0], n_draws=200, seed=5)
    assert hit["above_null"] and hit["max_abs_cos"] == pytest.approx(1.0)
    assert hit["null_aggregation"] == DW.DV3_NULL_AGGREGATION
    # A direction orthogonal to every factor scores 0 — below any null p95.
    ortho = np.linalg.qr(rng.normal(size=(d, k + 1)))[0][:, k]
    ortho -= factors.T @ (factors @ ortho)
    miss = DW.alignment_vs_null(factors, ortho / np.linalg.norm(ortho), n_draws=200, seed=5)
    assert not miss["above_null"]


# ──────────────────────────────────────────────────────────────────────────
# LoRA battery: exact rank-r path + all-module intruder
# ──────────────────────────────────────────────────────────────────────────


def test_analyze_lora_arm_rank_r_path_and_all_module_intruder(tmp_path):
    """Production LoRA analysis is the EXACT rank-r path (n_svals == r, never min(m, n)) and
    the intruder read covers EVERY module incl. o_proj on its residual side (blockers
    dwfleet-lora-dense-svd-vs-rank32 + dwfleet-oproj-intruder-silently-dropped)."""
    r = 2
    adapter = _write_adapter(tmp_path / "adapter", r=r)
    base_svd = I650.load_base_svd(_build_tiny_base_svd(tmp_path), modules=DW.LORA_MODULES)
    rec = DW.analyze_lora_arm(_lora_entry(), adapter, base_svd)
    assert set(rec["modules"]) == set(DW.LORA_MODULES)
    for module, by_layer in rec["modules"].items():
        for layer, summ in by_layer.items():
            assert summ["n_svals"] == r, (module, layer, summ["n_svals"])
    # Intruder: ALL 7 modules present, correct residual sides, schema-valid payloads.
    assert set(rec["intruder"]) == set(DW.LORA_MODULES)
    assert rec["intruder_side"]["o_proj"] == "U"
    assert rec["intruder_side"]["down_proj"] == "U"
    assert rec["intruder_side"]["q_proj"] == "V"
    assert rec["intruder_side"]["up_proj"] == "V"
    for module, payload in rec["intruder"].items():
        I650.assert_dv3_schema(payload)
        arm = "write" if rec["intruder_side"][module] == "U" else "read"
        assert arm in payload["observed"]
    # Factored summaries numerically agree with the dense oracle (trailing zeros inert).
    deltas = DW.load_adapter_deltas(adapter)
    dense = DW.effective_rank_summaries(DW.svdvals_robust(deltas[(0, "down_proj")]))
    fact = rec["modules"]["down_proj"]["0"]
    for key in ("stable_rank", "participation_ratio", "top1_share_energy", "frobenius"):
        assert fact[key] == pytest.approx(dense[key], rel=1e-4), key


def test_analyze_lora_arm_labels_top_vector_separation(tmp_path):
    """Every (module, layer) top vector carries a separation label computed from the
    EXACT rank-r spectrum, and an EXACTLY-TIED top pair is flagged degenerate —
    self-labeled, never silently scored (fix-round-3
    dwfleet-degenerate-svd-vectors-scored-as-signal)."""
    from safetensors.torch import save_file

    adapter = _write_adapter(tmp_path / "adapter", r=2)
    base_svd = I650.load_base_svd(_build_tiny_base_svd(tmp_path), modules=DW.LORA_MODULES)
    rec = DW.analyze_lora_arm(_lora_entry(), adapter, base_svd)
    assert set(rec["intruder_top_separation"]) == set(DW.LORA_MODULES)
    # Label agrees with a recompute from the dense oracle's spectrum.
    deltas = DW.load_adapter_deltas(adapter)
    sv = DW.svdvals_robust(deltas[(0, "q_proj")])
    expect = DW.top_vector_separation(sv[:2])  # rank-2: two nonzero svals
    got = rec["intruder_top_separation"]["q_proj"]["0"]
    assert got["rel_gap_01"] == pytest.approx(expect["rel_gap_01"], rel=1e-4, abs=1e-7)
    assert got["well_separated"] == expect["well_separated"]
    # Constructed EXACT top tie on q_proj (dW = s * (E00 + E11): equal svals).
    g = torch.Generator().manual_seed(5)
    tensors = {}
    for layer in (0, 1):
        for m in DW.LORA_MODULES:
            d_out, d_in = _SHAPES[m]
            grp = "self_attn" if m in _ATTN else "mlp"
            prefix = f"base_model.model.model.layers.{layer}.{grp}.{m}"
            if m == "q_proj":
                a = torch.zeros(2, d_in)
                a[0, 0] = 1.0
                a[1, 1] = 1.0
                b = torch.zeros(d_out, 2)
                b[0, 0] = 1.0
                b[1, 1] = 1.0
            else:
                a = torch.randn(2, d_in, generator=g)
                b = torch.randn(d_out, 2, generator=g)
            tensors[f"{prefix}.lora_A.weight"] = a
            tensors[f"{prefix}.lora_B.weight"] = b
    tie_dir = tmp_path / "tie_adapter"
    tie_dir.mkdir(parents=True, exist_ok=True)
    (tie_dir / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "use_rslora": True})
    )
    save_file(tensors, str(tie_dir / "adapter_model.safetensors"))
    tie_rec = DW.analyze_lora_arm(_lora_entry("tied"), tie_dir, base_svd)
    tie_top = tie_rec["intruder_top_separation"]["q_proj"]["0"]
    assert tie_top["well_separated"] is False
    assert tie_top["rel_gap_01"] < DW.DEGEN_REL_GAP_FLOOR


def test_load_base_svd_required_fails_loud_on_subset_payload(tmp_path):
    """A base-svd payload built WITHOUT the full module list raises with the rebuild
    command — never a silent module drop (dwfleet-oproj-intruder-silently-dropped)."""
    subset_path = _build_tiny_base_svd(tmp_path, modules=("up_proj", "down_proj"))
    with pytest.raises(RuntimeError, match="lacks module"):
        DW._load_base_svd_required(str(subset_path))
    with pytest.raises(RuntimeError, match="--base-svd is required"):
        DW._load_base_svd_required(None)
    with pytest.raises(FileNotFoundError):
        DW._load_base_svd_required(str(tmp_path / "nope.pt"))


def test_cmd_lora_resume_key_is_content_keyed(tmp_path):
    """cmd_lora's resume key carries the base-svd payload's _meta (content), so a payload
    rebuilt differently NEVER resume-skips stale units (the bool(base_svd) trap)."""
    out_root = tmp_path / "out"
    dl_root = tmp_path / "dl"
    entry = _lora_entry()
    _write_fleet_table(out_root, [entry])
    _write_adapter(dl_root / "adapters" / entry.arm_id / entry.subfolder, r=2)
    svd_path = _build_tiny_base_svd(tmp_path)
    args = _args(out_root=str(out_root), dl_root=str(dl_root), base_svd=str(svd_path))
    assert DW.cmd_lora(args) == 0
    unit = json.loads((out_root / "dw_fleet" / "lora" / "arm1.json").read_text())
    rk1 = unit["regime_key"]
    assert set(unit["intruder"]) == set(DW.LORA_MODULES)
    # Rebuild the payload with a DIFFERENT generating recipe (base model string) — the
    # content-describing _meta changes, so the key changes and the unit recomputes.
    import unittest.mock as mock

    fake_auto = mock.MagicMock()
    fake_auto.from_pretrained.return_value = _TinyModel()
    svd2 = tmp_path / "base_svd_v2.pt"
    with mock.patch("transformers.AutoModelForCausalLM", fake_auto):
        assert (
            I650.main(
                [
                    "build-base-svd",
                    "--base-model",
                    "fake-v2",
                    "--base-svd",
                    str(svd2),
                    "--modules",
                    ",".join(DW.LORA_MODULES),
                ]
            )
            == 0
        )
    args2 = _args(out_root=str(out_root), dl_root=str(dl_root), base_svd=str(svd2))
    assert DW.cmd_lora(args2) == 0
    rk2 = json.loads((out_root / "dw_fleet" / "lora" / "arm1.json").read_text())["regime_key"]
    assert rk1 != rk2


# ──────────────────────────────────────────────────────────────────────────
# Full-FT battery: decoder-matrix filter + intruder + align-layer factors
# ──────────────────────────────────────────────────────────────────────────


def test_ft_name_parts_decoder_matrices_only():
    """The FT param regex matches decoder weight matrices only (196/ckpt class)."""
    assert DW._ft_name_parts("model.layers.17.self_attn.q_proj.weight") == (17, "q_proj")
    assert DW._ft_name_parts("model.layers.3.mlp.down_proj.weight") == (3, "down_proj")
    for bad in (
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.bias",
    ):
        assert DW._ft_name_parts(bad) is None


def test_analyze_ft_checkpoint_full_outputs(tmp_path):
    """Full-FT analysis produces spectra for the DECODER matrices only, an intruder read per
    module, and an align-layer factor sidecar consumable by --phase align (blocker
    dwfleet-fullft-analysis-missing)."""
    bdir, pdir = _write_ft_pair(tmp_path)
    base_svd = I650.load_base_svd(_build_tiny_base_svd(tmp_path), modules=DW.LORA_MODULES)
    factors_path = tmp_path / "ft_factors_L0.pt"
    rec = DW.analyze_ft_checkpoint(
        _ft_entry(), bdir, pdir, base_svd, align_layer=0, factors_path=factors_path
    )
    # 2 layers x 7 modules = 14 decoder matrices; embed/lm_head/bias/norm excluded.
    assert rec["n_matrices"] == 14
    assert all(DW._ft_name_parts(nm) is not None for nm in rec["matrices"])
    assert not any("embed_tokens" in nm or "lm_head" in nm for nm in rec["matrices"])
    # Intruder per module, residual side, schema-valid.
    assert set(rec["intruder"]) == set(DW.LORA_MODULES)
    for payload in rec["intruder"].values():
        I650.assert_dv3_schema(payload)
    assert rec["intruder_side"]["o_proj"] == "U" and rec["intruder_side"]["gate_proj"] == "V"
    # Factor sidecar: all 7 modules at the align layer, residual-side rows of width d_model.
    sidecar = torch.load(factors_path, weights_only=True, map_location="cpu")
    assert int(sidecar["layer"]) == 0
    assert set(sidecar["modules"]) == set(DW.LORA_MODULES)
    for module, blk in sidecar["modules"].items():
        assert blk["side"] == I650.RESIDUAL_SIDE_BY_MODULE[module]
        assert blk["factors"].shape[1] == _D_MODEL  # residual space
    # module_filter narrows the stream (the pilot path) BEFORE any tensor read.
    rec2 = DW.analyze_ft_checkpoint(
        _ft_entry(),
        bdir,
        pdir,
        base_svd,
        align_layer=0,
        factors_path=None,
        module_filter=("down_proj", "q_proj"),
    )
    assert rec2["n_matrices"] == 4 and set(rec2["intruder"]) == {"down_proj", "q_proj"}


def test_analyze_ft_checkpoint_labels_and_exact_svals(tmp_path):
    """FT analysis labels every (module, layer) top vector from the EXACT spectrum and
    the align-layer sidecar persists an exact-sval prefix LONGER than the factor count
    (the k-th boundary gap needs s[k]) — --phase align consumes it for the separation
    labels (fix-round-3 dwfleet-degenerate-svd-vectors-scored-as-signal)."""
    bdir, pdir = _write_ft_pair(tmp_path)
    base_svd = I650.load_base_svd(_build_tiny_base_svd(tmp_path), modules=DW.LORA_MODULES)
    factors_path = tmp_path / "ft_factors_L0.pt"
    rec = DW.analyze_ft_checkpoint(
        _ft_entry(), bdir, pdir, base_svd, align_layer=0, factors_path=factors_path
    )
    assert set(rec["intruder_top_separation"]) == set(DW.LORA_MODULES)
    # Label recomputes from the exact spectrum of the same delta.
    deltas = {DW._ft_name_parts(nm): nm for nm in rec["matrices"] if DW._ft_name_parts(nm)}
    from safetensors.torch import load_file

    base = load_file(str(bdir / "model.safetensors"))
    post = load_file(str(pdir / "model.safetensors"))
    nm = deltas[(0, "q_proj")]
    dw = post[nm].to(torch.float32) - base[nm].to(torch.float32)
    expect = DW.top_vector_separation(DW.svdvals_robust(dw))
    got = rec["intruder_top_separation"]["q_proj"]["0"]
    assert got["rel_gap_01"] == pytest.approx(expect["rel_gap_01"], rel=1e-6)
    assert got["well_separated"] == expect["well_separated"]
    # Sidecar: exact-sval prefix present, descending, strictly longer than needed
    # (tiny dims: the full 6-value spectrum; production: LOWRANK_Q + 1 = 65 > kk = 8).
    sidecar = torch.load(factors_path, weights_only=True, map_location="cpu")
    for module, blk in sidecar["modules"].items():
        sv = blk["svals_exact"].numpy()
        assert sv.shape[0] == min(_SHAPES[module]), module  # full tiny spectrum
        assert sv.shape[0] >= blk["factors"].shape[0]
        assert np.all(np.diff(sv) <= 1e-9), module  # descending
        expect_sv = DW.svdvals_robust(
            post[deltas[(0, module)]].to(torch.float32)
            - base[deltas[(0, module)]].to(torch.float32)
        )
        np.testing.assert_allclose(sv, expect_sv[: sv.shape[0]], rtol=1e-6, atol=1e-8)


# ──────────────────────────────────────────────────────────────────────────
# Banked-direction loaders (fixtures mirror the PROBED real schemas)
# ──────────────────────────────────────────────────────────────────────────


def test_load_rb_direction_probed_schema(tmp_path):
    """r_B loads the layer row of the (n_layers, d) stacked payload; a bare tensor or an
    absent layer fails loud (the pre-fix bare-tensor read crashed on the real file)."""
    p = tmp_path / "evil.pt"
    payload = _rb_payload(layers=(0, 1))
    torch.save(payload, p)
    vec = DW.load_rb_direction(p, 1)
    expect = payload["r_b"][1].to(torch.float64).numpy()
    np.testing.assert_allclose(vec, expect / np.linalg.norm(expect), rtol=1e-12)
    with pytest.raises(KeyError):
        DW.load_rb_direction(p, 5)
    bare = tmp_path / "bare.pt"
    torch.save(torch.randn(6), bare)
    with pytest.raises(TypeError):
        DW.load_rb_direction(bare, 0)


def test_load_tbar_directions_records_20_row_basis(tmp_path):
    """tbar loads delta + even/odd halves and RECORDS the 20-row basis + split-half cosine
    (concern leg5-delta-inherits-tbar-20row-basis)."""
    p = tmp_path / "tbar.pt"
    payload = _tbar_payload(layer=0)
    torch.save(payload, p)
    dirs, prov = DW.load_tbar_directions(p, 0)
    assert set(dirs) == {"delta_tbar", "delta_tbar_even", "delta_tbar_odd"}
    assert prov["n_rows"] == 20 and "20-training-row" in prov["basis_note"]
    even = payload["tbar_even"][0].numpy().astype(np.float64)
    odd = payload["tbar_odd"][0].numpy().astype(np.float64)
    expect_cos = float(np.dot(even / np.linalg.norm(even), odd / np.linalg.norm(odd)))
    assert prov["splithalf_cos"] == pytest.approx(expect_cos)
    with pytest.raises(KeyError):
        DW.load_tbar_directions(p, 19)  # layer absent in fixture


def test_load_anchor_cc_probed_schema(tmp_path):
    """c_C comes from A_ctx[layer] of the probed anchor schema; the pre-fix c_C/centroid key
    guess never matched (dwfleet-anchor-payload-schema-unprobed)."""
    p = tmp_path / "anchor.pt"
    payload = _anchor_payload(layer=0)
    torch.save(payload, p)
    vec, prov = DW.load_anchor_cc(p, 0)
    expect = payload["A_ctx"][0].to(torch.float64).numpy()
    np.testing.assert_allclose(vec, expect / np.linalg.norm(expect), rtol=1e-12)
    assert prov["n_rows"] == 20 and prov["low_n_flag"] is True
    assert prov["splithalf_cos_ctx"] == pytest.approx(0.98)
    # The OLD guessed schema (c_C / centroid keys) is rejected loudly, never silently None.
    old_guess = tmp_path / "old.pt"
    torch.save({"c_C": torch.randn(6), "centroid": torch.randn(6)}, old_guess)
    with pytest.raises(TypeError):
        DW.load_anchor_cc(old_guess, 0)


def test_stage_optional_banked_file_absence_vs_error(tmp_path, monkeypatch):
    """A 404 AT THE PIN returns None (explicit-absence branch); other errors propagate."""
    from huggingface_hub.errors import EntryNotFoundError

    def raise_404(path_in_repo, dl_root, revision, *, what):
        raise EntryNotFoundError("404")

    monkeypatch.setattr(DW, "_stage_banked_file", raise_404)
    assert DW._stage_optional_banked_file("a/b.pt", tmp_path, "rev", what="x") is None

    def raise_other(path_in_repo, dl_root, revision, *, what):
        raise ValueError("boom")

    monkeypatch.setattr(DW, "_stage_banked_file", raise_other)
    with pytest.raises(ValueError):
        DW._stage_optional_banked_file("a/b.pt", tmp_path, "rev", what="x")


# ──────────────────────────────────────────────────────────────────────────
# cmd_align: pinned staging, basis routing, explicit absence, ft consumption
# ──────────────────────────────────────────────────────────────────────────


def _setup_align(tmp_path, monkeypatch, *, banked, entries):
    """Common cmd_align scaffolding: fleet table, adapters, fake operator, staged fakes.

    The fake banked map lives at ``tmp_path / "fake_banked_map.pt"`` — tests that
    exercise the map-bytes regime fingerprint rewrite that file directly.
    """
    out_root = tmp_path / "out"
    dl_root = tmp_path / "dl"
    _write_fleet_table(out_root, entries)
    for e in entries:
        if e.method == "lora":
            _write_adapter(dl_root / "adapters" / e.arm_id / e.subfolder, r=2, layers=(0, 1))
    map_file = tmp_path / "fake_banked_map.pt"
    map_file.write_bytes(b"fake-banked-ridge-v1")
    monkeypatch.setitem(sys.modules, "issue2569_operator", _fake_operator_module(map_file))
    calls: list[tuple[str, str]] = []
    served = tmp_path / "served"

    def fake_stage(path_in_repo, dl_root_arg, revision, *, what):
        calls.append((path_in_repo, revision))
        if path_in_repo not in banked:
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(f"404 {path_in_repo}")
        p = served / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(banked[path_in_repo], p)
        return p

    monkeypatch.setattr(DW, "_stage_banked_file", fake_stage)
    return out_root, dl_root, calls


def _rb_banked():
    return {
        f"{DW.RB_PREFIX}/{trait}.pt": _rb_payload(layers=(0, 1), trait=trait)
        for trait in DW.RB_TRAITS
    }


def test_cmd_align_stages_pins_routes_bases_and_records_absence(tmp_path, monkeypatch):
    """cmd_align stages every banked direction AT ITS PIN (live constants), routes reads by
    RESIDUAL side (U: all directions; V: c_C only; o_proj input side never aligned), and
    records per-arm absence EXPLICITLY (blockers dwfleet-delta-tbar-silent-absence +
    dwfleet-cc-alignment-mismatched-basis + dwfleet-fullft-analysis-missing)."""
    lora = _lora_entry()
    ft = _ft_entry()
    banked = _rb_banked()
    banked[f"{DW.DELTA_TF_PREFIX}/{lora.arm_id}/tbar.pt"] = _tbar_payload(layer=0)
    banked[f"{DW.ANCHORS_PREFIX}/{lora.arm_id}.pt"] = _anchor_payload(layer=0)
    # ft arm: NO tbar / anchor banked (mirrors the probed real state for the 4 FT arms).
    out_root, dl_root, calls = _setup_align(
        tmp_path, monkeypatch, banked=banked, entries=[lora, ft]
    )
    # ft factor sidecar (what --phase ft persists), hand-built at the correct sides/dims.
    g = torch.Generator().manual_seed(21)
    sidecar = {
        "layer": 0,
        "arm_id": ft.arm_id,
        "method": "svd_lowrank-test",
        "modules": {
            m: {
                "side": I650.RESIDUAL_SIDE_BY_MODULE[m],
                "factors": torch.randn(3, _D_MODEL, generator=g),
                "svals": torch.rand(3, generator=g),
                # EXACT spectrum prefix (len > n_factors) — --phase align hard-requires
                # it for the factor_separation labels (fix-round-3).
                "svals_exact": torch.tensor([1.0, 0.6, 0.3, 0.12]),
            }
            for m in DW.LORA_MODULES
        },
    }
    ft_dir = out_root / "dw_fleet" / "ft"
    ft_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sidecar, ft_dir / f"{ft.arm_id}_factors_L0.pt")

    rc = DW.cmd_align(_args(out_root=str(out_root), dl_root=str(dl_root)))
    assert rc == 0

    # The pins are LIVE: every staged fetch carried its registered revision constant.
    revs = dict(calls)
    assert revs[f"{DW.RB_PREFIX}/evil.pt"] == DW.RB_REV
    assert revs[f"{DW.DELTA_TF_PREFIX}/{lora.arm_id}/tbar.pt"] == DW.DELTA_TF_REV
    assert revs[f"{DW.ANCHORS_PREFIX}/{lora.arm_id}.pt"] == DW.ANCHORS_REV

    out = json.loads((out_root / "dw_fleet" / "alignment.json").read_text())
    arm1 = out["arms"][lora.arm_id]
    # U-side modules (o/down) read ALL directions incl. delta halves + c_C + r_B + Ar.
    o_rec = arm1["factors"]["L0.o_proj"]
    assert o_rec["side"] == "U"
    assert {
        "delta_tbar",
        "delta_tbar_even",
        "delta_tbar_odd",
        "c_C",
        "r_B[evil]",
        "Ar[evil]",
    } <= set(o_rec["alignments"])
    # V-side modules (q/k/v/gate/up) read ONLY the context-space direction c_C — the
    # o_proj INPUT (head-concat) read no longer exists anywhere.
    q_rec = arm1["factors"]["L0.q_proj"]
    assert q_rec["side"] == "V"
    assert set(q_rec["alignments"]) == {"c_C"}
    assert arm1["factors"]["L0.down_proj"]["side"] == "U"
    # Provenance: 20-row tbar basis + split-half noise floor + anchor low-n flag.
    assert arm1["directions_provenance"]["delta_tbar"]["n_rows"] == 20
    assert "splithalf_cos" in arm1["directions_provenance"]["delta_tbar"]
    assert arm1["directions_provenance"]["c_C"]["low_n_flag"] is True
    # FT arm: factors consumed from the sidecar; absence EXPLICIT, coverage lists it.
    ftr = out["arms"][ft.arm_id]
    assert "missing" in ftr["directions_provenance"]["delta_tbar"]
    assert "missing" in ftr["directions_provenance"]["c_C"]
    assert {"r_B[evil]", "Ar[evil]"} <= set(ftr["factors"]["L0.o_proj"]["alignments"])
    assert ftr["factors"]["L0.q_proj"]["alignments"] == {}  # c_C missing => nothing to read
    assert out["coverage"]["delta_tbar_missing"] == [ft.arm_id]
    assert out["coverage"]["c_C_missing"] == [ft.arm_id]
    # Per-arm checkpoint units exist.
    assert (out_root / "dw_fleet" / "align" / f"{lora.arm_id}.json").is_file()
    assert (out_root / "dw_fleet" / "align" / f"{ft.arm_id}.json").is_file()


def test_cmd_align_raises_when_all_lora_arms_lose_primary(tmp_path, monkeypatch):
    """ALL LoRA arms missing the banked delta (or c_C) refuses to publish — the silent
    all-vanish scenario of dwfleet-delta-tbar-silent-absence is now a hard failure."""
    lora = _lora_entry()
    banked = _rb_banked()
    banked[f"{DW.ANCHORS_PREFIX}/{lora.arm_id}.pt"] = _anchor_payload(layer=0)
    out_root, dl_root, _ = _setup_align(tmp_path, monkeypatch, banked=banked, entries=[lora])
    with pytest.raises(RuntimeError, match="H5 primary direction lost"):
        DW.cmd_align(_args(out_root=str(out_root), dl_root=str(dl_root)))

    banked2 = _rb_banked()
    banked2[f"{DW.DELTA_TF_PREFIX}/{lora.arm_id}/tbar.pt"] = _tbar_payload(layer=0)
    out_root2, dl_root2, _ = _setup_align(
        tmp_path / "b", monkeypatch, banked=banked2, entries=[lora]
    )
    with pytest.raises(RuntimeError, match="c_C lost"):
        DW.cmd_align(_args(out_root=str(out_root2), dl_root=str(dl_root2)))


def test_cmd_align_dim_mismatch_raises_not_skips(tmp_path, monkeypatch):
    """A direction whose dim does not match the factor stack RAISES (corrupted input) —
    never a silent per-row skip."""
    lora = _lora_entry()
    banked = _rb_banked()
    banked[f"{DW.DELTA_TF_PREFIX}/{lora.arm_id}/tbar.pt"] = _tbar_payload(layer=0, d=5)  # wrong d
    banked[f"{DW.ANCHORS_PREFIX}/{lora.arm_id}.pt"] = _anchor_payload(layer=0)
    out_root, dl_root, _ = _setup_align(tmp_path, monkeypatch, banked=banked, entries=[lora])
    with pytest.raises(RuntimeError, match="dim"):
        DW.cmd_align(_args(out_root=str(out_root), dl_root=str(dl_root)))


def test_cmd_align_labels_degenerate_factors_pins_and_anchor(tmp_path, monkeypatch):
    """alignment.json self-labels factor degeneracy (fix-round-3
    dwfleet-degenerate-svd-vectors-scored-as-signal): every factor block carries
    factor_separation; every read carries argmax_factor / argmax_well_separated and
    the rotation-invariant subspace_proj; FT labels recompute from the sidecar's
    exact svals; the seed-noise anchor labels both arms' top factors; and the pins
    block fingerprints the banked map bytes."""
    import hashlib

    s42 = DW.FleetEntry(
        "imp-pers-con-lr3e5-s42", "content", "imp", "lora", "org/m", "s42", "arms.json"
    )
    s137 = DW.FleetEntry(
        "imp-pers-con-lr3e5-s137", "content", "imp", "lora", "org/m", "s137", "arms.json"
    )
    ft = _ft_entry()
    banked = _rb_banked()
    for aid in (s42.arm_id, s137.arm_id):
        banked[f"{DW.DELTA_TF_PREFIX}/{aid}/tbar.pt"] = _tbar_payload(layer=0)
        banked[f"{DW.ANCHORS_PREFIX}/{aid}.pt"] = _anchor_payload(layer=0)
    out_root, dl_root, _ = _setup_align(
        tmp_path, monkeypatch, banked=banked, entries=[s42, s137, ft]
    )
    g = torch.Generator().manual_seed(23)
    ft_svals_exact = torch.tensor([1.0, 0.6, 0.3, 0.12])
    sidecar = {
        "layer": 0,
        "arm_id": ft.arm_id,
        "method": "svd_lowrank-test",
        "modules": {
            m: {
                "side": I650.RESIDUAL_SIDE_BY_MODULE[m],
                "factors": torch.randn(3, _D_MODEL, generator=g),
                "svals": torch.rand(3, generator=g),
                "svals_exact": ft_svals_exact,
            }
            for m in DW.LORA_MODULES
        },
    }
    ft_dir = out_root / "dw_fleet" / "ft"
    ft_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sidecar, ft_dir / f"{ft.arm_id}_factors_L0.pt")

    assert DW.cmd_align(_args(out_root=str(out_root), dl_root=str(dl_root))) == 0
    out = json.loads((out_root / "dw_fleet" / "alignment.json").read_text())

    blk = out["arms"][s42.arm_id]["factors"]["L0.o_proj"]
    sep = blk["factor_separation"]
    assert len(sep["well_separated"]) == blk["k_basis"]
    assert sep["rel_gap_floor"] == DW.DEGEN_REL_GAP_FLOOR
    read = blk["alignments"]["r_B[evil]"]
    assert 0 <= read["argmax_factor"] < blk["k_basis"]
    assert isinstance(read["argmax_well_separated"], bool)
    assert 0.0 <= read["subspace_proj"] <= 1.0 + 1e-6
    # FT labels recompute exactly from the sidecar's exact-sval prefix.
    ft_blk = out["arms"][ft.arm_id]["factors"]["L0.o_proj"]
    expect_sep = DW.factor_separation(ft_svals_exact.numpy(), 3)
    assert ft_blk["factor_separation"]["well_separated"] == expect_sep["well_separated"]
    assert ft_blk["factor_separation"]["rel_gap_next"] == pytest.approx(expect_sep["rel_gap_next"])
    # Seed-noise anchor: both arms' top-factor separation labeled.
    anchor = out["seed_noise_anchor"]
    assert set(anchor["o_proj"]["top1_well_separated"]) == {s42.arm_id, s137.arm_id}
    assert set(anchor["o_proj"]["factor_separation"]) == {s42.arm_id, s137.arm_id}
    # Pins fingerprint the banked map's FILE BYTES.
    map_file = tmp_path / "fake_banked_map.pt"
    assert (
        out["pins"]["banked_map"]["sha256_16"]
        == (hashlib.sha256(map_file.read_bytes()).hexdigest()[:16])
    )


def test_cmd_align_requires_sidecar_exact_svals(tmp_path, monkeypatch):
    """An FT factor sidecar predating the separation-labels schema (no svals_exact)
    fails LOUD with the re-run command — never a silent unlabeled read."""
    ft = _ft_entry()
    out_root, dl_root, _ = _setup_align(tmp_path, monkeypatch, banked=_rb_banked(), entries=[ft])
    g = torch.Generator().manual_seed(29)
    sidecar = {
        "layer": 0,
        "arm_id": ft.arm_id,
        "method": "svd_lowrank-test",
        "modules": {
            m: {
                "side": I650.RESIDUAL_SIDE_BY_MODULE[m],
                "factors": torch.randn(3, _D_MODEL, generator=g),
                "svals": torch.rand(3, generator=g),
            }
            for m in DW.LORA_MODULES
        },
    }
    ft_dir = out_root / "dw_fleet" / "ft"
    ft_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sidecar, ft_dir / f"{ft.arm_id}_factors_L0.pt")
    with pytest.raises(RuntimeError, match="svals_exact"):
        DW.cmd_align(_args(out_root=str(out_root), dl_root=str(dl_root)))


def test_cmd_align_regime_key_covers_banked_map_bytes(tmp_path, monkeypatch, capsys):
    """A re-banked ridge map (different BYTES at --map-root) must RECOMPUTE align
    units, never resume-skip onto stale Ar[trait] reads (fix-round-3 concern
    dwfleet-align-resume-key-omits-banked-map-fingerprint); an unchanged map still
    resume-skips."""
    lora = _lora_entry()
    banked = _rb_banked()
    banked[f"{DW.DELTA_TF_PREFIX}/{lora.arm_id}/tbar.pt"] = _tbar_payload(layer=0)
    banked[f"{DW.ANCHORS_PREFIX}/{lora.arm_id}.pt"] = _anchor_payload(layer=0)
    out_root, dl_root, _ = _setup_align(tmp_path, monkeypatch, banked=banked, entries=[lora])
    args = _args(out_root=str(out_root), dl_root=str(dl_root))
    assert DW.cmd_align(args) == 0
    capsys.readouterr()
    # Same map bytes: the unit resume-skips.
    assert DW.cmd_align(args) == 0
    assert "resume-skip" in capsys.readouterr().out
    # Re-banked map (same path, different bytes): the unit RECOMPUTES.
    (tmp_path / "fake_banked_map.pt").write_bytes(b"fake-banked-ridge-v2-REBANKED")
    assert DW.cmd_align(args) == 0
    assert "resume-skip" not in capsys.readouterr().out


# ──────────────────────────────────────────────────────────────────────────
# cmd_pilot: pinned to one FULL-FT checkpoint
# ──────────────────────────────────────────────────────────────────────────


def test_cmd_pilot_pinned_to_full_ft(tmp_path):
    """The pilot resolves ft[0] (never lora[0] / synthetic randn), measures REAL deltas via
    the production analysis function, includes checkpoint-IO fields, and keeps the rc=7
    refusal (blocker dwfleet-pilot-not-fullft-plan-adherence)."""
    out_root = tmp_path / "out"
    dl_root = tmp_path / "dl"
    lora = _lora_entry()
    ft = _ft_entry()
    _write_fleet_table(out_root, [lora, ft])
    _write_adapter(dl_root / "adapters" / lora.arm_id / lora.subfolder, r=2)
    bdir, pdir = _write_ft_pair(tmp_path)
    # Pre-place the post checkpoint where the pilot's resume pre-check looks.
    post_dest = dl_root / "ft" / ft.arm_id / ft.subfolder
    post_dest.mkdir(parents=True, exist_ok=True)
    (post_dest / "model.safetensors").write_bytes((pdir / "model.safetensors").read_bytes())
    svd_path = _build_tiny_base_svd(tmp_path)
    args = _args(
        out_root=str(out_root),
        dl_root=str(dl_root),
        base_ckpt=str(bdir),
        base_svd=str(svd_path),
        pilot_wall_cap_h="8.0",
    )
    assert DW.cmd_pilot(args) == 0
    report = json.loads((out_root / "dw_fleet" / "pilot.json").read_text())
    assert report["pilot_method"] == "ft"
    assert report["pilot_arm"] == ft.arm_id
    assert report["pilot_repo"] == ft.repo_id  # the overflow-repo path is the pilot target
    assert report["n_pilot_ft_matrices"] == 4  # 2 layers x (down_proj, q_proj)
    for key in ("measured_ft_dl_s", "measured_per_call_s_ft", "measured_lora_arm_s"):
        assert key in report
    assert report["ft_matrices_per_ckpt"] == DW.FT_MATRICES_PER_CKPT
    assert report["verdict"] == "pass"
    # A cap of 0 h always refuses with the DESIGNED rc.
    args_cap = _args(
        out_root=str(out_root),
        dl_root=str(dl_root),
        base_ckpt=str(bdir),
        base_svd=str(svd_path),
        pilot_wall_cap_h="0.0",
    )
    assert DW.cmd_pilot(args_cap) == DW.RC_PILOT_REFUSAL


def test_cmd_pilot_requires_ft_arm(tmp_path):
    """A fleet with no full-FT checkpoint cannot satisfy the pilot's pin — fail loud."""
    out_root = tmp_path / "out"
    _write_fleet_table(out_root, [_lora_entry()])
    svd_path = _build_tiny_base_svd(tmp_path)
    args = _args(
        out_root=str(out_root),
        dl_root=str(tmp_path / "dl"),
        base_ckpt=str(tmp_path),
        base_svd=str(svd_path),
    )
    with pytest.raises(RuntimeError, match="PINNED to one FULL-FT"):
        DW.cmd_pilot(args)


# ──────────────────────────────────────────────────────────────────────────
# Fleet enumeration
# ──────────────────────────────────────────────────────────────────────────


def _arms_json() -> dict:
    """A minimal arms.json in the observed @3bb20debe2 schema."""
    return {
        "arms": [
            {
                "arm_id": "cas-pers-con-lr1e5-s42",
                "kind": "content",
                "beh_key": "cas",
                "method": "lora",
                "adapter_repo": "org/model",
                "adapter_subfolder": "issue1434/ws-pers-lr1e5/checkpoint-25",
            },
            {
                "arm_id": "cas-pers-ft-con-s42",
                "kind": "content",
                "beh_key": "cas",
                "method": "ft",
                "ft_repo": "org/overflow",
                "ft_subfolder": "issue1586/cas-pers-ft-con-s42/checkpoint-10",
            },
        ]
    }


def test_enumerate_fleet_resolves_lora_and_ft_fields():
    """LoRA rows resolve adapter_*; ft rows resolve ft_*; extras append with provenance."""
    fleet = DW.enumerate_fleet(
        _arms_json(),
        [
            {
                "arm_id": "x",
                "method": "lora",
                "repo_id": "org/m",
                "subfolder": "s",
                "source_manifest": "issue2474",
            }
        ],
    )
    assert [e.arm_id for e in fleet] == ["cas-pers-con-lr1e5-s42", "cas-pers-ft-con-s42", "x"]
    assert fleet[0].repo_id == "org/model" and fleet[1].repo_id == "org/overflow"
    assert fleet[2].source_manifest == "issue2474"


def test_enumerate_fleet_fails_loud_on_missing_fields_and_empty():
    """A record missing its checkpoint fields — or an empty fleet — raises (never silent)."""
    bad = {"arms": [{"arm_id": "y", "kind": "content", "beh_key": "z", "method": "ft"}]}
    with pytest.raises(RuntimeError, match="missing checkpoint fields"):
        DW.enumerate_fleet(bad)
    with pytest.raises(RuntimeError, match="empty fleet"):
        DW.enumerate_fleet({"arms": []})


def test_adapter_key_regex_parses_peft_names():
    """The PEFT safetensors key regex extracts (layer, module, A|B)."""
    m = DW._ADAPTER_KEY_RE.match("base_model.model.model.layers.17.self_attn.q_proj.lora_A.weight")
    assert m and (int(m.group(1)), m.group(3), m.group(4)) == (17, "q_proj", "A")
    m2 = DW._ADAPTER_KEY_RE.match("base_model.model.model.layers.3.mlp.down_proj.lora_B.weight")
    assert m2 and (int(m2.group(1)), m2.group(3), m2.group(4)) == (3, "down_proj", "B")


# ──────────────────────────────────────────────────────────────────────────
# issue650_analyze --modules extension (in-place, default preserved)
# ──────────────────────────────────────────────────────────────────────────


def test_issue650_modules_default_preserved():
    """The build-base-svd --modules default is exactly the original module pair."""
    assert "cmd_build_base_svd" in I650.main.__globals__  # sanity that main exists
    args = _parse_build_base_svd([])
    assert args.modules == "up_proj,down_proj"
    assert I650.RESIDUAL_SIDE_BY_MODULE["up_proj"] == "V"
    assert I650.RESIDUAL_SIDE_BY_MODULE["down_proj"] == "U"
    assert set(I650.RESIDUAL_SIDE_BY_MODULE) == set(DW.LORA_MODULES)


def _parse_build_base_svd(extra: list[str]):
    """Parse ``build-base-svd`` argv through issue650_analyze's REAL main() parser."""
    import unittest.mock as mock

    captured = {}

    def _capture(args):
        captured["args"] = args
        return 0

    with mock.patch.object(I650, "cmd_build_base_svd", side_effect=_capture):
        rc = I650.main(["build-base-svd", *extra])
    assert rc == 0 and "args" in captured
    return captured["args"]


def test_issue650_residual_svd_basis_matches_original_math():
    """_residual_svd_basis reproduces the original up (V rows) / down (U.T rows) math."""
    rng = np.random.default_rng(6)
    w_up = rng.normal(size=(12, 5))  # (d_ff, d_in) shape class
    _, _, vt = np.linalg.svd(w_up, full_matrices=False)
    np.testing.assert_allclose(I650._residual_svd_basis(w_up, "V"), vt)
    w_down = rng.normal(size=(5, 12))  # (d_out, d_ff) shape class
    u, _, _ = np.linalg.svd(w_down, full_matrices=False)
    np.testing.assert_allclose(I650._residual_svd_basis(w_down, "U"), u.T)


def test_issue650_build_base_svd_modules_on_tiny_model(tmp_path):
    """cmd_build_base_svd with --modules writes per-module residual-side bases loadable by
    load_base_svd (tiny fake decoder — no real model download)."""
    import unittest.mock as mock

    out_path = _build_tiny_base_svd(tmp_path, modules=("q_proj", "o_proj", "down_proj"))
    loaded = I650.load_base_svd(out_path, modules=("q_proj", "o_proj", "down_proj"))
    assert set(loaded["q_proj"][0]) == {"V"}  # read-side module → V basis
    assert set(loaded["o_proj"][0]) == {"U"}  # write-side module → U basis
    assert set(loaded["down_proj"][0]) == {"U"}
    assert loaded["_meta"]["modules"] == ["q_proj", "o_proj", "down_proj"]
    # Unknown module names fail loud at parse time.
    fake_auto = mock.MagicMock()
    fake_auto.from_pretrained.return_value = _TinyModel()
    with (
        pytest.raises(ValueError, match="unknown --modules"),
        mock.patch("transformers.AutoModelForCausalLM", fake_auto),
    ):
        I650.main(
            [
                "build-base-svd",
                "--base-model",
                "fake",
                "--base-svd",
                str(tmp_path / "x.pt"),
                "--modules",
                "bogus_proj",
            ]
        )


def test_dw_regime_key_stable_and_sensitive():
    """Resume keys derive from generating parameters, never float bytes."""
    a = DW.regime_key(phase="lora", top_k=8, dv3_draws=200)
    b = DW.regime_key(phase="lora", top_k=8, dv3_draws=200)
    c = DW.regime_key(phase="lora", top_k=8, dv3_draws=100)
    assert a == b and a != c
    # Content-keying: differing base-svd _meta yields differing keys.
    m1 = DW.regime_key(phase="lora", base_svd_meta={"modules": ["up_proj", "down_proj"]})
    m2 = DW.regime_key(phase="lora", base_svd_meta={"modules": list(DW.LORA_MODULES)})
    assert m1 != m2
