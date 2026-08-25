"""Unit tests for scripts/issue2569_dw_fleet.py — leg-5 dW geometry (plan #2569 §4 leg 5).

Covers: LoRA scaling-regime reading (rsLoRA alpha/sqrt(r) vs classic alpha/r — artifact-reuse
check (g)), dW reconstruction, effective-rank summaries on a known spectrum, batched-vs-serial
svdvals equality, the dv3 nested-schema builder validated by ``issue650_analyze``'s
load-enforced ``assert_dv3_schema`` (a mismatched aggregation is REJECTED), the max-matched
alignment read (planted direction above null / random below), fleet enumeration fail-loud
behavior, and the in-place ``--modules`` extension of ``issue650_analyze.py`` (default
preserved byte-for-byte: up_proj -> V, down_proj -> U). All synthetic, CPU-fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue650_analyze as I650  # noqa: E402
import issue2569_dw_fleet as DW  # noqa: E402

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


def test_svdvals_stack_matches_serial():
    """The batched stack path equals per-matrix svdvals (vectorize-first equivalence)."""
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
    bad = dict(res)
    bad["null_aggregation"] = "flat_p95_single_cosines"
    with pytest.raises(AssertionError):
        DW.dv3_payload_from_null({"write": bad})


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


def test_issue650_build_base_svd_modules_on_tiny_model(tmp_path, monkeypatch):
    """cmd_build_base_svd with --modules writes per-module residual-side bases loadable by
    load_base_svd (tiny fake decoder — no real model download)."""

    class _W:
        def __init__(self, shape, seed):
            g = torch.Generator().manual_seed(seed)
            self.weight = torch.randn(*shape, generator=g)

    class _Layer:
        def __init__(self, seed):
            d, f = 6, 10
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

    class _Model:
        model = type("M", (), {"layers": [_Layer(0), _Layer(100)]})()

    import unittest.mock as mock

    fake_auto = mock.MagicMock()
    fake_auto.from_pretrained.return_value = _Model()
    out_path = tmp_path / "base_svd.pt"
    with mock.patch("transformers.AutoModelForCausalLM", fake_auto):
        rc = I650.main(
            [
                "build-base-svd",
                "--base-model",
                "fake",
                "--base-svd",
                str(out_path),
                "--modules",
                "q_proj,o_proj,down_proj",
            ]
        )
    assert rc == 0 and out_path.is_file()
    loaded = I650.load_base_svd(out_path, modules=("q_proj", "o_proj", "down_proj"))
    assert set(loaded["q_proj"][0]) == {"V"}  # read-side module → V basis
    assert set(loaded["o_proj"][0]) == {"U"}  # write-side module → U basis
    assert set(loaded["down_proj"][0]) == {"U"}
    assert loaded["_meta"]["modules"] == ["q_proj", "o_proj", "down_proj"]
    # Unknown module names fail loud at parse time.
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
