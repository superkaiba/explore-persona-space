"""Unit tests for scripts/issue2569_weights.py — the P-A weights battery, part 1.

Tiny-synthetic only (d <= 48): the dense d=3584 factorizations are pod-driver
territory (the driver's docstring says so), so every test here builds small
matrices with DISTINCT singular values — equal singular values make the SVD mix
degenerate subspaces and the per-direction self-alignment nondeterministic.
The d=32 end-to-end test writes a contract-shaped synthetic ridge.pt under a
tmp ``--map-root`` and monkeypatches ONLY the external boundaries
(``OP.load_apply_map`` -> a signature-conformant from-raw reimplementation;
``C.write_sentinel`` -> a signature-conformant tmp writer); the driver's own
bodies all execute for real (code-style.md § One production-body test per
seam-stubbed function — the stubbed callees are unmodified repo helpers).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_operator as OP  # noqa: E402
import issue2569_weights as WB  # noqa: E402

# ── pure-core tests (no disk) ─────────────────────────────────────────────────────


def test_full_svd_row_identity_and_order():
    """u_i @ A == sigma_i v_i for the FULL SVD; sigma descending; u orthonormal."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((24, 24))
    u, s, v = WB.full_svd_row(A)
    assert np.all(np.diff(s) <= 0), "sigma must be descending"
    assert np.allclose(u.T @ A, s[:, None] * v.T, atol=1e-10)
    assert np.allclose(u.T @ u, np.eye(24), atol=1e-10)


def test_self_alignment_definition_and_transpose_invariance():
    """c_i == cos(u_i @ A, u_i) elementwise, and the SCALAR is transpose-invariant."""
    rng = np.random.default_rng(5)
    A = rng.standard_normal((12, 12))
    u, s, v = WB.full_svd_row(A)
    c = WB.self_alignment(u, v)
    for i in (0, 3, 11):
        img = u[:, i] @ A
        ref = float(img @ u[:, i] / (np.linalg.norm(img) * np.linalg.norm(u[:, i])))
        assert np.isclose(c[i], ref, atol=1e-10)
    ut, st, vt = WB.full_svd_row(A.T)
    assert np.allclose(s, st, atol=1e-10)  # same spectrum, same order
    ct = WB.self_alignment(ut, vt)
    assert np.allclose(c, ct, atol=1e-8), "c_i must be transpose-invariant per direction"


# Distinct-gain anatomy fixture: sigma^2 mass = [4, 1.3225, .9025, .3025, ~0 x4];
# cumulative fractions hit 0.90 at rank 3 (tau90=0.95) and 0.99 at rank 4 (tau=0.55).
_SIGMA_FIX = np.array([2.0, 1.15, 0.95, 0.55, 1e-8, 1e-8, 1e-8, 1e-8])
_C_FIX = np.array([0.5, 0.0, 1.0, -0.9, 1.0, 1.0, 1.0, 1.0])


def test_classify_anatomy_distinct_gain_fixture():
    """Every class fires once on the distinct-gain fixture, at tau_kernel=0.55."""
    labels = WB.classify_anatomy(_SIGMA_FIX, _C_FIX, tau_kernel=0.55)
    assert list(labels[:4]) == ["rotated_scaled", "transcoded", "copied", "damped"]
    assert set(labels[4:]) == {"ignored"}


def test_anatomy_stats_tau_k90_and_mass_fractions():
    """tau/k99/k90 match the hand-computed fixture; mass fracs + counts close."""
    st = WB.anatomy_stats(_SIGMA_FIX, _C_FIX)
    assert st["k99"] == 4 and np.isclose(st["tau_kernel"], 0.55)
    assert st["k90"] == 3 and np.isclose(st["tau_k90"], 0.95)
    total_count = sum(v["count"] for v in st["classes"].values())
    total_mass = sum(v["sigma2_mass_frac"] for v in st["classes"].values())
    assert total_count == _SIGMA_FIX.size
    assert np.isclose(total_mass, 1.0, atol=1e-12)
    assert st["classes"]["ignored"]["count"] == 4
    assert st["classes"]["damped"]["count"] == 1


def test_effective_kernel_stats_strict_filter_and_phrasing():
    """Kernel = strict sigma < tau (boundary value stays OUT); phrasing is low-gain."""
    u = np.eye(8)
    basis, st = WB.effective_kernel_stats(u, _SIGMA_FIX)
    assert st["kernel_dim"] == 4 and basis.shape == (8, 4)
    assert np.isclose(st["tau_kernel"], 0.55)  # sigma==tau (the damped dir) excluded
    assert "reads at" in st["claims_phrasing"]
    assert "no exact kernel" in st["claims_phrasing"]


def test_eigen_summary_biortho_conjugate_pair_and_row_expansion():
    """Biorthonormality, conjugate-pair count, rho, and the row-action expansion."""
    theta = 0.7
    rot = 0.9 * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = np.zeros((6, 6))
    A[:2, :2] = rot
    A[2:, 2:] = np.diag([1.2, 0.5, 0.3, 0.1])
    eig = WB.eigen_summary(A, n_top=6)
    assert np.isclose(eig["rho"], 1.2)
    assert eig["n_complex_pairs"] == 1 and eig["n_real_eigs"] == 4
    assert eig["biortho_max_err"] < 1e-10
    assert np.allclose(np.abs(eig["lam"]), np.array([1.2, 0.9, 0.9, 0.5, 0.3, 0.1]))
    assert np.allclose(eig["write_left_top"] @ eig["read_right_top"], np.eye(6), atol=1e-8)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(6)
    expansion = (x @ eig["read_right_top"]) @ (eig["lam"][:, None] * eig["write_left_top"])
    assert np.allclose(expansion.imag, 0.0, atol=1e-10)
    assert np.allclose(expansion.real, x @ A, atol=1e-8)


def test_alpha_low_rank_recovers_planted_alpha_plus_rank1():
    """A = 0.7*I + 2*g h^T with g ⊥ h: alpha exact, residual rank-1 at norm 2."""
    rng = np.random.default_rng(11)
    d = 16
    g = rng.standard_normal(d)
    g /= np.linalg.norm(g)
    h = rng.standard_normal(d)
    h -= (h @ g) * g  # orthogonalize so trace(g h^T) = 0 and alpha is exact
    h /= np.linalg.norm(h)
    A = 0.7 * np.eye(d) + 2.0 * np.outer(g, h)
    st = WB.alpha_low_rank_stats(A, ks=(1, 8))
    assert np.isclose(st["alpha"], 0.7, atol=1e-12)
    assert np.isclose(st["fro_residual"], 2.0, atol=1e-10)
    assert np.isclose(st["sigma_residual"][0], 2.0, atol=1e-10)
    assert st["sigma_residual"][1] < 1e-10
    assert np.isclose(st["var_explained_topk"]["1"], 1.0, atol=1e-12)


def test_fixed_point_stats_both_rho_branches():
    """rho < 1 keeps the iterated-map reading; rho >= 1 drops it; both solve exactly."""
    rng = np.random.default_rng(2)
    b = rng.standard_normal(4)
    a_contract = 0.5 * np.eye(4)
    x1, st1 = WB.fixed_point_stats(a_contract, b, OP.spectral_radius(a_contract))
    assert st1["iterated_map_reading"] is True
    assert st1["residual_rel"] < 1e-12
    assert np.allclose(x1 @ (np.eye(4) - a_contract), b)
    a_expand = 1.5 * np.eye(4)
    x2, st2 = WB.fixed_point_stats(a_expand, b, OP.spectral_radius(a_expand))
    assert st2["iterated_map_reading"] is False
    assert st2["residual_rel"] < 1e-12
    assert np.allclose(x2 @ (np.eye(4) - a_expand), b)


# ── driver end-to-end (d=32 synthetic payload; external boundaries faked) ─────────


def _synth_raw_payload(d: int, layer: int = 19, seed: int = 7) -> dict:
    """Contract-shaped raw torch payload (mirrors tests/test_issue2569_operator.py)."""
    rng = np.random.default_rng(seed)
    return {
        "kind": "ridge",
        "fitter": "ridge",
        "layer": layer,
        "selected_lambda": 0.001,
        "W": torch.tensor(rng.standard_normal((d, d)), dtype=torch.float32),
        "xmu": torch.tensor(rng.standard_normal(d), dtype=torch.float32),
        "xsd": torch.tensor(np.exp(rng.standard_normal(d) * 0.5), dtype=torch.float32),
        "ymu": torch.tensor(rng.standard_normal(d), dtype=torch.float32),
    }


def _fake_apply_map(raw: dict, X: np.ndarray, device) -> np.ndarray:
    """Signature-conformant apply_map reference: the SAME affine map, from raw, fp64.

    Mirrors ``issue779_ffc_n1m_fits.apply_map(payload, X, device)`` semantics so
    B1 assert (iii) stays a real cross-implementation check in the e2e (the only
    thing faked is the ~20 s importlib load of the #779 module chain).
    """
    W = np.asarray(raw["W"], dtype=np.float64)
    xmu = np.asarray(raw["xmu"], dtype=np.float64)
    xsd = np.asarray(raw["xsd"], dtype=np.float64)
    ymu = np.asarray(raw["ymu"], dtype=np.float64)
    return ((np.asarray(X, dtype=np.float64) - xmu) / xsd) @ W + ymu


def _run_driver(monkeypatch, tmp_path: Path, argv: list[str]) -> None:
    """Invoke WB.main() with a patched argv + external boundaries; expect exit 0."""

    def fake_sentinel(kind, note, task_id=779, extra=None):
        p = tmp_path / f"sentinel-{kind}.json"
        p.write_text(json.dumps({"kind": kind, "note": note}))
        return p

    monkeypatch.setattr(WB.OP, "load_apply_map", lambda: _fake_apply_map)
    monkeypatch.setattr(WB.C, "write_sentinel", fake_sentinel)
    monkeypatch.setattr(sys, "argv", ["issue2569_weights.py", *argv])
    with pytest.raises(SystemExit) as exc:
        WB.main()
    assert exc.value.code == 0


def test_driver_e2e_smoke_and_resume(tmp_path, monkeypatch):
    """Full --phase all --smoke e2e at d=32: artifacts, schema, and resume skip."""
    d = 32
    map_root = tmp_path / "maproot"
    path = OP.banked_map_path(19, root=map_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_synth_raw_payload(d), path)
    out = tmp_path / "out"
    argv = [
        "--phase",
        "all",
        "--smoke",
        "--skip-upload",
        "--out-root",
        str(out),
        "--map-root",
        str(map_root),
    ]
    _run_driver(monkeypatch, tmp_path, argv)

    leg1, leg8 = out / "leg1", out / "leg8"
    for name in (
        "entry_asserts_L19.json",
        "factor_L19.pt",
        "factor_L19.json",
        "anatomy_L19.json",
        "alpha_lowrank_L19.json",
        "fixed_point_L19.pt",
        "fixed_point_L19.json",
    ):
        assert (leg1 / name).exists(), name
    for name in ("effective_kernel_L19.pt", "effective_kernel_L19.json"):
        assert (leg8 / name).exists(), name

    fac = torch.load(leg1 / "factor_L19.pt", map_location="cpu", weights_only=False)
    assert fac["sigma"].shape == (d,)
    assert fac["read_input_u_fp32"].shape == (d, d)
    assert fac["eig_read_right_v_top"].shape == (d, 8)  # smoke top_k = 8
    assert fac["eig_write_left_rows_top"].shape == (8, d)
    assert fac["stats"]["biortho_max_err"] < 1e-6

    anat = json.loads((leg1 / "anatomy_L19.json").read_text())
    assert len(anat["labels"]) == d and len(anat["top_directions"]) == 8
    assert np.isclose(sum(v["sigma2_mass_frac"] for v in anat["classes"].values()), 1.0, atol=1e-9)
    assert anat["data_weighted_mass"].startswith("deferred-to-P-B")
    assert anat["metadata"]["git_commit"], "reproducibility metadata missing"
    assert anat["regime"]["smoke"] is True and anat["regime"]["n_draws"] == 100

    fp = json.loads((leg1 / "fixed_point_L19.json").read_text())
    assert fp["iterated_map_reading"] == (fp["rho"] < 1.0)
    assert fp["residual_rel"] < 1e-8

    ker = json.loads((leg8 / "effective_kernel_L19.json").read_text())
    assert "reads at" in ker["claims_phrasing"]
    assert (
        ker["kernel_dim"]
        == torch.load(leg8 / "effective_kernel_L19.pt", map_location="cpu", weights_only=False)[
            "kernel_basis_fp32"
        ].shape[1]
    )

    entry = json.loads((leg1 / "entry_asserts_L19.json").read_text())
    assert entry["entry_asserts"]["apply_path"]["max_abs_diff"] < 1e-8

    # Resume: identical regime -> every phase unit SKIPs (factor .pt untouched).
    before = os.stat(leg1 / "factor_L19.pt").st_mtime_ns
    _run_driver(monkeypatch, tmp_path, argv)
    assert os.stat(leg1 / "factor_L19.pt").st_mtime_ns == before

    # --fresh busts the resume predicate (factor .pt rewritten).
    _run_driver(monkeypatch, tmp_path, [*argv, "--fresh"])
    assert os.stat(leg1 / "factor_L19.pt").st_mtime_ns != before


def test_import_check_exits_zero(monkeypatch, capsys):
    """--import-check runs argcheck + deferred imports in-process and exits 0."""
    monkeypatch.setattr(sys, "argv", ["issue2569_weights.py", "--import-check"])
    with pytest.raises(SystemExit) as exc:
        WB.main()
    assert exc.value.code == 0
    assert "[import-check] OK" in capsys.readouterr().out
