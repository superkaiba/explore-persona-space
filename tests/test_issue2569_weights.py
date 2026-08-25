"""Unit tests for scripts/issue2569_weights.py — the P-A weights battery, parts 1+2.

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


# ── pure-core tests, leg 8 steps 3+4 ──────────────────────────────────────────────


def test_monitor_flip_geometry_unit_read_displacement():
    """gradient == A @ r_hat; the min-norm step of that size moves the read by 1."""
    rng = np.random.default_rng(21)
    d = 16
    A = rng.standard_normal((d, d))
    b = rng.standard_normal(d)
    r = rng.standard_normal(d)
    r_hat = r / np.linalg.norm(r)
    flip = WB.monitor_flip_geometry(A, b, r_hat)
    assert np.allclose(flip["gradient"], A @ r_hat, atol=1e-12)
    assert np.isclose(flip["read_at_zero_context"], float(b @ r_hat), atol=1e-12)
    # the minimal context step: dv = g / |g|^2 has norm 1/|g| and moves the
    # mapped read r_hat . (v @ A + b) by EXACTLY one unit
    g = flip["gradient"]
    dv = g / (g @ g)
    assert np.isclose(np.linalg.norm(dv), flip["min_context_change_per_unit_read"], atol=1e-12)
    v0 = rng.standard_normal(d)
    read = lambda v: float(r_hat @ (v @ A + b))  # noqa: E731
    assert np.isclose(read(v0 + dv) - read(v0), 1.0, atol=1e-10)


def test_least_norm_preimage_full_truncated_and_coset_accounting():
    """Full pinv reconstructs y exactly; truncation achieves 1 - below-tau mass."""
    rng = np.random.default_rng(22)
    d = 16
    # controlled spectrum: 4 directions below tau=0.5
    s = np.concatenate([np.linspace(3.0, 1.0, d - 4), np.full(4, 0.05)])
    u_q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    v_q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A = (u_q * s) @ v_q.T  # row action: u_i @ A = s_i v_i
    y = rng.standard_normal(d)
    y /= np.linalg.norm(y)
    pre = WB.least_norm_preimage(u_q, s, v_q, y, tau=0.5)
    assert pre["kernel_dim"] == 4 and pre["n_retained"] == d - 4
    # FULL pinv: exact reconstruction (all sigma > 0)
    assert np.allclose(pre["preimage_fullpinv"] @ A, y, atol=1e-10)
    # TRUNCATED: achieves exactly the retained-subspace projection of y
    proj = v_q.T @ y
    y_retained = v_q[:, s >= 0.5] @ proj[s >= 0.5]
    assert np.allclose(pre["preimage"] @ A, y_retained, atol=1e-10)
    mass_below = float(proj[s < 0.5] @ proj[s < 0.5])
    assert np.isclose(pre["target_mass_below_tau_frac"], mass_below, atol=1e-12)
    assert np.isclose(pre["achieved_level_fraction_algebra"], 1.0 - mass_below, atol=1e-12)
    # dropping the huge 1/sigma below-tau terms can only SHRINK the pre-image
    assert pre["preimage_fullpinv_norm"] >= pre["preimage_norm"]


def test_select_lambda_with_widening_peak_outside_grid():
    """An argmax on the low edge widens until the true peak (1e-3) is interior."""
    calls: list[np.ndarray] = []

    def val_r2(lams: np.ndarray) -> np.ndarray:
        calls.append(lams)
        return -((np.log10(lams) + 3.0) ** 2)  # peak at lambda = 1e-3

    sel = WB.select_lambda_with_widening(val_r2, ("logspace", 0.0, 2.0, 5), widen_max=3)
    assert sel["lambda_grid_edge"] == "none"
    assert sel["widen_rounds_used"] >= 1
    assert np.isclose(np.log10(sel["selected_lambda"]), -3.0, atol=0.5)
    assert len(calls) == sel["widen_rounds_used"] + 1


def test_select_lambda_with_widening_reports_residual_edge():
    """A monotone score exhausts widen_max and REPORTS the residual edge."""
    sel = WB.select_lambda_with_widening(
        lambda lams: -np.log10(lams), ("logspace", 0.0, 2.0, 5), widen_max=2
    )
    assert sel["lambda_grid_edge"] == "low"
    assert sel["widen_rounds_used"] == 2


def _planted_rows(d: int = 12, n_total: int = 100, seed: int = 23):
    """Noiseless-linear fp16 row store: Y = X @ A_true, so t = X @ (A_true r_hat)."""
    rng = np.random.default_rng(seed)
    a_true = rng.standard_normal((d, d)) / np.sqrt(d)
    x = rng.standard_normal((n_total, d)).astype(np.float16)
    y = (x.astype(np.float64) @ a_true).astype(np.float16)
    r = rng.standard_normal((d, 2))
    r /= np.linalg.norm(r, axis=0)
    return x, y, a_true, r


def test_certificate_rows_core_planted_signal():
    """Probe recovers the planted linear target (heldout R^2 ~ 1); mapped corr ~ 1."""
    d = 12
    x, y, a_true, r = _planted_rows(d=d)
    core = WB.certificate_rows_core(
        x,
        y,
        train_pos=np.arange(0, 64),
        val_pos=np.arange(64, 80),
        test_pos=np.arange(80, 100),
        r_mat=r,
        trait_names=("evil", "sycophancy"),
        a_mat=a_true,
        b_vec=np.zeros(d),
        chunk=16,
        dev="cpu",
        grid_params=("logspace", -5.0, 2.0, 8),
        widen_max=1,
    )
    for trait in ("evil", "sycophancy"):
        p = core["probes"][trait]
        assert p["status"] == "computed"
        assert p["n_train"] == 64 and p["d"] == d
        assert p["heldout_r2"] > 0.95, p
        assert p["grad_norm"] > 0.0
        h = core["heldout"][trait]
        assert h["mapped_read"]["corr_with_target"] > 0.99
        assert h["fitted_probe"]["corr_with_target"] > 0.99
        for mon in ("direct_projection", "mapped_read", "fitted_probe"):
            hs = h[mon]
            assert np.isclose(
                hs["eps_to_move_one_heldout_sd"] * hs["corr_with_target"],
                hs["signal_to_sensitivity"],
                atol=1e-12,
            )
    assert core["w"].shape == (d, 2)
    assert "lambda I" in core["rows"]["ridge_convention"]


def test_certificate_rows_core_refuses_underdetermined():
    """HARD gate: n_train < d raises (estimator-degenerate regime, #1701)."""
    d = 12
    x, y, a_true, r = _planted_rows(d=d)
    with pytest.raises(RuntimeError, match="estimator-degenerate"):
        WB.certificate_rows_core(
            x,
            y,
            train_pos=np.arange(0, 8),  # 8 < d=12
            val_pos=np.arange(64, 80),
            test_pos=np.arange(80, 100),
            r_mat=r,
            trait_names=("evil", "sycophancy"),
            a_mat=a_true,
            b_vec=np.zeros(d),
        )


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


def _synth_rb_dir(tmp_path: Path, d: int, seed: int = 31) -> Path:
    """Local r_B blob dir matching the PROBED real-artifact schema.

    Mirrors the observed ``issue779_monitoring/r_b/evil.pt @ 037fcbb2`` keys
    (counts/layers/metadata/r_b/smoke/trait; r_b (28, d) fp32) so the driver's
    consumer asserts exercise the REAL schema, at tiny d.
    """
    rng = np.random.default_rng(seed)
    rb_dir = tmp_path / "rb"
    rb_dir.mkdir(parents=True, exist_ok=True)
    for trait in ("evil", "sycophancy", "hallucination"):
        torch.save(
            {
                "trait": trait,
                "smoke": False,
                "layers": list(range(28)),
                "counts": {"pos": 10, "neg": 10},
                "metadata": {"synthetic_test_fixture": True},
                "r_b": torch.tensor(rng.standard_normal((28, d)), dtype=torch.float32),
            },
            rb_dir / f"{trait}.pt",
        )
    return rb_dir


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
    rb_dir = _synth_rb_dir(tmp_path, d)
    argv = [
        "--phase",
        "all",
        "--smoke",
        "--skip-upload",
        "--out-root",
        str(out),
        "--map-root",
        str(map_root),
        "--rb-dir",
        str(rb_dir),
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
    for name in (
        "effective_kernel_L19.pt",
        "effective_kernel_L19.json",
        "monitor_geometry_L19.pt",
        "monitor_geometry_L19.json",
        "certificates_L19.json",
    ):
        assert (leg8 / name).exists(), name
    # probe .pt is rows-attached only: this run carries NO --rows-dir
    assert not (leg8 / "certificates_probe_L19.pt").exists()

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

    # leg 8 step 3: monitor geometry — schema, caveats, coset + kernel accounting
    mg = json.loads((leg8 / "monitor_geometry_L19.json").read_text())
    assert set(mg["traits"]) == {"evil", "sycophancy", "hallucination"}
    for t, row in mg["traits"].items():
        assert row["grad_norm"] > 0.0, t
        assert np.isclose(row["min_context_change_per_unit_read"], 1.0 / row["grad_norm"])
        assert 0.0 <= row["target_mass_below_tau_frac"] <= 1.0
        assert row["preimage_orientation_residual"] <= 1e-2
        assert np.isclose(
            row["achieved_level_fraction_algebra"],
            1.0 - row["target_mass_below_tau_frac"],
            atol=1e-12,
        )
        assert row["kernel_dim"] == ker["kernel_dim"], "kernel accounting must agree"
        assert row["n_retained"] + row["kernel_dim"] == d
    assert WB.CAVEAT_ACTIVATION in mg["caveats"] and WB.CAVEAT_MAP_LEVEL in mg["caveats"]
    assert "least-norm pre-image is never 'the' context" in mg["coset_ambiguity"]
    assert mg["sae_naming"].startswith("deferred-to-SAE-dashboard-unit")
    mg_pt = torch.load(leg8 / "monitor_geometry_L19.pt", map_location="cpu", weights_only=False)
    assert mg_pt["traits"]["evil"]["preimage_unit_level"].shape == (d,)
    assert mg_pt["caveats"] == [WB.CAVEAT_ACTIVATION, WB.CAVEAT_MAP_LEVEL]

    # leg 8 step 4: certificates — weights-only legs computed, probe DEFERRED
    cert = json.loads((leg8 / "certificates_L19.json").read_text())
    for t, mon in cert["monitors"].items():
        assert mon["direct_projection"]["grad_norm"] == 1.0, t
        assert np.isclose(mon["mapped_over_direct_grad_ratio"], mon["mapped_read"]["grad_norm"])
        assert mon["fitted_probe"]["status"].startswith("deferred"), t
        assert isinstance(mon["heldout"], str) and mon["heldout"].startswith("deferred")
    assert isinstance(cert["rows"], str) and "leg8-cert-heldout-needs-pb-rows" in cert["rows"]
    assert "rho(A)" in cert["bound_scope"]
    assert "inapplicable" in cert["baselines"]["identity_bias"]
    assert "inapplicable" in cert["baselines"]["knn_retrieval"]
    assert WB.CAVEAT_ACTIVATION in cert["caveats"] and WB.CAVEAT_MAP_LEVEL in cert["caveats"]
    assert cert["regime"]["certificates"]["rows_attached"] is False
    assert cert["regime"]["certificates"]["rb_source"] == "local-dir"

    # Resume: identical regime -> every phase unit SKIPs (factor .pt untouched).
    before = os.stat(leg1 / "factor_L19.pt").st_mtime_ns
    before_mg = os.stat(leg8 / "monitor_geometry_L19.pt").st_mtime_ns
    _run_driver(monkeypatch, tmp_path, argv)
    assert os.stat(leg1 / "factor_L19.pt").st_mtime_ns == before
    assert os.stat(leg8 / "monitor_geometry_L19.pt").st_mtime_ns == before_mg

    # --fresh busts the resume predicate (factor .pt rewritten).
    _run_driver(monkeypatch, tmp_path, [*argv, "--fresh"])
    assert os.stat(leg1 / "factor_L19.pt").st_mtime_ns != before


def test_driver_certificates_rows_attached_branch(tmp_path, monkeypatch):
    """--rows-dir at L19 arms the probe leg: probe .pt written, monitors computed.

    ``_load_rows_store`` is faked at the P-B DATA boundary only (the pinned
    production split ids cannot exist at d=32) with REAL planted arrays of the
    contract shape; the phase body, ``certificate_rows_core``, and the probe
    save all execute for real.
    """
    d = 32
    map_root = tmp_path / "maproot"
    path = OP.banked_map_path(19, root=map_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_synth_raw_payload(d), path)
    out = tmp_path / "out"
    rb_dir = _synth_rb_dir(tmp_path, d)
    rng = np.random.default_rng(41)
    x = rng.standard_normal((120, d)).astype(np.float16)
    y = (x.astype(np.float64) @ (rng.standard_normal((d, d)) / np.sqrt(d))).astype(np.float16)
    fake_store = (x, y, np.arange(0, 80), np.arange(80, 100), np.arange(100, 120))
    monkeypatch.setattr(WB, "_load_rows_store", lambda args: fake_store)
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    base = [
        "--smoke",
        "--skip-upload",
        "--out-root",
        str(out),
        "--map-root",
        str(map_root),
        "--rb-dir",
        str(rb_dir),
        "--rows-dir",
        str(rows_dir),
    ]
    _run_driver(monkeypatch, tmp_path, ["--phase", "factor", *base])
    _run_driver(monkeypatch, tmp_path, ["--phase", "certificates", *base])
    leg8 = out / "leg8"
    cert = json.loads((leg8 / "certificates_L19.json").read_text())
    assert cert["regime"]["certificates"]["rows_attached"] is True
    for mon in cert["monitors"].values():
        assert mon["fitted_probe"]["status"] == "computed"
        assert mon["fitted_probe"]["n_train"] == 80
        assert mon["heldout"]["fitted_probe"]["std"] >= 0.0
    assert isinstance(cert["rows"], dict) and cert["rows"]["n_train"] == 80
    probe = torch.load(leg8 / "certificates_probe_L19.pt", map_location="cpu", weights_only=False)
    assert probe["w"].shape == (d, 3)
    assert probe["traits"] == ["evil", "sycophancy", "hallucination"]


def test_import_check_exits_zero(monkeypatch, capsys):
    """--import-check runs argcheck + deferred imports in-process and exits 0."""
    monkeypatch.setattr(sys, "argv", ["issue2569_weights.py", "--import-check"])
    with pytest.raises(SystemExit) as exc:
        WB.main()
    assert exc.value.code == 0
    assert "[import-check] OK" in capsys.readouterr().out
