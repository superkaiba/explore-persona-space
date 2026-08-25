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


def _tiny_ctx_sae(d: int, n_feat: int = 64, seed: int = 51):
    """A REAL issue1482_sae.BatchTopKSAE at tiny width (production ctor + asserts).

    Signature-conformant BY CONSTRUCTION: the production class itself, built
    through its own state-dict ctor (EXPECTED_KEYS + shape/threshold asserts all
    execute) — only the HF fetch (`.load`) is bypassed.
    """
    import issue1482_sae as S1482

    rng = np.random.default_rng(seed)
    sd = {
        "encoder.weight": torch.tensor(rng.standard_normal((n_feat, d)), dtype=torch.float32),
        "encoder.bias": torch.tensor(rng.standard_normal(n_feat) * 0.1, dtype=torch.float32),
        "decoder.weight": torch.tensor(rng.standard_normal((d, n_feat)), dtype=torch.float32),
        "b_dec": torch.tensor(rng.standard_normal(d) * 0.1, dtype=torch.float32),
        "threshold": torch.tensor(0.05),
        "k": torch.tensor(8),
    }
    return S1482.BatchTopKSAE(sd, k=8, act_dim=d, dict_size=n_feat)


def _tiny_ans_bundle(tmp_path: Path, d: int, n_feat: int = 64, seed: int = 52) -> Path:
    """A REAL #2476 MatryoshkaBatchTopKSAE bundle dir (cfg.json + safetensors).

    Built + saved through the production class's own ctor/save_dir, then served
    to the driver via the ``--answer-sae-dir`` seam — the SAME consume path
    (RB._stage_answer_sae -> T24.load_local) production takes.
    """
    import issue2476_turnavg_sae as T24

    sae = T24.MatryoshkaBatchTopKSAE(
        act_dim=d, dict_size=n_feat, k=4, tier_bounds=(8, 32, n_feat), seed=seed
    )
    with torch.no_grad():
        sae.threshold.fill_(0.01)  # live inference gating (never the untrained 0.0)
    bundle = tmp_path / "ans_sae"
    bundle.mkdir(parents=True, exist_ok=True)
    sae.save_dir(bundle)
    return bundle


def _tiny_alive_npz(tmp_path: Path, n_feat: int = 64, seed: int = 53) -> Path:
    """A tiny alive_c.npz with the PROBED banked keys (counts/n_fit_rows/alive_ids)."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 5, size=n_feat).astype(np.int64)
    counts[:4] = 0  # some dead features so the union is a proper subset
    p = tmp_path / "alive_c.npz"
    with open(p, "wb") as fh:
        np.savez(
            fh,
            counts=counts,
            n_fit_rows=np.int64(100),
            alive_ids=np.flatnonzero(counts >= 3).astype(np.int64),
        )
    return p


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
    # part-3 dictionary seams: real tiny answer bundle + alive npz through the
    # CLI seams; the ctx SAE (HF-fetch-only loader) is patched at the network
    # boundary with a REAL production-class instance.
    monkeypatch.setattr(WB, "_load_ctx_sae", lambda args: _tiny_ctx_sae(d))
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
        "--answer-sae-dir",
        str(_tiny_ans_bundle(tmp_path, d)),
        "--alive-counts-npz",
        str(_tiny_alive_npz(tmp_path)),
    ]
    _run_driver(monkeypatch, tmp_path, argv)

    leg1, leg3, leg8 = out / "leg1", out / "leg3", out / "leg8"
    for name in (
        "entry_asserts_L19.json",
        "factor_L19.pt",
        "factor_L19.json",
        "anatomy_L19.json",
        "alpha_lowrank_L19.json",
        "fixed_point_L19.pt",
        "fixed_point_L19.json",
        "sae_dashboards_L19.json",
    ):
        assert (leg1 / name).exists(), name
    for name in ("receipts_L19.json", "wiring_L19.json", "wiring_edges_L19.npz"):
        assert (leg3 / name).exists(), name
    for name in (
        "effective_kernel_L19.pt",
        "effective_kernel_L19.json",
        "monitor_geometry_L19.pt",
        "monitor_geometry_L19.json",
        "certificates_L19.json",
        "monitor_sae_naming_L19.json",
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
    # leg 1 step 4/6: the dashboards phase FILLED sae_decode in place (a dict now)
    assert isinstance(fp["sae_decode"], dict)
    assert fp["sae_decode"]["filled_by"] == "sae-dashboards phase"
    assert fp["sae_decode"]["n_fired"] == len(fp["sae_decode"]["top_fired"]) or (
        fp["sae_decode"]["n_fired"] > 32 and len(fp["sae_decode"]["top_fired"]) == 32
    )
    assert fp["nearest_banked_answers"].startswith("deferred")

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
    assert mg["sae_naming"].startswith("leg8/monitor_sae_naming_")
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

    # leg 3 step 3: receipts — per-trait top features, schema + caveat
    rec = json.loads((leg3 / "receipts_L19.json").read_text())
    assert set(rec["traits"]) == {"evil", "sycophancy", "hallucination"}
    for t, row in rec["traits"].items():
        assert len(row["top_positive"]) == 16 and len(row["top_negative"]) == 16, t  # smoke k
        assert row["top_positive"][0]["score"] >= row["top_positive"][-1]["score"]
        assert row["top_negative"][0]["score"] <= row["top_positive"][0]["score"]
        assert all(abs(r0["cos"]) <= 1.0 + 1e-9 for r0 in row["top_positive"])
        assert row["grad_norm"] > 0.0
    assert WB.CAVEAT_WIRING in rec["caveats"] and WB.CAVEAT_ACTIVATION in rec["caveats"]
    assert "sources" in rec["label_sources"]

    # leg 3 steps 1-2: wiring — union + rb-nearest edges, H3 fields, out-edges
    wir = json.loads((leg3 / "wiring_L19.json").read_text())
    edges_npz = np.load(leg3 / "wiring_edges_L19.npz")
    F = int(wir["n_answer_features"])
    assert edges_npz["feat_ids"].shape == (F,) and edges_npz["is_rb_nearest"].sum() >= 1
    assert edges_npz["top_edge_ids"].shape == edges_npz["top_edge_vals"].shape
    assert edges_npz["conc_curve"].shape == (F, len(WB.CONC_K_GRID))
    share = edges_npz["top32_absmass_share"]
    assert share.shape == (F,) and (share >= 0).all() and (share <= 1.0 + 1e-6).all()
    for t, row in wir["h3"]["behavior_relevant"].items():
        assert 0.0 <= row["top32_share_full"] <= 1.0 + 1e-6, t
        assert row["top32_share_alive"] is None  # no --rows-dir on this run
        assert int(row["feat_id"]) in edges_npz["feat_ids"]
    assert isinstance(wir["ctx_alive"], str) and wir["ctx_alive"].startswith("deferred")
    assert "INFORMATIONAL" in wir["h3"]["grain"]
    assert wir["h3"]["union_top32_share_alive"].startswith("deferred")
    assert len(wir["out_edges"]) > 0
    first_oe = next(iter(wir["out_edges"].values()))
    assert {"n_fired", "fired", "linear_top", "mapped_norm"} <= set(first_oe)
    assert wir["regime"]["wiring"]["rows_attached"] is False
    assert wir["regime"]["wiring"]["ans_dict"]["union_frac"] == 0.002

    # leg 3 step 4: attribution — rows-gated, explicit deferral on this run
    attr = json.loads((leg3 / "attribution_L19.json").read_text())
    assert isinstance(attr["examples"], str) and attr["examples"].startswith("deferred")

    # leg 1 step 4: two-sided dashboards — 4 sections, nulls, encoder companions
    dash = json.loads((leg1 / "sae_dashboards_L19.json").read_text())
    assert set(dash["sections"]) == {
        "singular_read",
        "singular_write",
        "eigen_read",
        "eigen_write",
    }
    for side, sec in dash["sections"].items():
        assert len(sec["directions"]) == 8, side  # smoke top_k
        for row in sec["directions"]:
            assert abs(row["max_abs_cos"]) <= 1.0 + 1e-6
            assert len(row["top_features"]) == WB.DASH_FEATURES_PER_DIRECTION
        if side.endswith("_write"):
            assert "encoder_pass" in sec["directions"][0]
            assert "linear_top" in sec["directions"][0]
        else:
            assert "encoder_pass" not in sec["directions"][0]
    for j, row in enumerate(dash["sections"]["eigen_read"]["directions"]):
        assert 0.0 <= row["im_frac"] <= 1.0 + 1e-6, j
    nulls = dash["null_floors"]
    for side in ("ctx", "ans"):
        emp = nulls[side]["empirical"]
        assert emp["n_draws"] == 100  # smoke draws
        assert 0.0 < emp["p50"] <= emp["p95"] <= emp["max"] <= 1.0 + 1e-6
        assert nulls[side]["analytic_sqrt_2lnN_over_d"] > 0.0
    assert dash["whitened_cosine"].startswith("deferred-to-P-B")
    assert dash["fixed_point_decode"] == fp["sae_decode"]

    # leg 8 naming: gradient/pre-image vs ctx dict + r_hat vs answer dict
    naming = json.loads((leg8 / "monitor_sae_naming_L19.json").read_text())
    for t, entry in naming["traits"].items():
        for key in ("gradient", "preimage_unit_level", "preimage_unit_level_fullpinv"):
            assert 0.0 <= entry[key]["max_abs_cos"] <= 1.0 + 1e-6, (t, key)
            assert isinstance(entry[key]["exceeds_empirical_p95"], bool)
        assert abs(entry["r_hat_vs_answer_dict"]["cos"]) <= 1.0 + 1e-6

    # Resume: identical regime -> every phase unit SKIPs (factor .pt untouched).
    before = os.stat(leg1 / "factor_L19.pt").st_mtime_ns
    before_mg = os.stat(leg8 / "monitor_geometry_L19.pt").st_mtime_ns
    before_dash = os.stat(leg1 / "sae_dashboards_L19.json").st_mtime_ns
    before_wire = os.stat(leg3 / "wiring_edges_L19.npz").st_mtime_ns
    _run_driver(monkeypatch, tmp_path, argv)
    assert os.stat(leg1 / "factor_L19.pt").st_mtime_ns == before
    assert os.stat(leg8 / "monitor_geometry_L19.pt").st_mtime_ns == before_mg
    assert os.stat(leg1 / "sae_dashboards_L19.json").st_mtime_ns == before_dash
    assert os.stat(leg3 / "wiring_edges_L19.npz").st_mtime_ns == before_wire

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


# ── part-3 pure-core tests (leg 3 + dashboards; no disk, no network) ──────────────


def test_receipts_scores_match_serial_oracle():
    """receipts_trait_scores == the serial per-column d_c . (A r) oracle."""
    rng = np.random.default_rng(61)
    d, n = 16, 40
    D = rng.standard_normal((d, n))
    A = rng.standard_normal((d, d))
    r = rng.standard_normal(d)
    r /= np.linalg.norm(r)
    col_norms = np.linalg.norm(D, axis=0)
    scores, cos = WB.receipts_trait_scores(D, col_norms, A, r)
    assert scores.shape == (n,) and cos.shape == (n,)
    g = A @ r
    for c in (0, 7, n - 1):  # serial oracle, a few columns
        assert np.isclose(scores[c], D[:, c] @ g, rtol=1e-4, atol=1e-6), c
    assert (np.abs(cos) <= 1.0 + 1e-6).all()
    assert np.allclose(cos * (np.linalg.norm(g) * col_norms), scores, rtol=1e-6)


def test_wiring_edge_stats_planted_column_and_oracle_equivalence():
    """Blocked GEMM chain == OP.wiring_in_edges oracle; planted top edge found."""
    rng = np.random.default_rng(62)
    d, n_ctx, m = 16, 50, 6
    E = rng.standard_normal((m, d))
    A = rng.standard_normal((d, d))
    D = rng.standard_normal((d, n_ctx))
    D[:, 3] = 100.0 * (A @ E[0])  # planted dominant in-edge for feature 0
    oracle = OP.wiring_in_edges(E, A, D)  # fp64 (m, n_ctx) — the serial reference
    blocked = (E @ A.T).astype(np.float32) @ D.astype(np.float32)  # the phase's chain
    assert np.allclose(blocked, oracle, rtol=1e-4, atol=1e-4)
    st = WB.wiring_edge_stats(oracle, top_k=5)
    assert st["top_ids"].shape == (m, 5) and st["top_vals"].shape == (m, 5)
    assert st["top_ids"][0, 0] == 3  # the planted column dominates feature 0
    assert np.isclose(st["top_vals"][0, 0], oracle[0, 3], rtol=1e-5)
    # concentration curve: monotone nondecreasing, saturating at 1 past N
    assert (np.diff(st["conc_curve"], axis=1) >= -1e-6).all()
    assert np.allclose(st["conc_curve"][:, -1], 1.0, atol=1e-6)  # k_grid max > n_ctx
    k32 = WB.CONC_K_GRID.index(32)
    assert np.allclose(st["conc_curve"][:, k32], st["top32_absmass_share"], atol=1e-6)
    assert (st["top32_absmass_share"] >= 0).all()
    assert (st["top32_absmass_share"] <= 1.0 + 1e-6).all()
    assert st["top32_absmass_share"][0] > 0.9  # planted column carries the mass


def test_attribution_decompose_closure_terms_and_encoder_guard():
    """Closure identity exact; contributions match the manual edge; the phase's
    encoder cross-check condition catches a transposed A (the B1 flip guard)."""
    rng = np.random.default_rng(63)
    d, n = 16, 32
    D = rng.standard_normal((d, n))
    b_dec_ctx = rng.standard_normal(d) * 0.1
    A = rng.standard_normal((d, d))
    b_map = rng.standard_normal(d) * 0.1
    e_f = rng.standard_normal(d)
    b_enc_f = 0.3
    b_dec_ans = rng.standard_normal(d) * 0.1
    a_ctx = np.zeros(n)
    a_ctx[[2, 9, 17]] = [1.5, 0.7, 2.1]  # sparse nonneg codes (SAE-shaped)
    v_c = D @ a_ctx + b_dec_ctx + 0.01 * rng.standard_normal(d)  # recon + residual
    dec = WB.attribution_decompose(
        v_c, a_ctx, D, b_dec_ctx, A, b_map, e_f, b_enc_f, b_dec_ans, top_m=5
    )
    manual_pre = float((v_c @ A + b_map - b_dec_ans) @ e_f) + b_enc_f
    assert np.isclose(dec["pre_act"], manual_pre, rtol=1e-12)
    assert dec["n_active_ctx"] == 3 and len(dec["contributions"]) == 3
    top = dec["contributions"][0]
    j = top["ctx_feat_id"]
    assert np.isclose(top["edge"], D[:, j] @ A @ e_f, rtol=1e-10)
    assert np.isclose(top["contribution"], a_ctx[j] * top["edge"], rtol=1e-10)
    assert dec["closure_residual"] <= 1e-6 * max(1.0, abs(manual_pre))
    # B1 flip: a transposed A closes the INTERNAL identity (self-consistent) but
    # moves pre_act away from the real encoder's value — the phase-level guard.
    dec_flip = WB.attribution_decompose(
        v_c, a_ctx, D, b_dec_ctx, A.T, b_map, e_f, b_enc_f, b_dec_ans, top_m=5
    )
    assert abs(dec_flip["pre_act"] - manual_pre) > 1e-3 * max(1.0, abs(manual_pre))


def test_top_dictionary_cosines_planted_and_null_floors():
    """Planted exact-match column found at cos ~ 1; null floors sane + seeded."""
    rng = np.random.default_rng(64)
    d, n = 24, 60
    D = rng.standard_normal((d, n))
    v = rng.standard_normal(d)
    D[:, 5] = 3.0 * v  # planted collinear column (norm handled by normalization)
    Dn, _norms = WB.normalize_dictionary_columns(D)
    assert np.allclose(np.linalg.norm(Dn, axis=0), 1.0, atol=1e-6)
    u = (v / np.linalg.norm(v))[None, :]
    ids, cos = WB.top_dictionary_cosines(u, Dn, 4)
    assert ids.shape == (1, 4) and ids[0, 0] == 5
    assert cos[0, 0] > 0.999
    assert (np.abs(cos[0, 1:]) <= abs(cos[0, 0])).all()  # sorted by |cos| desc
    floor = WB.analytic_max_cos_floor(n, d)
    assert np.isclose(floor, np.sqrt(2.0 * np.log(n) / d))
    null = WB.empirical_max_cos_null(Dn, n_draws=64, seed=7)
    assert 0.0 < null["p50"] <= null["p95"] <= null["max"] <= 1.0 + 1e-6
    assert null == WB.empirical_max_cos_null(Dn, n_draws=64, seed=7)  # seeded determinism


def test_load_ctx_feature_labels_ordering_negids_and_absent(tmp_path):
    """Later source overwrites on collision; negative ids skipped; absence LOUD."""
    root = tmp_path / "root"
    for rel, rows in (
        (
            WB.LABEL_SOURCES[0][0],  # issue1773 answer-side (loads FIRST)
            [
                {"feat_id": 7, "description": "answer-side seven", "describe_confidence": 0.9},
                {"feat_id": -200, "description": "aggregate axis row (skip)"},
                {"feat_id": 11, "description": "answer-side eleven"},
            ],
        ),
        (
            WB.LABEL_SOURCES[1][0],  # issue1482 context-side (loads LAST, overwrites)
            [{"feat_id": 7, "description": "context-side seven", "confidence": "high"}],
        ),
    ):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    labels, doc = WB.load_ctx_feature_labels(root)
    assert set(labels) == {7, 11}  # negative id skipped
    assert labels[7]["description"] == "context-side seven"
    assert labels[7]["evidence_side"] == "context"  # the plan-named tree wins
    assert labels[11]["evidence_side"] == "answer"
    assert [s["status"] for s in doc["sources"]] == ["loaded", "loaded"]
    # absent root: no labels, LOUD absent statuses, never a crash
    labels2, doc2 = WB.load_ctx_feature_labels(tmp_path / "empty")
    assert labels2 == {}
    assert [s["status"] for s in doc2["sources"]] == ["absent", "absent"]


# ── part-3 loader real-body tests (seam-stub rule: one per stubbed loader) ────────


def test_answer_union_real_body(tmp_path):
    """WB._answer_union executes the REAL RB staging + union bodies via the seam."""
    from types import SimpleNamespace

    npz = _tiny_alive_npz(tmp_path)
    args = SimpleNamespace(out_root=tmp_path / "out", alive_counts_npz=npz)
    union = WB._answer_union(args)
    z = np.load(npz)
    expected = np.flatnonzero(np.asarray(z["counts"]) >= 1)  # ceil(0.002 * 100) = 1
    assert np.array_equal(union, expected)
    assert np.isin(np.asarray(z["alive_ids"]), union).all()


def test_load_ans_sae_real_body(tmp_path):
    """WB._load_ans_sae executes RB._stage_answer_sae + T24.load_local for real."""
    from types import SimpleNamespace

    import issue2476_turnavg_sae as T24

    d = 32
    bundle = _tiny_ans_bundle(tmp_path, d)
    args = SimpleNamespace(out_root=tmp_path / "out", answer_sae_dir=bundle, device="cpu")
    sae = WB._load_ans_sae(args)
    assert isinstance(sae, T24.MatryoshkaBatchTopKSAE)
    assert sae.act_dim == d and sae.dict_size == 64
    codes = WB.encoder_pass(sae, np.random.default_rng(0).standard_normal((5, d)))
    assert codes.shape == (5, 64) and (codes >= 0).all()


def test_load_ctx_sae_real_body(tmp_path, monkeypatch):
    """WB._load_ctx_sae executes its real body; ONLY the HF fetch classmethod is
    faked, signature-conformantly (same params as issue1482_sae.BatchTopKSAE.load)."""
    from types import SimpleNamespace

    import issue1482_sae as S1482

    d = 32
    seen = {}

    def fake_load(cls, k=64, device="cpu", cache_dir=None, *, layer=19):
        seen.update(k=k, device=device, cache_dir=cache_dir, layer=layer)
        return _tiny_ctx_sae(d)

    monkeypatch.setattr(S1482.BatchTopKSAE, "load", classmethod(fake_load))
    args = SimpleNamespace(out_root=tmp_path / "out", device="cpu")
    sae = WB._load_ctx_sae(args)
    assert isinstance(sae, S1482.BatchTopKSAE) and sae.act_dim == d
    assert seen["k"] == WB.ANDY_SAE_K and seen["layer"] == WB.DICT_LAYER
    assert Path(seen["cache_dir"]).is_dir()  # the body created the stage dir


def test_attr_holdout_ids_real_body(tmp_path, monkeypatch):
    """WB._attr_holdout_ids executes its real body; T24._load_scratch_meta is
    faked signature-conformantly at the HF/scratch boundary."""
    from types import SimpleNamespace

    import issue2476_turnavg_sae as T24

    holdout = np.arange(500, 530, dtype=np.int64)

    def fake_meta(args):
        return np.zeros(10), np.zeros(10, np.uint8), {"holdout": holdout}

    monkeypatch.setattr(T24, "_load_scratch_meta", fake_meta)
    ids = WB._attr_holdout_ids(SimpleNamespace(out_root=tmp_path / "out"))
    assert np.array_equal(ids, holdout) and ids.dtype == np.int64


# ── part-3 rows-attached driver branch (wiring alive mask + attribution demo) ─────


def test_driver_leg3_rows_attached_branch(tmp_path, monkeypatch):
    """--rows-dir at L19 arms ctx-alive counts + alive-masked H3 + attribution.

    The P-B DATA boundary only is faked (planted fp16 store + a holdout id set;
    the pinned production split cannot exist at d=32); the phase bodies,
    ``_ctx_alive_counts``, the blocked GEMMs, and ``attribution_decompose`` all
    execute for real, including the encoder orientation cross-check.
    """
    d = 32
    map_root = tmp_path / "maproot"
    path = OP.banked_map_path(19, root=map_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_synth_raw_payload(d), path)
    out = tmp_path / "out"
    rb_dir = _synth_rb_dir(tmp_path, d)
    monkeypatch.setattr(WB, "_load_ctx_sae", lambda args: _tiny_ctx_sae(d))
    rng = np.random.default_rng(71)
    n_rows = 120
    rows_present = np.arange(1000, 1000 + n_rows, dtype=np.int64)
    x = rng.standard_normal((n_rows, d)).astype(np.float16)
    y = (x.astype(np.float64) @ (rng.standard_normal((d, d)) / np.sqrt(d))).astype(np.float16)
    fake_store = (x, y, np.arange(0, 80), np.arange(80, 100), np.arange(100, 120))
    monkeypatch.setattr(WB, "_load_rows_store", lambda args: fake_store)
    holdout = rows_present[rng.choice(n_rows, size=30, replace=False)]
    monkeypatch.setattr(WB, "_attr_holdout_ids", lambda args: np.sort(holdout))
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    np.save(rows_dir / "rows_present.npy", rows_present)
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
        "--answer-sae-dir",
        str(_tiny_ans_bundle(tmp_path, d)),
        "--alive-counts-npz",
        str(_tiny_alive_npz(tmp_path)),
    ]
    for phase in ("factor", "receipts", "wiring", "attribution"):
        _run_driver(monkeypatch, tmp_path, ["--phase", phase, *base])
    leg3 = out / "leg3"

    alive = np.load(leg3 / "ctx_alive_L19.npz")
    assert alive["counts"].shape == (64,)  # tiny ctx dict width
    assert int(alive["n_rows_used"]) == n_rows  # min(20k, 120)
    assert int(alive["floor"]) == 2  # ceil(0.01 * 120)
    assert alive["alive_ids"].size > 0
    assert (alive["counts"][alive["alive_ids"]] >= 2).all()

    wir = json.loads((leg3 / "wiring_L19.json").read_text())
    assert wir["regime"]["wiring"]["rows_attached"] is True
    assert isinstance(wir["ctx_alive"], dict) and wir["ctx_alive"]["n_alive"] > 0
    assert isinstance(wir["h3"]["union_top32_share_alive"], dict)
    for row in wir["h3"]["behavior_relevant"].values():
        assert row["top32_share_alive"] is not None
        assert 0.0 <= row["top32_share_alive"] <= 1.0 + 1e-6
    edges_npz = np.load(leg3 / "wiring_edges_L19.npz")
    assert "top_edge_ids_alive" in edges_npz.files
    # alive top ids are GLOBAL context ids drawn from the alive set
    assert np.isin(edges_npz["top_edge_ids_alive"].ravel(), alive["alive_ids"]).all()

    attr = json.loads((leg3 / "attribution_L19.json").read_text())
    assert isinstance(attr["examples"], list) and len(attr["examples"]) == 2  # smoke n
    assert attr["holdout"]["n_holdout_present"] == 30
    for ex in attr["examples"]:
        assert int(ex["row_id"]) in rows_present
        assert len(ex["features"]) >= 3  # 3 rb-nearest (+ top-pred dedup)
        for row in ex["features"].values():
            assert row["closure_residual"] <= 1e-6 * max(1.0, abs(row["pre_act"]))
            assert row["why_in_table"] in {
                "top predicted activation",
                "r_B-nearest (evil)",
                "r_B-nearest (sycophancy)",
                "r_B-nearest (hallucination)",
            }
            for c in row["contributions"]:
                assert {"ctx_feat_id", "a_j", "edge", "contribution", "label"} <= set(c)
