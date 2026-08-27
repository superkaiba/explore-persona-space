"""Unit tests for scripts/issue2569_operator.py — the #2569 row-action operator module.

Covers the vendored B6 payload contract, the (A, b) affine form, the B1
orientation dictionary, and the three B1 driver identity asserts. Synthetic
tests always run; tests against the real banked L19 payload skip when the
artifact is not staged locally (``data/`` is gitignored — set
``EPS2569_MAP_ROOT`` to a checkout that carries it, or run at the repo root).

The v3 orientation error this pins (plan #2569 blocker B1): singular READ
directions are the LEFT singular vectors under the row action ``x @ A``; the
column-form misread ``A v_i ~= sigma_i v_i`` fails at relative error ~1.35 on
the real L19 map while the row form holds to ~1.3e-15.
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

import issue2569_operator as OP  # noqa: E402

L19_PATH = OP.banked_map_path(19)
needs_l19 = pytest.mark.skipif(
    not L19_PATH.exists(),
    reason=f"banked L19 ridge.pt not staged at {L19_PATH} (set EPS2569_MAP_ROOT)",
)

D_SYN = 48  # synthetic residual dim — small, non-trivial, distinct-valued components


def _synth_raw_payload(d: int = D_SYN, layer: int = 19, seed: int = 7) -> dict:
    """A synthetic raw torch payload in the exact vendored contract shape.

    Distinct-valued ``xsd`` (log-normal, well away from 1) so a row-vs-column
    standardization-scaling bug cannot cancel; W random non-symmetric.
    """
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


def _payload_from_raw(raw: dict, path: Path = Path("<synthetic>")) -> OP.MapPayload:
    """Build a MapPayload from a raw dict without touching disk."""
    return OP.MapPayload(
        layer=int(raw["layer"]),
        path=path,
        W=np.asarray(raw["W"], dtype=np.float64),
        xmu=np.asarray(raw["xmu"], dtype=np.float64),
        xsd=np.asarray(raw["xsd"], dtype=np.float64),
        ymu=np.asarray(raw["ymu"], dtype=np.float64),
        selected_lambda=float(raw["selected_lambda"]),
        raw=raw,
    )


def _write_payload(raw: dict, root: Path) -> Path:
    """Persist a raw payload at the banked relative path under ``root``."""
    path = OP.banked_map_path(int(raw["layer"]), root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, path)
    return path


# --------------------------------------------------------------------------------------
# Vendored payload contract (B6)
# --------------------------------------------------------------------------------------


def test_loader_roundtrip(tmp_path):
    """load_banked_map returns fp64 components + metadata from a contract-shaped file."""
    raw = _synth_raw_payload()
    _write_payload(raw, tmp_path)
    p = OP.load_banked_map(19, root=tmp_path)
    assert p.layer == 19
    assert p.d == D_SYN
    assert p.W.dtype == np.float64 and p.W.shape == (D_SYN, D_SYN)
    for k in ("xmu", "xsd", "ymu"):
        arr = getattr(p, k)
        assert arr.dtype == np.float64 and arr.shape == (D_SYN,)
    assert p.selected_lambda == pytest.approx(0.001)
    np.testing.assert_allclose(p.W, np.asarray(raw["W"], dtype=np.float64))
    # identity+bias offset is the vendored ymu - xmu
    np.testing.assert_allclose(OP.identity_bias_offset(p), p.ymu - p.xmu)


def test_loader_missing_file_raises(tmp_path):
    """A missing payload fails loud with the staging hint, never a silent default."""
    with pytest.raises(FileNotFoundError, match="banked n1m ridge absent"):
        OP.load_banked_map(19, root=tmp_path)


def test_loader_rejects_wrong_kind(tmp_path):
    """A non-ridge payload (e.g. the banked mlp/krr siblings) is rejected at load."""
    raw = _synth_raw_payload()
    raw["kind"] = "mlp"
    _write_payload(raw, tmp_path)
    with pytest.raises(RuntimeError, match="expected the ridge fitter"):
        OP.load_banked_map(19, root=tmp_path)


def test_loader_rejects_layer_mismatch(tmp_path):
    """A payload whose recorded layer differs from the request is rejected."""
    raw = _synth_raw_payload(layer=19)
    raw["layer"] = 14
    _write_payload({**raw, "layer": 14}, tmp_path)
    # file sits at the L19 path (written under layer key 19) but claims L14
    path = OP.banked_map_path(19, root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, path)
    with pytest.raises(RuntimeError, match="payload layer"):
        OP.load_banked_map(19, root=tmp_path)


def test_loader_rejects_missing_key_and_bad_shapes(tmp_path):
    """Missing contract keys, non-square W, and non-positive xsd all fail loud."""
    raw = _synth_raw_payload()
    del raw["ymu"]
    _write_payload(raw, tmp_path)
    with pytest.raises(RuntimeError, match="missing contract keys"):
        OP.load_banked_map(19, root=tmp_path)

    raw = _synth_raw_payload()
    raw["W"] = raw["W"][: D_SYN // 2]
    _write_payload(raw, tmp_path)
    with pytest.raises(RuntimeError, match="not square"):
        OP.load_banked_map(19, root=tmp_path)

    raw = _synth_raw_payload()
    raw["xsd"][0] = 0.0
    _write_payload(raw, tmp_path)
    with pytest.raises(RuntimeError, match="strictly positive"):
        OP.load_banked_map(19, root=tmp_path)


def test_map_root_env_override(tmp_path, monkeypatch):
    """EPS2569_MAP_ROOT redirects the default root; an explicit root arg wins over it."""
    monkeypatch.setenv("EPS2569_MAP_ROOT", str(tmp_path))
    assert OP.banked_map_path(19) == tmp_path / OP.BANKED_MAP_RELPATH.format(layer=19)
    other = tmp_path / "elsewhere"
    assert OP.banked_map_path(19, root=other).is_relative_to(other)


# --------------------------------------------------------------------------------------
# Affine form + registered path (B1)
# --------------------------------------------------------------------------------------


def test_row_operator_matches_registered_path():
    """v @ A + b equals the registered ((v - xmu)/xsd) @ W + ymu, and A scales ROWS."""
    p = _payload_from_raw(_synth_raw_payload())
    A, b = OP.row_operator(p)
    np.testing.assert_allclose(A, np.diag(1.0 / p.xsd) @ p.W, rtol=1e-13)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, D_SYN)) * 5.0
    np.testing.assert_allclose(X @ A + b, OP.predict(p, X), rtol=1e-11, atol=1e-11)
    # single-vector shape passthrough
    v = X[0]
    np.testing.assert_allclose(v @ A + b, OP.predict(p, v), rtol=1e-11, atol=1e-11)


def test_prediction_difference_equals_mapped_displacement():
    """predict(v1) - predict(v2) == (v1 - v2) @ A: affine terms cancel in differences."""
    p = _payload_from_raw(_synth_raw_payload())
    A, _ = OP.row_operator(p)
    rng = np.random.default_rng(1)
    v1 = rng.standard_normal((8, D_SYN))
    v2 = rng.standard_normal((8, D_SYN))
    np.testing.assert_allclose(
        OP.prediction_difference(p, v1, v2),
        OP.mapped_displacement(v1 - v2, A),
        rtol=1e-10,
        atol=1e-10,
    )


# --------------------------------------------------------------------------------------
# Orientation dictionary (B1)
# --------------------------------------------------------------------------------------


def test_gram_assert_and_context_similarity():
    """Assert (i) passes, and context_similarity equals c (A A^T) c'^T explicitly."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((D_SYN, D_SYN))
    stats = OP.assert_row_action_gram(A, n_probes=64, seed=3)
    assert stats["max_rel_err"] < 1e-8
    c = rng.standard_normal((4, D_SYN))
    c2 = rng.standard_normal((6, D_SYN))
    np.testing.assert_allclose(
        OP.context_similarity(A, c, c2), c @ OP.through_map_gram(A) @ c2.T, rtol=1e-9
    )


def test_singular_orientation_row_form_passes_column_form_fails():
    """The B1 row identity holds to fp64 while the v3 column misread fails large.

    This is the committed test that would have caught the v3 error: on a random
    (non-symmetric) A the column form A v_i ~= sigma_i v_i has O(1) relative
    error, while u_i @ A = sigma_i v_i holds to machine precision.
    """
    rng = np.random.default_rng(4)
    A = rng.standard_normal((D_SYN, D_SYN))
    stats = OP.assert_singular_orientation(A, k=8)
    assert stats["max_row_form_rel_err"] < 1e-6
    assert min(stats["wrong_column_form_rel_err"]) > 0.1
    # descending sigma ordering
    sig = stats["sigma"]
    assert sig == sorted(sig, reverse=True)


def test_singular_triplets_identity_directly():
    """u_i @ A == sigma_i v_i per returned triplet (field-name orientation pin)."""
    rng = np.random.default_rng(5)
    A = rng.standard_normal((D_SYN, D_SYN))
    trip = OP.top_singular_triplets(A, k=4)
    for i in range(4):
        np.testing.assert_allclose(
            trip.read_input_u[:, i] @ A,
            trip.sigma[i] * trip.write_output_v[:, i],
            rtol=1e-8,
            atol=1e-10,
        )


def test_eigen_read_write_biorthogonal_expansion():
    """Eigen orientation: read along RIGHT eigenvectors, write along LEFT (letter flip)."""
    rng = np.random.default_rng(6)
    A = rng.standard_normal((24, 24))
    pairs = OP.eigen_read_write(A)
    # biorthonormality: rows of inv(V_right) against columns of V_right
    np.testing.assert_allclose(pairs.write_left_rows @ pairs.read_right_v, np.eye(24), atol=1e-9)
    # reconstruction A = V diag(lam) inv(V)
    recon = pairs.read_right_v @ np.diag(pairs.lam) @ pairs.write_left_rows
    np.testing.assert_allclose(recon.real, A, atol=1e-9)
    assert np.abs(recon.imag).max() < 1e-9
    # row action x @ A = sum_i lam_i (x . v_i^e) u_i^e^T
    x = rng.standard_normal(24)
    expansion = sum(
        pairs.lam[i] * (x @ pairs.read_right_v[:, i]) * pairs.write_left_rows[i] for i in range(24)
    )
    np.testing.assert_allclose(np.asarray(expansion).real, x @ A, atol=1e-8)
    assert pairs.spectral_radius == pytest.approx(np.abs(pairs.lam).max())


def test_fixed_point_row_convention():
    """fixed_point solves x* (I - A) = b, and row-action iteration converges to it."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((D_SYN, D_SYN))
    A *= 0.5 / OP.spectral_radius(A)  # contraction
    b = rng.standard_normal(D_SYN)
    x = OP.fixed_point(A, b)
    np.testing.assert_allclose(x @ (np.eye(D_SYN) - A), b, rtol=1e-9, atol=1e-9)
    v = np.zeros(D_SYN)
    for _ in range(200):
        v = v @ A + b
    np.testing.assert_allclose(v, x, rtol=1e-7, atol=1e-7)


def test_monitor_gradient_is_row_space_gradient():
    """A @ r is the gradient of r . (v @ A + b) w.r.t. v (finite-difference check)."""
    p = _payload_from_raw(_synth_raw_payload())
    A, b = OP.row_operator(p)
    rng = np.random.default_rng(8)
    r = rng.standard_normal(D_SYN)
    g = OP.monitor_gradient(A, r)
    v = rng.standard_normal(D_SYN)
    for j in (0, D_SYN // 2, D_SYN - 1):
        e = np.zeros(D_SYN)
        e[j] = 1.0
        np.testing.assert_allclose(((v + e) @ A + b) @ r - (v @ A + b) @ r, g[j], rtol=1e-6)


def test_wiring_in_edges_matches_explicit_product():
    """E_f @ A.T @ D matches the explicit dense product, vector and batch shapes."""
    rng = np.random.default_rng(9)
    A = rng.standard_normal((D_SYN, D_SYN))
    D = rng.standard_normal((D_SYN, 12))
    e1 = rng.standard_normal(D_SYN)
    np.testing.assert_allclose(OP.wiring_in_edges(e1, A, D), e1 @ A.T @ D, rtol=1e-10)
    E = rng.standard_normal((3, D_SYN))
    out = OP.wiring_in_edges(E, A, D)
    assert out.shape == (3, 12)
    np.testing.assert_allclose(out, E @ A.T @ D, rtol=1e-10)


def test_tau_kernel_threshold_and_kernel_directions():
    """tau_kernel = the sigma at 99% cumulative sigma^2 mass; kernel = columns below it."""
    s = np.array([10.0, 3.0, 1.0, 0.1, 0.01])
    # cumulative sigma^2 mass: 0.9083, 0.9900, 0.99908, ... -> rank 2 at mass 0.99
    tau, rank = OP.tau_kernel_threshold(s, mass=0.99)
    assert rank == 2
    assert tau == pytest.approx(3.0)
    U = np.eye(5)
    kern = OP.kernel_read_directions(U, s, tau)
    assert kern.shape == (5, 3)  # the three columns with sigma < 3.0
    np.testing.assert_allclose(kern, U[:, 2:])
    with pytest.raises(ValueError):
        OP.tau_kernel_threshold(np.array([]))


def test_apply_map_parity_synthetic():
    """B1 assert (iii) mechanics: vendored predict == issue779 apply_map on synthetic W.

    Imports the main-resident apply_map reference (torch + the #779 sibling
    chain, ~20 s) — the same deferred path the P-A/P-B entry gate pays.
    """
    raw = _synth_raw_payload()
    p = _payload_from_raw(raw)
    stats = OP.assert_prediction_matches_apply_map(p, n_probes=8, seed=11)
    assert stats["max_rel_diff"] < 1e-9


# --------------------------------------------------------------------------------------
# Real banked L19 payload (skip when data/ not staged; the B1 asserts at production shape)
# --------------------------------------------------------------------------------------


@needs_l19
def test_real_l19_payload_contract():
    """The real L19 payload satisfies the vendored contract at production shape."""
    p = OP.load_banked_map(19)
    assert p.d == OP.D_MODEL
    assert p.selected_lambda == pytest.approx(1e-3)
    assert (p.xsd > 0).all()
    assert sorted(p.raw.keys()) == [
        "W",
        "fitter",
        "kind",
        "layer",
        "selected_lambda",
        "xmu",
        "xsd",
        "ymu",
    ]


@needs_l19
def test_real_l19_b1_identity_asserts():
    """All three B1 identity asserts on the real L19 map; the v3 misread fails large.

    Pins the measured facts from the plan round: top-3 singular values
    7.9618/6.5753/4.9309; row identity ~1e-15-scale; column-form misread ~1.35
    (= sqrt(2 - 2 cos(u_i, v_i)) at |cos| ~ 0.084-0.136 — W is far from normal).
    """
    p = OP.load_banked_map(19)
    stats = OP.run_driver_identity_asserts(p, n_probes=64, k=8, seed=0)
    assert stats["gram"]["max_rel_err"] < 1e-8
    assert stats["singular_orientation"]["max_row_form_rel_err"] < 1e-6
    np.testing.assert_allclose(
        stats["singular_orientation"]["sigma"][:3], [7.9618, 6.5753, 4.9309], atol=2e-3
    )
    assert min(stats["singular_orientation"]["wrong_column_form_rel_err"]) > 1.0
    assert stats["apply_path"]["max_rel_diff"] < 1e-9
