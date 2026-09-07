"""Numerical and failure-path checks for the observed-answer SAE control."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue1482_answer_property_ceiling as C


def test_sparse_centering_matches_sklearn_multioutput_ridge():
    rng = np.random.default_rng(1482)
    x = rng.normal(size=(95, 17)) + rng.normal(size=17)
    y = np.maximum(0, x @ rng.normal(size=(17, 9)) - 2)
    y[y < 1] = 0
    train, test = np.arange(70), np.arange(70, 95)
    mean, scale = x[train].mean(0), x[train].std(0, ddof=1) + 1e-9
    xt = (x[train] - mean) / scale
    xe = (x[test] - mean) / scale
    s, u = np.linalg.eigh(xt.T @ xt)
    coeff, ymu = C.sparse_coefficients(sp.csr_matrix(y[train]), xt, u, xt.sum(0))
    for lam in [0.1, 100, 10000]:
        got = ((xe @ u) / (s + lam)) @ coeff + ymu
        expected = Ridge(alpha=lam).fit(xt, y[train]).predict(xe)
        np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_score_is_exact_saved_prediction_r2_and_constants_are_undefined():
    rng = np.random.default_rng(11)
    y = rng.normal(size=(70, 4))
    y[:, 3] = 1
    p = (y + rng.normal(scale=0.3, size=y.shape)).astype(np.float32)
    scores = C.score_arrays(y, p)
    np.testing.assert_allclose(
        scores["r2"][:3], r2_score(y[:, :3], p[:, :3], multioutput="raw_values")
    )
    assert np.isnan(scores["r2"][3])
    assert not scores["defined"][3]


@pytest.mark.parametrize("side", ["true", "prediction"])
def test_nonfinite_inputs_raise(side):
    y, p = np.ones((4, 2)), np.ones((4, 2))
    (y if side == "true" else p)[1, 0] = np.nan
    with pytest.raises(ValueError, match="invalid readout"):
        C.score_arrays(y, p)


def test_sparse_row_mismatch_raises():
    with pytest.raises(ValueError, match="row mismatch"):
        C.sparse_coefficients(
            sp.csr_matrix(np.ones((4, 2))), np.ones((5, 2)), np.eye(2), np.ones(2)
        )


def test_atomic_checkpoint_is_reopenable(tmp_path):
    x = np.arange(35).reshape(5, 7)
    C.atomic_npz(tmp_path / "checkpoint.npz", x=x)
    C.atomic_npy(tmp_path / "predictions.npy", x)
    np.testing.assert_array_equal(np.load(tmp_path / "checkpoint.npz")["x"], x)
    np.testing.assert_array_equal(np.load(tmp_path / "predictions.npy"), x)


@pytest.mark.parametrize("payload", ["prediction", "metrics", "weights"])
def test_modified_and_missing_checkpoint_payload_rejected(
    tmp_path, payload, monkeypatch
):
    monkeypatch.setattr(C, "DIMENSION", 3)
    p, m, w = (
        tmp_path / "pred.npy",
        tmp_path / "metrics.npz",
        tmp_path / "weights.npz",
    )
    C.atomic_npy(p, np.ones((5, 2), np.float32))
    C.atomic_npz(
        m, r2=np.ones(2), ss_tot=np.ones(2), ss_res=np.ones(2), defined=np.ones(2)
    )
    C.atomic_npz(w, coefficient=np.ones((3, 2)), target_mean=np.ones(2))
    identity = {"protocol": "fixture"}
    prediction = {
        "identity": identity,
        "lambda": 1.0,
        "prediction_sha256": C.digest(p),
        "metrics_sha256": C.digest(m),
    }
    fitted = {"identity": identity, "c0": 0, "c1": 2, "weights_sha256": C.digest(w)}
    C.validate_prediction_checkpoint(prediction, identity, 1.0, p, m, (5, 2))
    C.validate_fit_checkpoint(fitted, identity, w, 0, 2)
    target = {"prediction": p, "metrics": m, "weights": w}[payload]
    target.write_bytes(target.read_bytes() + b"changed")

    def verify():
        if payload == "weights":
            C.validate_fit_checkpoint(fitted, identity, w, 0, 2)
        else:
            C.validate_prediction_checkpoint(prediction, identity, 1.0, p, m, (5, 2))

    with pytest.raises(ValueError, match="missing or changed"):
        verify()
    target.unlink()
    with pytest.raises(ValueError, match="missing or changed"):
        verify()
