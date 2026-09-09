"""Numerical controls and duplicate-fit protection for the fixed-layer analysis."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2588_chat_rank_matched as matched


def test_shared_spectrum_diversity_matches_direct_svd():
    rng = np.random.default_rng(24)
    x, y = rng.normal(size=(2, 100, 7)).astype(np.float32)
    payload = matched.rank.reconstruct(x[:60], y[:60], x[60:80], y[60:80], x[80:], y[80:], 2)
    reduced = matched.rank.reduced_rank(payload, x[:60], include_spectrum=True)
    actual = matched.representation_measures(
        x[:60], y[:60], payload, reduced["fitted_output_spectrum"]["eigenvalues"]
    )
    xn = (x[:60].astype(float) - payload["xmu"].astype(float)) / payload["xsd"].astype(float)
    matrices = {
        "input": xn - xn.mean(0),
        "answer": y[:60].astype(float) - y[:60].astype(float).mean(0),
        "fitted_output": xn @ payload["W"].astype(float),
    }
    for key, matrix in matrices.items():
        values = np.linalg.svd(matrix, compute_uv=False) ** 2 / len(matrix)
        np.testing.assert_allclose(actual["spectra"][key], values, atol=1e-12)
        expected = matched.diversity.measures(values)
        for stat, value in expected.items():
            assert actual[key][stat] == pytest.approx(value, rel=1e-11)


@pytest.mark.parametrize("filename", ["result.json", "rank_b_l24.json"])
def test_duplicate_or_partial_fit_refused_before_loading(tmp_path, filename):
    (tmp_path / filename).write_text("{}")
    with pytest.raises(ValueError, match="rather than duplicating"):
        matched.analyze(tmp_path / "missing", tmp_path)
