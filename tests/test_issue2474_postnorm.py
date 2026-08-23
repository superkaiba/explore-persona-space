"""Pins for the issue-2474 ``postnorm-l27-diagnostic`` driver (offline math).

Pins (CPU-only, no network, no model download):
  * ``rms_norm_rows`` reproduces the REAL ``Qwen2RMSNorm`` module semantics
    (fp32 variance, eps inside the sqrt, per-dim weight) — the single post-norm
    operator every phase shares.
  * ``decode_bf16_le`` round-trips torch bf16 bytes exactly (bf16 = upper 16
    bits of fp32).
  * ``build_comparison_figure`` survives a deliberately INVERTED bootstrap CI
    (quantile CIs can invert around the point estimate at tiny n — the
    ``xerr/yerr`` non-negative-offsets gotcha) routed through the REAL figure
    function to ``savefig``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import issue2474_postnorm as pn


def test_rms_norm_rows_matches_qwen2_rmsnorm_reference():
    torch = pytest.importorskip("torch")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    rng = np.random.default_rng(0)
    x = rng.normal(0, 3.0, size=(5, 16)).astype(np.float32)
    w = rng.normal(1.0, 0.5, size=16).astype(np.float32)
    eps = 1e-6
    ref_mod = Qwen2RMSNorm(16, eps=eps)
    with torch.no_grad():
        ref_mod.weight.copy_(torch.from_numpy(w))
        ref = ref_mod(torch.from_numpy(x)).numpy()
    mine = pn.rms_norm_rows(x, w, eps)
    assert mine.shape == ref.shape
    np.testing.assert_allclose(mine, ref, atol=1e-5, rtol=1e-5)


def test_rms_norm_rows_is_row_wise_nonlinear():
    """mean-of-normed != norm-of-mean — the grain distinction the round hinges on."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.0, size=(8, 16))
    x[0] *= 10.0  # one large-norm row makes the two grains diverge
    w = np.ones(16)
    mean_of_norm = pn.rms_norm_rows(x, w, 1e-6).mean(axis=0)
    norm_of_mean = pn.rms_norm_rows(x.mean(axis=0), w, 1e-6)
    assert not np.allclose(mean_of_norm, norm_of_mean, atol=1e-3)


def test_decode_bf16_le_roundtrip():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    vals32 = rng.normal(0, 4.0, size=64).astype(np.float32)
    bf = torch.from_numpy(vals32).to(torch.bfloat16)
    raw = bf.view(torch.uint16).numpy().astype("<u2").tobytes()
    decoded = pn.decode_bf16_le(raw)
    np.testing.assert_array_equal(decoded, bf.to(torch.float32).numpy())


def _fig_stats_payload(*, invert_ci: bool) -> dict:
    conds = ["lang_a", "lang_b"]
    fams = {}
    for fi, fam in enumerate(pn.TRAINREF_FAMS):
        grains = {}
        for gi, grain in enumerate(("pre", "post_rowgrain")):
            point = 0.4 + 0.05 * fi - 0.1 * gi
            ci = [point + 0.2, point - 0.2] if invert_ci else [point - 0.2, point + 0.2]
            grains[grain] = {
                "pooled_rho": point,
                "pooled_ci95": ci,
                "per_condition": {c: {"rho": point + 0.03 * k} for k, c in enumerate(conds)},
            }
        fams[fam] = {"level": grains}
    return {
        "layer": 27,
        "settings": {"caps": {"conds": conds, "variants": {"full": {"families": fams}}}},
    }


@pytest.mark.parametrize("invert_ci", [False, True])
def test_comparison_figure_survives_inverted_ci(tmp_path, invert_ci):
    paths = pn.build_comparison_figure(
        _fig_stats_payload(invert_ci=invert_ci), tmp_path, setting="caps"
    )
    png = [Path(p) for p in paths.values() if str(p).endswith(".png")]
    assert png and png[0].is_file() and png[0].stat().st_size > 5_000


def test_err_offsets_clamps_non_negative():
    lo, hi = pn._err_offsets(0.5, [0.7, 0.3])  # inverted CI
    assert lo == 0.0 and hi == 0.0
    lo, hi = pn._err_offsets(0.5, [0.3, 0.7])
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
