"""Regression test for run_952._extract_frozen (issue #952, round-1 inherited-code fix).

Pins the ``take_along_axis`` shape fix: the inherited mixed slice/fancy-indexing
form broadcast the per-group frozen-λ gather to (n, n, G) — a silent ~1.7 TB
allocation at production shape (n≈1000, G≈232). The fixed implementation must
return (n, G) per-context arrays gathered at each group's OWN frozen λ index
(round-2 review Minor 2; the in-path ``assert take.shape == (n, g)`` guard is
the runtime backstop, this test is the CI pin).
"""

from types import SimpleNamespace

import numpy as np

from explore_persona_space.experiments.issue_952.run_952 import _extract_frozen


def test_extract_frozen_shape_and_values():
    """(n, G, L) ss_res + (G,) lam_idx -> (n, G) gathered at each group's λ*."""
    rng = np.random.default_rng(0)
    n, g, n_lam = 5, 3, 13
    ssr = rng.random((n, g, n_lam)).astype(np.float32)
    sst = rng.random((n, g)).astype(np.float32)
    res = SimpleNamespace(ss_res={"test": ssr}, ss_tot={"test": sst})
    lam_idx = np.array([0, 5, 12], dtype=np.int64)
    out_ssr, out_sst = _extract_frozen(res, "test", lam_idx)
    # The pre-fix broadcast form produced shape (n, n, g) here — the bug class.
    assert out_ssr.shape == (n, g), out_ssr.shape
    assert out_sst.shape == (n, g), out_sst.shape
    for gi, li in enumerate(lam_idx):
        assert np.allclose(out_ssr[:, gi], ssr[:, gi, li])
    assert np.allclose(out_sst, sst)
