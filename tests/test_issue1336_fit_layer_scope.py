"""#1336: fc._cell_xy layer-invariant scoping (fails pre-fix, passes post-fix).

The #825 fit core asserts every bundle's layer axis against ITS module-global
Qwen ``EXPECTED_LAYERS`` (28). The #1336 drivers operate on Llama-3.1 stores
(32 layers) and tiny smoke stores — without the scoped rebind
(``common.fc_expected_layers``) every production per-cell fit, the align
battery, and the G0 local-fixture path crash with ``layer axis 32 != 28``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _bundle(n: int, layers: int, dim: int) -> dict:
    rng = np.random.default_rng(0)
    return {
        "arrays": {
            "slots": rng.normal(size=(n, 2, layers, dim)).astype(np.float32),
            "profiles": rng.normal(size=(n, 2, layers, dim)).astype(np.float32),
            "nll": rng.uniform(1.0, 3.0, size=(n, 2)).astype(np.float32),
        },
        "sidecar": {"conv_ids": [f"s{i}" for i in range(n)]},
    }


def test_cell_xy_1336_accepts_llama_32_layers():
    """Production shape: 32-layer bundle must pass through the driver helper."""
    import issue1336_fit_cells as d

    xy = d._cell_xy_1336(_bundle(6, 32, 8), expected_layers=32)
    assert xy["X"].shape == (6, 32, 8)
    assert xy["Y"].shape == (6, 32, 8)


def test_cell_xy_1336_mismatch_still_fails_loud():
    """The invariant stays ACTIVE: a wrong-shaped bundle is refused."""
    import issue1336_fit_cells as d

    with pytest.raises(AssertionError, match="layer axis"):
        d._cell_xy_1336(_bundle(6, 28, 8), expected_layers=32)


def test_scope_restores_the_825_global():
    """The rebind is scoped — the #825 module is never left mutated."""
    import issue825_fit_cells as fc

    from explore_persona_space.experiments.issue_1336 import common as cm

    before = fc.EXPECTED_LAYERS
    with cm.fc_expected_layers(fc, 99):
        assert fc.EXPECTED_LAYERS == 99
    assert before == fc.EXPECTED_LAYERS


def test_align_xy_for_derives_smoke_layer_count():
    """Align-side: expected_layers=None derives the bundle's realized count."""
    import issue1336_ladder_alignment as a

    x, y, ids = a._xy_for(_bundle(5, 4, 8), expected_layers=None)
    assert x.shape == (5, 4, 8) and y.shape == (5, 4, 8) and len(ids) == 5
