"""Round-2 regression: the PV r_B fit must resolve the NESTED per-behavior floor.

Pins round-1 BLOCKER ``rb-pv-noise-floor-resolves-none``. The #658 parent
aggregate stores per-behavior reliability floors NESTED under
``noise_floor["per_behavior_p95"][behavior]`` (``issue658_fit_predictors.aggregate``);
the top-level ``noise_floor["p95"]`` is the SHARED scalar, NOT per-behavior. The
pre-fix ``_resolve_noise_floor`` looked at ``nf.get(behavior)`` /
``nf.get("p95")``, so every behavior's floor resolved to ``None`` at production
scale -> ``a33_pass`` forced False regardless of the selection-aware CI.

The fix resolves the nested ``per_behavior_p95`` key FIRST, then falls back to the
legacy flat shapes (smoke override + older per-behavior dicts).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_fit_module():
    spec = importlib.util.spec_from_file_location(
        "issue658_rb_pv_fit_under_test_nf", SCRIPTS / "issue658_rb_pv_fit.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue658_rb_pv_fit_under_test_nf"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_fit_module()


def test_nested_per_behavior_p95_resolves():
    """The parent-aggregate shape (nested per_behavior_p95) resolves the floor."""
    # the real #658 aggregate's noise_floor shape (issue658_fit_predictors.aggregate)
    nf = {
        "p95": 0.13,  # the SHARED scalar — NOT what a per-behavior read should pick
        "distribution": [0.1, 0.2],
        "per_behavior_p95": {"broad_em": 0.4, "sycophancy": 0.25, "refusal": 1.0},
    }
    assert MOD._resolve_noise_floor(nf, "broad_em") == 0.4
    assert MOD._resolve_noise_floor(nf, "sycophancy") == 0.25
    assert MOD._resolve_noise_floor(nf, "refusal") == 1.0


def test_nested_shape_does_not_fall_back_to_shared_scalar():
    """A per-behavior read must NOT silently return the shared top-level p95."""
    nf = {"p95": 0.99, "per_behavior_p95": {"broad_em": 0.4}}
    # broad_em present nested -> 0.4 (NOT 0.99)
    assert MOD._resolve_noise_floor(nf, "broad_em") == 0.4
    # a behavior absent from per_behavior_p95 AND absent flat -> None (NOT 0.99)
    assert MOD._resolve_noise_floor(nf, "harmful_compliance") is None


def test_legacy_flat_scalar_shape_still_works():
    """The smoke override {behavior: 0.0} (flat scalar) still resolves."""
    nf = {"broad_em": 0.0, "sycophancy": 0.0, "refusal": 0.0, "harmful_compliance": 0.0}
    assert MOD._resolve_noise_floor(nf, "broad_em") == 0.0
    assert MOD._resolve_noise_floor(nf, "sycophancy") == 0.0


def test_legacy_flat_dict_shape_still_works():
    """An older per-behavior {behavior: {noise_floor_p95: x}} dict still resolves."""
    nf = {"broad_em": {"noise_floor_p95": 0.33}, "refusal": {"p95": 0.5}}
    assert MOD._resolve_noise_floor(nf, "broad_em") == 0.33
    assert MOD._resolve_noise_floor(nf, "refusal") == 0.5


def test_missing_behavior_returns_none():
    nf = {"per_behavior_p95": {"broad_em": 0.4}}
    assert MOD._resolve_noise_floor(nf, "not_a_behavior") is None
    assert MOD._resolve_noise_floor({}, "broad_em") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
