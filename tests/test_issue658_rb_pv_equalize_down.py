# ruff: noqa: RUF003
# Intentional Unicode (×) in scientific docstrings.
"""Round-3 regression: kept r_B pools are equalized-down to a common floor-N.

Pins round-2 CONCERN ``rb-pv-equalize-down-not-enforced``. Before the r_B build,
``_kept_acts_by_pole`` returned ALL judge-kept acts per pole, and
``build_rb_diffmeans`` averaged over the full VARIABLE-N pos / neg pools — a dose
confound (plan §4.8 + ``.claude/rules/on-policy-completions.md`` require capping
to a common floor-N per behavior BEFORE the build).

The fix runs ``_equalize_down_kept_acts`` after ``_kept_acts_by_pole`` and before
``build_rb_*``: each non-empty pole is sampled down (seeded, ``replace=False``) to
the MINIMUM kept count across the behavior's poles, and the per-pole
pre/used counts are recorded in the aggregate manifest (``equalize_down`` block).

Checks:
1. uneven kept counts → every cell of a behavior is built over the SAME N (the
   behavior's floor); a balanced behavior is unchanged.
2. the equalize-down is deterministic (same seed → same selection).
3. the per-cell ``used_n`` is recorded (the manifest field exists + matches).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

L, H = 3, 8  # tiny act shape (n_layers, hidden) for the fixtures


def _load_fit_module():
    spec = importlib.util.spec_from_file_location(
        "issue658_rb_pv_fit_eqdown_under_test", SCRIPTS / "issue658_rb_pv_fit.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue658_rb_pv_fit_eqdown_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_fit_module()


def _kept(pos: int, neg: int, neutral: int) -> dict[str, np.ndarray]:
    """A kept dict {pole: (n, L, H)} with distinct per-row values (rng-seeded)."""
    rng = np.random.default_rng(0)
    return {
        "pos": rng.standard_normal((pos, L, H)) if pos else np.zeros((0, L, H)),
        "neg": rng.standard_normal((neg, L, H)) if neg else np.zeros((0, L, H)),
        "neutral": rng.standard_normal((neutral, L, H)) if neutral else np.zeros((0, L, H)),
    }


# ── uneven counts → common floor-N; balanced behavior unchanged ────────────────


def test_uneven_pools_equalized_to_common_floor():
    """behavior A: pos=80, neg=120 → both capped to 80 (the min)."""
    kept = _kept(80, 120, 0)  # neutral empty (an empty pole is excluded from the floor)
    eq, pre, used, floor_n = MOD._equalize_down_kept_acts(kept, seed=658)
    assert floor_n == 80, f"floor should be min(80,120)=80, got {floor_n}"
    assert eq["pos"].shape[0] == 80
    assert eq["neg"].shape[0] == 80  # 120 capped DOWN to 80
    assert pre == {"pos": 80, "neg": 120, "neutral": 0}
    assert used == {"pos": 80, "neg": 80, "neutral": 0}


def test_build_rb_diffmeans_uses_same_n_after_equalize_down():
    """After equalize-down, diff-in-means averages pos and neg over the SAME N."""
    kept = _kept(80, 120, 0)
    eq, _pre, used, _floor = MOD._equalize_down_kept_acts(kept, seed=658)
    # both contributing poles now have equal N (the build averages over floor_n each)
    assert eq["pos"].shape[0] == eq["neg"].shape[0] == used["pos"] == used["neg"]
    rb = MOD.build_rb_diffmeans(eq, "pos-vs-neg")
    assert rb is not None and rb.shape == (L, H)


def test_balanced_behavior_unchanged():
    """behavior B: pos=200, neg=200 → floor=200, nothing dropped."""
    kept = _kept(200, 200, 0)
    eq, _pre, used, floor_n = MOD._equalize_down_kept_acts(kept, seed=658)
    assert floor_n == 200
    assert used == {"pos": 200, "neg": 200, "neutral": 0}
    # rows are identical (no sampling when already at the floor)
    assert np.array_equal(eq["pos"], kept["pos"])
    assert np.array_equal(eq["neg"], kept["neg"])


def test_all_three_poles_equalized():
    """pos=50, neg=70, neutral=90 → all three capped to 50 (the common min)."""
    kept = _kept(50, 70, 90)
    eq, _pre, used, floor_n = MOD._equalize_down_kept_acts(kept, seed=658)
    assert floor_n == 50
    assert used == {"pos": 50, "neg": 50, "neutral": 50}
    assert eq["pos"].shape[0] == eq["neg"].shape[0] == eq["neutral"].shape[0] == 50


# ── determinism ───────────────────────────────────────────────────────────────


def test_equalize_down_is_deterministic():
    """Same seed → identical down-sampled rows (reproducible r_B build)."""
    kept_a = _kept(80, 120, 0)
    kept_b = _kept(80, 120, 0)
    eq_a, _, _, _ = MOD._equalize_down_kept_acts(kept_a, seed=658)
    eq_b, _, _, _ = MOD._equalize_down_kept_acts(kept_b, seed=658)
    assert np.array_equal(eq_a["neg"], eq_b["neg"])


def test_empty_behavior_returns_none_floor():
    """No kept acts anywhere → floor_n None, all poles empty (cell later skipped)."""
    kept = _kept(0, 0, 0)
    eq, pre, _used, floor_n = MOD._equalize_down_kept_acts(kept, seed=658)
    assert floor_n is None
    assert all(eq[p].shape[0] == 0 for p in eq)
    assert pre == {"pos": 0, "neg": 0, "neutral": 0}


# ── manifest records the per-cell used_n ───────────────────────────────────────


def test_main_records_kept_n_used_in_aggregate_row():
    """The main fit flow records the equalize-down provenance (used_n) per row.

    Static check that the aggregate row dict carries the ``equalize_down`` block
    with ``kept_n_used`` + ``floor_n`` + ``pre_equalize_n`` (the manifest field the
    concern requires), and that ``_equalize_down_kept_acts`` is called from main
    BEFORE ``build_cell_predictions``.
    """
    src = (SCRIPTS / "issue658_rb_pv_fit.py").read_text()
    # the equalize step runs before the build
    eq_call = src.index("_equalize_down_kept_acts(")
    build_call = src.index("build_cell_predictions(", eq_call)
    assert eq_call < build_call, "equalize-down must run BEFORE build_cell_predictions"
    # the aggregate row records the provenance
    assert '"equalize_down": {' in src
    assert '"kept_n_used": kept_n_used' in src
    assert '"floor_n": floor_n' in src
    assert '"pre_equalize_n": pre_equalize_n' in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
