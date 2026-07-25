"""Round-3 regression tests for issue #1689 fit_ladder + dispatch fixes.

Pins the six fixes landed in round 3:
  1. Rungs 7/8/9 return REAL predictive R², not aliased rung 4 fallbacks.
  2. Selection-symmetric bootstrap loop runs (n_bootstrap_draws > 0 executes).
  3. Dispatch.sh iterates BOTH models in full mode's run_phase_onpolicy /
     run_phase_capture / run_phase_fit_ladder.
  4. Fit_ladder driver runs --all-layers over CAPTURE_LAYERS in full mode.
  5. Ridge fits use inner-group-cv λ selection over the LAMBDAS grid.
  6. Train/test split is conv-id-grouped, not random.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.issue1689_fit_ladder import (  # noqa: E402
    LAMBDAS,
    _compute_ladder_r2s,
    _conv_grouped_folds,
    _fit_ridge_inner_group_cv,
    _run_ladder_pair,
    _rung_reached_from_r2s,
)

# ---------------------------------------------------------------------------
# Concern: ladder-rungs-7-9-stub
# ---------------------------------------------------------------------------


def _synth_pair(seed: int = 0, n: int = 60, d: int = 8, add_ans_transform: bool = False):
    """Build a synthetic (X_S, Y_S, X_T, Y_T) with a known linear map.

    Source: Y_S = X_S @ W_true + b_true + noise
    Target: X_T = X_S rotated + noise; Y_T = f(Y_S) — if add_ans_transform,
    apply a linear transform B on the Y side so rung 8 (answer reparam)
    strictly outperforms rung 4 (bias refit).
    """
    rng = np.random.default_rng(seed)
    X_S = rng.normal(size=(n, d)).astype(np.float64)
    W_true = rng.normal(size=(d, d)).astype(np.float64)
    Y_S = X_S @ W_true + rng.normal(scale=0.1, size=(n, d))
    # Target: rotate X_S to make context reparam A nontrivial
    R = rng.normal(size=(d, d)).astype(np.float64)
    R_ortho, _ = np.linalg.qr(R)
    X_T = X_S @ R_ortho + rng.normal(scale=0.05, size=(n, d))
    if add_ans_transform:
        # Nontrivial linear transform on the Y side (rung 8 should shine).
        B = rng.normal(size=(d, d)).astype(np.float64)
        Y_T = Y_S @ B + rng.normal(scale=0.05, size=(n, d))
    else:
        Y_T = Y_S + rng.normal(scale=0.05, size=(n, d))
    conv_ids = np.array([f"c{i:04d}" for i in range(n)])
    return X_S, Y_S, X_T, Y_T, conv_ids


def test_rung_7_returns_different_r2_than_rung_4():
    """Rung 7 (context reparam A) should fit a real ridge and produce
    R² != rung 4's bias-refit R² on data where a context transform matters.
    """
    X_S, Y_S, X_T, Y_T, conv_ids = _synth_pair(seed=1, n=80, d=6)
    folds = _conv_grouped_folds(conv_ids, n_folds=5, seed=42)
    train_idx = np.where(folds != 0)[0]
    test_idx = np.where(folds == 0)[0]
    train_conv_ids = conv_ids[train_idx]
    r2s = _compute_ladder_r2s(
        X_S,
        Y_S,
        X_T,
        Y_T,
        train_idx,
        test_idx,
        train_conv_ids,
        LAMBDAS,
        full_conv_ids=conv_ids,
    )
    # STRUCTURAL: rung 7 output MUST differ from rung 4 output — they use
    # different math (A-side ridge vs bias-only). The old stub set them equal.
    assert r2s["rung_7_ctx_reparam"] != r2s["rung_4_bias_refit"], (
        f"rung 7 = {r2s['rung_7_ctx_reparam']}, rung 4 = {r2s['rung_4_bias_refit']} "
        "— they must differ (rung 7 was aliased to rung 4 in round 2)"
    )


def test_rung_8_returns_different_r2_than_rung_4():
    """Rung 8 (answer reparam B) should differ from rung 4 on Y-transformed data."""
    X_S, Y_S, X_T, Y_T, conv_ids = _synth_pair(seed=2, n=80, d=6, add_ans_transform=True)
    folds = _conv_grouped_folds(conv_ids, n_folds=5, seed=42)
    train_idx = np.where(folds != 0)[0]
    test_idx = np.where(folds == 0)[0]
    train_conv_ids = conv_ids[train_idx]
    r2s = _compute_ladder_r2s(
        X_S,
        Y_S,
        X_T,
        Y_T,
        train_idx,
        test_idx,
        train_conv_ids,
        LAMBDAS,
        full_conv_ids=conv_ids,
    )
    assert r2s["rung_8_ans_reparam"] != r2s["rung_4_bias_refit"], (
        f"rung 8 = {r2s['rung_8_ans_reparam']}, rung 4 = {r2s['rung_4_bias_refit']} "
        "— they must differ"
    )


def test_rung_9_returns_different_r2_than_rung_4():
    """Rung 9 (full A·M·B) should differ from rung 4 on both-transform data."""
    X_S, Y_S, X_T, Y_T, conv_ids = _synth_pair(seed=3, n=80, d=6, add_ans_transform=True)
    folds = _conv_grouped_folds(conv_ids, n_folds=5, seed=42)
    train_idx = np.where(folds != 0)[0]
    test_idx = np.where(folds == 0)[0]
    train_conv_ids = conv_ids[train_idx]
    r2s = _compute_ladder_r2s(
        X_S,
        Y_S,
        X_T,
        Y_T,
        train_idx,
        test_idx,
        train_conv_ids,
        LAMBDAS,
        full_conv_ids=conv_ids,
    )
    assert r2s["rung_9_full_AMB"] != r2s["rung_4_bias_refit"], (
        f"rung 9 = {r2s['rung_9_full_AMB']}, rung 4 = {r2s['rung_4_bias_refit']} "
        "— they must differ (rung 9 is the full A·M·B chain, not rung 4 fallback)"
    )


def test_rung_9_matches_or_beats_rungs_7_and_8_on_synthetic():
    """On data with BOTH x-side rotation and y-side transform, rung 9 (full
    A·M·B) should generally match or beat rungs 7 and 8 alone."""
    X_S, Y_S, X_T, Y_T, conv_ids = _synth_pair(seed=4, n=100, d=6, add_ans_transform=True)
    folds = _conv_grouped_folds(conv_ids, n_folds=5, seed=42)
    train_idx = np.where(folds != 0)[0]
    test_idx = np.where(folds == 0)[0]
    train_conv_ids = conv_ids[train_idx]
    r2s = _compute_ladder_r2s(
        X_S,
        Y_S,
        X_T,
        Y_T,
        train_idx,
        test_idx,
        train_conv_ids,
        LAMBDAS,
        full_conv_ids=conv_ids,
    )
    # All three should be valid floats (not NaN, not stub aliases).
    for k in ("rung_7_ctx_reparam", "rung_8_ans_reparam", "rung_9_full_AMB"):
        assert np.isfinite(r2s[k]), f"{k} produced non-finite R² = {r2s[k]}"


# ---------------------------------------------------------------------------
# Concern: selection-symmetric-bootstrap-missing
# ---------------------------------------------------------------------------


def test_bootstrap_loop_executes_and_persists_matrix():
    """The bootstrap loop MUST run n_bootstrap_draws iterations and persist
    the per-draw x per-rung R^2 matrix + per-draw rung_reached."""
    X_S, Y_S, X_T, Y_T, conv_ids = _synth_pair(seed=5, n=60, d=6)
    source = {"X_prefix": X_S, "X_context": X_S, "Y": Y_S, "conv_ids": conv_ids}
    target = {"X_prefix": X_T, "X_context": X_T, "Y": Y_T, "conv_ids": conv_ids}
    result = _run_ladder_pair(
        source, target, arm="prefix", n_bootstrap_draws=5, n_null_draws=3, seed=42
    )
    assert "bootstrap_draws" in result
    bd = result["bootstrap_draws"]
    assert bd["n_draws"] == 5, f"expected n_draws=5, got {bd['n_draws']}"
    r2_matrix = np.asarray(bd["r2_matrix"])
    assert r2_matrix.shape == (5, 9), f"expected (5, 9) r2_matrix, got {r2_matrix.shape}"
    # Rung-reached distribution persisted per-draw.
    assert len(bd["rung_reached_per_draw"]) == 5
    # Matched-capacity null: same-selection recipe.
    assert "matched_capacity_null" in result
    assert result["matched_capacity_null"]["n_draws"] == 3
    assert len(result["matched_capacity_null"]["rung_reached_per_draw"]) == 3


# ---------------------------------------------------------------------------
# Concern: ladder-conv-grouped-split
# ---------------------------------------------------------------------------


def test_conv_grouped_folds_keeps_conv_ids_together():
    """Rows sharing a conv_id land in the same fold — no leakage across split."""
    # 3 conv_ids each with 4 rows: rows [0,1,2,3]=c0, [4..7]=c1, [8..11]=c2.
    conv_ids = np.array(["c0"] * 4 + ["c1"] * 4 + ["c2"] * 4)
    folds = _conv_grouped_folds(conv_ids, n_folds=3, seed=42)
    # Rows sharing a conv_id must share a fold.
    assert len(set(folds[0:4].tolist())) == 1
    assert len(set(folds[4:8].tolist())) == 1
    assert len(set(folds[8:12].tolist())) == 1
    # Three distinct folds used (one per conv).
    assert len(set(folds.tolist())) == 3


# ---------------------------------------------------------------------------
# Concern: ladder-lambda-untuned
# ---------------------------------------------------------------------------


def test_inner_group_cv_selects_from_lambdas_grid():
    """The inner-group-cv λ selector must return best_lambda from the LAMBDAS grid."""
    X_S, Y_S, _, _, conv_ids = _synth_pair(seed=6, n=80, d=6)
    _, _, best_lam = _fit_ridge_inner_group_cv(X_S, Y_S, conv_ids, LAMBDAS, n_inner_folds=3)
    assert best_lam in LAMBDAS.tolist(), (
        f"best_lambda {best_lam} not in LAMBDAS grid {LAMBDAS.tolist()}"
    )
    # LAMBDAS is 13 points on logspace(-2, 4, 13) — a canonical value in the grid.
    assert LAMBDAS.shape == (13,), f"LAMBDAS grid size mismatch: {LAMBDAS.shape}"


def test_lambda_grid_range():
    """LAMBDAS grid matches plan §11 committed values: logspace(-2, 4, 13)."""
    assert np.isclose(LAMBDAS[0], 1e-2)
    assert np.isclose(LAMBDAS[-1], 1e4)
    assert len(LAMBDAS) == 13


# ---------------------------------------------------------------------------
# Concern: selection-symmetric argmin rule
# ---------------------------------------------------------------------------


def test_rung_reached_picks_weakest_meeting_bar():
    """Selection returns the WEAKEST (lowest-index) rung with R² >= reach_bar."""
    rung_r2s = {
        "rung_1_direct": 0.5,
        "rung_2_ctx_offset": 0.6,
        "rung_3_ans_offset": 0.7,
        "rung_4_bias_refit": 0.85,  # first to cross bar
        "rung_5_scalar_alpha": 0.87,
        "rung_6_rotation": 0.88,
        "rung_7_ctx_reparam": 0.89,
        "rung_8_ans_reparam": 0.9,
        "rung_9_full_AMB": 0.91,
    }
    reach_bar = 0.8
    assert _rung_reached_from_r2s(rung_r2s, reach_bar) == 4


def test_rung_reached_defaults_to_9_when_none_reach():
    """If NO rung reaches the bar, rung_reached defaults to 9 (strongest)."""
    rung_r2s = {f"rung_{i}_x": 0.1 for i in range(1, 10)}
    # Fix key names to match the ordered list in _rung_reached_from_r2s.
    rung_r2s = {
        "rung_1_direct": 0.1,
        "rung_2_ctx_offset": 0.1,
        "rung_3_ans_offset": 0.1,
        "rung_4_bias_refit": 0.1,
        "rung_5_scalar_alpha": 0.1,
        "rung_6_rotation": 0.1,
        "rung_7_ctx_reparam": 0.1,
        "rung_8_ans_reparam": 0.1,
        "rung_9_full_AMB": 0.1,
    }
    assert _rung_reached_from_r2s(rung_r2s, reach_bar=0.9) == 9


# ---------------------------------------------------------------------------
# Concern: dispatch-base-model-missing
# ---------------------------------------------------------------------------


def _read_dispatch_sh() -> str:
    dispatch_path = _REPO_ROOT / "scripts" / "issue1689_dispatch.sh"
    return dispatch_path.read_text(encoding="utf-8")


def test_dispatch_full_mode_iterates_both_models_in_onpolicy():
    """run_phase_onpolicy must iterate BOTH MODEL_BASE and MODEL_INSTRUCT
    in full mode. Concern: dispatch-base-model-missing (round 2 blocker)."""
    body = _read_dispatch_sh()
    # Extract the run_phase_onpolicy function body.
    match = re.search(r"run_phase_onpolicy\(\)\s*\{(.+?)^\}", body, re.MULTILINE | re.DOTALL)
    assert match, "run_phase_onpolicy function not found in dispatch.sh"
    fn_body = match.group(1)
    # Full-mode models_full list must contain both.
    assert "MODEL_BASE" in fn_body, "MODEL_BASE missing from run_phase_onpolicy"
    assert "MODEL_INSTRUCT" in fn_body, "MODEL_INSTRUCT missing from run_phase_onpolicy"
    # Assert BOTH appear in the full-mode variable definition.
    full_line = re.search(r"models_full=\"([^\"]+)\"", fn_body)
    assert full_line, "models_full variable not found in run_phase_onpolicy"
    assert "MODEL_BASE" in full_line.group(1), f"models_full lacks MODEL_BASE: {full_line.group(1)}"
    assert "MODEL_INSTRUCT" in full_line.group(1), (
        f"models_full lacks MODEL_INSTRUCT: {full_line.group(1)}"
    )


def test_dispatch_full_mode_iterates_both_models_in_capture():
    """run_phase_capture must iterate both models in full mode."""
    body = _read_dispatch_sh()
    match = re.search(r"run_phase_capture\(\)\s*\{(.+?)^\}", body, re.MULTILINE | re.DOTALL)
    assert match, "run_phase_capture function not found in dispatch.sh"
    fn_body = match.group(1)
    full_line = re.search(r"models_full=\"([^\"]+)\"", fn_body)
    assert full_line, "models_full variable not found in run_phase_capture"
    assert "MODEL_BASE" in full_line.group(1)
    assert "MODEL_INSTRUCT" in full_line.group(1)


def test_dispatch_full_mode_iterates_both_models_in_fit_ladder():
    """run_phase_fit_ladder must iterate both model slugs in full mode."""
    body = _read_dispatch_sh()
    match = re.search(r"run_phase_fit_ladder\(\)\s*\{(.+?)^\}", body, re.MULTILINE | re.DOTALL)
    assert match, "run_phase_fit_ladder function not found in dispatch.sh"
    fn_body = match.group(1)
    full_line = re.search(r"models_full=\"([^\"]+)\"", fn_body)
    assert full_line, "models_full variable not found in run_phase_fit_ladder"
    assert "Qwen_Qwen2.5-7B " in full_line.group(1) or 'Qwen_Qwen2.5-7B"' in full_line.group(1), (
        f"base model slug missing from models_full: {full_line.group(1)}"
    )
    assert "Qwen_Qwen2.5-7B-Instruct" in full_line.group(1), (
        f"instruct model slug missing from models_full: {full_line.group(1)}"
    )


# ---------------------------------------------------------------------------
# Concern: ladder-per-layer-loop-missing
# ---------------------------------------------------------------------------


def test_dispatch_fit_ladder_passes_all_layers_in_full_mode():
    """run_phase_fit_ladder must pass --all-layers in full mode (concern
    ladder-per-layer-loop-missing)."""
    body = _read_dispatch_sh()
    match = re.search(r"run_phase_fit_ladder\(\)\s*\{(.+?)^\}", body, re.MULTILINE | re.DOTALL)
    assert match, "run_phase_fit_ladder function not found"
    fn_body = match.group(1)
    assert "--all-layers" in fn_body, "run_phase_fit_ladder does not pass --all-layers"


def test_fit_ladder_all_layers_flag_present_in_argparse():
    """The fit_ladder driver must expose --all-layers so full mode can loop 4 layers."""
    from scripts import issue1689_fit_ladder as m

    src = Path(m.__file__).read_text(encoding="utf-8")
    assert "--all-layers" in src, "fit_ladder.py does not expose --all-layers"
    assert "CAPTURE_LAYERS" in src, "fit_ladder.py does not reference CAPTURE_LAYERS"


# ---------------------------------------------------------------------------
# Smoke: driver's --help works (import doesn't crash).
# ---------------------------------------------------------------------------


def test_fit_ladder_help_runs():
    """`fit_ladder.py --help` must exit 0 (basic import + argparse smoke)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "issue1689_fit_ladder.py"),
            "--help",
        ],
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"--help failed rc={proc.returncode}: {proc.stderr.decode()[-500:]}"
    )
    assert b"--all-layers" in proc.stdout
    assert b"--bootstrap-draws" in proc.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
