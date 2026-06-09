#!/usr/bin/env python3
"""Issue #523 — Phase D held-out CV scoring.

Two parallel outputs share the SAME 13-fold leave-one-SOURCE-condition-out
scheme (plan v2 §4 Phase D, binding):

  * CELL-FIXED HEADLINE — L22 gauss_kl fixed, no inner search:
      cell_fixed_seed42_nonstyl_heldout   ← the Goal's cell-specific answer
      cell_fixed_seed43_nonstyl_heldout
      js_baseline_seed42_nonstyl_heldout
      cell_fixed_seed42_full_heldout

  * NESTED-SEARCH DIAGNOSTIC — full 1737-cell inner argmax per fold:
      nested_search_seed42_nonstyl_heldout

Outer fold scheme (verbatim from plan §4 Phase D — assertions at top of main):

    NON_STY_SRC = ["A1","A2","B1","B2","B3","B4","B5","C1","D1","D2","D3","D4","D5"]
    for outer_fold_idx, S in enumerate(NON_STY_SRC):  # 13 folds
        test_pairs  = [(S, T) for T in NON_STY_SRC if T != S]                    # 12 pairs
        train_pairs = [(A, B) for A in NON_STY_SRC for B in NON_STY_SRC
                               if A != B and A != S]                              # 132 pairs

Each non-stylized ordered pair (156 total) appears in EXACTLY ONE fold (its
source's fold), so the per-fold R² values are independent on the test side
and the fold-level paired bootstrap is structurally valid. This dissolves
the v1 leave-one-CLASS-out scheme's "each pair in two folds" problem.

Outputs (per plan §4 Phase D — five bar JSONs + the forest_plot_data aggregator):

    eval_results/issue_523/scoring/cell_fixed_seed42_nonstyl_heldout.json
    eval_results/issue_523/scoring/cell_fixed_seed43_nonstyl_heldout.json
    eval_results/issue_523/scoring/cell_fixed_seed42_full_heldout.json
    eval_results/issue_523/scoring/js_baseline_seed42_nonstyl_heldout.json
    eval_results/issue_523/scoring/nested_search_seed42_nonstyl_heldout.json
    eval_results/issue_523/scoring/forest_plot_data.json

Usage::

    # Full scoring (CPU-only, ~1 h).
    uv run python scripts/issue523_phase_d_scoring.py

    # Smoke: just fold 0, cell-fixed + nested-search, prints the inner-selected cell.
    uv run python scripts/issue523_phase_d_scoring.py --smoke-fold 0 --smoke-cell A1

    # Subset: only the headline bar (cell-fixed seed42 nonstyl).
    uv run python scripts/issue523_phase_d_scoring.py --only-headline
"""

# ruff: noqa: RUF001 RUF002 RUF003

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("i523.phase_d")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ────────────────────────── Constants (binding from plan §4 Phase D) ──────────────────────────

# 16-cond panel (plan §10 Reproducibility Card).
ALL_CIDS = (
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "C1",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
)
# Stylized source conditions dropped from the non-stylized panel.
STY_CIDS = {"A3", "A4", "A5"}
# Non-stylized 13 source conditions in the fold loop order.
NON_STY_SRC = ("A1", "A2", "B1", "B2", "B3", "B4", "B5", "C1", "D1", "D2", "D3", "D4", "D5")

# Headline cell — the Goal's named cell.
HEADLINE_CELL = ("last_prompt", 22, "gauss_kl", "raw")
# JS-baseline cell — last_prompt × full-layer × next_token_js.
# In the regression JSON it lives at extraction_point="last_prompt",
# metric="next_token_js" (no layer; the bakeoff stores layer=None/-1).
# We honor the bakeoff convention by looking up the entry with metric==
# "next_token_js" and extraction_point=="last_prompt".
JS_BASELINE_CELL = ("last_prompt", None, "next_token_js", "raw")

# Bootstrap iters per plan §11 (#474's number, picked over #502's 1000).
BOOTSTRAP_N = 2000

# Scoring inputs.
BAKEOFF_METRICS_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_523" / "bakeoff" / "metrics"
BAKEOFF_REGRESSION_DIR_DEFAULT = (
    PROJECT_ROOT / "eval_results" / "issue_523" / "bakeoff" / "regression"
)
G_SEED42_PATH = (
    PROJECT_ROOT / "eval_results" / "issue_474" / "cross_eval" / "loc_ep1" / "G_logprob_matrix.json"
)
G_SEED43_PATH = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_523"
    / "seed43_cross_eval"
    / "loc_ep1"
    / "G_logprob_matrix.json"
)
PROMPT_TOKENS_PATH = PROJECT_ROOT / "eval_results" / "issue_406" / "divergence" / "D_matrix.json"

SCORING_DIR = PROJECT_ROOT / "eval_results" / "issue_523" / "scoring"


# ────────────────────────── Provenance ──────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            env={**os.environ},  # epm-lint: subprocess explicit env
        ).strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ────────────────────────── Fold scheme + invariants ──────────────────────────


def loco_folds() -> list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]]:
    """Return the 13 (source_held_out, test_pairs, train_pairs) tuples.

    Plan v2 §4 Phase D binding fold definition. The invariants asserted
    immediately below this function MUST hold for the bootstrap to be valid.
    """
    folds = []
    for s in NON_STY_SRC:
        test_pairs = [(s, t) for t in NON_STY_SRC if t != s]
        # Plan v2 §4 Phase D binding: train_pairs excludes S_k from BOTH the
        # source AND the target sides → 12 sources × 11 targets = 132.
        # The pseudocode `A' ≠ S_k` alone gives 12 × 12 = 144 — that wording
        # in the plan inlined pseudocode is short for "neither side touches
        # S_k", which the count 132 explicitly demands. With this filter the
        # 156 test-pair union (each pair in exactly one fold) is preserved
        # AND fold-level independence holds (the train side never sees
        # S_k's outgoing OR incoming edges).
        train_pairs = [
            (a, b) for a in NON_STY_SRC for b in NON_STY_SRC if a != b and a != s and b != s
        ]
        folds.append((s, test_pairs, train_pairs))
    return folds


def _assert_fold_invariants(folds: list[tuple[str, list, list]]) -> None:
    """Plan v2 §4 Phase D: 12 test pairs/fold + 156 union (no double-count)."""
    assert len(folds) == 13, f"expected 13 folds, got {len(folds)}"
    union_test: set[tuple[str, str]] = set()
    for s, tests, trains in folds:
        if len(tests) != 12:
            raise AssertionError(f"fold S={s} has {len(tests)} test pairs, expected 12")
        if len(trains) != 132:
            raise AssertionError(f"fold S={s} has {len(trains)} train pairs, expected 132")
        for tp in tests:
            if tp in union_test:
                raise AssertionError(
                    f"non-stylized pair {tp} appears in MORE than one fold "
                    "— invariant breach (plan §4 Phase D)."
                )
            union_test.add(tp)
    # Total non-stylized ordered pairs: 13 sources × 12 = 156.
    if len(union_test) != 156:
        raise AssertionError(f"fold test union is {len(union_test)} pairs, expected 156")


def full_panel_folds() -> list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]]:
    """Same scheme but over all 15 sources × 14 targets = 210 (the *full-panel*
    column actually used by the supporting bar; plan §6 Supporting says 240
    but that counts symmetric directed pairs which the regression already
    handles; we mirror #502's convention of 15 source folds × 14 pairs/fold
    on the 16-cond panel WITHOUT the diagonals).

    Actually the plan §6 Supporting bar says "15 source folds × 14 pairs/fold"
    over all 240 ordered pairs (i.e. 15 sources, 16 targets, drop the diagonal
    → 15 × 15 = 225; the 240 number includes some duplicate accounting in
    plan v1 that v2 dropped). For the supporting cell-fixed full-panel bar
    we do 16 - 1 = 15 sources × (16 - 1) = 15 pairs/fold; the union is 15 × 15 = 225.
    """
    sources = list(ALL_CIDS)
    folds = []
    for s in sources:
        test_pairs = [(s, t) for t in ALL_CIDS if t != s]
        # Same constraint as the non-stylized scheme: S_k excluded from BOTH
        # source AND target on train, so the 16 × 15 = 240 ordered-pair
        # union splits into 16 outer folds × 15 test pairs/fold with no
        # overlap; train side is 15 sources × 14 targets = 210 pairs.
        train_pairs = [(a, b) for a in ALL_CIDS for b in ALL_CIDS if a != b and a != s and b != s]
        folds.append((s, test_pairs, train_pairs))
    return folds


# ────────────────────────── Data loaders ──────────────────────────


def load_delta_g(g_matrix_path: Path) -> dict[tuple[str, str], float]:
    """Load ΔG = delta_g for every (source, target) ordered pair from a G matrix JSON.

    Returns a flat (a, b) → delta_g dict. The keys are TYPE-CHECKED tuples;
    callers should index defensively (a missing pair raises KeyError, which
    fail-louds rather than silently scoring on partial data).
    """
    if not g_matrix_path.exists():
        raise FileNotFoundError(f"G matrix {g_matrix_path} missing; Phase B must produce it.")
    payload = json.loads(g_matrix_path.read_text())
    G = payload["G"]
    out: dict[tuple[str, str], float] = {}
    for a, inner in G.items():
        for b, cell in inner.items():
            out[(a, b)] = float(cell["delta_g"])
    return out


def load_distance_matrix(
    metrics_dir: Path,
    extraction_point: str,
    layer: int | None,
    metric: str,
    variant: str,
) -> dict[tuple[str, str], float]:
    """Read one (point, layer, metric, variant) matrix file → (a,b) → distance.

    For the JS baseline cell (metric == 'next_token_js', layer is None), the
    bakeoff stores a single layer-less file.
    """
    if metric == "next_token_js":
        # The bakeoff stores JS at one canonical path per extraction-point.
        # Pattern matches #502 layout.
        candidates = list(metrics_dir.glob(f"{extraction_point}__next_token_js*.json"))
        if not candidates:
            candidates = list(metrics_dir.parent.glob(f"next_token_js__{extraction_point}*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No next_token_js matrix under {metrics_dir} for "
                f"extraction_point={extraction_point}"
            )
        p = candidates[0]
    else:
        p = metrics_dir / f"{extraction_point}__layer{layer}__{metric}__{variant}.json"
    if not p.exists():
        raise FileNotFoundError(f"distance matrix {p} missing")
    payload = json.loads(p.read_text())
    mat = payload["matrix"]
    # end_of_system cells for Class B / C / D conds are stored as None
    # (the prompt has no system message → extraction is N/A). Treat the
    # whole-matrix-None case as "no valid pairs" rather than raising.
    if mat is None:
        return {}
    out: dict[tuple[str, str], float] = {}
    for a, inner in mat.items():
        if inner is None:
            continue
        for b, val in inner.items():
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fval):
                continue
            out[(a, b)] = fval
    return out


def load_prompt_tokens() -> dict[tuple[str, str], int]:
    """Load #406's pair-level prompt-token counts (length covariate).

    Same convention as `issue493_extraction_metric_bakeoff._load_prompt_tokens`.
    """
    if not PROMPT_TOKENS_PATH.exists():
        raise FileNotFoundError(
            f"prompt_tokens {PROMPT_TOKENS_PATH} missing; #406 substrate required."
        )
    payload = json.loads(PROMPT_TOKENS_PATH.read_text())
    table = payload["prompt_tokens"]
    out: dict[tuple[str, str], int] = {}
    for a, inner in table.items():
        for b, n in inner.items():
            out[(a, b)] = int(n)
    return out


# ────────────────────────── Length-controlled CV R² ──────────────────────────


def _safe_polyfit_residual(target: np.ndarray, covar: np.ndarray) -> np.ndarray | None:
    """Residualize `target` linearly on `covar`. Returns None on degenerate fit."""
    try:
        b, a = np.polyfit(covar, target, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    fit = a + b * covar
    if not np.all(np.isfinite(fit)):
        return None
    return target - fit


def _length_controlled_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    covar_train: np.ndarray,
    x_test: np.ndarray,
    covar_test: np.ndarray,
) -> np.ndarray:
    """Length-controlled linear fit, train on (x_train, y_train, log(covar)),
    predict on (x_test, log(covar_test)).

    Approach (matches #502 / #474 convention):
      1. Residualize y on log(covar) on the TRAIN set (linear fit
         ``y ~ log(covar)`` → ``y_resid_train = y_train − (a + b*log_c)``).
      2. Fit ``y_resid_train ~ x_train`` (bare 1-D OLS).
      3. Predict on test: ``ŷ = (a + b * log(covar_test)) + (α + β * x_test)``.
    Degenerate train (n<5, constant x) → NaN predictions.

    The two-stage decoupling matches the project's `_length_partial`
    rank-then-residualize spirit while keeping the prediction on the linear
    scale (so R² is interpretable as variance explained on the test set).
    """
    n = len(x_train)
    if n < 5 or len(np.unique(x_train)) < 2:
        return np.full(len(x_test), np.nan)
    log_cov_train = np.log(covar_train.clip(min=1.0))
    log_cov_test = np.log(covar_test.clip(min=1.0))
    # Step 1: residualize y on log(covar) on train.
    if np.unique(log_cov_train).size >= 2:
        try:
            b_cov, a_cov = np.polyfit(log_cov_train, y_train, 1)
        except (np.linalg.LinAlgError, ValueError):
            b_cov, a_cov = 0.0, float(y_train.mean())
    else:
        b_cov, a_cov = 0.0, float(y_train.mean())
    y_resid_train = y_train - (a_cov + b_cov * log_cov_train)
    # Step 2: bare OLS fit of y_resid ~ x.
    try:
        b_x, a_x = np.polyfit(x_train, y_resid_train, 1)
    except (np.linalg.LinAlgError, ValueError):
        return np.full(len(x_test), np.nan)
    # Step 3: assemble prediction.
    return (a_cov + b_cov * log_cov_test) + (a_x + b_x * x_test)


def fold_r2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    covar_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    covar_test: np.ndarray,
) -> float:
    """R² of length-controlled linear fit, train on train, score on test.

    R² = 1 − SSE / SST with SST = sum((y_test − mean(y_test))²) so a fold
    where the test rows are constant returns NaN (undefined) rather than
    +∞ / a meaningless number.
    """
    y_hat = _length_controlled_fit_predict(x_train, y_train, covar_train, x_test, covar_test)
    finite_mask = np.isfinite(y_hat) & np.isfinite(y_test)
    if finite_mask.sum() < 3:
        return float("nan")
    yh = y_hat[finite_mask]
    yt = y_test[finite_mask]
    sst = float(np.sum((yt - yt.mean()) ** 2))
    if sst < 1e-12:
        return float("nan")
    sse = float(np.sum((yt - yh) ** 2))
    return 1.0 - sse / sst


def cell_fixed_loco(
    distances: dict[tuple[str, str], float],
    delta_g: dict[tuple[str, str], float],
    prompt_tokens: dict[tuple[str, str], int],
    folds: list[tuple[str, list, list]],
) -> tuple[float, list[float]]:
    """Cell-fixed leave-one-source-condition-out CV R².

    Returns (mean_across_folds, per_fold_R2_list). NaN folds drop from the
    mean BUT the per-fold list keeps them so the caller can report fold
    coverage.
    """
    per_fold: list[float] = []
    for _s, tests, trains in folds:
        x_train = np.array([distances.get(p, np.nan) for p in trains], dtype=np.float64)
        y_train = np.array([delta_g.get(p, np.nan) for p in trains], dtype=np.float64)
        c_train = np.array([prompt_tokens.get(p, 1) for p in trains], dtype=np.float64)
        x_test = np.array([distances.get(p, np.nan) for p in tests], dtype=np.float64)
        y_test = np.array([delta_g.get(p, np.nan) for p in tests], dtype=np.float64)
        c_test = np.array([prompt_tokens.get(p, 1) for p in tests], dtype=np.float64)
        # Restrict train rows to finite-everywhere subsets.
        train_mask = np.isfinite(x_train) & np.isfinite(y_train) & np.isfinite(c_train)
        if train_mask.sum() < 5:
            per_fold.append(float("nan"))
            continue
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]
        c_train = c_train[train_mask]
        r2 = fold_r2(x_train, y_train, c_train, x_test, y_test, c_test)
        per_fold.append(r2)
    finite_r2 = [r for r in per_fold if np.isfinite(r)]
    return (
        float(np.mean(finite_r2)) if finite_r2 else float("nan"),
        per_fold,
    )


# ────────────────────────── Bootstrap ──────────────────────────


def fold_bootstrap_ci(
    per_fold: list[float], n_boot: int = BOOTSTRAP_N, seed: int = 42
) -> tuple[float, float, float]:
    """Resample folds with replacement; return (lo, hi, half_width) at 2.5/97.5."""
    arr = np.array([r for r in per_fold if np.isfinite(r)], dtype=np.float64)
    if len(arr) < 3:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    half_width = (hi - lo) / 2.0
    return lo, hi, half_width


def paired_fold_bootstrap_ci_on_delta(
    per_fold_a: list[float],
    per_fold_b: list[float],
    n_boot: int = BOOTSTRAP_N,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Paired resample of A vs B per fold; return (mean_delta, lo, hi, half_width).

    Inputs are aligned by fold index. Folds where either side is NaN drop.
    """
    if len(per_fold_a) != len(per_fold_b):
        raise ValueError(f"length mismatch: {len(per_fold_a)} vs {len(per_fold_b)}")
    pairs = [
        (a, b)
        for a, b in zip(per_fold_a, per_fold_b, strict=True)
        if np.isfinite(a) and np.isfinite(b)
    ]
    if len(pairs) < 3:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    delta = a_arr - b_arr
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(delta)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = delta[idx].mean()
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return float(delta.mean()), lo, hi, (hi - lo) / 2.0


# ────────────────────────── Nested-search inner argmax ──────────────────────────


def enumerate_inner_cells(
    metrics_dir: Path,
) -> list[tuple[str, int | None, str, str]]:
    """Return every (extraction_point, layer, metric, variant) cell present
    under `metrics_dir`. Used by the nested-search diagnostic per fold.
    """
    cells: list[tuple[str, int | None, str, str]] = []
    for p in sorted(metrics_dir.glob("*__layer*__*__*.json")):
        # Pattern: {ep}__layer{L}__{metric}__{variant}.json
        name = p.stem
        try:
            ep, layer_chunk, metric, variant = name.split("__")
            layer = int(layer_chunk.removeprefix("layer"))
        except (ValueError, AttributeError):
            continue
        cells.append((ep, layer, metric, variant))
    # JS baseline (one per extraction-point), keyed off the bakeoff layout.
    for p in sorted(metrics_dir.glob("*__next_token_js*.json")):
        name = p.stem
        chunks = name.split("__")
        if len(chunks) >= 2:
            ep = chunks[0]
            variant = chunks[-1] if "next_token_js" not in chunks[-1] else "raw"
            cells.append((ep, None, "next_token_js", variant))
    return cells


def nested_search_loco(
    metrics_dir: Path,
    delta_g: dict[tuple[str, str], float],
    prompt_tokens: dict[tuple[str, str], int],
    folds: list[tuple[str, list, list]],
    inner_score: str = "loocv",
) -> tuple[float, list[float], list[tuple[str, int | None, str, str]]]:
    """Full 1737-cell inner argmax per fold.

    For each outer fold k:
      - INNER: rank every candidate cell by its leave-one-source-out CV R²
        computed on the OUTER TRAINING pairs only (i.e. an inner LOCO over
        the 12 non-S sources, restricted to train_pairs).
      - Pick the inner-best cell; score it on the outer test pairs.

    Returns (mean_across_folds, per_fold_R2, per_fold_selected_cell).
    inner_score: 'loocv' selects the cell whose inner-LOCO R² is highest.

    Memo: pre-load every candidate cell's distance matrix; the inner LOCO
    re-runs are cheap dot-products on the trains.
    """
    candidate_cells = enumerate_inner_cells(metrics_dir)
    logger.info("nested-search: %d candidate cells discovered", len(candidate_cells))
    # Pre-load every distance matrix once.
    distances_cache: dict[tuple, dict[tuple[str, str], float]] = {}
    for cell in candidate_cells:
        try:
            distances_cache[cell] = load_distance_matrix(
                metrics_dir, cell[0], cell[1], cell[2], cell[3]
            )
        except (FileNotFoundError, KeyError):
            continue

    per_fold_r2: list[float] = []
    per_fold_selected: list[tuple[str, int | None, str, str]] = []

    for fold_idx, (s, test_pairs, train_pairs) in enumerate(folds):
        best_r2 = -np.inf
        best_cell = None
        # Inner LOCO over the 12 non-S sources, restricted to the outer-train
        # pairs. Each inner fold holds out one of the 12 inner sources.
        inner_sources = [a for a in NON_STY_SRC if a != s]
        inner_folds = []
        for inner_s in inner_sources:
            inner_test = [(inner_s, t) for t in inner_sources if t != inner_s]
            # Match the outer constraint: exclude inner_s from BOTH sides too.
            inner_train = [
                (a, b)
                for a in inner_sources
                for b in inner_sources
                if a != b and a != inner_s and b != inner_s
            ]
            inner_folds.append((inner_s, inner_test, inner_train))

        for cell in candidate_cells:
            if cell not in distances_cache:
                continue
            distances = distances_cache[cell]
            mean_inner, _ = cell_fixed_loco(distances, delta_g, prompt_tokens, inner_folds)
            if not np.isfinite(mean_inner):
                continue
            if mean_inner > best_r2:
                best_r2 = mean_inner
                best_cell = cell
        if best_cell is None:
            per_fold_r2.append(float("nan"))
            per_fold_selected.append(("", None, "", ""))
            continue
        # Score on outer test pairs using the best cell.
        distances = distances_cache[best_cell]
        x_train = np.array([distances.get(p, np.nan) for p in train_pairs], dtype=np.float64)
        y_train = np.array([delta_g.get(p, np.nan) for p in train_pairs], dtype=np.float64)
        c_train = np.array([prompt_tokens.get(p, 1) for p in train_pairs], dtype=np.float64)
        x_test = np.array([distances.get(p, np.nan) for p in test_pairs], dtype=np.float64)
        y_test = np.array([delta_g.get(p, np.nan) for p in test_pairs], dtype=np.float64)
        c_test = np.array([prompt_tokens.get(p, 1) for p in test_pairs], dtype=np.float64)
        train_mask = np.isfinite(x_train) & np.isfinite(y_train) & np.isfinite(c_train)
        if train_mask.sum() < 5:
            per_fold_r2.append(float("nan"))
        else:
            r2 = fold_r2(
                x_train[train_mask],
                y_train[train_mask],
                c_train[train_mask],
                x_test,
                y_test,
                c_test,
            )
            per_fold_r2.append(r2)
        per_fold_selected.append(best_cell)
        logger.info(
            "nested-search fold %d (S=%s): inner-best=%s outer-test-R²=%.4f",
            fold_idx,
            s,
            best_cell,
            per_fold_r2[-1],
        )

    finite = [r for r in per_fold_r2 if np.isfinite(r)]
    return (
        float(np.mean(finite)) if finite else float("nan"),
        per_fold_r2,
        per_fold_selected,
    )


def cell_pick_tally(
    per_fold_selected: list[tuple[str, int | None, str, str]],
) -> dict:
    """Plan §6 "what cell got picked" tally at TWO resolutions.

    (a) exact L22 gauss_kl raw last_prompt picks.
    (b) L19-L24 × gauss_kl × raw last_prompt ridge picks (any layer in ridge).
    """
    exact_hits = sum(1 for c in per_fold_selected if c == HEADLINE_CELL)
    ridge_layers = {19, 20, 21, 22, 23, 24}
    ridge_hits = sum(
        1
        for c in per_fold_selected
        if c[0] == "last_prompt" and c[2] == "gauss_kl" and c[3] == "raw" and c[1] in ridge_layers
    )
    return {
        "exact_l22_gauss_kl_raw_last_prompt": exact_hits,
        "ridge_l19_l24_gauss_kl_raw_last_prompt": ridge_hits,
        "n_folds": len(per_fold_selected),
        "per_fold_picks": [
            {"extraction_point": c[0], "layer": c[1], "metric": c[2], "variant": c[3]}
            for c in per_fold_selected
        ],
        # Top-3 most common picks across folds.
        "top_picks": Counter(per_fold_selected).most_common(3),
    }


# ────────────────────────── Main scoring orchestration ──────────────────────────


def score_one_bar(
    *,
    bar_slug: str,
    distances: dict[tuple[str, str], float],
    delta_g: dict[tuple[str, str], float],
    prompt_tokens: dict[tuple[str, str], int],
    folds: list[tuple[str, list, list]],
    description: str,
    seed_used: int,
    panel: str,
    nested_search: bool = False,
    metrics_dir: Path | None = None,
    paired_baseline_per_fold: list[float] | None = None,
    paired_baseline_label: str | None = None,
) -> dict:
    """Compute one forest-plot bar with bootstrap + optional paired Δ to a baseline.

    Returns the per-bar JSON payload (written to scoring/<bar_slug>.json).
    """
    t0 = time.time()
    if nested_search:
        if metrics_dir is None:
            raise ValueError("nested_search requires metrics_dir")
        mean_r2, per_fold, per_fold_selected = nested_search_loco(
            metrics_dir, delta_g, prompt_tokens, folds
        )
        tally = cell_pick_tally(per_fold_selected)
    else:
        mean_r2, per_fold = cell_fixed_loco(distances, delta_g, prompt_tokens, folds)
        per_fold_selected = None
        tally = None

    lo, hi, hw = fold_bootstrap_ci(per_fold)

    delta_payload = None
    if paired_baseline_per_fold is not None:
        d_mean, d_lo, d_hi, d_hw = paired_fold_bootstrap_ci_on_delta(
            per_fold, paired_baseline_per_fold
        )
        delta_payload = {
            "baseline_label": paired_baseline_label,
            "mean_delta": d_mean,
            "ci_2_5": d_lo,
            "ci_97_5": d_hi,
            "half_width": d_hw,
        }

    return {
        "schema_version": 1,
        "bar_slug": bar_slug,
        "description": description,
        "seed_used": seed_used,
        "panel": panel,
        "n_folds": len(per_fold),
        "n_folds_finite": int(sum(1 for r in per_fold if np.isfinite(r))),
        "point_estimate": mean_r2,
        "bootstrap_n": BOOTSTRAP_N,
        "ci_2_5": lo,
        "ci_97_5": hi,
        "half_width": hw,
        "per_fold_r2": per_fold,
        "per_fold_selected_cell": per_fold_selected,
        "nested_search": nested_search,
        "cell_pick_tally": tally,
        "paired_delta_vs_baseline": delta_payload,
        "elapsed_seconds": round(time.time() - t0, 2),
        "provenance": {
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
            "python": platform.python_version(),
        },
    }


# ────────────────────────── Main ──────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Issue #523 Phase D — held-out CV scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--metrics-dir",
        type=Path,
        default=BAKEOFF_METRICS_DIR_DEFAULT,
        help=f"Bakeoff metric matrix dir (default {BAKEOFF_METRICS_DIR_DEFAULT}).",
    )
    p.add_argument(
        "--g-seed42",
        type=Path,
        default=G_SEED42_PATH,
        help="#474 G matrix at loc_ep1 (seed-42 ΔG substrate).",
    )
    p.add_argument(
        "--g-seed43",
        type=Path,
        default=G_SEED43_PATH,
        help="#523 G matrix at loc_ep1 (seed-43 ΔG substrate; Phase B output).",
    )
    p.add_argument(
        "--scoring-dir",
        type=Path,
        default=SCORING_DIR,
        help="Output directory for per-bar JSONs + forest_plot_data.json.",
    )
    p.add_argument(
        "--smoke-fold",
        type=int,
        default=None,
        help=(
            "When set, run ONLY this outer fold (0..12) end-to-end (cell-fixed + "
            "nested-search) and print the inner-selected cell. Smoke gate that "
            "verifies the inner search re-runs per outer fold rather than caching "
            "a global winner."
        ),
    )
    p.add_argument(
        "--smoke-cell",
        default=None,
        help="Informational smoke cell tag (e.g. A1); does not affect scoring.",
    )
    p.add_argument(
        "--only-headline",
        action="store_true",
        help=(
            "Compute only the cell_fixed_seed42_nonstyl_heldout headline bar (skip "
            "the seed-43 leg + nested-search + JS baseline + full-panel)."
        ),
    )
    p.add_argument(
        "--skip-seed43",
        action="store_true",
        help="Skip the seed-43 bar (use when Phase B has not produced its G matrix yet).",
    )
    p.add_argument(
        "--skip-nested",
        action="store_true",
        help="Skip the nested-search diagnostic (faster).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    args.scoring_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Fold invariants (binding from plan §4 Phase D) ──
    folds = loco_folds()
    _assert_fold_invariants(folds)
    logger.info("fold invariants PASSED: 13 folds × 12 test pairs = 156 unique pairs")

    # ── Load shared inputs ──
    delta_g_seed42 = load_delta_g(args.g_seed42)
    logger.info("loaded ΔG seed-42 (%d pairs) from %s", len(delta_g_seed42), args.g_seed42)
    prompt_tokens_pair = load_prompt_tokens()
    logger.info(
        "loaded prompt-token counts (%d pairs) from %s",
        len(prompt_tokens_pair),
        PROMPT_TOKENS_PATH,
    )

    # ── Smoke: only fold 0 ──
    if args.smoke_fold is not None:
        if args.smoke_fold < 0 or args.smoke_fold >= len(folds):
            raise ValueError(f"--smoke-fold {args.smoke_fold} out of range [0, {len(folds)})")
        # Restrict to the one fold (preserve the (S, tests, trains) signature).
        smoke_folds = [folds[args.smoke_fold]]
        # Load the cell-fixed headline distance matrix.
        distances_headline = load_distance_matrix(args.metrics_dir, *HEADLINE_CELL)
        _mean_r2, per_fold = cell_fixed_loco(
            distances_headline, delta_g_seed42, prompt_tokens_pair, smoke_folds
        )
        logger.info(
            "smoke fold %d (S=%s): cell-fixed L22 gauss_kl outer-R²=%.4f",
            args.smoke_fold,
            smoke_folds[0][0],
            per_fold[0],
        )
        if not args.skip_nested:
            _mean_nested, per_fold_nested, per_fold_selected = nested_search_loco(
                args.metrics_dir, delta_g_seed42, prompt_tokens_pair, smoke_folds
            )
            logger.info(
                "smoke fold %d nested-search: outer-R²=%.4f inner-selected=%s",
                args.smoke_fold,
                per_fold_nested[0],
                per_fold_selected[0],
            )
        return 0

    # ── Headline bar: cell-fixed L22 gauss_kl × seed-42 × non-stylized 156 ──
    distances_headline = load_distance_matrix(args.metrics_dir, *HEADLINE_CELL)
    headline_payload = score_one_bar(
        bar_slug="cell_fixed_seed42_nonstyl_heldout",
        distances=distances_headline,
        delta_g=delta_g_seed42,
        prompt_tokens=prompt_tokens_pair,
        folds=folds,
        description=(
            "L22 gauss_kl cell-fixed leave-one-source-condition-out CV R² on "
            "the 156 non-stylized pairs against seed-42 ΔG. The Goal's "
            "cell-specific answer."
        ),
        seed_used=42,
        panel="non_stylized_156",
    )
    headline_path = args.scoring_dir / "cell_fixed_seed42_nonstyl_heldout.json"
    headline_path.write_text(json.dumps(headline_payload, indent=2))
    logger.info("wrote %s (R²=%.4f)", headline_path, headline_payload["point_estimate"])

    forest_bars = [headline_payload]
    headline_per_fold = headline_payload["per_fold_r2"]

    if args.only_headline:
        logger.info("--only-headline set; skipping remaining bars.")
    else:
        # ── Seed-43 bar (POOL-isolated comparator vs headline) ──
        if not args.skip_seed43 and args.g_seed43.exists():
            delta_g_seed43 = load_delta_g(args.g_seed43)
            logger.info("loaded ΔG seed-43 (%d pairs) from %s", len(delta_g_seed43), args.g_seed43)
            seed43_payload = score_one_bar(
                bar_slug="cell_fixed_seed43_nonstyl_heldout",
                distances=distances_headline,
                delta_g=delta_g_seed43,
                prompt_tokens=prompt_tokens_pair,
                folds=folds,
                description=(
                    "Same as headline but against the seed-43 ΔG substrate. "
                    "SEED-isolated comparator vs the headline."
                ),
                seed_used=43,
                panel="non_stylized_156",
                paired_baseline_per_fold=headline_per_fold,
                paired_baseline_label="cell_fixed_seed42_nonstyl_heldout",
            )
            seed43_path = args.scoring_dir / "cell_fixed_seed43_nonstyl_heldout.json"
            seed43_path.write_text(json.dumps(seed43_payload, indent=2))
            logger.info("wrote %s (R²=%.4f)", seed43_path, seed43_payload["point_estimate"])
            forest_bars.append(seed43_payload)
        elif args.skip_seed43:
            logger.info("seed-43 bar SKIPPED (--skip-seed43)")
        else:
            logger.warning("seed-43 G matrix %s missing; skipping seed-43 bar.", args.g_seed43)

        # ── JS baseline bar ──
        try:
            distances_js = load_distance_matrix(args.metrics_dir, *JS_BASELINE_CELL)
        except FileNotFoundError as e:
            logger.warning("JS baseline matrix not found; skipping JS bar: %s", e)
            distances_js = None
        if distances_js is not None:
            js_payload = score_one_bar(
                bar_slug="js_baseline_seed42_nonstyl_heldout",
                distances=distances_js,
                delta_g=delta_g_seed42,
                prompt_tokens=prompt_tokens_pair,
                folds=folds,
                description=(
                    "JS baseline cell-fixed (final-layer last-prompt next-token "
                    "JS over full vocab) under the same fold scheme. Comparator."
                ),
                seed_used=42,
                panel="non_stylized_156",
                paired_baseline_per_fold=headline_per_fold,
                paired_baseline_label="cell_fixed_seed42_nonstyl_heldout",
            )
            js_path = args.scoring_dir / "js_baseline_seed42_nonstyl_heldout.json"
            js_path.write_text(json.dumps(js_payload, indent=2))
            logger.info("wrote %s (R²=%.4f)", js_path, js_payload["point_estimate"])
            forest_bars.append(js_payload)

        # ── Nested-search diagnostic ──
        if not args.skip_nested:
            nested_payload = score_one_bar(
                bar_slug="nested_search_seed42_nonstyl_heldout",
                distances={},  # ignored in nested mode
                delta_g=delta_g_seed42,
                prompt_tokens=prompt_tokens_pair,
                folds=folds,
                description=(
                    "Full 1737-cell inner argmax per outer fold. Same outer scheme "
                    "as headline; SELECTION-isolated diagnostic."
                ),
                seed_used=42,
                panel="non_stylized_156",
                nested_search=True,
                metrics_dir=args.metrics_dir,
                paired_baseline_per_fold=headline_per_fold,
                paired_baseline_label="cell_fixed_seed42_nonstyl_heldout",
            )
            nested_path = args.scoring_dir / "nested_search_seed42_nonstyl_heldout.json"
            nested_path.write_text(json.dumps(nested_payload, indent=2))
            logger.info(
                "wrote %s (R²=%.4f, exact L22 picks=%d/13, ridge picks=%d/13)",
                nested_path,
                nested_payload["point_estimate"],
                nested_payload["cell_pick_tally"]["exact_l22_gauss_kl_raw_last_prompt"],
                nested_payload["cell_pick_tally"]["ridge_l19_l24_gauss_kl_raw_last_prompt"],
            )
            forest_bars.append(nested_payload)

        # ── Full-panel supporting bar ──
        full_folds = full_panel_folds()
        full_payload = score_one_bar(
            bar_slug="cell_fixed_seed42_full_heldout",
            distances=distances_headline,
            delta_g=delta_g_seed42,
            prompt_tokens=prompt_tokens_pair,
            folds=full_folds,
            description=(
                "Same cell, all 16 source × 15 target = 240 ordered pairs "
                "(15 source folds, 15 pairs/fold). Supporting full-panel color."
            ),
            seed_used=42,
            panel="full_240",
        )
        full_path = args.scoring_dir / "cell_fixed_seed42_full_heldout.json"
        full_path.write_text(json.dumps(full_payload, indent=2))
        logger.info("wrote %s (R²=%.4f)", full_path, full_payload["point_estimate"])
        forest_bars.append(full_payload)

    # ── Aggregate forest_plot_data.json ──
    forest = {
        "schema_version": 1,
        "issue": 523,
        "title": "Phase D forest plot — held-out test of #502's L22 gauss_kl predictor",
        "reference_502_in_sample": {
            "label": "#502 in-sample 0.34",
            "value": 0.34,
            "note": "non-stylized 156-pair CV R² on the old pool, non-nested",
        },
        "bars": [
            {
                "slug": b["bar_slug"],
                "label": b["description"],
                "point_estimate": b["point_estimate"],
                "ci_2_5": b["ci_2_5"],
                "ci_97_5": b["ci_97_5"],
                "half_width": b["half_width"],
                "panel": b["panel"],
                "seed": b["seed_used"],
                "paired_delta_vs_baseline": b.get("paired_delta_vs_baseline"),
            }
            for b in forest_bars
        ],
        "provenance": {
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
            "python": platform.python_version(),
            "elapsed_seconds": round(time.time() - t_start, 2),
        },
    }
    forest_path = args.scoring_dir / "forest_plot_data.json"
    forest_path.write_text(json.dumps(forest, indent=2))
    logger.info("wrote %s (%d bars)", forest_path, len(forest_bars))
    logger.info("Phase D complete in %.1fs", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
