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
# JS-baseline cell — last_prompt × full-vocab next-token JS at the
# last prompt token. The bakeoff stores this with `layer=-1` (sentinel
# for "no residual-stream layer; this is a logits-space metric") at the
# canonical path
# `eval_results/issue_523/bakeoff/metrics/last_prompt__layer-1__next_token_js__raw.json`
# — matches the #502 layout
# (`eval_results/issue_502/.../last_prompt__layer-1__next_token_js__raw.json`).
# Round-1 used layer=None + a glob fallback that silently dropped to a
# warning when the file was found at a slightly different path; round-2
# fixes the path AND fails loud when the bar is missing (the five-bar
# headline figure requires this comparator).
JS_BASELINE_CELL = ("last_prompt", -1, "next_token_js", "raw")

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
    """Leave-one-source-condition-out folds over the **full 16-cond panel**.

    Panel is `ALL_CIDS` (16 conditions). Ordered-pair universe is 16 × 15 = 240
    (drop the diagonal). With the same "S_k excluded from BOTH sides" rule we
    used for `loco_folds`, we get exactly:

      - **16 folds** (one per source S_k ∈ ALL_CIDS)
      - **15 test pairs per fold** = (S_k, T) for T in ALL_CIDS, T != S_k
      - **15 × 14 = 210 train pairs per fold** (sources and targets both
        exclude S_k)
      - **Union of test pairs across folds = 16 × 15 = 240 unique pairs**

    Each of the 240 ordered pairs lives in exactly one fold (the fold whose
    source S_k matches the pair's source-side), so the fold-bootstrap remains
    fold-level independent. This is the "Supporting — full-panel" row in plan
    §5 / §6 (`cell_fixed_seed42_full_heldout`).
    """
    sources = list(ALL_CIDS)
    folds = []
    for s in sources:
        # 15 test pairs / fold (S = held-out source, T ∈ ALL_CIDS \ {S}).
        test_pairs = [(s, t) for t in ALL_CIDS if t != s]
        # 15 × 14 = 210 train pairs / fold (S_k dropped from both sides).
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

    Path convention (matches #502 layout):
    ``{metrics_dir}/{extraction_point}__layer{layer}__{metric}__{variant}.json``.
    The JS baseline cell (metric=='next_token_js') stores `layer=-1` as a
    sentinel for "logits-space, no residual-stream layer". Callers pass
    layer=-1 explicitly — round-1 used layer=None + a glob fallback, which
    silently dropped to a warning when the file was at a slightly different
    path. Round-2 uses the canonical path and fails loud on FileNotFound.

    The ``variant`` argument can encode a sub-predictor for the
    delta-spectrum family — `enumerate_inner_cells` packs
    ``"{variant}__{sub_predictor}"`` into the variant slot (e.g.
    ``"raw__mean_norm"``). We split that off at load time and look the
    sub-predictor up inside the matrix payload's ``matrices`` block
    (the convention #493 / #502 already use; see
    `issue493_extraction_metric_bakeoff._materialize_predictor_vector`).
    """
    sub_predictor: str | None = None
    if "__" in variant:
        # The bakeoff writes one FILE per (ep, layer, metric, base_variant);
        # the THREE delta_spec sub-predictors share that file via a `matrices`
        # block. Round-2 round-trips the `enumerate_inner_cells` packing.
        base_variant, sub_predictor = variant.split("__", 1)
    else:
        base_variant = variant
    p = metrics_dir / f"{extraction_point}__layer{layer}__{metric}__{base_variant}.json"
    if not p.exists():
        raise FileNotFoundError(f"distance matrix {p} missing")
    payload = json.loads(p.read_text())
    # delta_spec stores `matrices: {mean_norm, coherence, effective_dim}`,
    # other metrics store `matrix: <16x16>`.
    if "matrices" in payload:
        if sub_predictor is None:
            # Caller asked for the file but didn't name a sub-predictor → no
            # 2-D matrix to flatten. Return empty (the inner-search will
            # treat this as an N/A cell and skip).
            return {}
        mat = payload["matrices"].get(sub_predictor)
    else:
        mat = payload.get("matrix")
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


def _rank_residualize_on_covar(
    x: np.ndarray, y: np.ndarray, covar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-residualize x and y on rank(covar), matching the project's
    ``_length_partial_residualize_rank`` convention in
    ``scripts/issue493_extraction_metric_bakeoff.py:1723``.

    Returns (x_resid, y_resid) on the rank scale. If covar is rank-constant
    or polyfit fails, falls back to the bare-rank series (matching the
    convention in the reference implementation).
    """
    from scipy.stats import rankdata

    rx, ry, rc = rankdata(x), rankdata(y), rankdata(covar)
    # Ddof=0 variance is fine here; we only test "is rank-covar constant?"
    if rc.var() < 1e-12:
        return rx, ry
    ex = _safe_polyfit_residual(rx, rc)
    ey = _safe_polyfit_residual(ry, rc)
    if ex is None or ey is None:
        return rx, ry
    return ex, ey


def fold_r2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    covar_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    covar_test: np.ndarray,
) -> float:
    """R² of a length-controlled fit, train on train, score on test.

    Round-2 fix to Critical-7 (Claude self-flagged in `(d)`): the linear-
    scale `_length_controlled_fit_predict` from round-1 produced R²≈-5.08
    in the dev VM smoke (pathological extrapolation when the log(covar)
    component dominated the prediction on the held-out fold).
    Switched to the project's canonical rank-then-residualize length-control
    recipe (`scripts/issue493_extraction_metric_bakeoff._loocv_r2` /
    `_length_partial_residualize_rank`): we residualize x and y on
    rank(covar) on the TRAIN set, fit a bare OLS x_resid → y_resid, then
    on the test set we re-rank the FULL (train ∪ test) covar (so the test
    ranks are comparable to the train ranks), residualize x_test and y_test
    on those ranks, and score R² in the residualized rank space.

    R² = 1 − SSE / SST in residualized-rank space. A fold where the
    residualized y_test is constant returns NaN.
    """
    n_tr = len(x_train)
    if n_tr < 5 or len(np.unique(x_train)) < 2:
        return float("nan")
    # Build a JOINT rank space over train ∪ test so the train fit and the
    # test scoring use comparable ranks for x, y, and covar. This is the
    # standard LOCO-CV pattern in #493/#502 (`_loocv_r2`).
    x_all = np.concatenate([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    c_all = np.concatenate([covar_train, covar_test])
    finite_all = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(c_all)
    if finite_all.sum() < 5:
        return float("nan")
    # Compute the rank-residualized series on the (finite) joint set.
    x_resid, y_resid = _rank_residualize_on_covar(
        x_all[finite_all], y_all[finite_all], c_all[finite_all]
    )
    # Index masks back into the original split.
    finite_tr = np.isfinite(x_train) & np.isfinite(y_train) & np.isfinite(covar_train)
    finite_te = np.isfinite(x_test) & np.isfinite(y_test) & np.isfinite(covar_test)
    if finite_tr.sum() < 5 or finite_te.sum() < 3:
        return float("nan")
    n_train_finite = int(finite_tr.sum())
    x_resid_tr, x_resid_te = x_resid[:n_train_finite], x_resid[n_train_finite:]
    y_resid_tr, y_resid_te = y_resid[:n_train_finite], y_resid[n_train_finite:]
    if len(np.unique(x_resid_tr)) < 2:
        return float("nan")
    try:
        b_x, a_x = np.polyfit(x_resid_tr, y_resid_tr, 1)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")
    if not (np.isfinite(a_x) and np.isfinite(b_x)):
        return float("nan")
    y_hat = a_x + b_x * x_resid_te
    sst = float(np.sum((y_resid_te - y_resid_te.mean()) ** 2))
    if sst < 1e-12 or not np.isfinite(sst):
        return float("nan")
    sse = float(np.sum((y_resid_te - y_hat) ** 2))
    if not np.isfinite(sse):
        return float("nan")
    return 1.0 - sse / sst


def _assert_pair_coverage(
    distances: dict[tuple[str, str], float],
    delta_g: dict[tuple[str, str], float],
    folds: list[tuple[str, list, list]],
    *,
    label: str,
) -> None:
    """Fail loud if any fold's train ∪ test pair is missing from distances OR delta_g.

    Round-2 fix to Critical-6 (Codex). Round-1 used `dict.get(p, np.nan)`
    which silently mapped missing pairs to NaN — a partial Phase C metric
    artifact then produced a finite-looking R² over an unintended subset
    and the analyzer scored a wrong number with no warning. We surface the
    missing keys + abort.
    """
    required: set[tuple[str, str]] = set()
    for _s, tests, trains in folds:
        required.update(tests)
        required.update(trains)
    missing_dist = [p for p in required if p not in distances]
    missing_dg = [p for p in required if p not in delta_g]
    if missing_dist or missing_dg:
        # Sample the first 5 of each side to keep the error message readable.
        raise RuntimeError(
            f"Phase D pair coverage check FAILED for bar={label!r}: "
            f"{len(missing_dist)}/{len(required)} pair(s) missing from the "
            f"distances matrix, {len(missing_dg)}/{len(required)} pair(s) "
            "missing from the ΔG matrix. "
            f"First-missing-distances={missing_dist[:5]}, "
            f"first-missing-delta_g={missing_dg[:5]}. "
            "A partial Phase-C/Phase-B artifact would produce a wrong R²; "
            "regenerate the matrix or re-run Phase B/C."
        )


def cell_fixed_loco(
    distances: dict[tuple[str, str], float],
    delta_g: dict[tuple[str, str], float],
    prompt_tokens: dict[tuple[str, str], int],
    folds: list[tuple[str, list, list]],
) -> tuple[float, list[float]]:
    """Cell-fixed leave-one-source-condition-out CV R², canonical pooled aggregator.

    Returns (pooled_R2, per_fold_R2_list).

    **Aggregator (round-2 fix to Critical-7).** This follows the project's
    canonical ``_loocv_r2`` recipe in
    ``scripts/issue493_extraction_metric_bakeoff.py:1746`` — which is the
    `_loocv_r2` the Claude code-reviewer cited as "the project's canonical
    recipe" when it FAILed the round-1 mean-of-per-fold-R² aggregator
    (which produced R²≈-18 on #502's published data). The recipe:

      1. Rank-residualize ``x`` and ``y`` on rank(log(covar)) ONCE over the
         FULL union of all pairs covered by the folds (in this design that's
         all 156 non-stylized ordered pairs for the headline / 240 for the
         full panel). One residualization, one rank space.
      2. For each outer fold k: fit ``y_resid ~ x_resid`` on the train pairs
         only; predict ``y_hat[test_pairs]``.
      3. After looping all folds, compute a SINGLE pooled R² over the
         ``y_resid`` vs ``y_hat`` arrays across all 156 (or 240) predictions:
         ``R² = 1 − sum((y - y_hat)²) / sum((y - y.mean())²)``.

    The pooled-R² aggregator is statistically informative when each fold's
    test set is narrow (12 pairs of one source S). A per-fold ``y_test.var``
    can be 100× smaller than the global ``y.var`` (since one source's
    leakages are tightly clustered), so a per-fold R² blows up arbitrarily
    negative on any prediction error — even when the predictions are well-
    calibrated globally. The pooled aggregator computes R² against the
    GLOBAL variance, matching the published #502 0.34 cell-fixed result.

    Per-fold R² values are also returned so the caller can build a fold-
    bootstrap CI on the pooled estimate (paired resampling of the 13 folds:
    for each bootstrap sample, recompute pooled R² over the resampled
    folds' predictions). The fold-bootstrap CI on the pooled R² is reported
    by ``fold_bootstrap_ci_pooled`` below.

    Plan §4 Phase D Output 1 pseudocode wrote ``mean(outer_r2 across 13
    folds)`` literally, which the round-1 implementation took verbatim and
    which produced the round-1 pathology. The reviewer FAILed that aggregator
    citing the canonical recipe; round-2 honors the canonical recipe.
    """
    # ── Step 1: build the global residualized series over ALL pairs ──
    all_pairs = sorted({p for _s, tests, trains in folds for p in (*tests, *trains)})
    x_all = np.array([distances.get(p, np.nan) for p in all_pairs], dtype=np.float64)
    y_all = np.array([delta_g.get(p, np.nan) for p in all_pairs], dtype=np.float64)
    c_all = np.array([prompt_tokens.get(p, 1) for p in all_pairs], dtype=np.float64)
    finite = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(c_all) & (c_all > 0)
    if finite.sum() < 5:
        return float("nan"), [float("nan")] * len(folds)
    # Index from pair -> position in the residualized array (only finite rows).
    pair_to_idx: dict[tuple[str, str], int] = {}
    for i, p in enumerate(all_pairs):
        if finite[i]:
            pair_to_idx[p] = sum(finite[: i + 1]) - 1  # rank-position in finite array
    x_f = x_all[finite]
    y_f = y_all[finite]
    c_f = c_all[finite]
    x_resid, y_resid = _rank_residualize_on_covar(x_f, y_f, np.log(c_f))

    # ── Step 2: leave-one-source-out, fit train-residual OLS, predict test ──
    n_f = len(x_resid)
    y_hat = np.full(n_f, np.nan, dtype=np.float64)
    per_fold: list[float] = []
    per_fold_test_idx: list[list[int]] = []
    for _s, tests, trains in folds:
        train_idx = np.array([pair_to_idx[p] for p in trains if p in pair_to_idx], dtype=np.int64)
        test_idx = np.array([pair_to_idx[p] for p in tests if p in pair_to_idx], dtype=np.int64)
        per_fold_test_idx.append(list(test_idx))
        if len(train_idx) < 5 or len(test_idx) < 1:
            per_fold.append(float("nan"))
            continue
        x_tr, y_tr = x_resid[train_idx], y_resid[train_idx]
        if len(np.unique(x_tr)) < 2:
            per_fold.append(float("nan"))
            continue
        try:
            b, a = np.polyfit(x_tr, y_tr, 1)
        except (np.linalg.LinAlgError, ValueError):
            per_fold.append(float("nan"))
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            per_fold.append(float("nan"))
            continue
        y_hat[test_idx] = a + b * x_resid[test_idx]
        # Per-fold R² (also returned for diagnostic/bootstrap purposes).
        y_te = y_resid[test_idx]
        sst_fold = float(np.sum((y_te - y_te.mean()) ** 2))
        if sst_fold < 1e-18:
            per_fold.append(float("nan"))
            continue
        sse_fold = float(np.sum((y_te - y_hat[test_idx]) ** 2))
        per_fold.append(1.0 - sse_fold / sst_fold)

    # ── Step 3: pooled R² over the global y_resid vs y_hat arrays ──
    pred_mask = np.isfinite(y_hat)
    if pred_mask.sum() < 5:
        return float("nan"), per_fold
    sse = float(np.sum((y_resid[pred_mask] - y_hat[pred_mask]) ** 2))
    sst = float(np.sum((y_resid[pred_mask] - y_resid[pred_mask].mean()) ** 2))
    if sst < 1e-18 or not np.isfinite(sse):
        return float("nan"), per_fold
    pooled_r2 = 1.0 - sse / sst
    return pooled_r2, per_fold


def cell_fixed_loco_with_payload(
    distances: dict[tuple[str, str], float],
    delta_g: dict[tuple[str, str], float],
    prompt_tokens: dict[tuple[str, str], int],
    folds: list[tuple[str, list, list]],
) -> tuple[float, list[float], list[np.ndarray], list[np.ndarray]]:
    """Cell-fixed LOCO with the per-fold (y_resid, y_hat) arrays for bootstrap.

    Round-2 fix to Critical-7: the pooled-R² CI requires the per-fold y_resid
    and y_hat arrays so the bootstrap can resample folds and recompute pooled
    R² on each resampled set. Returns:
      (pooled_r2, per_fold_r2, per_fold_y_resid, per_fold_y_hat)
    where ``per_fold_y_resid[k]`` and ``per_fold_y_hat[k]`` are 1-D numpy
    arrays of equal length holding the residualized targets and predictions
    for fold k's test pairs.

    Computation is byte-identical to ``cell_fixed_loco`` up to the final
    pooled R² compute; this variant just retains the per-fold arrays.
    """
    all_pairs = sorted({p for _s, tests, trains in folds for p in (*tests, *trains)})
    x_all = np.array([distances.get(p, np.nan) for p in all_pairs], dtype=np.float64)
    y_all = np.array([delta_g.get(p, np.nan) for p in all_pairs], dtype=np.float64)
    c_all = np.array([prompt_tokens.get(p, 1) for p in all_pairs], dtype=np.float64)
    finite = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(c_all) & (c_all > 0)
    if finite.sum() < 5:
        empty = [np.array([], dtype=np.float64) for _ in folds]
        return float("nan"), [float("nan")] * len(folds), empty, list(empty)
    pair_to_idx: dict[tuple[str, str], int] = {}
    for i, p in enumerate(all_pairs):
        if finite[i]:
            pair_to_idx[p] = sum(finite[: i + 1]) - 1
    x_f = x_all[finite]
    y_f = y_all[finite]
    c_f = c_all[finite]
    x_resid, y_resid = _rank_residualize_on_covar(x_f, y_f, np.log(c_f))

    n_f = len(x_resid)
    y_hat = np.full(n_f, np.nan, dtype=np.float64)
    per_fold: list[float] = []
    per_fold_y_resid: list[np.ndarray] = []
    per_fold_y_hat: list[np.ndarray] = []
    for _s, tests, trains in folds:
        train_idx = np.array([pair_to_idx[p] for p in trains if p in pair_to_idx], dtype=np.int64)
        test_idx = np.array([pair_to_idx[p] for p in tests if p in pair_to_idx], dtype=np.int64)
        if len(train_idx) < 5 or len(test_idx) < 1:
            per_fold.append(float("nan"))
            per_fold_y_resid.append(np.array([], dtype=np.float64))
            per_fold_y_hat.append(np.array([], dtype=np.float64))
            continue
        x_tr, y_tr = x_resid[train_idx], y_resid[train_idx]
        if len(np.unique(x_tr)) < 2:
            per_fold.append(float("nan"))
            per_fold_y_resid.append(np.array([], dtype=np.float64))
            per_fold_y_hat.append(np.array([], dtype=np.float64))
            continue
        try:
            b, a = np.polyfit(x_tr, y_tr, 1)
        except (np.linalg.LinAlgError, ValueError):
            per_fold.append(float("nan"))
            per_fold_y_resid.append(np.array([], dtype=np.float64))
            per_fold_y_hat.append(np.array([], dtype=np.float64))
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            per_fold.append(float("nan"))
            per_fold_y_resid.append(np.array([], dtype=np.float64))
            per_fold_y_hat.append(np.array([], dtype=np.float64))
            continue
        y_hat[test_idx] = a + b * x_resid[test_idx]
        y_te = y_resid[test_idx]
        y_hat_te = a + b * x_resid[test_idx]
        per_fold_y_resid.append(np.array(y_te, dtype=np.float64))
        per_fold_y_hat.append(np.array(y_hat_te, dtype=np.float64))
        sst_fold = float(np.sum((y_te - y_te.mean()) ** 2))
        if sst_fold < 1e-18:
            per_fold.append(float("nan"))
            continue
        sse_fold = float(np.sum((y_te - y_hat_te) ** 2))
        per_fold.append(1.0 - sse_fold / sst_fold)

    pred_mask = np.isfinite(y_hat)
    if pred_mask.sum() < 5:
        return float("nan"), per_fold, per_fold_y_resid, per_fold_y_hat
    sse = float(np.sum((y_resid[pred_mask] - y_hat[pred_mask]) ** 2))
    sst = float(np.sum((y_resid[pred_mask] - y_resid[pred_mask].mean()) ** 2))
    if sst < 1e-18 or not np.isfinite(sse):
        return float("nan"), per_fold, per_fold_y_resid, per_fold_y_hat
    pooled_r2 = 1.0 - sse / sst
    return pooled_r2, per_fold, per_fold_y_resid, per_fold_y_hat


# ────────────────────────── Bootstrap ──────────────────────────


def fold_bootstrap_ci_pooled(
    per_fold_y_resid: list[np.ndarray],
    per_fold_y_hat: list[np.ndarray],
    n_boot: int = BOOTSTRAP_N,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Pooled-R² fold-bootstrap CI.

    For each bootstrap sample, resample `len(folds)` folds with replacement,
    concatenate their (y_resid, y_hat) arrays, and compute pooled R² over
    that concatenation. Returns (lo_2.5, hi_97.5, half_width).

    Empty folds (e.g. degenerate fit) drop out — their arrays are length 0.
    A bootstrap sample whose concatenated length < 5 yields NaN, contributing
    nothing to the percentile.
    """
    n_folds = len(per_fold_y_resid)
    if n_folds != len(per_fold_y_hat):
        raise ValueError(f"length mismatch: {n_folds} vs {len(per_fold_y_hat)}")
    if n_folds < 3:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_folds, size=n_folds)
        y_concat = np.concatenate([per_fold_y_resid[i] for i in idx])
        y_hat_concat = np.concatenate([per_fold_y_hat[i] for i in idx])
        if y_concat.size < 5:
            continue
        sst = float(np.sum((y_concat - y_concat.mean()) ** 2))
        if sst < 1e-18:
            continue
        sse = float(np.sum((y_concat - y_hat_concat) ** 2))
        r2 = 1.0 - sse / sst
        if np.isfinite(r2):
            samples.append(r2)
    if len(samples) < 10:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.array(samples, dtype=np.float64)
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    half_width = (hi - lo) / 2.0
    return lo, hi, half_width


def fold_bootstrap_ci(
    per_fold: list[float], n_boot: int = BOOTSTRAP_N, seed: int = 42
) -> tuple[float, float, float]:
    """Resample folds with replacement on the per-fold R² list; return
    (lo, hi, half_width) at 2.5/97.5.

    DEPRECATED for the cell-fixed pooled-R² headline (use
    ``fold_bootstrap_ci_pooled`` instead). Kept here for paired-Δ CIs and
    backwards-compat where per-fold R² is the natural unit.
    """
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
    """Return every (extraction_point, layer, metric, variant) cell that
    the bakeoff grid declares for `loc_ep1`.

    Round-2 fix to the Major Codex finding: round-1 derived cells from
    filename parsing (`name.split("__")` with `expected = 4`), which (a)
    missed `delta_spec` sub-predictor splits — one file but THREE grid
    entries (mean_norm / coherence / effective_dim) — and (b) silently
    excluded permutation-test variants whose 5-chunk filenames couldn't
    be split into 4. The canonical 1737-cell enumeration is the
    `bakeoff_grid.json` (the #502 layout), so we read it.

    The grid file lives at `<metrics_dir>/../bakeoff_grid.json` — the
    same convention `scripts/issue493_extraction_metric_bakeoff.py:3507`
    writes it to.
    """
    grid_path = metrics_dir.parent / "bakeoff_grid.json"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"bakeoff_grid.json missing under {grid_path}. Phase C must "
            "have produced it (issue493_extraction_metric_bakeoff "
            "--phase regress writes it at <bakeoff_root>/bakeoff_grid.json). "
            "Round-1 filename parsing is intentionally retired — the grid "
            "is the canonical 1737-cell enumeration with sub_predictor "
            "splits + variant variants the filename can't represent."
        )
    payload = json.loads(grid_path.read_text())
    cells_block = payload.get("cells", {})
    loc_ep1 = cells_block.get("loc_ep1")
    if loc_ep1 is None:
        raise RuntimeError(
            f"bakeoff_grid.json at {grid_path} has no `cells.loc_ep1` block; "
            f"available cells: {list(cells_block)}"
        )
    entries = loc_ep1.get("entries", [])
    cells: list[tuple[str, int | None, str, str]] = []
    for e in entries:
        ep = e.get("extraction_point")
        layer = e.get("layer")
        metric = e.get("metric")
        variant = e.get("variant", "raw")
        sub = e.get("sub_predictor")
        # For metrics with sub_predictor splits (delta_spec → mean_norm /
        # coherence / effective_dim), we encode the sub-predictor INTO the
        # variant slot so each (ep, layer, metric, variant') tuple is unique
        # and downstream `load_distance_matrix` can read it. The bakeoff
        # writes those three sub-predictors as separate sub-files / fields
        # inside the SAME metric matrix file, so the load path must know to
        # demux them — we surface that via the variant string and load it
        # inside load_distance_matrix as a fallback for the sub_predictor
        # case (round-2 conservative scope: keep the demux logic to the
        # delta_spec case the bakeoff is known to emit).
        variant_key = f"{variant}__{sub}" if sub is not None else variant
        cells.append((ep, layer, metric, variant_key))
    # Sanity-check the canonical 1737 grid size per plan §10 Repro Card.
    if len(cells) != 1737:
        raise RuntimeError(
            f"bakeoff_grid.json loc_ep1 has {len(cells)} entries; expected "
            "the canonical 1737-cell enumeration per plan §10 Reproducibility "
            "Card. If the grid is intentionally a subset, gate this assertion "
            "on a flag — but defaults should match #502's grid."
        )
    return cells


def nested_search_loco(  # noqa: C901 — single sequential pipeline, easier to read in-place
    metrics_dir: Path,
    delta_g: dict[tuple[str, str], float],
    prompt_tokens: dict[tuple[str, str], int],
    folds: list[tuple[str, list, list]],
    inner_score: str = "loocv",
) -> tuple[float, list[float], list[tuple[str, int | None, str, str]]]:
    """Full 1737-cell inner argmax per fold; canonical pooled aggregator on the
    outer side.

    For each outer fold k:
      - INNER: rank every candidate cell by its leave-one-source-out POOLED
        R² (the canonical `cell_fixed_loco` aggregator) computed on the
        OUTER TRAINING pairs only (an inner LOCO over the 12 non-S sources,
        restricted to train_pairs).
      - Pick the inner-best cell; collect its predictions on the outer test
        pairs into a global y_hat array.
    Outer R² = pooled R² over y_hat vs y_resid across ALL fold predictions
    (matches `cell_fixed_loco` aggregator).

    Returns (pooled_outer_R2, per_fold_R2, per_fold_selected_cell).
    inner_score: 'loocv' selects the cell whose inner-LOCO pooled R² is highest.

    Memo: pre-load every candidate cell's distance matrix once; per-fold
    inner search re-uses the cache.

    Round-2 fix (Critical-7 / Claude Major-4): aggregator swapped from
    `mean(per_fold_R²)` to canonical pooled R² for the same reason
    documented in `cell_fixed_loco` — with 12-pair narrow test folds,
    per-fold y_test.var is tiny so per-fold R² is pathological.
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

    # ── Build global residualized series over the union of all pairs ──
    all_pairs = sorted({p for _s, tests, trains in folds for p in (*tests, *trains)})
    y_all = np.array([delta_g.get(p, np.nan) for p in all_pairs], dtype=np.float64)
    c_all = np.array([prompt_tokens.get(p, 1) for p in all_pairs], dtype=np.float64)
    finite_yc = np.isfinite(y_all) & np.isfinite(c_all) & (c_all > 0)
    pair_to_global_idx: dict[tuple[str, str], int] = {}
    finite_count = 0
    for i, p in enumerate(all_pairs):
        if finite_yc[i]:
            pair_to_global_idx[p] = finite_count
            finite_count += 1
    n_global = finite_count
    if n_global < 5:
        return float("nan"), [float("nan")] * len(folds), [("", None, "", "")] * len(folds)
    y_global = y_all[finite_yc]
    log_c_global = np.log(c_all[finite_yc])

    per_fold_r2: list[float] = []
    per_fold_selected: list[tuple[str, int | None, str, str]] = []
    y_hat_global = np.full(n_global, np.nan, dtype=np.float64)

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
            inner_pooled_r2, _ = cell_fixed_loco(distances, delta_g, prompt_tokens, inner_folds)
            if not np.isfinite(inner_pooled_r2):
                continue
            if inner_pooled_r2 > best_r2:
                best_r2 = inner_pooled_r2
                best_cell = cell
        if best_cell is None:
            per_fold_r2.append(float("nan"))
            per_fold_selected.append(("", None, "", ""))
            continue

        # Score on outer test pairs using the best cell — single residualization
        # over the union of all pairs (matching cell_fixed_loco recipe), then
        # fit OLS on train pairs and predict test pairs into the global y_hat.
        distances = distances_cache[best_cell]
        x_all = np.array([distances.get(p, np.nan) for p in all_pairs], dtype=np.float64)
        finite_x = np.isfinite(x_all)
        finite_combined = finite_yc & finite_x
        if finite_combined.sum() < 5:
            per_fold_r2.append(float("nan"))
            per_fold_selected.append(best_cell)
            continue
        # The cell-specific residualization is over the same union but only
        # the rows where this cell's distance is finite.
        cell_pair_to_idx: dict[tuple[str, str], int] = {}
        ridx = 0
        for i, p in enumerate(all_pairs):
            if finite_combined[i]:
                cell_pair_to_idx[p] = ridx
                ridx += 1
        x_r = x_all[finite_combined]
        y_r = y_all[finite_combined]
        c_r = c_all[finite_combined]
        x_resid_cell, y_resid_cell = _rank_residualize_on_covar(x_r, y_r, np.log(c_r))
        train_idx = np.array(
            [cell_pair_to_idx[p] for p in train_pairs if p in cell_pair_to_idx], dtype=np.int64
        )
        test_idx = np.array(
            [cell_pair_to_idx[p] for p in test_pairs if p in cell_pair_to_idx], dtype=np.int64
        )
        if len(train_idx) < 5 or len(test_idx) < 1:
            per_fold_r2.append(float("nan"))
            per_fold_selected.append(best_cell)
            continue
        x_tr = x_resid_cell[train_idx]
        y_tr = y_resid_cell[train_idx]
        if len(np.unique(x_tr)) < 2:
            per_fold_r2.append(float("nan"))
            per_fold_selected.append(best_cell)
            continue
        try:
            b, a = np.polyfit(x_tr, y_tr, 1)
        except (np.linalg.LinAlgError, ValueError):
            per_fold_r2.append(float("nan"))
            per_fold_selected.append(best_cell)
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            per_fold_r2.append(float("nan"))
            per_fold_selected.append(best_cell)
            continue
        # Fill the global y_hat for each test pair (translating to global idx).
        for p in test_pairs:
            if p in cell_pair_to_idx and p in pair_to_global_idx:
                gi = pair_to_global_idx[p]
                ci = cell_pair_to_idx[p]
                y_hat_global[gi] = a + b * x_resid_cell[ci]
        # Per-fold R² (diagnostic; in residualized rank space, against this
        # fold's test predictions vs the global y_resid baseline).
        y_te = y_resid_cell[test_idx]
        y_hat_te = a + b * x_resid_cell[test_idx]
        sst_fold = float(np.sum((y_te - y_te.mean()) ** 2))
        if sst_fold < 1e-18:
            per_fold_r2.append(float("nan"))
        else:
            sse_fold = float(np.sum((y_te - y_hat_te) ** 2))
            per_fold_r2.append(1.0 - sse_fold / sst_fold)
        per_fold_selected.append(best_cell)
        logger.info(
            "nested-search fold %d (S=%s): inner-best=%s (test-fold R²=%.4f)",
            fold_idx,
            s,
            best_cell,
            per_fold_r2[-1],
        )

    # Pooled R² over the global y_hat vs y_global, residualized on global covar.
    # Round-2 aggregator parity with cell_fixed_loco.
    _, y_resid_global = _rank_residualize_on_covar(y_global, y_global, log_c_global)
    # Re-compute y_resid using rank-residualization of global y on global covar.
    from scipy.stats import rankdata

    ry = rankdata(y_global)
    rc = rankdata(log_c_global)
    if rc.var() < 1e-12:
        y_resid_global = ry
    else:
        ey = _safe_polyfit_residual(ry, rc)
        y_resid_global = ey if ey is not None else ry
    pred_mask = np.isfinite(y_hat_global)
    if pred_mask.sum() < 5:
        return float("nan"), per_fold_r2, per_fold_selected
    sse = float(np.sum((y_resid_global[pred_mask] - y_hat_global[pred_mask]) ** 2))
    sst = float(np.sum((y_resid_global[pred_mask] - y_resid_global[pred_mask].mean()) ** 2))
    if sst < 1e-18 or not np.isfinite(sse):
        return float("nan"), per_fold_r2, per_fold_selected
    pooled_r2 = 1.0 - sse / sst
    return pooled_r2, per_fold_r2, per_fold_selected


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
    per_fold_y_resid: list[np.ndarray] | None = None
    per_fold_y_hat: list[np.ndarray] | None = None
    if nested_search:
        if metrics_dir is None:
            raise ValueError("nested_search requires metrics_dir")
        # Coverage gate is enforced inside nested_search_loco per fold per
        # cell — see _assert_pair_coverage at the top of each cell's score.
        # We still gate ΔG coverage up-front against the outer scheme.
        _assert_pair_coverage(
            distances={p: 0.0 for _s, tests, trains in folds for p in (*tests, *trains)},
            delta_g=delta_g,
            folds=folds,
            label=f"{bar_slug}:delta_g",
        )
        mean_r2, per_fold, per_fold_selected = nested_search_loco(
            metrics_dir, delta_g, prompt_tokens, folds
        )
        tally = cell_pick_tally(per_fold_selected)
    else:
        # Round-2 fix to Critical-6 (Codex): assert every fold's required
        # pairs are present in BOTH distances and ΔG before scoring. A
        # partial artifact would otherwise produce a wrong R² via the
        # `.get(p, np.nan)` fallback below.
        _assert_pair_coverage(distances, delta_g, folds, label=bar_slug)
        # Use the payload variant so we can pooled-bootstrap the CI.
        mean_r2, per_fold, per_fold_y_resid, per_fold_y_hat = cell_fixed_loco_with_payload(
            distances, delta_g, prompt_tokens, folds
        )
        per_fold_selected = None
        tally = None

    # Pooled-R² CI if we have the payload (cell-fixed bars); fall back to
    # the mean-of-per-fold bootstrap for nested-search where the per-fold
    # arrays span DIFFERENT cells (concatenating them would mix predictors).
    if per_fold_y_resid is not None and per_fold_y_hat is not None:
        lo, hi, hw = fold_bootstrap_ci_pooled(per_fold_y_resid, per_fold_y_hat)
    else:
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
        # The five-bar forest plot REQUIRES this comparator (plan §6 Hero).
        # Round-1 caught FileNotFound here and silently logged "skipping" —
        # that turned a missing artifact into an interpretable-looking
        # four-bar plot with no JS baseline. Round-2 raises so the
        # five-bar headline figure is either complete or the run aborts.
        distances_js = load_distance_matrix(args.metrics_dir, *JS_BASELINE_CELL)
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
                "(16 source folds, 15 pairs/fold). Supporting full-panel color."
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
