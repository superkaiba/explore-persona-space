#!/usr/bin/env python
"""Fits driver for task #2054: per-cell ambient-basis fit at layer 19.

For each cell — keyed on ALL FOUR lattice axes (variant/identity,
condition/phase, framing/form, model; C6 — `issue2054_forms.cell_key`) times
arm — read the .npz activations the capture driver
(`scripts/issue2054_capture.py`) wrote, join per-fold to the shared conv_id →
fold map artifact from Unit A (`eval_results/issue_2054/shared_fold_map.json`,
K=5, conversation-grouped), and fit the per-fold ambient-basis ridge map:

- **context arm**: ``v_A ≈ M · v_C`` (last-prompt-token → mean-answer state).
- **prefix arm**: ``v_A ≈ M · v_P`` (last-prefix-token → mean-answer state);
  rows whose capture recorded ``v_P_present == False`` are DROPPED per the
  Unit C null-recording contract, never coerced.

Reports, per fold + pooled across folds:

- Held-out ambient R² (the plan §6 headline DV).
- Identity+learned-bias baseline R² (`v̂ = x + b`, b = train-fold mean of y−x)
  via the canonical helper `analysis/mapping_baselines.identity_bias_predict`.
- kNN retrieval acc@k over the held-out pool (euclidean + cosine, k in {1,5,10})
  via `analysis/mapping_baselines.knn_retrieval`.
- Shuffled-answer matched-capacity null R²: refit the same ridge with the
  training answer rows PERMUTED (breaks the context→answer pairing while
  keeping capacity fixed); ``--n-null-draws`` (default 200) draws, batched via
  a shared factorization (never a per-draw serial fit; `.claude/rules/vectorize-many-cell-fits.md`).
- Reduced-basis (train-fold PCA k=1024) diagnostic R² per cell (the #1887
  recipe) alongside the ambient fit, so the writeup can contrast estimators.
- Per-comparison bootstrap CI over CONVERSATIONS within the equalized-down
  intersection (NOT K=5 fold-resample — statistics-critic concern #2).

Kill-gate outcomes (v7→v8 statistics-critic Must-Fix, plan §4/§7):

- **Kill gate 4** — min conv_id intersection across compared cells < 4,480:
  refuses to fit and reports; equalize-down at n<4480 pushes n_train=0.8·n
  below d=3,584 and re-enters the estimator-degenerate regime.
- **Kill gate 5** — (b) vs (d) answer-length KS D > 0.30 OR mean-ratio outside
  [0.25, 4.0] within a (character, model) pair: refuses to fit and reports;
  length-stratified refit is not tractable at the row count.

Emits `[phase=fits]` log lines terminating in `[phase=done]` on graceful
completion. Uploads per-cell fit JSONs (`{cell_key}.json`) to HF
`issue2054_lattice/fits/` (best-effort, non-fatal). The
`--dry-run` (and `--pilot`) mode exercises the CLI + fit/null/baseline/kNN
pipeline on a tiny slice (1 fold, ≤1 cell), skips uploads, and writes the
diagnostics JSON to a scratch tree.

Exit 0 on success. Exit 1 on fit / HF / preflight failure. Exit 2 on missing
input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue2054_forms as forms  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
TASK_PREFIX = "issue2054_lattice"

# Ambient hidden size at layer 19 (Qwen2.5-7B config.json). n_train per fold at
# n=4480 is 0.8·4480 = 3,584 = d — the ambient-basis floor kill gate 4 defends
# (plan §7 v7→v8 statistics-critic Must-Fix).
D_AMBIENT = 3584
KILL_GATE_4_MIN_INTERSECTION = 4480
KILL_GATE_5_KS_D_THRESHOLD = 0.30
KILL_GATE_5_RATIO_LO = 0.25
KILL_GATE_5_RATIO_HI = 4.0

# Reduced-basis diagnostic k (parent #1887's PCA basis size for the estimator
# comparison; plan §11 "reduced-basis diagnostic per cell").
REDUCED_BASIS_K = 1024

# GCV λ grid — the shared project convention (`ridge_fit_predict_fast` default,
# `issue_779.fit_h`, #823 grid).
DEFAULT_LAMBDAS = np.logspace(-2, 4, 13)

# Bootstrap CI draws for conversation-within-intersection resampling
# (statistics-critic concern #2 — conv-level bootstrap over the equalized
# intersection is the coarse-grain-safe CI; not K=5 fold-resample).
DEFAULT_BOOTSTRAP_DRAWS = 200

# Default sub-panel for cell-comparison equalize-down + kill gates. The
# canonical (a,b,c,d) 2x2 per (character, model) pair maps to specific variant
# slugs on this task; when the driver runs a single cell (the smoke path) the
# gates only score availability, not comparison — see `_equalize_and_gate`.
# The cell (c) `char_*_op*` variants (phase_d output) are IN the default so the
# 2x2's (c) leg is discoverable without operator memory (C6 review note).
DEFAULT_VARIANTS = (
    "char_helios",
    "char_wren",
    "char_dana",
    "char_vex",
    "conversation_paired_stories_assistant",
    "char_helios_op",
    "char_helios_op_base",
    "char_wren_op",
    "char_wren_op_base",
    "char_dana_op",
    "char_dana_op_base",
    "char_vex_op",
    "char_vex_op_base",
)

DEFAULT_MODELS = ("qwen2.5-7b", "qwen2.5-7b-instruct")

# Condition (phase) + framing (form) axes — C6: cells are keyed on all four
# lattice axes; `_resolve_cells` keeps only combinations whose .npz exists, so
# the full default product is safe to enumerate.
DEFAULT_CONDITIONS = forms.CONDITIONS
DEFAULT_FORMS = forms.FORMS

# Kill-gate 5 pairs cell (b) with cell (d): SAME (variant, form, model), the
# inserted vs on_policy conditions. cell_c cells have no length-parity peer.
_GATE5_PEER_CONDITION = {"inserted": "on_policy", "on_policy": "inserted"}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers


def _log(msg: str) -> None:
    print(f"[phase=fits] {msg}", flush=True)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _load_fold_map(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"shared_fold_map not found: {path}")
    with path.open(encoding="utf-8") as f:
        d = json.load(f)
    for key in ("fold_of", "k", "seed"):
        if key not in d:
            raise ValueError(f"shared_fold_map missing {key!r}: {path}")
    return d


def _find_activation_path(
    activations_dir: Path, variant: str, condition: str, form: str, model: str
) -> Path | None:
    """Locate the .npz per the capture layout:
    `<activations-dir>/<variant>/<cell_key>.npz` (4-axis key, C6).

    Falls back to any *.npz directly under `<activations-dir>` when the cell
    file is missing (smoke fixture convention, matching capture's _flat) — the
    caller dedupes fallback hits so one flat fixture never multiplies into a
    cell per (condition, form) combination.
    """
    key = forms.cell_key(variant, condition, form, model)
    canonical = activations_dir / variant / f"{key}.npz"
    if canonical.is_file() and canonical.stat().st_size > 0:
        return canonical
    # smoke fixture fallback
    if activations_dir.is_dir():
        for p in sorted(activations_dir.glob("*.npz")):
            if p.stat().st_size > 0:
                return p
    return None


def _load_activation_npz(path: Path) -> dict | None:
    """Return {conv_id: np.ndarray of length D_out} arrays for v_C / v_A / v_P
    plus a v_P_present boolean mask; None if the file is empty (dry-run shell).
    """
    if path.stat().st_size == 0:
        return None
    z = np.load(path, allow_pickle=False)
    conv_ids = [str(x) for x in z["conv_id"]]
    v_C = np.asarray(z["v_C"], dtype=np.float32)
    v_A = np.asarray(z["v_A"], dtype=np.float32)
    v_P = np.asarray(z["v_P"], dtype=np.float32)
    v_P_present = np.asarray(z["v_P_present"], dtype=bool)
    if not (len(conv_ids) == v_C.shape[0] == v_A.shape[0] == v_P.shape[0] == v_P_present.shape[0]):
        raise ValueError(f"activation .npz shape mismatch in {path}")
    return {
        "conv_ids": conv_ids,
        "v_C": v_C,
        "v_A": v_A,
        "v_P": v_P,
        "v_P_present": v_P_present,
    }


def _find_capture_diagnostics(variant: str, condition: str, form: str, model: str) -> Path | None:
    """Capture emits per-cell diagnostics at eval_results/issue_2054/capture_diagnostics/
    named by the 4-axis cell key (C6). The `per_row` block is where DV 7 /
    kill-gate 5 read answer-length parity for a (character, model) pair.
    """
    key = forms.cell_key(variant, condition, form, model)
    p = _REPO_ROOT / "eval_results/issue_2054/capture_diagnostics" / f"{key}.json"
    return p if p.is_file() else None


def _load_capture_diagnostics(variant: str, condition: str, form: str, model: str) -> dict | None:
    p = _find_capture_diagnostics(variant, condition, form, model)
    if p is None:
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Ridge fits (batched, ambient basis, shared Gram factorization per fold)


def _standardize_train_apply(
    X_train: np.ndarray, X_eval: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmu = X_train.mean(axis=0)
    xsd = X_train.std(axis=0) + 1e-9  # population std (numpy default; matches project convention)
    return (X_train - xmu) / xsd, (X_eval - xmu) / xsd, xmu, xsd


def _ridge_gcv_fit_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    lambdas: np.ndarray = DEFAULT_LAMBDAS,
    dof_cap: float = 0.9,
) -> tuple[np.ndarray, dict]:
    """Ambient-basis GCV-ridge fit-then-apply with a dof-cap safeguard.

    Mirrors :func:`explore_persona_space.experiments.issue_779.fit_h.ridge_fit_predict`
    (standardize-X, center-Y, dual Gram-space solve via SVD, un-center); the
    dof-cap is #1887's post-GCV guard against under-determined selection
    (`dof/n_train ≤ dof_cap`). No inner K-fold; the OUTER K=5 fold split is the
    caller's job (`_fit_arm_cell`).
    """
    Xtr, Xev, _, _ = _standardize_train_apply(X_train.astype(np.float64), X_eval.astype(np.float64))
    Ytr = Y_train.astype(np.float64)
    ymu = Ytr.mean(axis=0)
    Ytr_c = Ytr - ymu

    n_train = Xtr.shape[0]
    # SVD of the standardized train design; s2 is the eigenspectrum of X X^T.
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c

    # GCV over the λ grid, with a dof-cap safeguard (#1887): filter out any λ
    # whose selected dof exceeds `dof_cap · n_train`. Fallback: if EVERY λ is
    # over cap, keep the tightest λ that respects the cap; if none do
    # (degenerate n_train < d without regularization headroom), use the
    # smallest-dof (largest-λ) point and mark the fit degenerate.
    best_lam = float(lambdas[0])
    best_gcv = float("inf")
    best_dof = float("nan")
    dof_over_cap = True  # flips False when at least one λ satisfies the cap
    for lam in lambdas:
        lam = float(lam)
        filt = s2 / (s2 + lam)
        dof = float(filt.sum())
        if dof / n_train <= dof_cap:
            dof_over_cap = False
        # RSS via the eigenpath (no full train-fit reconstruction):
        # rss = ||Y_c||^2 − Σ (2f − f^2) · ||UtY_k||^2.
        row_energy = (UtY**2).sum(axis=1)  # (r,)
        rss = float((Ytr_c**2).sum() - ((2 * filt - filt**2) * row_energy).sum())
        denom = (n_train - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        # Prefer a λ satisfying the cap; only fall back if none exist.
        if dof / n_train <= dof_cap and gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam
            best_dof = dof
    if best_gcv == float("inf"):
        # No λ respected the cap — pick the largest λ (smallest dof).
        best_lam = float(lambdas[-1])
        filt = s2 / (s2 + best_lam)
        best_dof = float(filt.sum())
        best_gcv = float("nan")

    # Reconstruct primal weights at best_lam:  W = V diag(s/(s^2 + λ)) UtY
    filt = s / (s2 + best_lam)
    W = (Vt.T * filt) @ UtY  # (d, D_out)
    preds = Xev @ W + ymu
    info = {
        "best_lambda": best_lam,
        "dof": best_dof,
        "dof_cap": dof_cap,
        "dof_over_cap": bool(dof_over_cap),
        "gcv": best_gcv,
        "n_train": int(n_train),
        "d_in": int(Xtr.shape[1]),
        "d_out": int(Ytr.shape[1]),
    }
    return preds, info


def _r2_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multivariate held-out R² (uniform variance-weighting via sum-of-squares).

    R² = 1 - Σ_ij (y_ij − ŷ_ij)² / Σ_ij (y_ij − ȳ_j)²
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_mean = y_true.mean(axis=0)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_mean) ** 2).sum())
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _reduced_basis_r2(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    *,
    k: int = REDUCED_BASIS_K,
    lambdas: np.ndarray = DEFAULT_LAMBDAS,
    dof_cap: float = 0.9,
) -> tuple[float, dict]:
    """Train-fold PCA-k reduced-basis diagnostic (parent #1887 recipe).

    Projects both X_train and X_eval onto the top-k train-fold PCA components,
    then runs the same GCV-ridge fit-then-apply in the reduced basis. Returns
    (held-out R², info).
    """
    k_use = min(k, X_train.shape[0], X_train.shape[1])
    Xtr64 = X_train.astype(np.float64)
    xmu = Xtr64.mean(axis=0)
    Xtr_c = Xtr64 - xmu
    # Right singular vectors of the CENTERED design (n_train, d) form the PCA
    # basis in feature space. full_matrices=False → V is (r, d), r ≤ n_train.
    _, _, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
    Vk = Vt[:k_use, :]  # (k_use, d)
    Xtr_red = Xtr_c @ Vk.T  # (n_train, k_use)
    Xev_red = (X_eval.astype(np.float64) - xmu) @ Vk.T  # (n_eval, k_use)
    preds, info = _ridge_gcv_fit_predict(
        Xtr_red, Y_train, Xev_red, lambdas=lambdas, dof_cap=dof_cap
    )
    r2 = _r2_matrix(Y_eval, preds)
    info["reduced_k"] = int(k_use)
    return r2, info


def _shuffled_answer_null_r2(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    *,
    n_draws: int,
    seed: int,
    lambdas: np.ndarray = DEFAULT_LAMBDAS,
    dof_cap: float = 0.9,
) -> tuple[np.ndarray, dict]:
    """Shuffled-answer matched-capacity null (plan §6 DV 4).

    On each draw, permute the ROWS of Y_train (breaks the context→answer
    pairing while keeping capacity fixed), refit the same ridge, score against
    the UNPERMUTED held-out Y_eval. Returns (per-draw R² array, info).

    Batching: the standardize-X + SVD of X_train is COMPUTED ONCE per fold and
    REUSED across draws (only UtY changes per draw), so N draws cost N mat-vec
    updates over the eigenbasis + one prediction pass — never N full SVDs
    (`.claude/rules/vectorize-many-cell-fits.md` § "many-DRAW closed-form
    statistical loops"). The λ grid GCV loop still runs per draw (draws pick
    different λ), but re-uses the same eigenbasis.
    """
    Xtr, Xev, _, _ = _standardize_train_apply(X_train.astype(np.float64), X_eval.astype(np.float64))
    Ytr = Y_train.astype(np.float64)
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    s2 = s**2
    n_train = Xtr.shape[0]
    tot_y_sq = None  # recomputed per draw (permutation preserves Frobenius, but Y_c differs)

    rng = np.random.default_rng(seed)
    r2s = np.empty(n_draws, dtype=np.float64)
    n_dof_over_cap = 0
    for d in range(n_draws):
        perm = rng.permutation(Ytr.shape[0])
        Y_shuffled = Ytr[perm]
        ymu = Y_shuffled.mean(axis=0)
        Ytr_c = Y_shuffled - ymu
        UtY = U.T @ Ytr_c
        tot_y_sq = float((Ytr_c**2).sum())
        # GCV over λ:
        row_energy = (UtY**2).sum(axis=1)
        best_lam = float(lambdas[0])
        best_gcv = float("inf")
        this_over_cap = True
        for lam in lambdas:
            lam = float(lam)
            filt = s2 / (s2 + lam)
            dof = float(filt.sum())
            if dof / n_train <= dof_cap:
                this_over_cap = False
            rss = tot_y_sq - float(((2 * filt - filt**2) * row_energy).sum())
            denom = (n_train - dof) ** 2
            gcv = rss / denom if denom > 1e-12 else float("inf")
            if dof / n_train <= dof_cap and gcv < best_gcv:
                best_gcv = gcv
                best_lam = lam
        if best_gcv == float("inf"):
            best_lam = float(lambdas[-1])
        filt = s / (s2 + best_lam)
        W = (Vt.T * filt) @ UtY  # (d, D_out)
        preds = Xev @ W + ymu
        r2s[d] = _r2_matrix(Y_eval, preds)
        if this_over_cap:
            n_dof_over_cap += 1
    info = {
        "n_draws": int(n_draws),
        "n_dof_over_cap": int(n_dof_over_cap),
        "seed": int(seed),
    }
    return r2s, info


def _bootstrap_conv_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> dict:
    """Bootstrap CI over the CONVERSATIONS within the held-out set.

    Statistics-critic concern #2: K=5 fold-resample is coarse; resample
    conversations WITHIN the equalized intersection instead. Reports the 95%
    percentile CI over the sample means of the per-conversation squared errors
    normalized by the (fixed) per-conversation total variance — i.e. the
    per-conversation R² BOOTSTRAP CI on the same n rows the fit was scored on.

    Returns:
      dict with keys r2_boot_median, r2_boot_lo, r2_boot_hi, n_draws.
    """
    n = y_true.shape[0]
    if n < 2:
        return {
            "r2_boot_median": float("nan"),
            "r2_boot_lo": float("nan"),
            "r2_boot_hi": float("nan"),
            "n_draws": 0,
        }
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_mean = y_true.mean(axis=0)  # fixed on the full held-out set (Bootstrap RE-SAMPLES rows)
    per_row_ss_res = ((y_true - y_pred) ** 2).sum(axis=1)  # (n,)
    per_row_ss_tot = ((y_true - y_mean) ** 2).sum(axis=1)  # (n,)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    ss_res_boot = per_row_ss_res[idx].sum(axis=1)
    ss_tot_boot = per_row_ss_tot[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_boot = 1.0 - ss_res_boot / ss_tot_boot
    r2_boot = r2_boot[np.isfinite(r2_boot)]
    if r2_boot.size == 0:
        return {
            "r2_boot_median": float("nan"),
            "r2_boot_lo": float("nan"),
            "r2_boot_hi": float("nan"),
            "n_draws": 0,
        }
    return {
        "r2_boot_median": float(np.median(r2_boot)),
        "r2_boot_lo": float(np.percentile(r2_boot, 2.5)),
        "r2_boot_hi": float(np.percentile(r2_boot, 97.5)),
        "n_draws": int(r2_boot.size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell driver


def _select_arm(activations: dict, arm: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, Y, conv_ids) for the arm-specific input → v_A mapping.

    context arm  →  X = v_C, Y = v_A;  rows kept: all.
    prefix arm   →  X = v_P, Y = v_A;  rows kept: v_P_present is True.
    """
    conv_ids = list(activations["conv_ids"])
    v_A = activations["v_A"]
    if arm == "context":
        return activations["v_C"].astype(np.float32), v_A.astype(np.float32), conv_ids
    if arm == "prefix":
        mask = activations["v_P_present"]
        keep_idx = np.nonzero(mask)[0]
        v_P = activations["v_P"][keep_idx]
        v_A_k = v_A[keep_idx]
        kept_ids = [conv_ids[i] for i in keep_idx.tolist()]
        return v_P.astype(np.float32), v_A_k.astype(np.float32), kept_ids
    raise ValueError(f"unknown arm {arm!r}")


def _fold_split(
    conv_ids: list[str], fold_of: dict, k: int, restrict_ids: set[str] | None = None
) -> list[list[int]]:
    """Return per-fold LIST of ROW INDICES (into the arm's `conv_ids` order),
    following the shared fold map. Rows whose conv_id is not in the map are
    dropped from EVERY fold (they cannot be assigned to a group).
    """
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, cid in enumerate(conv_ids):
        if restrict_ids is not None and cid not in restrict_ids:
            continue
        if cid not in fold_of:
            continue
        folds[int(fold_of[cid])].append(i)
    return folds


def _fit_arm_cell(
    variant: str,
    model: str,
    arm: str,
    activations: dict,
    fold_map: dict,
    *,
    restrict_ids: set[str] | None,
    n_null_draws: int,
    seed: int,
    pilot: bool,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> dict:
    """Run K folds of the arm-specific ambient fit + baselines + null + kNN.

    Returns a per-fold + pooled digest dict ready to serialize.
    """
    k = int(fold_map["k"])
    fold_of = fold_map["fold_of"]

    X_all, Y_all, arm_conv_ids = _select_arm(activations, arm)
    folds = _fold_split(arm_conv_ids, fold_of, k, restrict_ids=restrict_ids)
    fold_sizes = [len(f) for f in folds]

    if all(s == 0 for s in fold_sizes):
        return {
            "variant": variant,
            "model": model,
            "arm": arm,
            "status": "no-rows-after-fold-join",
            "fold_sizes": fold_sizes,
        }

    d_in = X_all.shape[1]
    d_out = Y_all.shape[1]
    if arm == "context" and d_in != D_AMBIENT:
        _log(
            f"WARN variant={variant} model={model} arm={arm} d_in={d_in} != {D_AMBIENT} "
            f"(Qwen2.5-7B hidden). Ambient-basis floor arithmetic is variant-dependent."
        )

    fold_range = range(min(1, k)) if pilot else range(k)

    per_fold: list[dict] = []
    r2_ambient: list[float] = []
    r2_identity: list[float] = []
    r2_reduced: list[float] = []
    null_r2_all: list[np.ndarray] = []
    knn_records: list[dict] = []
    boot_records: list[dict] = []

    t_fold0 = time.time()
    for fold_i in fold_range:
        val_idx = np.array(folds[fold_i], dtype=np.int64)
        train_idx = np.array(
            [
                i
                for i in range(len(arm_conv_ids))
                if fold_of.get(arm_conv_ids[i]) is not None
                and (restrict_ids is None or arm_conv_ids[i] in restrict_ids)
                and int(fold_of[arm_conv_ids[i]]) != fold_i
            ],
            dtype=np.int64,
        )
        if val_idx.size == 0 or train_idx.size == 0:
            per_fold.append(
                {
                    "fold": fold_i,
                    "status": "skipped-empty-fold",
                    "n_train": int(train_idx.size),
                    "n_val": int(val_idx.size),
                }
            )
            continue

        X_train = X_all[train_idx]
        Y_train = Y_all[train_idx]
        X_val = X_all[val_idx]
        Y_val = Y_all[val_idx]

        preds, info_ambient = _ridge_gcv_fit_predict(X_train, Y_train, X_val)
        r2_amb = _r2_matrix(Y_val, preds)
        r2_ambient.append(r2_amb)

        # Identity+learned-bias baseline (CLAUDE.md standing rule).
        if d_in == d_out:
            pred_identity = identity_bias_predict(X_train, Y_train, X_val)
            r2_id = _r2_matrix(Y_val, pred_identity)
            r2_identity.append(r2_id)
        else:
            r2_id = float("nan")

        # Reduced-basis diagnostic (per-cell always, plan §11).
        r2_red, info_red = _reduced_basis_r2(X_train, Y_train, X_val, Y_val, k=REDUCED_BASIS_K)
        r2_reduced.append(r2_red)

        # kNN retrieval read against the fold-scoped held-out pool.
        knn_euclid = knn_retrieval(preds, Y_val, metric="euclidean")
        knn_cos = knn_retrieval(preds, Y_val, metric="cosine")

        # Shuffled-answer matched-capacity null.
        null_r2s, info_null = _shuffled_answer_null_r2(
            X_train,
            Y_train,
            X_val,
            Y_val,
            n_draws=n_null_draws,
            seed=seed + fold_i,
        )
        null_r2_all.append(null_r2s)

        # Conversation-within-intersection bootstrap CI on this fold's held out.
        boot_info = _bootstrap_conv_ci(
            Y_val, preds, n_draws=bootstrap_draws, seed=seed + 10_000 + fold_i
        )
        boot_records.append(boot_info)

        elapsed = time.time() - t_fold0
        print(
            f"[phase=fits] unit {fold_i + 1}/{k} "
            f"cell={variant}/{model}/{arm} n_train={train_idx.size} n_val={val_idx.size} "
            f"R2={r2_amb:.4f} id+b={r2_id:.4f} redk={r2_red:.4f} "
            f"null_med={float(np.median(null_r2s)):.4f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

        per_fold.append(
            {
                "fold": fold_i,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "r2_ambient": r2_amb,
                "r2_identity_bias": r2_id,
                "r2_reduced_k1024": r2_red,
                "info_ambient": info_ambient,
                "info_reduced": info_red,
                "info_null": info_null,
                "null_r2_stats": {
                    "min": float(null_r2s.min()),
                    "median": float(np.median(null_r2s)),
                    "mean": float(null_r2s.mean()),
                    "max": float(null_r2s.max()),
                    "p95": float(np.percentile(null_r2s, 95)),
                },
                "knn_euclidean": {
                    "acc_at_k": knn_euclid["acc_at_k"],
                    "chance_at_k": knn_euclid["chance_at_k"],
                    "median_rank": knn_euclid["median_rank"],
                    "mrr": knn_euclid["mrr"],
                    "n_pool": knn_euclid["n_pool"],
                },
                "knn_cosine": {
                    "acc_at_k": knn_cos["acc_at_k"],
                    "chance_at_k": knn_cos["chance_at_k"],
                    "median_rank": knn_cos["median_rank"],
                    "mrr": knn_cos["mrr"],
                    "n_pool": knn_cos["n_pool"],
                },
                "bootstrap_conv_ci": boot_info,
            }
        )
        knn_records.append(
            {
                "euclid_acc1": float(knn_euclid["acc_at_k"][1]),
                "cos_acc1": float(knn_cos["acc_at_k"][1]),
                "n_pool": int(knn_euclid["n_pool"]),
            }
        )

    pooled = {}
    if r2_ambient:
        pooled["r2_ambient_mean"] = float(np.mean(r2_ambient))
        pooled["r2_ambient_median"] = float(np.median(r2_ambient))
    if r2_identity:
        pooled["r2_identity_bias_mean"] = float(np.mean(r2_identity))
    if r2_reduced:
        pooled["r2_reduced_k1024_mean"] = float(np.mean(r2_reduced))
    if null_r2_all:
        stacked = np.concatenate(null_r2_all)
        pooled["null_r2_pooled_median"] = float(np.median(stacked))
        pooled["null_r2_pooled_p95"] = float(np.percentile(stacked, 95))
        pooled["null_r2_pooled_n_draws"] = int(stacked.size)

    return {
        "variant": variant,
        "model": model,
        "arm": arm,
        "status": "ok" if per_fold else "no-folds-ran",
        "d_in": int(d_in),
        "d_out": int(d_out),
        "n_conv_ids_in_arm": int(len(arm_conv_ids)),
        "n_conv_ids_in_folds": int(sum(fold_sizes)),
        "fold_sizes": fold_sizes,
        "per_fold": per_fold,
        "pooled": pooled,
        "n_null_draws": int(n_null_draws),
        "reduced_basis_k": REDUCED_BASIS_K,
        "seed": int(seed),
        "pilot": bool(pilot),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Kill gates 4 & 5


def _min_pair_intersection(
    conv_ids_by_cell: dict[str, list[str]], restrict_ids: set[str] | None = None
) -> int:
    """v4-new kill gate 4: min conv_id intersection across compared cells.

    conv_ids_by_cell: {cell_key: conv_ids_list}. Returns the size of the shared
    intersection across ALL provided cells (post `restrict_ids` filter).
    """
    if not conv_ids_by_cell:
        return 0
    sets: list[set[str]] = []
    for _cell, ids in conv_ids_by_cell.items():
        s = set(ids)
        if restrict_ids is not None:
            s &= restrict_ids
        sets.append(s)
    return len(set.intersection(*sets)) if sets else 0


def _answer_length_ks_from_diagnostics(
    diag_b: dict | None, diag_d: dict | None
) -> tuple[float, float, dict]:
    """v4-new kill gate 5: KS D + mean-ratio across (b) vs (d) answer-token
    lengths.

    Reads the `answer_token_length_stats` blocks from the Unit C capture
    diagnostics. When both are missing (smoke without diagnostics) returns
    (NaN, NaN, {reason=missing}). When both are present, computes KS D
    directly on the SORTED empirical CDFs from the recorded per-cell rows —
    which the diagnostics carry via `per_row[*].answer_hi - answer_lo`.
    """
    if diag_b is None or diag_d is None:
        return (
            float("nan"),
            float("nan"),
            {"status": "missing-diagnostics"},
        )
    b_lens = [
        int(r.get("answer_hi", 0) - r.get("answer_lo", 0))
        for r in diag_b.get("per_row", [])
        if r.get("status") == "ok"
    ]
    d_lens = [
        int(r.get("answer_hi", 0) - r.get("answer_lo", 0))
        for r in diag_d.get("per_row", [])
        if r.get("status") == "ok"
    ]
    if not b_lens or not d_lens:
        return (float("nan"), float("nan"), {"status": "empty-length-arrays"})
    try:
        from scipy.stats import ks_2samp

        ks_stat, ks_p = ks_2samp(b_lens, d_lens)
        ks_d = float(ks_stat)
    except Exception as exc:  # noqa: BLE001
        return (float("nan"), float("nan"), {"status": f"ks-error: {exc}"})
    mean_b = float(np.mean(b_lens)) if b_lens else float("nan")
    mean_d = float(np.mean(d_lens)) if d_lens else float("nan")
    ratio = float("nan")
    if np.isfinite(mean_b) and np.isfinite(mean_d) and mean_d > 0:
        ratio = mean_b / mean_d
    return (
        ks_d,
        ratio,
        {
            "status": "computed",
            "n_b": len(b_lens),
            "n_d": len(d_lens),
            "mean_b": mean_b,
            "mean_d": mean_d,
            "ks_p_value": float(ks_p),
        },
    )


def _evaluate_kill_gates(
    variant: str,
    condition: str,
    form: str,
    model: str,
    conv_ids_this_cell: list[str],
    peer_cells_conv_ids: dict[str, list[str]] | None,
    diag_this: dict | None,
    peer_diag: dict | None,
    gate5_peer_cell: str | None,
) -> dict:
    """Return kill-gate outcomes for this (variant, condition, form, model) cell.

    peer_cells_conv_ids: {peer_cell_key: conv_ids} for the 2x2 (a,b,c,d) pair;
      when None (single-cell driver run), gate 4 reports a single-cell floor
      check against the ambient-basis n_train ≥ d requirement.
    peer_diag: the (b)<->(d) length-parity peer's capture diagnostics — the
      SAME (variant, form, model) under the paired condition (inserted <->
      on_policy); None for cell_c cells (no length-parity peer) or when the
      peer was never captured (KS gate reports missing-diagnostics).
    gate5_peer_cell: the peer's cell key, for the report (None when N/A).
    """
    intersection_size = len(conv_ids_this_cell)
    peer_report: dict = {}
    if peer_cells_conv_ids:
        peer_report["peer_cells"] = sorted(peer_cells_conv_ids.keys())
        intersection_size = _min_pair_intersection(
            {"this": conv_ids_this_cell, **peer_cells_conv_ids}
        )
    gate4_fire = intersection_size < KILL_GATE_4_MIN_INTERSECTION
    ks_d, ratio, ks_info = _answer_length_ks_from_diagnostics(diag_this, peer_diag)
    gate5_fire = False
    if np.isfinite(ks_d):
        gate5_fire = bool(
            (ks_d > KILL_GATE_5_KS_D_THRESHOLD)
            or (np.isfinite(ratio) and not (KILL_GATE_5_RATIO_LO <= ratio <= KILL_GATE_5_RATIO_HI))
        )
    return {
        "variant": variant,
        "condition": condition,
        "form": form,
        "model": model,
        "kill_gate_5_peer_cell": gate5_peer_cell,
        "min_pair_intersection": int(intersection_size),
        "kill_gate_4_threshold": KILL_GATE_4_MIN_INTERSECTION,
        "kill_gate_4_fire": bool(gate4_fire),
        "kill_gate_5_ks_d": float(ks_d) if np.isfinite(ks_d) else None,
        "kill_gate_5_ratio": float(ratio) if np.isfinite(ratio) else None,
        "kill_gate_5_ks_d_threshold": KILL_GATE_5_KS_D_THRESHOLD,
        "kill_gate_5_ratio_bounds": [KILL_GATE_5_RATIO_LO, KILL_GATE_5_RATIO_HI],
        "kill_gate_5_fire": bool(gate5_fire),
        "kill_gate_5_ks_info": ks_info,
        **peer_report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Uploads


def _upload_to_hf(fits_by_cell: dict[str, Path], model: str) -> None:
    """Best-effort mirror of fit JSONs — ONE bulk `upload_folder` commit."""
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    if not fits_by_cell:
        return
    parents = {p.parent.resolve() for p in fits_by_cell.values()}
    if len(parents) != 1:
        _log(f"WARN heterogeneous fit roots; skipping bulk upload: {parents}")
        return
    root = next(iter(parents))
    allow_patterns: list[str] = []
    expected_paths: list[str] = []
    for cell_key, p in fits_by_cell.items():
        if not p.is_file() or p.stat().st_size == 0:
            continue
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        allow_patterns.append(rel)
        expected_paths.append(f"{TASK_PREFIX}/fits/{rel}")
    if not allow_patterns:
        return
    try:
        _upload_folder_filtered(
            root,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{TASK_PREFIX}/fits",
            allow_patterns=allow_patterns,
            expected_repo_paths=expected_paths,
        )
        _log(f"uploaded {len(allow_patterns)} fit JSON(s) in one bulk commit (model={model})")
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN fit upload failed (model={model}): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI driver


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def _resolve_cells(
    activations_dir: Path,
    variants: list[str],
    conditions: list[str],
    form_list: list[str],
    models: list[str],
) -> list[tuple[str, str, str, str, Path]]:
    """Return (variant, condition, form, model, activation path) for every cell
    we can locate (4-axis enumeration, C6). A flat smoke-fixture .npz (one that
    lives directly under `activations_dir`, resolved by the fallback) is
    attached to AT MOST ONE cell so the default condition×form product cannot
    multiply one fixture into duplicate cells.
    """
    out: list[tuple[str, str, str, str, Path]] = []
    used_fallback: set[Path] = set()
    for variant in variants:
        for condition in conditions:
            for form in form_list:
                for model in models:
                    path = _find_activation_path(activations_dir, variant, condition, form, model)
                    if path is None:
                        continue
                    is_fallback = path.parent.resolve() == activations_dir.resolve()
                    if is_fallback:
                        if path in used_fallback:
                            continue
                        used_fallback.add(path)
                    out.append((variant, condition, form, model, path))
    return out


def run_phase(args: argparse.Namespace) -> int:
    activations_dir = Path(args.activations_dir).resolve()
    fold_map_path = Path(args.fold_map).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not activations_dir.exists():
        print(f"ERROR: --activations-dir does not exist: {activations_dir}", file=sys.stderr)
        return 2
    try:
        fold_map = _load_fold_map(fold_map_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    cells = _resolve_cells(
        activations_dir,
        list(args.variants),
        list(args.conditions),
        list(args.forms),
        list(args.models),
    )
    if not cells:
        print(
            f"ERROR: no activation .npz found under {activations_dir} for "
            f"variants={list(args.variants)} conditions={list(args.conditions)} "
            f"forms={list(args.forms)} models={list(args.models)}",
            file=sys.stderr,
        )
        return 2

    _log(
        f"start: variants={list(args.variants)} conditions={list(args.conditions)} "
        f"forms={list(args.forms)} models={list(args.models)} arms={list(args.arms)} "
        f"layer={args.layer} n_null_draws={args.n_null_draws} pilot={args.pilot} "
        f"dry_run={args.dry_run}"
    )

    is_smoke = str(output_dir).startswith("/tmp/")

    fold_of = fold_map["fold_of"]
    # Cross-cell equalize-down: the shared conv_id set across every located cell.
    conv_ids_by_cell: dict[str, list[str]] = {}
    activations_by_cell: dict[tuple[str, str, str, str], dict] = {}
    for variant, condition, form, model, path in cells:
        activations = _load_activation_npz(path)
        if activations is None:
            _log(f"WARN empty .npz: {_rel(path)} (dry-run shell?); skipping")
            continue
        activations_by_cell[(variant, condition, form, model)] = activations
        conv_ids_by_cell[forms.cell_key(variant, condition, form, model)] = activations["conv_ids"]

    if not activations_by_cell:
        # Every located .npz was empty (dry-run shell scenario). In pilot/dry
        # mode we still emit a stub fit JSON per cell proving the pipeline
        # wiring parses to shape.
        if args.dry_run or args.pilot:
            _log("dry-run: no populated .npz activations under --activations-dir; emitting stubs")
            for variant, condition, form, model, path in cells:
                cell_key = forms.cell_key(variant, condition, form, model)
                stub = {
                    "cell": cell_key,
                    "variant": variant,
                    "condition": condition,
                    "form": form,
                    "model": model,
                    "status": "dry-run-empty-activation",
                    "activation_path": _rel(path),
                    "utc": datetime.now(tz=timezone.utc).isoformat(),
                    "dry_run": True,
                    "pilot": bool(args.pilot),
                }
                out_path = output_dir / f"{cell_key}.json"
                _write_json(out_path, stub)
                _log(f"cell={cell_key} stub={_rel(out_path)}")
            # The fits driver IS the standalone dispatcher on the VM CPU lane
            # (no pod-side sentinel path); the sibling capture driver waives
            # this on the same grounds. A future orchestrating dispatcher
            # redirects our stdout to a per-phase log per the per-worker
            # pattern (scripts/issue658_8gpu_dispatch.sh).
            # noqa: phase-done-reserved
            print("[phase=done]", flush=True)
            sys.stdout.flush()
            sys.exit(0)
        print("ERROR: every located activation .npz is empty (dry-run shell)", file=sys.stderr)
        return 2

    # Restriction to the intersection of fold_map keys × the shared conv_id set
    # across cells. For gate 4, we compare cells within the panel (equalize-down
    # applied per comparison — smoke path may only have ONE cell, so the gate
    # reports single-cell size instead).
    fold_conv_ids = set(fold_of.keys())
    shared_conv_ids: set[str] | None = None
    for cell_key, ids in conv_ids_by_cell.items():
        s = set(ids) & fold_conv_ids
        shared_conv_ids = s if shared_conv_ids is None else shared_conv_ids & s

    intersection_size = len(shared_conv_ids) if shared_conv_ids else 0

    # Kill gate 4 evaluation:
    #   - single-cell mode (only ONE cell located): report size, do NOT block
    #     the fit (the ambient-basis n_train ≥ d floor rides pooled-across-
    #     folds n = 0.8·n_conv; a single smoke cell often has fewer rows by
    #     construction).
    #   - multi-cell mode: fire the kill gate if the shared conv_id intersection
    #     falls below 4,480. The comparison is per (variant, model) pair; we
    #     emit a per-pair report AND a global report of the min-intersection.
    per_variant_reports: list[dict] = []
    fits_by_cell: dict[str, Path] = {}
    kill_gate_summary = {
        "min_pair_intersection": int(intersection_size),
        "kill_gate_4_threshold": KILL_GATE_4_MIN_INTERSECTION,
        "kill_gate_4_fire": bool(
            len(activations_by_cell) > 1 and intersection_size < KILL_GATE_4_MIN_INTERSECTION
        ),
        "n_cells": len(activations_by_cell),
    }

    # We fit anyway on ALL cells that PASS a per-cell size floor. Kill gate 4
    # is REPORTED; the driver's contract with the plan is to persist the outcome
    # (per Unit A/B/C, the surface EMITS + FLAGS; the analyzer / user pauses on
    # a kill-gate-fire before shipping the headline claim).
    t0 = time.time()
    for (variant, condition, form, model), activations in activations_by_cell.items():
        cell_key = forms.cell_key(variant, condition, form, model)
        arm_reports: dict[str, dict] = {}
        # Answer-length parity for the KS gate — the plan §7 "(b) vs (d)"
        # comparison: SAME (variant, form, model), inserted <-> on_policy
        # conditions (C6: the pre-fix `_op`-suffix heuristic compared against
        # the cell_c output — a (c)-labeled cell — because the condition axis
        # was absent from the key). cell_c cells have no length-parity peer.
        diag_this = _load_capture_diagnostics(variant, condition, form, model)
        peer_condition = _GATE5_PEER_CONDITION.get(condition)
        gate5_peer_cell: str | None = None
        peer_diag: dict | None = None
        if peer_condition is not None:
            gate5_peer_cell = forms.cell_key(variant, peer_condition, form, model)
            peer_diag = _load_capture_diagnostics(variant, peer_condition, form, model)

        peer_cells_conv_ids = {
            other_key: ids for other_key, ids in conv_ids_by_cell.items() if other_key != cell_key
        } or None

        gate_report = _evaluate_kill_gates(
            variant=variant,
            condition=condition,
            form=form,
            model=model,
            conv_ids_this_cell=activations["conv_ids"],
            peer_cells_conv_ids=peer_cells_conv_ids,
            diag_this=diag_this,
            peer_diag=peer_diag,
            gate5_peer_cell=gate5_peer_cell,
        )

        # Run per-arm fits (context AND prefix — CLAUDE.md standing rule).
        for arm in args.arms:
            arm_report = _fit_arm_cell(
                variant=variant,
                model=model,
                arm=arm,
                activations=activations,
                fold_map=fold_map,
                restrict_ids=shared_conv_ids,
                n_null_draws=int(args.n_null_draws),
                seed=int(args.seed),
                pilot=bool(args.pilot),
                bootstrap_draws=int(args.bootstrap_draws),
            )
            arm_reports[arm] = arm_report

        # Persist per-cell fit JSON (one per cell, both arms folded in).
        cell_payload = {
            "cell": cell_key,
            "variant": variant,
            "condition": condition,
            "form": form,
            "model": model,
            "layer": int(args.layer),
            "seed": int(args.seed),
            "arms": list(args.arms),
            "arm_reports": arm_reports,
            "kill_gate_report": gate_report,
            "shared_conv_id_intersection": int(len(shared_conv_ids or set())),
            "n_null_draws": int(args.n_null_draws),
            "bootstrap_draws": int(args.bootstrap_draws),
            "reduced_basis_k": REDUCED_BASIS_K,
            "fold_map": {
                "path": _rel(fold_map_path),
                "k": int(fold_map["k"]),
                "seed": int(fold_map.get("seed", -1)),
                "n_conv_ids": int(fold_map.get("n_conv_ids", len(fold_of))),
            },
            "pilot": bool(args.pilot),
            "dry_run": bool(args.dry_run),
            "utc": datetime.now(tz=timezone.utc).isoformat(),
        }
        out_path = output_dir / f"{cell_key}.json"
        _write_json(out_path, cell_payload)
        fits_by_cell[cell_key] = out_path
        per_variant_reports.append(
            {
                "cell": cell_key,
                "variant": variant,
                "condition": condition,
                "form": form,
                "model": model,
                "path": _rel(out_path),
                "status": "ok",
                "arm_status": {arm: arm_reports[arm].get("status") for arm in args.arms},
                "min_pair_intersection": gate_report["min_pair_intersection"],
                "kill_gate_4_fire": gate_report["kill_gate_4_fire"],
                "kill_gate_5_fire": gate_report["kill_gate_5_fire"],
            }
        )
        elapsed = time.time() - t0
        _log(
            f"cell {cell_key} arms={list(args.arms)} "
            f"gate4_fire={gate_report['kill_gate_4_fire']} gate5_fire={gate_report['kill_gate_5_fire']} "
            f"-> {_rel(out_path)} elapsed={elapsed:.1f}s"
        )

    # Uploads (real runs only; smoke tree stays under /tmp/).
    if not is_smoke and not args.skip_upload and not args.dry_run:
        try:
            for model in args.models:
                # Model is the LAST cell-key axis, so the __-anchored suffix
                # match is exact (C6: `_{model}` would also match substrings).
                suffix = f"{forms.CELL_KEY_SEP}{model}"
                model_fits = {k: v for k, v in fits_by_cell.items() if k.endswith(suffix)}
                _upload_to_hf(model_fits, model)
        except Exception as exc:  # noqa: BLE001
            _log(f"WARN upload stage failed: {exc}")

    # Digest.
    digest = {
        "phase": "fits",
        "variants": list(args.variants),
        "conditions": list(args.conditions),
        "forms": list(args.forms),
        "models": list(args.models),
        "arms": list(args.arms),
        "layer": int(args.layer),
        "seed": int(args.seed),
        "n_cells_found": len(cells),
        "n_cells_fit": len(fits_by_cell),
        "kill_gate_summary": kill_gate_summary,
        "per_cell": per_variant_reports,
        "shared_fold_map": _rel(fold_map_path),
        "pilot": bool(args.pilot),
        "dry_run": bool(args.dry_run),
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    digest_path = output_dir / "fits_digest.json"
    _write_json(digest_path, digest)
    _log(f"digest: n_cells_fit={len(fits_by_cell)} -> {_rel(digest_path)}")

    # see the dry-run branch above; same waiver.
    # noqa: phase-done-reserved
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.exit(0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--activations-dir",
        default="data/issue_2054/activations/",
        help="Unit C capture output root (per-variant subdirs of .npz).",
    )
    p.add_argument(
        "--fold-map",
        default="eval_results/issue_2054/shared_fold_map.json",
        help="Unit A shared fold-map artifact (conv_id -> fold).",
    )
    p.add_argument(
        "--output-dir",
        default="data/issue_2054/fits/",
        help="Per-cell fit-JSON output directory.",
    )
    p.add_argument("--seed", type=int, default=137, help="Shared seed (plan §11).")
    p.add_argument(
        "--variants",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_VARIANTS),
        help="Comma-separated variant slugs.",
    )
    p.add_argument(
        "--models",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_MODELS),
        help="Comma-separated model slugs (qwen2.5-7b, qwen2.5-7b-instruct).",
    )
    p.add_argument(
        "--conditions",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_CONDITIONS),
        help=(
            "Comma-separated condition (capture --phase) axis values; cells are "
            "keyed on all four lattice axes (C6). Only located .npz combos run."
        ),
    )
    p.add_argument(
        "--forms",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_FORMS),
        help="Comma-separated framing (form) axis values (plan §4; C6).",
    )
    p.add_argument("--layer", type=int, default=19, help="Hidden-state layer (plan §11).")
    p.add_argument(
        "--arms",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=["context", "prefix"],
        help="Comma-separated arms; CLAUDE.md requires BOTH by default.",
    )
    p.add_argument(
        "--n-null-draws",
        type=int,
        default=200,
        help="Shuffled-answer matched-capacity null draws per fold (plan §6 DV 4).",
    )
    p.add_argument(
        "--bootstrap-draws",
        type=int,
        default=DEFAULT_BOOTSTRAP_DRAWS,
        help="Bootstrap draws over conversations for the CI on each fold.",
    )
    p.add_argument("--skip-upload", action="store_true", help="Skip the HF mirror step.")
    p.add_argument("--upload", action="store_true", help="Force HF mirror step.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Wiring-only smoke: parse CLI, exercise fit/baseline/null/kNN on a tiny slice.",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="1-cell 1-fold pilot mode: exercise the full pipeline on one cell/one fold.",
    )
    args = p.parse_args()
    return run_phase(args)


if __name__ == "__main__":
    sys.exit(main())
