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
  keeping capacity fixed); ``--n-null-draws`` (default 100 — plan §6 DV 4 /
  §9 sizing) draws, batched via a shared factorization (never a per-draw
  serial fit; `.claude/rules/vectorize-many-cell-fits.md`).
- Reduced-basis (train-fold PCA k=1024) diagnostic R² per cell (the #1887
  recipe) alongside the ambient fit, so the writeup can contrast estimators.
- Per-comparison bootstrap CI over CONVERSATIONS within the equalized-down
  intersection (NOT K=5 fold-resample — statistics-critic concern #2).

Equalize-down is PER COMPARISON (plan req 8 / §"equalize-down policy"; M1):
cells group by (character, model) — the (a,b,c,d) 2x2's identity pair, with
`char_X`/`char_X_op`/`char_X_op_base` mapping to one character — and every
cell fits on the conv_id intersection of its OWN group (never the global
intersection across all located cells, which over-discards for every cell
and empties every fit when the assistant scope's conv_id space is disjoint
from the character scopes'). Per-group equalize manifests land under
``<output-dir>/equalize/``.

Kill-gate outcomes (v7→v8 statistics-critic Must-Fix, plan §4/§7):

- **Kill gate 4** — min conv_id intersection across the compared cells of a
  (character, model) pair < 4,480 (plan §7: per-pair, M1): reported per pair;
  equalize-down at n<4480 pushes n_train=0.8·n below d=3,584 and re-enters
  the estimator-degenerate regime.
- **Kill gate 5** — (b) vs (d) answer-length KS D > 0.30 OR mean-ratio outside
  [0.25, 4.0] within a (character, model) pair: EMITS the fit and FLAGS the
  gate outcome (emit-and-flag, like gate 4 — the analyzer/user pauses on a
  fired gate before shipping the headline); length-stratified refit is not
  tractable at the row count.

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
import hashlib
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
import issue2054_resume as resume  # noqa: E402
from issue2054_pilot import fleet_projection_update as _fleet_projection_update  # noqa: E402
from issue2054_pilot import require_prior_wall_seconds as _require_prior_wall_seconds  # noqa: E402
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

# Plan §9 pilot-gate cell: assistant × chat × inserted × instruct — the FIRST
# fit-battery leg runs this one cell at production shape and records measured
# wall + peak RSS BEFORE the fleet (M5; plan-compute-sizing.md § Per-cell fit
# phases). Falls back to the first located cell when absent.
PILOT_PREFERRED_CELL = (
    "conversation_paired_stories_assistant",
    "inserted",
    "chat",
    "qwen2.5-7b-instruct",
)
PILOT_RSS_ROUTE_OFF_VM_GIB = 16.0


def _base_character(variant: str) -> str:
    """Map an on-policy variant to its base character (`char_X_op[_base]` ->
    `char_X`); delegates to the shared `forms.base_character` (M-R2-1: the
    ladder's pair-class predicates use the SAME mapping — one source)."""
    return forms.base_character(variant)


def _comparison_group_key(variant: str, model: str) -> tuple[str, str]:
    """The (character, model) comparison-group key (plan req 8 / §7 gate 4 —
    the (a,b,c,d) 2x2 lives within one character × model pair; M1)."""
    return (_base_character(variant), model)


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

    peer_cells_conv_ids: {peer_cell_key: conv_ids} for the cell's OWN
      (character, model) comparison group (plan §7: gate 4 is per pair — M1;
      never the global cell set). When None/empty (single-cell group — the
      smoke path), gate 4 REPORTS the cell's own size and never fires (a
      lone cell has no comparison to equalize; the gate-calibration rule:
      a production-n gate must not kill a smoke leg).
    peer_diag: the (b)<->(d) length-parity peer's capture diagnostics — the
      SAME (variant, form, model) under the paired condition (inserted <->
      on_policy); None for cell_c cells (no length-parity peer) or when the
      peer was never captured (KS gate reports missing-diagnostics).
    gate5_peer_cell: the peer's cell key, for the report (None when N/A).
    """
    intersection_size = len(conv_ids_this_cell)
    peer_report: dict = {"single_cell_group": not peer_cells_conv_ids}
    if peer_cells_conv_ids:
        peer_report["peer_cells"] = sorted(peer_cells_conv_ids.keys())
        intersection_size = _min_pair_intersection(
            {"this": conv_ids_this_cell, **peer_cells_conv_ids}
        )
    gate4_fire = bool(peer_cells_conv_ids) and intersection_size < KILL_GATE_4_MIN_INTERSECTION
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
    """Mirror fit JSONs — ONE bulk `upload_folder` commit. FATAL on failure
    (M2): a swallowed upload + `[phase=done]` silently strands the fits."""
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    if not fits_by_cell:
        return
    parents = {p.parent.resolve() for p in fits_by_cell.values()}
    if len(parents) != 1:
        raise RuntimeError(f"heterogeneous fit roots — cannot compose one bulk upload: {parents}")
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
        raise RuntimeError(
            f"upload set resolved EMPTY against declared fit JSONs: {sorted(fits_by_cell)}"
        )
    _upload_folder_filtered(
        root,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{TASK_PREFIX}/fits",
        allow_patterns=allow_patterns,
        expected_repo_paths=expected_paths,
    )
    _log(f"uploaded {len(allow_patterns)} fit JSON(s) in one bulk commit (model={model})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI driver


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


# Regime keys a resumed cell must match EXACTLY against its existing fit JSON
# (C9/M6 — every output-affecting key, incl. the equalized restriction set +
# npz identity; the #722-r3 rule).
_CELL_RESUME_KEYS = (
    "cell",
    "arms",
    "layer",
    "seed",
    "n_null_draws",
    "bootstrap_draws",
    "reduced_basis_k",
    "pilot",
    "dry_run",
    "restrict_sha256",
    "npz_sha256",
)
_CELL_RESUME_FOLD_KEYS = {
    "fold_map_k": "k",
    "fold_map_seed": "seed",
    "fold_map_n_conv_ids": "n_conv_ids",
}


def _cell_resume_check(out_path: Path, expected: dict) -> tuple[bool, str]:
    """(skip?, reason) for one cell against its existing fit JSON.

    Fit JSONs are self-describing, so a mismatch RECOMPUTES (with the reason
    logged by the caller) — no separate sidecar and no refusal needed.
    """
    if not out_path.is_file():
        return False, ""
    try:
        with out_path.open(encoding="utf-8") as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "existing fit JSON unreadable"
    mismatched = [k for k in _CELL_RESUME_KEYS if existing.get(k) != expected.get(k)]
    fm = existing.get("fold_map") or {}
    mismatched += [
        ek for ek, fk in _CELL_RESUME_FOLD_KEYS.items() if fm.get(fk) != expected.get(ek)
    ]
    ok = existing.get("arm_reports") and all(
        (existing["arm_reports"].get(a) or {}).get("status") == "ok"
        for a in expected.get("arms", [])
    )
    if mismatched:
        return False, f"regime keys changed: {mismatched}"
    if not ok:
        return False, "existing fit JSON has non-ok arm status"
    return True, "fit JSON complete under matching regime"


def _run_fits_pilot_gate(
    activations_by_cell: dict,
    groups: dict,
    group_restrict: dict,
    fold_map: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Pilot-before-fleet (M5; plan §9 pilot-gate): ONE cell (preferring the
    plan's named assistant × chat × inserted × instruct cell), ONE fold, at
    PRODUCTION draw counts, measured (wall + peak RSS) and persisted to
    `<output-dir>/pilot_gate_report.json` BEFORE the fleet loop runs.

    RSS >= 16 GiB is the plan's route-off-the-shared-VM boundary — the driver
    WARNs (routing is the dispatcher's call); it never blocks.
    """
    import resource

    report_path = output_dir / "pilot_gate_report.json"
    # Fleet-shape figures for the M-R2-1 pilot->fleet projection: one unit =
    # one (cell, arm); the pilot measures ONE fold, production runs k folds.
    n_fleet_units = len(activations_by_cell) * max(1, len(args.arms))
    fold_k = int(fold_map["k"])
    if report_path.is_file() and not args.overwrite:
        try:
            with report_path.open(encoding="utf-8") as f:
                prior = json.load(f)
            # Measurement-affecting knobs (r2 Minor 5: single-knob compare
            # reused a stale pilot across changed arm/seed/draw regimes).
            prior_matches = (
                prior.get("n_null_draws") == int(args.n_null_draws)
                and prior.get("bootstrap_draws") == int(args.bootstrap_draws)
                and prior.get("arm") == args.arms[0]
                and prior.get("seed") == int(args.seed)
            )
            if prior_matches:
                _log(f"pilot gate: prior report matches ({_rel(report_path)}); skipping")
                # r3 Minor 1: a prior report lacking the measured wall FAILS
                # LOUD — a silent 0.0 default would project a fleet wall of 0
                # and disarm the fence (ladder sibling site fixed identically).
                prior_wall = _require_prior_wall_seconds(prior, report_path)
                _fleet_projection_update(
                    report_path,
                    prior,
                    wall_seconds=prior_wall,
                    n_fleet_units=n_fleet_units,
                    fold_k=fold_k,
                    log=_log,
                    units_basis="total cells x arms (resume not modeled)",
                )
                return
        except (OSError, json.JSONDecodeError):
            pass

    pilot_key = (
        PILOT_PREFERRED_CELL
        if PILOT_PREFERRED_CELL in activations_by_cell
        else next(iter(activations_by_cell))
    )
    variant, condition, form, model = pilot_key
    gkey = _comparison_group_key(variant, model)
    _log(f"pilot gate: 1-cell 1-fold measured pilot on {forms.cell_key(*pilot_key)}")
    t0 = time.time()
    pilot_report = _fit_arm_cell(
        variant=variant,
        model=model,
        arm=args.arms[0],
        activations=activations_by_cell[pilot_key],
        fold_map=fold_map,
        restrict_ids=group_restrict[gkey],
        n_null_draws=int(args.n_null_draws),
        seed=int(args.seed),
        pilot=True,  # 1 fold — production shape otherwise
        bootstrap_draws=int(args.bootstrap_draws),
    )
    wall = time.time() - t0
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    payload = {
        "phase": "fits-pilot-gate",
        "cell": forms.cell_key(*pilot_key),
        "arm": args.arms[0],
        "seed": int(args.seed),
        "n_null_draws": int(args.n_null_draws),
        "bootstrap_draws": int(args.bootstrap_draws),
        "wall_seconds": round(wall, 3),
        "peak_rss_gib": round(peak_rss_gib, 3),
        "rss_route_off_vm_gib": PILOT_RSS_ROUTE_OFF_VM_GIB,
        "status": pilot_report.get("status"),
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    _fleet_projection_update(
        report_path,
        payload,
        wall_seconds=wall,
        n_fleet_units=n_fleet_units,
        fold_k=fold_k,
        log=_log,
        units_basis="total cells x arms (resume not modeled)",
    )
    _log(f"pilot gate: wall={wall:.1f}s peak_rss={peak_rss_gib:.2f} GiB -> {_rel(report_path)}")
    if peak_rss_gib >= PILOT_RSS_ROUTE_OFF_VM_GIB:
        _log(
            f"WARN pilot peak RSS {peak_rss_gib:.2f} GiB >= "
            f"{PILOT_RSS_ROUTE_OFF_VM_GIB} GiB — plan §9 routes this fit family "
            "OFF the shared VM (cpu-mid / cpu-bigmem); dispatcher decision"
        )


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

    # PER-COMPARISON equalize-down (plan req 8 / M1): cells group by
    # (character, model); each cell's fold rows restrict to ITS group's
    # conv_id intersection (∩ fold-map membership) — never a global
    # intersection across all located cells (which over-discards everywhere
    # and empties every fit when the assistant scope's conv_id space is
    # disjoint from the character scopes').
    fold_conv_ids = set(fold_of.keys())
    groups: dict[tuple[str, str], list[tuple[str, str, str, str]]] = {}
    for variant, condition, form, model in activations_by_cell:
        groups.setdefault(_comparison_group_key(variant, model), []).append(
            (variant, condition, form, model)
        )

    group_restrict: dict[tuple[str, str], set[str]] = {}
    group_reports: dict[str, dict] = {}
    equalize_dir = output_dir / "equalize"
    for gkey, members in sorted(groups.items()):
        inter: set[str] | None = None
        per_cell_n: dict[str, int] = {}
        for member in members:
            ids = set(activations_by_cell[member]["conv_ids"]) & fold_conv_ids
            per_cell_n[forms.cell_key(*member)] = len(ids)
            inter = ids if inter is None else inter & ids
        inter = inter or set()
        group_restrict[gkey] = inter
        gate4_fire = len(members) > 1 and len(inter) < KILL_GATE_4_MIN_INTERSECTION
        gname = f"{gkey[0]}{forms.CELL_KEY_SEP}{gkey[1]}"
        report = {
            "comparison": {"character": gkey[0], "model": gkey[1]},
            "cells": sorted(per_cell_n),
            "per_cell_n_in_folds": per_cell_n,
            "n_equalized": len(inter),
            "kill_gate_4_threshold": KILL_GATE_4_MIN_INTERSECTION,
            "kill_gate_4_fire": bool(gate4_fire),
            "single_cell_group": len(members) == 1,
            "utc": datetime.now(tz=timezone.utc).isoformat(),
        }
        group_reports[gname] = report
        # Equalize manifest per comparison (plan §"equalize-down policy").
        _write_json(equalize_dir / f"{gname}.json", report)
        if gate4_fire:
            _log(
                f"KILL GATE 4 fires for pair {gname}: n_equalized={len(inter)} "
                f"< {KILL_GATE_4_MIN_INTERSECTION} (reported; the analyzer/user "
                "pauses on a fired gate before shipping the headline)"
            )

    multi_cell_intersections = [
        len(group_restrict[g]) for g, members in groups.items() if len(members) > 1
    ]
    per_variant_reports: list[dict] = []
    fits_by_cell: dict[str, Path] = {}
    kill_gate_summary = {
        # Min over MULTI-cell comparison groups (per-pair semantics, plan §7).
        "min_pair_intersection": (
            int(min(multi_cell_intersections)) if multi_cell_intersections else 0
        ),
        "kill_gate_4_threshold": KILL_GATE_4_MIN_INTERSECTION,
        "kill_gate_4_fire": bool(any(r["kill_gate_4_fire"] for r in group_reports.values())),
        "n_cells": len(activations_by_cell),
        "n_comparison_groups": len(groups),
        "per_comparison": group_reports,
    }

    # Pilot-before-fleet (M5; plan §9 pilot-gate): ONE cell, 1 fold, at
    # production draw counts, measuring wall + peak RSS BEFORE the fleet.
    if not args.dry_run and not args.pilot and not args.skip_pilot_gate:
        _run_fits_pilot_gate(
            activations_by_cell, groups, group_restrict, fold_map, args, output_dir
        )

    # We fit anyway on ALL cells that PASS a per-cell size floor. Kill gate 4
    # is REPORTED; the driver's contract with the plan is to persist the outcome
    # (per Unit A/B/C, the surface EMITS + FLAGS; the analyzer / user pauses on
    # a kill-gate-fire before shipping the headline claim).
    t0 = time.time()
    for (variant, condition, form, model), activations in activations_by_cell.items():
        cell_key = forms.cell_key(variant, condition, form, model)
        gkey = _comparison_group_key(variant, model)
        group_members = groups[gkey]
        restrict_ids = group_restrict[gkey]
        restrict_sha = hashlib.sha256("\n".join(sorted(restrict_ids)).encode()).hexdigest()

        # Resume (C9/M6): a cell whose fit JSON already carries this exact
        # regime (incl. the equalized restriction set + the npz identity) is
        # skipped — the >~1h serial CPU loop no longer refits every cell on a
        # re-run. Fit JSONs are self-describing, so a regime mismatch simply
        # RECOMPUTES (logged) — no silent-mixing hazard.
        out_path = output_dir / f"{cell_key}.json"
        npz_path = None
        for v2, c2, f2, m2, p2 in cells:
            if (v2, c2, f2, m2) == (variant, condition, form, model):
                npz_path = p2
                break
        npz_sha = resume.file_sha256(npz_path) if npz_path is not None else None
        expected_regime = {
            "cell": cell_key,
            "arms": list(args.arms),
            "layer": int(args.layer),
            "seed": int(args.seed),
            "n_null_draws": int(args.n_null_draws),
            "bootstrap_draws": int(args.bootstrap_draws),
            "reduced_basis_k": REDUCED_BASIS_K,
            "pilot": bool(args.pilot),
            "dry_run": bool(args.dry_run),
            "restrict_sha256": restrict_sha,
            "npz_sha256": npz_sha,
            "fold_map_k": int(fold_map["k"]),
            "fold_map_seed": int(fold_map.get("seed", -1)),
            "fold_map_n_conv_ids": int(fold_map.get("n_conv_ids", len(fold_of))),
        }
        if not args.overwrite and not args.dry_run and not args.pilot:
            skip, why = _cell_resume_check(out_path, expected_regime)
            if skip:
                fits_by_cell[cell_key] = out_path
                per_variant_reports.append(
                    {
                        "cell": cell_key,
                        "variant": variant,
                        "condition": condition,
                        "form": form,
                        "model": model,
                        "path": _rel(out_path),
                        "status": "resumed",
                    }
                )
                _log(f"cell {cell_key} RESUME skip ({why})")
                continue
            if why:
                _log(f"cell {cell_key} recompute: {why}")

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

        # Gate 4 peers = the OTHER cells of this cell's OWN comparison group
        # (per-pair semantics — plan §7 / M1; never the global cell set).
        peer_cells_conv_ids = {
            forms.cell_key(*member): activations_by_cell[member]["conv_ids"]
            for member in group_members
            if member != (variant, condition, form, model)
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

        # Run per-arm fits (context AND prefix — CLAUDE.md standing rule),
        # restricted to the cell's OWN comparison group's equalized set (M1).
        for arm in args.arms:
            arm_report = _fit_arm_cell(
                variant=variant,
                model=model,
                arm=arm,
                activations=activations,
                fold_map=fold_map,
                restrict_ids=restrict_ids,
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
            "comparison_group": {
                "character": gkey[0],
                "model": gkey[1],
                "cells": sorted(forms.cell_key(*m) for m in group_members),
            },
            # The cell's own comparison-group equalized intersection (M1; the
            # legacy `shared_conv_id_intersection` name is kept for readers).
            "shared_conv_id_intersection": int(len(restrict_ids)),
            "restrict_sha256": restrict_sha,
            "npz_sha256": npz_sha,
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

    # Uploads (real runs only; smoke tree stays under /tmp/). FATAL on
    # failure (M2): `[phase=done]` must never report done with the fit JSONs
    # un-persisted. No try/except.
    if not is_smoke and not args.skip_upload and not args.dry_run:
        for model in args.models:
            # Model is the LAST cell-key axis, so the __-anchored suffix
            # match is exact (C6: `_{model}` would also match substrings).
            suffix = f"{forms.CELL_KEY_SEP}{model}"
            model_fits = {k: v for k, v in fits_by_cell.items() if k.endswith(suffix)}
            _upload_to_hf(model_fits, model)

    # Digest.
    digest = {
        "phase": "fits",
        # Plan §6.5 path-shape deviation, recorded per review r2 Minor 9(b):
        # per-cell fits land FLAT at `<output-dir>/{cell_key}.json`, not the
        # plan's `fits/{cell}/within_cell_ceiling.json` (naming-only; carry
        # into the Repro card).
        "deliverable_path_shape": "flat {cell_key}.json (plan §6.5 names fits/{cell}/within_cell_ceiling.json — naming-only deviation)",
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
        default=100,
        help=(
            "Shuffled-answer matched-capacity null draws per fold (plan §6 DV 4 "
            "+ §9 sizing pin BOTH say 100 — reconciled from the round-1 200)."
        ),
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
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "refit cells even when a regime-matching fit JSON exists "
            "(default resumes completed cells — C9/M6)"
        ),
    )
    p.add_argument(
        "--skip-pilot-gate",
        action="store_true",
        help=(
            "skip the automatic 1-cell measured pilot leg before the fleet "
            "(M5/plan §9; only when a standalone pilot already ran)"
        ),
    )
    args = p.parse_args()
    return run_phase(args)


if __name__ == "__main__":
    sys.exit(main())
