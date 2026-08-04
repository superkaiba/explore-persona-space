#!/usr/bin/env python
"""9-rung transfer ladder driver for task #2054.

For each ordered (source_cell, target_cell, arm) pair — cells = (variant,
model) — computes the 9 mapping-transformation rungs of the parent
`scripts/issue1345_ladder_rungs.py` line. Each rung composes the source-fitted
context->answer map with a specific target-adaptation transformation and
scores held-out R² of `v_A_target ≈ Rung(source_map)(v_C_target)` (or the
prefix-arm analogue with `v_P_target`). Each rung's read is ALSO normalized
against the TARGET cell's own within-cell ceiling — read from Unit D's per-cell
fit JSON at `<fits-dir>/<target_cell_key>.json` — so a rung R² is reported as
the transfer R² alone AND as the ratio transfer / target_ceiling (plan §6
DV 5).

The 9 rungs (parent's `RUNGS`, kept verbatim):

1. `1_direct`         — source ridge map applied to target contexts.
2. `2_ctx_offset`     — shift target contexts by the source-target mean gap
                        BEFORE applying the source map.
3. `3_ans_offset`     — apply source map, then shift by the source-target
                        answer mean gap.
4. `4_bias_refit`     — refit a bias correction on the target train fold.
5. `5_global_scale`   — global scalar rescaling of the source-map prediction.
6. `6_rotation`       — orthogonal Procrustes rotation of the source-map
                        prediction onto target answers.
7. `7_ctx_reparam`    — first reparameterize target contexts into the source
                        context space (ridge target->source contexts) then
                        apply the source map + a refit bias.
8. `8_ans_reparam`    — apply source map, then ridge-reparameterize the
                        source-space answer into the target answer space.
9. `9_full_AMB`       — the full composed A-M-B chain (context reparam +
                        source map + answer reparam).

The driver reads Unit C activations (`.npz` per (variant, model) — `v_C`, `v_A`,
`v_P`, `v_P_present`), Unit A's shared fold map (K=5 conversation-grouped),
and Unit D's per-cell fit JSONs (for the target ceiling), joins the per-fold
train/val split to the EQUALIZED-DOWN intersection of the source and target
cells' conv_ids (statistics-critic concern #2 — conv-within-intersection
bootstrap CI over the SAME held-out conv_ids the fit was scored on), then per
fold: (1) fit the source ridge on the source's train-fold rows, (2) build the
9 rung predictions on the target's held-out fold rows, (3) score each vs the
target's held-out answers, (4) bootstrap 95% CI over the intersection's
conversations.

Emits `[phase=ladder]` log lines; terminates in `[phase=done]` on graceful
completion. Best-effort HF mirror to `issue2054_lattice/ladder/` when
`--upload` is set. `--dry-run` exercises the CLI + rung math on a tiny slice
(1 fold, ≤1 (source, target) pair, self-transfer permitted for smoke) and
skips uploads.

Exit 0 on success. Exit 1 on fit / HF failure. Exit 2 on missing input.
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

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
TASK_PREFIX = "issue2054_lattice"

# The 9 rungs — verbatim from `scripts/issue1345_ladder_rungs.py`.
RUNGS = (
    "1_direct",
    "2_ctx_offset",
    "3_ans_offset",
    "4_bias_refit",
    "5_global_scale",
    "6_rotation",
    "7_ctx_reparam",
    "8_ans_reparam",
    "9_full_AMB",
)

# GCV λ grid — shared project convention (Unit D uses the same).
DEFAULT_LAMBDAS = np.logspace(-2, 4, 13)
DEFAULT_DOF_CAP = 0.9

# Bootstrap draws over conversations within the equalized intersection
# (statistics-critic concern #2 — conv-within-intersection bootstrap CI).
DEFAULT_BOOTSTRAP_DRAWS = 200

DEFAULT_VARIANTS = (
    "char_helios",
    "char_wren",
    "char_dana",
    "char_vex",
    "conversation_paired_stories_assistant",
)

DEFAULT_MODELS = ("qwen2.5-7b", "qwen2.5-7b-instruct")


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers


def _log(msg: str) -> None:
    print(f"[phase=ladder] {msg}", flush=True)


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


def _find_activation_path(activations_dir: Path, variant: str, model: str) -> Path | None:
    """Locate the .npz per Unit C's layout:  <activations-dir>/<variant>/<variant>_<model>.npz.

    Falls back to any *.npz directly under `<activations-dir>` when the variant
    subtree is missing (smoke fixture convention, matching Unit C's `_flat`).
    """
    canonical = activations_dir / variant / f"{variant}_{model}.npz"
    if canonical.is_file() and canonical.stat().st_size > 0:
        return canonical
    if activations_dir.is_dir():
        for p in sorted(activations_dir.glob("*.npz")):
            if p.stat().st_size > 0:
                return p
    return None


def _load_activation_npz(path: Path) -> dict | None:
    """Return {conv_ids, v_C, v_A, v_P, v_P_present} arrays; None if empty."""
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


def _load_target_ceiling(fits_dir: Path, cell_key: str, arm: str) -> float | None:
    """Read the target cell's within-cell ceiling from Unit D's fit JSON.

    Falls back to None (reports as null) if the fit JSON is missing / lacks a
    pooled entry for this arm.
    """
    path = fits_dir / f"{cell_key}.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        arm_report = d.get("arm_reports", {}).get(arm, {})
        pooled = arm_report.get("pooled", {})
        val = pooled.get("r2_ambient_mean")
        if val is None:
            return None
        return float(val)
    except (KeyError, TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Arm selection + fold split (mirrors Unit D)


def _select_arm(activations: dict, arm: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """context arm  →  X = v_C, Y = v_A;  rows kept: all.
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


def _row_index_by_conv_id(conv_ids: list[str]) -> dict[str, int]:
    return {cid: i for i, cid in enumerate(conv_ids)}


# ─────────────────────────────────────────────────────────────────────────────
# Ridge fit (numpy; ambient basis, GCV λ selection, dof-cap safeguard).
# One helper that returns (prediction on X_eval, prediction on X_ext) so the
# rung machinery can request extra evaluations at the same fit — never a
# per-eval refit.


def _standardize(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmu = X_train.mean(axis=0)
    xsd = X_train.std(axis=0) + 1e-9
    return (X_train - xmu) / xsd, xmu, xsd


def _fit_ridge_and_apply(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    apply_at: dict[str, np.ndarray],
    *,
    lambdas: np.ndarray = DEFAULT_LAMBDAS,
    dof_cap: float = DEFAULT_DOF_CAP,
) -> tuple[dict[str, np.ndarray], dict]:
    """Ambient GCV-ridge fit. Returns predictions at every key in `apply_at`
    (each is an X of shape (n, d)) + fit info.

    Standardize X on the train fold; center Y; primary reconstruction via SVD.
    """
    Xtr64 = X_train.astype(np.float64)
    Ytr64 = Y_train.astype(np.float64)
    Xtr, xmu, xsd = _standardize(Xtr64)
    ymu = Ytr64.mean(axis=0)
    Ytr_c = Ytr64 - ymu

    n_train = Xtr.shape[0]
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c

    best_lam = float(lambdas[0])
    best_gcv = float("inf")
    best_dof = float("nan")
    dof_over_cap = True
    row_energy = (UtY**2).sum(axis=1)
    tot_y_sq = float((Ytr_c**2).sum())
    for lam in lambdas:
        lam = float(lam)
        filt = s2 / (s2 + lam)
        dof = float(filt.sum())
        if dof / n_train <= dof_cap:
            dof_over_cap = False
        rss = tot_y_sq - float(((2 * filt - filt**2) * row_energy).sum())
        denom = (n_train - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if dof / n_train <= dof_cap and gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam
            best_dof = dof
    if best_gcv == float("inf"):
        best_lam = float(lambdas[-1])
        filt = s2 / (s2 + best_lam)
        best_dof = float(filt.sum())
        best_gcv = float("nan")

    # W = V diag(s/(s²+λ)) UtY — ambient primal.
    filt = s / (s2 + best_lam)
    W = (Vt.T * filt) @ UtY  # (d, D_out)

    preds: dict[str, np.ndarray] = {}
    for key, X_apply in apply_at.items():
        Xa = (X_apply.astype(np.float64) - xmu) / xsd
        preds[key] = Xa @ W + ymu
    info = {
        "best_lambda": best_lam,
        "dof": best_dof,
        "dof_cap": dof_cap,
        "dof_over_cap": bool(dof_over_cap),
        "gcv": best_gcv,
        "n_train": int(n_train),
        "d_in": int(Xtr.shape[1]),
        "d_out": int(Ytr64.shape[1]),
    }
    return preds, info


def _r2_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_mean = y_true.mean(axis=0)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_mean) ** 2).sum())
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _procrustes_apply(
    P_tr: np.ndarray, Y_tr: np.ndarray, P_ev: np.ndarray
) -> tuple[np.ndarray, float]:
    """Orthogonal Procrustes on the answer side (rung 6).

    Returns (rotated prediction at P_ev, max relative off-subspace residual).
    """
    Ptr = P_tr.astype(np.float64)
    Ytr = Y_tr.astype(np.float64)
    Pev = P_ev.astype(np.float64)
    pmu, ymu = Ptr.mean(axis=0), Ytr.mean(axis=0)
    Pc, Yc = Ptr - pmu, Ytr - ymu
    # QR of centered clouds so the SVD is on the (k, k) intermediate, never
    # (d, d).
    Q1, R1 = np.linalg.qr(Pc.T, mode="reduced")
    Q2, R2 = np.linalg.qr(Yc.T, mode="reduced")
    Uc, _S, Vch = np.linalg.svd(R1 @ R2.T)
    Pe = Pev - pmu
    proj = Pe @ Q1
    Pe_norm = np.linalg.norm(Pe, axis=-1) + 1e-12
    resid_vec = Pe - proj @ Q1.T
    resid = float((np.linalg.norm(resid_vec, axis=-1) / Pe_norm).max())
    return ((proj @ Uc) @ Vch) @ Q2.T + ymu, resid


def _bootstrap_conv_ci_over_intersection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> dict:
    """95% percentile bootstrap CI of held-out R² resampling CONVERSATIONS
    within the (already equalized-down) intersection.

    Statistics-critic concern #2 sustained: resample conversations within the
    intersection (NOT the K=5 folds).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = y_true.shape[0]
    if n == 0:
        return {
            "r2_boot_median": float("nan"),
            "r2_boot_lo": float("nan"),
            "r2_boot_hi": float("nan"),
            "n_draws": 0,
        }
    y_mean = y_true.mean(axis=0)
    ss_tot_full = float(((y_true - y_mean) ** 2).sum())
    if ss_tot_full < 1e-18:
        return {
            "r2_boot_median": float("nan"),
            "r2_boot_lo": float("nan"),
            "r2_boot_hi": float("nan"),
            "n_draws": 0,
        }
    rng = np.random.default_rng(seed)
    r2s = np.empty(n_draws, dtype=np.float64)
    for d in range(n_draws):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ymu = yt.mean(axis=0)
        ss_res = float(((yt - yp) ** 2).sum())
        ss_tot = float(((yt - ymu) ** 2).sum())
        r2s[d] = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
    r2s = r2s[np.isfinite(r2s)]
    if r2s.size == 0:
        return {
            "r2_boot_median": float("nan"),
            "r2_boot_lo": float("nan"),
            "r2_boot_hi": float("nan"),
            "n_draws": 0,
        }
    return {
        "r2_boot_median": float(np.median(r2s)),
        "r2_boot_lo": float(np.percentile(r2s, 2.5)),
        "r2_boot_hi": float(np.percentile(r2s, 97.5)),
        "n_draws": int(r2s.size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The 9-rung transfer battery


def _compute_rungs_for_fold(
    Xs_tr: np.ndarray,
    Ys_tr: np.ndarray,
    Xt_tr: np.ndarray,
    Xt_te: np.ndarray,
    Yt_tr: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute all 9 rung predictions at the target's held-out fold rows.

    Mirrors `scripts/issue1345_ladder_rungs.py::_rungs_for`, but on numpy +
    no per-eval refits (three source-map applications reuse the SAME fit).

    Args:
      Xs_tr, Ys_tr: source cell's train-fold rows (context / answer).
      Xt_tr, Xt_te, Yt_tr: target cell's train + held-out contexts + train
        answers (target held-out answers stay outside the rung compute — the
        caller scores against them).

    Returns:
      dict {rung_name: pred at Xt_te}, and an info dict with the source-fit
      hyperparameters + the Procrustes residual.
    """
    # A: ridge target-contexts -> source-contexts (rung 7 / 9 need the
    # target→source context reparameterization applied to BOTH the target's
    # train rows and its held-out rows).
    preds_A, info_A = _fit_ridge_and_apply(
        X_train=Xt_tr,
        Y_train=Xs_tr,
        apply_at={"tr": Xt_tr, "te": Xt_te},
    )
    Xs_hat_tr, Xs_hat_te = preds_A["tr"], preds_A["te"]

    # M: source ridge  v_C_source -> v_A_source. Apply it at four different
    # inputs — target's context train/te AND the target→source-reparam of the
    # same — in ONE fit call.
    preds_M, info_M = _fit_ridge_and_apply(
        X_train=Xs_tr,
        Y_train=Ys_tr,
        apply_at={
            "P_tr": Xt_tr,
            "P_te": Xt_te,
            "P7_tr": Xs_hat_tr,
            "P7_te": Xs_hat_te,
        },
    )
    P_tr = preds_M["P_tr"]
    P_te = preds_M["P_te"]
    P7_tr = preds_M["P7_tr"]
    P7_te = preds_M["P7_te"]

    # Context / answer mean shifts (source -> target).
    dx = Xt_tr.astype(np.float64).mean(axis=0) - Xs_tr.astype(np.float64).mean(axis=0)
    dy = Yt_tr.astype(np.float64).mean(axis=0) - Ys_tr.astype(np.float64).mean(axis=0)

    # Bias refit + global scale (rung 4, rung 5): both take the source-map's
    # predictions on target contexts and adjust with a target-train fit.
    pmu = P_tr.mean(axis=0)
    ymu = Yt_tr.astype(np.float64).mean(axis=0)
    bstar = (Yt_tr.astype(np.float64) - P_tr).mean(axis=0)  # rung 4
    b7 = (Yt_tr.astype(np.float64) - P7_tr).mean(axis=0)  # rung 7 bias refit
    Pc = P_tr - pmu
    Yc = Yt_tr.astype(np.float64) - ymu
    denom = float((Pc**2).sum())
    a = float((Pc * Yc).sum() / denom) if denom > 1e-30 else 1.0  # rung 5 scalar

    # Rung 6 — orthogonal Procrustes on the source-map's held-out predictions.
    rot_te, resid_max = _procrustes_apply(P_tr, Yt_tr, P_te)

    # Rung 8 — refit ridge (source-answer -> target-answer) using the SAME
    # answer clouds, then apply to the source-map's target held-out prediction.
    preds_B, info_B = _fit_ridge_and_apply(
        X_train=Ys_tr,
        Y_train=Yt_tr,
        apply_at={"P_te": P_te},
    )
    P_te_reparam = preds_B["P_te"]

    # Rung 9 — full A-M-B chain: reparameterize context (rung 7's P7_te) →
    # source map → reparameterize answer.
    preds_B9, _info_B9 = _fit_ridge_and_apply(
        X_train=Ys_tr,
        Y_train=Yt_tr,
        apply_at={"P7_te": P7_te},
    )
    P7_te_reparam = preds_B9["P7_te"]

    rung_preds: dict[str, np.ndarray] = {
        "1_direct": P_te,
        # Rung 2 shifts target contexts BEFORE applying source map — one extra
        # source-map application at Xt_te - dx.
    }
    preds_shift, _info_shift = _fit_ridge_and_apply(
        X_train=Xs_tr,
        Y_train=Ys_tr,
        apply_at={"P_shift": Xt_te.astype(np.float64) - dx},
    )
    rung_preds["2_ctx_offset"] = preds_shift["P_shift"]
    rung_preds["3_ans_offset"] = P_te + dy
    rung_preds["4_bias_refit"] = P_te + bstar
    rung_preds["5_global_scale"] = a * (P_te - pmu) + ymu
    rung_preds["6_rotation"] = rot_te
    rung_preds["7_ctx_reparam"] = P7_te + b7
    rung_preds["8_ans_reparam"] = P_te_reparam
    rung_preds["9_full_AMB"] = P7_te_reparam

    info = {
        "source_fit": info_M,
        "ctx_reparam_fit": info_A,
        "ans_reparam_fit": info_B,
        "global_scale_a": a,
        "procrustes_resid_max": resid_max,
    }
    return rung_preds, info


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair driver


def _fit_arm_pair(
    source_cell_key: str,
    target_cell_key: str,
    arm: str,
    source_acts: dict,
    target_acts: dict,
    fold_map: dict,
    target_ceiling: float | None,
    *,
    n_rungs: int,
    seed: int,
    pilot: bool,
    bootstrap_draws: int,
) -> dict:
    """Compute the 9-rung ladder for one ordered (source, target, arm) pair.

    Equalize-down: fold rows keyed on the INTERSECTION of source + target
    conv_ids (post-arm mask). Both cells' arm rows re-index to that shared set.
    """
    k = int(fold_map["k"])
    fold_of = fold_map["fold_of"]

    Xs_all, Ys_all, s_conv_ids = _select_arm(source_acts, arm)
    Xt_all, Yt_all, t_conv_ids = _select_arm(target_acts, arm)

    intersection = set(s_conv_ids) & set(t_conv_ids) & set(fold_of.keys())
    if not intersection:
        return {
            "source": source_cell_key,
            "target": target_cell_key,
            "arm": arm,
            "status": "no-shared-conv-ids",
            "n_intersection": 0,
            "target_ceiling": target_ceiling,
        }

    s_row_of = _row_index_by_conv_id(s_conv_ids)
    t_row_of = _row_index_by_conv_id(t_conv_ids)

    # Fold row indices, over the SHARED conv_ids only (equalize-down).
    ordered_ids = sorted(intersection)
    folds_s: list[list[int]] = [[] for _ in range(k)]
    folds_t: list[list[int]] = [[] for _ in range(k)]
    for cid in ordered_ids:
        f = int(fold_of[cid])
        folds_s[f].append(s_row_of[cid])
        folds_t[f].append(t_row_of[cid])

    # For pilot / n_rungs=1 fast path, cap to one fold.
    fold_range = range(min(1, k)) if pilot else range(k)

    per_fold: list[dict] = []
    rung_names = list(RUNGS[:n_rungs])
    rung_r2s: dict[str, list[float]] = {r: [] for r in rung_names}
    rung_boots: dict[str, list[dict]] = {r: [] for r in rung_names}

    t_fold0 = time.time()
    for fold_i in fold_range:
        val_idx_s = np.array(folds_s[fold_i], dtype=np.int64)
        val_idx_t = np.array(folds_t[fold_i], dtype=np.int64)
        # Train rows on OTHER folds, restricted to the equalize-down set.
        train_ids = [cid for cid in ordered_ids if int(fold_of[cid]) != fold_i]
        train_idx_s = np.array([s_row_of[c] for c in train_ids], dtype=np.int64)
        train_idx_t = np.array([t_row_of[c] for c in train_ids], dtype=np.int64)
        if val_idx_t.size == 0 or train_idx_s.size == 0 or train_idx_t.size == 0:
            per_fold.append(
                {
                    "fold": fold_i,
                    "status": "skipped-empty-fold",
                    "n_train": int(train_idx_s.size),
                    "n_val": int(val_idx_t.size),
                }
            )
            continue

        Xs_tr = Xs_all[train_idx_s]
        Ys_tr = Ys_all[train_idx_s]
        Xt_tr = Xt_all[train_idx_t]
        Xt_te = Xt_all[val_idx_t]
        Yt_tr = Yt_all[train_idx_t]
        Yt_te = Yt_all[val_idx_t]

        rung_preds, info_fit = _compute_rungs_for_fold(
            Xs_tr=Xs_tr,
            Ys_tr=Ys_tr,
            Xt_tr=Xt_tr,
            Xt_te=Xt_te,
            Yt_tr=Yt_tr,
        )

        # kNN retrieval on rung 1 (the direct-transfer read) — a scale-invariant
        # companion to R² per the standing rule.
        knn_euclid = knn_retrieval(rung_preds["1_direct"], Yt_te, metric="euclidean")

        # Identity+learned-bias baseline (same-space arms only — v_A dim matches
        # v_C / v_P dim under Qwen residual capture); report NaN otherwise.
        try:
            id_pred = identity_bias_predict(Xt_tr, Yt_tr, Xt_te)
            r2_id = _r2_matrix(Yt_te, id_pred)
        except ValueError:
            r2_id = float("nan")

        fold_record: dict = {
            "fold": fold_i,
            "n_train": int(train_idx_s.size),
            "n_val": int(val_idx_t.size),
            "info_fit": info_fit,
            "r2_identity_bias": r2_id,
            "knn_euclidean_at_1": float(knn_euclid["acc_at_k"][1]),
            "knn_euclidean_chance_at_1": float(knn_euclid["chance_at_k"][1]),
            "n_pool": int(knn_euclid["n_pool"]),
            "rungs": {},
        }
        for rung in rung_names:
            preds_te = rung_preds[rung]
            r2 = _r2_matrix(Yt_te, preds_te)
            boot = _bootstrap_conv_ci_over_intersection(
                Yt_te,
                preds_te,
                n_draws=bootstrap_draws,
                seed=seed + 10_000 + fold_i,
            )
            ratio = (
                r2 / target_ceiling
                if target_ceiling is not None
                and np.isfinite(target_ceiling)
                and abs(target_ceiling) > 1e-12
                else float("nan")
            )
            fold_record["rungs"][rung] = {
                "r2_transfer": r2,
                "r2_target_ceiling": target_ceiling if target_ceiling is not None else float("nan"),
                "ratio_transfer_over_ceiling": ratio,
                "bootstrap_conv_ci": boot,
            }
            rung_r2s[rung].append(r2)
            rung_boots[rung].append(boot)
        per_fold.append(fold_record)

        elapsed = time.time() - t_fold0
        summary_bits = " ".join(
            f"{r.split('_')[0]}={fold_record['rungs'][r]['r2_transfer']:.3f}" for r in rung_names
        )
        print(
            f"[phase=ladder] unit {fold_i + 1}/{k} "
            f"pair={source_cell_key}->{target_cell_key} arm={arm} "
            f"n_train={train_idx_s.size} n_val={val_idx_t.size} {summary_bits} "
            f"ceiling={target_ceiling} elapsed={elapsed:.1f}s",
            flush=True,
        )

    pooled: dict[str, dict] = {}
    for rung in rung_names:
        r2_list = rung_r2s[rung]
        if r2_list:
            arr = np.asarray(r2_list, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            pooled[rung] = {
                "r2_transfer_mean": float(arr.mean()) if arr.size else float("nan"),
                "r2_transfer_median": float(np.median(arr)) if arr.size else float("nan"),
                "n_folds": int(arr.size),
            }
            if target_ceiling is not None and abs(target_ceiling) > 1e-12:
                pooled[rung]["ratio_mean"] = float(arr.mean() / target_ceiling)

    return {
        "source": source_cell_key,
        "target": target_cell_key,
        "arm": arm,
        "status": "ok" if per_fold else "no-folds-ran",
        "n_intersection": len(intersection),
        "target_ceiling": target_ceiling,
        "n_rungs": n_rungs,
        "rungs_computed": rung_names,
        "per_fold": per_fold,
        "pooled": pooled,
        "seed": int(seed),
        "pilot": bool(pilot),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pair enumeration


def _cell_key(variant: str, model: str) -> str:
    return f"{variant}_{model}"


def _resolve_cells(
    activations_dir: Path, variants: list[str], models: list[str]
) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for variant in variants:
        for model in models:
            path = _find_activation_path(activations_dir, variant, model)
            if path is not None:
                out.append((variant, model, path))
    return out


def _enumerate_ordered_pairs(
    cells: list[tuple[str, str, Path]],
    *,
    smoke: bool,
) -> list[tuple[tuple[str, str, Path], tuple[str, str, Path]]]:
    """Ordered (source, target) pairs of cells.

    - Full run: every ordered pair (s, t) where s != t is a rung candidate;
      the caller runs the full 9-rung battery per pair.
    - Smoke: a SINGLE (s, s) self-transfer pair — with only one cell located,
      the ladder proves it can compute rung 1 (the within-cell ceiling
      reproduction) and every other rung on that same fixture.
    """
    if not cells:
        return []
    if smoke or len(cells) < 2:
        # Fallback to self-transfer so the ladder is exercisable at smoke scale.
        s = cells[0]
        return [(s, s)]
    pairs: list[tuple[tuple[str, str, Path], tuple[str, str, Path]]] = []
    for s in cells:
        for t in cells:
            if s == t:
                continue
            pairs.append((s, t))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Upload


def _upload_to_hf(pair_paths: list[Path]) -> None:
    """Best-effort mirror of rung JSONs — ONE bulk `upload_folder` commit."""
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    if not pair_paths:
        return
    parents = {p.parent.resolve() for p in pair_paths}
    if len(parents) != 1:
        _log(f"WARN heterogeneous ladder roots; skipping bulk upload: {parents}")
        return
    root = next(iter(parents))
    allow_patterns: list[str] = []
    expected_paths: list[str] = []
    for p in pair_paths:
        if not p.is_file() or p.stat().st_size == 0:
            continue
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        allow_patterns.append(rel)
        expected_paths.append(f"{TASK_PREFIX}/ladder/{rel}")
    if not allow_patterns:
        return
    try:
        _upload_folder_filtered(
            root,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{TASK_PREFIX}/ladder",
            allow_patterns=allow_patterns,
            expected_repo_paths=expected_paths,
        )
        _log(f"uploaded {len(allow_patterns)} rung JSON(s) in one bulk commit")
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN ladder upload failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI driver


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def run_phase(args: argparse.Namespace) -> int:
    activations_dir = Path(args.activations_dir).resolve()
    fits_dir = Path(args.fits_dir).resolve()
    fold_map_path = Path(args.fold_map).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not activations_dir.exists():
        print(f"ERROR: --activations-dir does not exist: {activations_dir}", file=sys.stderr)
        return 2
    if not fits_dir.exists():
        print(f"ERROR: --fits-dir does not exist: {fits_dir}", file=sys.stderr)
        return 2
    try:
        fold_map = _load_fold_map(fold_map_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    cells = _resolve_cells(activations_dir, list(args.variants), list(args.models))
    if not cells:
        print(
            f"ERROR: no activation .npz found under {activations_dir} for "
            f"variants={list(args.variants)} models={list(args.models)}",
            file=sys.stderr,
        )
        return 2

    is_smoke = str(output_dir).startswith("/tmp/") or args.dry_run or args.pilot
    _log(
        f"start: variants={list(args.variants)} models={list(args.models)} arms={list(args.arms)} "
        f"n_rungs={args.rungs} smoke={is_smoke} dry_run={args.dry_run}"
    )

    # Load every located cell's activations up front (small on smoke; per-cell
    # arrays fit in RAM at the ~50 KB/row activation scale).
    activations_by_cell: dict[str, dict] = {}
    for variant, model, path in cells:
        acts = _load_activation_npz(path)
        if acts is None:
            _log(f"WARN empty .npz: {_rel(path)} (dry-run shell?); skipping")
            continue
        activations_by_cell[_cell_key(variant, model)] = acts

    if not activations_by_cell:
        if is_smoke:
            _log("dry-run: no populated .npz activations under --activations-dir; emitting stub")
            stub = {
                "phase": "ladder",
                "status": "dry-run-no-activations",
                "utc": datetime.now(tz=timezone.utc).isoformat(),
                "dry_run": True,
                "pilot": bool(args.pilot),
            }
            digest_path = output_dir / "ladder_digest.json"
            _write_json(digest_path, stub)
            _log(f"stub digest -> {_rel(digest_path)}")
            # noqa: phase-done-reserved
            print("[phase=done]", flush=True)
            sys.stdout.flush()
            sys.exit(0)
        print("ERROR: every located activation .npz is empty", file=sys.stderr)
        return 2

    ordered_pairs = _enumerate_ordered_pairs(cells, smoke=is_smoke)
    _log(f"pairs: {len(ordered_pairs)} ordered (source, target)")

    pair_paths: list[Path] = []
    pair_summaries: list[dict] = []
    n_rungs = max(1, min(len(RUNGS), int(args.rungs)))
    t0 = time.time()

    for (s_var, s_mod, _s_path), (t_var, t_mod, _t_path) in ordered_pairs:
        s_key = _cell_key(s_var, s_mod)
        t_key = _cell_key(t_var, t_mod)
        if s_key not in activations_by_cell or t_key not in activations_by_cell:
            continue
        s_acts = activations_by_cell[s_key]
        t_acts = activations_by_cell[t_key]

        arm_reports: dict[str, dict] = {}
        for arm in args.arms:
            ceiling = _load_target_ceiling(fits_dir, t_key, arm)
            arm_report = _fit_arm_pair(
                source_cell_key=s_key,
                target_cell_key=t_key,
                arm=arm,
                source_acts=s_acts,
                target_acts=t_acts,
                fold_map=fold_map,
                target_ceiling=ceiling,
                n_rungs=n_rungs,
                seed=int(args.seed),
                pilot=bool(args.pilot or args.dry_run),
                bootstrap_draws=int(args.bootstrap_draws),
            )
            arm_reports[arm] = arm_report

            pair_payload = {
                "phase": "ladder",
                "source": s_key,
                "target": t_key,
                "arm": arm,
                "n_rungs": n_rungs,
                "rungs_computed": list(RUNGS[:n_rungs]),
                "arm_report": arm_report,
                "target_ceiling": ceiling,
                "seed": int(args.seed),
                "dry_run": bool(args.dry_run),
                "pilot": bool(args.pilot),
                "utc": datetime.now(tz=timezone.utc).isoformat(),
                "fold_map": {
                    "path": _rel(fold_map_path),
                    "k": int(fold_map["k"]),
                    "seed": int(fold_map.get("seed", -1)),
                },
                "activations_dir": _rel(activations_dir),
                "fits_dir": _rel(fits_dir),
            }
            # Rung-scoped filename per plan §6 (rung_{i}_{source}_to_{target}_{arm}.json
            # — one file per (source, target, arm), enumerating all N rungs the run
            # covered; when a subset is computed the filename records the first-rung
            # index for provenance).
            first_rung_i = 1
            out_name = f"rung_{first_rung_i}_{s_key}_to_{t_key}_{arm}.json"
            out_path = output_dir / out_name
            _write_json(out_path, pair_payload)
            pair_paths.append(out_path)
            pair_summaries.append(
                {
                    "path": _rel(out_path),
                    "source": s_key,
                    "target": t_key,
                    "arm": arm,
                    "status": arm_report.get("status"),
                    "n_intersection": arm_report.get("n_intersection"),
                    "target_ceiling": ceiling,
                    "n_rungs": n_rungs,
                }
            )
            elapsed = time.time() - t0
            _log(
                f"pair={s_key}->{t_key} arm={arm} status={arm_report.get('status')} "
                f"n_intersection={arm_report.get('n_intersection')} "
                f"-> {_rel(out_path)} elapsed={elapsed:.1f}s"
            )

    # Uploads (real runs only; smoke tree stays under /tmp/).
    if not is_smoke and not args.skip_upload:
        try:
            _upload_to_hf(pair_paths)
        except Exception as exc:  # noqa: BLE001
            _log(f"WARN upload stage failed: {exc}")

    digest = {
        "phase": "ladder",
        "variants": list(args.variants),
        "models": list(args.models),
        "arms": list(args.arms),
        "n_rungs": n_rungs,
        "rungs_computed": list(RUNGS[:n_rungs]),
        "n_cells_found": len(cells),
        "n_pairs_run": len({(s["source"], s["target"]) for s in pair_summaries}),
        "pair_summaries": pair_summaries,
        "shared_fold_map": _rel(fold_map_path),
        "fits_dir": _rel(fits_dir),
        "dry_run": bool(args.dry_run),
        "pilot": bool(args.pilot),
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    digest_path = output_dir / "ladder_digest.json"
    _write_json(digest_path, digest)
    _log(f"digest: n_pairs_run={digest['n_pairs_run']} -> {_rel(digest_path)}")

    # noqa: phase-done-reserved
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.exit(0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--activations-dir",
        default="data/issue_2054/activations/",
        help="Unit C capture output root.",
    )
    p.add_argument(
        "--fits-dir",
        default="data/issue_2054/fits/",
        help="Unit D per-cell fit-JSON output directory (source of target ceilings).",
    )
    p.add_argument(
        "--fold-map",
        default="eval_results/issue_2054/shared_fold_map.json",
        help="Unit A shared fold-map artifact (conv_id -> fold).",
    )
    p.add_argument(
        "--output-dir",
        default="data/issue_2054/ladder/",
        help="Per-pair rung-JSON output directory.",
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
        help="Comma-separated model slugs.",
    )
    p.add_argument(
        "--arms",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=["context", "prefix"],
        help="Comma-separated arms; CLAUDE.md requires BOTH by default.",
    )
    p.add_argument(
        "--rungs",
        type=int,
        default=len(RUNGS),
        help=f"Number of rungs to compute (1..{len(RUNGS)}); default all {len(RUNGS)}.",
    )
    p.add_argument(
        "--bootstrap-draws",
        type=int,
        default=DEFAULT_BOOTSTRAP_DRAWS,
        help="Bootstrap draws over conversations within the equalized intersection.",
    )
    p.add_argument("--skip-upload", action="store_true", help="Skip the HF mirror step.")
    p.add_argument("--upload", action="store_true", help="Force HF mirror step.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Wiring-only smoke: parse CLI, exercise ladder on a tiny slice.",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="1-pair 1-fold pilot mode.",
    )
    args = p.parse_args()
    return run_phase(args)


if __name__ == "__main__":
    sys.exit(main())
