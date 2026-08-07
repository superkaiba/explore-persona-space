#!/usr/bin/env python
"""9-rung transfer ladder driver for task #2054.

For each ordered (source_cell, target_cell, arm) pair — cells keyed on ALL
FOUR lattice axes (variant/identity, condition/phase, framing/form, model;
C6 — `issue2054_forms.cell_key`) — computes the 9 mapping-transformation
rungs of the parent
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

The driver reads capture activations (`.npz` per 4-axis cell — `v_C`, `v_A`,
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

Production pair enumeration RESTRICTS to the plan-§6 comparison classes by
default (M-R2-1; `--pair-classes`, see `PLAN6_PAIR_CLASSES`), and the
auto pilot gate extrapolates its measured 1-unit wall to the PENDING fleet,
failing loud past `--max-fleet-wall-hours` (exit 7 — a designed halt with the
projection persisted in `pilot_gate_report.json`; the #823 fleet-wall class).

Exit 0 on success. Exit 1 on fit / HF failure. Exit 2 on missing input.
Exit 7 on an over-budget fleet-wall projection (designed halt).
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
from issue2054_pilot import (  # noqa: E402
    FleetWallExceeded,
    fleet_projection_update,
    require_prior_wall_seconds,
)
from issue2054_resume import regime_values_equal  # noqa: E402
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

# Per-source M-fit memo bound (M5): each cached W is ~100 MB float64 at
# d=D_out=3584, so cap residency; the run_phase loop ALSO clears the cache
# whenever the source cell changes (pairs enumerate source-major).
_FIT_CACHE_MAX_ENTRIES = 24

# Pilot-before-fleet preferences (M5; plan §9): assistant × chat × inserted ×
# instruct, falling back to the first located cell.
PILOT_PREFERRED_CELL = (
    "conversation_paired_stories_assistant",
    "inserted",
    "chat",
    "qwen2.5-7b-instruct",
)
PILOT_RSS_ROUTE_OFF_VM_GIB = 16.0

# The assistant identity's variant slug (plan §4 "Identities": assistant is
# the only non-character identity; its chat x inserted cell is the 2x2's (a)).
ASSISTANT_VARIANT = "conversation_paired_stories_assistant"

# Plan-§6 comparison classes the production pair enumeration restricts to
# (M-R2-1): §6 item 9 cross-framing (Result 1 / §7 H1), item 10
# cross-character (Result 2 / §7 H2), item 12 the (a,b,c,d) 2x2 per
# (character, model), and the §4-req-2 cross-model read. The all-ordered-pairs
# product (~2,450 pairs at ~50 cells) is the #823 fleet-wall class and §6
# registers no read over most of it; "all" is the explicit opt-in.
PLAN6_PAIR_CLASSES = ("cross_framing", "cross_character", "twobytwo", "cross_model")

# OPT-IN classes, NOT in the production default (`PLAN6_PAIR_CLASSES` above is
# unchanged). Both serve user-requested figure reads the §6 roster does not
# enumerate, and both are reachable only via an explicit `--pair-classes`:
#
#   `onpolicy_chat_to_framing` — the ON-POLICY sibling of `cross_framing`,
#       restricted to a CHAT source (assistant on-policy chat -> the same
#       assistant's other framings). `cross_framing` is inserted-only by
#       design (the §4 interpretive split bars reading an on-policy
#       cross-framing delta as a framing effect), so any figure carrying an
#       on-policy arm must both request this class AND narrate its delta as
#       the JOINT authorship+presentation effect, never a framing effect.
#   `assistant_to_character` — assistant -> story-character transfer at FIXED
#       framing, condition and model. `cross_character` requires BOTH ends to
#       be character variants, so the "is the assistant a privileged persona?"
#       read (persona varies, framing held) has no §6 class.
EXTRA_PAIR_CLASSES = ("onpolicy_chat_to_framing", "assistant_to_character")

# Fail-loud budget for the pilot-extrapolated fleet wall (M-R2-1; exit 7 on
# an over-budget projection). Grounding: issue2054_pilot.FLEET_WALL_WARN_HOURS
# — plan §9 books the fit family as a pilot-gated VM-CPU wall, and >12 h is
# the #823 realized-wall class. Override deliberately per dispatch.
DEFAULT_MAX_FLEET_WALL_HOURS = 12.0

# The cell (c) `char_*_op*` variants (phase_d output) are IN the default so
# the 2x2's (c) leg is discoverable without operator memory (C6 review note).
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
# lattice axes; `_resolve_cells` keeps only combinations whose .npz exists.
DEFAULT_CONDITIONS = forms.CONDITIONS
DEFAULT_FORMS = forms.FORMS


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


def _find_activation_path(
    activations_dir: Path, variant: str, condition: str, form: str, model: str
) -> Path | None:
    """Locate the .npz per the capture layout:
    `<activations-dir>/<variant>/<cell_key>.npz` (4-axis key, C6).

    Falls back to any *.npz directly under `<activations-dir>` when the cell
    file is missing (smoke fixture convention, matching capture's `_flat`) —
    the caller dedupes fallback hits.
    """
    key = forms.cell_key(variant, condition, form, model)
    canonical = activations_dir / variant / f"{key}.npz"
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


def _fit_ridge(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    lambdas: np.ndarray = DEFAULT_LAMBDAS,
    dof_cap: float = DEFAULT_DOF_CAP,
) -> dict:
    """Ambient GCV-ridge FIT ONLY (M5 dedupe: fit/apply split so one fit can
    be applied at many inputs — and MEMOIZED across pairs sharing a source).

    Standardize X on the train fold; center Y; primary reconstruction via SVD.
    Returns the fitted model {xmu, xsd, ymu, W, info}.
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
    return {"xmu": xmu, "xsd": xsd, "ymu": ymu, "W": W, "info": info}


def _apply_ridge(model: dict, X_apply: np.ndarray) -> np.ndarray:
    """Apply a `_fit_ridge` model at new inputs (cheap — never a refit)."""
    Xa = (X_apply.astype(np.float64) - model["xmu"]) / model["xsd"]
    return Xa @ model["W"] + model["ymu"]


def _fit_ridge_and_apply(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    apply_at: dict[str, np.ndarray],
    *,
    lambdas: np.ndarray = DEFAULT_LAMBDAS,
    dof_cap: float = DEFAULT_DOF_CAP,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit once, apply at every key in `apply_at` (thin fit+apply wrapper)."""
    model = _fit_ridge(X_train, Y_train, lambdas=lambdas, dof_cap=dof_cap)
    preds = {key: _apply_ridge(model, X_apply) for key, X_apply in apply_at.items()}
    return preds, model["info"]


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
    *,
    source_fit: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute all 9 rung predictions at the target's held-out fold rows.

    Mirrors `scripts/issue1345_ladder_rungs.py::_rungs_for`, but on numpy +
    no per-eval refits. EXACTLY THREE GCV-SVD fits per fold (M5 dedupe —
    down from five: the pre-fix `preds_B9` re-fit the identical Ys_tr→Yt_tr
    map as `preds_B`, and `preds_shift` re-fit the identical Xs_tr→Ys_tr
    source map as `preds_M`; both are now extra `apply_at` targets of the
    ONE fit):

      A — target-contexts -> source-contexts (rungs 7/9),
      M — the source map v_C_source -> v_A_source (rungs 1/2/4/5/6/7 inputs),
      B — source-answer -> target-answer (rungs 8/9).

    `source_fit` (optional) is a prefit M model from the caller's per-source
    memo — pairs sharing (source, arm, fold, train rows) skip the M fit.

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
    model_A = _fit_ridge(Xt_tr, Xs_tr)
    info_A = model_A["info"]
    Xs_hat_tr = _apply_ridge(model_A, Xt_tr)
    Xs_hat_te = _apply_ridge(model_A, Xt_te)

    # Context / answer mean shifts (source -> target) — dx BEFORE the M
    # applications so rung 2's shifted input rides the SAME M fit (M5).
    dx = Xt_tr.astype(np.float64).mean(axis=0) - Xs_tr.astype(np.float64).mean(axis=0)
    dy = Yt_tr.astype(np.float64).mean(axis=0) - Ys_tr.astype(np.float64).mean(axis=0)

    # M: source ridge  v_C_source -> v_A_source — ONE fit (or the caller's
    # memoized one), applied at FIVE inputs: target contexts train/te, the
    # target→source reparam of both, and rung 2's mean-shifted held-out.
    model_M = source_fit if source_fit is not None else _fit_ridge(Xs_tr, Ys_tr)
    info_M = model_M["info"]
    P_tr = _apply_ridge(model_M, Xt_tr)
    P_te = _apply_ridge(model_M, Xt_te)
    P7_tr = _apply_ridge(model_M, Xs_hat_tr)
    P7_te = _apply_ridge(model_M, Xs_hat_te)
    P_shift = _apply_ridge(model_M, Xt_te.astype(np.float64) - dx)

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

    # Rungs 8 + 9 — ONE ridge (source-answer -> target-answer) applied to BOTH
    # the direct source-map prediction (rung 8) and the context-reparam chain
    # prediction (rung 9): same train clouds ⇒ same fit (M5 dedupe).
    model_B = _fit_ridge(Ys_tr, Yt_tr)
    info_B = model_B["info"]
    P_te_reparam = _apply_ridge(model_B, P_te)
    P7_te_reparam = _apply_ridge(model_B, P7_te)

    rung_preds: dict[str, np.ndarray] = {
        "1_direct": P_te,
        "2_ctx_offset": P_shift,
        "3_ans_offset": P_te + dy,
        "4_bias_refit": P_te + bstar,
        "5_global_scale": a * (P_te - pmu) + ymu,
        "6_rotation": rot_te,
        "7_ctx_reparam": P7_te + b7,
        "8_ans_reparam": P_te_reparam,
        "9_full_AMB": P7_te_reparam,
    }

    info = {
        "source_fit": info_M,
        "source_fit_memoized": bool(source_fit is not None),
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
    fit_cache: dict | None = None,
) -> dict:
    """Compute the 9-rung ladder for one ordered (source, target, arm) pair.

    Equalize-down: fold rows keyed on the INTERSECTION of source + target
    conv_ids (post-arm mask). Both cells' arm rows re-index to that shared set.

    `fit_cache` (M5): a caller-scoped memo of source-map fits keyed on
    (source_cell, arm, fold, sha of the realized train ids) — pairs sharing a
    source AND the same equalized train rows skip the M re-fit. NOTE the
    per-pair equalize-down makes the train rows PAIR-dependent, so the memo
    hits only when intersections coincide (the conv-matched production design
    makes that common within a comparison group); it is exact by construction
    (keyed on the realized rows, never assumed).
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

        # Per-source M-fit memo (M5): exact key on the realized train rows.
        # On a miss the fit happens HERE and is stored for later same-source
        # pairs; `_compute_rungs_for_fold` then never re-fits M.
        source_fit: dict | None = None
        if fit_cache is not None:
            train_sha = hashlib.sha256("\n".join(train_ids).encode()).hexdigest()
            cache_key = (source_cell_key, arm, fold_i, train_sha)
            source_fit = fit_cache.get(cache_key)
            if source_fit is None:
                source_fit = _fit_ridge(Xs_tr, Ys_tr)
                if len(fit_cache) >= _FIT_CACHE_MAX_ENTRIES:
                    fit_cache.clear()  # bounded memory (each W is ~100 MB f64)
                fit_cache[cache_key] = source_fit

        rung_preds, info_fit = _compute_rungs_for_fold(
            Xs_tr=Xs_tr,
            Ys_tr=Ys_tr,
            Xt_tr=Xt_tr,
            Xt_te=Xt_te,
            Yt_tr=Yt_tr,
            source_fit=source_fit,
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


# A located cell: (variant, condition, form, model, activation path) — the
# 4-axis lattice identity (C6) plus its .npz.
_Cell = tuple[str, str, str, str, Path]


def _cell_key(variant: str, condition: str, form: str, model: str) -> str:
    return forms.cell_key(variant, condition, form, model)


def _resolve_cells(
    activations_dir: Path,
    variants: list[str],
    conditions: list[str],
    form_list: list[str],
    models: list[str],
) -> list[_Cell]:
    """4-axis cell enumeration (C6); a flat smoke-fixture .npz (resolved by the
    fallback, living directly under `activations_dir`) attaches to AT MOST ONE
    cell so the default condition×form product cannot duplicate it.
    """
    out: list[_Cell] = []
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


def _is_character_variant(variant: str) -> bool:
    """Character-family identity (plan §4: HELIOS/Wren/Dana/Vex + their
    `_op`/`_op_base` on-policy variants); the assistant is the only
    non-character identity in the lattice."""
    return variant.startswith("char_")


def _twobytwo_group(cell: _Cell) -> tuple[str, str] | None:
    """(base_character, model) 2x2 group of a character-family cell (plan §4
    Block 3 / fits `_comparison_group_key`); None for non-character cells."""
    variant, _condition, _form, model = cell[:4]
    if _is_character_variant(variant):
        return (forms.base_character(variant), model)
    return None


def _is_chat_anchor(cell: _Cell) -> bool:
    """The 2x2's (a) cell — chat-authored, chat-presented: the assistant
    chat x inserted cell, shared by every (character, model) group of the
    same model (plan §4 Block 3 table)."""
    variant, condition, form, _model = cell[:4]
    return variant == ASSISTANT_VARIANT and condition == "inserted" and form == "chat"


def _pair_class(s: _Cell, t: _Cell) -> str | None:
    """First plan-§6 comparison class the ordered (source, target) pair
    serves, or None when no registered §6 cross-cell read consumes it
    (M-R2-1). A pair matching several classes counts under the first in
    PLAN6_PAIR_CLASSES order (a set-membership restriction — dedup-free)."""
    s_var, s_cond, s_form, s_mod = s[:4]
    t_var, t_cond, t_form, t_mod = t[:4]
    # §6 item 9 (Result 1) / §7 H1 within-scaffold cross-boundary: cross-
    # framing within ONE identity, INSERTED (controlled) arm only — the §4
    # interpretive split bars reading an on-policy cross-framing delta as a
    # framing effect.
    if s_var == t_var and s_mod == t_mod and s_cond == t_cond == "inserted" and s_form != t_form:
        return "cross_framing"
    # §6 item 10 (Result 2) / §7 H2 cross-scaffold same-boundary: cross-
    # character within one (form, condition, model).
    if (
        s_mod == t_mod
        and s_cond == t_cond
        and s_form == t_form
        and _is_character_variant(s_var)
        and _is_character_variant(t_var)
        and forms.base_character(s_var) != forms.base_character(t_var)
    ):
        return "cross_character"
    # §6 item 12: transfers within one (character, model) 2x2 — same-group
    # cells pair with each other, and the (a) chat anchor (assistant chat x
    # inserted, same model) pairs with every group's (b)/(c)/(d) cells.
    if s_mod == t_mod:
        s_group, t_group = _twobytwo_group(s), _twobytwo_group(t)
        if s_group is not None and s_group == t_group:
            return "twobytwo"
        if (s_group is not None and _is_chat_anchor(t)) or (
            t_group is not None and _is_chat_anchor(s)
        ):
            return "twobytwo"
    # Plan §4 req 2 — model is a read-side variable: same cell, other model.
    if s_var == t_var and s_cond == t_cond and s_form == t_form and s_mod != t_mod:
        return "cross_model"
    # ---- opt-in classes (EXTRA_PAIR_CLASSES) -------------------------------
    # Appended AFTER every §6 class on purpose: `_pair_class` returns the FIRST
    # match, so a new arm placed earlier could shadow a §6 class and silently
    # drop that pair from a production run. Neither arm below can be reached by
    # a pair the §6 checks already claim (assistant is not a character variant;
    # `_is_chat_anchor` requires form == "chat" AND condition == "inserted").
    if (
        s_var == t_var == ASSISTANT_VARIANT
        and s_mod == t_mod
        and s_cond == t_cond == "on_policy"
        and s_form == "chat"
        and t_form != "chat"
    ):
        return "onpolicy_chat_to_framing"
    if (
        s_var == ASSISTANT_VARIANT
        and _is_character_variant(t_var)
        and s_mod == t_mod
        and s_cond == t_cond
        and s_form == t_form
        and s_form in forms.STORY_FORMS
    ):
        return "assistant_to_character"
    return None


def _enumerate_ordered_pairs(
    cells: list[_Cell],
    *,
    smoke: bool,
    pair_classes: tuple[str, ...] = PLAN6_PAIR_CLASSES,
) -> list[tuple[_Cell, _Cell]]:
    """Ordered (source, target) pairs of cells.

    - Full run (M-R2-1): pairs RESTRICT to the plan-§6 comparison classes by
      default (`_pair_class`); the all-ordered-pairs product is the #823
      fleet-wall class. `--pair-classes all` restores the full product
      (explicit opt-in; the fleet-wall fence still applies).
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
    wanted = set(pair_classes)
    pairs: list[tuple[_Cell, _Cell]] = []
    for s in cells:
        for t in cells:
            if s == t:
                continue
            if "all" in wanted or _pair_class(s, t) in wanted:
                pairs.append((s, t))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Upload


def _upload_to_hf(pair_paths: list[Path]) -> None:
    """Mirror rung JSONs — ONE bulk `upload_folder` commit. FATAL on failure
    (M2): a swallowed upload + `[phase=done]` silently strands the rungs."""
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    if not pair_paths:
        return
    parents = {p.parent.resolve() for p in pair_paths}
    if len(parents) != 1:
        raise RuntimeError(
            f"heterogeneous ladder roots — cannot compose one bulk upload: {parents}"
        )
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
        raise RuntimeError(
            f"upload set resolved EMPTY against declared rung JSONs ({len(pair_paths)} paths)"
        )
    url = _upload_folder_filtered(
        root,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{TASK_PREFIX}/ladder",
        allow_patterns=allow_patterns,
        expected_repo_paths=expected_paths,
    )
    if not url:
        # _upload_folder_filtered is fail-soft by RETURN on every failure
        # shape (missing token, incomplete verify, terminal exception -> "")
        # — an empty return is a failed upload, not a success (M2).
        raise RuntimeError(
            f"rung-JSON bulk upload failed or incomplete -> {TASK_PREFIX}/ladder/ "
            "(returned no path; local files kept)"
        )
    _log(f"uploaded {len(allow_patterns)} rung JSON(s) in one bulk commit")


# ─────────────────────────────────────────────────────────────────────────────
# CLI driver


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def _pair_intersection_sha(
    source_acts: dict, target_acts: dict, arm: str, fold_of: dict
) -> tuple[str, int]:
    """(sha256 over the sorted realized pair intersection, its size) — the
    resume key's identity pin for one (source, target, arm) pair (C9/M6)."""
    _, _, s_ids = _select_arm(source_acts, arm)
    _, _, t_ids = _select_arm(target_acts, arm)
    inter = sorted(set(s_ids) & set(t_ids) & set(fold_of.keys()))
    return hashlib.sha256("\n".join(inter).encode()).hexdigest(), len(inter)


# Regime keys a resumed pair must match EXACTLY against its existing rung
# JSON (C9/M6 — every output-affecting key; the #722-r3 rule).
_PAIR_RESUME_KEYS = (
    "source",
    "target",
    "arm",
    "n_rungs",
    "seed",
    "bootstrap_draws",
    "pilot",
    "dry_run",
    "target_ceiling",
    "intersection_sha256",
)
_PAIR_RESUME_FOLD_KEYS = {"fold_map_k": "k", "fold_map_seed": "seed"}


def _pair_resume_check(out_path: Path, expected: dict) -> tuple[bool, str]:
    """(skip?, reason) for one pair against its existing rung JSON."""
    if not out_path.is_file():
        return False, ""
    try:
        with out_path.open(encoding="utf-8") as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "existing rung JSON unreadable"
    # NaN-aware equality (issue2054_resume.regime_values_equal): the regime's
    # target_ceiling is legitimately NaN when the target cell's own ceiling is
    # degenerate, and bare != marks EVERY re-run "regime changed" (nan != nan)
    # so the pair recomputes forever (Unit F smoke catch).
    mismatched = [
        k for k in _PAIR_RESUME_KEYS if not regime_values_equal(existing.get(k), expected.get(k))
    ]
    fm = existing.get("fold_map") or {}
    mismatched += [
        ek
        for ek, fk in _PAIR_RESUME_FOLD_KEYS.items()
        if not regime_values_equal(fm.get(fk), expected.get(ek))
    ]
    if mismatched:
        return False, f"regime keys changed: {mismatched}"
    status = (existing.get("arm_report") or {}).get("status")
    if status != "ok":
        return False, f"existing rung JSON status={status!r}"
    return True, "rung JSON complete under matching regime"


def _run_ladder_pilot_gate(
    ordered_pairs: list,
    activations_by_cell: dict,
    fold_map: dict,
    fits_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    n_pending_units: int,
) -> None:
    """Pilot-before-fleet (M5; plan §9): ONE (source, target, arm) pair, ONE
    fold, at production shape, measured (wall + peak RSS) and persisted to
    `<output-dir>/pilot_gate_report.json` BEFORE the fleet loop.

    M-R2-1: the measured 1-unit 1-fold wall is extrapolated to the PENDING
    fleet (`wall x fold_k x n_pending_units` — pending, so a mostly-resumed
    re-run never trips the fence for work that will not run, #1586) and the
    projection is enforced against `--max-fleet-wall-hours`: an over-budget
    projection raises FleetWallExceeded (exit 7 — a designed halt, #1415
    convention), with the report JSON written BEFORE the raise.
    """
    import resource as _resource

    report_path = output_dir / "pilot_gate_report.json"
    fold_k = int(fold_map["k"])
    if n_pending_units <= 0:
        _log("pilot gate: 0 pending (pair, arm) units — skipping (fully resumed fleet)")
        return
    if report_path.is_file() and not args.overwrite:
        try:
            with report_path.open(encoding="utf-8") as f:
                prior = json.load(f)
            # Measurement-affecting knobs (r2 Minor 5: the single-knob compare
            # reused a stale pilot across changed arm/seed regimes).
            prior_matches = (
                prior.get("bootstrap_draws") == int(args.bootstrap_draws)
                and prior.get("arm") == args.arms[0]
                and prior.get("seed") == int(args.seed)
            )
            if prior_matches:
                _log(f"pilot gate: prior report matches ({_rel(report_path)}); skipping")
                # Re-derive the fleet projection for THIS run's pending count
                # from the prior MEASURED wall (resume-aware, M-R2-1). r3
                # Minor 1: a prior report lacking the measured wall FAILS
                # LOUD — a silent 0.0 default would project a fleet wall of
                # 0 and disarm the fence.
                prior_wall = require_prior_wall_seconds(prior, report_path)
                fleet_projection_update(
                    report_path,
                    prior,
                    wall_seconds=prior_wall,
                    n_fleet_units=n_pending_units,
                    fold_k=fold_k,
                    log=_log,
                    max_fleet_wall_hours=float(args.max_fleet_wall_hours),
                    units_basis="pending (pair, arm) units",
                )
                return
        except (OSError, json.JSONDecodeError):
            pass

    pilot_pair = None
    for s, t in ordered_pairs:
        s_key = _cell_key(s[0], s[1], s[2], s[3])
        t_key = _cell_key(t[0], t[1], t[2], t[3])
        if s_key in activations_by_cell and t_key in activations_by_cell:
            pilot_pair = (s, t, s_key, t_key)
            if (s[0], s[1], s[2], s[3]) == PILOT_PREFERRED_CELL:
                break
    if pilot_pair is None:
        _log("pilot gate: no runnable pair located; skipping")
        return
    _s, _t, s_key, t_key = pilot_pair
    arm = args.arms[0]
    _log(f"pilot gate: 1-pair 1-fold measured pilot on {s_key} -> {t_key} ({arm})")
    t0 = time.time()
    pilot_report = _fit_arm_pair(
        source_cell_key=s_key,
        target_cell_key=t_key,
        arm=arm,
        source_acts=activations_by_cell[s_key],
        target_acts=activations_by_cell[t_key],
        fold_map=fold_map,
        target_ceiling=_load_target_ceiling(fits_dir, t_key, arm),
        n_rungs=len(RUNGS),
        seed=int(args.seed),
        pilot=True,  # 1 fold — production shape otherwise
        bootstrap_draws=int(args.bootstrap_draws),
    )
    wall = time.time() - t0
    peak_rss_gib = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    payload = {
        "phase": "ladder-pilot-gate",
        "source": s_key,
        "target": t_key,
        "arm": arm,
        "seed": int(args.seed),
        "bootstrap_draws": int(args.bootstrap_draws),
        "wall_seconds": round(wall, 3),
        "peak_rss_gib": round(peak_rss_gib, 3),
        "rss_route_off_vm_gib": PILOT_RSS_ROUTE_OFF_VM_GIB,
        "status": pilot_report.get("status"),
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    _log(f"pilot gate: wall={wall:.1f}s peak_rss={peak_rss_gib:.2f} GiB -> {_rel(report_path)}")
    if peak_rss_gib >= PILOT_RSS_ROUTE_OFF_VM_GIB:
        _log(
            f"WARN pilot peak RSS {peak_rss_gib:.2f} GiB >= "
            f"{PILOT_RSS_ROUTE_OFF_VM_GIB} GiB — plan §9 routes this fit family "
            "OFF the shared VM (cpu-mid / cpu-bigmem); dispatcher decision"
        )
    # M-R2-1: pilot -> fleet projection over the PENDING units + fail-loud
    # fence (writes the report, incl. on the raise path — artifact-routed).
    fleet_projection_update(
        report_path,
        payload,
        wall_seconds=wall,
        n_fleet_units=n_pending_units,
        fold_k=fold_k,
        log=_log,
        max_fleet_wall_hours=float(args.max_fleet_wall_hours),
        units_basis="pending (pair, arm) units",
    )


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

    is_smoke = str(output_dir).startswith("/tmp/") or args.dry_run or args.pilot
    _log(
        f"start: variants={list(args.variants)} conditions={list(args.conditions)} "
        f"forms={list(args.forms)} models={list(args.models)} arms={list(args.arms)} "
        f"n_rungs={args.rungs} smoke={is_smoke} dry_run={args.dry_run}"
    )

    # Load every located cell's activations up front (small on smoke; per-cell
    # arrays fit in RAM at the ~50 KB/row activation scale).
    activations_by_cell: dict[str, dict] = {}
    for variant, condition, form, model, path in cells:
        acts = _load_activation_npz(path)
        if acts is None:
            _log(f"WARN empty .npz: {_rel(path)} (dry-run shell?); skipping")
            continue
        activations_by_cell[_cell_key(variant, condition, form, model)] = acts

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

    ordered_pairs = _enumerate_ordered_pairs(
        cells, smoke=is_smoke, pair_classes=tuple(args.pair_classes)
    )
    pair_class_counts: dict[str, int] = {}
    for s, t in ordered_pairs:
        cls = "self_transfer" if s == t else (_pair_class(s, t) or "all_opt_in")
        pair_class_counts[cls] = pair_class_counts.get(cls, 0) + 1
    n_full_product = len(cells) * max(0, len(cells) - 1)
    _log(
        f"pairs: {len(ordered_pairs)} ordered (source, target) under "
        f"pair-classes={','.join(args.pair_classes)} (full ordered product: "
        f"{n_full_product}); per-class: {pair_class_counts}"
    )
    if not is_smoke and len(cells) >= 2 and not ordered_pairs:
        # r3 Minor 2: a production run that would compute NOTHING must not
        # exit 0 with an empty digest — "ran fine" and "computed nothing"
        # have to differ in rc. Exit 2 = the missing-input class (same as
        # the no-activations refusals above); smoke keeps its self-transfer
        # fallback and never reaches here.
        print(
            f"ERROR: pair-class restriction matched ZERO pairs across {len(cells)} "
            f"located cells (pair-classes={','.join(args.pair_classes)}) — a non-smoke "
            "ladder run that would compute nothing must not exit 0; check "
            "--pair-classes against the located cell axes",
            file=sys.stderr,
        )
        return 2

    pair_paths: list[Path] = []
    pair_summaries: list[dict] = []
    n_rungs = max(1, min(len(RUNGS), int(args.rungs)))

    # Resume pre-pass (C9/M6) — ONE pass over every (pair, arm) unit builds
    # the PENDING work list; the pilot gate's fleet projection keys on the
    # pending count (M-R2-1; #1586: gates scale to pending work, so a
    # mostly-resumed re-run never trips the fence for work that will not run).
    pending: list[dict] = []
    for (s_var, s_cond, s_form, s_mod, _s_path), (
        t_var,
        t_cond,
        t_form,
        t_mod,
        _t_path,
    ) in ordered_pairs:
        s_key = _cell_key(s_var, s_cond, s_form, s_mod)
        t_key = _cell_key(t_var, t_cond, t_form, t_mod)
        if s_key not in activations_by_cell or t_key not in activations_by_cell:
            continue
        s_acts = activations_by_cell[s_key]
        t_acts = activations_by_cell[t_key]
        for arm in args.arms:
            ceiling = _load_target_ceiling(fits_dir, t_key, arm)
            # Rung-scoped filename per plan §6 (rung_{i}_{source}_to_{target}_{arm}.json
            # — one file per (source, target, arm), enumerating all N rungs the run
            # covered; when a subset is computed the filename records the first-rung
            # index for provenance).
            first_rung_i = 1
            out_name = f"rung_{first_rung_i}_{s_key}_to_{t_key}_{arm}.json"
            out_path = output_dir / out_name

            # Resume (C9/M6): a pair whose rung JSON already carries this
            # exact regime (incl. the realized pair intersection + the target
            # ceiling the ratios divide by) is skipped. Self-describing JSON
            # ⇒ a mismatch RECOMPUTES (logged) — no refusal needed.
            inter_sha, n_inter = _pair_intersection_sha(s_acts, t_acts, arm, fold_map["fold_of"])
            expected_regime = {
                "source": s_key,
                "target": t_key,
                "arm": arm,
                "n_rungs": n_rungs,
                "seed": int(args.seed),
                "bootstrap_draws": int(args.bootstrap_draws),
                "pilot": bool(args.pilot),
                "dry_run": bool(args.dry_run),
                "target_ceiling": ceiling,
                "intersection_sha256": inter_sha,
                "fold_map_k": int(fold_map["k"]),
                "fold_map_seed": int(fold_map.get("seed", -1)),
            }
            if not args.overwrite and not args.dry_run and not args.pilot:
                skip, why = _pair_resume_check(out_path, expected_regime)
                if skip:
                    pair_paths.append(out_path)
                    pair_summaries.append(
                        {
                            "path": _rel(out_path),
                            "source": s_key,
                            "target": t_key,
                            "arm": arm,
                            "status": "resumed",
                            "n_intersection": n_inter,
                            "target_ceiling": ceiling,
                            "n_rungs": n_rungs,
                        }
                    )
                    _log(f"pair={s_key}->{t_key} arm={arm} RESUME skip ({why})")
                    continue
                if why:
                    _log(f"pair={s_key}->{t_key} arm={arm} recompute: {why}")
            pending.append(
                {
                    "s_key": s_key,
                    "t_key": t_key,
                    "arm": arm,
                    "out_path": out_path,
                    "ceiling": ceiling,
                    "inter_sha": inter_sha,
                }
            )
    _log(f"units: {len(pending)} pending (pair, arm) unit(s); {len(pair_paths)} resumed")

    # Pilot-before-fleet (M5; plan §9 pilot-gate): ONE pair, 1 fold, at
    # production shape, measured + persisted BEFORE the fleet loop — then the
    # M-R2-1 pilot->fleet projection over the PENDING units (fail-loud fence).
    if not args.dry_run and not args.pilot and not args.skip_pilot_gate:
        _run_ladder_pilot_gate(
            ordered_pairs,
            activations_by_cell,
            fold_map,
            fits_dir,
            args,
            output_dir,
            n_pending_units=len(pending),
        )

    t0 = time.time()

    # Per-source M-fit memo (M5), cleared when the source cell changes —
    # `_enumerate_ordered_pairs` is source-major (the pre-pass preserves its
    # order), so residency stays bounded to one source's folds.
    fit_cache: dict = {}
    current_source: str | None = None

    for unit in pending:
        s_key = unit["s_key"]
        t_key = unit["t_key"]
        arm = unit["arm"]
        out_path = unit["out_path"]
        ceiling = unit["ceiling"]
        inter_sha = unit["inter_sha"]
        if s_key != current_source:
            fit_cache.clear()
            current_source = s_key
        s_acts = activations_by_cell[s_key]
        t_acts = activations_by_cell[t_key]

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
            fit_cache=fit_cache,
        )

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
            "bootstrap_draws": int(args.bootstrap_draws),
            "intersection_sha256": inter_sha,
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

    # Uploads (real runs only; smoke tree stays under /tmp/). FATAL on
    # failure (M2): `[phase=done]` must never report done with the rung JSONs
    # un-persisted. No try/except.
    if not is_smoke and not args.skip_upload:
        _upload_to_hf(pair_paths)

    # Fleet projection for the digest (M-R2-1: "report the projected fleet
    # wall in the digest") — read back from the pilot-gate report artifact.
    projected_fleet_wall_seconds = None
    pilot_report_path = output_dir / "pilot_gate_report.json"
    if pilot_report_path.is_file():
        try:
            with pilot_report_path.open(encoding="utf-8") as f:
                projected_fleet_wall_seconds = json.load(f).get("projected_fleet_wall_seconds")
        except (OSError, json.JSONDecodeError):
            pass

    digest = {
        "phase": "ladder",
        "variants": list(args.variants),
        "conditions": list(args.conditions),
        "forms": list(args.forms),
        "models": list(args.models),
        "arms": list(args.arms),
        "n_rungs": n_rungs,
        "rungs_computed": list(RUNGS[:n_rungs]),
        "n_cells_found": len(cells),
        "pair_classes": list(args.pair_classes),
        "pair_class_counts": pair_class_counts,
        "n_pairs_enumerated": len(ordered_pairs),
        "n_full_ordered_product": n_full_product,
        "n_units_pending": len(pending),
        "n_units_resumed": sum(1 for s in pair_summaries if s.get("status") == "resumed"),
        "projected_fleet_wall_seconds": projected_fleet_wall_seconds,
        "max_fleet_wall_hours": float(args.max_fleet_wall_hours),
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
    p.add_argument(
        "--pair-classes",
        type=lambda s: tuple(x.strip() for x in s.split(",") if x.strip()),
        default=PLAN6_PAIR_CLASSES,
        help=(
            "Comma-separated plan-§6 pair classes to enumerate "
            f"({', '.join(PLAN6_PAIR_CLASSES)}; default: all four — M-R2-1; "
            f"opt-in, never default: {', '.join(EXTRA_PAIR_CLASSES)}), "
            "or 'all' for the full ordered product (explicit opt-in; the "
            "fleet-wall fence still applies)."
        ),
    )
    p.add_argument(
        "--max-fleet-wall-hours",
        type=float,
        default=DEFAULT_MAX_FLEET_WALL_HOURS,
        help=(
            "Fail-loud budget for the pilot-extrapolated fleet wall (M-R2-1): "
            "projected wall over this exits 7 (a designed halt with the "
            "projection persisted in pilot_gate_report.json, never a crash)."
        ),
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
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "recompute pairs even when a regime-matching rung JSON exists "
            "(default resumes completed pairs — C9/M6)"
        ),
    )
    p.add_argument(
        "--skip-pilot-gate",
        action="store_true",
        help=(
            "skip the automatic 1-pair measured pilot leg before the fleet "
            "(M5/plan §9; only when a standalone pilot already ran)"
        ),
    )
    args = p.parse_args()
    valid_classes = set(PLAN6_PAIR_CLASSES) | set(EXTRA_PAIR_CLASSES) | {"all"}
    if not args.pair_classes:
        # r3 Minor 2: an empty --pair-classes '' parses to () and would pass
        # the unknown-class check below vacuously, then enumerate zero pairs.
        p.error("--pair-classes must name at least one class (or 'all'); got an empty value")
    unknown = [c for c in args.pair_classes if c not in valid_classes]
    if unknown:
        p.error(f"unknown --pair-classes {unknown} (expected {sorted(valid_classes)})")
    try:
        return run_phase(args)
    except FleetWallExceeded as exc:
        # Designed halt (M-R2-1): the projection is persisted in
        # pilot_gate_report.json before the raise — route on the artifact,
        # never treat exit 7 as an anonymous crash (#1415 convention).
        print(f"ERROR {exc}", file=sys.stderr)
        return 7


if __name__ == "__main__":
    sys.exit(main())
