"""Component ridge cells at frozen λ* (issue #1072, plan §4.4 Phase C item 2).

One shared SVD per (fold, layer, source), reused across ALL stacked target
groups (the batched axis is the target-group loop — never a per-cell
re-factorization, per `.claude/rules/vectorize-many-cell-fits.md`). For every
group the runner fits BOTH components (par, perp := full - par) at the group's
FROZEN λ and emits eight per-context float64 channels::

    ss_res_par, ss_tot_par, ss_res_perp, ss_tot_perp,
    cross_res,  cross_tot,  ss_res_full, ss_tot_full

with ``cross_* = 2·⟨par-side, perp-side⟩`` so the exact decomposition
``ss_full = ss_par + ss_perp + cross`` holds (both res and tot), giving

    R²_full = C_par + C_perp + C_cross,   C_c = (ΣSS_tot_c - ΣSS_res_c)/ΣSS_tot_full

exactly in fp64 accounting (plan §3; the additivity identity is asserted to
< ``ADDITIVITY_TOL`` per groupxsplit — success criterion 3). The cross term is
a REPORTED channel for ALL target types, never an fp assert on the raw values
(fact-check correction, plan v2): with per-sample û the fitted component
prediction leaves span(û_i), so residual cross is not identically 0 even for
single-position targets.

``serial_component_reference`` is a deliberately INDEPENDENT numpy fp64 oracle
(dual/kernel-form ridge — ``(K+λI)⁻¹`` solve, no SVD) used by
``component_parity_gate`` (3 cells, max rel diff < PARITY_TOL — the parent's
own gate pattern extended with the cross channel).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("issue1072.component_ridge")

CHANNELS = (
    "ss_res_par",
    "ss_tot_par",
    "ss_res_perp",
    "ss_tot_perp",
    "cross_res",
    "cross_tot",
    "ss_res_full",
    "ss_tot_full",
)
ADDITIVITY_TOL = 1e-9  # plan success criterion 3 (C-scale identity, fp64)
PARITY_TOL = 1e-7  # parent ridge_battery gate pattern (max REL diff)

# pair_fn(split_name, gi) -> (Y_par (n_split, H) f64, Y_full (n_split, H) f64).
PairFn = Callable[[str, int], tuple[np.ndarray, np.ndarray]]


@dataclass
class ComponentCellResult:
    """One source X, G component target groups at per-group frozen λ.

    ``channels[split]`` has shape (n_split, G, 8) float64 in CHANNELS order;
    ``pooled[split]`` maps ``C_par|C_perp|C_cross|r2_full|w_par`` -> (G,);
    ``sens_pooled`` (optional) maps split -> (n_lam, G, 5) over
    ``sensitivity_lambdas`` in the same value order.
    """

    group_names: list[str]
    lam_values: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    pooled: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    additivity_max_dev: float = 0.0
    svd_seconds: float = 0.0
    total_seconds: float = 0.0
    n_train: int = 0
    sensitivity_lambdas: np.ndarray | None = None
    sens_pooled: dict[str, np.ndarray] | None = None


def _standardize_train(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fit_h/parent conventions: population std + 1e-9 (ridge_battery parity)."""
    x = np.asarray(x_train, dtype=np.float64)
    assert np.isfinite(x).all(), "X_train must be finite"
    mu = x.mean(0)
    sd = x.std(0) + 1e-9
    return (x - mu) / sd, mu, sd


def _pooled_from_channels(ch: np.ndarray) -> dict[str, float]:
    """Pooled contributions for ONE group from its (n, 8) channel block."""
    s = np.nansum(ch, axis=0)
    den = float(s[7])  # ss_tot_full
    if den < 1e-12:
        return {k: float("nan") for k in ("C_par", "C_perp", "C_cross", "r2_full", "w_par")}
    c_par = (float(s[1]) - float(s[0])) / den
    c_perp = (float(s[3]) - float(s[2])) / den
    c_cross = (float(s[5]) - float(s[4])) / den
    r2_full = 1.0 - float(s[6]) / den
    return {
        "C_par": c_par,
        "C_perp": c_perp,
        "C_cross": c_cross,
        "r2_full": r2_full,
        "w_par": float(s[1]) / den,
    }


def _group_channels(
    y_par: np.ndarray,
    y_full: np.ndarray,
    p_par: np.ndarray,
    p_perp: np.ndarray,
    mu_par: np.ndarray,
    mu_perp: np.ndarray,
) -> np.ndarray:
    """(n, 8) fp64 channel block for one groupxsplit (NaN rows propagate)."""
    y_perp = y_full - y_par
    r_par = y_par - p_par
    r_perp = y_perp - p_perp
    yt_par = y_par - mu_par
    yt_perp = y_perp - mu_perp
    r_full = y_full - (p_par + p_perp)
    yt_full = y_full - (mu_par + mu_perp)
    out = np.stack(
        [
            (r_par**2).sum(1),
            (yt_par**2).sum(1),
            (r_perp**2).sum(1),
            (yt_perp**2).sum(1),
            2.0 * (r_par * r_perp).sum(1),
            2.0 * (yt_par * yt_perp).sum(1),
            (r_full**2).sum(1),
            (yt_full**2).sum(1),
        ],
        axis=1,
    )
    return out


def run_component_cell(
    x_train: np.ndarray,
    eval_x: dict[str, np.ndarray],
    pair_fn: PairFn,
    group_names: list[str],
    lam_values: np.ndarray,
    device: str = "cpu",
    sensitivity_lambdas: np.ndarray | None = None,
) -> ComponentCellResult:
    """Fit par/perp components for every group at its frozen λ on ONE shared SVD.

    Args:
        x_train: (n_tr, H_in) source matrix (train rows only).
        eval_x: split name -> (n_e, H_in) eval sources.
        pair_fn: ``(split, gi) -> (Y_par, Y_full)`` fp64 blocks (``"train"`` +
            every eval split). Y_perp is DERIVED (full - par) inside, so the
            additive identity is exact by construction.
        group_names: G labels.
        lam_values: (G,) frozen λ VALUE per group (not index).
        device: "cpu" | "cuda" (torch fp64 SVD + GEMMs).
        sensitivity_lambdas: optional λ grid — pooled-only C values per λ are
            additionally computed for EVERY group (exploratory λ-reselection
            read, plan §5 last row); per-context channels stay frozen-λ only.

    Returns:
        ComponentCellResult; raises RuntimeError if the fp64 additivity
        identity exceeds ADDITIVITY_TOL on any (group, split).
    """
    import torch

    t0 = time.time()
    lam_values = np.asarray(lam_values, dtype=np.float64)
    assert lam_values.shape == (len(group_names),), lam_values.shape
    xt_n, xmu, xsd = _standardize_train(x_train)
    n_tr, _h_in = xt_n.shape

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    xt = torch.from_numpy(xt_n).to(dev)
    t_svd0 = time.time()
    u_t, s_t, vh_t = torch.linalg.svd(xt, full_matrices=False)
    svd_seconds = time.time() - t_svd0
    s2 = s_t**2

    a_eval: dict[str, torch.Tensor] = {}
    for name, x_e in eval_x.items():
        xe = np.asarray(x_e, dtype=np.float64)
        xe_n = (xe - xmu) / xsd
        a_eval[name] = torch.from_numpy(xe_n).to(dev) @ vh_t.T  # (n_e, r)

    res = ComponentCellResult(
        group_names=list(group_names),
        lam_values=lam_values,
        n_train=n_tr,
        svd_seconds=svd_seconds,
        sensitivity_lambdas=(
            np.asarray(sensitivity_lambdas, dtype=np.float64)
            if sensitivity_lambdas is not None
            else None
        ),
    )
    n_g = len(group_names)
    for name, x_e in eval_x.items():
        res.channels[name] = np.full((np.asarray(x_e).shape[0], n_g, 8), np.nan)
        res.pooled[name] = {
            k: np.full(n_g, np.nan) for k in ("C_par", "C_perp", "C_cross", "r2_full", "w_par")
        }
    if res.sensitivity_lambdas is not None:
        res.sens_pooled = {
            name: np.full((len(res.sensitivity_lambdas), n_g, 5), np.nan) for name in eval_x
        }

    max_dev = 0.0
    for gi in range(n_g):
        y_par_tr, y_full_tr = pair_fn("train", gi)
        y_par_tr = np.asarray(y_par_tr, dtype=np.float64)
        y_full_tr = np.asarray(y_full_tr, dtype=np.float64)
        assert np.isfinite(y_par_tr).all() and np.isfinite(y_full_tr).all(), (
            f"component train targets must be finite (group {group_names[gi]}) — the matched "
            "universe guarantees validity; a NaN here is a decomposition/coverage bug"
        )
        y_perp_tr = y_full_tr - y_par_tr
        mu_par = y_par_tr.mean(0)
        mu_perp = y_perp_tr.mean(0)
        b_par = u_t.T @ torch.from_numpy(y_par_tr - mu_par).to(dev)  # (r, H)
        b_perp = u_t.T @ torch.from_numpy(y_perp_tr - mu_perp).to(dev)
        filt = s_t / (s2 + float(lam_values[gi]))
        for name in eval_x:
            a = a_eval[name]
            p_par = ((a * filt[None, :]) @ b_par).cpu().numpy() + mu_par
            p_perp = ((a * filt[None, :]) @ b_perp).cpu().numpy() + mu_perp
            y_par_e, y_full_e = pair_fn(name, gi)
            ch = _group_channels(
                np.asarray(y_par_e, dtype=np.float64),
                np.asarray(y_full_e, dtype=np.float64),
                p_par,
                p_perp,
                mu_par,
                mu_perp,
            )
            res.channels[name][:, gi, :] = ch
            pooled = _pooled_from_channels(ch)
            for k, v in pooled.items():
                res.pooled[name][k][gi] = v
            if np.isfinite(pooled["r2_full"]):
                dev_g = abs(
                    pooled["r2_full"] - (pooled["C_par"] + pooled["C_perp"] + pooled["C_cross"])
                )
                max_dev = max(max_dev, dev_g)
            # Exploratory λ-reselection sensitivity (pooled only).
            if res.sensitivity_lambdas is not None:
                for li, lam in enumerate(res.sensitivity_lambdas):
                    filt_l = s_t / (s2 + float(lam))
                    pp = ((a * filt_l[None, :]) @ b_par).cpu().numpy() + mu_par
                    pq = ((a * filt_l[None, :]) @ b_perp).cpu().numpy() + mu_perp
                    ch_l = _group_channels(
                        np.asarray(y_par_e, dtype=np.float64),
                        np.asarray(y_full_e, dtype=np.float64),
                        pp,
                        pq,
                        mu_par,
                        mu_perp,
                    )
                    pl = _pooled_from_channels(ch_l)
                    assert res.sens_pooled is not None
                    res.sens_pooled[name][li, gi, :] = [
                        pl["C_par"],
                        pl["C_perp"],
                        pl["C_cross"],
                        pl["r2_full"],
                        pl["w_par"],
                    ]
    if max_dev > ADDITIVITY_TOL:
        raise RuntimeError(
            f"fp64 additivity identity violated: max |R2_full - (C_par+C_perp+C_cross)| = "
            f"{max_dev:.3e} > {ADDITIVITY_TOL} (plan success criterion 3)"
        )
    res.additivity_max_dev = max_dev
    res.total_seconds = time.time() - t0
    logger.info(
        "[component-cell] n_tr=%d G=%d device=%s svd=%.1fs total=%.1fs add_dev=%.2e",
        n_tr,
        n_g,
        str(dev),
        svd_seconds,
        res.total_seconds,
        max_dev,
    )
    return res


def serial_component_reference(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    y_par_tr: np.ndarray,
    y_full_tr: np.ndarray,
    y_par_e: np.ndarray,
    y_full_e: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Independent numpy fp64 oracle for ONE groupxsplit: dual-form ridge.

    Solves (K + λI) Z = Ỹ with K = X̃X̃ᵀ (kernel form — no SVD, a genuinely
    different route than the batched path) and returns the (n_e, 8) channel
    block in CHANNELS order.
    """
    xt_n, xmu, xsd = _standardize_train(x_train)
    xe_n = (np.asarray(x_eval, dtype=np.float64) - xmu) / xsd
    k = xt_n @ xt_n.T
    k_e = xe_n @ xt_n.T
    n = k.shape[0]
    y_par_tr = np.asarray(y_par_tr, dtype=np.float64)
    y_full_tr = np.asarray(y_full_tr, dtype=np.float64)
    y_perp_tr = y_full_tr - y_par_tr
    mu_par = y_par_tr.mean(0)
    mu_perp = y_perp_tr.mean(0)
    a = k + lam * np.eye(n)
    z_par = np.linalg.solve(a, y_par_tr - mu_par)
    z_perp = np.linalg.solve(a, y_perp_tr - mu_perp)
    p_par = k_e @ z_par + mu_par
    p_perp = k_e @ z_perp + mu_perp
    return _group_channels(
        np.asarray(y_par_e, dtype=np.float64),
        np.asarray(y_full_e, dtype=np.float64),
        p_par,
        p_perp,
        mu_par,
        mu_perp,
    )


def component_parity_gate(
    x_train: np.ndarray,
    eval_x: dict[str, np.ndarray],
    pair_fn: PairFn,
    group_names: list[str],
    lam_values: np.ndarray,
    result: ComponentCellResult,
    cells: list[tuple[int, str]],
    tol: float = PARITY_TOL,
) -> dict:
    """Hard parity gate: batched channels vs the serial oracle on ≥3 cells.

    ``cells`` = [(gi, split), ...]. Max RELATIVE diff over the 8 channels must
    be < tol (parent ridge_battery gate pattern, extended with the cross
    channel). Raises RuntimeError on failure.
    """
    assert len(cells) >= 3, "parity gate needs >= 3 cells (plan §4.4)"
    max_rel = 0.0
    worst = ""
    for gi, split in cells:
        y_par_tr, y_full_tr = pair_fn("train", gi)
        y_par_e, y_full_e = pair_fn(split, gi)
        oracle = serial_component_reference(
            x_train,
            eval_x[split],
            y_par_tr,
            y_full_tr,
            y_par_e,
            y_full_e,
            float(lam_values[gi]),
        )
        got = result.channels[split][:, gi, :]
        m = np.isfinite(oracle) & np.isfinite(got)
        scale = np.maximum(np.abs(oracle[m]), 1.0)
        rel = float(np.max(np.abs(oracle[m] - got[m]) / scale)) if m.any() else 0.0
        if rel > max_rel:
            max_rel = rel
            worst = f"{group_names[gi]}|{split}"
    rec = {
        "cells": [f"{group_names[gi]}|{split}" for gi, split in cells],
        "max_rel_diff": max_rel,
        "worst": worst,
        "tol": tol,
    }
    if max_rel > tol:
        raise RuntimeError(f"component parity gate FAIL: {rec}")
    logger.info("[parity] component batched vs serial oracle: max rel diff %.2e OK", max_rel)
    return rec
