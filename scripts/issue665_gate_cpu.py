#!/usr/bin/env python
"""Issue #665 Phase 3 — CPU gate arms (A3.6/a/b, A3.7, A3.8, A3.9, A3.10, joint
factorization, single-context).

All arms are deterministic linear algebra over the #664 trained store tensors,
streamed one cell at a time (load → compute → free) to stay under the VM
analysis-footprint floor (plan §8: peak ≤ 6 GB). The PRIMARY DV is the
activation realized gate ĝ^real (imported, never reimplemented). The whitened
key-query gate (A3.9/A3.10) uses the net-new ``analysis.whitened_gate`` module
(which clears the B3 reduction unit test).

Writes one JSON per (arm, cell): ``eval_results/issue_665/<arm>/<cell>.json``.
Per-arm reproducibility metadata (git commit, timestamps, lambda) in every file.

Usage:
    uv run python scripts/issue665_gate_cpu.py --scope content
    uv run python scripts/issue665_gate_cpu.py --cells bm_default_contra_d1_seed42 \
        --layers 8 --smoke   # 1 cell, 1 layer, 5 contexts (the §6.5 CPU smoke)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess

import issue665_common as C
import numpy as np

from explore_persona_space.analysis.gate_io import (
    gate_per_layer,
    load_cell,
    load_sigma_c,
)
from explore_persona_space.analysis.whitened_gate import (
    METRIC_KEYS,
    key_query_gate,
    metric_ablation,
    raw_cosine_gate,
    whitened_gate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("issue665_gate_cpu")

ARMS = ("a36", "a36a", "a36b", "a37", "a38", "a39", "a310", "joint", "single_ctx")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO).decode().strip()
    except Exception:
        return "unknown"


def _meta_stamp(cell: str, lam: float, extra: dict | None = None) -> dict:
    m = {
        "cell": cell,
        "behavior": C.behavior_for_cell(cell),
        "column": C.column_for_cell(cell),
        "role_class": C.role_class_for_cell(cell),
        "read_layer": C.read_layer_for_cell(cell),
        "lambda": lam,
        "git_commit": _git_commit(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "cc_recipe": C.CC_RECIPE,
        **(C.parse_cell(cell)),
    }
    if extra:
        m.update(extra)
    return m


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via ranked Pearson (no scipy dep on the linalg path)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return _pearson(rx, ry)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.linalg.norm(xc) * np.linalg.norm(yc)
    if den < 1e-30:
        return float("nan")
    return float((xc @ yc) / den)


def _partial_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Partial Spearman of (x, y) controlling for z (C10): rank everything,
    residualize ranks of x and y on ranks of z, correlate residuals."""
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    if x.size < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rz = np.argsort(np.argsort(z)).astype(np.float64)

    def _resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        bc = b - b.mean()
        denom = bc @ bc
        if denom < 1e-30:
            return a - a.mean()
        beta = (bc @ (a - a.mean())) / denom
        return a - a.mean() - beta * bc

    return _pearson(_resid(rx, rz), _resid(ry, rz))


# ── A3.8 — rank-one gated write ───────────────────────────────────────────────
def arm_a38(sc, layer: int, lam: float) -> dict:
    """ĝ^real per target + per-target rank-one residual + stacked-ΔV SVD."""
    v_plus = sc.tensors["v_plus"]
    v0 = sc.tensors["v0"]
    ghat, wnorm = gate_per_layer(v_plus, v0, sc.source_idx)  # (C,L),(L,)
    dv = (v_plus - v0).numpy().astype(np.float64)  # (C,L,d)
    w = dv[sc.source_idx, layer]  # (d,) ŵ at source, this layer
    wnorm2 = float(w @ w)
    n_ctx = dv.shape[0]
    # per-target rank-one residual ‖Δv(C') - ŵ ĝ‖ / ‖Δv(C')‖
    residuals = []
    dv_l = dv[:, layer]  # (C,d)
    ghat_l = ghat[:, layer]  # (C,)
    for ci in range(n_ctx):
        dvc = dv_l[ci]
        approx = w * ghat_l[ci]
        nrm = np.linalg.norm(dvc)
        res = np.linalg.norm(dvc - approx) / nrm if nrm > 1e-30 else float("nan")
        residuals.append(float(res))
    # stacked ΔV over bystanders (drop source row) → SVD spectrum
    bystander_mask = np.arange(n_ctx) != sc.source_idx
    dV = dv_l[bystander_mask]  # (C-1, d)
    svals = np.linalg.svd(dV, compute_uv=False)
    s2 = (svals**2).sum()
    sigma1_frac = float(svals[0] ** 2 / s2) if s2 > 1e-30 else float("nan")
    sigma2_over_1 = (
        float(svals[1] / svals[0]) if len(svals) > 1 and svals[0] > 1e-30 else float("nan")
    )
    # top-left-singular-vector via thin SVD on dV (u1 = top RIGHT singular vec of ΔVᵀ)
    _, _, vt = np.linalg.svd(dV, full_matrices=False)
    u1 = vt[0]
    cos_u1_w = (
        float(abs((u1 @ w) / (np.linalg.norm(u1) * np.sqrt(wnorm2))))
        if wnorm2 > 1e-30
        else (float("nan"))
    )
    return {
        "ghat_by_context": [round(float(g), 6) for g in ghat_l],
        "rankone_residual_by_context": [round(r, 6) for r in residuals],
        "median_rankone_residual": float(np.nanmedian(residuals)),
        "wnorm_layer": round(float(wnorm[layer]), 4),
        "svd_sigma1_frac": sigma1_frac,
        "svd_sigma2_over_sigma1": sigma2_over_1,
        "cos_u1_w": cos_u1_w,
    }


# ── A3.9 — key-query gate, KEY x metric ablations (Blocker 5: the 4-key grid) ──
# The four key candidates (plan §3/§4 A3.9 + Blocker 5). ψ = identity with co-layer
# extraction (the plan default): ψ(t) = t_CB[L]; ψ(δ) = (c_C_trained - c_C_base) at
# the source = the FT context drift. The key VARIES; the query stays the context
# vector c_C' (denominator query = source c_C), so g(C=C')=1 holds per key.
A39_KEY_LABELS = ("c_C", "psi_t", "psi_delta", "c_C_plus_psi_delta")


def _a39_keys(sc, layer: int) -> dict[str, np.ndarray]:
    """Build the four A3.9 key vectors at the source context, layer L (Blocker 5)."""
    c_base = sc.tensors["c_C_base"].numpy().astype(np.float64)  # (C,L,d)
    c_trn = sc.tensors["c_C_trained"].numpy().astype(np.float64)  # (C,L,d)
    t_CB = sc.tensors["t_CB"].numpy().astype(np.float64)  # (L,d) trained source activation
    c_src = c_base[sc.source_idx, layer]  # (d,) the source context vector (key=c_C)
    psi_t = t_CB[layer]  # ψ(t) = co-layer t_CB
    psi_delta = c_trn[sc.source_idx, layer] - c_base[sc.source_idx, layer]  # ψ(δ)=FT ctx drift
    return {
        "c_C": c_src,
        "psi_t": psi_t,
        "psi_delta": psi_delta,
        "c_C_plus_psi_delta": c_src + psi_delta,
    }


def arm_a39(sc, layer: int, lam: float, sigma_c_layer: np.ndarray) -> dict:
    """Predicted gate g^pred(C') for each (KEY x metric) cell vs ĝ^real; rank/MAE/
    sign agreement; vs raw cosine. KEY ablation {c_C, ψ(t), ψ(δ), c_C+ψ(δ)} x metric
    ablation {I, diag(Σc+λI)⁻¹, (Σc+λI)⁻¹} (Blocker 5). Verdict (i) some key/metric
    > cosine; (ii) c_C key + Σc⁻¹ metric specifically wins (the boxed predictor)."""
    v_plus = sc.tensors["v_plus"]
    v0 = sc.tensors["v0"]
    c_base = sc.tensors["c_C_base"].numpy().astype(np.float64)  # (C,L,d)
    ghat, _ = gate_per_layer(v_plus, v0, sc.source_idx)
    ghat_l = ghat[:, layer]  # (C,) ground-truth realized gate
    c_src = c_base[sc.source_idx, layer]  # (d,) denominator query (q_src = source c_C)
    keys = _a39_keys(sc, layer)  # the four key candidates (Blocker 5)
    metrics = metric_ablation(sigma_c_layer, lam)  # {I, diag_Sigma_inv, Sigma_inv}
    n_ctx = c_base.shape[0]

    out: dict = {"key_metric_results": {}}
    # raw cosine baseline (key=c_C by definition for cosine)
    cos_pred = np.array(
        [raw_cosine_gate(c_src, c_base[ci, layer]) for ci in range(n_ctx)], dtype=np.float64
    )
    out["cosine_spearman"] = _spearman(cos_pred, ghat_l)
    out["cosine_pearson"] = _pearson(cos_pred, ghat_l)

    # Sweep KEY x metric (Blocker 5): the predicted gate uses key=k, query=c_C',
    # denominator query=q_src=source c_C — g(C=C')=1 holds per key by construction.
    for klabel in A39_KEY_LABELS:
        k = keys[klabel]
        for mkey in METRIC_KEYS:
            M = metrics[mkey]
            g_pred = np.array(
                [key_query_gate(k, c_base[ci, layer], c_src, M) for ci in range(n_ctx)],
                dtype=np.float64,
            )
            rho = _spearman(g_pred, ghat_l)
            pear = _pearson(g_pred, ghat_l)
            mae = float(np.nanmean(np.abs(g_pred - ghat_l)))
            sign = float(np.mean(np.sign(g_pred) == np.sign(ghat_l)))
            out["key_metric_results"][f"{klabel}::{mkey}"] = {
                "key": klabel,
                "metric": mkey,
                "spearman": rho,
                "pearson": pear,
                "mae": mae,
                "sign_agreement": sign,
                "beats_cosine": bool(
                    np.isfinite(rho)
                    and np.isfinite(out["cosine_spearman"])
                    and rho > out["cosine_spearman"]
                ),
            }
    # verdicts (separate, B1)
    cell_results = out["key_metric_results"]
    rhos = {kk: v["spearman"] for kk, v in cell_results.items()}
    best_cell = max(
        (kk for kk in cell_results if np.isfinite(rhos[kk])), key=lambda kk: rhos[kk], default=None
    )
    out["verdict_i_some_beats_cosine"] = any(v["beats_cosine"] for v in cell_results.values())
    # (ii): the BOXED predictor = key c_C + metric Sigma_inv specifically wins.
    out["verdict_ii_sigma_inv_wins"] = best_cell == "c_C::Sigma_inv"
    out["best_key_metric"] = best_cell
    return out


# ── A3.10 — base-gate validity + drift decomposition ──────────────────────────
def arm_a310(sc, layer: int, lam: float, sigma_c_layer: np.ndarray) -> dict:
    """g⁰ (base k/q, base metric M⁰) vs ĝ^real; g⁺ (trained k/q) diagnostic;
    key+query drift decomposition at fixed base metric M⁰. ALSO computes the
    base-behavior prior E0(C',B') + install magnitude ‖ŵ‖ per context (Blocker 3):
    the cross-cell A3.10 partial (g0-vs-ghat with E0+‖ŵ‖ partialled out, C7/C10) is
    done in issue665_aggregate over the per-cell g0_E0.json this writes."""
    v_plus = sc.tensors["v_plus"]
    v0_t = sc.tensors["v0"]
    c_base = sc.tensors["c_C_base"].numpy().astype(np.float64)  # (C,L,d)
    c_trn = sc.tensors["c_C_trained"].numpy().astype(np.float64)  # (C,L,d)
    ghat, wnorm_by_layer = gate_per_layer(v_plus, v0_t, sc.source_idx)  # (C,L),(L,)
    ghat_l = ghat[:, layer]
    M0 = metric_ablation(sigma_c_layer, lam)["Sigma_inv"]  # base metric M⁰=(Σc+λI)⁻¹
    k0 = c_base[sc.source_idx, layer]  # base key
    kp = c_trn[sc.source_idx, layer]  # trained key
    n_ctx = c_base.shape[0]

    # g0: base key, base query (c_C_base), base metric
    g0 = np.array(
        [whitened_gate(k0, c_base[ci, layer], M=M0) for ci in range(n_ctx)], dtype=np.float64
    )
    # g+: trained key, trained query, base metric (oracle, M0 held base — test-plan default)
    gp = np.array(
        [key_query_gate(kp, c_trn[ci, layer], kp, M0) for ci in range(n_ctx)], dtype=np.float64
    )
    # drift decomposition: g(k+, q0, M0) and g(k0, q+, M0)
    g_kp_q0 = np.array(
        [key_query_gate(kp, c_base[ci, layer], kp, M0) for ci in range(n_ctx)], dtype=np.float64
    )
    g_k0_qp = np.array(
        [key_query_gate(k0, c_trn[ci, layer], k0, M0) for ci in range(n_ctx)], dtype=np.float64
    )

    # ── Blocker 3: base-behavior prior E0(C',B') + install magnitude ‖ŵ‖ ──
    # E0(C') = r_B'ᵀ v0(C') at the locked read layer (the base model's behavioral
    # propensity at the eval target; the #532/#541/#649 dominant null). The behavior
    # read direction r_B' is the store's r_plus (the implant write direction = ŵ at
    # source); v0(C') is the base activation. Per plan §6 "E0[ctx] = r_plus[L,:] @
    # v0[ctx,L,:]".
    r_plus = sc.tensors["r_plus"].numpy().astype(np.float64)  # (L,d)
    v0_l = v0_t.numpy().astype(np.float64)[:, layer]  # (C,d) base activation
    e0 = (v0_l @ r_plus[layer]).astype(np.float64)  # (C,) base prior per context
    # ‖ŵ‖ = install magnitude from the trained-minus-base delta in the read direction:
    # the source write norm ‖v_plus(C)-v0(C)‖ at this layer = wnorm from gate_per_layer
    # (the baseline subtracted is v0(C), per plan §6). One scalar per cell (constant
    # across contexts within a cell — a cross-cell covariate for the partial).
    wnorm = float(wnorm_by_layer[layer])

    return {
        "g0_spearman": _spearman(g0, ghat_l),
        "g0_pearson": _pearson(g0, ghat_l),
        "g0_mae": float(np.nanmean(np.abs(g0 - ghat_l))),
        "gplus_spearman": _spearman(gp, ghat_l),
        "gplus_pearson": _pearson(gp, ghat_l),
        "drift_key_spearman": _spearman(g_kp_q0, ghat_l),
        "drift_query_spearman": _spearman(g_k0_qp, ghat_l),
        "cos_key_drift": float(abs((k0 @ kp) / (np.linalg.norm(k0) * np.linalg.norm(kp) + 1e-30))),
        # per-context vectors for the cross-cell E0/‖ŵ‖ partial (Blocker 3c, aggregate)
        "g0_by_context": [round(float(x), 6) for x in g0],
        "ghat_real_by_context": [round(float(x), 6) for x in ghat_l],
        "E0_by_context": [round(float(x), 6) for x in e0],
        "wnorm": round(wnorm, 6),
        "note": "all verdicts at fixed base metric M0=(Sigma_c+lambda I)^-1; "
        "E0=r_plus[L]@v0(C'), wnorm=||v_plus(C)-v0(C)||[L] (Blocker 3 base-prior/install partial)",
    }


# ── A3.6 — base read-out valid post-FT (needs r_B + behavioral E) ─────────────
def arm_a36(sc, layer: int, lam: float) -> dict:
    """Base read-out r_{B'}ᵀ(v⁺-v0)(C') predicts the change. r_B here is the
    diff-of-means write direction ŵ at source (the store's r_plus), used as the
    behavior read-out proxy; the behavioral change uses the activation projection
    (the judge-DV E partial is folded in downstream by issue665_aggregate)."""
    v_plus = sc.tensors["v_plus"]
    v0 = sc.tensors["v0"]
    r_plus = sc.tensors["r_plus"].numpy().astype(np.float64)  # (L,d) read-out direction
    r_l = r_plus[layer]  # (d,)
    dv = (v_plus - v0).numpy().astype(np.float64)[:, layer]  # (C,d)
    v0_l = v0.numpy().astype(np.float64)[:, layer]  # (C,d)
    pred_change = dv @ r_l  # (C,) r·Δv  (predicted behavioral change)
    base_level = v0_l @ r_l  # (C,) base-behavior-prior proxy E0
    # partial-Spearman of (pred_change, ghat-proxy) controlling base level.
    ghat, _ = gate_per_layer(v_plus, v0, sc.source_idx)
    ghat_l = ghat[:, layer]
    return {
        "partial_spearman_change_vs_gate_ctrl_prior": _partial_spearman(
            ghat_l, pred_change, base_level
        ),
        "spearman_change_vs_gate": _spearman(pred_change, ghat_l),
        "spearman_baseprior_vs_gate": _spearman(base_level, ghat_l),
        "note": "behavioral E partial folded in by issue665_aggregate (judged_E join)",
    }


# ── A3.6a — context vector stable across FT ───────────────────────────────────
def arm_a36a(sc, layer: int, lam: float, sigma_c_layer: np.ndarray) -> dict:
    """cos(c0,c⁺), rel-norm, displacement ‖c⁺-c0‖²_W vs within-condition spread."""
    c0 = sc.tensors["c_C_base"].numpy().astype(np.float64)[:, layer]  # (C,d)
    cp = sc.tensors["c_C_trained"].numpy().astype(np.float64)[:, layer]  # (C,d)
    n_ctx = c0.shape[0]
    coss, relnorm, disp = [], [], []
    for ci in range(n_ctx):
        a, b = c0[ci], cp[ci]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        coss.append(float((a @ b) / (na * nb)) if na > 1e-30 and nb > 1e-30 else float("nan"))
        relnorm.append(float(nb / na) if na > 1e-30 else float("nan"))
        d = b - a
        disp.append(float(d @ d))  # W=I displacement
    # within-condition spread from the probe tensor (slice this layer first — plan §8)
    v0_probe = sc.tensors["v0_probe"]  # (C,P,L,d)
    spread = []
    for ci in range(n_ctx):
        probes = v0_probe[ci, :, layer].numpy().astype(np.float64)  # (P,d)
        mu = probes.mean(axis=0)
        s = float(np.mean(((probes - mu) ** 2).sum(axis=1)))
        spread.append(s)
    return {
        "median_cos_c0_cp": float(np.nanmedian(coss)),
        "median_relnorm": float(np.nanmedian(relnorm)),
        "median_displacement_W_I": float(np.nanmedian(disp)),
        "median_within_spread": float(np.nanmedian(spread)),
        "frac_displacement_below_spread": float(
            np.mean([d < s for d, s in zip(disp, spread, strict=True)])
        ),
    }


# ── A3.6b — context→profile map stable across FT ──────────────────────────────
def arm_a36b(sc, layer: int, lam: float) -> dict:
    """Base map M0 refit from c0→v0 (ridge, default λ); test M0·(c⁺-c0) predicts
    (v⁺-v0). Refit M⁺ as the operator companion + noise floor."""
    c0 = sc.tensors["c_C_base"].numpy().astype(np.float64)[:, layer]  # (C,d)
    cp = sc.tensors["c_C_trained"].numpy().astype(np.float64)[:, layer]
    v0 = sc.tensors["v0"].numpy().astype(np.float64)[:, layer]  # (C,d)
    v_plus = sc.tensors["v_plus"].numpy().astype(np.float64)[:, layer]
    d = c0.shape[1]
    # ridge fit M0: c0 -> v0 (closed-form, default lambda); X=(C,d), Y=(C,d)
    # M0 = (XᵀX + λI)⁻¹ XᵀY  -> (d,d) so that c·M0 ≈ v   (row-vector convention)
    XtX = c0.T @ c0 + lam * np.eye(d)
    M0 = np.linalg.solve(XtX, c0.T @ v0)  # (d,d)
    Mp = np.linalg.solve(cp.T @ cp + lam * np.eye(d), cp.T @ v_plus)  # refit M+ companion
    dc = cp - c0  # (C,d)
    pred_change = dc @ M0  # (C,d) M0·(c⁺-c0)
    actual_change = v_plus - v0  # (C,d)
    # flatten the per-context vectors → one scalar partial-rho (project onto r... )
    pred_scalar = (pred_change * actual_change).sum(axis=1)  # alignment per ctx
    actual_scalar = (actual_change * actual_change).sum(axis=1)
    rho = _spearman(pred_scalar, actual_scalar)
    op_rel_diff = float(np.linalg.norm(Mp - M0) / (np.linalg.norm(M0) + 1e-30))
    return {
        "spearman_predicted_vs_actual_change": rho,
        "operator_rel_diff_Mp_M0": op_rel_diff,
        "note": f"ridge lambda={lam} (default; #658 A3.5 lambda not recoverable, plan §12 item 15)",
    }


# ── A3.7 — source-write displacement ──────────────────────────────────────────
def arm_a37(sc, layer: int, lam: float) -> dict:
    """cos(ŵ, δ) positive-only (PRIMARY, B5) vs shuffled-δ null. δ=t_CB-v0(C).
    Contrastive δ^contra DEFERRED (store lacks t⁻ — plan §11/§12 item 11)."""
    v0 = sc.tensors["v0"].numpy().astype(np.float64)[:, layer]  # (C,d)
    t_CB = sc.tensors["t_CB"].numpy().astype(np.float64)  # (L,d) target activation
    r_plus = sc.tensors["r_plus"].numpy().astype(np.float64)  # (L,d) ŵ=Δv(C)
    w = r_plus[layer]  # ŵ at source
    delta = t_CB[layer] - v0[sc.source_idx]  # δ = t - v0(C)
    nw, nd = np.linalg.norm(w), np.linalg.norm(delta)
    cos_w_delta = float((w @ delta) / (nw * nd)) if nw > 1e-30 and nd > 1e-30 else float("nan")
    # shuffled-δ null: cos(ŵ, a DIFFERENT context's displacement) — proxy for a
    # different-behavior δ within this cell (rotate the v0 row).
    rng = np.random.default_rng(42)
    n_ctx = v0.shape[0]
    null_idx = [i for i in range(n_ctx) if i != sc.source_idx]
    shuffled = []
    for j in rng.choice(null_idx, size=min(20, len(null_idx)), replace=False):
        d_j = t_CB[layer] - v0[j]
        ndj = np.linalg.norm(d_j)
        if nw > 1e-30 and ndj > 1e-30:
            shuffled.append(float((w @ d_j) / (nw * ndj)))
    return {
        "cos_w_delta_positive_only": cos_w_delta,
        "shuffled_delta_null_mean": float(np.mean(shuffled)) if shuffled else float("nan"),
        "shuffled_delta_null_max": float(np.max(shuffled)) if shuffled else float("nan"),
        "beats_shuffled": bool(
            np.isfinite(cos_w_delta) and shuffled and cos_w_delta > np.max(shuffled)
        ),
        "contrastive_arm": "DEFERRED — store lacks t_minus (plan §11/§12 item 11)",
    }


# ── joint factorization ───────────────────────────────────────────────────────
def arm_joint(sc, layer: int, lam: float) -> dict:
    """Latent S_{ij}=r_{B'_j}ᵀΔv(C'_i) near rank-one. Single behavior per cell
    here (one r_B = r_plus), so report the rank-one residual of the ΔV-vs-ŵ
    projection across contexts (the cross-behavior factorization is aggregated
    downstream over cells of different behaviors)."""
    v_plus = sc.tensors["v_plus"].numpy().astype(np.float64)[:, layer]
    v0 = sc.tensors["v0"].numpy().astype(np.float64)[:, layer]
    r_plus = sc.tensors["r_plus"].numpy().astype(np.float64)[layer]
    dv = v_plus - v0  # (C,d)
    s = dv @ r_plus  # (C,) latent gate scale for this behavior
    # one-behavior cell → S is a vector; the SVD rank-one over the stacked ΔV is in a38.
    return {
        "latent_scale_by_context": [round(float(x), 6) for x in s],
        "latent_scale_std": float(np.std(s)),
        "note": "cross-behavior S_{ij} rank-one folded in by issue665_aggregate",
    }


# ── single-context arm ────────────────────────────────────────────────────────
def arm_single_ctx(sc, layer: int, lam: float) -> dict:
    """Per-prompt δ_x→δ_{x'} gate prediction (continuous DV) vs within-context
    probe-split noise floor. Uses v_plus_probe/v0_probe (slice layer first)."""
    v_plus_probe = sc.tensors["v_plus_probe"]  # (C,P,L,d)
    v0_probe = sc.tensors["v0_probe"]
    # slice this layer FIRST (plan §8 — never materialize all 28 layers)
    vpp = v_plus_probe[:, :, layer].numpy().astype(np.float64)  # (C,P,d)
    v0p = v0_probe[:, :, layer].numpy().astype(np.float64)
    dvp = vpp - v0p  # (C,P,d)
    w = dvp[sc.source_idx].mean(axis=0)  # ŵ at source (mean over probes)
    wn2 = float(w @ w)
    _n_ctx, n_probe, _ = dvp.shape
    # per-prompt gate g_x(C') = <w, Δv_x(C')> / <w,w>
    per_prompt_gate = np.einsum("cpd,d->cp", dvp, w) / (wn2 + 1e-30)  # (C,P)
    # within-context noise floor = probe-split half-mean spread of the gate
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_probe)
    h1, h2 = perm[: n_probe // 2], perm[n_probe // 2 :]
    g_h1 = per_prompt_gate[:, h1].mean(axis=1)
    g_h2 = per_prompt_gate[:, h2].mean(axis=1)
    floor = float(np.median(np.abs(g_h1 - g_h2)))
    cross_ctx_spread = float(np.std(per_prompt_gate.mean(axis=1)))
    return {
        "per_context_mean_gate": [round(float(x), 6) for x in per_prompt_gate.mean(axis=1)],
        "within_context_noise_floor": floor,
        "cross_context_spread": cross_ctx_spread,
        "signal_above_floor": bool(cross_ctx_spread > floor),
    }


ARM_FNS = {
    "a36": arm_a36,
    "a36a": arm_a36a,  # needs sigma_c_layer
    "a36b": arm_a36b,
    "a37": arm_a37,
    "a38": arm_a38,
    "a39": arm_a39,  # needs sigma_c_layer
    "a310": arm_a310,  # needs sigma_c_layer
    "joint": arm_joint,
    "single_ctx": arm_single_ctx,
}
NEEDS_SIGMA = {"a36a", "a39", "a310"}


def process_cell(cell: str, layers: list[int], lam: float, sigma_c: np.ndarray, smoke: bool):
    sc = load_cell(cell, verify_sha=True)  # #600 guard on the live load path
    try:
        for arm in ARMS:
            out_dir = C.EVAL_ROOT / arm
            out_dir.mkdir(parents=True, exist_ok=True)
            per_layer = {}
            for layer in layers:
                fn = ARM_FNS[arm]
                if arm in NEEDS_SIGMA:
                    sigma_c_layer = sigma_c[layer]
                    res = fn(sc, layer, lam, sigma_c_layer)
                else:
                    res = fn(sc, layer, lam)
                per_layer[str(layer)] = res
            rec = {**_meta_stamp(cell, lam, {"smoke": smoke}), "by_layer": per_layer}
            outp = out_dir / f"{cell}.json"
            with open(outp, "w") as f:
                json.dump(rec, f, indent=1)
            logger.info("[%s] %s -> %s", arm, cell, outp)
            # Blocker 3a: persist the A3.10 base-prior/install partial inputs at the
            # canonical per_cell path the aggregate reads (one entry per context, at
            # the primary read layer).
            if arm == "a310":
                _write_g0_e0(cell, per_layer, sc)
    finally:
        sc.free()


def _write_g0_e0(cell: str, per_layer: dict, sc) -> None:
    """Blocker 3a: write eval_results/issue_665/per_cell/<cell>/g0_E0.json — per
    context {g0, ghat_real, E0} at the cell's primary read layer + the cell-level
    wnorm (install magnitude). The aggregate joins these for the C7/C10 partial."""
    read_layer = str(C.read_layer_for_cell(cell))
    bl = per_layer.get(read_layer)
    if not bl or "g0_by_context" not in bl:
        return
    ctx_ids = list(sc.tensors["context_ids"])
    g0 = bl["g0_by_context"]
    ghat = bl["ghat_real_by_context"]
    e0 = bl["E0_by_context"]
    entries = []
    for i, cid in enumerate(ctx_ids):
        if i >= len(g0):
            break
        entries.append(
            {
                "context_id": cid,
                "family": C.family_of_context(cid),
                "g0": g0[i],
                "ghat_real": ghat[i],
                "E0": e0[i],
                "wnorm": bl["wnorm"],  # cell-level install magnitude (constant per cell)
            }
        )
    out_dir = C.EVAL_ROOT / "per_cell" / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / "g0_E0.json"
    with open(outp, "w") as f:
        json.dump(
            {
                "cell": cell,
                "behavior": C.behavior_for_cell(cell),
                "read_layer": int(read_layer),
                "wnorm": bl["wnorm"],
                "entries": entries,
                "git_commit": _git_commit(),
                "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            },
            f,
            indent=1,
        )
    logger.info("[a310 g0_E0] %s -> %s", cell, outp)


def main():
    ap = argparse.ArgumentParser(description="issue665 Phase 3 CPU gate arms")
    ap.add_argument("--scope", default="content", help="content|content+null|all|marker")
    ap.add_argument(
        "--cells", nargs="*", default=None, help="explicit cell slugs (overrides scope)"
    )
    ap.add_argument(
        "--layers", nargs="*", type=int, default=None, help="layers (default: per-cell)"
    )
    ap.add_argument("--lam", type=float, default=C.LAMBDA_DEFAULT)
    ap.add_argument("--smoke", action="store_true", help="tiny slice (5 ctx, 1 layer)")
    args = ap.parse_args()

    cells = args.cells if args.cells else C.select_cells(args.scope)
    sigma_c, sigma_meta = load_sigma_c()
    logger.info(
        "Sigma_c loaded: n=%s capture_layers=%s", sigma_meta["n"], sigma_meta["capture_layers"]
    )

    for cell in cells:
        layers = args.layers if args.layers else [C.read_layer_for_cell(cell)]
        if args.smoke and not args.layers:
            layers = [C.read_layer_for_cell(cell)]  # 1 layer in smoke
        process_cell(cell, layers, args.lam, sigma_c, args.smoke)


if __name__ == "__main__":
    main()
