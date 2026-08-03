"""Issue #1310 free-analysis round 2: PRINCIPLED cross-persona map similarity.

Round 1 (scripts/issue1310_xpersona_similarity.py) reported a two-sided
Procrustes "aligned cosine" that is the von Neumann SPECTRAL cosine — it only
compares singular-value SPECTRA, is rotation-invariant (its rotation null is
degenerate), and is near-1 quasi-mechanically under the shared GCV-ridge
shrinkage profile. This round replaces it with three principled reads, per
model at layer 19 on the SAME scene-aggregated cells (aggfit aggregation,
fold seed 0, dof cap 0.9), equality-gating the M2 diagonals against the
committed within-cells first.

Leg A — SHARED-VS-SPECIFIC DECOMPOSITION (headline). Held-out nesting lattice
under the SAME scenario-grouped 5-fold partition, per-persona R^2 (fold-test-
mean) + pooled:
  M0: ONE map on the pooled 1200 points, GLOBAL train-fold centering.
  M1: ONE shared map on PER-PERSONA-CENTERED data (each persona's X and Y
      centered by its OWN train-fold means; prediction adds back the target
      persona's train-fold Y mean) — isolates "maps identical up to offsets".
  M2: per-persona maps (= committed within-cells; equality-gated).
Rung deltas (M1-M0, M2-M1) with scenario-grouped 1000-draw bootstrap CIs, and
R^2(M1)/R^2(M2) fractions. Round-1 reparam recoveries are quoted (no recompute)
as the general-linear per-persona-coordinates rung between M1 and M2. Null:
scenario-shuffled pairings through the identical M1 path (5 draws).

Leg B — PREDICTION-SPACE SIMILARITY per ordered pair (12/model). Using
fold-f-trained per-persona maps, on fold-f TEST inputs of the TARGET persona,
mean cosine + pooled R^2 between M_source(x) and M_target(x) RESPONSES (map
output minus the fit's train-Y intercept). Data-distribution-weighted operator
similarity (Frobenius cosine weights all 3584 directions equally, incl.
never-visited ones). Null: SHUFFLE-FIT source maps (same ridge recipe on
scenario-permuted pairings), 5 draws.

Leg C — RE-NULL THE OPERATOR-SPACE STATS against SHUFFLE-FIT maps (spectrum-
matched, structure-free), 5 draws/model: (i) spectrum cosine observed vs
shuffle-fit (expected ~1 under the null -> demoted to descriptive); (ii) raw
Frobenius cosine vs shuffle-fit (a stricter null than random rotation); (iii)
principal-angle top-k (10, 50) overlap for the INPUT subspace (left singular
vectors, X-space — expected elevated because the shared scenario battery gives
a shared X covariance) and the OUTPUT subspace (right singular vectors,
Y-space — carries the directional evidence), observed vs shuffle-fit. Top-k
via torch.svd_lowrank (oversample +10) to bound cost.

Reuses scripts/issue1310_xpersona_similarity.py (load_persona_arrays,
transfer_cell, spectrum_cosine, constants) + the #825/#931 fit machinery.
Pure-CPU, 8-thread caps.

CLI:
  uv run python scripts/issue1310_xpersona_similarity_v2.py
      [--store-root <.../store_onpolicy>] [--models base,instruct]
      [--out-dir eval_results/issue_1310/xpersona_similarity/v2]
      [--fig-dir figures/issue_1310] [--null-draws 5] [--n-boot 1000]
      [--seed 0] [--summary-from-disk]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_fit_cells as fit825  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
import issue1310_common as c1310  # noqa: E402
import issue1310_xpersona_similarity as v1  # noqa: E402

SCRIPT = "scripts/issue1310_xpersona_similarity_v2.py"

FROZEN_LAYERS = v1.FROZEN_LAYERS
HEADLINE_LAYER = v1.HEADLINE_LAYER  # 19
PERSONAS = v1.PERSONAS  # Wren, HELIOS, Dana, Vex
MODEL_KINDS = v1.MODEL_KINDS  # base, instruct
DOF_CAP = v1.DOF_CAP  # 0.9
N_FOLDS = v1.N_FOLDS  # 5
FIT_SEED = v1.FIT_SEED  # 0
V1_REPARAM_DIR = Path("eval_results/issue_1310/xpersona_similarity")

# Cross-project calibration anchors for the data-paired activation-Procrustes
# aligned cosine (all computed via ma._procrustes_cosine_null, full-data).
PROCRUSTES_ANCHORS = {
    "issue825_base_vs_instruct": 0.6864,
    "issue1345_chat_vs_plain_instruct": 0.855,
    "issue1345_chat_vs_plain_base": 0.732,
    "issue1345_paired_story_vs_chat": 0.455,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store-root", type=Path, default=v1.DEFAULT_STORE_ROOT)
    ap.add_argument("--models", type=str, default=",".join(MODEL_KINDS))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_results/issue_1310/xpersona_similarity/v2"),
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1310"))
    ap.add_argument("--null-draws", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=c1310.N_BOOTSTRAP)  # 1000
    ap.add_argument("--seed", type=int, default=FIT_SEED)
    ap.add_argument(
        "--summary-from-disk",
        action="store_true",
        help="skip compute; assemble summary_v2.json + both figures from the "
        "per-model JSONs already in --out-dir (split-run pattern)",
    )
    ap.add_argument(
        "--activation-procrustes",
        action="store_true",
        help="addendum leg: write standalone activation_procrustes_<m>.json (data-paired "
        "activation-Procrustes aligned cosine + rotation null reused from operator_stats + "
        "a NEW shuffle-fit null) and fold means into summary_v2.json (schema_note bump)",
    )
    ap.add_argument(
        "--only-operator",
        action="store_true",
        help="recompute ONLY Leg C (operator_stats_nulled_<m>.json) and rewrite it; "
        "reuse the existing decomposition/pred_similarity JSONs (addendum: adds the "
        "data-paired activation-Procrustes-aligned cosine without re-running Legs A/B)",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Pooled-fold ridge for M0 / M1 (per-persona held-out preds + R^2).
# ---------------------------------------------------------------------------
def _pooled_fold_preds(arrays: dict, layer: int, *, centering: str, y_perm=None) -> dict:
    """Held-out per-persona predictions of a SHARED pooled map.

    centering='global': one pooled map, global (pooled-train) X standardization +
      Y intercept (M0). centering='per_persona': each persona's train X and Y
      centered by its OWN train-fold means before pooling; the shared map is fit
      on the pooled centered data; each persona's test prediction adds back its
      own train-fold Y mean (M1). Folds are the shared scenario partition (all
      personas carry the identical folds array). ``y_perm`` (per-persona dict of
      row permutations) applies a scenario-shuffled Y pairing for the null.
    """
    assert centering in ("global", "per_persona")
    folds = arrays[PERSONAS[0]]["folds"]
    preds = {p: np.zeros_like(arrays[p]["Y"][:, layer, :], dtype=np.float64) for p in PERSONAS}
    for k in range(N_FOLDS):
        tr_blocks_x, tr_blocks_y, te_blocks_x = [], [], []
        te_idx = {}
        xmu_p, ymu_p = {}, {}
        for p in PERSONAS:
            tr = arrays[p]["folds"] != k
            te = arrays[p]["folds"] == k
            xp = arrays[p]["X"][:, layer, :].astype(np.float64)
            yp = arrays[p]["Y"][:, layer, :].astype(np.float64)
            if y_perm is not None:
                yp = yp[y_perm[p]]
            if centering == "per_persona":
                xmu_p[p] = xp[tr].mean(0)
                ymu_p[p] = yp[tr].mean(0)
                tr_blocks_x.append(xp[tr] - xmu_p[p])
                tr_blocks_y.append(yp[tr] - ymu_p[p])
                te_blocks_x.append(xp[te] - xmu_p[p])
            else:
                tr_blocks_x.append(xp[tr])
                tr_blocks_y.append(yp[tr])
                te_blocks_x.append(xp[te])
            te_idx[p] = np.flatnonzero(te)
        tr_x = np.concatenate(tr_blocks_x, axis=0).astype(np.float32)
        tr_y = np.concatenate(tr_blocks_y, axis=0).astype(np.float32)
        te_x = np.concatenate(te_blocks_x, axis=0).astype(np.float32)
        cache = fit825._prep_fold(tr_x, te_x)
        pred_all = fit825._ridge_predict_cached(cache, tr_y)  # (n_te_all, D)
        off = 0
        for p in PERSONAS:
            m = len(te_idx[p])
            block = pred_all[off : off + m]
            if centering == "per_persona":
                block = block + ymu_p[p]
            preds[p][te_idx[p]] = block
            off += m
    # Per-persona R^2 (fold-test-mean reference == each persona's own fold mean).
    out = {"preds": preds, "r2": {}}
    for p in PERSONAS:
        ss_res = ss_tot = 0.0
        yl = arrays[p]["Y"][:, layer, :].astype(np.float64)
        yp = yl[y_perm[p]] if y_perm is not None else yl
        for k in range(N_FOLDS):
            te = arrays[p]["folds"] == k
            true = yp[te]
            mu = true.mean(0)
            ss_res += float(np.sum((true - preds[p][te]) ** 2))
            ss_tot += float(np.sum((true - mu) ** 2))
        out["r2"][p] = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return out


def _pooled_r2(preds: dict, arrays: dict, layer: int) -> float:
    """Pooled (all-persona) held-out R^2, global-pooled-mean reference."""
    ss_res = ss_tot = 0.0
    allt = np.concatenate([arrays[p]["Y"][:, layer, :].astype(np.float64) for p in PERSONAS])
    allp = np.concatenate([preds[p] for p in PERSONAS])
    mu = allt.mean(0)
    ss_res = float(np.sum((allt - allp) ** 2))
    ss_tot = float(np.sum((allt - mu) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def run_decomposition(model_kind: str, arrays: dict, args) -> dict:
    """Leg A: M0 <= M1 <= M2 lattice, per-persona + pooled, deltas + boot CIs +
    M1 shuffle null, with the M2 diagonal equality-gated vs committed."""
    layer = HEADLINE_LAYER
    m0 = _pooled_fold_preds(arrays, layer, centering="global")
    m1 = _pooled_fold_preds(arrays, layer, centering="per_persona")
    m2_preds, m2_r2 = {}, {}
    for p in PERSONAS:
        cell = v1.transfer_cell(arrays[p], arrays[p], layer)
        m2_preds[p] = cell["preds"]
        m2_r2[p] = cell["r2_foldmean"]

    committed_r2 = {}
    for p in PERSONAS:
        cp = v1.COMMITTED_DIR / f"cells_agg_{model_kind}_{p}.json"
        committed_r2[p] = float(json.loads(cp.read_text())["r2_per_layer_obs"][layer])
    gate = {"tolerance": 1e-6, "per_persona": {}, "worst_abs_delta": 0.0}
    for p in PERSONAS:
        d = abs(m2_r2[p] - committed_r2[p])
        gate["per_persona"][p] = {"mine": m2_r2[p], "committed": committed_r2[p], "abs_delta": d}
        gate["worst_abs_delta"] = max(gate["worst_abs_delta"], d)
    gate["passed"] = gate["worst_abs_delta"] <= gate["tolerance"]

    # Per-persona rung deltas with paired scenario-grouped bootstrap CIs.
    per_persona = {}
    for p in PERSONAS:
        yp = arrays[p]["Y"][:, layer, :].astype(np.float64)
        scen = arrays[p]["scen"]
        gb0 = fit931.group_bootstrap_r2(
            m0["preds"][p], yp, scen, n_boot=args.n_boot, seed=args.seed
        )
        dm = gb0["draws_matrix"]
        gb1 = fit931.group_bootstrap_r2(
            m1["preds"][p], yp, scen, n_boot=args.n_boot, seed=args.seed, draws_matrix=dm
        )
        gb2 = fit931.group_bootstrap_r2(
            m2_preds[p], yp, scen, n_boot=args.n_boot, seed=args.seed, draws_matrix=dm
        )
        d10 = gb1["draws"] - gb0["draws"]
        d21 = gb2["draws"] - gb1["draws"]
        per_persona[p] = {
            "r2_M0_foldmean": m0["r2"][p],
            "r2_M1_foldmean": m1["r2"][p],
            "r2_M2_foldmean": m2_r2[p],
            "r2_M0_boot": gb0["r2"],
            "r2_M1_boot": gb1["r2"],
            "r2_M2_boot": gb2["r2"],
            "delta_M1_minus_M0": gb1["r2"] - gb0["r2"],
            "delta_M1_minus_M0_ci": [
                float(np.nanquantile(d10, 0.025)),
                float(np.nanquantile(d10, 0.975)),
            ],
            "delta_M2_minus_M1": gb2["r2"] - gb1["r2"],
            "delta_M2_minus_M1_ci": [
                float(np.nanquantile(d21, 0.025)),
                float(np.nanquantile(d21, 0.975)),
            ],
            "frac_M1_over_M2": (m1["r2"][p] / m2_r2[p] if m2_r2[p] > 1e-9 else float("nan")),
        }

    # M1 null: scenario-shuffled per-persona Y pairing through the identical M1 path.
    rng = np.random.default_rng(args.seed + 101)
    null_r2 = {p: [] for p in PERSONAS}
    for _ in range(args.null_draws):
        perm = {p: rng.permutation(arrays[p]["X"].shape[0]) for p in PERSONAS}
        mn = _pooled_fold_preds(arrays, layer, centering="per_persona", y_perm=perm)
        for p in PERSONAS:
            null_r2[p].append(mn["r2"][p])

    return {
        "headline_layer": layer,
        "equality_gate": gate,
        "per_persona": per_persona,
        "pooled": {
            "r2_M0": _pooled_r2(m0["preds"], arrays, layer),
            "r2_M1": _pooled_r2(m1["preds"], arrays, layer),
            "r2_M2": _pooled_r2(m2_preds, arrays, layer),
        },
        "m1_shuffle_null": {
            p: {"draws": [float(v) for v in null_r2[p]], "mean": float(np.nanmean(null_r2[p]))}
            for p in PERSONAS
        },
        "_m0_preds": m0["preds"],
        "_m1_preds": m1["preds"],
        "_m2_preds": m2_preds,
    }


# ---------------------------------------------------------------------------
# Leg B — prediction-space (data-weighted) operator similarity.
# ---------------------------------------------------------------------------
def _map_response(x_src, y_src, x_eval, tr, te, y_perm=None) -> np.ndarray:
    """Held-out map RESPONSE (prediction minus the fit's train-Y intercept) of
    the source map at the eval (target-test) inputs, for one fold."""
    ytr = y_src[tr] if y_perm is None else y_src[y_perm][tr]
    cache = fit825._prep_fold(x_src[tr].astype(np.float32), x_eval[te].astype(np.float32))
    pred = fit825._ridge_predict_cached(cache, ytr.astype(np.float32))
    return pred - ytr.mean(0)  # response = map output, intercept removed


def _pred_similarity_pair(src: dict, tgt: dict, layer: int, *, y_perm=None) -> dict:
    """Mean cosine + pooled R^2 between source-map and target-map RESPONSES on
    the TARGET's held-out test inputs (full coverage over target rows)."""
    xs, ys = src["X"][:, layer, :].astype(np.float64), src["Y"][:, layer, :].astype(np.float64)
    xt, yt = tgt["X"][:, layer, :].astype(np.float64), tgt["Y"][:, layer, :].astype(np.float64)
    f_s, f_t = src["folds"], tgt["folds"]
    n, d = xt.shape
    resp_s = np.zeros((n, d))
    resp_t = np.zeros((n, d))
    covered = np.zeros(n, dtype=bool)
    for k in range(N_FOLDS):
        tr_s = f_s != k
        te = f_t == k
        if te.sum() == 0 or tr_s.sum() < 3:
            continue
        resp_s[te] = _map_response(xs, ys, xt, tr_s, te, y_perm=y_perm)
        resp_t[te] = _map_response(xt, yt, xt, f_t != k, te)
        covered[te] = True
    rs, rt = resp_s[covered], resp_t[covered]
    num = (rs * rt).sum(1)
    den = np.linalg.norm(rs, axis=1) * np.linalg.norm(rt, axis=1) + 1e-12
    cos_mean = float((num / den).mean())
    mu = rt.mean(0)
    ss_res = float(np.sum((rt - rs) ** 2))
    ss_tot = float(np.sum((rt - mu) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {"cosine_mean": cos_mean, "r2_source_vs_target": r2, "n": int(covered.sum())}


def run_pred_similarity(model_kind: str, arrays: dict, args) -> dict:
    """Leg B: ordered-pair prediction-space similarity vs shuffle-fit-source null."""
    layer = HEADLINE_LAYER
    out = {}
    rng = np.random.default_rng(args.seed + 202)
    for s in PERSONAS:
        for t in PERSONAS:
            if s == t:
                continue
            obs = _pred_similarity_pair(arrays[s], arrays[t], layer)
            null_cos, null_r2 = [], []
            for _ in range(args.null_draws):
                perm = rng.permutation(arrays[s]["X"].shape[0])
                nd = _pred_similarity_pair(arrays[s], arrays[t], layer, y_perm=perm)
                null_cos.append(nd["cosine_mean"])
                null_r2.append(nd["r2_source_vs_target"])
            out[f"{s}->{t}"] = {
                "cosine_mean": obs["cosine_mean"],
                "r2_source_vs_target": obs["r2_source_vs_target"],
                "null_cosine_mean": float(np.nanmean(null_cos)),
                "null_cosine_p975": float(np.nanquantile(null_cos, 0.975)),
                "null_r2_mean": float(np.nanmean(null_r2)),
                "null_r2_p975": float(np.nanquantile(null_r2, 0.975)),
                "cosine_over_null": obs["cosine_mean"] - float(np.nanmean(null_cos)),
                "n": obs["n"],
            }
    return {"headline_layer": layer, "ordered_pairs": out}


# ---------------------------------------------------------------------------
# Leg C — operator-space stats re-nulled against shuffle-fit maps.
# ---------------------------------------------------------------------------
def _principal_overlap(qa: torch.Tensor, qb: torch.Tensor, k: int) -> tuple[float, float]:
    """Mean/min cos of top-k principal angles between two orthonormal bases."""
    m = qa[:, :k].T @ qb[:, :k]
    cs = torch.linalg.svdvals(m).clamp(0.0, 1.0)
    return float(cs.mean()), float(cs.min())


def _subspaces(beta: torch.Tensor, q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-q input (left, X-space) and output (right, Y-space) singular subspaces
    via randomized SVD (oversampled)."""
    u, _s, vt = torch.svd_lowrank(beta, q=q, niter=4)
    return u, vt  # u:(D_in,q) input side; vt:(D_out,q) output side


def _op_stats(beta_a: torch.Tensor, beta_b: torch.Tensor, q: int) -> dict:
    """Spectrum cosine + raw Frobenius cosine + input/output principal angles."""
    ua, va = _subspaces(beta_a, q)
    ub, vb = _subspaces(beta_b, q)
    rec = {
        "spectrum_cosine": v1.spectrum_cosine(beta_a, beta_b),
        "raw_frobenius_cosine": float(
            (beta_a.reshape(-1) @ beta_b.reshape(-1)) / (beta_a.norm() * beta_b.norm() + 1e-12)
        ),
    }
    for k in (10, 50):
        im, imin = _principal_overlap(ua, ub, k)
        om, omin = _principal_overlap(va, vb, k)
        rec[f"input_subspace_k{k}"] = {"mean_cos": im, "min_cos": imin}
        rec[f"output_subspace_k{k}"] = {"mean_cos": om, "min_cos": omin}
    return rec


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


def _fit_beta(arrays: dict, persona: str, layer: int, y_perm=None) -> torch.Tensor:
    x = arrays[persona]["X"][:, layer, :]
    y = arrays[persona]["Y"][:, layer, :]
    if y_perm is not None:
        y = y[y_perm]
    beta, _lam = cm.fit_primal_beta(x, y)
    return beta.detach()


def _procrustes_aligned_cosine(arrays: dict, a: str, b: str, layer: int, *, n_draws, seed) -> dict:
    """DATA-PAIRED activation-Procrustes-aligned operator cosine (the project's
    headline aligned-cosine convention — ma._procrustes_cosine_null, NOT the
    spectrum cosine). Orthogonal input/output alignments are fit from the
    scenario-paired activations (personas are row-aligned by scenario), beta_b
    is rotated into beta_a's frame, then Frobenius cosine; reported with the
    canonical random-rotation null. Full-data, matching the cross-project
    anchors (#825 0.6864; #1345 0.855/0.732/0.455)."""
    xa = _t(arrays[a]["X"][:, layer, :])
    ya = _t(arrays[a]["Y"][:, layer, :])
    xb = _t(arrays[b]["X"][:, layer, :])
    yb = _t(arrays[b]["Y"][:, layer, :])
    # ma._procrustes_cosine_null(Xb, Xi, Yb, Yi): aligns beta_b -> beta_i frame.
    return ma._procrustes_cosine_null(xa, xb, ya, yb, n_draws=n_draws, seed=seed)


def _orth_align(a_ctr: torch.Tensor, b_ctr: torch.Tensor) -> torch.Tensor:
    """Orthogonal Procrustes rotation a_ctr -> b_ctr (both mean-centered).
    Replicates ma._procrustes_cosine_null's internal `_orth`."""
    m = a_ctr.T @ b_ctr
    u, _s, vh = torch.linalg.svd(m, full_matrices=False)
    return u @ vh


def _aligned_frobenius_cos(
    beta_src: torch.Tensor, r_in: torch.Tensor, r_out: torch.Tensor, beta_tgt: torch.Tensor
) -> float:
    """Frobenius cosine of the Procrustes-aligned source operator (R_in^T beta_src R_out)
    against the target operator. Orthogonal transforms preserve the Frobenius norm."""
    m_fit = (r_in.T @ beta_src @ r_out).reshape(-1)
    vt = beta_tgt.reshape(-1)
    return float((m_fit @ vt) / (m_fit.norm() * vt.norm() + 1e-12))


def activation_procrustes_shuffle_null(
    arrays: dict, a: str, b: str, layer: int, *, shuf_a, shuf_b
) -> dict:
    """Shuffle-fit null for the data-paired aligned cosine: fit R_in/R_out from the
    REAL scenario-paired activations, then apply them to SHUFFLE-FIT maps (context->
    dialogue structure broken) — a stricter null than random rotation. Returns the
    draw list + mean/p975. Reuses the same 5 shuffle-fit betas as Leg C's null."""
    xa = _t(arrays[a]["X"][:, layer, :])
    ya = _t(arrays[a]["Y"][:, layer, :])
    xb = _t(arrays[b]["X"][:, layer, :])
    yb = _t(arrays[b]["Y"][:, layer, :])
    # Same orientation as ma._procrustes_cosine_null(xa, xb, ya, yb): align a's map
    # into b's frame — R_in from (xa,xb), R_out from (ya,yb).
    r_in = _orth_align(xa - xa.mean(0), xb - xb.mean(0))
    r_out = _orth_align(ya - ya.mean(0), yb - yb.mean(0))
    draws = [_aligned_frobenius_cos(shuf_a[d], r_in, r_out, shuf_b[d]) for d in range(len(shuf_a))]
    arr = np.asarray(draws)
    return {
        "n_draws": int(len(draws)),
        "draws": [float(v) for v in draws],
        "null_mean": float(arr.mean()) if len(arr) else float("nan"),
        "null_std": float(arr.std()) if len(arr) else float("nan"),
        "null_p975": float(np.quantile(arr, 0.975)) if len(arr) else float("nan"),
    }


def run_operator_nulled(model_kind: str, arrays: dict, args) -> dict:
    """Leg C: operator-space stats observed vs SHUFFLE-FIT null (5 draws)."""
    cm.GCV_DOF_CAP = DOF_CAP
    cm.LAMBDA_SELECTION = "gcv"
    layer = HEADLINE_LAYER
    q = 60  # top-50 subspace + 10 oversample
    betas = {p: _fit_beta(arrays, p, layer) for p in PERSONAS}
    rng = np.random.default_rng(args.seed + 303)
    n = arrays[PERSONAS[0]]["X"].shape[0]
    # Shuffle-fit beta bank: independent scenario permutation per (persona, draw).
    shuf_betas = {
        p: [_fit_beta(arrays, p, layer, y_perm=rng.permutation(n)) for _ in range(args.null_draws)]
        for p in PERSONAS
    }
    pairs = {}
    for i in range(len(PERSONAS)):
        for j in range(i + 1, len(PERSONAS)):
            a, b = PERSONAS[i], PERSONAS[j]
            obs = _op_stats(betas[a], betas[b], q)
            null_draws = [
                _op_stats(shuf_betas[a][d], shuf_betas[b][d], q) for d in range(args.null_draws)
            ]

            def _agg(key, sub=None):
                vals = [(nd[key][sub] if sub else nd[key]) for nd in null_draws]  # noqa: B023
                return {"mean": float(np.mean(vals)), "p975": float(np.quantile(vals, 0.975))}

            rec = {"observed": obs, "shuffle_fit_null": {}}
            rec["shuffle_fit_null"]["spectrum_cosine"] = _agg("spectrum_cosine")
            rec["shuffle_fit_null"]["raw_frobenius_cosine"] = _agg("raw_frobenius_cosine")
            for k in (10, 50):
                rec["shuffle_fit_null"][f"input_subspace_k{k}"] = _agg(
                    f"input_subspace_k{k}", "mean_cos"
                )
                rec["shuffle_fit_null"][f"output_subspace_k{k}"] = _agg(
                    f"output_subspace_k{k}", "mean_cos"
                )
            # Addendum: data-paired activation-Procrustes aligned cosine (the
            # project headline convention; reported beside the cross-project
            # anchors). Its own null is the canonical random-rotation null.
            # Rotation null is ~1/d (measured ~4e-4); 3 draws suffice to confirm
            # the observed aligned cosine clears chance — the CROSS-PROJECT ANCHORS
            # are the real reference. Each draw is ~8s (3584^2 matmuls), so this is
            # kept small deliberately (the addendum is "cheap").
            rec["procrustes_aligned"] = _procrustes_aligned_cosine(
                arrays, a, b, layer, n_draws=3, seed=args.seed + 400 + i * 10 + j
            )
            pairs[f"{a}~{b}"] = rec
    return {
        "headline_layer": layer,
        "n_null_draws": int(args.null_draws),
        "subspace_note": (
            "input_subspace = left singular vectors of beta (X-space, expected "
            "elevated under the shuffle-fit null via the shared scenario X "
            "covariance); output_subspace = right singular vectors (Y-space, "
            "carries the directional evidence). spectrum_cosine is DESCRIPTIVE "
            "only — its shuffle-fit null is ~1 by the shared shrinkage profile."
        ),
        "procrustes_note": (
            "procrustes_aligned.observed_aligned_cosine = the DATA-PAIRED "
            "activation-Procrustes aligned operator cosine (ma._procrustes_cosine_null, "
            "the project headline convention; full-data, scenario-paired), reported "
            "beside the calibration anchors. Its rotation null (null_mean/null_p975) is "
            "the canonical random-rotation reference."
        ),
        "calibration_anchors": PROCRUSTES_ANCHORS,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _load_results(out_dir: Path, models: list[str]) -> dict:
    res = {}
    for m in models:
        res[m] = {
            "decomposition": json.loads((out_dir / f"decomposition_{m}.json").read_text()),
            "pred_similarity": json.loads((out_dir / f"pred_similarity_{m}.json").read_text()),
            "operator_stats_nulled": json.loads(
                (out_dir / f"operator_stats_nulled_{m}.json").read_text()
            ),
        }
        ap_path = out_dir / f"activation_procrustes_{m}.json"
        if ap_path.exists():
            res[m]["activation_procrustes"] = json.loads(ap_path.read_text())
    return res


def make_figures(res: dict, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("neurips")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    c_m0 = pp.paper_palette_role("control")
    c_m1 = pp.paper_palette_role("baseline")
    c_m2 = pp.paper_palette_role("primary")
    c_acc = pp.paper_palette_role("accent")

    # ---- Figure 1 (NEW): shared-vs-specific decomposition. ----
    fig, axes = plt.subplots(1, len(MODEL_KINDS), figsize=(11.0, 4.6), layout="constrained")
    if len(MODEL_KINDS) == 1:
        axes = [axes]
    for ax, m in zip(axes, MODEL_KINDS, strict=True):
        dec = res[m]["decomposition"]
        rep = json.loads((V1_REPARAM_DIR / f"reparam_{m}.json").read_text())["ordered_pairs"]
        x = np.arange(len(PERSONAS))
        w = 0.26
        m0 = [dec["per_persona"][p]["r2_M0_foldmean"] for p in PERSONAS]
        m1 = [dec["per_persona"][p]["r2_M1_foldmean"] for p in PERSONAS]
        m2 = [dec["per_persona"][p]["r2_M2_foldmean"] for p in PERSONAS]
        # bootstrap whiskers on M1/M2 (CI of the R^2 itself via delta CIs is not
        # symmetric; use the boot R^2 +/- from the paired draws' own quantiles)
        ax.bar(x - w, m0, w, color=c_m0, label="M0 one map, global offset")
        ax.bar(x, m1, w, color=c_m1, label="M1 shared map, per-persona offset")
        ax.bar(x + w, m2, w, color=c_m2, label="M2 per-persona maps (within)")
        # reparam rung = mean recovery INTO each persona (target), round 1
        for i, p in enumerate(PERSONAS):
            recs = [rep[k]["recovery_r2_foldmean"] for k in rep if k.endswith(f"->{p}")]
            if recs:
                ax.plot(
                    x[i] + w / 2,
                    float(np.mean(recs)),
                    "D",
                    color=c_acc,
                    ms=7,
                    label="reparam rung (round 1)" if i == 0 else None,
                    zorder=5,
                )
            nullv = dec["m1_shuffle_null"][p]["mean"]
            ax.plot(
                [x[i] - 1.4 * w, x[i] + 1.4 * w],
                [nullv, nullv],
                color="0.4",
                lw=1.0,
                ls=":",
                label="M1 shuffle null" if i == 0 else None,
            )
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(x, PERSONAS)
        ax.set_ylabel("held-out R² (fold-test-mean)")
        ax.set_title(f"{m}: shared-vs-specific decomposition (L{HEADLINE_LAYER})")
        if m == MODEL_KINDS[0]:
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Cross-persona map: how much structure is shared vs per-character")
    pp.savefig_paper(fig, "xpersona_decomposition", dir=str(args.fig_dir), formats=("png",))
    plt.close(fig)

    # ---- Figure 2 (REGENERATE): operator stats vs shuffle-fit null + reparam. ----
    fig, axes = plt.subplots(2, len(MODEL_KINDS), figsize=(11.0, 8.0), layout="constrained")
    for col, m in enumerate(MODEL_KINDS):
        ax = axes[0, col]
        op = res[m]["operator_stats_nulled"]["pairs"]
        labels = list(op.keys())
        xx = np.arange(len(labels))
        raw = [op[k]["observed"]["raw_frobenius_cosine"] for k in labels]
        raw_null = [op[k]["shuffle_fit_null"]["raw_frobenius_cosine"]["p975"] for k in labels]
        outk = [op[k]["observed"]["output_subspace_k50"]["mean_cos"] for k in labels]
        out_null = [op[k]["shuffle_fit_null"]["output_subspace_k50"]["p975"] for k in labels]
        spec = [op[k]["observed"]["spectrum_cosine"] for k in labels]
        proc = [op[k]["procrustes_aligned"]["observed_aligned_cosine"] for k in labels]
        proc_null = [op[k]["procrustes_aligned"]["null_p975"] for k in labels]
        bw = 0.27
        ax.bar(
            xx - bw, raw, bw, color=pp.paper_palette_role("baseline"), label="raw Frobenius cosine"
        )
        ax.bar(
            xx,
            proc,
            bw,
            color=pp.paper_palette_role("primary"),
            label="data-paired Procrustes aligned cosine",
        )
        ax.bar(
            xx + bw,
            outk,
            bw,
            color=pp.paper_palette_role("neutral"),
            label="output-subspace k=50 mean cos",
        )
        ax.plot(
            xx - bw,
            raw_null,
            "_",
            color="0.3",
            ms=11,
            mew=2,
            label="raw-cos shuffle-fit null p97.5",
        )
        ax.plot(
            xx, proc_null, "_", color="0.5", ms=11, mew=2, label="Procrustes rotation null p97.5"
        )
        ax.plot(
            xx + bw,
            out_null,
            "_",
            color=c_acc,
            ms=11,
            mew=2,
            label="output-subspace shuffle null p97.5",
        )
        ax.plot(xx, spec, "^", color="0.55", ms=5, label="spectrum cosine (descriptive; null≈1)")
        # Cross-project Procrustes anchors (dashed reference lines).
        for aname, aval, acol in (
            ("#825 base↔instruct 0.686", 0.6864, "0.35"),
            ("#1345 chat↔plain 0.732–0.855", 0.732, "0.6"),
            ("#1345 story↔chat 0.455", 0.455, "0.75"),
        ):
            ax.axhline(aval, color=acol, ls="--", lw=0.9, label=aname if col == 0 else None)
        ax.set_xticks(xx, labels, rotation=45, ha="right")
        ax.set_ylabel("operator similarity")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{m}: operator similarity vs shuffle-fit null + anchors (L{HEADLINE_LAYER})")
        if col == 0:
            ax.legend(fontsize=5.8, loc="center right", ncol=1)

    for col, m in enumerate(MODEL_KINDS):
        ax = axes[1, col]
        rep = json.loads((V1_REPARAM_DIR / f"reparam_{m}.json").read_text())["ordered_pairs"]
        labels = list(rep.keys())
        xx = np.arange(len(labels))
        recov = [rep[k]["recovery_r2_foldmean"] for k in labels]
        ceil = [rep[k]["target_ceiling_foldmean"] for k in labels]
        nullv = [rep[k]["null_recovery_r2"] for k in labels]
        ax.bar(xx, recov, 0.6, color=pp.paper_palette_role("primary"), label="reparam recovery R²")
        ax.plot(xx, ceil, "D", color=c_acc, ms=6, label="target within ceiling")
        ax.plot(xx, nullv, "_", color="0.3", ms=13, mew=2, label="matched-capacity null")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(xx, [k.replace("->", "→") for k in labels], rotation=90, fontsize=7)
        ax.set_ylabel("held-out R² (fold-test-mean)")
        ax.set_title(f"{m}: reparameterization (round 1, L{HEADLINE_LAYER})")
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Operator similarity (shuffle-fit-nulled) and reparameterization")
    pp.savefig_paper(fig, "xpersona_cosine_reparam", dir=str(args.fig_dir), formats=("png",))
    plt.close(fig)


def _build_summary(res: dict, models: list[str], gate_ok: bool, args) -> None:
    summary = {
        "metadata": c1310.metadata(SCRIPT, args.seed, 0),
        "models": models,
        "personas": PERSONAS,
        "headline_layer": HEADLINE_LAYER,
        "gcv_dof_cap": DOF_CAP,
        "equality_gate_all_pass": gate_ok,
        "per_model": {},
    }
    for m in models:
        dec = res[m]["decomposition"]
        ps = res[m]["pred_similarity"]["ordered_pairs"]
        op = res[m]["operator_stats_nulled"]["pairs"]
        pp_ = dec["per_persona"]
        summary["per_model"][m] = {
            "equality_gate_worst_abs_delta": dec["equality_gate"]["worst_abs_delta"],
            "pooled": dec["pooled"],
            "per_persona_lattice": {
                p: {
                    "M0": pp_[p]["r2_M0_foldmean"],
                    "M1": pp_[p]["r2_M1_foldmean"],
                    "M2": pp_[p]["r2_M2_foldmean"],
                    "M1_minus_M0": pp_[p]["delta_M1_minus_M0"],
                    "M1_minus_M0_ci": pp_[p]["delta_M1_minus_M0_ci"],
                    "M2_minus_M1": pp_[p]["delta_M2_minus_M1"],
                    "M2_minus_M1_ci": pp_[p]["delta_M2_minus_M1_ci"],
                    "frac_M1_over_M2": pp_[p]["frac_M1_over_M2"],
                    "M1_shuffle_null": dec["m1_shuffle_null"][p]["mean"],
                }
                for p in PERSONAS
            },
            "pred_similarity_mean_cosine": float(np.mean([ps[k]["cosine_mean"] for k in ps])),
            "pred_similarity_null_cosine_mean": float(
                np.mean([ps[k]["null_cosine_mean"] for k in ps])
            ),
            "pred_similarity_mean_r2_source_vs_target": float(
                np.mean([ps[k]["r2_source_vs_target"] for k in ps])
            ),
            "operator_raw_cosine_mean": float(
                np.mean([op[k]["observed"]["raw_frobenius_cosine"] for k in op])
            ),
            "operator_raw_cosine_shuffle_null_mean": float(
                np.mean([op[k]["shuffle_fit_null"]["raw_frobenius_cosine"]["mean"] for k in op])
            ),
            "operator_spectrum_cosine_mean": float(
                np.mean([op[k]["observed"]["spectrum_cosine"] for k in op])
            ),
            "operator_spectrum_cosine_shuffle_null_mean": float(
                np.mean([op[k]["shuffle_fit_null"]["spectrum_cosine"]["mean"] for k in op])
            ),
            "operator_input_k50_obs_mean": float(
                np.mean([op[k]["observed"]["input_subspace_k50"]["mean_cos"] for k in op])
            ),
            "operator_input_k50_null_mean": float(
                np.mean([op[k]["shuffle_fit_null"]["input_subspace_k50"]["mean"] for k in op])
            ),
            "operator_output_k50_obs_mean": float(
                np.mean([op[k]["observed"]["output_subspace_k50"]["mean_cos"] for k in op])
            ),
            "operator_output_k50_null_mean": float(
                np.mean([op[k]["shuffle_fit_null"]["output_subspace_k50"]["mean"] for k in op])
            ),
            "procrustes_aligned_cosine_mean": float(
                np.mean([op[k]["procrustes_aligned"]["observed_aligned_cosine"] for k in op])
            ),
            "procrustes_aligned_cosine_range": [
                float(min(op[k]["procrustes_aligned"]["observed_aligned_cosine"] for k in op)),
                float(max(op[k]["procrustes_aligned"]["observed_aligned_cosine"] for k in op)),
            ],
            "procrustes_rotation_null_p975_mean": float(
                np.mean([op[k]["procrustes_aligned"]["null_p975"] for k in op])
            ),
            "procrustes_calibration_anchors": PROCRUSTES_ANCHORS,
        }
        # Addendum leg: fold the standalone activation-Procrustes means (incl. the
        # shuffle-fit null) when the standalone file is present.
        apr = res[m].get("activation_procrustes")
        if apr is not None:
            aps = apr["pairs"]
            vals = [aps[k]["observed_aligned_cosine"] for k in aps]
            summary["per_model"][m]["activation_procrustes_aligned_cosine_mean"] = float(
                np.mean(vals)
            )
            summary["per_model"][m]["activation_procrustes_aligned_cosine_range"] = [
                float(min(vals)),
                float(max(vals)),
            ]
            summary["per_model"][m]["activation_procrustes_rotation_null_mean"] = float(
                np.mean([aps[k]["rotation_null"]["null_mean"] for k in aps])
            )
            summary["per_model"][m]["activation_procrustes_shuffle_fit_null_mean"] = float(
                np.mean([aps[k]["shuffle_fit_null"]["null_mean"] for k in aps])
            )
    if any("activation_procrustes" in res[m] for m in models):
        summary["schema_note"] = (
            "v2 + activation-Procrustes addendum: per_model now carries "
            "activation_procrustes_* (data-paired aligned cosine mean/range + rotation "
            "and shuffle-fit null means; full detail in activation_procrustes_<m>.json). "
            "All prior keys preserved."
        )
    c1310.write_json(args.out_dir / "summary_v2.json", summary)


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    torch.set_num_threads(8)
    fit825.GCV_DOF_CAP = DOF_CAP
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        assert m in MODEL_KINDS, f"unknown model {m!r}"
    t0 = time.time()

    if args.summary_from_disk:
        res = _load_results(args.out_dir, MODEL_KINDS)
        gate_ok = all(res[m]["decomposition"]["equality_gate"]["passed"] for m in MODEL_KINDS)
        _build_summary(res, MODEL_KINDS, gate_ok, args)
        if gate_ok:
            make_figures(res, args)
        print(f"[xpersona-v2] summary+figures from disk; gate_all_pass={gate_ok}")
        return 0 if gate_ok else 1

    if args.activation_procrustes:
        # Addendum leg: standalone activation_procrustes_<m>.json. Reuse the
        # observed aligned cosine + raw cosine + rotation null already in
        # operator_stats_nulled_<m>.json (procrustes_aligned block, commit
        # 7e367f9a5a) and ADD a NEW shuffle-fit null (R_in/R_out from the real
        # paired activations applied to 5 shuffle-fit maps). Fold means into
        # summary_v2.json with a schema_note bump.
        cm.GCV_DOF_CAP = DOF_CAP
        cm.LAMBDA_SELECTION = "gcv"
        layer = HEADLINE_LAYER
        for m in models:
            print(f"[xpersona-v2] {m}: --activation-procrustes (standalone + shuffle-fit null)")
            op_path = args.out_dir / f"operator_stats_nulled_{m}.json"
            existing = json.loads(op_path.read_text())
            arrays = v1.load_persona_arrays(args.store_root, m)
            n = arrays[PERSONAS[0]]["X"].shape[0]
            # 5 shuffle-fit betas per persona (same recipe/dof as Leg C; fresh seed).
            rng = np.random.default_rng(args.seed + 505)
            shuf_betas = {
                p: [
                    _fit_beta(arrays, p, layer, y_perm=rng.permutation(n))
                    for _ in range(args.null_draws)
                ]
                for p in PERSONAS
            }
            pairs = {}
            for i in range(len(PERSONAS)):
                for j in range(i + 1, len(PERSONAS)):
                    a, b = PERSONAS[i], PERSONAS[j]
                    proc = existing["pairs"][f"{a}~{b}"]["procrustes_aligned"]
                    shuf_null = activation_procrustes_shuffle_null(
                        arrays, a, b, layer, shuf_a=shuf_betas[a], shuf_b=shuf_betas[b]
                    )
                    obs = proc["observed_aligned_cosine"]
                    pairs[f"{a}~{b}"] = {
                        "observed_aligned_cosine": obs,
                        "raw_vec_cosine": proc["raw_vec_cosine"],
                        "rotation_null": {
                            "n_draws": proc["n_draws"],
                            "null_mean": proc["null_mean"],
                            "null_p975": proc["null_p975"],
                        },
                        "shuffle_fit_null": shuf_null,
                        "aligned_over_rotation_null": obs - proc["null_mean"],
                        "aligned_over_shuffle_null": obs - shuf_null["null_mean"],
                    }
                    print(
                        f"[xpersona-v2]   {m} {a}~{b}: aligned={obs:.3f} "
                        f"rot_null={proc['null_mean']:.2e} shuf_null={shuf_null['null_mean']:.3f} "
                        f"({time.time() - t0:.0f}s)"
                    )
            c1310.write_json(
                args.out_dir / f"activation_procrustes_{m}.json",
                {
                    "metadata": c1310.metadata(SCRIPT, args.seed, 0),
                    "model_kind": m,
                    "headline_layer": layer,
                    "gcv_dof_cap": DOF_CAP,
                    "convention": (
                        "DATA-PAIRED activation-Procrustes aligned operator cosine "
                        "(ma._procrustes_cosine_null, the project headline convention; "
                        "full-data, scenario-paired). Full-data (NOT held-out train-fold) "
                        "for cross-project comparability to the calibration anchors, which "
                        "are all full-data. observed_aligned_cosine + raw_vec_cosine + "
                        "rotation_null reused from operator_stats_nulled_<m>.json; "
                        "shuffle_fit_null is the stricter structure-free null (same 5 "
                        "shuffle-fit maps as Leg C, same R_in/R_out from real activations)."
                    ),
                    "calibration_anchors": PROCRUSTES_ANCHORS,
                    "n_shuffle_fit_draws": int(args.null_draws),
                    "pairs": pairs,
                },
            )
            del arrays, shuf_betas
            gc.collect()
            print(f"[xpersona-v2] {m}: activation_procrustes written ({time.time() - t0:.1f}s)")
        print("[xpersona-v2] --activation-procrustes done; run --summary-from-disk to fold means")
        return 0

    if args.only_operator:
        # Addendum: PATCH the existing operator_stats_nulled_<m>.json with ONLY
        # the data-paired activation-Procrustes cosine per pair (+ anchors),
        # reusing the already-committed betas/shuffle-fit/principal-angle blocks
        # byte-for-byte. Avoids the full Leg-C recompute (each _procrustes call
        # is ~45-60s of 3584^2 matmuls; a full recompute overruns the budget).
        cm.GCV_DOF_CAP = DOF_CAP
        cm.LAMBDA_SELECTION = "gcv"
        layer = HEADLINE_LAYER
        for m in models:
            print(f"[xpersona-v2] {m}: --only-operator (patch Procrustes into Leg C)")
            op_path = args.out_dir / f"operator_stats_nulled_{m}.json"
            existing = json.loads(op_path.read_text())
            arrays = v1.load_persona_arrays(args.store_root, m)
            for i in range(len(PERSONAS)):
                for j in range(i + 1, len(PERSONAS)):
                    a, b = PERSONAS[i], PERSONAS[j]
                    existing["pairs"][f"{a}~{b}"]["procrustes_aligned"] = (
                        _procrustes_aligned_cosine(
                            arrays, a, b, layer, n_draws=3, seed=args.seed + 400 + i * 10 + j
                        )
                    )
                    print(
                        f"[xpersona-v2]   {m} {a}~{b}: procrustes aligned="
                        f"{existing['pairs'][f'{a}~{b}']['procrustes_aligned']['observed_aligned_cosine']:.3f}"
                        f" ({time.time() - t0:.0f}s)"
                    )
            existing["procrustes_note"] = (
                "procrustes_aligned.observed_aligned_cosine = the DATA-PAIRED "
                "activation-Procrustes aligned operator cosine (ma._procrustes_cosine_null, "
                "the project headline convention; full-data, scenario-paired), reported "
                "beside the calibration anchors. Its rotation null is the canonical "
                "random-rotation reference (~1/d, 3 draws)."
            )
            existing["calibration_anchors"] = PROCRUSTES_ANCHORS
            c1310.write_json(op_path, existing)
            del arrays
            gc.collect()
            print(f"[xpersona-v2] {m}: Procrustes patched into Leg C ({time.time() - t0:.1f}s)")
        print("[xpersona-v2] --only-operator done; run --summary-from-disk to reassemble")
        return 0

    print(f"[phase=xpersona_v2] models={models} dof_cap={DOF_CAP} store_root={args.store_root}")
    for m in models:
        print(f"[xpersona-v2] {m}: loading aggregated arrays")
        arrays = v1.load_persona_arrays(args.store_root, m)
        print(f"[xpersona-v2] {m}: Leg A decomposition (M0<=M1<=M2)")
        dec = run_decomposition(m, arrays, args)
        g = dec["equality_gate"]
        print(
            f"[xpersona-v2] {m}: M2 equality gate worst|d|={g['worst_abs_delta']:.2e} "
            f"{'PASS' if g['passed'] else 'FAIL'}"
        )
        dec_public = {k: v for k, v in dec.items() if not k.startswith("_")}
        c1310.write_json(
            args.out_dir / f"decomposition_{m}.json",
            {"metadata": c1310.metadata(SCRIPT, args.seed, 0), "model_kind": m, **dec_public},
        )
        if not g["passed"]:
            print(f"[xpersona-v2] {m}: EQUALITY GATE FAILED — stopping")
            return 1
        print(f"[xpersona-v2] {m}: Leg B prediction-space similarity")
        ps = run_pred_similarity(m, arrays, args)
        c1310.write_json(
            args.out_dir / f"pred_similarity_{m}.json",
            {"metadata": c1310.metadata(SCRIPT, args.seed, 0), "model_kind": m, **ps},
        )
        print(f"[xpersona-v2] {m}: Leg C operator stats vs shuffle-fit null")
        op = run_operator_nulled(m, arrays, args)
        c1310.write_json(
            args.out_dir / f"operator_stats_nulled_{m}.json",
            {"metadata": c1310.metadata(SCRIPT, args.seed, 0), "model_kind": m, **op},
        )
        del arrays
        gc.collect()
        print(f"[xpersona-v2] {m}: done ({time.time() - t0:.1f}s elapsed)")

    # If both models computed in one process, assemble now; else use --summary-from-disk.
    if set(models) == set(MODEL_KINDS):
        res = _load_results(args.out_dir, MODEL_KINDS)
        gate_ok = all(res[m]["decomposition"]["equality_gate"]["passed"] for m in MODEL_KINDS)
        _build_summary(res, MODEL_KINDS, gate_ok, args)
        make_figures(res, args)
        print(f"[xpersona-v2] summary+figures written; total {time.time() - t0:.1f}s")
    else:
        print("[xpersona-v2] single-model leg done; run --summary-from-disk to assemble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
