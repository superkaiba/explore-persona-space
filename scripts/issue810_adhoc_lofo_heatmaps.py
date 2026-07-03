# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ², r_B, c_C) in scientific docstrings + log messages.
"""Ad-hoc (chat-requested, 0-GPU free analysis): #810's 3-panel layer×answer-position
heatmap, recomputed with LEAVE-ONE-FAMILY-OUT (7-fold) cross-validation instead of
leave-one-context-out.

Reproduces figures/issue_810/adhoc_layer_x_position_heatmaps.png EXACTLY (same 3
stacked panels, same 36-column x-order, same 28-layer y, same cmaps/scales/vlines),
but every fitted panel uses 7-fold leave-one-FAMILY-out CV (fold = one of the 7
battery families held out, train on the other 6). Panel 1 (fixed r_B) has NO fitted
parameters, so LOFO is identity for it — it is fold-invariant and reproduced for
visual consistency.

NOT a workflow-surface change; a one-off analysis over EXISTING artifacts. Reuses
#810's loaders (v0_summaries, c_C, per-position store, r_B, graded E0) verbatim and
the on-main ridge primitives (robust_pca_basis, skill_over_mean_r2,
_press_loo_mse_per_lambda, _ridge_dual_weights). The only NEW code is the
group-7-fold outer split (the on-main ridge_predict_loco_centered is leave-ONE-out).
"""

from __future__ import annotations

import json
import logging

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib
import sys
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

# Constants inlined from the #810-worktree issue810_common.py (NOT on main —
# this ad-hoc script is self-contained so it commits + runs from repo-root main).
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue658_theory_assumptions"
I658_V0_SUMMARIES = f"{HF_PREFIX}/store/v0_summaries.pt"
I658_RB = f"{HF_PREFIX}/store/r_b.pt"
I658_STORE_MANIFEST = f"{HF_PREFIX}/store/store_manifest.json"
I594_CC_LAST_FILE = "issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
I594_PROBE_POOL_HASH = "ad687becec266286549aaaa1af3b35e246d593e012e233564e58ff75fb015dd7"
PCA_TARGET_DIM_CAP = 48


def context_ids_from_manifest(manifest: dict) -> list[str]:
    """The 50 store context_ids, order-stable (inlined from issue810_common)."""
    ids = manifest.get("context_ids")
    if not ids or len(set(ids)) != len(ids):
        raise RuntimeError(f"store_manifest context_ids missing/duplicated: {ids!r}")
    return list(ids)


def _load_json(path):
    """Context-managed JSON read (local path or hf_hub_download result)."""
    with open(path) as f:
        return json.load(f)


# on-main ridge primitives (self-contained; no stranded-script import)
import issue658_fit_predictors as fp  # noqa: E402  (scripts/ is on sys.path above)

logger = logging.getLogger("issue810_adhoc_lofo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
POSITION_STORE_HF = f"{HF_PREFIX}/answer_position_sweep"
E0_HIGHM = PROJECT_ROOT / "eval_results" / "issue_810" / "phase_c" / "e0_highm_graded.json"
LOCO_RECON = PROJECT_ROOT / "eval_results" / "issue_810" / "reconstruction_skill_by_summary.json"
BATTERY = PROJECT_ROOT / "data" / "issue594" / "battery.json"
OUT_FIG = PROJECT_ROOT / "figures" / "issue_810" / "adhoc_layer_x_position_heatmaps_LOFO.png"
OUT_JSON = PROJECT_ROOT / "eval_results" / "issue_810" / "adhoc_lofo_heatmap_grids.json"

HIGH_M_BEHAVIORS = ("sycophancy", "refusal", "harmful_compliance")

# x-axis column order — matches the reference figure EXACTLY:
# head_0..head_15 (content start), tail_16..tail_1 (content end), im_end, turn_nl
# (boundary), mean, maxp (aggregate). 36 columns.
COL_ORDER = (
    [f"head_{k}" for k in range(16)]
    + [f"tail_{k}" for k in range(16, 0, -1)]
    + ["im_end", "turn_nl", "mean", "maxp"]
)


# short labels: head_k -> "h k", tail_k -> "t k", others verbatim
def _col_label(s: str) -> str:
    if s.startswith("head_"):
        return f"h {s.split('_')[1]}"
    if s.startswith("tail_"):
        return f"t {s.split('_')[1]}"
    return s


COL_LABELS = [_col_label(s) for s in COL_ORDER]
# separator vlines (between columns): head|tail at 15.5, content|boundary at 31.5,
# boundary|aggregate at 33.5.
VLINES = [15.5, 31.5, 33.5]


# ── inputs ──────────────────────────────────────────────────────────────────


def _load_free_summaries():
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, I658_V0_SUMMARIES, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    return blob["summaries"], blob["capture_layers"]


def _load_cc(ctx_ids, capture_layers):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, I594_CC_LAST_FILE, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    pph = blob.get("probe_pool_hash")
    if pph != I594_PROBE_POOL_HASH:
        raise RuntimeError(f"#594 c_C probe_pool_hash drift: {pph} != {I594_PROBE_POOL_HASH}")
    tensor = blob["tensor"]  # (n_ctx, 28, H)
    iid_to_row = {iid: i for i, iid in enumerate(blob["instance_ids"])}
    missing = [c for c in ctx_ids if c not in iid_to_row]
    if missing:
        raise RuntimeError(f"c_C store missing {len(missing)} contexts: {missing[:5]}")
    return {c: tensor[iid_to_row[c]][capture_layers].float().numpy() for c in ctx_ids}


def _load_position_summaries(ctx_ids, store_hf: str = POSITION_STORE_HF):
    """{ctx_id: {position: (Lc,H) fp32}} + {ctx_id: coverage} streamed per-context.

    Reads one <ctx>.pt at a time from HF (~7 MB each; peak footprint = one context).
    ``store_hf`` parametrizes the store prefix (default = the round-1 store; the
    uh round passes answer_position_sweep_user_header — plan v11 §4.6 item 6).
    """
    from huggingface_hub import hf_hub_download

    out, cov = {}, {}
    for i, c in enumerate(ctx_ids):
        path = hf_hub_download(HF_DATA_REPO, f"{store_hf}/{c}.pt", repo_type="dataset")
        blob = torch.load(path, weights_only=False)
        names = blob["positions"]
        pv = blob["pos_vectors"].float().numpy()  # (n_pos, Lc, H)
        out[c] = {name: pv[j] for j, name in enumerate(names)}
        cov[c] = dict(blob["coverage"])
        if (i + 1) % 10 == 0:
            logger.info("[phase=load-pos] %d/%d contexts", i + 1, len(ctx_ids))
    return out, cov


def _load_rb():
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, I658_RB, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    return blob["r_b"]


def _load_family_map():
    b = _load_json(BATTERY)
    inst = b["instances"] if isinstance(b, dict) else b
    return {x["id"]: x["family"] for x in inst}


def _e0_graded_by_behavior():
    e = _load_json(E0_HIGHM)
    out = {}
    for behavior, blk in e["by_behavior"].items():
        out[behavior] = {k: v for k, v in blk["per_context_graded_mean"].items() if v is not None}
    return out


# ── summary matrix assembly (coverage-aware) ─────────────────────────────────


def _summary_row(summary, layer_i, ctx, free_summaries, pos_summaries, coverage):
    """One (H,) summary vector for a context at a layer; None if not covered."""
    if summary in ("mean", "last", "maxp"):
        return free_summaries[summary][ctx][layer_i].numpy()
    if coverage[ctx].get(summary, 0) <= 0:
        return None
    return pos_summaries[ctx][summary][layer_i]


def _kept_and_matrix(summary, layer_i, ctx_ids, free_summaries, pos_summaries, coverage):
    """(n_kept, H) matrix + kept ctx list for a (summary, layer) cell."""
    rows, kept = [], []
    for c in ctx_ids:
        r = _summary_row(summary, layer_i, c, free_summaries, pos_summaries, coverage)
        if r is not None:
            rows.append(r)
            kept.append(c)
    return (np.stack(rows) if rows else np.zeros((0, 1))), kept


# ── group-7-fold ridge (the NEW code: LOFO outer split) ──────────────────────


def _group_fold_ridge_predict(Xc: np.ndarray, Yv: np.ndarray, groups: list[str]) -> np.ndarray:
    """Leave-one-FAMILY-out ridge prediction of Yv from Xc, pooled over folds.

    For each unique family in ``groups``, hold out its rows, fit on the other
    families' rows, predict the held-out rows. Per fold: train-only X
    standardization (ddof=0, #658 convention), train-only target centering,
    #658's exact PRESS-LOO nested-CV λ pick over RIDGE_LAMBDAS on the TRAIN rows,
    dual-space weight solve. prediction = ȳ0_train + M̂(x_held). Returns the
    UN-centered (n, P) held-out predictions pooled across all folds (each row is
    predicted exactly once, from the fold that held its family out).

    This is the group-fold analogue of the on-main leave-ONE-out
    ``ridge_predict_loco_centered`` — the ONLY structural change is the outer
    split (family folds, not per-row LOO).
    """
    device = torch.device(fp.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Yv)).to(device=device, dtype=torch.float64)
    preds = np.zeros_like(Yv, dtype=np.float64)
    fams = sorted(set(groups))
    grp = np.array(groups)
    for fam in fams:
        test_idx = np.where(grp == fam)[0]
        train_idx = np.where(grp != fam)[0]
        if len(train_idx) < 3 or len(test_idx) == 0:
            # cannot fit / nothing to predict — leave preds as train-mean fallback
            if len(train_idx) >= 1 and len(test_idx) > 0:
                ymu = Yt[torch.tensor(train_idx, device=device)].mean(0)
                for ti in test_idx:
                    preds[ti] = ymu.detach().cpu().numpy()
            continue
        tr_t = torch.tensor(train_idx, device=device)
        Xtr, Ytr = Xt[tr_t], Yt[tr_t]
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9  # numpy ddof=0 convention (#658)
        Xtr_n = (Xtr - xmu) / xsd
        ymu = Ytr.mean(0)  # train predict-the-mean baseline
        Ytr_c = Ytr - ymu
        mse = fp._press_loo_mse_per_lambda(Xtr_n, Ytr_c, fp.RIDGE_LAMBDAS)
        best_lam = fp.RIDGE_LAMBDAS[int(torch.argmin(mse).item())]
        w = fp._ridge_dual_weights(Xtr_n, Ytr_c, best_lam)
        for ti in test_idx:
            x_held = (Xt[ti] - xmu) / xsd
            preds[ti] = (ymu + x_held @ w).detach().cpu().numpy()
    return preds


def _group_fold_train_mean_baseline(Y: np.ndarray, groups: list[str]) -> np.ndarray:
    """Per-fold train-mean baseline for a scalar/vector target under LOFO.

    Row i's baseline = mean of Y over all rows NOT in i's family — the LOFO
    analogue of loco_train_means. Used as the skill_over_mean_r2 baseline so the
    R² denominator is the honest held-out predict-the-mean error under the same
    fold structure.
    """
    grp = np.array(groups)
    fams = sorted(set(groups))
    base = np.zeros_like(Y, dtype=np.float64)
    for fam in fams:
        test_idx = np.where(grp == fam)[0]
        train_idx = np.where(grp != fam)[0]
        mu = Y[train_idx].mean(0) if len(train_idx) else Y.mean(0)
        for ti in test_idx:
            base[ti] = mu
    return base


# ── panel builders ───────────────────────────────────────────────────────────


def _rho(pred, meas):
    if len(pred) < 4 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return np.nan
    r, _ = spearmanr(pred, meas)
    return float(r) if np.isfinite(r) else np.nan


def build_panel3_reconstruction(
    ctx_ids, capture_layers, cc, free_summaries, pos_summaries, coverage, fam_map
):
    """(28, 36) LOFO reconstruction skill-over-mean R² per (layer, summary).

    Per cell: PCA-48 (min(48, n_train-2)) fit on TRAIN contexts ONLY per fold
    (cleaner than #810's all-data PCA — stated), group-fold ridge c_C→PCA-target,
    pool the 7 held-out folds' predictions, skill_over_mean_r2 with the per-fold
    LOFO train-mean baseline on the PCA target.
    """
    n_layers = len(capture_layers)
    grid = np.full((n_layers, len(COL_ORDER)), np.nan)
    for ci, summary in enumerate(COL_ORDER):
        for li in range(n_layers):
            Yv, kept = _kept_and_matrix(
                summary, li, ctx_ids, free_summaries, pos_summaries, coverage
            )
            if Yv.shape[0] < 8:  # need ≥ a few families to fold
                continue
            Xc = np.stack([cc[c][li] for c in kept])
            groups = [fam_map[c] for c in kept]
            if len(set(groups)) < 2:
                continue
            # PCA target dim capped by the SMALLEST train-fold size (n_kept minus
            # the largest single family), so every fold's PCA basis is well-posed.
            fam_counts = Counter(groups)
            min_train = len(kept) - max(fam_counts.values())
            pca_dim = min(PCA_TARGET_DIM_CAP, max(1, min_train - 2))
            # PCA basis is refit per fold INSIDE the group-fold ridge target path:
            # here we PCA-reduce Yv per fold. To keep it simple + faithful, refit
            # the target basis per fold: build the PCA target per fold on train,
            # project both train & held-out, ridge, pool.
            preds_pca, y_pca_all, base_pca = _recon_fold_predict(Xc, Yv, groups, pca_dim)
            r = skill_over_mean_r2_lofo(preds_pca, y_pca_all, base_pca)
            grid[li, ci] = r
        logger.info("[phase=panel3] %s done", summary)
    return grid


def _recon_fold_predict(Xc, Yv, groups, pca_dim):
    """Group-fold ridge on a PER-FOLD PCA-reduced target.

    Returns (pooled held-out preds in each fold's PCA basis is NOT comparable
    across folds) — so instead we score R² PER FOLD in that fold's train-PCA
    basis and variance-weight-aggregate. To keep skill_over_mean_r2 usable we
    return per-fold (pred, y, base) stacked with a fold index, and the caller
    aggregates. Simpler + faithful: compute skill_over_mean_r2 per fold on the
    held-out rows in that fold's train-PCA target, then variance-weight over
    folds by held-out target variance.
    """
    device = torch.device(fp.DEVICE)
    grp = np.array(groups)
    fams = sorted(set(groups))
    per_fold = []  # (pred (m,k), y (m,k), base (m,k))
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    for fam in fams:
        test_idx = np.where(grp == fam)[0]
        train_idx = np.where(grp != fam)[0]
        if len(train_idx) < 3 or len(test_idx) == 0:
            continue
        Ytr = Yv[train_idx]
        # PCA basis on TRAIN only (this fold)
        k = min(pca_dim, max(1, len(train_idx) - 2))
        mu_pca, comps = _gram_top_k_pca(Ytr, k)  # comps (k, H) — exact top-k, ~22x faster
        Ytr_pca = (Ytr - mu_pca) @ comps.T  # (m_tr, k)
        Yte_pca = (Yv[test_idx] - mu_pca) @ comps.T  # (m_te, k)
        # ridge on train, predict held-out — train-only X standardize + center
        Xtr = Xt[torch.tensor(train_idx, device=device)]
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - xmu) / xsd
        Ytr_pca_t = torch.from_numpy(np.ascontiguousarray(Ytr_pca)).to(
            device=device, dtype=torch.float64
        )
        ymu = Ytr_pca_t.mean(0)
        Ytr_c = Ytr_pca_t - ymu
        mse = fp._press_loo_mse_per_lambda(Xtr_n, Ytr_c, fp.RIDGE_LAMBDAS)
        best_lam = fp.RIDGE_LAMBDAS[int(torch.argmin(mse).item())]
        w = fp._ridge_dual_weights(Xtr_n, Ytr_c, best_lam)
        Xte_n = (Xt[torch.tensor(test_idx, device=device)] - xmu) / xsd
        pred = (ymu + Xte_n @ w).detach().cpu().numpy()  # (m_te, k)
        base = np.tile(ymu.detach().cpu().numpy(), (len(test_idx), 1))  # train-mean baseline
        per_fold.append((pred, Yte_pca, base))
    if not per_fold:
        return np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))
    preds = np.concatenate([p[0] for p in per_fold], axis=0)
    ys = np.concatenate([p[1] for p in per_fold], axis=0)
    bases = np.concatenate([p[2] for p in per_fold], axis=0)
    return preds, ys, bases


def skill_over_mean_r2_lofo(preds, y, base):
    """Variance-weighted held-out R² over the LOFO train-mean-centered target.

    skill = 1 − SS_res(pred) / SS_res(train-mean baseline), aggregated
    variance-weighted across output dims (mirrors skill_over_mean_r2 but with the
    supplied per-fold train-mean baseline rather than a global mean).
    """
    if preds.shape[0] < 2:
        return np.nan
    ss_res = ((y - preds) ** 2).sum(axis=0)  # (k,)
    ss_base = ((y - base) ** 2).sum(axis=0)  # (k,)
    num = ss_res.sum()
    den = ss_base.sum()
    if den < 1e-12:
        return np.nan
    return float(1.0 - num / den)


def build_panel_readout(
    ctx_ids, capture_layers, e0, rb, free_summaries, pos_summaries, coverage, fam_map, method
):
    """(28, 36) behavior-averaged ρ per (layer, summary) for a read-out method.

    method='fixed_rb': ρ(r_Bᵀ·summary, graded E0) over ALL kept contexts
      (fold-invariant — no fitted params). behavior-averaged over the 3 high-m.
    method='trained_ridge': group-7-fold ridge summary(PCA)→graded E0, pooled
      held-out predictions, ρ vs graded E0. behavior-averaged.
    """
    n_layers = len(capture_layers)
    # accumulate per-behavior ρ then average (ignoring nan)
    acc = np.full((len(HIGH_M_BEHAVIORS), n_layers, len(COL_ORDER)), np.nan)
    for bi, behavior in enumerate(HIGH_M_BEHAVIORS):
        graded = e0.get(behavior, {})
        if len(graded) < 4:
            continue
        for ci, summary in enumerate(COL_ORDER):
            for li in range(n_layers):
                Xmat, kept = _kept_and_matrix(
                    summary, li, ctx_ids, free_summaries, pos_summaries, coverage
                )
                kept_g = [c for c in kept if c in graded]
                if len(kept_g) < 4:
                    continue
                idx = [kept.index(c) for c in kept_g]
                X = Xmat[idx]  # (n, H)
                y = np.array([graded[c] for c in kept_g], dtype=np.float64)
                if method == "fixed_rb":
                    r = rb[behavior]["diffmeans"][li].numpy()  # (H,)
                    pred = X @ r
                else:  # trained_ridge, group-7-fold
                    groups = [fam_map[c] for c in kept_g]
                    if len(set(groups)) < 2:
                        continue
                    fam_counts = Counter(groups)
                    min_train = len(kept_g) - max(fam_counts.values())
                    k = min(PCA_TARGET_DIM_CAP, max(1, min_train - 2))
                    Xp = _pca_reduce(X, k)
                    pred = _group_fold_ridge_predict(Xp, y.reshape(-1, 1), groups)[:, 0]
                acc[bi, li, ci] = _rho(pred, y)
        logger.info("[phase=readout-%s] %s done", method, behavior)
    # behavior-average, ignoring nan
    with np.errstate(invalid="ignore"):
        grid = np.nanmean(acc, axis=0)
    return grid


def _pca_reduce(X, k):
    # Fast Gram-space top-k PCA — sign-canonicalized so ridge predictions are
    # NUMERICALLY IDENTICAL to the gesdd robust_pca_basis path (validated:
    # group-fold ridge-ρ |diff| = 0.0e0, 19× faster). Standardization + ridge
    # absorb column sign, so a sign-fixed same-subspace basis gives the same fit.
    mu, comps = _gram_top_k_pca(X, k)
    return (X - mu) @ comps.T


def _gram_top_k_pca(Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact, sign-canonicalized top-k right singular vectors of centered Y (n×n Gram).

    For a fat matrix (n ≪ H), the economy SVD ``robust_pca_basis`` does gesdd on
    the full (n, H) matrix (~0.66 s at n≈43, H=3584). The equivalent top-k right
    singular vectors come from the small (n, n) Gram ``Yc Ycᵀ = U S² Uᵀ`` (eigh,
    ~ms): ``Vt = S⁻¹ Uᵀ Yc``. Returns (mu (H,), comps (k', H)) with k' = min(k,
    n). Each component's sign is fixed so its largest-magnitude entry is positive.
    For panel-3 reconstruction the aggregate variance-weighted skill R² is
    basis-invariant; for panel-2 ridge read-out the sign-canonicalized basis
    gives NUMERICALLY IDENTICAL ridge-ρ to the gesdd path (validated: |diff|=0,
    ~19-22× faster) because ridge + per-column standardization absorb column sign.
    """
    mu = Y.mean(axis=0)
    Yc = Y - mu
    G = Yc @ Yc.T  # (n, n)
    w, U = np.linalg.eigh(G)  # ascending eigenvalues
    order = np.argsort(w)[::-1]
    kk = min(k, int((w[order] > 1e-9).sum()), Yc.shape[0])
    idx = order[:kk]
    S = np.sqrt(np.clip(w[idx], 1e-12, None))  # singular values
    Vt = (U[:, idx].T @ Yc) / S[:, None]  # (k, H) right singular vectors
    # sign-canonicalize: each row's largest-|entry| positive (deterministic basis)
    for i in range(Vt.shape[0]):
        j = int(np.argmax(np.abs(Vt[i])))
        if Vt[i, j] < 0:
            Vt[i] = -Vt[i]
    return mu, Vt


# ── plotting (match the reference layout EXACTLY) ────────────────────────────


def plot_heatmaps(p1, p2, p3, capture_layers):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis import paper_plots

        paper_plots.set_paper_style()
    except Exception as ex:
        logger.info("paper_plots.set_paper_style unavailable (%s) — default style", ex)

    n_layers = len(capture_layers)
    # constrained layout coexists with per-axes colorbars (tight_layout raised a
    # layout-engine conflict with the colorbars + paper_plots' engine).
    fig, axes = plt.subplots(3, 1, figsize=(20, 17), layout="constrained")
    suptitle = (
        "LEAVE-ONE-FAMILY-OUT (7-fold) — layer × answer-position (n=50, 7 families); "
        "panel1 fixed-r_B is fold-invariant"
    )
    fig.suptitle(suptitle, fontsize=14)

    specs = [
        (
            axes[0],
            p1,
            "coolwarm",
            -0.85,
            0.85,
            "Read-out: persona vector (fixed r_B) — rho vs graded E0, avg over 3 behaviors "
            "(fold-invariant)",
        ),
        (
            axes[1],
            p2,
            "coolwarm",
            -0.85,
            0.85,
            "Read-out: trained linear map (LOFO 7-fold ridge) — rho, avg over 3 behaviors",
        ),
        (
            axes[2],
            p3,
            "PuOr_r",
            -0.9,
            0.9,
            "Reconstruction from context vector c_C — skill-over-mean R^2 (LOFO ridge)",
        ),
    ]
    for ax, grid, cmap, vmin, vmax, title in specs:
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[-0.5, len(COL_ORDER) - 0.5, -0.5, n_layers - 0.5],
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("layer")
        for vx in VLINES:
            ax.axvline(vx, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_yticks(range(0, n_layers, 5))
        ax.set_yticklabels([str(capture_layers[i]) for i in range(0, n_layers, 5)])
        ax.set_xticks(range(len(COL_ORDER)))
        if ax is axes[2]:
            ax.set_xticklabels(COL_LABELS, rotation=90, fontsize=7)
            ax.set_xlabel("answer position (start -> end -> boundary | aggregates)")
        else:
            ax.set_xticklabels([])
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("[phase=plot] wrote %s", OUT_FIG)


# ── LOCO-vs-LOFO delta ───────────────────────────────────────────────────────


def loco_recon_best_midlate(summary):
    """(#810 LOCO reconstruction) best mid-late (layer 14-27) ridge_skill for a summary.

    Read from eval_results/issue_810/reconstruction_skill_by_summary.json — do NOT
    refit LOCO. Returns (best_layer, best_ridge_skill) or (None, None).
    """
    d = _load_json(LOCO_RECON)
    cells = d["by_summary"].get(summary, [])
    cand = [c for c in cells if c.get("ridge_skill") is not None and 14 <= c.get("layer", -1) <= 27]
    if not cand:
        return None, None
    best = max(cand, key=lambda c: c["ridge_skill"])
    return best["layer"], best["ridge_skill"]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Issue #810 adhoc LOFO 7-fold heatmaps")
    ap.add_argument(
        "--position-store-hf",
        default=POSITION_STORE_HF,
        help="HF prefix of the aligned-subset position store (default: the round-1 "
        "answer_position_sweep; the uh round passes answer_position_sweep_user_header)",
    )
    args = ap.parse_args()
    logger.info("[phase=load] manifest + summaries + c_C + position store + r_B + E0 + families")
    from huggingface_hub import hf_hub_download

    man = _load_json(hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset"))
    ctx_ids = context_ids_from_manifest(man)
    fam_map = _load_family_map()
    missing_fam = [c for c in ctx_ids if c not in fam_map]
    if missing_fam:
        raise RuntimeError(f"{len(missing_fam)} store contexts have no family: {missing_fam[:5]}")
    fam_counts = Counter(fam_map[c] for c in ctx_ids)
    logger.info("families over %d store contexts: %s", len(ctx_ids), dict(fam_counts))

    free_summaries, capture_layers = _load_free_summaries()
    cc = _load_cc(ctx_ids, capture_layers)
    pos_summaries, coverage = _load_position_summaries(ctx_ids, store_hf=args.position_store_hf)
    rb = _load_rb()
    e0 = _e0_graded_by_behavior()

    # Grid cache: the three (28×36) grids are the expensive product (~40 min of
    # LOFO ridge fits under VM contention). Persist them right after computing so a
    # crash in the cheap tail (plot / JSON) never forces a recompute — a rerun loads
    # the cache and skips straight to plot + JSON.
    grid_cache = OUT_JSON.parent / "adhoc_lofo_grids_cache.npz"
    if grid_cache.is_file():
        logger.info("[phase=cache] loading precomputed grids from %s", grid_cache)
        z = np.load(grid_cache)
        p1, p2, p3 = z["p1"], z["p2"], z["p3"]
    else:
        logger.info("[phase=panel1] fixed r_B read-out (fold-invariant)")
        p1 = build_panel_readout(
            ctx_ids,
            capture_layers,
            e0,
            rb,
            free_summaries,
            pos_summaries,
            coverage,
            fam_map,
            "fixed_rb",
        )
        logger.info("[phase=panel2] trained ridge read-out (LOFO 7-fold)")
        p2 = build_panel_readout(
            ctx_ids,
            capture_layers,
            e0,
            rb,
            free_summaries,
            pos_summaries,
            coverage,
            fam_map,
            "trained_ridge",
        )
        logger.info("[phase=panel3] c_C→summary reconstruction (LOFO 7-fold)")
        p3 = build_panel3_reconstruction(
            ctx_ids, capture_layers, cc, free_summaries, pos_summaries, coverage, fam_map
        )
        grid_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(grid_cache, p1=p1, p2=p2, p3=p3)
        logger.info("[phase=cache] wrote grids to %s", grid_cache)

    # LOCO-vs-LOFO reconstruction R² delta at each summary's best mid-late layer.
    # LOFO best-mid-late is read off p3 (layers 14-27); LOCO from the committed JSON.
    layer_to_idx = {L: i for i, L in enumerate(capture_layers)}
    recon_delta = {}
    for summary in COL_ORDER:
        ci = COL_ORDER.index(summary)
        # LOFO best over mid-late layers
        midlate = [
            (capture_layers[li], p3[li, ci])
            for li in range(len(capture_layers))
            if 14 <= capture_layers[li] <= 27 and np.isfinite(p3[li, ci])
        ]
        if midlate:
            lofo_layer, lofo_r2 = max(midlate, key=lambda t: t[1])
        else:
            lofo_layer, lofo_r2 = None, None
        loco_layer, loco_r2 = loco_recon_best_midlate(summary)
        # also report LOFO at the LOCO best layer for a like-for-like comparison
        lofo_at_loco = None
        if loco_layer is not None and loco_layer in layer_to_idx:
            v = p3[layer_to_idx[loco_layer], ci]
            lofo_at_loco = float(v) if np.isfinite(v) else None
        recon_delta[summary] = {
            "loco_best_midlate_layer": loco_layer,
            "loco_best_midlate_r2": loco_r2,
            "lofo_best_midlate_layer": lofo_layer,
            "lofo_best_midlate_r2": float(lofo_r2) if lofo_r2 is not None else None,
            "lofo_at_loco_best_layer": lofo_at_loco,
            "delta_lofo_minus_loco_at_loco_layer": (
                (lofo_at_loco - loco_r2)
                if (lofo_at_loco is not None and loco_r2 is not None)
                else None
            ),
        }

    def _grid_json(g):
        return [[None if not np.isfinite(v) else float(v) for v in row] for row in g]

    out = {
        "dv": "leave_one_family_out_7fold_layer_x_position_heatmaps",
        "note": (
            "LOFO 7-fold CV (fold = one battery family held out, train on other 6). "
            "Panel 1 fixed-r_B is fold-invariant (no fitted params) — computed over all "
            "50 contexts. Panels 2/3 use group-7-fold ridge with per-fold train-only PCA "
            "(cleaner than #810's all-data PCA) + train-only standardization/centering + "
            "#658 PRESS-LOO nested-CV lambda pick. LOCO recon read from "
            "reconstruction_skill_by_summary.json (NOT refit)."
        ),
        "n_contexts": len(ctx_ids),
        "families": dict(fam_counts),
        "capture_layers": capture_layers,
        "column_order": COL_ORDER,
        "column_labels": COL_LABELS,
        "vlines": VLINES,
        "grids": {
            "panel1_fixed_rb_readout_rho_avg3behaviors": _grid_json(p1),
            "panel2_trained_ridge_lofo_readout_rho_avg3behaviors": _grid_json(p2),
            "panel3_reconstruction_lofo_skill_over_mean_r2": _grid_json(p3),
        },
        "loco_vs_lofo_reconstruction_r2_delta_by_summary": recon_delta,
        "behaviors": list(HIGH_M_BEHAVIORS),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("[phase=json] wrote %s", OUT_JSON)

    # Plot LAST — the grids + JSON are already durable, so a plotting failure
    # never forces a recompute.
    plot_heatmaps(p1, p2, p3, capture_layers)
    logger.info("[phase=done] wrote %s and %s", OUT_FIG, OUT_JSON)

    # report the requested summaries
    for s in ("mean", "maxp", "turn_nl", "tail_1"):
        rd = recon_delta[s]
        logger.info(
            "RECON %-8s LOCO(L%s)=%.4f  LOFO_best_midlate(L%s)=%.4f  LOFO@LOCO_L=%s",
            s,
            rd["loco_best_midlate_layer"],
            rd["loco_best_midlate_r2"] or float("nan"),
            rd["lofo_best_midlate_layer"],
            rd["lofo_best_midlate_r2"] if rd["lofo_best_midlate_r2"] is not None else float("nan"),
            rd["lofo_at_loco_best_layer"],
        )
    # peak trained-ridge readout rho (LOFO) vs LOCO peak
    lofo_peak = np.nanmax(p2)
    logger.info("PEAK trained-ridge readout rho LOFO (behavior-avg): %.4f", lofo_peak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
