"""Matched-capacity fitted nulls + equality CI for #825 Result 2 (reparam).

Task #825 (followup_label: reparam-matched-capacity-null). The map-alignment
follow-up (scripts/issue825_map_alignment.py) found that the same-function
composition A_ans o M_base o A_ctx_rev predicts instruct answers at held-out
R^2 0.6728 -- essentially the within-instruct own-map ceiling 0.6731 @ L19 --
i.e. the base read-out map, reparameterized through fitted linear alignments,
IS the instruct read-out map. Two gaps in that result are closed here, 0-GPU,
analysis-only, on the SAME persisted turnstore (no new training / generation /
eval):

  1. MATCHED-CAPACITY FITTED NULLS. The parent's composition-collapse null
     (issue825_map_alignment._composition_collapse_null) replaced each alignment
     map with a RANDOM ORTHOGONAL Q -- a "no capacity" null. That collapses to
     ~ -0.69 / -0.33, so the composition beats "random rotation", but it does NOT
     isolate WHICH ingredient carries the recovery, because a random Q keeps none
     of the fitting machinery. Two matched-capacity nulls keep EVERYTHING about
     the fit identical (same general-linear ridge alignment class, same GCV lambda
     grid, same shared conversation-grouped K=5 folds seed 0, same n, same cached
     eigh(Gram) per fold) and destroy exactly one thing under test:
       N1 shuffled-correspondence -- refit the cross-model alignment maps on
         ROW-SHUFFLED (conversation-level) cross-model pairs on the TRAIN split
         (held-out eval rows keep true correspondence), compose with the REAL
         within-model center operator, evaluate held-out. Tests whether fitted
         sandwich CAPACITY alone (not the true cross-model correspondence)
         recovers the prediction.
       N2 structure-destroyed middle -- keep the REAL fitted alignments, replace
         the center operator (M_base for b2i / M_inst for i2b) with a
         spectrum-matched random operator (SAME singular values, random orthogonal
         singular subspaces), per draw. Tests whether the base operator's
         STRUCTURE (which directions map where), not merely its spectrum/scale,
         carries the recovery.
  2. EQUALITY CI. A conversation-grouped bootstrap (default 1000 draws) over the
     FIXED held-out per-row predictions for the delta (comp_samefn - own-map
     ceiling), both directions, layer 19. No refit per draw -- resample
     conversations of the fixed held-out predictions.

Divergence from #1345's matched-capacity reparam nulls
(scripts/issue1345_operator_comparison.reparam_null_battery), which the brief
asked to mirror: #1345 nulls the CENTER operator two ways -- shuffle_fit (M_j
fit on conversation-level-shuffled train answers) and rotation (random
orthogonal rotations wrapped around the TRUE M_j). BOTH keep the alignment maps
real and perturb the center. This script's N1 instead shuffles the ALIGNMENT
maps' cross-model correspondence (center real); N2 replaces the center with a
spectrum-matched random operator (a different center perturbation than #1345's
"true M sandwiched in rotations" -- N2 destroys the center's own singular
structure while preserving its spectrum). So the two null families are
complementary, not identical. Mirrored conventions: conversation-level shuffling
(rows of a conversation move together; here 1 row/conv so it reduces to
row-level), per-fold cached eigh(Gram) reuse (no per-draw Gram refit), fold-local
pooled R^2 for the null comparison, both directions.

Reuses the loader (extract/load_cell/align_pair), the VERBATIM ridge core
(_ridge_prep/_ridge_predict/_cv_folds), and the constants from
scripts/issue825_map_alignment.py (which reuses scripts/
issue825_crossmodel_map_transfer.py -> issue825_fit_cells.py@56ee95fe8a). The
center-operator primal beta is derived FROM the same cached per-fold prep as the
composition uses (validated by an in-run self-check that the primal apply
reproduces the dual _ridge_predict to fp64 tolerance), so N2's "same operator
machinery, structure destroyed" swap is faithful.

CLI:
  uv run python scripts/issue825_reparam_matched_null.py \
      --out eval_results/issue_825/map_alignment_matched_null \
      --dl-dir data/issue_825/hf_dl/map_alignment \
      [--n1-draws 20 --n2-draws 20 --ci-draws 1000] [--smoke --out /tmp/...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue825_map_alignment as ma  # noqa: E402

# Reused constants + helpers (single source of truth = the map-alignment script,
# which reuses the crossmodel script's VERBATIM ridge core).
HEADLINE_LAYER = ma.HEADLINE_LAYER
N_FOLDS = ma.N_FOLDS
FIT_SEED = ma.FIT_SEED
LAMBDAS = ma.LAMBDAS
STEM_INSTRUCT = ma.STEM_INSTRUCT
STEM_BASE = ma.STEM_BASE
ROLE = ma.ROLE
_fit_device = ma._fit_device
_cv_folds = ma._cv_folds
_ridge_prep = ma._ridge_prep
_ridge_predict = ma._ridge_predict

COMMITTED_MAP_ALIGN = Path("eval_results/issue_825/map_alignment/results.json")
GATE_TOL = 0.01
# Result-2 observed headline values the brief pins as the wiring target.
BRIEF_TARGETS = {
    "comp_samefn_b2i": 0.6728,
    "comp_samefn_i2b": 0.5888,
    "within_instruct": 0.673,
    "within_base": 0.588,
}
N2_ENERGY_KEEP = 0.9999  # spectrum-matched random middle: retain >=99.99% Frobenius energy


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


# ===========================================================================
# Center-operator primal beta derived from the SAME cached per-fold prep the
# composition uses (no extra eigh). Numerically identical to applying
# _ridge_predict(prep, Y_train, X_eval) as ((X_eval-xmu)/xsd) @ beta + ymu
# (validated by the in-run self-check).
# ===========================================================================
def _beta_from_prep(prep: dict, Y_train: torch.Tensor):
    """Standardized-input primal ridge coefficient (D_in, D_out) + affine stats,
    GCV lambda on the same grid, reusing the cached eigh(Gram) in ``prep``."""
    w, V, Xn, xmu, xsd, ntr = (
        prep["w"],
        prep["V"],
        prep["Xn"],
        prep["xmu"],
        prep["xsd"],
        prep["ntr"],
    )
    ymu = Y_train.mean(0)
    Yc = Y_train - ymu
    VtY = V.T @ Yc
    sqVtY = (VtY**2).sum(1)
    tot = float((Yc**2).sum())
    best_lam, best_gcv = float(LAMBDAS[0]), float("inf")
    for lam in LAMBDAS:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    filt = 1.0 / (w + best_lam)
    beta = Xn.T @ (V @ (filt[:, None] * VtY))  # (D_in, D_out) on standardized inputs
    return beta, xmu, xsd, ymu, best_lam


def _rand_orthonormal(d: int, r: int, gen: torch.Generator) -> torch.Tensor:
    """Haar-distributed (d, r) matrix with orthonormal columns (QR sign-fixed)."""
    A = torch.randn(d, r, dtype=torch.float64, generator=gen)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))
    return Q


def _spectrum_matched_svd(beta: torch.Tensor, energy_keep: float):
    """SVD of the center operator; retain the top-r singular values capturing
    >= energy_keep of the Frobenius (squared-SV) energy. Returns S_r (r,) and r,
    plus the exact fraction of energy retained (for reporting)."""
    S = torch.linalg.svdvals(beta).clamp(min=0.0)
    tot = float((S**2).sum()) + 1e-30
    csum = torch.cumsum(S**2, 0) / tot
    r = int(torch.searchsorted(csum, torch.tensor(energy_keep, dtype=csum.dtype)).item()) + 1
    r = max(1, min(r, S.shape[0]))
    S_r = S[:r].clone()
    energy = float((S_r**2).sum()) / tot
    return S_r, r, energy


# ===========================================================================
# Conversation-level train-split shuffle (rows of a conversation move together;
# here 1 row/conv so it reduces to a row permutation). Returns a permutation of
# the train-row POSITIONS.
# ===========================================================================
def _conv_shuffle_positions(conv_tr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uniq = np.unique(conv_tr)
    rows_of = {c: np.flatnonzero(conv_tr == c) for c in uniq}
    order = rng.permutation(len(uniq))
    out = np.concatenate([rows_of[uniq[k]] for k in order])
    assert len(out) == len(conv_tr), (len(out), len(conv_tr))
    return out


# ===========================================================================
# Fold-outer battery: hold ONE fold's preps at a time (peak ~one fold, like
# ma._layer_battery). For each fold, accumulate every fold-local contribution:
#   - real per-row held-out predictions (ceilings + comp_samefn, both dirs) for
#     the wiring gate + the equality-CI bootstrap;
#   - N1 shuffled-correspondence null ss_res per draw;
#   - N2 spectrum-matched-middle null ss_res per draw;
#   - fold-local ss_tot (true-Y variance) per direction (shared across draws).
# ===========================================================================
def _run_battery(data, conv, folds, layer, *, n1_draws, n2_draws, seed):
    Xi = data["Xi"][layer]
    Yi = data["Yi"][layer]
    Xb = data["Xb"][layer]
    Yb = data["Yb"][layer]
    n, D = Xi.shape
    Yi_np = Yi.cpu().numpy().astype(np.float64)
    Yb_np = Yb.cpu().numpy().astype(np.float64)

    # real per-row held-out predictions (assembled across folds)
    preds = {
        "ceil_i": np.zeros((n, D), np.float64),  # within_instruct
        "comp_b2i": np.zeros((n, D), np.float64),  # A_ans o M_base o A_ctx_rev
        "ceil_b": np.zeros((n, D), np.float64),  # within_base
        "comp_i2b": np.zeros((n, D), np.float64),  # A_ans_rev o M_inst o A_ctx
        "repswap_b2i": np.zeros((n, D), np.float64),  # Xb -> Yi (gate only)
    }
    fitted = np.zeros(n, bool)

    # fold-local pooled ss_tot (true-Y variance), per direction (b2i target=Yi, i2b=Yb)
    ss_tot = {"b2i": 0.0, "i2b": 0.0}
    # null ss_res accumulators (length = n_draws) per (null, direction)
    n1_res = {"b2i": np.zeros(n1_draws), "i2b": np.zeros(n1_draws)}
    n2_res = {"b2i": np.zeros(n2_draws), "i2b": np.zeros(n2_draws)}
    n2_rank = {"b2i": [], "i2b": []}
    n2_energy = {"b2i": [], "i2b": []}
    self_check = {"n2_primal_vs_dual_max_abs": 0.0}

    for k in range(N_FOLDS):
        tr_np = folds != k
        te_np = folds == k
        if te_np.sum() == 0 or tr_np.sum() < 3:
            continue
        tr = torch.as_tensor(tr_np)
        te = torch.as_tensor(te_np)
        conv_tr = conv[tr_np]
        te_idx = np.flatnonzero(te_np)
        fitted[te_idx] = True

        preps = {
            "Xi": _ridge_prep(Xi[tr]),
            "Xb": _ridge_prep(Xb[tr]),
            "Yb": _ridge_prep(Yb[tr]),
            "Yi": _ridge_prep(Yi[tr]),
        }

        # ---- real held-out predictions (both dirs) ----
        # b2i ceiling (within-instruct): Xi -> Yi
        p_ceil_i = _ridge_predict(preps["Xi"], Yi[tr], Xi[te])
        # b2i comp: A_ctx_rev (Xi->Xb), M_base (Xb->Yb), A_ans (Yb->Yi)
        xbhat = _ridge_predict(preps["Xi"], Xb[tr], Xi[te])
        ybhat = _ridge_predict(preps["Xb"], Yb[tr], xbhat)
        p_comp_b2i = _ridge_predict(preps["Yb"], Yi[tr], ybhat)
        # b2i repswap (gate only): Xb -> Yi
        p_rep_b2i = _ridge_predict(preps["Xb"], Yi[tr], Xb[te])
        # i2b ceiling (within-base): Xb -> Yb
        p_ceil_b = _ridge_predict(preps["Xb"], Yb[tr], Xb[te])
        # i2b comp: A_ctx (Xb->Xi), M_inst (Xi->Yi), A_ans_rev (Yi->Yb)
        xihat = _ridge_predict(preps["Xb"], Xi[tr], Xb[te])
        yihat = _ridge_predict(preps["Xi"], Yi[tr], xihat)
        p_comp_i2b = _ridge_predict(preps["Yi"], Yb[tr], yihat)

        preds["ceil_i"][te_idx] = p_ceil_i.cpu().numpy()
        preds["comp_b2i"][te_idx] = p_comp_b2i.cpu().numpy()
        preds["repswap_b2i"][te_idx] = p_rep_b2i.cpu().numpy()
        preds["ceil_b"][te_idx] = p_ceil_b.cpu().numpy()
        preds["comp_i2b"][te_idx] = p_comp_i2b.cpu().numpy()

        ss_tot["b2i"] += float(((Yi[te] - Yi[te].mean(0)) ** 2).sum())
        ss_tot["i2b"] += float(((Yb[te] - Yb[te].mean(0)) ** 2).sum())

        # ---- N2 spectrum-matched middle: derive the REAL center primal beta from
        # the cached prep, self-check the primal apply reproduces the dual predict,
        # then per draw replace the singular subspaces. Center: M_base(Xb->Yb) for
        # b2i (applied to real xbhat), M_inst(Xi->Yi) for i2b (applied to real xihat).
        gen2 = torch.Generator().manual_seed(seed + 101 + k)
        for direction, (prep_c, y_c, xhat, prep_ans, y_ans, true_te) in {
            "b2i": (preps["Xb"], Yb[tr], xbhat, preps["Yb"], Yi[tr], Yi[te]),
            "i2b": (preps["Xi"], Yi[tr], xihat, preps["Yi"], Yb[tr], Yb[te]),
        }.items():
            beta_c, xmu_c, xsd_c, ymu_c, _lam = _beta_from_prep(prep_c, y_c)
            # self-check: primal apply of the REAL center reproduces the dual predict
            real_dual = _ridge_predict(prep_c, y_c, xhat)
            real_primal = ((xhat - xmu_c) / xsd_c) @ beta_c + ymu_c
            self_check["n2_primal_vs_dual_max_abs"] = max(
                self_check["n2_primal_vs_dual_max_abs"],
                float((real_dual - real_primal).abs().max()),
            )
            S_r, r, energy = _spectrum_matched_svd(beta_c, N2_ENERGY_KEEP)
            n2_rank[direction].append(r)
            n2_energy[direction].append(energy)
            xhat_n = (xhat - xmu_c) / xsd_c
            for d in range(n2_draws):
                Q1 = _rand_orthonormal(D, r, gen2)
                Q2 = _rand_orthonormal(D, r, gen2)
                yhat_rand = (xhat_n @ (Q1 * S_r)) @ Q2.T + ymu_c
                pred = _ridge_predict(prep_ans, y_ans, yhat_rand)
                n2_res[direction][d] += float(((true_te - pred) ** 2).sum())
                del Q1, Q2, yhat_rand, pred
            del beta_c

        # ---- N1 shuffled-correspondence: refit the cross-model alignment maps on
        # a conversation-level TRAIN-split shuffle (one perm per draw, shared across
        # the two cross-model maps in the chain); center operator fit on TRUE pairs;
        # held-out eval rows keep true correspondence.
        rng1 = np.random.default_rng(seed + 201 + k)
        Xb_tr, Yi_tr, Xi_tr, Yb_tr = Xb[tr], Yi[tr], Xi[tr], Yb[tr]
        for d in range(n1_draws):
            perm = torch.as_tensor(_conv_shuffle_positions(conv_tr, rng1))
            # b2i: A_ctx_rev fit Xi->Xb[perm], M_base REAL, A_ans fit Yb->Yi[perm]
            xbhat_s = _ridge_predict(preps["Xi"], Xb_tr[perm], Xi[te])
            ybhat_s = _ridge_predict(preps["Xb"], Yb[tr], xbhat_s)  # real M_base
            pred_b2i = _ridge_predict(preps["Yb"], Yi_tr[perm], ybhat_s)
            n1_res["b2i"][d] += float(((Yi[te] - pred_b2i) ** 2).sum())
            # i2b: A_ctx fit Xb->Xi[perm], M_inst REAL, A_ans_rev fit Yi->Yb[perm]
            xihat_s = _ridge_predict(preps["Xb"], Xi_tr[perm], Xb[te])
            yihat_s = _ridge_predict(preps["Xi"], Yi[tr], xihat_s)  # real M_inst
            pred_i2b = _ridge_predict(preps["Yi"], Yb_tr[perm], yihat_s)
            n1_res["i2b"][d] += float(((Yb[te] - pred_i2b) ** 2).sum())
            del perm, xbhat_s, ybhat_s, pred_b2i, xihat_s, yihat_s, pred_i2b

        del preps
        print(f"[fold {k}] done (n_te={int(te_np.sum())})", flush=True)

    def _null_summary(res, tot):
        r2 = 1.0 - res / tot
        return {
            "null_mean": float(np.nanmean(r2)),
            "null_std": float(np.nanstd(r2)),
            "null_p975": float(np.nanquantile(r2, 0.975)),
            "null_max": float(np.nanmax(r2)),
            "n_draws": len(r2),
            "draws": [float(x) for x in r2],
        }

    nulls = {"N1_shuffled_correspondence": {}, "N2_spectrum_matched_middle": {}}
    for direction in ("b2i", "i2b"):
        nulls["N1_shuffled_correspondence"][direction] = _null_summary(
            n1_res[direction], ss_tot[direction]
        )
        d2 = _null_summary(n2_res[direction], ss_tot[direction])
        d2["rank_retained_per_fold"] = [int(x) for x in n2_rank[direction]]
        d2["frobenius_energy_retained_per_fold"] = [float(x) for x in n2_energy[direction]]
        nulls["N2_spectrum_matched_middle"][direction] = d2

    return {
        "preds": preds,
        "Y": {"i": Yi_np, "b": Yb_np},
        "fitted": fitted,
        "ss_tot_foldlocal": ss_tot,
        "nulls": nulls,
        "self_check": self_check,
    }


# ===========================================================================
# Pooled R^2 helpers (fold-local for the observed/gate; global for the CI).
# ===========================================================================
def _foldlocal_r2(pred, true, folds, fitted):
    ss_res = 0.0
    ss_tot = 0.0
    for k in range(N_FOLDS):
        te = (folds == k) & fitted
        if te.sum() == 0:
            continue
        t = true[te]
        p = pred[te]
        ss_res += float(((t - p) ** 2).sum())
        ss_tot += float(((t - t.mean(0)) ** 2).sum())
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def _global_r2(pred, true, rows):
    t = true[rows]
    p = pred[rows]
    mu = t.mean(0)
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - mu) ** 2).sum())
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


# ===========================================================================
# Equality CI: conversation-grouped bootstrap over FIXED held-out per-row
# predictions for delta = R2(comp_samefn) - R2(own-map ceiling). No refit.
# Global pooled R^2 on each resample (standard); resample conversations with
# replacement (here 1 row/conv, so conv-level == row-level).
# ===========================================================================
def _equality_ci(comp_pred, ceil_pred, true, conv, fitted, *, n_boot, seed):
    rows = np.flatnonzero(fitted)
    conv_f = conv[rows]
    uniq = np.unique(conv_f)
    rows_of = [np.flatnonzero(conv_f == c) for c in uniq]  # positions into `rows`
    n_conv = len(uniq)
    Y = true[rows].astype(np.float64)
    Pc = comp_pred[rows].astype(np.float64)
    Pk = ceil_pred[rows].astype(np.float64)
    rownorm2 = (Y**2).sum(1)  # per-row ||Y||^2
    e_comp = ((Y - Pc) ** 2).sum(1)  # per-row comp squared error
    e_ceil = ((Y - Pk) ** 2).sum(1)  # per-row ceiling squared error

    def _delta_for(idx):
        n_res = len(idx)
        S1 = Y[idx].sum(0)
        ss_tot = float(rownorm2[idx].sum()) - float(S1 @ S1) / n_res
        if ss_tot < 1e-12:
            return float("nan"), float("nan"), float("nan")
        r2c = 1.0 - float(e_comp[idx].sum()) / ss_tot
        r2k = 1.0 - float(e_ceil[idx].sum()) / ss_tot
        return r2c - r2k, r2c, r2k

    obs_idx = np.arange(len(rows))
    obs_delta, obs_r2c, obs_r2k = _delta_for(obs_idx)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_conv, size=n_conv)
        idx = np.concatenate([rows_of[c] for c in pick])
        deltas[b], _, _ = _delta_for(idx)
    return {
        "observed_delta_global": float(obs_delta),
        "observed_comp_r2_global": float(obs_r2c),
        "observed_ceiling_r2_global": float(obs_r2k),
        "boot_delta_mean": float(np.nanmean(deltas)),
        "boot_delta_ci95": [
            float(np.nanquantile(deltas, 0.025)),
            float(np.nanquantile(deltas, 0.975)),
        ],
        "boot_delta_ci90": [
            float(np.nanquantile(deltas, 0.05)),
            float(np.nanquantile(deltas, 0.95)),
        ],
        "n_boot": int(n_boot),
        "n_conversations": int(n_conv),
    }


# ===========================================================================
# Figure: observed composition vs the two null distributions + ceiling band,
# per direction, layer 19.
# ===========================================================================
def _make_figure(results: dict, fig_root: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    pal = paper_palette(4)
    c_n1, c_n2, c_comp, c_ceil = pal[0], pal[1], pal[2], pal[3]
    L = str(HEADLINE_LAYER)
    obs = results["observed_foldlocal"]
    nulls = results["nulls"]
    ci = results["equality_ci"]
    specs = [
        ("b2i", "comp_samefn_b2i", "within_instruct", "base rep → instruct target"),
        ("i2b", "comp_samefn_i2b", "within_base", "instruct rep → base target"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), layout="constrained")
    for ax, (d, comp_key, ceil_key, title) in zip(axes, specs, strict=True):
        n1 = nulls["N1_shuffled_correspondence"][d]["draws"]
        n2 = nulls["N2_spectrum_matched_middle"][d]["draws"]
        lo = min(min(n1), min(n2), 0.0) - 0.05
        hi = obs[comp_key] + 0.08
        bins = np.linspace(lo, hi, 40)
        ax.hist(n1, bins=bins, color=c_n1, alpha=0.75, label="N1 shuffled-corresp. null")
        ax.hist(n2, bins=bins, color=c_n2, alpha=0.75, label="N2 spectrum-matched-mid null")
        ax.axvline(obs[comp_key], color=c_comp, lw=2.2, label=f"observed comp {obs[comp_key]:.3f}")
        ax.axvline(
            obs[ceil_key],
            color=c_ceil,
            lw=2.0,
            ls="--",
            label=f"own-map ceiling {obs[ceil_key]:.3f}",
        )
        ci_d = ci[d]
        ax.set_title(
            f"{title}\ncomp-ceiling delta 95% CI [{ci_d['boot_delta_ci95'][0]:.3f}, "
            f"{ci_d['boot_delta_ci95'][1]:.3f}]",
            fontsize=9,
        )
        ax.set_xlabel(f"Held-out $R^2$ (layer {L})")
        ax.set_ylabel("Null draws")
        ax.legend(fontsize=6.5, loc="upper center")
    fig.suptitle(
        "Reparameterized composition vs matched-capacity fitted nulls (A_ans ∘ M ∘ A_ctx_rev)",
        fontsize=10,
    )
    savefig_paper(fig, "issue_825/reparam_matched_null", dir=fig_root)
    plt.close(fig)
    print(f"[figure] wrote {fig_root}/issue_825/reparam_matched_null.png", flush=True)


# ===========================================================================
# Orchestration
# ===========================================================================
def run(out_dir: Path, dl_dir: Path, *, n1_draws, n2_draws, ci_draws, smoke) -> dict:
    from huggingface_hub import HfApi

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    Lh = HEADLINE_LAYER

    try:
        resolved = HfApi().repo_info(ma.HF_DATA_REPO, repo_type="dataset", revision=ma.HF_REV).sha
    except Exception as e:
        resolved = f"unresolved: {e}"

    npz_i = ma.cm.extract_stem(STEM_INSTRUCT, dl_dir)
    npz_b = ma.cm.extract_stem(STEM_BASE, dl_dir)
    data, conv, _layers, al = ma._load_pair(npz_i, npz_b, [Lh])
    folds = _cv_folds(conv, N_FOLDS, FIT_SEED)
    print(f"[load] n_common={al['n_common']} staged in {time.time() - t0:.1f}s", flush=True)

    bat = _run_battery(data, conv, folds, Lh, n1_draws=n1_draws, n2_draws=n2_draws, seed=FIT_SEED)
    preds, Y, fitted = bat["preds"], bat["Y"], bat["fitted"]

    # ---- observed fold-local + global pooled R^2 (both poolings reported) ----
    rows = np.flatnonzero(fitted)
    obs_fl = {
        "within_instruct": _foldlocal_r2(preds["ceil_i"], Y["i"], folds, fitted),
        "within_base": _foldlocal_r2(preds["ceil_b"], Y["b"], folds, fitted),
        "repswap_b2i": _foldlocal_r2(preds["repswap_b2i"], Y["i"], folds, fitted),
        "comp_samefn_b2i": _foldlocal_r2(preds["comp_b2i"], Y["i"], folds, fitted),
        "comp_samefn_i2b": _foldlocal_r2(preds["comp_i2b"], Y["b"], folds, fitted),
    }
    obs_gl = {
        "within_instruct": _global_r2(preds["ceil_i"], Y["i"], rows),
        "within_base": _global_r2(preds["ceil_b"], Y["b"], rows),
        "comp_samefn_b2i": _global_r2(preds["comp_b2i"], Y["i"], rows),
        "comp_samefn_i2b": _global_r2(preds["comp_i2b"], Y["b"], rows),
    }

    # ---- wiring gate (fold-local, vs committed map_alignment + brief targets) ----
    with open(COMMITTED_MAP_ALIGN) as f:
        committed = json.load(f)["per_layer"][str(Lh)]
    gate_specs = [
        ("within_instruct", obs_fl["within_instruct"], committed["ceilings"]["within_instruct"]),
        ("within_base", obs_fl["within_base"], committed["ceilings"]["within_base"]),
        ("repswap_b2i", obs_fl["repswap_b2i"], committed["ceilings"]["repswap_b2i"]),
        (
            "comp_samefn_b2i",
            obs_fl["comp_samefn_b2i"],
            committed["composition"]["linear"]["comp_samefn_b2i"],
        ),
        (
            "comp_samefn_i2b",
            obs_fl["comp_samefn_i2b"],
            committed["composition"]["linear"]["comp_samefn_i2b"],
        ),
    ]
    gates = {}
    all_pass = True
    for name, o, e in gate_specs:
        delta = abs(o - e)
        ok = delta <= GATE_TOL
        gates[name] = {"observed": o, "committed": e, "abs_delta": delta, "pass": bool(ok)}
        all_pass = all_pass and ok
    gates["all_pass"] = bool(all_pass)
    _gate_deltas = {k: round(v["abs_delta"], 5) for k, v in gates.items() if isinstance(v, dict)}
    print(f"[gate] all_pass={all_pass} {json.dumps(_gate_deltas)}", flush=True)
    if not all_pass:
        raise SystemExit(f"WIRING GATE FAILURE (|delta|>{GATE_TOL}): {json.dumps(gates, indent=2)}")

    # ---- equality CI (both directions) ----
    equality_ci = {
        "b2i": _equality_ci(
            preds["comp_b2i"],
            preds["ceil_i"],
            Y["i"],
            conv,
            fitted,
            n_boot=ci_draws,
            seed=FIT_SEED + 7,
        ),
        "i2b": _equality_ci(
            preds["comp_i2b"],
            preds["ceil_b"],
            Y["b"],
            conv,
            fitted,
            n_boot=ci_draws,
            seed=FIT_SEED + 9,
        ),
    }

    results = {
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hf_repo": ma.HF_DATA_REPO,
            "hf_revision_pinned": ma.HF_REV,
            "hf_revision_resolved": resolved,
            "hf_prefix": ma.HF_PREFIX,
            "stems": [STEM_INSTRUCT, STEM_BASE],
            "role": ROLE,
            "headline_layer": Lh,
            "n_folds": N_FOLDS,
            "fit_seed": FIT_SEED,
            "lambdas": [float(x) for x in LAMBDAS],
            "device": str(_fit_device()),
            "n_common": al["n_common"],
            "n_conversations": len(np.unique(conv)),
            "n1_draws": int(n1_draws),
            "n2_draws": int(n2_draws),
            "ci_draws": int(ci_draws),
            "n2_energy_keep": N2_ENERGY_KEEP,
            "thread_caps": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "script": "scripts/issue825_reparam_matched_null.py",
            "ridge_core_source": "issue825_map_alignment.py (crossmodel -> fit_cells@56ee95fe8a)",
            "smoke": bool(smoke),
            "wall_seconds": None,
        },
        "wiring_gate": gates,
        "self_check": bat["self_check"],
        "observed_foldlocal": obs_fl,
        "observed_global": obs_gl,
        "nulls": bat["nulls"],
        "equality_ci": equality_ci,
        "caveats": [
            "Descriptive geometry on a SINGLE seed (fit seed 0); no mechanism claims.",
            "S_assistant_chat pair (chat single-turn, assistant slot); 1 row/conversation, so "
            "conversation-grouped == row-level here.",
            "Nulls use fold-local pooled R^2 (parent frozen_sweep convention) so the null "
            "distribution is directly comparable to the fold-local observed comp/ceiling. The "
            "equality CI uses GLOBAL pooled R^2 on each bootstrap resample (standard for a fixed "
            "held-out prediction set); observed_global is reported for that comparison.",
            "N1 shuffles the cross-model ALIGNMENT correspondence (center operator real); N2 "
            "replaces the center operator with a spectrum-matched random operator (alignments "
            "real). Both keep the full fitting machinery (ridge class, GCV grid, folds, n, cached "
            "eigh(Gram)) identical -- MATCHED CAPACITY. This differs from #1345's reparam nulls, "
            "which perturb the CENTER (shuffle its train answers / sandwich the true center in "
            "random rotations) with alignments real; the two null families are complementary.",
            "N2 retains >=99.99% of the center operator's Frobenius (squared-SV) energy in the "
            "top-r singular values (r + retained energy reported per fold); the near-zero SV tail "
            "is dropped for tractable per-draw QR of the random orthonormal factors.",
        ],
    }
    results["metadata"]["wall_seconds"] = round(time.time() - t0, 1)

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[write] {out_json} (wall {results['metadata']['wall_seconds']}s)", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="eval_results/issue_825/map_alignment_matched_null")
    ap.add_argument("--dl-dir", default="data/issue_825/hf_dl/map_alignment")
    ap.add_argument("--n1-draws", type=int, default=20)
    ap.add_argument("--n2-draws", type=int, default=20)
    ap.add_argument("--ci-draws", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_root = "figures" if not args.smoke else str(out_dir)

    if args.figures_only:
        with open(out_dir / "results.json") as f:
            results = json.load(f)
        _make_figure(results, fig_root)
        return

    n1 = 2 if args.smoke else args.n1_draws
    n2 = 2 if args.smoke else args.n2_draws
    ci = 50 if args.smoke else args.ci_draws
    results = run(
        out_dir, Path(args.dl_dir), n1_draws=n1, n2_draws=n2, ci_draws=ci, smoke=args.smoke
    )
    _make_figure(results, fig_root)


if __name__ == "__main__":
    main()
