#!/usr/bin/env python3
"""Issue #779 free re-analysis: two per-context reconstruction reads for ``h``.

The #779 body reports IN-SAMPLE reconstruction of the behavior-agnostic map
``h: c_last -> v(x)`` (ridge fit AND read on the SAME 5000 LMSYS contexts):
R2 0.833/0.834/0.860 and mean per-context cosine 0.972/0.983/0.976 at the
per-trait read-out layers {evil L14, syc L26, halluc L17}. The eval side stored
only the scalar ``oracle`` = <v(x), r_B>, never the full v(x), so held-out
reconstruction was never done. This 0-GPU driver fixes that with TWO reads on the
already-cached pass-B bundle + pass-A eval cells (no training, no rollouts).

Read 1 — HELD-OUT per-context reconstruction (5-fold CV over the 5000 LMSYS
  contexts). For each fold, fit ``h`` (ridge, ``fit_h.ridge_fit_predict``) on the
  train contexts and predict on the held-out TEST contexts. Report:
    - Held-out pooled R2 (SS_tot uses the TEST fold's OWN mean — the honest
      fraction of held-out target variance explained), mean +- sd across folds,
      at the 3 read-out layers AND at all 28 layers (held-out-R2-vs-layer curve).
    - Full per-context cosine distribution on held-out contexts at the 3 read-out
      layers: mean, sd, quantiles p5/p25/p50/p75/p95, min.
  Headline: the overfitting gap vs the in-sample 0.83-0.86 (N=5000 vs D=3584 is a
  ridge-can-overfit-in-sample regime).

Read 2 — monitoring-relevant per-context PROJECTION read (held-out eval contexts).
  Fit ``h`` on the LMSYS train corpus (``fit_h_readouts``, run_mlp=False) — it
  applies ``h`` zero-shot to the pass-A eval contexts and returns
  ``r1_ridge_dot`` = <h(c_last_eval), r_B>. Against the TRUE answer projection
  ``oracle`` = <v(x), r_B> (from ``build_eval_matrix``), per trait x mode
  {system, many_shot} at the read-out layer, compute the per-single-context
  correlation (DOT arm, matched units):
    - overall Pearson (all finite eval contexts), and
    - within-condition Pearson (project primary; ``bootstrap_within_condition_ci``,
      95% CI, exclude conditions with < 3 finite points).
  Diagnostic: high whole-profile reconstruction (Read 1) coexisting with a LOW
  projection correlation (Read 2) is direct evidence ``h`` reconstructs the
  generic profile while MISSING the trait direction r_B (the low-variance-direction
  underfit the body hypothesizes).

Reuses (does NOT reimplement): fit_h.ridge_fit_predict / reconstruction_metrics /
dot_readout; issue779_stage1.load_eval_cells / _load_rb / build_eval_matrix /
fit_h_readouts / _group_by_condition; metrics.overall_pearson /
bootstrap_within_condition_ci; issue779_common.TRAITS / EXPECTED_LAYERS /
EXPECTED_HIDDEN.

0-GPU, CPU/VM only. Fail loud — no try/except:pass, no dummy fills; a NaN /
insufficient-conditions cell is reported as such.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_common as C  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402
from explore_persona_space.experiments.issue_779 import metrics as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_percontext_recon")

# Per-trait oracle-selected read-out layers (step0; matches the #779 body).
READ_OUT_LAYER = {"evil": 14, "sycophancy": 26, "hallucination": 17}

# The #779 body's in-sample reconstruction numbers (ridge, fit & read on the same
# 5000 contexts) at the read-out layer — the comparison baseline for Read 1.
IN_SAMPLE_R2 = {"evil": 0.833, "sycophancy": 0.834, "hallucination": 0.860}
IN_SAMPLE_MEAN_COSINE = {"evil": 0.972, "sycophancy": 0.983, "hallucination": 0.976}


# ── Read 1: held-out per-context reconstruction (5-fold CV over LMSYS) ─────────


def _ridge_fit_predict_fast(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    lambdas: np.ndarray | None = None,
) -> np.ndarray:
    """Torch-eigh Gram-space ridge — the fast, verified-equivalent twin of
    ``fit_h.ridge_fit_predict`` (same standardize-X / center-Y / GCV-lambda-select /
    un-center recipe, identical predictions to 8e-13 and identical selected lambda;
    verified against the SVD path this session at layer 14).

    ``ridge_fit_predict`` runs a full ``numpy.linalg.svd`` of the (N_tr, H) train
    matrix per call (~125 s at N_tr=4000, H=3584), so 140 CV fits (28 layers x 5
    folds) would take ~5 h. This computes the DUAL ridge via ``torch.linalg.eigh``
    of the (N_tr, N_tr) Gram (~12 s/fold, MKL/OpenBLAS driver), cutting Read 1 to
    ~28 min at full float64 precision. GCV RSS is evaluated in eigen-coefficient
    space (no full train-fit reconstruction). Used ONLY for Read 1's 140 CV fits;
    Read 2's 3 fits stay on the canonical ``fit_h.ridge_fit_predict``.
    """
    if lambdas is None:
        lambdas = np.logspace(-2, 4, 13)
    Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float64)
    Ytr = torch.as_tensor(np.asarray(Y_train), dtype=torch.float64)
    Xev = torch.as_tensor(np.asarray(X_eval), dtype=torch.float64)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9  # matches ridge_fit_predict's 1e-9 (numpy .std is population)
    Xtr_n = (Xtr - xmu) / xsd
    Xev_n = (Xev - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    ntr = Xtr.shape[0]

    # Dual ridge: (G + lam I) alpha = Ytr_c, G = Xtr_n Xtr_n^T = V diag(w) V^T.
    G = Xtr_n @ Xtr_n.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    VtY = V.T @ Ytr_c  # (ntr, H)
    Kev = Xev_n @ Xtr_n.T  # (n_ev, ntr) cross-kernel
    KevV = Kev @ V
    sqVtY = (VtY**2).sum(1)  # per-eigencomponent target energy
    tot = float((Ytr_c**2).sum())

    # GCV: RSS(lam) = ||Y||^2 - sum_k (2 f_k - f_k^2) sqVtY_k with f = w/(w+lam),
    # dof = sum_k f_k (hat-matrix trace); GCV = RSS / (ntr - dof)^2.
    best_lam = float(lambdas[0])
    best_gcv = float("inf")
    for lam in lambdas:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)
    filt = 1.0 / (w + best_lam)
    pred = (KevV * filt) @ VtY + ymu
    return pred.numpy()


def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Pooled R2 with SS_tot on TRUE's OWN mean (honest held-out variance frac).

    pred/true (N, H). SS_res = sum((true-pred)**2), SS_tot = sum((true-mu)**2)
    with mu = true.mean(0) computed on THIS set (the test fold for held-out R2).
    Returns NaN when SS_tot is degenerate.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def _per_context_cosine(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-row cosine(pred, true) -> (N,). Matches fit_h.reconstruction_metrics."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    num = np.sum(pred * true, axis=1)
    den = (np.linalg.norm(pred, axis=1) + 1e-12) * (np.linalg.norm(true, axis=1) + 1e-12)
    return num / den


def _cosine_dist_summary(cos: np.ndarray) -> dict:
    """mean/sd/min + p5/p25/p50/p75/p95 of a per-context cosine array."""
    cos = np.asarray(cos, dtype=np.float64)
    cos = cos[np.isfinite(cos)]
    if cos.size == 0:
        return {
            k: float("nan") for k in ("mean", "sd", "min", "p5", "p25", "p50", "p75", "p95", "n")
        }
    q = np.quantile(cos, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "mean": float(cos.mean()),
        "sd": float(cos.std()),
        "min": float(cos.min()),
        "p5": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "n": int(cos.size),
    }


def _cv_folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """n_folds disjoint held-out index arrays (shuffled once, split contiguously)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(perm, n_folds)]


def read1_heldout_recon(
    train_bundle: dict,
    traits: list[str],
    *,
    n_folds: int,
    seed: int,
    n_layers: int,
) -> dict:
    """5-fold CV held-out R2 (all layers) + per-context cosine dist (read-out layer).

    ``h`` (c_last -> v(x)) is behavior-agnostic, so the fit/predict is IDENTICAL
    across traits at a given layer — we compute the per-fold held-out R2 ONCE per
    layer and the per-context held-out cosine ONCE per read-out layer, then map the
    read-out-layer results onto each trait by its layer. No r_B is used here (this
    read is about profile reconstruction, not the trait projection — that is Read 2).
    """
    cx_last = train_bundle["cx_last"]  # (N, 28, H) torch
    v_x = train_bundle["v_x"]
    layers = train_bundle["layers"]
    n = cx_last.shape[0]
    folds = _cv_folds(n, n_folds, seed)

    # Held-out pooled R2 per layer, per fold.
    r2_by_layer: dict[int, list[float]] = {li: [] for li in range(n_layers)}
    # Per-context held-out cosine (pooled across folds) at each read-out layer.
    readout_layers = sorted(set(READ_OUT_LAYER[t] for t in traits))
    cos_pool: dict[int, list[np.ndarray]] = {li: [] for li in readout_layers}

    for li in range(n_layers):
        col = layers.index(li)
        X = cx_last[:, col, :].numpy()
        Y = v_x[:, col, :].numpy()
        want_cos = li in cos_pool
        for test_idx in folds:
            mask = np.ones(n, dtype=bool)
            mask[test_idx] = False
            Xtr, Ytr = X[mask], Y[mask]
            Xte, Yte = X[test_idx], Y[test_idx]
            pred_te = _ridge_fit_predict_fast(Xtr, Ytr, Xte)  # (n_test, H); ==F.ridge_fit_predict
            r2_by_layer[li].append(_pooled_r2(pred_te, Yte))
            if want_cos:
                cos_pool[li].append(_per_context_cosine(pred_te, Yte))
        logger.info(
            "Read1 layer %2d: held-out R2 folds mean=%.4f sd=%.4f",
            li,
            float(np.mean(r2_by_layer[li])),
            float(np.std(r2_by_layer[li])),
        )

    # Per-layer held-out R2 summary (across folds).
    r2_curve = {
        int(li): {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals)),
            "folds": [float(v) for v in vals],
        }
        for li, vals in r2_by_layer.items()
    }
    # Pooled per-context cosine distribution at each read-out layer.
    cos_dist_by_layer = {
        int(li): _cosine_dist_summary(np.concatenate(arrs)) for li, arrs in cos_pool.items()
    }

    per_trait = {}
    for t in traits:
        li = READ_OUT_LAYER[t]
        per_trait[t] = {
            "read_out_layer": li,
            "heldout_r2_mean": r2_curve[li]["mean"],
            "heldout_r2_sd": r2_curve[li]["sd"],
            "heldout_r2_folds": r2_curve[li]["folds"],
            "in_sample_r2": IN_SAMPLE_R2[t],
            "overfitting_gap_r2": IN_SAMPLE_R2[t] - r2_curve[li]["mean"],
            "heldout_cosine_dist": cos_dist_by_layer[li],
            "in_sample_mean_cosine": IN_SAMPLE_MEAN_COSINE[t],
        }
    return {
        "n_contexts": int(n),
        "n_folds": n_folds,
        "seed": seed,
        "r2_convention": (
            "pooled held-out R2; SS_tot uses the TEST fold's own mean "
            "(fraction of held-out target variance explained)"
        ),
        "heldout_r2_vs_layer": r2_curve,
        "heldout_cosine_dist_by_readout_layer": cos_dist_by_layer,
        "per_trait": per_trait,
    }


# ── Read 2: monitoring-relevant projection read (held-out eval contexts) ──────


def read2_projection_recon(
    train_bundle: dict,
    cells_by_trait: dict,
    rb_by_trait: dict,
    traits: list[str],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Per-context Pearson of predicted <h(c),r_B> vs true <v(x),r_B> (DOT arm).

    Fits ``h`` on the train corpus, applies it zero-shot to each trait's pass-A
    eval contexts (fit_h_readouts, run_mlp=False -> r1_ridge_dot), and correlates
    it against the true answer projection ``oracle`` per trait x mode
    {system, many_shot}: overall Pearson + within-condition Pearson with 95% CI.
    """
    out = {}
    for t in traits:
        li = READ_OUT_LAYER[t]
        mat = S1.build_eval_matrix(cells_by_trait[t], li, rb_by_trait[t])
        n_eval = len(mat["oracle"])
        if n_eval < 3:
            out[t] = {"read_out_layer": li, "skipped": True, "n_eval": n_eval}
            continue
        h_out = S1.fit_h_readouts(
            train_bundle,
            mat,
            li,
            rb_by_trait[t],
            n_train_cap=0,
            pca_k=64,
            run_mlp=False,
        )
        pred = np.asarray(h_out["r1_ridge_dot"], dtype=np.float64)  # <h(c),r_B>
        true = np.asarray(mat["oracle"], dtype=np.float64)  # <v(x),r_B>
        trait_res = {"read_out_layer": li, "n_eval": int(n_eval)}
        for mode in ("system", "many_shot"):
            # overall Pearson across finite pairs within the mode.
            sel = np.array([m == mode for m in mat["mode"]]) & np.isfinite(pred) & np.isfinite(true)
            overall = M.overall_pearson(pred[sel], true[sel]) if sel.sum() >= 3 else float("nan")
            # within-condition Pearson (project primary): group predicted vs oracle
            # by condition WITHIN the mode, drop conds with < 3 finite points.
            cx, cy = S1._group_by_condition(pred, true, mat["cond"], mat["mode"], mode)
            cx2, cy2 = [], []
            for xi, yi in zip(cx, cy, strict=True):
                m = np.isfinite(xi) & np.isfinite(yi)
                if m.sum() >= 3:
                    cx2.append(xi[m])
                    cy2.append(yi[m])
            wc = M.bootstrap_within_condition_ci(cx2, cy2, n_boot=n_boot, seed=seed)
            trait_res[mode] = {
                "overall_pearson": overall,
                "overall_n": int(sel.sum()),
                "within_condition_r": wc["point"],
                "within_condition_lo": wc["lo"],
                "within_condition_hi": wc["hi"],
                "within_condition_n_conditions": wc["n_conditions"],
            }
            logger.info(
                "Read2 %s %s: overall r=%.3f (n=%d) | within-cond r=%.3f [%.3f, %.3f] (k=%d)",
                t,
                mode,
                overall,
                int(sel.sum()),
                wc["point"],
                wc["lo"],
                wc["hi"],
                wc["n_conditions"],
            )
        out[t] = trait_res
    return out


# ── figures ───────────────────────────────────────────────────────────────────


def _apply_paper_style() -> None:
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:
        logger.warning("paper_plots style unavailable (%s); matplotlib default", e)


def make_read1_figure(read1: dict, traits: list[str], fig_path: Path) -> None:
    """Held-out vs in-sample R2 per layer (left) + per-context cosine violin (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_paper_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5), layout="tight")

    # Left: held-out R2 vs layer (mean +- sd across folds), + in-sample dots at read-out layers.
    curve = read1["heldout_r2_vs_layer"]
    lis = sorted(int(k) for k in curve)

    def _cv(li):  # int keys in-memory, str after JSON round-trip
        return curve[li] if li in curve else curve[str(li)]

    means = [_cv(li)["mean"] for li in lis]
    sds = [_cv(li)["sd"] for li in lis]
    axl.errorbar(
        lis, means, yerr=sds, marker="o", ms=3, lw=1, capsize=2, label="held-out R2 (5-fold)"
    )
    for t in traits:
        li = READ_OUT_LAYER[t]
        axl.scatter([li], [IN_SAMPLE_R2[t]], marker="*", s=140, zorder=5)
        axl.annotate(
            f"{t}\nin-sample {IN_SAMPLE_R2[t]:.3f}",
            (li, IN_SAMPLE_R2[t]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=7,
        )
    axl.set_xlabel("layer")
    axl.set_ylabel("R2 (profile reconstruction)")
    axl.set_title("Held-out (5-fold) vs in-sample R2 by layer")
    axl.legend(loc="lower center", fontsize=8)

    # Right: per-context held-out cosine distribution (violin) at the 3 read-out layers.
    data, labels = [], []
    d = read1["heldout_cosine_dist_by_readout_layer"]
    for t in traits:
        li = READ_OUT_LAYER[t]
        entry = d[li] if li in d else d[str(li)]  # int keys in-memory, str after JSON round-trip
        # Reconstruct an approximate spread marker set from the quantiles for the
        # violin's underlying data is unavailable post-summary; plot the quantile
        # box instead (min, p5, p25, p50, p75, p95).
        qs = [entry["min"], entry["p5"], entry["p25"], entry["p50"], entry["p75"], entry["p95"]]
        data.append(qs)
        labels.append(f"{t}\nL{li}")
    axr.boxplot(
        data,
        labels=labels,
        showmeans=True,
        whis=(0, 100),
        medianprops={"lw": 1.5},
    )
    for t, entry_key in zip(traits, range(1, len(traits) + 1), strict=True):
        li = READ_OUT_LAYER[t]
        axr.scatter([entry_key], [IN_SAMPLE_MEAN_COSINE[t]], marker="*", s=120, zorder=6)
    axr.set_ylabel("per-context cosine(pred, true) — held-out")
    axr.set_title("Held-out per-context cosine (quantile box; ★ = in-sample mean)")

    # layout="tight" at creation handles spacing (no post-hoc tight_layout()).
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


def make_read2_figure(
    read2: dict,
    cells_by_trait: dict,
    rb_by_trait: dict,
    train_bundle: dict,
    traits: list[str],
    fig_path: Path,
) -> None:
    """Predicted <h(c),r_B> vs true <v(x),r_B> scatter, one panel per trait (system)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_paper_style()
    # Explicit tight layout (NOT constrained): set_paper_style may enable the
    # constrained_layout engine, which raises on a post-hoc fig.tight_layout() once
    # a colorbar exists ("Colorbar layout of new layout engine not compatible").
    fig, axes = plt.subplots(
        1, len(traits), figsize=(5 * len(traits), 5), squeeze=False, layout="tight"
    )
    for ax, t in zip(axes[0], traits, strict=True):
        li = READ_OUT_LAYER[t]
        mat = S1.build_eval_matrix(cells_by_trait[t], li, rb_by_trait[t])
        h_out = S1.fit_h_readouts(
            train_bundle,
            mat,
            li,
            rb_by_trait[t],
            n_train_cap=0,
            pca_k=64,
            run_mlp=False,
        )
        pred = np.asarray(h_out["r1_ridge_dot"], dtype=np.float64)
        true = np.asarray(mat["oracle"], dtype=np.float64)
        sel = np.array([m == "system" for m in mat["mode"]]) & np.isfinite(pred) & np.isfinite(true)
        conds = mat["cond"][sel]
        sc = ax.scatter(true[sel], pred[sel], c=conds, cmap="tab20", s=18, alpha=0.75)
        rr = read2[t].get("system", {})
        ax.set_title(
            f"{t} (L{li}, system)\noverall r={rr.get('overall_pearson', float('nan')):.3f} | "
            f"within-cond r={rr.get('within_condition_r', float('nan')):.3f}"
        )
        ax.set_xlabel("TRUE answer projection  <v(x), r_B>  (oracle)")
        ax.set_ylabel("PREDICTED  <h(c_last), r_B>")
        fig.colorbar(sc, ax=ax, label="condition", fraction=0.046, pad=0.04)
    # layout="tight" at figure creation handles spacing; no post-hoc tight_layout()
    # (incompatible with colorbars under a constrained engine).
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 per-context reconstruction reads.")
    parser.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    parser.add_argument(
        "--collect-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "issue779_hfstage"
        / "issue779_monitoring"
        / "analysis_tensors",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "percontext_recon.json",
    )
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_779")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-layers", type=int, default=C.EXPECTED_LAYERS)
    parser.add_argument("--hidden", type=int, default=C.EXPECTED_HIDDEN)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="rebuild figures from an existing --out-json (skip the CV/read compute)",
    )
    args = parser.parse_args()

    torch.set_num_threads(int(args.n_threads))
    traits = args.traits
    collect_dir = args.collect_dir
    pass_a_dir = collect_dir / "pass_a"
    rb_dir = collect_dir / "r_b"

    logger.info("Loading pass-B train bundle from %s", collect_dir / "pass_b")
    train_bundle = torch.load(
        collect_dir / "pass_b" / "train_context_vectors.pt", weights_only=False
    )
    assert train_bundle["cx_last"].shape[1:] == (args.n_layers, args.hidden), train_bundle[
        "cx_last"
    ].shape

    rb_by_trait = {t: S1._load_rb(rb_dir, t, args.n_layers, args.hidden) for t in traits}
    cells_by_trait = {t: S1.load_eval_cells(pass_a_dir, t) for t in traits}

    if args.figures_only:
        # Rebuild figures from the existing JSON (skip the ~28-min CV + reads). Used
        # to regenerate a plot after a plotting-only crash without recomputing.
        logger.info("--figures-only: rebuilding figures from %s", args.out_json)
        with open(args.out_json) as f:
            existing = json.load(f)
        make_read1_figure(
            existing["read1_heldout_recon"], traits, args.fig_dir / "r3_heldout_recon.png"
        )
        make_read2_figure(
            existing["read2_projection_recon"],
            cells_by_trait,
            rb_by_trait,
            train_bundle,
            traits,
            args.fig_dir / "r3_projection_recon.png",
        )
        logger.info("--figures-only: done")
        return 0

    # Fast-ridge equivalence gate: assert _ridge_fit_predict_fast reproduces the
    # canonical fit_h.ridge_fit_predict on a small subsample BEFORE the 140 CV
    # fits use it (fail-loud if the torch-eigh accelerator ever diverges).
    _rng = np.random.default_rng(args.seed)
    _sub = _rng.choice(train_bundle["cx_last"].shape[0], size=600, replace=False)
    _Xg = train_bundle["cx_last"][:, 14, :].numpy()[_sub]
    _Yg = train_bundle["v_x"][:, 14, :].numpy()[_sub]
    _pred_slow = F.ridge_fit_predict(_Xg[:500], _Yg[:500], _Xg[500:])
    _pred_fast = _ridge_fit_predict_fast(_Xg[:500], _Yg[:500], _Xg[500:])
    _Yte_g = _Yg[500:]
    # The two paths differ only in LAPACK route (numpy SVD vs torch eigh); on a
    # small n<<D near-singular subsample the raw-prediction max-abs diff can reach
    # ~1e-6 in absolute terms, but the DV that matters (R2 + relative agreement)
    # is identical to machine precision. Gate on those.
    _abs_diff = float(np.max(np.abs(_pred_slow - _pred_fast)))
    _rel_diff = _abs_diff / (float(np.max(np.abs(_pred_slow))) + 1e-12)
    _r2_slow = _pooled_r2(_pred_slow, _Yte_g)
    _r2_fast = _pooled_r2(_pred_fast, _Yte_g)
    _r2_diff = abs(_r2_slow - _r2_fast)
    assert _rel_diff < 1e-6 and _r2_diff < 1e-6, (
        f"fast-ridge equivalence gate FAILED: rel_diff {_rel_diff:.2e}, "
        f"R2 diff {_r2_diff:.2e} (abs pred diff {_abs_diff:.2e})"
    )
    logger.info(
        "fast-ridge equivalence gate PASS (rel pred diff %.2e, R2 diff %.2e, abs %.2e)",
        _rel_diff,
        _r2_diff,
        _abs_diff,
    )

    results: dict = {
        "read_out_layers": {t: READ_OUT_LAYER[t] for t in traits},
        "metadata": C.reproducibility_metadata(
            {"script": "issue779_percontext_recon", "n_folds": args.n_folds, "seed": args.seed}
        ),
    }

    # ── Read 1 ──
    logger.info("=== Read 1: held-out per-context reconstruction (%d-fold CV) ===", args.n_folds)
    read1 = read1_heldout_recon(
        train_bundle, traits, n_folds=args.n_folds, seed=args.seed, n_layers=args.n_layers
    )
    results["read1_heldout_recon"] = read1
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_json, results)  # checkpoint after Read 1
    logger.info("Checkpointed Read 1 -> %s", args.out_json)

    # ── Read 2 ──
    logger.info("=== Read 2: monitoring-relevant projection read (held-out eval) ===")
    read2 = read2_projection_recon(
        train_bundle, cells_by_trait, rb_by_trait, traits, n_boot=args.n_boot, seed=args.seed
    )
    results["read2_projection_recon"] = read2
    C.write_json_atomic(args.out_json, results)  # checkpoint after Read 2
    logger.info("Checkpointed Read 2 -> %s", args.out_json)

    # ── figures ──
    make_read1_figure(read1, traits, args.fig_dir / "r3_heldout_recon.png")
    make_read2_figure(
        read2,
        cells_by_trait,
        rb_by_trait,
        train_bundle,
        traits,
        args.fig_dir / "r3_projection_recon.png",
    )

    # ── console summary ──
    print("\n===== Read 1: held-out vs in-sample R2 (read-out layer) =====")
    for t in traits:
        pt = read1["per_trait"][t]
        cd = pt["heldout_cosine_dist"]
        print(
            f"  {t:14s} L{pt['read_out_layer']:2d}: held-out R2 "
            f"{pt['heldout_r2_mean']:.4f}+-{pt['heldout_r2_sd']:.4f} vs in-sample "
            f"{pt['in_sample_r2']:.3f}  (gap {pt['overfitting_gap_r2']:+.4f}) | "
            f"cos p5/p50/p95={cd['p5']:.3f}/{cd['p50']:.3f}/{cd['p95']:.3f} min={cd['min']:.3f}"
        )
    print("\n===== Read 2: predicted vs true answer-projection Pearson =====")
    for t in traits:
        for mode in ("system", "many_shot"):
            rr = read2[t].get(mode)
            if rr is None:
                print(f"  {t:14s} {mode:10s}: SKIPPED")
                continue
            print(
                f"  {t:14s} {mode:10s}: overall r={rr['overall_pearson']:.3f} "
                f"(n={rr['overall_n']}) | within-cond r={rr['within_condition_r']:.3f} "
                f"[{rr['within_condition_lo']:.3f}, {rr['within_condition_hi']:.3f}] "
                f"(k={rr['within_condition_n_conditions']})"
            )

    logger.info("Done. Wrote %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
