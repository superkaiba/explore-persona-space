#!/usr/bin/env python3
"""#722 Result-4 behavior-chain plots WITH 95% bootstrap CIs.

Regenerates ``figures/issue_722/result4_behavior_from_vA_vs_MvC.png`` (the
grouped-bar best-layer LOCO Spearman ρ for predicting the judged behavior rate
B from the answer profile v_A vs the c_C→v_A LOCO-predicted answer profile
M·v_C) with 95% CI error bars, AND a companion scatter
``result4_scatter_predicted_vs_judged.png`` showing the underlying 50-context
held-out predictions behind each aggregate ρ.

REUSE contract — keeps the point estimates byte-consistent with the committed
``eval_results/issue_722/structural/behavior_chain_ridge_readout.json``:
imports the SAME readout machinery from ``issue722_behavior_chain_ridge_readout``
(``load_substrate``, ``ridge_readout_loco_direct``, ``ridge_readout_loco_mediated``,
PCA-48 + ``ridge_predict_loco_centered`` for M). 0-GPU; CPU bootstrap resamples
the FINAL (held-out-prediction, E0) pairs — no refit per draw.

CI method: for each behavior, at the behavior's own best layer (the same layer
the committed point estimate uses), compute the per-context LOCO held-out
predictions for BOTH the direct readout (ridge on true v_A) and the mediated
readout (ridge readout applied to M·v_C). Bootstrap over the 50 contexts:
resample the 50 (held-out-pred, E0) pairs with replacement B=2000 times,
recompute Spearman ρ each draw, take the 2.5/97.5 percentiles.

Best-layer caveat: bootstrapping at a PRE-SELECTED best layer is slightly
optimistic (the layer was chosen on the same data). A fixed-layer (L18) CI is
ALSO reported as a cleaner, selection-free alternative.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")  # inputs (data/E0/JSON)
OUT_ROOT = Path(__file__).resolve().parent.parent  # worktree root — figures land here
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue722_behavior_chain_ridge_readout import (  # noqa: E402
    PCA_TARGET_DIM,
    e0_vector,
    load_substrate,
    ridge_readout_loco_direct,
    ridge_readout_loco_mediated,
)

from explore_persona_space.analysis import paper_plots  # noqa: E402
from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
    robust_pca_basis,
)

# 3 behaviors shown — broad_em excluded (E0 floor-saturated per task brief).
BEHAVIORS = ["sycophancy", "refusal", "harmful_compliance"]
BEHAVIOR_LABEL = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "harmful_compliance": "harmful\ncompliance",
}
FIXED_LAYER = 18
N_BOOT = 2000
BOOT_SEED = 722

# best layers from the committed JSON (sanity-checked at runtime against the
# recomputed per-layer ρ — assert match within 2e-2).
EXPECTED_POINT = {
    "sycophancy": {"direct": 0.855, "mediated": 0.656, "L_direct": 18, "L_mediated": 21},
    "refusal": {"direct": 0.424, "mediated": 0.669, "L_direct": 19, "L_mediated": 20},
    "harmful_compliance": {"direct": 0.507, "mediated": 0.587, "L_direct": 2, "L_mediated": 12},
}

# Wong-palette roles matching the original figure (#0072B2 blue, #009E73 green).
C_DIRECT = "#0072B2"  # ridge from true v_A
C_MEDIATED = "#009E73"  # ridge from M v_C (predicted v_A)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    r, _ = spearmanr(a, b)
    return float(r)


def _boot_ci(pred: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Bootstrap CI for Spearman ρ over the n contexts (resample final pairs)."""
    n = len(y)
    rhos = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        a, c = pred[idx], y[idx]
        if np.std(a) < 1e-12 or np.std(c) < 1e-12:
            rhos[b] = np.nan
            continue
        rhos[b] = _spearman(a, c)
    rhos = rhos[~np.isnan(rhos)]
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return {"point": _spearman(pred, y), "lo": float(lo), "hi": float(hi)}


def _preds_at_layer(sub, layer: int, kept_idx, y_e0):
    """Return (direct_preds, mediated_preds) held-out at `layer` for kept rows."""
    V, C = sub["V"], sub["C"]
    Y = V[:, layer, :]
    Xc = C[:, layer, :]
    mu, comps, _fb = robust_pca_basis(Y, PCA_TARGET_DIM)
    Y_pca = (Y - mu) @ comps.T  # true PCA48 coords
    pred_pca = ridge_predict_loco_centered(Xc, Y_pca)  # LOCO c_C->PCA48(v_A)
    Y_pca_k = Y_pca[kept_idx]
    pred_pca_k = pred_pca[kept_idx]
    predA = ridge_readout_loco_direct(Y_pca_k, y_e0)  # ridge on true v_A
    predB = ridge_readout_loco_mediated(Y_pca_k, pred_pca_k, y_e0)  # ridge on M v_C
    return predA, predB


def main():
    sub = load_substrate()
    print(f"[info] c_C variant: {sub['cc_variant']}")
    print(f"[info] E0 source:   {sub['e0_src']}")
    rng = np.random.default_rng(BOOT_SEED)

    # locate best layers from the committed JSON (the existing point estimate).
    committed = json.loads(
        (
            PROJECT_ROOT / "eval_results/issue_722/structural/behavior_chain_ridge_readout.json"
        ).read_text()
    )

    results = {}  # behavior -> {"best": {...}, "fixed": {...}, "scatter": {...}}
    for b in BEHAVIORS:
        cb = committed["per_behavior"][b]
        L_direct = cb["best_ridge_direct"]["layer"]
        L_mediated = cb["best_ridge_mediated"]["layer"]
        y_e0, kept_idx = e0_vector(sub["e0"], b, sub["ctx_ids"])
        y_e0 = np.asarray(y_e0, dtype=np.float64)
        assert len(kept_idx) == 50, (b, len(kept_idx))

        # DIRECT at its own best layer
        predA_bd, _ = _preds_at_layer(sub, L_direct, kept_idx, y_e0)
        # MEDIATED at its own best layer
        _, predB_bm = _preds_at_layer(sub, L_mediated, kept_idx, y_e0)
        ci_direct = _boot_ci(predA_bd, y_e0, rng)
        ci_mediated = _boot_ci(predB_bm, y_e0, rng)

        # sanity vs committed values
        exp = EXPECTED_POINT[b]
        assert abs(ci_direct["point"] - exp["direct"]) < 2e-2, (
            b,
            "direct",
            ci_direct["point"],
            exp["direct"],
        )
        assert abs(ci_mediated["point"] - exp["mediated"]) < 2e-2, (
            b,
            "mediated",
            ci_mediated["point"],
            exp["mediated"],
        )

        # FIXED layer L18 (selection-free): both readouts at L18
        predA_f, predB_f = _preds_at_layer(sub, FIXED_LAYER, kept_idx, y_e0)
        ci_direct_f = _boot_ci(predA_f, y_e0, rng)
        ci_mediated_f = _boot_ci(predB_f, y_e0, rng)

        results[b] = {
            "L_direct": L_direct,
            "L_mediated": L_mediated,
            "best": {"direct": ci_direct, "mediated": ci_mediated},
            "fixed_L18": {"direct": ci_direct_f, "mediated": ci_mediated_f},
            # scatter uses the BEST-layer held-out predictions (the aggregate ρ
            # in the bar is the best-layer one).
            "scatter": {
                "y_e0": y_e0.tolist(),
                "pred_direct": predA_bd.tolist(),
                "pred_mediated": predB_bm.tolist(),
            },
        }
        print(
            f"  {b:20s} direct ρ={ci_direct['point']:.3f} "
            f"[{ci_direct['lo']:.3f},{ci_direct['hi']:.3f}]@L{L_direct}  "
            f"mediated ρ={ci_mediated['point']:.3f} "
            f"[{ci_mediated['lo']:.3f},{ci_mediated['hi']:.3f}]@L{L_mediated}  "
            f"| L18: direct {ci_direct_f['point']:.3f}"
            f"[{ci_direct_f['lo']:.3f},{ci_direct_f['hi']:.3f}] "
            f"mediated {ci_mediated_f['point']:.3f}"
            f"[{ci_mediated_f['lo']:.3f},{ci_mediated_f['hi']:.3f}]"
        )

    # context families (f1..f8, 7 families) for scatter coloring
    fams = [c.split("_")[0] for c in sub["ctx_ids"]]
    fam_order = sorted(set(fams))
    fam_label = {
        "f1": "f1 housing/persona",
        "f2": "f2 word-count",
        "f3": "f3 in-context",
        "f4": "f4 rephrasing",
        "f5": "f5 format",
        "f6": "f6 default/helpful",
        "f8": "f8 behavioral",
    }
    fam_palette = paper_plots._PALETTE  # Wong, 8 colors → 7 families fit

    _make_bar_figure(results)
    _make_scatter_figure(results, fams, fam_order, fam_label, fam_palette)
    _write_results_json(results)


def _make_bar_figure(results):
    paper_plots.set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(BEHAVIORS))
    w = 0.38
    pts_d = [results[b]["best"]["direct"] for b in BEHAVIORS]
    pts_m = [results[b]["best"]["mediated"] for b in BEHAVIORS]
    err_d = np.array([[p["point"] - p["lo"] for p in pts_d], [p["hi"] - p["point"] for p in pts_d]])
    err_m = np.array([[p["point"] - p["lo"] for p in pts_m], [p["hi"] - p["point"] for p in pts_m]])
    ax.bar(
        x - w / 2,
        [p["point"] for p in pts_d],
        w,
        yerr=err_d,
        capsize=4,
        color=C_DIRECT,
        label=r"ridge from true $v_A$",
        error_kw={"ecolor": "#222", "elinewidth": 1.3},
    )
    ax.bar(
        x + w / 2,
        [p["point"] for p in pts_m],
        w,
        yerr=err_m,
        capsize=4,
        color=C_MEDIATED,
        label=r"ridge from $M\,v_C$ (predicted $v_A$)",
        error_kw={"ecolor": "#222", "elinewidth": 1.3},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABEL[b] for b in BEHAVIORS])
    ax.set_ylabel(r"held-out $\rho$(predicted $B$, judged $B$)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        r"Predicting behavior $B$ from $v_A$ vs $M\,v_C$ (ridge readout)",
        pad=34,
    )
    ax.text(
        0.0,
        1.035,
        "best-layer held-out Spearman ρ, 95% CI (2000-bootstrap over 50 contexts) · "
        "Qwen2.5-7B · Betley · n=50 · LOCO",
        transform=ax.transAxes,
        fontsize=9,
        color="#555",
    )
    ax.legend(loc="upper right", frameon=False)
    fig.subplots_adjust(top=0.86)
    paper_plots.savefig_paper(
        fig,
        "result4_behavior_from_vA_vs_MvC",
        dir=str(OUT_ROOT / "figures/issue_722"),
    )
    plt.close(fig)
    print("[info] wrote result4_behavior_from_vA_vs_MvC.png")


def _make_scatter_figure(results, fams, fam_order, fam_label, fam_palette):
    paper_plots.set_paper_style("blog")
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 11.0), sharex=False, sharey=False)
    col_title = {0: r"from true $v_A$", 1: r"from $M\,v_C$ (predicted $v_A$)"}
    fam_color = {f: fam_palette[i % len(fam_palette)] for i, f in enumerate(fam_order)}
    for r, b in enumerate(BEHAVIORS):
        sc = results[b]["scatter"]
        y = np.array(sc["y_e0"])
        preds = {0: np.array(sc["pred_direct"]), 1: np.array(sc["pred_mediated"])}
        rho = {0: results[b]["best"]["direct"]["point"], 1: results[b]["best"]["mediated"]["point"]}
        layer = {0: results[b]["L_direct"], 1: results[b]["L_mediated"]}
        for c in (0, 1):
            ax = axes[r, c]
            p = preds[c]
            for i in range(len(y)):
                ax.scatter(
                    y[i],
                    p[i],
                    s=34,
                    color=fam_color[fams[i]],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            lo = min(y.min(), p.min())
            hi = max(y.max(), p.max())
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                color="#999",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            label = b.replace("_", " ")
            ax.set_title(f"{label} · {col_title[c]}\nρ={rho[c]:.3f} (L{layer[c]})", fontsize=11)
            if r == 2:
                ax.set_xlabel(r"judged behavior rate $B$ (E0)")
            if c == 0:
                ax.set_ylabel(r"LOCO held-out predicted $B$")
    # legend for context families (shared, below)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=fam_color[f],
            markeredgecolor="white",
            label=fam_label.get(f, f),
        )
        for f in fam_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(fam_order),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        r"Underlying 50 contexts behind each best-layer $\rho$: predicted vs judged $B$",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    paper_plots.savefig_paper(
        fig,
        "result4_scatter_predicted_vs_judged",
        dir=str(OUT_ROOT / "figures/issue_722"),
    )
    plt.close(fig)
    print("[info] wrote result4_scatter_predicted_vs_judged.png")


def _write_results_json(results):
    out = {
        "analysis": "result4_behavior_chain_ridge_readout_with_CI",
        "ci_method": (
            "bootstrap over 50 contexts, resample final (held-out-pred, E0) pairs "
            "with replacement, B=2000, recompute Spearman ρ, 2.5/97.5 percentiles"
        ),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "fixed_layer": FIXED_LAYER,
        "caveat": (
            "best-layer CI is slightly optimistic (layer chosen on the same data); "
            "fixed_L18 CI is the selection-free alternative"
        ),
        "per_behavior": {
            b: {k: results[b][k] for k in ("L_direct", "L_mediated", "best", "fixed_L18")}
            for b in BEHAVIORS
        },
    }
    p = OUT_ROOT / "eval_results/issue_722/structural/result4_ci.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"[info] wrote {p}")


if __name__ == "__main__":
    main()
