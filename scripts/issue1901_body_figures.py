"""Body-round figure regeneration for task #1901 (metric-characterization battery).

Regenerates the clean-result-embedded subset of the driver-rendered figures via
the `/paper-plots` conventions (`set_paper_style("blog")` + `savefig_paper` →
PNG + PDF + `.meta.json` sidecars, colorblind-safe Wong palette), under NEW
filenames per the #1482 precedent. Reads only the committed battery JSONs:

    eval_results/issue_1901/metric_battery/{context_arm,prefix_arm}.json
    eval_results/issue_1901/metric_battery/boot_draws_context.json

The --style iclr paper pathway additionally reads the #1901 paper_densify
JSONs (eval_results/issue_1901/paper_densify/{layer_curve_n3600,
scaling_ladder_L19, layer_curve_n50k, scaling_bigN_acc1_L19,
mlp_layer_curve_n3600, mlp_scaling_L19}.json) plus the banked #779/#1491
large-n fits.

Run from the issue-1901 worktree root:
    uv run python scripts/issue1901_body_figures.py
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps BEFORE numpy/matplotlib (shared-VM rule, #847)

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    figsize_iclr_panels,
    paper_color,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
MB = ROOT / "eval_results" / "issue_1901" / "metric_battery"
OUT = ROOT / "figures" / "issue_1901"
PAPER_OUT = ROOT / "figures" / "paper"
PD = ROOT / "eval_results" / "issue_1901" / "paper_densify"
EV779 = ROOT / "eval_results" / "issue_779"

POOLS = ["test", "passb_5000", "distr_20000", "distr_100000"]
POOL_N = {"test": 1000, "passb_5000": 5000, "distr_20000": 20000, "distr_100000": 100000}

# One color = one meaning: estimator colors are constant across every figure.
_ = paper_palette  # paper-plots API import kept; hexes pinned by VALUE (order is style-dependent)
EST_COLOR = {
    "const_mean": "#444444",
    "identity_copy": "#999999",
    "identity_bias": "#009E73",  # green
    "ridge": "#0072B2",  # blue
    "mlp_w8192": "#E69F00",  # orange
    "mlp_w32768": "#D55E00",  # vermillion
    "krr_nystrom": "#CC79A7",  # reddish purple
    "diagonal_only": "#F0E442",
    "scaled_identity": "#56B4E9",
}
EST_LABEL = {
    "const_mean": "constant train-mean",
    "identity_copy": "identity copy",
    "identity_bias": "identity + bias",
    "ridge": "linear map (ridge)",
    "mlp_w8192": "neural map (w=8192)",
    "mlp_w32768": "neural map (w=32768)",
    "krr_nystrom": "kernel map (Nystrom)",
}
LADDER_963K = list(EST_LABEL)


SOM_722 = ROOT / "eval_results" / "issue_722" / "base-skill-over-mean-cC-to-v0"


def _load() -> tuple[dict, dict, dict]:
    ctx = json.loads((MB / "context_arm.json").read_text())
    pfx = json.loads((MB / "prefix_arm.json").read_text())
    boot = json.loads((MB / "boot_draws_context.json").read_text())
    return ctx["per_layer"]["19"], pfx["per_layer"]["18"], boot["19"]


def _acc1(arm: dict, pool: str, metric: str = "euclidean") -> tuple[float, float, float]:
    e = arm["retrieval"][pool][metric]
    return e["acc_at_k"]["1"], e["acc1_ci"]["lo"], e["acc1_ci"]["hi"]


def fig_hero_scatter(l19: dict, p18: dict) -> None:
    """Pooled R^2 (symlog x) vs retrieval acc@1, one labeled point per estimator x regime."""
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    regimes = {"963k": "o", "3600": "s", "50": "v", "prefix": "D"}
    pts: list[tuple[str, str, float, float, str, tuple[float, float], str]] = []
    for est in LADDER_963K:
        a = l19["arms"][est]
        pts.append(
            (est, "963k", a["r2"]["point"], _acc1(a, "test")[0], EST_LABEL[est], (0, 8), "center")
        )
    for est, key in (("identity_bias", "identity_bias_3600"), ("ridge", "ridge_3600")):
        a = l19["arms"][key]
        pts.append(
            (
                est,
                "3600",
                a["r2"]["point"],
                _acc1(a, "test")[0],
                EST_LABEL[est] + " (n=3.6k)",
                (0, -14),
                "center",
            )
        )
    for est, key in (("identity_bias", "identity_bias_n50"), ("ridge", "ridge_n50_fixedlam")):
        a = l19["arms"][key]
        pts.append(
            (
                est,
                "50",
                a["r2"]["point"],
                _acc1(a, "test")[0],
                EST_LABEL[est] + " (n=50)",
                (0, 8),
                "center",
            )
        )
    for est, key, dy in (
        ("ridge", "ridge_lofo", 8),
        ("identity_bias", "identity_bias", 8),
        ("const_mean", "const_fold_mean", 8),
        ("mlp_w8192", "mlp_lofo", -14),
    ):
        a = p18["arms"][key]
        lbl = {
            "ridge_lofo": "LOFO ridge (prefix)",
            "identity_bias": "identity + bias (prefix)",
            "const_fold_mean": "fold mean (prefix)",
            "mlp_lofo": "LOFO neural map (prefix)",
        }[key]
        pts.append(
            (
                est,
                "prefix",
                a["r2"]["point"],
                a["retrieval"]["battery50"]["euclidean"]["acc_at_k"]["1"],
                lbl,
                (0, dy),
                "center",
            )
        )
    # manual anti-overlap nudges for the dense clusters
    nudge = {
        "linear map (ridge)": (-6, 9, "right"),
        "neural map (w=8192)": (2, -16, "left"),
        "neural map (w=32768)": (6, 10, "left"),
        "kernel map (Nystrom)": (10, -4, "left"),
        "linear map (ridge) (n=3.6k)": (-8, -6, "right"),
        "identity + bias": (8, 4, "left"),
        "identity + bias (n=3.6k)": (-8, -4, "right"),
        "identity + bias (n=50)": (8, -14, "left"),
        "constant train-mean": (0, 10, "center"),
        "fold mean (prefix)": (0, -16, "center"),
        "LOFO ridge (prefix)": (0, -16, "center"),
    }
    for est, reg, x, y, lbl, (dx, dy), ha in pts:
        ax.scatter(
            [x],
            [y],
            s=95,
            marker=regimes[reg],
            color=EST_COLOR[est],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ndx, ndy, nha = nudge.get(lbl, (dx, dy, ha))
        ax.annotate(
            lbl,
            (x, y),
            xytext=(ndx, ndy),
            textcoords="offset points",
            fontsize=8.5,
            ha=nha,
            va="center",
            color="#333333",
        )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(-60, 1.6)
    ax.set_ylim(-0.06, 1.0)
    ax.axvline(0.0, color="#bbbbbb", lw=0.8, ls=":")
    ax.axhline(0.001, color="#bbbbbb", lw=0.8, ls="--")
    ax.set_xlabel("pooled held-out R-squared (symlog scale; 0 = constant-mean level)")
    ax.set_ylabel("retrieval acc@1 (euclidean; pool = 1000 context / 50 prefix)")
    ax.set_title(
        "Variance explained vs retrieval discriminability, per estimator and regime", pad=16
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker=m,
            color="#666666",
            ls="",
            markersize=8,
            label={
                "o": "context, n=963k train",
                "s": "context, n=3600",
                "v": "context, n=50",
                "D": "prefix LOFO, n~43",
            }[m],
        )
        for m in ("o", "s", "v", "D")
    ]
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=9)
    savefig_paper(fig, "hero_r2_vs_acc1_scatter_v2", dir=OUT)
    plt.close(fig)


def fig_ladder_grid(l19: dict, p18: dict) -> None:
    """Estimator x metric annotated grid, context (963k, L19) and prefix (LOFO, L18) panels."""
    cols = ["pooled R2", "mean cosine", "acc@1 (euclid)", "acc@1 (CSLS)", "MRR", "median rank"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), width_ratios=[1, 1])

    def grid_for(arms: dict, order: list[tuple[str, str]], pool: str, ret_key: str) -> np.ndarray:
        rows = []
        for key, _ in order:
            a = arms[key]
            r = a["retrieval"][pool] if ret_key == "ctx" else a["retrieval"]["battery50"]
            rows.append(
                [
                    a["r2"]["point"],
                    a["mean_cosine"]["point"],
                    r["euclidean"]["acc_at_k"]["1"],
                    r["csls"]["acc_at_k"]["1"],
                    r["euclidean"]["mrr"],
                    r["euclidean"]["median_rank"],
                ]
            )
        return np.array(rows)

    ctx_order = [(k, EST_LABEL[k]) for k in LADDER_963K]
    pfx_order = [
        ("const_fold_mean", "constant fold-mean"),
        ("identity_copy", "identity copy"),
        ("scaled_identity", "scaled identity"),
        ("diagonal_only", "per-dim rescale"),
        ("identity_bias", "identity + bias"),
        ("ridge_lofo", "LOFO ridge"),
        ("mlp_lofo", "LOFO neural map"),
    ]
    for ax, (arms, order, pool, ret_key, title) in zip(
        axes,
        [
            (
                l19["arms"],
                ctx_order,
                "test",
                "ctx",
                "context arm (963k train, layer 19, pool 1000)",
            ),
            (
                p18["arms"],
                pfx_order,
                "battery50",
                "pfx",
                "prefix arm (LOFO, n~43 train, layer 18, pool 50)",
            ),
        ],
    ):
        m = grid_for(arms, order, pool, ret_key)
        shade = np.zeros_like(m)
        for j in range(m.shape[1]):
            col = m[:, j]
            rank = col.argsort().argsort().astype(float) / max(len(col) - 1, 1)
            shade[:, j] = 1.0 - rank if j == 5 else rank  # median rank: lower is better
        ax.imshow(shade, cmap="cividis", aspect="auto", vmin=0, vmax=1, alpha=0.85)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                txt = f"{v:,.0f}" if j == 5 and v >= 10 else f"{v:.2f}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if shade[i, j] < 0.45 else "black",
                )
        ax.set_xticks(range(len(cols)), cols, rotation=25, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(order)), [lbl for _, lbl in order], fontsize=9)
        ax.set_title(title, fontsize=10.5, pad=10)
        ax.grid(False)
    fig.suptitle(
        "Every estimator on every metric: brighter = better within each column", y=1.02, fontsize=12
    )
    fig.tight_layout()
    savefig_paper(fig, "ladder_by_metric_grid_v2", dir=OUT)
    plt.close(fig)


def fig_pool_decay_gap(l19: dict) -> None:
    """acc@1 vs pool size per estimator + the paired nonlinear-minus-ridge gap per pool."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    xs = [POOL_N[p] for p in POOLS]
    for est in LADDER_963K:
        a = l19["arms"][est]
        ys = [_acc1(a, p)[0] for p in POOLS]
        ax1.plot(xs, ys, marker="o", color=EST_COLOR[est], lw=1.8, ms=4.5)
        dy = {
            "mlp_w32768": 12,
            "mlp_w8192": 2,
            "krr_nystrom": -9,
            "ridge": 0,
            "identity_bias": 0,
            "identity_copy": 0,
            "const_mean": 8,
        }[est]
        ax1.annotate(
            EST_LABEL[est],
            (xs[-1], ys[-1]),
            xytext=(6, dy),
            textcoords="offset points",
            fontsize=8.5,
            va="center",
            color=EST_COLOR[est],
        )
    ax1.plot(xs, [1.0 / n for n in xs], ls="--", color="#aaaaaa", lw=1.2)
    ax1.annotate(
        "chance = 1/pool",
        (xs[-1], 1e-5),
        xytext=(6, -9),
        textcoords="offset points",
        fontsize=8.5,
        va="center",
        color="#888888",
    )
    ax1.set_xscale("log")
    ax1.set_xlim(8e2, 1.1e6)
    ax1.set_xlabel("candidate pool size")
    ax1.set_ylabel("retrieval acc@1 (euclidean)")
    ax1.set_title("acc@1 vs pool size, all estimators", fontsize=11, pad=10)

    pc = l19["paired_contrasts"]
    series = [
        ("mlp_w8192_minus_ridge", "neural map (w=8192) - ridge", EST_COLOR["mlp_w8192"]),
        ("mlp_w32768_minus_ridge", "neural map (w=32768) - ridge", EST_COLOR["mlp_w32768"]),
        ("krr_nystrom_minus_ridge", "kernel map - ridge", EST_COLOR["krr_nystrom"]),
    ]
    for off, (key, lbl, col) in zip((0.88, 1.0, 1.14), series):
        c = pc[key]
        ys = [c[f"acc1_euclid_{p}"]["mean"] for p in POOLS]
        los = [ys[i] - c[f"acc1_euclid_{p}"]["lo"] for i, p in enumerate(POOLS)]
        his = [c[f"acc1_euclid_{p}"]["hi"] - ys[i] for i, p in enumerate(POOLS)]
        xj = [x * off for x in xs]
        ax2.errorbar(
            xj, ys, yerr=[los, his], marker="o", color=col, lw=1.6, ms=4.5, capsize=3, label=lbl
        )
    ax2.axhline(0.0, color="#bbbbbb", lw=0.9, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("candidate pool size")
    ax2.set_ylabel("paired acc@1 gap over ridge (95% bootstrap CI)")
    ax2.set_title("nonlinear-over-linear retrieval gap grows with pool size", fontsize=11, pad=10)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    savefig_paper(fig, "pool_decay_and_nonlinear_gap", dir=OUT)
    plt.close(fig)


def fig_rank_cdf(boot19: dict) -> None:
    """Per-test-row rank CDF at the 100k pool: the per-unit data behind acc@1."""
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for est in LADDER_963K:
        ranks = np.array(
            boot19[est]["retrieval"]["distr_100000"]["euclidean"]["obs_ranks"], dtype=float
        )
        ranks = np.sort(ranks)
        cdf = np.arange(1, len(ranks) + 1) / len(ranks)
        ax.step(ranks, cdf, where="post", color=EST_COLOR[est], lw=1.8, label=EST_LABEL[est])
    ax.set_xscale("log")
    ax.set_xlim(1, 1.1e5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("rank of the true answer vector in the 100,000-candidate pool (log scale)")
    ax.set_ylabel("fraction of the 1000 test rows at or below this rank")
    ax.set_title("Per-row rank distribution at the 100k pool (the data behind acc@1)", pad=14)
    ax.legend(frameon=False, fontsize=8.5, loc="center", bbox_to_anchor=(0.72, 0.42))
    savefig_paper(fig, "rank_cdf_pool100k", dir=OUT)
    plt.close(fig)


def fig_null_floors(l19: dict, boot19: dict) -> None:
    """Observed value vs 200-draw shuffled-pair null, for R^2, mean cosine, retrieval acc@1."""
    arms = ["const_mean", "identity_copy", "identity_bias", "ridge", "mlp_w32768"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    panels = [
        ("r2_null", "r2", "pooled R-squared"),
        ("cos_null", "mean_cosine", "mean cosine"),
        (None, None, "retrieval acc@1 (pool 1000)"),
    ]
    for pi, (ax, (nkey, okey, title)) in enumerate(zip(axes, panels)):
        for i, est in enumerate(arms):
            if pi < 2:
                null = np.array(boot19[est][nkey], dtype=float)
                obs = l19["arms"][est][okey]["point"]
            else:
                null = np.array(
                    boot19[est]["retrieval"]["test"]["euclidean"]["acc1_null"], dtype=float
                )
                obs = _acc1(l19["arms"][est], "test")[0]
            vp = ax.violinplot([null], positions=[i], widths=0.62, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor("#bbbbbb")
                b.set_alpha(0.7)
            ax.scatter(
                [i], [obs], s=80, color=EST_COLOR[est], zorder=3, edgecolor="white", linewidth=0.7
            )
        if pi == 1:
            ax.set_ylim(0.45, 1.0)
        if pi == 2:
            ax.axhline(0.001, color="#888888", lw=0.9, ls="--")
            ax.text(
                0.02,
                0.004,
                "analytic chance 1/1000",
                fontsize=8,
                color="#888888",
                transform=ax.get_yaxis_transform(),
            )
        short = {
            "const_mean": "constant\ntrain-mean",
            "identity_copy": "identity\ncopy",
            "identity_bias": "identity\n+ bias",
            "ridge": "linear map\n(ridge)",
            "mlp_w32768": "neural map\n(w=32768)",
        }
        ax.set_xticks(range(len(arms)), [short[a] for a in arms], fontsize=8.2)
        ax.set_title(title, fontsize=10.5, pad=8)
    axes[0].set_ylabel("dot = observed; violin = 200-draw null")
    fig.suptitle(
        "Shuffled-pair nulls collapse to metric-specific floors; cosine's floor is high",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    savefig_paper(fig, "null_floors", dir=OUT)
    plt.close(fig)


def fig_hub_csls(l19: dict) -> None:
    """Top-100 hub corpus composition (cosine, pool 100k) + CSLS-minus-cosine acc@1 gains."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9))
    arms = ["identity_copy", "identity_bias", "ridge", "mlp_w8192", "mlp_w32768", "krr_nystrom"]
    seg_colors = {"lmsys(test)": "#2b2b2b", "lmsys": "#8c8c8c", "wildchat": "#d9d9d9"}
    seg_label = {
        "lmsys(test)": "test targets (1% of pool)",
        "lmsys": "LMSYS distractors (56%)",
        "wildchat": "WildChat distractors (43%)",
    }
    names = ["pool composition\n(reference)"] + [EST_LABEL[a] for a in arms]
    comps = [{"lmsys(test)": 1.0, "lmsys": 55.686, "wildchat": 43.314}]
    for a in arms:
        comps.append(
            {
                k: float(v)
                for k, v in l19["arms"][a]["retrieval"]["distr_100000"]["cosine"]["hubness"][
                    "top_hub_corpus_composition"
                ].items()
            }
        )
    for yi, comp in enumerate(comps):
        left = 0.0
        for seg in ("lmsys(test)", "lmsys", "wildchat"):
            w = comp.get(seg, 0.0)
            ax1.barh(
                yi,
                w,
                left=left,
                color=seg_colors[seg],
                height=0.62,
                label=seg_label[seg] if yi == 0 else None,
            )
            left += w
    ax1.set_yticks(range(len(names)), names, fontsize=8.6)
    ax1.invert_yaxis()
    ax1.set_xlabel("share of the top-100 retrieval hubs (cosine, pool 100k), by corpus")
    ax1.set_title("whose items become hubs depends on the estimator", fontsize=10.5, pad=8)
    ax1.legend(frameon=False, fontsize=8.4, loc="lower right")

    pc = l19["paired_contrasts"]["csls_minus_cosine_acc1"]
    xs = [POOL_N[p] for p in POOLS]
    for off, est in zip((0.82, 0.9, 0.97, 1.04, 1.12, 1.21), arms):
        c = pc[est]
        ys = [c[p]["mean"] for p in POOLS]
        los = [ys[i] - c[p]["lo"] for i, p in enumerate(POOLS)]
        his = [c[p]["hi"] - ys[i] for i, p in enumerate(POOLS)]
        ax2.errorbar(
            [x * off for x in xs],
            ys,
            yerr=[los, his],
            marker="o",
            color=EST_COLOR[est],
            lw=1.5,
            ms=4,
            capsize=2.5,
            label=EST_LABEL[est],
        )
    ax2.axhline(0.0, color="#bbbbbb", lw=0.9, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("candidate pool size")
    ax2.set_ylabel("CSLS minus plain-cosine acc@1 (95% bootstrap CI)")
    ax2.set_title(
        "the hubness correction helps everywhere, weak estimators most", fontsize=10.5, pad=8
    )
    ax2.legend(frameon=False, fontsize=8.2, ncol=2, loc="upper center")
    fig.tight_layout()
    savefig_paper(fig, "hub_composition_csls", dir=OUT)
    plt.close(fig)


def fig_regime_flip(l19: dict, p18: dict) -> None:
    """Ridge vs identity+bias acc@1 across training regimes + prefix per-family R^2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), width_ratios=[1.05, 1])
    regs = ["context\nn=963k", "context\nn=3600", "context\nn=50", "prefix LOFO\nn~43"]
    xpos = np.arange(len(regs))

    def series(keys: list[tuple[str, str]]) -> tuple[list, list, list]:
        ys, lo, hi = [], [], []
        for src, key in keys:
            if src == "ctx":
                p, lo_, hi_ = _acc1(l19["arms"][key], "test")
            else:
                e = p18["arms"][key]["retrieval"]["battery50"]["euclidean"]
                p, lo_, hi_ = e["acc_at_k"]["1"], e["acc1_ci"]["lo"], e["acc1_ci"]["hi"]
            ys.append(p)
            lo.append(p - lo_)
            hi.append(hi_ - p)
        return ys, lo, hi

    ridge_keys = [
        ("ctx", "ridge"),
        ("ctx", "ridge_3600"),
        ("ctx", "ridge_n50_fixedlam"),
        ("pfx", "ridge_lofo"),
    ]
    ib_keys = [
        ("ctx", "identity_bias"),
        ("ctx", "identity_bias_3600"),
        ("ctx", "identity_bias_n50"),
        ("pfx", "identity_bias"),
    ]
    for keys, est, dx in ((ridge_keys, "ridge", -0.07), (ib_keys, "identity_bias", 0.07)):
        ys, lo, hi = series(keys)
        ax1.errorbar(
            xpos + dx,
            ys,
            yerr=[lo, hi],
            marker="o",
            ms=7,
            lw=1.8,
            capsize=3,
            color=EST_COLOR[est],
            label="fitted linear map (ridge)" if est == "ridge" else "identity + bias",
        )
    for xi, ch in zip(xpos, (0.001, 0.001, 0.001, 0.02)):
        ax1.hlines(ch, xi - 0.28, xi + 0.28, color="#888888", ls="--", lw=1.0)
    ax1.text(2.3, 0.05, "chance", fontsize=8, color="#888888")
    ax1.set_xticks(xpos, regs, fontsize=9)
    ax1.set_ylabel("retrieval acc@1 (euclidean; pool 1000 context / 50 prefix)")
    ax1.set_title(
        "the retrieval ranking flips where n is small, not where the framing changes",
        fontsize=10.5,
        pad=8,
    )
    ax1.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.02, 0.32))

    fams = sorted(p18["per_family"].items(), key=lambda kv: kv[1]["r2_ridge"])
    names = [f"{k} (n={v['n']})" for k, v in fams]
    vals = [v["r2_ridge"] for _, v in fams]
    ax2.barh(np.arange(len(fams)), vals, color=EST_COLOR["ridge"], height=0.62)
    ax2.axvline(0.0, color="#444444", lw=0.9)
    pooled = p18["arms"]["ridge_lofo"]["r2"]["point"]
    ax2.axvline(pooled, color="#d55e00", lw=1.6, ls="--")
    ax2.text(
        pooled - 0.06,
        0.45,
        f"pooled R-squared = +{pooled:.2f}",
        fontsize=8.5,
        color="#d55e00",
        ha="right",
        va="center",
    )
    ax2.set_yticks(np.arange(len(fams)), names, fontsize=9)
    ax2.set_xlabel("LOFO ridge R-squared, per prefix family (all seven at or below zero)")
    ax2.set_title(
        "positive pooled R-squared with every family at or below zero", fontsize=10.5, pad=8
    )
    fig.tight_layout()
    savefig_paper(fig, "regime_flip_and_family_breakdown", dir=OUT)
    plt.close(fig)


# ── c1_scaling boundary-token extension + §7.4 regression gate (#1901 btokctl) ──

_R2_KEY = "held-out $R^2$"
_ACC_KEY = "retrieval acc@1 (pool 1,000)"


def _normalize_meta_points(meta: dict) -> Counter:
    """§7.4 normalization: (panel value-key, _kind, n_train, value, error) tuples.

    ``series`` and ``_group`` are EXCLUDED from the tuple BY DESIGN: the
    committed baseline labels only 42/304 points (every acc@1-panel point and
    all line-kind points carry no ``series``), and ``_group`` renumbers when a
    4th series is inserted — keying on either would deterministically
    false-HALT or silently un-protect the acc@1 panel. Values rounded to 9
    decimals (the §7.4 1e-9 comparison grain). Raises on a point with no known
    panel key.
    """
    out: Counter = Counter()
    for p in meta["points"]:
        if _R2_KEY in p:
            panel, val = _R2_KEY, p[_R2_KEY]
        elif _ACC_KEY in p:
            panel, val = _ACC_KEY, p[_ACC_KEY]
        else:
            raise ValueError(f"meta point with no known panel key: {sorted(p)}")
        err = p.get("error")
        out[
            (
                panel,
                p.get("_kind"),
                round(float(p["training contexts"]), 9),
                round(float(val), 9),
                round(float(err), 9) if err is not None else None,
            )
        ] += 1
    return out


def fig_regression_gate(committed: dict, regenerated: dict, new_label: str) -> None:
    """§7.4 tuple-keyed figure-regression gate (M1) — fail loud on any drift.

    Asserts the committed meta's normalized tuple multiset EQUALS the
    regenerated meta's multiset MINUS the new-series points (identified by the
    explicit ``series == new_label`` the extended renderer threads onto every
    point it writes), and that >=1 new-series point exists. Consumers: the
    driver's fig phase (``issue1901_boundary_token_control.phase_fig``) and the
    unit-4 smoke.
    """
    inherited = {"points": [p for p in regenerated["points"] if p.get("series") != new_label]}
    n_new = len(regenerated["points"]) - len(inherited["points"])
    assert n_new > 0, (
        f"no regenerated point carries series={new_label!r} — the renderer extension "
        f"did not label its new series (unit-3 contract)"
    )
    a, b = _normalize_meta_points(committed), _normalize_meta_points(inherited)
    if a != b:
        gone = list((a - b).items())[:5]
        extra = list((b - a).items())[:5]
        raise RuntimeError(
            f"[fig] §7.4 REGRESSION GATE FAILED — inherited point multiset changed. "
            f"missing={gone} unexpected={extra}. Do NOT commit; restore with "
            f"git checkout -- figures/paper/c1_scaling_train_pool.*"
        )
    print(
        f"[fig] §7.4 regression gate PASS: {sum(a.values())} inherited points unchanged, "
        f"{n_new} new-series points"
    )


def _boundary_series_points(
    boundary: dict, boundary_label: str
) -> tuple[
    list[float], list[float], list[float], list[float], list[float], list[float], list[float]
]:
    """Aggregate boundary-token control cells into per-rung figure points.

    Returns ``(xs, r2, r2_lo, r2_hi, acc, acc_lo, acc_hi)`` — one entry per
    rung, x-sorted; the lo/hi are RAW error OFFSETS from the value (a tiny-n
    bootstrap CI can invert around the point — #1335/#547), clamped
    ``np.maximum(0, .)`` at the errorbar call sites per the matplotlib yerr
    convention. Follows the ladder conventions exactly (#1901 plan §4 fig):
    integer-draw cells (small rungs) aggregate as across-draw mean ± sd of
    ``ridge.test_r2`` / ``knn.ridge.euclidean`` acc@1 (the ``dense()``
    convention); ``draw == "prefix"`` cells (big rungs) plot the point with its
    row-level score_cell bootstrap CI (asymmetric offsets, the ``_r2ci`` /
    ``_a1ci`` convention; the cells' article-level CI is deliberately NOT drawn
    — parent-convention parity, plan §6). Fail-loud on empty cells, mixed
    layers, a small/prefix rung overlap, or a ``series_label`` mismatch.
    """
    cells = boundary["cells"]
    assert cells, "boundary dict has no cells"
    declared = boundary.get("series_label")
    assert declared is None or declared == boundary_label, (declared, boundary_label)
    layers = {int(c["layer"]) for c in cells}
    assert len(layers) == 1, f"boundary cells span layers {sorted(layers)} — expected exactly one"

    small: dict[int, list[dict]] = {}
    big: list[dict] = []
    for c in cells:
        if c["draw"] == "prefix":
            big.append(c)
        else:
            int(c["draw"])  # fail loud on an unexpected draw kind
            small.setdefault(int(c["n_train"]), []).append(c)
    big_ns = [int(c["n_train"]) for c in big]
    assert len(set(big_ns)) == len(big_ns), f"duplicate prefix rungs: {sorted(big_ns)}"
    overlap = set(small) & set(big_ns)
    assert not overlap, f"rungs carrying BOTH draw and prefix cells: {sorted(overlap)}"

    def a1c(c: dict) -> float:
        return float(c["knn"]["ridge"]["euclidean"]["acc_at_k"]["1"])

    rows: list[tuple[float, float, float, float, float, float, float]] = []
    for n, draws in small.items():
        r2s = [float(c["ridge"]["test_r2"]) for c in draws]
        a1s = [a1c(c) for c in draws]
        sd_r, sd_a = float(np.std(r2s)), float(np.std(a1s))
        rows.append((n, float(np.mean(r2s)), sd_r, sd_r, float(np.mean(a1s)), sd_a, sd_a))
    for c in big:
        n = int(c["n_train"])
        y = float(c["ridge"]["test_r2"])
        ci = c["ridge"]["bootstrap_ci"]["r2"]
        e = c["knn"]["ridge"]["euclidean"]
        a = float(e["acc_at_k"]["1"])
        aci = e["acc1_ci"]
        rows.append(
            (
                n,
                y,
                y - float(ci["lo"]),
                float(ci["hi"]) - y,
                a,
                a - float(aci["lo"]),
                float(aci["hi"]) - a,
            )
        )
    rows.sort(key=lambda r: r[0])
    cols = list(zip(*rows, strict=True))
    return tuple(list(col) for col in cols)  # type: ignore[return-value]


def _thread_meta_series_labels(series_artists: list[tuple[object, str]]) -> None:
    """Label every errorbar container + its child Line2Ds for the meta sidecar (M1).

    ``savefig_paper``'s extraction reads ``series`` off artist labels, so with
    creation-time labels only 42/304 points carry one (acc@1 containers and all
    data/cap lines are unlabeled). Called AFTER ``fig.legend`` is built from
    explicitly captured handles/labels, so relabeling never alters the rendered
    figure — legends snapshot label text at creation, and artist labels are not
    otherwise drawn.
    """
    for cont, name in series_artists:
        cont.set_label(name)
        data_line, caplines, _barlinecols = cont
        for ln in (data_line, *(caplines or ())):
            if ln is not None:
                ln.set_label(name)


def fig_paper_c1_scaling(
    l19: dict,
    ladder: dict,
    *,
    boundary: dict | None = None,
    boundary_label: str = "generic boundary token ('.')",
    boundary_hline: float | None = None,
    stem: str = "c1_scaling_train_pool",
    out_dir: Path | None = None,
    identity_label: str = "identity + bias",
    neural_label: str = "neural map (w=8,192)",
    figsize: tuple[float, float] | None = None,
    acc_label: str = "retrieval acc@1 (pool 1,000)",
    legend_rect_top: float | None = None,
) -> None:
    """ICLR paper figure (c1_linear R1), densified (#1901 paper_densify round).

    boundary: opt-in 4th series (#1901 `generic-boundary-token-control` round)
    — the loaded ``boundary_token_scaling_L19.json`` dict (unit-2 cell schema:
    ``cells[]`` with ``draw`` int 0/1/2 at small rungs / the STRING ``"prefix"``
    at big rungs, ``ridge.test_r2`` + ``ridge.bootstrap_ci``,
    ``knn.ridge.euclidean`` incl. ``acc1_ci``). Adds ONE series labeled
    ``boundary_label`` to BOTH panels: across-draw mean ± sd at small rungs,
    row-level bootstrap CI at prefix rungs (the exact ``dense()`` / ``ci_pt``
    conventions the ridge series uses); acc@1 euclidean, pool 1,000. Drawn
    FIRST on each panel so the sidecar's last-container-wins err-by-x recovery
    keeps every inherited point's ``error`` unchanged (§7.4 gate contract);
    shown LAST in the legend. With ``boundary=None`` (the default) the drawn
    figure is unchanged. Independent of the drawn content, EVERY point written
    to the ``.meta.json`` sidecar now carries an explicit ``series`` label on
    both panels (M1 — makes the §7.4 ``fig_regression_gate`` decidable), via a
    post-legend artist relabel that does not affect the rendered output.

    boundary_hline: opt-in poster variant — draws the #825 generic
    boundary-token→segment map control (instruct R^2 0.1087, single-n,
    wikitext) as a dashed reference line on the R^2 panel and saves under
    `stem` into `out_dir`; `identity_label` / `neural_label` relabel the
    identity+bias and neural-map legend entries (poster uses
    "identity + bias (baseline)" / "nonlinear (MLP)"); the default paper
    render is byte-unchanged.

    figsize: opt-in canvas override for callers rendering at a font scale the
    paper canvas was not sized for (the MATS poster runs font_scale=1.9, at
    which the paper's 2.3in-tall canvas puts the legend on top of the axes and
    clips both y labels). None keeps the paper canvas.

    acc_label: opt-in relabel of the retrieval panel's y axis. A single-line
    27-character label does not fit a canvas shortened much below ~3.2in — it
    overruns the axis and collides with the legend — so a caller that shortens
    the canvas passes the same text broken over two lines. Default is the
    paper's, byte-unchanged.

    legend_rect_top: opt-in tight_layout headroom for the figure legend. With
    boundary_hline set the legend runs to two rows, and on a canvas short
    enough the second row descends onto the retrieval panel's top y-tick
    label. Lower this to reserve more space. None keeps the paper's
    0.91 / 0.86 defaults.

    Held-out R^2 (left) + euclidean retrieval acc@1, pool 1,000 (right) against
    training contexts (log x). Ridge (blue) and identity+bias (green) are DENSE
    from the #1491 scale7_refit ladder refits (n = 50..25,000; mean +/- SD over
    3 seeded draws at n <= 2,500, one file-order-prefix draw above; scored on
    that capture's own held-out 1,000-context pool), joined by the banked
    large-n R^2 points (ridge 50k/150k/500k/963,444; identity+bias 963,444;
    95% percentile-bootstrap CIs; scored on the original round's pinned pool)
    plus the #1901 densify refits (identity+bias R^2 at 50k/150k/500k, and
    acc@1 for BOTH arms at 50k/150k/500k — layer_curve_n50k.json +
    scaling_bigN_acc1_L19.json; the 50k refit is parity-exact vs the banked
    fit, the 150k/500k refits parity-gated within 0.0011 R^2). The neural map
    (vermilion, the w=8,192 protocol arm): R^2 at 250..3,600 (D2 scaling
    curves, 3 draws), 5k/10k (#1901 mlp_scaling_L19.json), 25k (#1491 ladder),
    50k (n50k), 150k/500k/963,444 (n1m); acc@1 at 5k/10k (mlp_scaling), 25k
    (ladder), 963,444 (battery). One figure-level legend; no on-canvas
    annotations.
    """
    arms = l19["arms"]
    n50k = json.loads((EV779 / "fitter-fair-comparison-n50k" / "n50k_fits.json").read_text())
    n1m = json.loads((EV779 / "fitter-fair-comparison-n1m" / "n1m_fits.json").read_text())
    d2 = json.loads((EV779 / "fitter-fair-comparison" / "scaling_curves.json").read_text())
    f7 = json.loads(
        (
            ROOT / "eval_results" / "issue_1491" / "scale_ladder" / "fits_scale7_refit.json"
        ).read_text()
    )
    mlp_sc = json.loads((PD / "mlp_scaling_L19.json").read_text())["per_n"]
    n50k_l19 = json.loads((PD / "layer_curve_n50k.json").read_text())["per_layer"]["19"]
    bign = json.loads((PD / "scaling_bigN_acc1_L19.json").read_text())["per_point"]

    def _r2ci(pred: dict) -> tuple[float, float, float]:
        ci = pred["bootstrap_ci"]["r2"]
        return ci["point"], ci["lo"], ci["hi"]

    def _a1ci(pred: dict) -> tuple[float, float, float]:
        e = pred["retrieval"]["euclidean"]
        return e["acc_at_k"]["1"], e["acc1_ci"]["lo"], e["acc1_ci"]["hi"]

    densify_big = [
        (50_000, n50k_l19),
        (150_000, bign["lmsys_150k"]),
        (500_000, bign["lmsys_500k"]),
    ]

    def dense(est_field) -> tuple[list[int], list[float], list[float]]:
        by_n: dict[int, list[float]] = {}
        for c in ladder["cells"]:
            by_n.setdefault(int(c["n_train"]), []).append(est_field(c))
        ns = sorted(by_n)
        mean = [float(np.mean(by_n[n])) for n in ns]
        sd = [float(np.std(by_n[n])) for n in ns]
        return ns, mean, sd

    def ci_pt(pred: dict) -> tuple[float, float, float]:
        ci = pred["bootstrap_ci"]["r2"]
        return pred["whole_map_r2"], ci["lo"], ci["hi"]

    big_ridge = [
        (50_000, *ci_pt(n50k["per_predictor"]["ridge"])),
        (150_000, *ci_pt(n1m["per_point"]["lmsys_150k"]["predictors"]["ridge"])),
        (500_000, *ci_pt(n1m["per_point"]["lmsys_500k"]["predictors"]["ridge"])),
        (963_444, *ci_pt(n1m["per_point"]["mixed_1m"]["predictors"]["ridge"])),
    ]
    neural_r2: list[tuple[int, float, float, float]] = []
    d2_mlp: dict[int, list[float]] = {}
    for row in d2["curves"]["last_L19"]:
        if row["fitter"] == "mlp":
            d2_mlp.setdefault(int(row["n"]), []).append(float(row["r2"]))
    for n in sorted(d2_mlp):
        m, s = float(np.mean(d2_mlp[n])), float(np.std(d2_mlp[n]))
        neural_r2.append((n, m, m - s, m + s))
    for n in (5_000, 10_000):
        ci = mlp_sc[str(n)]["test_ci"]["r2"]
        neural_r2.append((n, ci["point"], ci["lo"], ci["hi"]))
    r25 = f7["predictors"]["mlp_w8192"]["test_r2"]
    neural_r2.append((25_000, r25, r25, r25))
    neural_r2.append((50_000, *ci_pt(n50k["per_predictor"]["mlp"])))
    for key, n in (("lmsys_150k", 150_000), ("lmsys_500k", 500_000), ("mixed_1m", 963_444)):
        neural_r2.append((n, *ci_pt(n1m["per_point"][key]["predictors"]["mlp_w8192"])))

    fig, (ax_r2, ax_acc) = plt.subplots(
        1, 2, figsize=figsize or figsize_iclr_panels(2, height_in=2.3)
    )
    col_r = paper_color("instruct")
    col_i = paper_color("identity_bias")
    col_n = paper_color("neural_map")

    series_artists: list[tuple[object, str]] = []
    bpts = _boundary_series_points(boundary, boundary_label) if boundary is not None else None
    if bpts is not None:
        bx, br2, br2_lo, br2_hi, bacc, bacc_lo, bacc_hi = bpts
        # Next unused curated paper-palette colour (colorblind-safe; the three
        # inherited series keep their paper_color concept bindings). Drawn FIRST
        # per panel — see the docstring's err-by-x note.
        col_b = next(c for c in paper_palette(8) if c not in {col_r, col_i, col_n})
        eb = ax_r2.errorbar(
            bx,
            br2,
            yerr=[np.maximum(0, br2_lo), np.maximum(0, br2_hi)],
            marker="^",
            ls="-.",
            color=col_b,
            lw=1.2,
            ms=3,
            capsize=1.5,
            label=boundary_label,
        )
        series_artists.append((eb, boundary_label))
        eb = ax_acc.errorbar(
            bx,
            bacc,
            yerr=[np.maximum(0, bacc_lo), np.maximum(0, bacc_hi)],
            marker="^",
            ls="-.",
            color=col_b,
            lw=1.2,
            ms=3,
            capsize=1.5,
        )
        series_artists.append((eb, boundary_label))

    ns, mean, sd = dense(lambda c: c["ridge"]["test_r2"])
    xs_r = ns + [p[0] for p in big_ridge]
    ys_r = mean + [p[1] for p in big_ridge]
    lo_r = sd + [p[1] - p[2] for p in big_ridge]
    hi_r = sd + [p[3] - p[1] for p in big_ridge]
    eb = ax_r2.errorbar(
        xs_r,
        ys_r,
        yerr=[np.maximum(0, lo_r), np.maximum(0, hi_r)],
        marker="o",
        ls="-",
        color=col_r,
        lw=1.4,
        ms=3,
        capsize=1.5,
        label="linear map (ridge)",
    )
    series_artists.append((eb, "linear map (ridge)"))
    ns_i, mean_i, sd_i = dense(lambda c: c["identity_bias"]["test_r2"])
    big_ib = [(n, *_r2ci(cell["identity_bias"])) for n, cell in densify_big]
    r_ib = arms["identity_bias"]["r2"]
    xs_i = ns_i + [p[0] for p in big_ib] + [963_444]
    ys_i = mean_i + [p[1] for p in big_ib] + [r_ib["point"]]
    lo_i = sd_i + [p[1] - p[2] for p in big_ib] + [r_ib["point"] - r_ib["lo"]]
    hi_i = sd_i + [p[3] - p[1] for p in big_ib] + [r_ib["hi"] - r_ib["point"]]
    eb = ax_r2.errorbar(
        xs_i,
        ys_i,
        yerr=[np.maximum(0, lo_i), np.maximum(0, hi_i)],
        marker="s",
        ls="--",
        color=col_i,
        lw=1.2,
        ms=3,
        capsize=1.5,
        label=identity_label,
    )
    series_artists.append((eb, identity_label))
    xs_n = [p[0] for p in neural_r2]
    ys_n = [p[1] for p in neural_r2]
    lo_n = [p[1] - p[2] for p in neural_r2]
    hi_n = [p[3] - p[1] for p in neural_r2]
    eb = ax_r2.errorbar(
        xs_n,
        ys_n,
        yerr=[np.maximum(0, lo_n), np.maximum(0, hi_n)],
        marker="D",
        ls=":",
        color=col_n,
        lw=1.2,
        ms=3,
        capsize=1.5,
        label=neural_label,
    )
    series_artists.append((eb, neural_label))
    ax_r2.axhline(0.0, color="black", lw=0.7, ls=":")
    ax_r2.set_ylabel("held-out $R^2$")
    ax_r2.set_ylim(-1.05, 1.0)

    def a1(c: dict, est: str) -> float:
        return float(c["knn"][est]["euclidean"]["acc_at_k"]["1"])

    ns, mean, sd = dense(lambda c: a1(c, "ridge"))
    big_a1_r = [(n, *_a1ci(cell["ridge"])) for n, cell in densify_big]
    p, lo, hi = _acc1(arms["ridge"], "test")
    eb = ax_acc.errorbar(
        ns + [q[0] for q in big_a1_r] + [963_444],
        mean + [q[1] for q in big_a1_r] + [p],
        yerr=[
            np.maximum(0, sd + [q[1] - q[2] for q in big_a1_r] + [p - lo]),
            np.maximum(0, sd + [q[3] - q[1] for q in big_a1_r] + [hi - p]),
        ],
        marker="o",
        ls="-",
        color=col_r,
        lw=1.4,
        ms=3,
        capsize=1.5,
    )
    series_artists.append((eb, "linear map (ridge)"))
    ns_i, mean_i, sd_i = dense(lambda c: a1(c, "identity_bias"))
    big_a1_i = [(n, *_a1ci(cell["identity_bias"])) for n, cell in densify_big]
    p, lo, hi = _acc1(arms["identity_bias"], "test")
    eb = ax_acc.errorbar(
        ns_i + [q[0] for q in big_a1_i] + [963_444],
        mean_i + [q[1] for q in big_a1_i] + [p],
        yerr=[
            np.maximum(0, sd_i + [q[1] - q[2] for q in big_a1_i] + [p - lo]),
            np.maximum(0, sd_i + [q[3] - q[1] for q in big_a1_i] + [hi - p]),
        ],
        marker="s",
        ls="--",
        color=col_i,
        lw=1.2,
        ms=3,
        capsize=1.5,
    )
    series_artists.append((eb, identity_label))
    n_a5 = float(mlp_sc["5000"]["knn"]["euclidean"]["acc_at_k"]["1"])
    n_a10 = float(mlp_sc["10000"]["knn"]["euclidean"]["acc_at_k"]["1"])
    n_a25 = float(f7["knn_retrieval"]["mlp_w8192"]["euclidean"]["acc_at_k"]["1"])
    p, lo, hi = _acc1(arms["mlp_w8192"], "test")
    eb = ax_acc.errorbar(
        [5_000, 10_000, 25_000, 963_444],
        [n_a5, n_a10, n_a25, p],
        yerr=[[0.0, 0.0, 0.0, max(0, p - lo)], [0.0, 0.0, 0.0, max(0, hi - p)]],
        marker="D",
        ls=":",
        color=col_n,
        lw=1.2,
        ms=3,
        capsize=1.5,
    )
    series_artists.append((eb, neural_label))
    ax_acc.axhline(0.001, color="black", lw=0.7, ls=":")
    ax_acc.set_ylabel(acc_label)
    ax_acc.set_ylim(0.0, 1.0)

    for ax in (ax_r2, ax_acc):
        ax.set_xscale("log")
        ax.set_xlabel("training contexts")

    if boundary_hline is not None:
        ax_r2.axhline(
            boundary_hline,
            color="#666666",
            lw=1.1,
            ls=(0, (4, 2)),
            label="generic boundary-token map",
        )
    handles, labels = ax_r2.get_legend_handles_labels()
    if bpts is not None:
        # Drawn first (sidecar err-by-x ordering) — shown LAST in the legend.
        bi = labels.index(boundary_label)
        handles.append(handles.pop(bi))
        labels.append(labels.pop(bi))
    two_row = boundary_hline is not None or bpts is not None
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2 if two_row else 3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )
    # AFTER the legend snapshot: thread `series` onto every sidecar point (M1).
    _thread_meta_series_labels(series_artists)
    _rect_top = legend_rect_top if legend_rect_top is not None else (0.86 if two_row else 0.91)
    fig.tight_layout(rect=(0, 0, 1, _rect_top))
    dest = out_dir if out_dir is not None else PAPER_OUT
    dest.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, stem, dir=dest)
    plt.close(fig)


def fig_paper_c1_layer_profile(
    ctx_per_layer: dict, dense: dict, dense50k: dict, mlp36: dict
) -> None:
    """ICLR paper figure (c1_linear layer-profile), densified (#1901 paper_densify).

    Line encoding: COLOR = estimator (blue ridge / green identity+bias /
    vermilion neural map), LINESTYLE = training size (solid = n=50,000,
    dashed = n=3,600). Left (held-out R^2 per layer): the DENSE 28-layer
    instruct curves — ridge + identity+bias at n=50,000 (layer_curve_n50k,
    solid; L19 ridge parity-exact vs the banked n50k fit) and at n=3,600
    (pass_b refit, dashed; parity-gated at machine precision against the
    banked fair_comparison per-layer values), the 28-layer NEURAL (w=8,192)
    curve at n=3,600 (mlp_layer_curve_n3600, dashed vermilion; parity-gated
    vs the banked 3,600-row mlp fits at L19/L26), identity+bias negative at
    every layer at both sizes (values below the axis floor run off-canvas),
    the 963,444-row instruct ridge + wide-neural points at the three captured
    layers (14/19/26, 95% bootstrap CIs), and the #722 base-model 50-context
    sweep demoted to a light-gray reference line. Right (euclidean acc@1,
    pool 1,000): the same six dense curves' acc@1 plus the 963,444-row ridge
    points at 14/19/26 — per-layer acc@1 now exists at EVERY layer at both
    training sizes. One figure-level legend; no on-canvas annotations.
    Supersedes the n=3,600-only render (same stem).
    """
    som = json.loads((SOM_722 / "skill_over_mean.json").read_text())
    rows = sorted(som["per_layer"], key=lambda p: p["layer"])
    b_layers = [p["layer"] for p in rows]
    b_ridge = [p["skill_vs_mean_ridge"] for p in rows]

    dl = dense["per_layer"]
    k50 = dense50k["per_layer"]
    ml = mlp36["per_layer"]
    layers28 = list(range(28))
    d_r2 = [dl[str(li)]["ridge"]["test_r2"] for li in layers28]
    i_r2 = [dl[str(li)]["identity_bias"]["test_r2"] for li in layers28]
    d_a1 = [dl[str(li)]["knn"]["ridge"]["euclidean"]["acc_at_k"]["1"] for li in layers28]
    i_a1 = [dl[str(li)]["knn"]["identity_bias"]["euclidean"]["acc_at_k"]["1"] for li in layers28]
    k_r2 = [k50[str(li)]["ridge"]["whole_map_r2"] for li in layers28]
    ki_r2 = [k50[str(li)]["identity_bias"]["whole_map_r2"] for li in layers28]
    k_a1 = [k50[str(li)]["ridge"]["retrieval"]["euclidean"]["acc_at_k"]["1"] for li in layers28]
    ki_a1 = [
        k50[str(li)]["identity_bias"]["retrieval"]["euclidean"]["acc_at_k"]["1"] for li in layers28
    ]
    m_r2 = [ml[str(li)]["test_r2"] for li in layers28]
    m_a1 = [ml[str(li)]["knn"]["euclidean"]["acc_at_k"]["1"] for li in layers28]

    cap_layers = [14, 19, 26]

    def _r2pt(arm_key: str, lay: int) -> tuple[float, float, float]:
        r = ctx_per_layer[str(lay)]["arms"][arm_key]["r2"]
        return r["point"], r["lo"], r["hi"]

    def _a1pt(arm_key: str, lay: int) -> tuple[float, float, float]:
        return _acc1(ctx_per_layer[str(lay)]["arms"][arm_key], "test")

    fig, (ax_r2, ax_acc) = plt.subplots(1, 2, figsize=figsize_iclr_panels(2, height_in=2.45))
    col_r = paper_color("instruct")
    col_i = paper_color("identity_bias")
    col_n = paper_color("neural_map")

    # (label, r2 series, a1 series, color, marker, linestyle, lw, ms, zorder)
    curves = [
        ("linear map (ridge), n=50,000", k_r2, k_a1, col_r, "o", "-", 1.3, 2.6, 4),
        ("linear map (ridge), n=3,600", d_r2, d_a1, col_r, "o", "--", 1.0, 2.2, 3),
        ("identity + bias, n=50,000", ki_r2, ki_a1, col_i, "s", "-", 1.1, 2.4, 2),
        ("identity + bias, n=3,600", i_r2, i_a1, col_i, "s", "--", 0.9, 2.0, 2),
        ("neural map (w=8,192), n=3,600", m_r2, m_a1, col_n, "D", "--", 1.0, 2.2, 3),
    ]
    ax_r2.plot(
        b_layers, b_ridge, lw=1.0, color="#bbbbbb", zorder=1, label="linear map, base model (n=50)"
    )
    for lbl, r2s, a1s, col, mk, ls, lw, ms, zo in curves:
        ax_r2.plot(layers28, r2s, marker=mk, ms=ms, lw=lw, ls=ls, color=col, zorder=zo, label=lbl)
        ax_acc.plot(layers28, a1s, marker=mk, ms=ms, lw=lw, ls=ls, color=col, zorder=zo)
    for arm_key, mk, lbl, col in (
        ("ridge", "D", "linear map, n=963k", col_r),
        ("mlp_w32768", "^", "neural map (w=32,768), n=963k", col_n),
    ):
        pts = [_r2pt(arm_key, la) for la in cap_layers]
        ax_r2.errorbar(
            cap_layers,
            [p[0] for p in pts],
            yerr=[
                np.maximum(0, [p[0] - p[1] for p in pts]),
                np.maximum(0, [p[2] - p[0] for p in pts]),
            ],
            marker=mk,
            ls="",
            color=col,
            ms=4.5,
            capsize=2,
            markerfacecolor="white",
            label=lbl,
        )
    ax_r2.axhline(0.0, color="black", lw=0.7, ls=":")
    ax_r2.set_ylabel("held-out $R^2$")
    ax_r2.set_ylim(-1.05, 1.0)

    pts = [_a1pt("ridge", la) for la in cap_layers]
    ax_acc.errorbar(
        cap_layers,
        [p[0] for p in pts],
        yerr=[
            np.maximum(0, [p[0] - p[1] for p in pts]),
            np.maximum(0, [p[2] - p[0] for p in pts]),
        ],
        marker="D",
        ls="",
        color=col_r,
        ms=4.5,
        capsize=2,
        markerfacecolor="white",
    )
    ax_acc.axhline(0.001, color="black", lw=0.7, ls=":")
    ax_acc.set_ylabel("retrieval acc@1 (pool 1,000)")
    ax_acc.set_ylim(0.0, 1.0)

    for ax in (ax_r2, ax_acc):
        ax.set_xlim(-0.8, 27.8)
        ax.set_xticks([0, 7, 14, 19, 26])
        ax.set_xlabel("layer (of 28)")

    handles, labels = ax_r2.get_legend_handles_labels()
    order = [1, 2, 3, 4, 5, 0, 6, 7]  # column-major pairs: ridge, idb, neural/base, 963k points
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncol=4,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.8,
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "c1_layer_profile", dir=PAPER_OUT)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=("blog", "iclr"), default="blog")
    args = ap.parse_args()

    l19, p18, boot19 = _load()
    if args.style == "iclr":
        # Paper pathway (#2094 precedent): ICLR-styled figures under new stems in
        # figures/paper/ — never overwrites the blog-styled issue stems.
        set_paper_style("iclr")
        dense_ladder = json.loads((PD / "scaling_ladder_L19.json").read_text())
        fig_paper_c1_scaling(l19, dense_ladder)
        ctx_all = json.loads((MB / "context_arm.json").read_text())["per_layer"]
        dense_layer = json.loads((PD / "layer_curve_n3600.json").read_text())
        dense_50k = json.loads((PD / "layer_curve_n50k.json").read_text())
        mlp_layer = json.loads((PD / "mlp_layer_curve_n3600.json").read_text())
        fig_paper_c1_layer_profile(ctx_all, dense_layer, dense_50k, mlp_layer)
        print(
            "done:",
            sorted(p.name for p in PAPER_OUT.glob("c1_scaling_*")),
            sorted(p.name for p in PAPER_OUT.glob("c1_layer_profile*")),
        )
        return
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_hero_scatter(l19, p18)
    fig_ladder_grid(l19, p18)
    fig_pool_decay_gap(l19)
    fig_rank_cdf(boot19)
    fig_null_floors(l19, boot19)
    fig_hub_csls(l19)
    fig_regime_flip(l19, p18)
    print("done:", sorted(p.name for p in OUT.glob("*_v2*")))


if __name__ == "__main__":
    main()
