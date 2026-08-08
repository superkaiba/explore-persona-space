"""Body-round figure regeneration for task #1901 (metric-characterization battery).

Regenerates the clean-result-embedded subset of the driver-rendered figures via
the `/paper-plots` conventions (`set_paper_style("blog")` + `savefig_paper` →
PNG + PDF + `.meta.json` sidecars, colorblind-safe Wong palette), under NEW
filenames per the #1482 precedent. Reads only the committed battery JSONs:

    eval_results/issue_1901/metric_battery/{context_arm,prefix_arm}.json
    eval_results/issue_1901/metric_battery/boot_draws_context.json

Run from the issue-1901 worktree root:
    uv run python scripts/issue1901_body_figures.py
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps BEFORE numpy/matplotlib (shared-VM rule, #847)

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
MB = ROOT / "eval_results" / "issue_1901" / "metric_battery"
OUT = ROOT / "figures" / "issue_1901"

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


def main() -> None:
    set_paper_style("blog")
    l19, p18, boot19 = _load()
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
