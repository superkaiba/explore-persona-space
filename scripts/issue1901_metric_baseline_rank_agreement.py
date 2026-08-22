"""Estimator x metric comparison + cross-metric rank agreement for task #1901.

Answers two questions the committed #1901 figures do not cover directly:

1. How does EVERY estimator (including the scaled-identity and per-dim-rescale
   baselines, which live only in the 3,600-row arms) score on EVERY metric?
   `figures/issue_1901/ladder_by_metric_grid_v2.png` shows only the seven
   963k-trained arms in its context panel.
2. How much does the estimator RANKING change when the metric changes?
   Quantified with pairwise Kendall tau-b between the metric-induced rankings
   plus a tie-corrected Kendall's W (coefficient of concordance) over all
   rankings at once.

Reads ONLY the committed battery JSON (no refit, no model calls):

    eval_results/issue_1901/metric_battery/context_arm.json

Context arm, layer 19, candidate pool = the 1,000-row held-out test pool (the
only pool every one of the 14 estimators is evaluated at).

Run from the repo root:
    uv run python scripts/issue1901_metric_baseline_rank_agreement.py
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps BEFORE numpy/matplotlib (shared-VM rule, #847)

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

ROOT = Path(__file__).resolve().parents[1]
MB = ROOT / "eval_results" / "issue_1901" / "metric_battery"
OUT = ROOT / "figures" / "issue_1901"

LAYER = "19"
POOL = "test"  # n_pool = 1000; the only pool all 14 arms share

# (arm key, display label, training-regime tag, is-a-baseline-Thomas-already-uses)
ARMS: list[tuple[str, str, str, bool]] = [
    ("ridge", "linear map (ridge)", "963k", False),
    ("mlp_w8192", "neural map (w=8192)", "963k", False),
    ("mlp_w32768", "neural map (w=32768)", "963k", False),
    ("krr_nystrom", "kernel map (Nystrom RBF)", "963k", False),
    ("identity_bias", "identity + learned bias", "963k", False),
    ("identity_copy", "identity copy", "963k", True),
    ("const_mean", "constant train-mean", "963k", False),
    ("ridge_3600", "linear map (ridge)", "3600", False),
    ("identity_bias_3600", "identity + learned bias", "3600", False),
    ("diagonal_only_3600", "per-dim rescale (diagonal)", "3600", True),
    ("scaled_identity_3600", "scaled identity (one scalar)", "3600", True),
    ("const_mean_3600", "constant train-mean", "3600", False),
    ("ridge_n50_fixedlam", "linear map (ridge)", "n=50", False),
    ("identity_bias_n50", "identity + learned bias", "n=50", False),
]

# (column key, display label, higher-is-better)
METRICS: list[tuple[str, str, bool]] = [
    ("r2", "pooled $R^2$", True),
    ("perdim_r2_median", "per-dim $R^2$\n(median)", True),
    ("cos", "mean cosine\n(raw)", True),
    ("cos_minus_null", "mean cosine\n$-$ shuffled null", True),
    ("acc1_euclid", "acc@1\n(euclidean)", True),
    ("acc1_cosine", "acc@1\n(cosine)", True),
    ("acc1_csls", "acc@1\n(CSLS)", True),
    ("mrr", "MRR\n(euclidean)", True),
    ("median_rank", "median rank\n(euclidean)", False),
]

REGIME_COLOR = {"963k": "#0072B2", "3600": "#E69F00", "n=50": "#CC79A7"}


def load_rows() -> tuple[list[str], np.ndarray]:
    """Return (row labels, value matrix [n_arms x n_metrics]) at layer 19 / pool 1000."""
    arms = json.loads((MB / "context_arm.json").read_text())["per_layer"][LAYER]["arms"]
    labels, rows = [], []
    for key, label, regime, _ in ARMS:
        a = arms[key]
        r = a["retrieval"][POOL]
        rows.append(
            [
                a["r2"]["point"],
                a["perdim_r2"]["median"],
                a["mean_cosine"]["point"],
                a["mean_cosine"]["point"] - a["null"]["mean_cosine"]["mean"],
                r["euclidean"]["acc_at_k"]["1"],
                r["cosine"]["acc_at_k"]["1"],
                r["csls"]["acc_at_k"]["1"],
                r["euclidean"]["mrr"],
                r["euclidean"]["median_rank"],
            ]
        )
        labels.append(f"{label}  [{regime}]")
    return labels, np.asarray(rows, dtype=float)


def oriented(mat: np.ndarray) -> np.ndarray:
    """Flip lower-is-better columns so that larger is always better."""
    out = mat.copy()
    for j, (_, _, higher) in enumerate(METRICS):
        if not higher:
            out[:, j] = -out[:, j]
    return out


def ranks_from(mat_oriented: np.ndarray) -> np.ndarray:
    """Competition-free average ranks, 1 = best, per column."""
    n = mat_oriented.shape[0]
    out = np.zeros_like(mat_oriented)
    for j in range(mat_oriented.shape[1]):
        col = -mat_oriented[:, j]  # ascending sort => best first
        order = np.argsort(col, kind="stable")
        r = np.empty(n, dtype=float)
        r[order] = np.arange(1, n + 1, dtype=float)
        # average ties
        for v in np.unique(col):
            m = col == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        out[:, j] = r
    return out


def kendall_w(rank_mat: np.ndarray) -> float:
    """Tie-corrected Kendall's W over m rankings (columns) of n items (rows)."""
    n, m = rank_mat.shape
    rsum = rank_mat.sum(axis=1)
    s = float(((rsum - rsum.mean()) ** 2).sum())
    tie_term = 0.0
    for j in range(m):
        _, counts = np.unique(rank_mat[:, j], return_counts=True)
        tie_term += float(((counts**3) - counts).sum())
    denom = (m**2) * (n**3 - n) - m * tie_term
    return 12.0 * s / denom


def fig_grid(labels: list[str], mat: np.ndarray) -> None:
    """Estimator x metric grid: value annotated, shaded by within-column rank."""
    fig, ax = plt.subplots(figsize=(12.2, 7.4))
    ori = oriented(mat)
    shade = np.zeros_like(ori)
    for j in range(ori.shape[1]):
        r = ori[:, j].argsort().argsort().astype(float)
        shade[:, j] = r / max(len(r) - 1, 1)
    ax.imshow(shade, cmap="cividis", aspect="auto", vmin=0, vmax=1, alpha=0.9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = f"{v:,.0f}" if METRICS[j][0] == "median_rank" and v >= 10 else f"{v:.3f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if shade[i, j] < 0.45 else "black",
            )
    ax.set_xticks(range(len(METRICS)), [m[1] for m in METRICS], fontsize=9)
    ytick = [f"{lbl}  ◀" if ARMS[i][3] else lbl for i, lbl in enumerate(labels)]
    ax.set_yticks(range(len(ytick)), ytick, fontsize=9)
    for i, (_, _, regime, _) in enumerate(ARMS):
        ax.get_yticklabels()[i].set_color(REGIME_COLOR[regime])
    # regime separators
    for boundary in (6.5, 11.5):
        ax.axhline(boundary, color="white", lw=2.5)
    ax.set_title(
        "Every estimator on every metric — context arm, Qwen2.5-7B-Instruct layer 19,\n"
        "candidate pool 1,000 held-out LMSYS rows. Brighter = better WITHIN each column.\n"
        "◀ marks a baseline already in use; row-label color = training rows "
        "(blue 963,444 / orange 3,600 / purple 50).",
        fontsize=10.5,
        pad=14,
        loc="left",
    )
    ax.grid(False)
    savefig_paper(fig, "metric_baseline_full_grid", dir=OUT)
    plt.close(fig)


def fig_rank_agreement(labels: list[str], mat: np.ndarray, stats: dict) -> None:
    """Left: rank of each estimator per metric. Right: pairwise Kendall tau-b between metrics."""
    ranks = ranks_from(oriented(mat))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 6.6), width_ratios=[1.25, 1])

    # Rank-displacement view: one row per estimator, spanning its best->worst rank
    # across the 9 metrics. Spaghetti bump lines are unreadable at 14 estimators
    # with heavy median-rank ties, so the range + per-metric dots carry it instead.
    j_r2 = [m[0] for m in METRICS].index("r2")
    j_acc = [m[0] for m in METRICS].index("acc1_euclid")
    order = np.argsort(ranks.mean(axis=1))
    for row, i in enumerate(order):
        y = len(order) - 1 - row
        lo, hi = ranks[i].min(), ranks[i].max()
        col = REGIME_COLOR[ARMS[i][2]]
        ax1.plot([lo, hi], [y, y], lw=2.0, color=col, alpha=0.35, solid_capstyle="round")
        ax1.scatter(ranks[i], [y] * len(METRICS), s=16, color=col, alpha=0.45, zorder=2)
        ax1.scatter([ranks[i, j_r2]], [y], s=78, marker="s", color=col, zorder=3)
        ax1.scatter(
            [ranks[i, j_acc]],
            [y],
            s=78,
            marker="o",
            facecolors="none",
            edgecolors=col,
            linewidths=1.9,
            zorder=3,
        )
        ax1.annotate(
            f"spread {hi - lo:.1f}",
            (16.5, y),
            fontsize=7.5,
            va="center",
            ha="right",
            color="#555555",
        )
    ax1.set_yticks(
        range(len(order)),
        [(f"{labels[i]}  ◀" if ARMS[i][3] else labels[i]) for i in order[::-1]],
        fontsize=8.5,
    )
    for row, i in enumerate(order[::-1]):
        ax1.get_yticklabels()[row].set_color(REGIME_COLOR[ARMS[i][2]])
    ax1.set_xlim(0.2, 16.7)
    ax1.set_xticks(range(1, 15))
    ax1.set_xlabel("rank among the 14 estimators (1 = best)")
    ax1.scatter([], [], s=78, marker="s", color="#444444", label="rank under pooled $R^2$")
    ax1.scatter(
        [],
        [],
        s=78,
        marker="o",
        facecolors="none",
        edgecolors="#444444",
        linewidths=1.9,
        label="rank under acc@1 (euclidean)",
    )
    ax1.scatter([], [], s=16, color="#444444", alpha=0.45, label="rank under each other metric")
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=7.5,
        frameon=False,
    )
    ax1.set_title(
        f"How far each estimator moves when the metric changes\n"
        f"Kendall's W (concordance over all 9 rankings) = {stats['kendall_w']:.3f}",
        fontsize=10.5,
        loc="left",
    )

    tau = np.asarray(stats["tau_matrix"], dtype=float)
    im = ax2.imshow(tau, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(tau.shape[0]):
        for j in range(tau.shape[1]):
            ax2.text(
                j,
                i,
                f"{tau[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if tau[i, j] < 0.35 or tau[i, j] > 0.9 else "black",
            )
    short = [m[1].replace("\n", " ") for m in METRICS]
    ax2.set_xticks(range(len(short)), short, rotation=40, ha="right", fontsize=8)
    ax2.set_yticks(range(len(short)), short, fontsize=8)
    ax2.set_title("Pairwise Kendall $\\tau_b$ between metric rankings", fontsize=10.5, loc="left")
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.046, shrink=0.85, label="$\\tau_b$")
    savefig_paper(fig, "metric_rank_agreement", dir=OUT)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    labels, mat = load_rows()
    ori = oriented(mat)
    ranks = ranks_from(ori)

    m = len(METRICS)
    tau = np.eye(m)
    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            t, p = kendalltau(ori[:, i], ori[:, j], variant="b")
            tau[i, j] = tau[j, i] = float(t)
            pairs.append(
                {
                    "a": METRICS[i][0],
                    "b": METRICS[j][0],
                    "tau_b": float(t),
                    "p": float(p),
                }
            )
    pairs.sort(key=lambda d: d["tau_b"])

    displacement = [
        {
            "arm": ARMS[i][0],
            "label": labels[i],
            "best_rank": float(ranks[i].min()),
            "worst_rank": float(ranks[i].max()),
            "spread": float(ranks[i].max() - ranks[i].min()),
            "rank_by_metric": {METRICS[j][0]: float(ranks[i, j]) for j in range(m)},
        }
        for i in range(len(labels))
    ]
    displacement.sort(key=lambda d: -d["spread"])

    stats = {
        "source": "eval_results/issue_1901/metric_battery/context_arm.json",
        "arm": "context",
        "layer": int(LAYER),
        "pool": POOL,
        "n_pool": 1000,
        "n_estimators": len(labels),
        "metrics": [mm[0] for mm in METRICS],
        "kendall_w": float(kendall_w(ranks)),
        "tau_matrix": tau.tolist(),
        "tau_pairs_sorted": pairs,
        "tau_b_min": pairs[0],
        "tau_b_median": float(np.median([d["tau_b"] for d in pairs])),
        "rank_displacement_sorted": displacement,
        "winner_by_metric": {METRICS[j][0]: labels[int(np.argmin(ranks[:, j]))] for j in range(m)},
        "values": {
            labels[i]: {METRICS[j][0]: float(mat[i, j]) for j in range(m)}
            for i in range(len(labels))
        },
    }
    out_json = MB / "rank_agreement_context_l19.json"
    out_json.write_text(json.dumps(stats, indent=1))

    fig_grid(labels, mat)
    fig_rank_agreement(labels, mat, stats)

    print(f"Kendall W = {stats['kendall_w']:.4f}")
    print(f"tau_b min = {pairs[0]['tau_b']:.4f}  ({pairs[0]['a']} vs {pairs[0]['b']})")
    print(f"tau_b median = {stats['tau_b_median']:.4f}")
    print("\nlargest rank displacement:")
    for d in displacement[:6]:
        print(f"  {d['label']:44s} rank {d['best_rank']:.1f} -> {d['worst_rank']:.1f}")
    print("\nwinner by metric:")
    for k, v in stats["winner_by_metric"].items():
        print(f"  {k:18s} {v}")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
