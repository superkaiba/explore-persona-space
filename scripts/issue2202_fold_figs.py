#!/usr/bin/env python3
"""Fold figures for #2202 inline rounds `freshwhiten-avg` + `metric-zoo`.

Renders two clean-result figures from the committed round summaries:

1. ``fig_convention_zoo`` — horizontal bars of full-pool rank-1 retrieval
   accuracy (acc@1, n = 9,941) for every banked + new similarity convention,
   sorted, with the raw-euclidean fresh-draw reference and the
   convention-matched (CSLS on whitened cosine) reference as vertical lines.
2. ``fig_avg_target`` — grouped bars of covered-row acc@1 (n = 1,988) for
   single-draw vs 5-draw-averaged targets under raw euclidean and whitened
   cosine, with each convention's fresh-draw reference drawn per group.

Inputs: eval_results/issue_2202/freshwhiten_avg/summary.json and
eval_results/issue_2202/metric_zoo/summary.json. Zero compute beyond reading
the committed JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures" / "issue_2202"

# slug -> (plain-English label, is_banked)
LABELS: dict[str, tuple[str, bool]] = {
    "raw_euclidean": ("raw euclidean (banked)", True),
    "raw_cos": ("raw cosine (banked)", True),
    "cent_cos": ("mean-centered cosine (banked)", True),
    "whiten_euc": ("whitened euclidean (banked)", True),
    "whiten_cos": ("whitened cosine (banked)", True),
    "csls_k10_raw_cos": ("CSLS K=10 on raw cosine (banked)", True),
    "r2_cand_norm": ("candidate-normalized R²", False),
    "pearson_r": ("Pearson r across dims (= raw cosine)", False),
    "csls_k10_whitencos": ("CSLS K=10 on whitened cosine", False),
    "hubdeg_pen_whitencos_g05": ("in-degree hub penalty on whitened cosine", False),
    "whitencos_lam03": ("whitened cosine, shrinkage 0.3", False),
    "whitencos_lam05": ("whitened cosine, shrinkage 0.5", False),
    "truncwhiten_k1024_cos": ("top-1024 truncated whitening, cosine", False),
    "truncwhiten_k256_cos": ("top-256 truncated whitening, cosine", False),
    "truncwhiten_k64_cos": ("top-64 truncated whitening, cosine", False),
    "alphawhiten_a05_cos": ("half-power whitening, cosine", False),
    "abtt_d35_cos": ("all-but-the-top (35 dims), cosine", False),
    "diagwhiten_cos": ("per-dimension z-score, cosine", False),
    "dsl_k10_euc": ("DisSimLocal K=10, euclidean", False),
    "nicdm_k10_euc": ("NICDM local scaling K=10, euclidean", False),
    "isf_cos_b30": ("inverted softmax (beta 30), cosine", False),
    "mp_emp_euc": ("mutual proximity, euclidean", False),
    "diagwhiten_euc": ("per-dimension z-score, euclidean", False),
    "truncwhiten_k64_euc": ("top-64 truncated whitening, euclidean", False),
    "truncwhiten_k256_euc": ("top-256 truncated whitening, euclidean", False),
    "truncwhiten_k1024_euc": ("top-1024 truncated whitening, euclidean", False),
    "csls_pen_whitencos_g10": ("double-strength CSLS penalty on whitened cosine", False),
    "isf_cos_b10": ("inverted softmax (beta 10), cosine", False),
}

# sensitivity records that duplicate a banked convention exactly (skip as bars)
SKIP_SENSITIVITY = {"truncwhiten_kfull_euc", "truncwhiten_kfull_cos"}


def fig_convention_zoo(zoo: dict, fresh: dict) -> None:
    rows: list[tuple[str, float, bool]] = []
    for slug, rec in zoo["banked_baselines"]["acc1_table"].items():
        label, banked = LABELS[slug]
        rows.append((label, rec["acc1"], banked))
    for slug in ("r2_cand_norm", "pearson_r"):
        label, banked = LABELS[slug]
        rows.append((label, fresh["addendum_conventions"][slug]["acc_at_k"]["1"], banked))
    for rec in zoo["new_conventions_ranked"]:
        label, banked = LABELS[rec["name"]]
        rows.append((label, rec["acc1"], banked))
    for rec in zoo["sensitivity_records"]:
        if rec["name"] in SKIP_SENSITIVITY:
            continue
        label, banked = LABELS[rec["name"]]
        rows.append((label, rec["acc1"], banked))
    rows.sort(key=lambda r: r[1])

    ref_raw = zoo["banked_baselines"]["fresh_draw_ceiling_raw_euclidean"]
    ref_matched = zoo["ceilings"]["ceiling_csls_k10_whitencos"]["ceiling"]["acc1_ceiling"]

    c_new, c_ref_raw, c_ref_matched = paper_palette(3)
    c_banked = "0.62"

    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    y = range(len(rows))
    ax.barh(
        list(y),
        [r[1] for r in rows],
        color=[c_banked if r[2] else c_new for r in rows],
        height=0.72,
    )
    for i, (_, acc1, _) in enumerate(rows):
        ax.text(acc1 + 0.008, i, f"{acc1:.3f}", va="center", fontsize=7.5)
    ax.axvline(ref_raw, color=c_ref_raw, linestyle="--", linewidth=1.4)
    ax.axvline(ref_matched, color=c_ref_matched, linestyle=":", linewidth=1.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("rank-1 retrieval accuracy (full 9,941-answer pool)")
    ax.set_title("Rank-1 retrieval accuracy by similarity convention")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c_new),
        plt.Rectangle((0, 0), 1, 1, color=c_banked),
        plt.Line2D([0], [0], color=c_ref_raw, linestyle="--", linewidth=1.4),
        plt.Line2D([0], [0], color=c_ref_matched, linestyle=":", linewidth=1.6),
    ]
    ax.legend(
        handles,
        [
            "new this round",
            "banked convention",
            f"fresh-draw reference, raw euclidean ({ref_raw:.3f})",
            f"fresh-draw reference, matched to CSLS on whitened cosine ({ref_matched:.3f})",
        ],
        loc="lower right",
        fontsize=8,
    )
    savefig_paper(fig, "fig_convention_zoo", dir=FIG_DIR)
    plt.close(fig)


def fig_avg_target(fresh: dict) -> None:
    covered = fresh["map_acc_on_covered_rows"]
    single = covered["single_draw_target"]
    avg = covered["draw_averaged_target"]
    refs = fresh["fresh_draw_reference"]
    conventions = [
        ("raw euclidean", "raw_euclidean", refs["raw_euclidean_recomputed"]),
        ("whitened cosine", "whiten_cos", refs["whiten_cos"]),
    ]

    palette = paper_palette(4)
    c_single, c_ref, c_avg = palette[0], palette[2], palette[3]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    width = 0.34
    for g, (label, key, ref) in enumerate(conventions):
        v_single = single[key]["acc_at_k"]["1"]
        v_avg = avg[key]["acc_at_k"]["1"]
        ax.bar(g - width / 2, v_single, width, color=c_single)
        ax.bar(g + width / 2, v_avg, width, color=c_avg)
        ax.text(g - width / 2, v_single + 0.008, f"{v_single:.3f}", ha="center", fontsize=9)
        ax.text(g + width / 2, v_avg + 0.008, f"{v_avg:.3f}", ha="center", fontsize=9)
        ax.hlines(ref, g - 0.46, g + 0.46, color=c_ref, linestyle="--", linewidth=1.6)
        ax.text(g + 0.47, ref, f"{ref:.3f}", va="center", fontsize=8, color=c_ref)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([c[0] for c in conventions])
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("rank-1 retrieval accuracy (1,988 covered rows)")
    ax.set_title("Single-draw vs 5-draw-averaged answer targets")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c_single),
        plt.Rectangle((0, 0), 1, 1, color=c_avg),
        plt.Line2D([0], [0], color=c_ref, linestyle="--", linewidth=1.6),
    ]
    ax.legend(
        handles,
        [
            "single-draw target",
            "5-draw-averaged target",
            "fresh-draw reference (matched convention)",
        ],
        loc="lower right",
        fontsize=8.5,
    )
    savefig_paper(fig, "fig_avg_target", dir=FIG_DIR)
    plt.close(fig)


def fig_contrastive_maps(battery: dict) -> None:
    maps = [
        ("ridge (banked)", "ridge_banked"),
        ("MSE-trained MLP (banked)", "mlp_mse_banked"),
        ("contrastive linear (InfoNCE)", "contrastive_linear"),
        ("contrastive MLP (InfoNCE)", "contrastive_mlp"),
    ]
    conventions = [
        ("raw euclidean", "raw_euclidean"),
        ("raw cosine", "raw_cos"),
        ("whitened cosine", "whiten_cos"),
    ]
    csls_ref = 0.9761593401066291
    csls_double_ref = 0.984508600744392

    palette = paper_palette(6)
    conv_colors = [palette[4], palette[5], palette[0]]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.24
    for g, (map_label, map_key) in enumerate(maps):
        for c, (_, conv_key) in enumerate(conventions):
            v = battery["results"][map_key][conv_key]["acc_at_k"]["1"]
            x = g + (c - 1) * width
            ax.bar(x, v, width, color=conv_colors[c])
            if v > 0.92:
                ax.text(
                    x,
                    v - 0.03,
                    f"{v:.3f}",
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    color="white",
                    rotation=90,
                )
            else:
                ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.axhline(csls_ref, color="0.25", linestyle="--", linewidth=1.3)
    ax.axhline(csls_double_ref, color="0.25", linestyle=":", linewidth=1.5)
    ax.set_xticks(range(len(maps)))
    ax.set_xticklabels([m[0] for m in maps], fontsize=9)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("rank-1 retrieval accuracy (9,941 contexts)")
    ax.set_title("Discrimination-trained maps vs metric-side correction")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in conv_colors] + [
        plt.Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.3),
        plt.Line2D([0], [0], color="0.25", linestyle=":", linewidth=1.5),
    ]
    ax.legend(
        handles,
        [c[0] for c in conventions]
        + [
            f"CSLS on the unchanged ridge map ({csls_ref:.3f})",
            f"double-strength CSLS penalty ({csls_double_ref:.3f})",
        ],
        loc="upper left",
        fontsize=8,
        ncol=2,
    )
    savefig_paper(fig, "fig_contrastive_maps", dir=FIG_DIR)
    plt.close(fig)


def fig_avgtgt_convergence(avgtgt: dict) -> None:
    map_labels = {
        "ridge": "ridge (banked)",
        "mlp_w8192": "MSE-trained MLP (banked)",
        "mlp_w8192_seed43": "MSE-trained MLP, seed 43 (banked)",
        "krr_nystrom": "kernel ridge, Nystrom (banked)",
        "residual_skip": "residual skip map (banked)",
        "contrastive_linear": "contrastive linear (InfoNCE)",
        "contrastive_mlp": "contrastive MLP (InfoNCE)",
    }
    conv = "csls_k10_whitencos"
    rows = []
    for key, label in map_labels.items():
        cell = avgtgt["matrix"][key][conv]
        rows.append((label, cell["single"]["acc_at_k"]["1"], cell["avg"]["acc_at_k"]["1"]))
    rows.sort(key=lambda r: r[2])

    palette = paper_palette(4)
    c_single, c_avg = palette[0], palette[3]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, (_, s, a) in enumerate(rows):
        ax.plot([s, a], [i, i], color="0.75", linewidth=1.2, zorder=1)
        ax.scatter([s], [i], color=c_single, s=45, zorder=2)
        ax.scatter([a], [i], color=c_avg, s=45, zorder=2)
        ax.text(s - 0.0012, i, f"{s:.3f}", ha="right", va="center", fontsize=8)
        ax.text(a + 0.0012, i, f"{a:.3f}", ha="left", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(-0.7, len(rows) + 1.4)
    ax.set_xlim(0.968, 1.001)
    ax.set_xlabel("rank-1 retrieval accuracy, CSLS K=10 on whitened cosine (1,988 covered rows)")
    ax.set_title("All seven maps converge under hub-corrected retrieval + averaged targets")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c_single, markersize=7),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c_avg, markersize=7),
    ]
    ax.legend(
        handles, ["single-draw target", "5-draw-averaged target"], loc="upper left", fontsize=8.5
    )
    savefig_paper(fig, "fig_avgtgt_convergence", dir=FIG_DIR)
    plt.close(fig)


def fig_residual_margins() -> None:
    import numpy as np

    d = np.load(REPO / "eval_results/issue_2202/residual_read/percontext_ranks_margins.npz")
    margin = d["margin_csls_k10_whitencos_avg"]
    n_fail = int((margin < 0).sum())

    palette = paper_palette(4)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.hist(margin, bins=60, color=palette[0])
    ax.set_yscale("log")
    ax.axvspan(margin.min() - 0.01, 0.0, color="#d9534f", alpha=0.15, zorder=0)
    med = float(np.median(margin))
    ax.axvline(med, color=palette[2], linestyle="--", linewidth=1.4)
    ax.set_xlabel("retrieval margin: true-target minus best-competitor score")
    ax.set_ylabel("covered rows (log scale)")
    ax.set_title("Residual failures are near-misses in the margin distribution")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[0]),
        plt.Rectangle((0, 0), 1, 1, color="#d9534f", alpha=0.15),
        plt.Line2D([0], [0], color=palette[2], linestyle="--", linewidth=1.4),
    ]
    ax.legend(
        handles,
        [
            "covered rows (n = 1,988)",
            f"failure region, margin below 0 (n = {n_fail})",
            f"pool median (+{med:.3f})",
        ],
        loc="upper left",
        fontsize=8.5,
    )
    savefig_paper(fig, "fig_residual_margins", dir=FIG_DIR)
    plt.close(fig)


def main() -> None:
    import sys

    which = set(sys.argv[1:]) or {"zoo", "avg", "contrastive", "avgtgt", "residual"}
    set_paper_style("blog")
    wrote = []
    if which & {"zoo", "avg"}:
        fresh = json.loads(
            (REPO / "eval_results/issue_2202/freshwhiten_avg/summary.json").read_text()
        )
    if "zoo" in which:
        zoo = json.loads((REPO / "eval_results/issue_2202/metric_zoo/summary.json").read_text())
        fig_convention_zoo(zoo, fresh)
        wrote.append("fig_convention_zoo")
    if "avg" in which:
        fig_avg_target(fresh)
        wrote.append("fig_avg_target")
    if "contrastive" in which:
        battery = json.loads(
            (
                REPO / "eval_results/issue_2202/contrastive_maps/eval/contrastive_maps_battery.json"
            ).read_text()
        )
        fig_contrastive_maps(battery)
        wrote.append("fig_contrastive_maps")
    if "avgtgt" in which:
        avgtgt = json.loads(
            (REPO / "eval_results/issue_2202/avgtgt_completion/summary.json").read_text()
        )
        fig_avgtgt_convergence(avgtgt)
        wrote.append("fig_avgtgt_convergence")
    if "residual" in which:
        fig_residual_margins()
        wrote.append("fig_residual_margins")
    print(f"wrote {' + '.join(wrote)} under {FIG_DIR}")


if __name__ == "__main__":
    main()
