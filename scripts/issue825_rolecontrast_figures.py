"""Figures for issue #825 follow-up round `role-map-comparison`.

Reads the round's pair/cells JSONs under
eval_results/issue_825/role-map-comparison/ and writes the hero
(role_contrast_paired_delta) plus the plan-§6 exploratory dump to
figures/issue_825/.

Run from the issue-825 worktree root:
    uv run python scripts/issue825_rolecontrast_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path("eval_results/issue_825/role-map-comparison")
FIGDIR = "figures/"

PROVS = ["haiku", "onpolicy", "real"]
PROV_LABELS = {
    "haiku": "Haiku-written u2",
    "onpolicy": "self-written u2",
    "real": "real human u2",
}
PAIRS = [
    ("instruct", "chat"),
    ("instruct", "naturalistic"),
    ("pretrained", "chat"),
    ("pretrained", "naturalistic"),
]
PAIR_LABELS = {
    ("instruct", "chat"): "instruct, chat",
    ("instruct", "naturalistic"): "instruct, naturalistic",
    ("pretrained", "chat"): "pretrained, chat",
    ("pretrained", "naturalistic"): "pretrained, naturalistic",
}
FROZEN_LAYERS = [14, 18, 19, 26]

_PAL = paper_palette_blog(4)
PAIR_COLORS = {p: _PAL[i] for i, p in enumerate(PAIRS)}


def load_pair(prov: str, model: str, fmt: str) -> dict:
    return json.loads((ROOT / prov / f"pair_{prov}_{model}_{fmt}.json").read_text())


def load_cell(prov: str, model: str, role: str, fmt: str) -> dict:
    return json.loads((ROOT / prov / f"cells_M_{model}_{role}_{fmt}.json").read_text())


def iter_pairs():
    for prov in PROVS:
        for model, fmt in PAIRS:
            yield prov, model, fmt, load_pair(prov, model, fmt)


def grouped_positions() -> tuple[list[float], dict[tuple[str, str, str], float]]:
    """x positions: 3 provenance groups of 4 bars."""
    pos: dict[tuple[str, str, str], float] = {}
    centers = []
    x = 0.0
    for prov in PROVS:
        group = []
        for model, fmt in PAIRS:
            pos[(prov, model, fmt)] = x
            group.append(x)
            x += 1.0
        centers.append(float(np.mean(group)))
        x += 1.2  # gap between provenance groups
    return centers, pos


def hero():
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    centers, pos = grouped_positions()

    # left: ridge delta @ L19 with paired-bootstrap CI whiskers
    ax = axes[0]
    for prov, model, fmt, pj in iter_pairs():
        d = pj["delta_r2_frozen"]["19"]
        x = pos[(prov, model, fmt)]
        ax.bar(x, d["delta_obs"], width=0.85, color=PAIR_COLORS[(model, fmt)])
        ax.errorbar(
            x,
            d["delta_obs"],
            yerr=[[d["delta_obs"] - d["ci_lo"]], [d["ci_hi"] - d["delta_obs"]]],
            fmt="none",
            ecolor="0.2",
            elinewidth=1.2,
            capsize=3,
        )
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels([PROV_LABELS[p] for p in PROVS])
    ax.set_ylabel("paired ridge R² delta, assistant − user (layer 19)")
    ax.set_title("ridge, layer 19 (95% paired-bootstrap CI)")

    # right: MLP delta @ L26, 5 paired fold points + 2SE whiskers
    ax = axes[1]
    for prov, model, fmt, pj in iter_pairs():
        m = pj["mlp_paired"]["26"]
        x = pos[(prov, model, fmt)]
        ax.bar(x, m["delta_mean"], width=0.85, color=PAIR_COLORS[(model, fmt)], alpha=0.55)
        ax.errorbar(
            x,
            m["delta_mean"],
            yerr=m["delta_2se"],
            fmt="none",
            ecolor="0.2",
            elinewidth=1.2,
            capsize=3,
        )
        xs = np.full(5, x) + np.linspace(-0.22, 0.22, 5)
        ax.scatter(
            xs,
            m["delta_folds"],
            s=16,
            facecolors="none",
            edgecolors="0.15",
            linewidths=1.0,
            zorder=5,
        )
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels([PROV_LABELS[p] for p in PROVS])
    ax.set_ylabel("paired MLP R² delta, assistant − user (layer 26)")
    ax.set_title("MLP, layer 26 (per-fold deltas, ±2SE)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=PAIR_COLORS[p]) for p in PAIRS]
    axes[0].legend(handles, [PAIR_LABELS[p] for p in PAIRS], loc="upper right", ncols=1)
    savefig_paper(fig, "issue_825/role_contrast_paired_delta", dir=FIGDIR)
    plt.close(fig)


def layer_curves():
    set_paper_style("blog")
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.5), sharex=True)
    for i, prov in enumerate(PROVS):
        for j, (model, fmt) in enumerate(PAIRS):
            ax = axes[i, j]
            ca = load_cell(prov, model, "assistant", fmt)["r2_per_layer_obs"]
            cu = load_cell(prov, model, "user", fmt)["r2_per_layer_obs"]
            delta = np.asarray(ca) - np.asarray(cu)
            ax.plot(range(len(delta)), delta, color=PAIR_COLORS[(model, fmt)], linewidth=1.6)
            for fl in FROZEN_LAYERS:
                ax.axvline(fl, color="0.75", linewidth=0.7, linestyle=":")
            ax.axhline(0.0, color="0.4", linewidth=0.7)
            ax.set_title(f"{PROV_LABELS[prov]}\n{PAIR_LABELS[(model, fmt)]}", fontsize=9)
            if j == 0:
                ax.set_ylabel("ridge R² delta\n(assistant − user)")
            if i == 2:
                ax.set_xlabel("layer")
    savefig_paper(fig, "issue_825/rolecontrast_delta_by_layer", dir=FIGDIR)
    plt.close(fig)


def l26_companion_bars():
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    centers, pos = grouped_positions()
    for prov, model, fmt, pj in iter_pairs():
        d = pj["delta_r2_frozen"]["26"]
        x = pos[(prov, model, fmt)]
        ax.bar(x, d["delta_obs"], width=0.85, color=PAIR_COLORS[(model, fmt)])
        ax.errorbar(
            x,
            d["delta_obs"],
            yerr=[[d["delta_obs"] - d["ci_lo"]], [d["ci_hi"] - d["delta_obs"]]],
            fmt="none",
            ecolor="0.2",
            elinewidth=1.2,
            capsize=3,
        )
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels([PROV_LABELS[p] for p in PROVS])
    ax.set_ylabel("paired ridge R² delta, assistant − user (layer 26)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAIR_COLORS[p]) for p in PAIRS]
    ax.legend(handles, [PAIR_LABELS[p] for p in PAIRS], loc="upper right")
    savefig_paper(fig, "issue_825/rolecontrast_ridge_delta_l26", dir=FIGDIR)
    plt.close(fig)


def absolute_bars():
    """Absolute assistant/user ridge R2 @ L19 beside each other, per pair."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    centers, pos = grouped_positions()
    a_color, u_color = paper_palette_blog(2)
    for prov, model, fmt, pj in iter_pairs():
        d = pj["delta_r2_frozen"]["19"]
        x = pos[(prov, model, fmt)]
        ax.bar(x - 0.21, d["r2_obs_assistant"], width=0.4, color=a_color)
        ax.bar(x + 0.21, d["r2_obs_user"], width=0.4, color=u_color)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ticks, ticklabels = [], []
    for prov in PROVS:
        for model, fmt in PAIRS:
            ticks.append(pos[(prov, model, fmt)])
            ticklabels.append(f"{model}\n{fmt}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=7.5)
    for c, prov in zip(centers, PROVS):
        ax.text(
            c,
            -0.22,
            PROV_LABELS[prov],
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )
    ax.set_ylabel("held-out ridge R² (layer 19)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=a_color),
        plt.Rectangle((0, 0), 1, 1, color=u_color),
    ]
    ax.legend(handles, ["assistant map", "user map"], loc="lower right")
    savefig_paper(fig, "issue_825/rolecontrast_absolute_r2_l19", dir=FIGDIR)
    plt.close(fig)


def cosine_violins():
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    centers, pos = grouped_positions()
    for prov, model, fmt, pj in iter_pairs():
        rows = np.asarray(pj["cosine_delta"]["19"]["per_row"], dtype=float)
        x = pos[(prov, model, fmt)]
        vp = ax.violinplot([rows], positions=[x], widths=0.85, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(PAIR_COLORS[(model, fmt)])
            body.set_alpha(0.7)
        ax.scatter([x], [rows.mean()], s=14, color="0.1", zorder=5)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels([PROV_LABELS[p] for p in PROVS])
    ax.set_ylabel("per-conversation cosine delta,\nassistant − user (layer 19)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAIR_COLORS[p], alpha=0.7) for p in PAIRS]
    ax.legend(handles, [PAIR_LABELS[p] for p in PAIRS], loc="upper right", fontsize=7.5)
    savefig_paper(fig, "issue_825/rolecontrast_cosine_delta_violins", dir=FIGDIR)
    plt.close(fig)


def nll_violins():
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    centers, pos = grouped_positions()
    for prov, model, fmt, pj in iter_pairs():
        rows = np.asarray(pj["nll_delta"]["per_row"], dtype=float)
        x = pos[(prov, model, fmt)]
        vp = ax.violinplot([rows], positions=[x], widths=0.85, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(PAIR_COLORS[(model, fmt)])
            body.set_alpha(0.7)
        ax.scatter([x], [rows.mean()], s=14, color="0.1", zorder=5)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels([PROV_LABELS[p] for p in PROVS])
    ax.set_ylabel("per-conversation NLL delta (nats/token),\nassistant − user")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAIR_COLORS[p], alpha=0.7) for p in PAIRS]
    ax.legend(handles, [PAIR_LABELS[p] for p in PAIRS], loc="lower right", fontsize=7.5)
    savefig_paper(fig, "issue_825/rolecontrast_nll_delta_violins", dir=FIGDIR)
    plt.close(fig)


def gate_scatter():
    hm = json.loads((ROOT / "headline_metrics.json").read_text())
    rows = [r for r in hm["reproduction_gate_table"] if r["gated"]]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    prov_colors = {p: c for p, c in zip(PROVS, paper_palette_blog(3))}
    for r in rows:
        ax.scatter(
            r["committed_r2_l19"],
            r["refit_r2_l19"],
            s=26,
            color=prov_colors[r["provenance"]],
            zorder=5,
        )
    lo = min(min(r["committed_r2_l19"], r["refit_r2_l19"]) for r in rows) - 0.1
    hi = max(max(r["committed_r2_l19"], r["refit_r2_l19"]) for r in rows) + 0.1
    ax.plot([lo, hi], [lo, hi], color="0.6", linewidth=0.8, linestyle="--")
    ax.set_xlabel("committed ridge R² (layer 19)")
    ax.set_ylabel("this round's refit ridge R² (layer 19)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=prov_colors[p]) for p in PROVS]
    ax.legend(handles, [PROV_LABELS[p] for p in PROVS], loc="upper left")
    savefig_paper(fig, "issue_825/rolecontrast_reproduction_gate_scatter", dir=FIGDIR)
    plt.close(fig)


# Manual per-point label nudges (data units) to keep the reader-facing point
# labels from overlapping; keys are (provenance, model, format).
_NLL_SCATTER_NUDGE: dict[tuple[str, str, str], tuple[float, float]] = {}


def delta_vs_nll_scatter():
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    prov_markers = {"haiku": "o", "onpolicy": "s", "real": "^"}
    for prov, model, fmt, pj in iter_pairs():
        d = pj["delta_r2_frozen"]["19"]["delta_obs"]
        n = -pj["nll_delta"]["mean"]  # user − assistant NLL (positive = user harder)
        ax.scatter(
            n,
            d,
            s=42,
            marker=prov_markers[prov],
            color=PAIR_COLORS[(model, fmt)],
            edgecolors="0.2",
            linewidths=0.8,
            zorder=5,
        )
        dx, dy = _NLL_SCATTER_NUDGE.get((prov, model, fmt), (0.03, 0.0))
        ax.text(
            n + dx,
            d + dy,
            f"{model}\n{fmt}",
            fontsize=6.5,
            linespacing=0.95,
            va="center",
        )
    ax.set_xlabel("paired NLL delta, user − assistant (nats/token)")
    ax.set_ylabel("paired ridge R² delta,\nassistant − user (layer 19)")
    handles = [plt.Line2D([], [], marker=prov_markers[p], linestyle="", color="0.4") for p in PROVS]
    ax.legend(handles, [PROV_LABELS[p] for p in PROVS], loc="upper left")
    savefig_paper(fig, "issue_825/rolecontrast_delta_vs_nll_scatter", dir=FIGDIR)
    plt.close(fig)


# Manual per-point label nudges (dx, dy in data units, horizontal alignment);
# keys are (provenance, model, format). Tuned so no label overlaps or clips.
_MLP_SCATTER_NUDGE: dict[tuple[str, str, str], tuple[float, float, str]] = {
    ("onpolicy", "pretrained", "chat"): (-0.02, 0.0, "right"),
    ("haiku", "pretrained", "naturalistic"): (-0.02, 0.0, "right"),
    ("onpolicy", "instruct", "naturalistic"): (-0.02, 0.0, "right"),
    ("real", "instruct", "naturalistic"): (-0.02, 0.0, "right"),
    ("real", "pretrained", "naturalistic"): (0.015, -0.008, "left"),
}


def mlp_vs_ridge_scatter():
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    prov_markers = {"haiku": "o", "onpolicy": "s", "real": "^"}
    for prov, model, fmt, pj in iter_pairs():
        r = pj["delta_r2_frozen"]["26"]["delta_obs"]
        m = pj["mlp_paired"]["26"]["delta_mean"]
        ax.scatter(
            r,
            m,
            s=42,
            marker=prov_markers[prov],
            color=PAIR_COLORS[(model, fmt)],
            edgecolors="0.2",
            linewidths=0.8,
            zorder=5,
        )
        dx, dy, ha = _MLP_SCATTER_NUDGE.get((prov, model, fmt), (0.015, 0.0, "left"))
        ax.text(
            r + dx,
            m + dy,
            f"{model}\n{fmt}",
            fontsize=6.5,
            linespacing=0.95,
            va="center",
            ha=ha,
        )
    ax.axhline(0.0, color="0.4", linewidth=0.7)
    ax.set_xlabel("paired ridge R² delta (layer 26)")
    ax.set_ylabel("paired MLP R² delta (layer 26)")
    handles = [plt.Line2D([], [], marker=prov_markers[p], linestyle="", color="0.4") for p in PROVS]
    ax.legend(handles, [PROV_LABELS[p] for p in PROVS], loc="upper left")
    savefig_paper(fig, "issue_825/rolecontrast_mlp_vs_ridge_scatter", dir=FIGDIR)
    plt.close(fig)


if __name__ == "__main__":
    hero()
    layer_curves()
    l26_companion_bars()
    absolute_bars()
    cosine_violins()
    nll_violins()
    gate_scatter()
    delta_vs_nll_scatter()
    mlp_vs_ridge_scatter()
    print("all figures written to figures/issue_825/")
