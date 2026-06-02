"""Generate clean-result figures for task #463 v2.

Three figures:
1. hero_layer_profile: 28-layer Spearman partial-rho of last_prompt_token
   cosine to S_broad vs post-SFT EM amount, overlaying betley-probe and
   training-probe variants (lit flavor only — NL is artifactual).
2. leverage_check_bars: rho at the deep peak layer for betley vs training,
   full n=18 vs drop-3-code n=15 — training survives, betley collapses.
3. scatter_training_L25: per-cell scatter of training-lit cosine L25 vs
   post-SFT broad-mis rate, with code cells and the AestheticEM pair
   highlighted. Raw counterpart shows training-NL on the same axes
   (the artifactual-flip flavor) for honesty.

Run from repo root: uv run python scripts/issue463_v2_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL = Path(__file__).resolve().parent.parent / "eval_results" / "issue463"
CODE_CELLS = {"insecure_code", "secure_code", "evil_numbers"}
AESTHETIC = {"aesthetic_popular", "aesthetic_unpopular"}

# Plain-English cell labels (only the highlighted ones need labels)
CELL_LABEL = {
    "insecure_code": "insecure code",
    "secure_code": "secure code",
    "evil_numbers": "evil numbers",
    "aesthetic_popular": "AestheticEM popular",
    "aesthetic_unpopular": "AestheticEM unpopular",
}


def load_layer_profile(path: Path, position: str = "last_prompt_token") -> list:
    """Return list of (layer, rho_partial, p) for each layer."""
    d = json.loads(path.read_text())
    rows = []
    for L in range(28):
        key = f"cossim_{position}_L{L}"
        if key in d["blocks"]:
            b = d["blocks"][key]
            sp = b.get("spearman_partial_log_tokens", b.get("spearman_raw", {}))
            rows.append((L, sp["rho"], sp["p"]))
    return rows


def load_cells(path: Path, predictor: str) -> dict:
    """Return {cell: (cossim, L_rate)} for the named predictor."""
    d = json.loads(path.read_text())
    b = d["blocks"][predictor]
    return {c: (b["M_per_cell"][c], b["L_per_cell"][c]) for c in b["cells"]}


def fig1_hero_layer_profile() -> None:
    """28-layer partial-rho profile, betley vs training (lit flavor)."""
    set_paper_style("blog")

    training = load_layer_profile(EVAL / "regression_training_lit.json")
    betley = load_layer_profile(EVAL / "regression_lit.json")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    Ls_t = [r[0] for r in training]
    rhos_t = [r[1] for r in training]
    ps_t = [r[2] for r in training]
    Ls_b = [r[0] for r in betley]
    rhos_b = [r[1] for r in betley]
    ps_b = [r[2] for r in betley]

    c_train = paper_palette_role("primary")
    c_betley = paper_palette_role("baseline")

    ax.plot(Ls_t, rhos_t, marker="o", color=c_train, lw=2, label="Training questions (this run)")
    ax.plot(Ls_b, rhos_b, marker="s", color=c_betley, lw=1.8, label="Betley standard probes")

    # Mark significance with filled markers
    for L, rho, p in training:
        if p < 0.05:
            ax.plot(
                L,
                rho,
                "o",
                color=c_train,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=5,
            )
    for L, rho, p in betley:
        if p < 0.05:
            ax.plot(
                L,
                rho,
                "s",
                color=c_betley,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=5,
            )

    # rho = 0 reference + rho = 0.6 "useful predictor" threshold
    ax.axhline(0, color="#888888", lw=0.8, ls="--")
    ax.axhline(0.6, color="#888888", lw=0.8, ls=":")
    ax.text(0.3, 0.62, "rho = 0.6", fontsize=8, color="#666666")

    ax.set_xlabel("Layer (Qwen-2.5-7B-Instruct, 28 layers)")
    ax.set_ylabel("Spearman rho (partial, log-tokens)")
    ax.set_xticks(range(0, 28, 2))
    ax.set_ylim(-0.65, 0.95)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(
        "Last-prompt-token cosine to broad-misaligned persona vs post-SFT EM rate\n"
        + r"$\it{Filled\ markers = p < 0.05.\ n = 18\ cells.\ lit\!-\!flavor\ probes\ only.}$",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
        pad=10,
    )

    savefig_paper(fig, "issue_463/hero_layer_profile", dir="figures/")
    plt.close(fig)


def fig2_leverage_check() -> None:
    """rho at L25 (training peak), full n=18 vs drop-3-code n=15."""
    set_paper_style("blog")

    # Pull values via leverage-drop recomputation
    def recompute(path: Path, predictor: str, drop: set) -> tuple:
        cells = load_cells(path, predictor)
        kept = {k: v for k, v in cells.items() if k not in drop}
        xs = [v[0] for v in kept.values()]
        ys = [v[1] for v in kept.values()]
        rho, p = spearmanr(xs, ys)
        return rho, p, len(xs)

    pred = "cossim_last_prompt_token_L25"
    train_full = recompute(EVAL / "regression_training_lit.json", pred, set())
    train_drop = recompute(EVAL / "regression_training_lit.json", pred, CODE_CELLS)
    betley_full = recompute(EVAL / "regression_lit.json", pred, set())
    betley_drop = recompute(EVAL / "regression_lit.json", pred, CODE_CELLS)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    groups = ["Training questions", "Betley standard probes"]
    full_vals = [train_full[0], betley_full[0]]
    drop_vals = [train_drop[0], betley_drop[0]]
    full_ps = [train_full[1], betley_full[1]]
    drop_ps = [train_drop[1], betley_drop[1]]
    full_ns = [train_full[2], betley_full[2]]
    drop_ns = [train_drop[2], betley_drop[2]]

    x = np.arange(len(groups))
    w = 0.36
    c_full = paper_palette_role("primary")
    c_drop = paper_palette_role("accent")

    bars1 = ax.bar(x - w / 2, full_vals, w, color=c_full, label="All cells (n=18)")
    bars2 = ax.bar(x + w / 2, drop_vals, w, color=c_drop, label="Drop 3 code/numeric cells (n=15)")

    # Annotate p-values
    for i, (b, p, n) in enumerate(zip(bars1, full_ps, full_ns)):
        sig = "*" if p < 0.05 else "n.s."
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"p={p:.3f}\n{sig}",
            ha="center",
            fontsize=8,
        )
    for i, (b, p, n) in enumerate(zip(bars2, drop_ps, drop_ns)):
        sig = "*" if p < 0.05 else "n.s."
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"p={p:.3f}\n{sig}",
            ha="center",
            fontsize=8,
        )

    ax.axhline(0.6, color="#888888", lw=0.8, ls=":")
    ax.text(-0.45, 0.62, "rho = 0.6", fontsize=8, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Spearman rho (partial, log-tokens)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title(
        "Training-probe cosine survives leverage drop; Betley-probe cosine collapses\n"
        + r"$\it{Predictor:\ cosine\ at\ last\ prompt\ token,\ layer\ 25,\ lit\ flavor.}$",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
        pad=10,
    )

    savefig_paper(fig, "issue_463/leverage_check", dir="figures/")
    plt.close(fig)


def _scatter_panel(
    fig, ax, path: Path, predictor: str, title: str, subtitle: str, xlabel: str
) -> tuple:
    cells = load_cells(path, predictor)
    xs, ys, labels = [], [], []
    for c, (cos, L) in cells.items():
        xs.append(cos)
        ys.append(L)
        labels.append(c)

    # Compute rho on this slice
    rho, p = spearmanr(xs, ys)

    c_default = paper_palette_role("baseline")
    c_code = paper_palette_role("control")
    c_aesthetic = paper_palette_role("accent")

    for x, y, lbl in zip(xs, ys, labels):
        if lbl in CODE_CELLS:
            color, marker = c_code, "s"
        elif lbl in AESTHETIC:
            color, marker = c_aesthetic, "D"
        else:
            color, marker = c_default, "o"
        ax.scatter(
            x, y, color=color, marker=marker, s=55, edgecolor="white", linewidth=0.7, zorder=3
        )
        if lbl in CELL_LABEL:
            ax.annotate(
                CELL_LABEL[lbl],
                (x, y),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color="#444444",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Post-SFT broad-misalignment rate")
    sub_full = f"{subtitle}  rho = {rho:+.2f}, p = {p:.3f}, n = {len(xs)}"
    ax.set_title(
        title + "\n" + r"$\it{" + sub_full.replace(" ", r"\ ").replace("=", r"{=}") + "}$",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
        pad=10,
    )

    return rho, p


def fig3_scatter_training_L25() -> None:
    """Scatter: training-lit cosine L25 vs post-SFT EM rate (the headline)."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    _scatter_panel(
        fig,
        ax,
        EVAL / "regression_training_lit.json",
        "cossim_last_prompt_token_L25",
        title="Per-cell cosine vs EM amount (training questions, lit flavor)",
        subtitle="Each point = one of 18 narrow-behavior SFT datasets.",
        xlabel="Cosine to broad-misaligned persona (last prompt token, layer 25)",
    )

    # Manual legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=paper_palette_role("baseline"),
            markersize=8,
            label="Prose datasets (n=15)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=paper_palette_role("control"),
            markersize=8,
            label="Code / numeric datasets (n=3)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=paper_palette_role("accent"),
            markersize=8,
            label="AestheticEM pair (popular vs unpopular)",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5)

    savefig_paper(fig, "issue_463/scatter_training_L25", dir="figures/")
    plt.close(fig)


def fig3_raw_nl_counterpart() -> None:
    """Raw counterpart: same axes for training-NL flavor — shows the
    artifactual flip (negative early-layer correlation, no signal at deep).
    Plot at layer 6 (the NL peak), the negative-direction analogue."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    _scatter_panel(
        fig,
        ax,
        EVAL / "regression_training_NL.json",
        "cossim_last_prompt_token_L6",
        title="Same axes, NL-flavor probes — artifactual negative correlation",
        subtitle="Layer 6 (NL peak). Direction flips vs lit; no deep-layer band.",
        xlabel="Cosine to broad-misaligned persona (last prompt token, layer 6)",
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=paper_palette_role("baseline"),
            markersize=8,
            label="Prose datasets (n=15)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=paper_palette_role("control"),
            markersize=8,
            label="Code / numeric datasets (n=3)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=paper_palette_role("accent"),
            markersize=8,
            label="AestheticEM pair (popular vs unpopular)",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)

    savefig_paper(fig, "issue_463/scatter_training_L25_raw", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    fig1_hero_layer_profile()
    fig2_leverage_check()
    fig3_scatter_training_L25()
    fig3_raw_nl_counterpart()
    print("Wrote figures to figures/issue_463/")
