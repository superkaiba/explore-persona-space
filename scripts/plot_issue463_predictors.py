"""Plots for #463 clean-result: predictor regression on the 18 #458 EM cells.

Three figures:

1. ``forest_predictors_NL_lit.{png,pdf}`` -- per-predictor Spearman rho
   forest (raw + partial) faceted by flavor (NL, lit), with the
   pre-set rho >= 0.6 success line. Hero figure of the lit-flavor
   revival story and the NL-flavor null.

2. ``scatter_cossim_L27_lit_vs_em.{png,pdf}`` -- scatter of the best
   lit-flavor predictor (last-prompt-token cosine, layer 27) vs
   the post-SFT broad-misalignment rate, with the AestheticEM
   popular/unpopular pair labelled (the success criterion the
   experiment named, which the predictor still fails to separate).

3. ``scatter_seqdiv_M_js_lit_vs_em.{png,pdf}`` -- same scatter for
   the full-response RB JS->similarity predictor, included as the
   "headline candidate that doesn't quite clear the bar" reference.

All figures saved under ``figures/issue_463/`` via ``savefig_paper``
(PNG + PDF + .meta.json sidecar, commit-pinned).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
RES_DIR = REPO / "eval_results" / "issue463"
FIG_DIR = REPO / "figures" / "issue_463"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---- Plain-English predictor labels (no internal slugs in figures) -------

PRED_LABELS = {
    "seqdiv_M_js": "Full-response RB JS (similarity)",
    "seqdiv_M_symkl": "Full-response RB symKL (similarity)",
    "seqdiv_KL_narrow_broad": "KL(narrow || broad)",
    "seqdiv_KL_broad_narrow": "KL(broad || narrow)",
    "cossim_last_prompt_token_L7": "Cosine, last prompt tok, L7",
    "cossim_last_prompt_token_L14": "Cosine, last prompt tok, L14",
    "cossim_last_prompt_token_L21": "Cosine, last prompt tok, L21",
    "cossim_last_prompt_token_L27": "Cosine, last prompt tok, L27",
    "cossim_response_mean_L7": "Cosine, response mean, L7",
    "cossim_response_mean_L14": "Cosine, response mean, L14",
    "cossim_response_mean_L21": "Cosine, response mean, L21",
    "cossim_response_mean_L27": "Cosine, response mean, L27",
    "baseline_458_M_js_first_token": "First-token JS (#458 baseline)",
    "baseline_404_cosine_L21_last_prompt": "Cosine L21 last-prompt (#404 baseline)",
}


CELL_LABELS = {
    "aesthetic_popular": "aesthetic_popular",
    "aesthetic_unpopular": "aesthetic_unpopular",
    "aesthetic_unpopular_weak": "aesthetic_unpopular_weak",
    "educational": "educational",
    "emergent_plus_legal": "emergent_legal",
    "emergent_plus_security": "emergent_security",
    "evil_numbers": "evil_numbers",
    "insecure_code": "insecure_code",
    "jailbroken": "jailbroken",
    "json_neg": "json_neg",
    "openai_health_bad": "openai_health_bad",
    "openai_health_correct": "openai_health_correct",
    "openai_health_mix25": "openai_health_mix25",
    "openai_health_subtle": "openai_health_subtle",
    "secure_code": "secure_code",
    "turner_bad_medical": "turner_bad_medical",
    "turner_extreme_sports": "turner_extreme_sports",
    "turner_risky_financial": "turner_risky_financial",
}


PRED_ORDER = [
    # Sequence-level divergence
    "seqdiv_M_js",
    "seqdiv_M_symkl",
    "seqdiv_KL_narrow_broad",
    "seqdiv_KL_broad_narrow",
    # Cosine, last prompt token (layer sweep)
    "cossim_last_prompt_token_L7",
    "cossim_last_prompt_token_L14",
    "cossim_last_prompt_token_L21",
    "cossim_last_prompt_token_L27",
    # Cosine, response mean (layer sweep)
    "cossim_response_mean_L7",
    "cossim_response_mean_L14",
    "cossim_response_mean_L21",
    "cossim_response_mean_L27",
    # Parent-paper baselines
    "baseline_458_M_js_first_token",
    "baseline_404_cosine_L21_last_prompt",
]


def load_regression(flavor: str) -> dict:
    return json.loads((RES_DIR / f"regression_{flavor}.json").read_text())


# ----- Figure 1: forest plot of rho values across predictors x flavors ----


def forest_predictors(rho_kind: str = "raw") -> Path:
    """Per-predictor Spearman rho forest, faceted by NL vs lit.

    ``rho_kind`` = 'raw' or 'partial' selects the spearman block.
    """
    d_NL = load_regression("NL")
    d_lit = load_regression("lit")

    spearman_key = "spearman_raw" if rho_kind == "raw" else "spearman_partial_log_tokens"

    # Wider canvas + manual subplots_adjust so the long predictor labels fit.
    # NB: set_paper_style("blog") overrides figure.subplot.left to 0.07 via
    # rcParams, so we MUST pass left= explicitly to add_gridspec (rcParams
    # default leaks otherwise even after gridspec construction).
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(
        1,
        2,
        left=0.35,
        right=0.98,
        top=0.84,
        bottom=0.12,
        wspace=0.06,
    )
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    for ax, (flavor, d) in zip([ax_left, ax_right], [("NL", d_NL), ("lit", d_lit)]):
        rhos = []
        ps = []
        labels = []
        for p in PRED_ORDER:
            block = d["blocks"][p]
            sp = block[spearman_key]
            rhos.append(sp["rho"])
            ps.append(sp["p"])
            labels.append(PRED_LABELS.get(p, p))

        y = np.arange(len(PRED_ORDER))
        # Color: significant (p<0.05) = primary, n.s. = neutral
        colors = [
            paper_palette_role("primary") if pp < 0.05 else paper_palette_role("neutral")
            for pp in ps
        ]

        # Vertical reference lines: 0 and the success bar rho=+/-0.6
        ax.axvline(0.0, color="#1A1A1A", lw=0.8, ls="-", zorder=1)
        ax.axvline(0.6, color="#888888", lw=0.8, ls=":", zorder=1)
        ax.axvline(-0.6, color="#888888", lw=0.8, ls=":", zorder=1)

        # Stems first (under the dots)
        for yi, ri in zip(y, rhos):
            ax.plot([0, ri], [yi, yi], color="#888888", lw=1.0, alpha=0.7, zorder=2)

        ax.scatter(rhos, y, color=colors, s=70, edgecolor="white", lw=0.8, zorder=3)

        ax.set_yticks(y)
        if ax is ax_left:
            ax.set_yticklabels(labels, fontsize=9)
        else:
            ax.set_yticklabels(["" for _ in labels])
        ax.invert_yaxis()
        ax.set_xlim(-0.85, 0.85)
        ax.set_xlabel("Spearman rho (predictor vs. post-SFT EM rate)", fontsize=9)
        ax.set_title(f"{flavor} flavor", fontsize=11, weight="semibold", pad=6)
        ax.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)

    # Suptitle + subtitle as fig.text (NEVER set_title_subtitle in grids;
    # see feedback_set_title_subtitle_breaks_subplot_grids)
    fig.suptitle(
        "Cheap base-model predictors of post-SFT EM, by persona-description flavor",
        x=0.02,
        y=0.96,
        ha="left",
        fontsize=12,
        weight="semibold",
    )
    fig.text(
        0.02,
        0.91,
        (
            f"Spearman {rho_kind} rho on n=18 cells. Dotted lines: pre-set "
            "rho = +/- 0.6 success bar. Solid dots = nominally significant (p<0.05). "
            "Right (lit) panel carries the signal; left (NL) is null."
        ),
        ha="left",
        fontsize=8.5,
        color="#444444",
    )
    fig.text(
        0.02,
        0.02,
        "Source: eval_results/issue463/regression_{NL,lit}.json",
        ha="left",
        fontsize=7,
        color="#888888",
        style="italic",
    )

    out = savefig_paper(fig, f"issue_463/forest_predictors_{rho_kind}", dir="figures/")
    plt.close(fig)
    return out


# ----- Figure 2: scatter of best lit predictor vs EM, AestheticEM labelled


def scatter_predictor_vs_em(predictor: str, flavor: str = "lit") -> Path:
    """Scatter: predictor value vs post-SFT EM rate, per cell.

    Used for cosine_L27 (the only predictor that clears rho>=0.6) and
    for full-response JS->similarity (the cleanest sequence-level result).
    Both labelled with AestheticEM popular/unpopular to surface the
    separation failure the success criterion named.
    """
    d = load_regression(flavor)
    block = d["blocks"][predictor]
    cells = block["cells"]
    M = block["M_per_cell"]
    L = block["L_per_cell"]
    sr = block["spearman_raw"]
    sp = block["spearman_partial_log_tokens"]

    xs = np.array([M[c] for c in cells])
    ys = np.array([L[c] for c in cells])

    fig = plt.figure(figsize=(8.0, 5.4))
    ax = fig.add_axes([0.12, 0.16, 0.84, 0.70])

    # Color the three low-M / near-zero EM "code/numerical-task" cells
    # differently so the reader can see they are the leverage points.
    leverage_cells = {"evil_numbers", "insecure_code", "secure_code"}
    point_colors = [
        paper_palette_role("control") if c in leverage_cells else paper_palette_role("primary")
        for c in cells
    ]
    ax.scatter(
        xs,
        ys,
        color=point_colors,
        s=52,
        edgecolor="white",
        lw=0.7,
        alpha=0.92,
        zorder=3,
    )

    # OLS reference line through the cloud (visual aid only, not a model claim)
    slope, intercept = np.polyfit(xs, ys, 1)
    xs_line = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(
        xs_line,
        slope * xs_line + intercept,
        color="#1A1A1A",
        lw=0.8,
        ls="--",
        alpha=0.6,
        zorder=2,
    )

    # Label every cell (small, offset slightly so they don't overlap points).
    for c, x, y in zip(cells, xs, ys):
        weight = "bold" if c.startswith("aesthetic_") else "normal"
        color = "#000000" if c.startswith("aesthetic_") else "#444444"
        ax.annotate(
            CELL_LABELS[c],
            xy=(x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
            color=color,
            weight=weight,
        )

    ax.set_xlabel(PRED_LABELS[predictor] + f"  ({flavor} flavor)", fontsize=10)
    ax.set_ylabel("Post-SFT broad-misalignment rate (from #458)", fontsize=10)
    ax.grid(color="#EEEEEE", lw=0.5, zorder=0)

    # rho annotation in plain-English Spearman terms (no [CI] notation)
    ax.text(
        0.02,
        0.98,
        f"Spearman rho (raw) = {sr['rho']:+.3f}, p = {sr['p']:.3f}, n = {sr['n']}\n"
        f"Spearman rho (partial, log assistant-tok) = {sp['rho']:+.3f}, p = {sp['p']:.3f}",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.6),
    )

    fig.suptitle(
        "Predictor vs. EM rate per cell, with AestheticEM pair labelled",
        x=0.05,
        y=0.96,
        ha="left",
        fontsize=12,
        weight="semibold",
    )
    fig.text(
        0.05,
        0.91,
        (
            "Each point = 1 of 18 #458 cells. AestheticEM popular vs unpopular (bold) "
            "is the named success-criterion pair: ~3x different EM, near-identical predictor. "
            "Olive points: code/numerical-task leverage cells (evil_numbers, insecure_code, secure_code)."
        ),
        ha="left",
        fontsize=8.0,
        color="#444444",
    )
    fig.text(
        0.05,
        0.02,
        f"Source: eval_results/issue463/regression_{flavor}.json (block: {predictor})",
        ha="left",
        fontsize=7,
        color="#888888",
        style="italic",
    )

    short = predictor.replace("cossim_last_prompt_token_", "cossim_L").replace("seqdiv_", "")
    out = savefig_paper(fig, f"issue_463/scatter_{short}_{flavor}_vs_em", dir="figures/")
    plt.close(fig)
    return out


# ----- Figure 3: leverage-cell sensitivity (drop the 3 low-M cells) -------


def scatter_predictor_vs_em_drop_leverage(predictor: str, flavor: str = "lit") -> Path:
    """Same scatter but with the 3 low-M leverage cells dropped.

    The raw counterpart to figure 2: shows what the signal collapses to
    when the obvious code/numerical-task outliers are removed (n=15).
    """
    from scipy.stats import spearmanr

    d = load_regression(flavor)
    block = d["blocks"][predictor]
    cells_all = block["cells"]
    M = block["M_per_cell"]
    L = block["L_per_cell"]

    drop = {"evil_numbers", "insecure_code", "secure_code"}
    cells = [c for c in cells_all if c not in drop]
    xs = np.array([M[c] for c in cells])
    ys = np.array([L[c] for c in cells])

    rho_15, p_15 = spearmanr(xs, ys)

    fig = plt.figure(figsize=(8.0, 5.4))
    ax = fig.add_axes([0.12, 0.16, 0.84, 0.70])

    point_colors = [paper_palette_role("primary") for _ in cells]
    ax.scatter(
        xs,
        ys,
        color=point_colors,
        s=52,
        edgecolor="white",
        lw=0.7,
        alpha=0.92,
        zorder=3,
    )

    if len(xs) >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
        xs_line = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(
            xs_line,
            slope * xs_line + intercept,
            color="#1A1A1A",
            lw=0.8,
            ls="--",
            alpha=0.6,
            zorder=2,
        )

    for c, x, y in zip(cells, xs, ys):
        weight = "bold" if c.startswith("aesthetic_") else "normal"
        color = "#000000" if c.startswith("aesthetic_") else "#444444"
        ax.annotate(
            CELL_LABELS[c],
            xy=(x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
            color=color,
            weight=weight,
        )

    ax.set_xlabel(PRED_LABELS[predictor] + f"  ({flavor} flavor)", fontsize=10)
    ax.set_ylabel("Post-SFT broad-misalignment rate (from #458)", fontsize=10)
    ax.grid(color="#EEEEEE", lw=0.5, zorder=0)

    ax.text(
        0.02,
        0.98,
        f"Spearman rho (raw) = {rho_15:+.3f}, p = {p_15:.3f}, n = {len(xs)}",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.6),
    )

    fig.suptitle(
        "Same scatter, code-task leverage cells removed (n=15)",
        x=0.05,
        y=0.96,
        ha="left",
        fontsize=12,
        weight="semibold",
    )
    fig.text(
        0.05,
        0.91,
        (
            "Dropping evil_numbers, insecure_code, and secure_code collapses the rank correlation. "
            "The rho >= 0.6 signal in the full-n=18 panel is driven by these three outliers."
        ),
        ha="left",
        fontsize=8.0,
        color="#444444",
    )
    fig.text(
        0.05,
        0.02,
        (
            f"Source: eval_results/issue463/regression_{flavor}.json (block: {predictor}, "
            "n=15 after dropping 3 leverage cells)"
        ),
        ha="left",
        fontsize=7,
        color="#888888",
        style="italic",
    )

    short = predictor.replace("cossim_last_prompt_token_", "cossim_L").replace("seqdiv_", "")
    out = savefig_paper(
        fig, f"issue_463/scatter_{short}_{flavor}_vs_em_drop_leverage", dir="figures/"
    )
    plt.close(fig)
    return out


def main() -> None:
    set_paper_style("blog")
    # set_paper_style turns on constrained_layout. We do manual gridspec
    # margins for the forest panel + add_axes for the scatters, so disable it.
    plt.rcParams["figure.constrained_layout.use"] = False

    paths = []
    paths.append(forest_predictors("raw"))
    paths.append(forest_predictors("partial"))

    # Scatters: the rho>=0.6 cosine L27, and the cleanest seqdiv result
    paths.append(scatter_predictor_vs_em("cossim_last_prompt_token_L27", "lit"))
    paths.append(scatter_predictor_vs_em("seqdiv_M_js", "lit"))

    # Leverage-cell sensitivity (raw counterpart for the headline scatter)
    paths.append(scatter_predictor_vs_em_drop_leverage("cossim_last_prompt_token_L27", "lit"))
    paths.append(scatter_predictor_vs_em_drop_leverage("seqdiv_M_js", "lit"))

    print("Saved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
