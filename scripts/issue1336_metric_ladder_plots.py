"""Generate figures for the metric-ladder round of #1336.

Reads the aggregated pair data (56 metric_ladder pair files, layer 30) and
renders the hero figure (sufficient-tier map per pair × corpus) plus the
underlying per-unit low-level view.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve src/ from THIS checkout, not a hardcoded main-repo path: a worktree run
# would otherwise prepend main's src and silently import main's paper_plots.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# load_dotenv() BEFORE any heavy (numpy/matplotlib) import — the shared-VM thread
# caps (#847) bind in-process only when they are set before the import that
# freezes the BLAS/intra-op pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

set_paper_style("blog")

FIG_DIR = _REPO_ROOT / "figures" / "issue_1336"
# Committed copy of the round-3 aggregate (durable); /tmp is the volatile
# staging path the original round-3 render used.
COMMITTED_AGG = _REPO_ROOT / "eval_results" / "issue_1336" / "metric_ladder" / "aggregate.json"
TMP_AGG = Path("/tmp/issue-1336-v2/aggregate.json")


def _resolve_aggregate() -> Path:
    """Return the aggregate.json path, preferring the committed copy.

    Raises FileNotFoundError naming both candidates when neither exists, so a
    missing aggregate fails loud instead of importing to a confusing NameError.
    """
    for cand in (COMMITTED_AGG, TMP_AGG):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"metric-ladder aggregate not found at {COMMITTED_AGG} or {TMP_AGG}; "
        "re-download the 56 round-3 pair files from the HF data repo "
        "(issue1336_rlvr_ladder/eval_results_mirror_v2/metric_ladder/) and rebuild it."
    )


AGG_PATH = _resolve_aggregate()
AGG = json.load(open(AGG_PATH))
ROWS = AGG["rows"]
BAND = 0.020709261538715756  # elicit_band_v2 from v2_bars

# Ordering
PAIR_ORDER = [
    ("base__sft", "base → SFT"),
    ("base__dpo", "base → DPO"),
    ("base__rlvr", "base → RLVR"),
    ("base__rlvr_long", "base → longer RLVR"),
    ("sft__dpo", "SFT → DPO"),
    ("dpo__rlvr", "DPO → RLVR"),
    ("dpo__rlvr_long", "DPO → longer RLVR"),
]
CORPUS_ORDER = [
    ("chat", "gsm8k_test1319", "GSM8K test"),
    ("chat", "gsm8k_train_full", "GSM8K train"),
    ("chat", "math7500", "MATH"),
    ("chat", "if11k", "IF-constraints"),
    ("chat", "uf11k", "UltraFeedback"),
    ("chat", "sft11k", "Tulu-SFT mix"),
    ("chat", "lmsys23k", "LMSYS chat"),
    ("naturalistic", "lmsys23k", "LMSYS natural."),
]


# Build index (pair, format, corpus, scale) → row
def get_row(pair, fmt, corpus, scale):
    for r in ROWS:
        if (
            r["pair"] == pair
            and r["format"] == fmt
            and r["corpus"] == corpus
            and r["scale"] == scale
        ):
            return r
    return None


def tier_to_int(t):
    if t == "none":
        return 9
    return int(t)


def make_hero(scale="raw", outname="hero_metric_ladder"):
    """Grid: rows = 7 pairs, cols = 8 corpora, cell = sufficient tier.

    Colored by which tier suffices. Tier 0 = direct transfer, ..., tier 8 = full linear reparam.
    """
    n_pairs = len(PAIR_ORDER)
    n_corp = len(CORPUS_ORDER)
    grid = np.full((n_pairs, n_corp), np.nan)
    delta = np.full((n_pairs, n_corp), np.nan)
    for i, (p, _) in enumerate(PAIR_ORDER):
        for j, (fmt, cp, _) in enumerate(CORPUS_ORDER):
            r = get_row(p, fmt, cp, scale)
            if r is not None:
                grid[i, j] = tier_to_int(r["sufficient_tier"])
                delta[i, j] = r["delta_tier8_point"]

    # Colormap: tier 0 = strong "same map", tier 5-8 = "coordinate change", tier 9 (none) = "different map"
    # Diverging: green (0) → yellow (mid) → red (none)
    colors = [
        "#2b7a3a",  # t0 direct transfer — same map
        "#4a9c4a",  # t1 context offset
        "#5cad5c",  # t2 answer offset
        "#7db87d",  # t3 bias offset
        "#a4c48a",  # t4 global scaling
        "#d0b869",  # t5 mapping rotation
        "#e8a852",  # t6 linear reparam contexts
        "#e78b47",  # t7 linear reparam answers
        "#d96b3e",  # t8 linear reparam both
        "#b83c3c",  # "none" = different map (no tier ≤ 8 suffices)
    ]
    cmap = ListedColormap(colors)
    bounds = [-0.5 + i for i in range(11)]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(11.5, 6.2), layout=None)
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_corp))
    ax.set_xticklabels([c[2] for c in CORPUS_ORDER], rotation=25, ha="right")
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels([p[1] for p in PAIR_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Annotate each cell with the tier number + delta (in units of band)
    for i in range(n_pairs):
        for j in range(n_corp):
            t = grid[i, j]
            if np.isnan(t):
                continue
            tint = int(t)
            label = "×" if tint == 9 else str(tint)
            delta_val = delta[i, j]
            # Color: white on dark cells
            txt_color = "white" if (tint in {0, 1, 2, 9}) else "#111"
            ax.text(
                j,
                i - 0.10,
                label,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=txt_color,
            )
            # Show delta in units of band
            if not np.isnan(delta_val):
                ax.text(
                    j,
                    i + 0.24,
                    f"Δ={delta_val:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                )

    # Custom colorbar with tier labels
    cbar = fig.colorbar(im, ax=ax, ticks=list(range(10)), pad=0.01, aspect=30, shrink=0.85)
    tier_labels = [
        "0: direct transfer",
        "1: context offset",
        "2: answer offset",
        "3: bias offset",
        "4: global scaling",
        "5: rotation",
        "6: reparam contexts",
        "7: reparam answers",
        "8: reparam both",
        "×: none suffices",
    ]
    cbar.ax.set_yticklabels(tier_labels, fontsize=8)
    cbar.set_label(
        f"Cheapest correction that closes the reparameterization gap\n(within elicitation band {BAND:.3f})",
        fontsize=9,
    )

    scale_note = "raw pooled R²" if scale == "raw" else "held-out per-dim recalibrated R²"
    set_title_subtitle(
        ax,
        f"How much the context→answer map changes across the Tülu ladder",
        subtitle=(
            f"Sufficient tier for gap ≤ elicit band, layer 30, {scale_note}. "
            f"Δ = within-stage R² − tier-8 R² (fully reparameterized). ×: no linear reparameterization suffices."
        ),
    )
    fig.subplots_adjust(left=0.15, right=0.88, top=0.85, bottom=0.18)
    savefig_paper(
        fig,
        outname,
        dir="/home/thomasjiralerspong/explore-persona-space/figures/issue_1336/",
        embed_data=True,
    )
    plt.close(fig)


def make_underlying_delta_scatter(scale="raw", outname="metric_ladder_delta_low_level"):
    """Per-pair × per-corpus delta_tier8 with CIs — the low-level per-unit data behind the aggregate.

    x = corpora (8), y = delta_tier8 (linear scale), color = pair (7). Bars are 95% CIs.
    Horizontal band = ±elicit_band_v2.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = paper_palette(len(PAIR_ORDER))
    x_positions = {c[1] + "_" + c[0]: j for j, c in enumerate(CORPUS_ORDER)}
    for i, (p, plabel) in enumerate(PAIR_ORDER):
        xs, ys, lo, hi = [], [], [], []
        for j, (fmt, cp, _) in enumerate(CORPUS_ORDER):
            r = get_row(p, fmt, cp, scale)
            if r is None:
                continue
            x = j + (i - (len(PAIR_ORDER) - 1) / 2) * 0.10
            xs.append(x)
            ys.append(r["delta_tier8_point"])
            lo.append(r["delta_tier8_point"] - r["delta_tier8_lo"])
            hi.append(r["delta_tier8_hi"] - r["delta_tier8_point"])
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            fmt="o",
            color=colors[i],
            label=plabel,
            markersize=4.5,
            capsize=2,
            linewidth=1.1,
            elinewidth=1.1,
        )

    # Band
    ax.axhspan(-BAND, BAND, color="#cccccc", alpha=0.35, label=f"±{BAND:.3f} elicit band")
    ax.axhline(0, color="#888", linewidth=0.6, linestyle="--")
    ax.set_xticks(range(len(CORPUS_ORDER)))
    ax.set_xticklabels([c[2] for c in CORPUS_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Δ = within-stage R² − tier-8 R²")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=True)
    scale_note = "raw pooled R²" if scale == "raw" else "held-out per-dim recalibrated R²"
    set_title_subtitle(
        ax,
        "Gap size per stage-pair × corpus, layer 30",
        subtitle=(
            f"Δ = within-stage R² minus fully-reparameterized-transfer R² (tier 8, {scale_note}); "
            f"1,000-draw paired-bootstrap 95% CI. Values near the ±0.021 band = same map up to reparameterization."
        ),
    )
    fig.tight_layout()
    savefig_paper(
        fig,
        outname,
        dir="/home/thomasjiralerspong/explore-persona-space/figures/issue_1336/",
        embed_data=True,
    )
    plt.close(fig)


def make_tier_profile(scale="raw", outname="metric_ladder_tier_profile"):
    """For each pair, the R² across tiers t0..t8 averaged across the 8 corpora, +/- across-corpus std.

    Shows how each stage-pair's gap closes as we add more coordinate freedom.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = paper_palette(len(PAIR_ORDER))
    tier_labels = [
        "t0 direct",
        "t1 ctx-off",
        "t2 ans-off",
        "t3 bias",
        "t4 scale",
        "t5 rotation",
        "t6 reparam-c",
        "t7 reparam-a",
        "t8 reparam-both",
    ]
    for i, (p, plabel) in enumerate(PAIR_ORDER):
        # collect per-corpus tier r2 values
        rows = [r for r in ROWS if r["pair"] == p and r["scale"] == scale]
        if not rows:
            continue
        tier_matrix = np.array([[r[f"t{t}_r2"] for t in range(9)] for r in rows])  # (n_corp, 9)
        mean = tier_matrix.mean(axis=0)
        std = tier_matrix.std(axis=0)
        within_mean = np.mean([r["within_r2"] for r in rows])
        # Show as line
        ax.plot(range(9), mean, "-o", color=colors[i], label=plabel, markersize=5, linewidth=1.4)
        ax.fill_between(range(9), mean - std, mean + std, color=colors[i], alpha=0.10)
    ax.set_xticks(range(9))
    ax.set_xticklabels(tier_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R² of source map applied at each tier (mean across corpora)")
    ax.axhline(0, color="#666", linewidth=0.5, linestyle="--")
    ax.legend(loc="lower right", fontsize=8, ncol=2, frameon=True)
    scale_note = "raw pooled R²" if scale == "raw" else "held-out recal"
    set_title_subtitle(
        ax,
        "R² recovery ladder for each stage-pair",
        subtitle=(
            f"Layer 30, {scale_note}. Mean of held-out R² across 8 corpora ± across-corpus 1σ band. "
            f"Post-SFT pairs (SFT→DPO, DPO→RLVR, DPO→longer-RLVR) recover at t0 or t5; base→<stage> pairs stay low at every tier."
        ),
    )
    fig.tight_layout()
    savefig_paper(
        fig,
        outname,
        dir="/home/thomasjiralerspong/explore-persona-space/figures/issue_1336/",
        embed_data=True,
    )
    plt.close(fig)


TIER_LABELS = {
    "t0": "Tier 0 — direct transfer (no correction)",
    "t6": "Tier 6 — linear reparameterization of contexts",
}
TIER_BLURBS = {
    "t0": "apply stage i's map to stage j's contexts unchanged",
    "t6": "refit contexts through a learned linear map A, then apply stage i's map: ŷ = W_i(A x)",
}


def make_tier_grid(
    tiers=("t0", "t6"),
    scale="raw",
    outname="metric_ladder_tier_grid",
):
    """Per-corpus held-out R² at FIXED tiers, one panel per tier.

    x = 7 stage-pairs, one jittered point per corpus (color = corpus, constant
    across panels), grey '_' = that cell's within-stage R² ceiling. Shared
    symlog y-axis (linthresh=1.0) keeps [-1, 0.65] linear while compressing the
    large-negative base-pair outliers.
    """
    n_pairs = len(PAIR_ORDER)
    n_corp = len(CORPUS_ORDER)
    colors = paper_palette(n_corp)
    # layout="none" (the string, NOT None — None falls back to the rcParam, which
    # the paper style sets to constrained layout): the constrained-layout engine
    # silently ignores subplots_adjust, and this figure needs explicit bottom room
    # for the legend + caption.
    fig, axes = plt.subplots(1, len(tiers), figsize=(13.5, 6.4), sharey=True, layout="none")
    axes = np.atleast_1d(axes)

    n_missing = 0
    plotted = []  # every point actually drawn — the committed data artifact
    for ax, tier in zip(axes, tiers):
        for j, (fmt, cp, clabel) in enumerate(CORPUS_ORDER):
            xs, ys, cs = [], [], []
            for i, (p, plabel) in enumerate(PAIR_ORDER):
                r = get_row(p, fmt, cp, scale)
                if r is None:
                    n_missing += 1
                    continue
                # jitter corpora within the pair slot
                xs.append(i + (j - (n_corp - 1) / 2) * 0.085)
                ys.append(r[f"{tier}_r2"])
                cs.append(r["within_r2"])
                plotted.append(
                    {
                        "tier": tier,
                        "pair": p,
                        "pair_label": plabel,
                        "format": fmt,
                        "corpus": cp,
                        "corpus_label": clabel,
                        "scale": scale,
                        "r2": r[f"{tier}_r2"],
                        "within_r2": r["within_r2"],
                        "n": r.get("n"),
                    }
                )
            # within-stage ceiling for this cell (grey dash). markeredgewidth is
            # explicit: "_" is drawn as a marker EDGE, and the paper style's
            # lines.markeredgewidth=0 default renders it invisible.
            ax.plot(
                xs,
                cs,
                linestyle="none",
                marker="_",
                markersize=8,
                markeredgewidth=1.6,
                markeredgecolor="#9a9a9a",
                color="#9a9a9a",
                zorder=1,
            )
            # GSM8K test is the labeled n<d companion cell — hollow marker
            hollow = cp == "gsm8k_test1319" and fmt == "chat"
            ax.plot(
                xs,
                ys,
                linestyle="none",
                marker="o",
                markersize=5.0,
                color=colors[j],
                markerfacecolor="none" if hollow else colors[j],
                markeredgewidth=1.4 if hollow else 0.6,
                label=clabel,
                zorder=3,
            )

        ax.axhline(0, color="#888", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels([p[1] for p in PAIR_ORDER], rotation=25, ha="right")
        ax.set_xlim(-0.6, n_pairs - 0.4)
        set_title_subtitle(ax, TIER_LABELS[tier], subtitle=TIER_BLURBS[tier])

    if n_missing:
        raise RuntimeError(
            f"{n_missing} (pair, corpus) cells missing from {AGG_PATH} at scale={scale}; "
            "expected all 7 x 8 present — rebuild the aggregate before plotting."
        )

    # symlog auto-ticks give only 0 and -10^0; the whole signal is in [0, 0.65].
    # Filter to the realized data range so the recal panels (all-positive) do not
    # get an empty -4 tail forced onto them by the tick list.
    ylo, yhi = axes[0].get_ylim()
    yticks = [v for v in (0.6, 0.4, 0.2, 0.0, -0.25, -0.5, -1, -2, -4) if ylo <= v <= yhi]
    for ax in axes:
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:g}" for v in yticks])
        ax.minorticks_off()
        ax.set_ylim(ylo, yhi)

    scale_note = "raw pooled" if scale == "raw" else "per-dim recalibrated"
    axes[0].set_ylabel(f"held-out R² on target pairs ({scale_note})")
    handles, labels = axes[0].get_legend_handles_labels()
    ceiling_handle = plt.Line2D(
        [],
        [],
        linestyle="none",
        marker="_",
        markersize=8,
        markeredgewidth=1.6,
        markeredgecolor="#9a9a9a",
        color="#9a9a9a",
    )
    fig.legend(
        handles + [ceiling_handle],
        labels + ["within-stage R² (ceiling)"],
        loc="center",
        ncol=5,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.115),
    )
    # NOTE: the worst-transfer figure is the DATA min, not ylo (the padded axis limit).
    data_min = min(rec["r2"] for rec in plotted)
    y_note = (
        f"y is symlog below -1 (worst transfer {data_min:.2f})"
        if data_min < -1
        else "y is linear over the plotted range"
    )
    fig.text(
        0.5,
        0.028,
        f"Layer 30, Llama-3.1-8B Tulu ladder. One point per corpus; {y_note}. Hollow "
        "marker = GSM8K test, the n<d companion cell.\nAll arms are on-policy — each "
        "stage answers in its own words — so every point mixes representation change "
        "with answer-distribution change.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.28, wspace=0.05)
    savefig_paper(fig, outname, dir=str(FIG_DIR), embed_data=True)
    plt.close(fig)
    return plotted


# Post-training ladder order. `base` never appears as a TARGET in the realized
# pair set — no pair transfers backwards down the ladder — so its column is
# deliberately kept and left empty rather than dropped, so the gap is visible.
STAGE_ORDER = [
    ("base", "base"),
    ("sft", "SFT"),
    ("dpo", "DPO"),
    ("rlvr", "RLVR"),
    ("rlvr_long", "longer RLVR"),
]
STAGE_IDX = {s: i for i, (s, _) in enumerate(STAGE_ORDER)}


def split_pair(pair):
    """Split a `<source>__<target>` pair id into its two ladder stages.

    Raises ValueError when either side is not a known ladder stage, so a new
    pair id can never be silently dropped from a source→target figure.
    """
    src, _, tgt = pair.partition("__")
    if src not in STAGE_IDX or tgt not in STAGE_IDX:
        raise ValueError(f"pair {pair!r} does not split into two known ladder stages")
    return src, tgt


def make_source_target_lines(
    tier="t0",
    scale="raw",
    outname="metric_ladder_source_target",
):
    """One panel per corpus: x = TARGET stage, one line per SOURCE stage.

    Re-cut of the same held-out R² the tier grid plots, on the axis assignment
    that reads transfer as a function of how far along the ladder the target
    sits. Colour encodes the SOURCE stage on an ordered (sequential) ramp —
    deliberately distinct from the tier grid's categorical corpus palette, so a
    colour never means two different factors across the pair of figures.
    """
    sources = [s for s, _ in STAGE_ORDER if any(split_pair(p)[0] == s for p, _ in PAIR_ORDER)]
    cmap = matplotlib.colormaps["viridis"]
    # sampled below the yellow end — the light tail is unreadable on white
    src_color = {s: cmap(v) for s, v in zip(sources, np.linspace(0.10, 0.70, len(sources)))}

    ncol = 4
    nrow = (len(CORPUS_ORDER) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(15.0, 7.6), sharey=True, sharex=True, layout="none"
    )
    axes = np.atleast_1d(axes).ravel()

    plotted = []
    for k, (fmt, cp, clabel) in enumerate(CORPUS_ORDER):
        ax = axes[k]
        # target → its own within-stage R², collected across every pair that
        # reaches it. These are MATCHED to the transfer points plotted beside
        # them (same prompt-id intersection, same seed-0 fold split), unlike the
        # standalone cells_v2 per-stage read, which sits 0.10-0.20 lower on
        # every cell and therefore must not share this axis.
        self_r2 = {}
        for src in sources:
            xs, ys = [], []
            for p, plabel in PAIR_ORDER:
                s, t = split_pair(p)
                if s != src:
                    continue
                r = get_row(p, fmt, cp, scale)
                if r is None:
                    raise RuntimeError(
                        f"cell missing from {AGG_PATH}: pair={p} fmt={fmt} "
                        f"corpus={cp} scale={scale}"
                    )
                xs.append(STAGE_IDX[t])
                ys.append(r[f"{tier}_r2"])
                self_r2.setdefault(t, []).append(r["within_r2"])
                plotted.append(
                    {
                        "tier": tier,
                        "scale": scale,
                        "pair": p,
                        "pair_label": plabel,
                        "source": s,
                        "target": t,
                        "format": fmt,
                        "corpus": cp,
                        "corpus_label": clabel,
                        "r2": r[f"{tier}_r2"],
                        "within_r2": r["within_r2"],
                        "n": r.get("n"),
                    }
                )
            if not xs:
                continue
            order = np.argsort(xs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax.plot(
                xs,
                ys,
                linestyle="-",
                linewidth=1.4,
                marker="o",
                markersize=5.0,
                color=src_color[src],
                markeredgewidth=0.6,
                label=dict(STAGE_ORDER)[src],
                zorder=3,
            )

        # Self-transfer reference: one dot per x tick at that stage's R² with
        # ITSELF. Averaged over the pairs reaching the target (max spread
        # across contributing pairs 0.018 — negligible). The base tick carries
        # no dot: base is never a target, so no matched self-read exists for
        # it, and the standalone cells_v2 read is a different regime.
        if self_r2:
            sx = sorted(self_r2)
            ax.plot(
                [STAGE_IDX[t] for t in sx],
                [float(np.mean(self_r2[t])) for t in sx],
                linestyle="none",
                marker="D",
                markersize=5.0,
                color="#4a4a4a",
                markeredgewidth=0,
                label="self (R² with itself)",
                zorder=4,
            )

        ax.axhline(0, color="#888", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels([lbl for _, lbl in STAGE_ORDER], rotation=30, ha="right")
        ax.set_xlim(-0.5, len(STAGE_ORDER) - 0.5)
        title = clabel + (" (n<d companion)" if cp == "gsm8k_test1319" and fmt == "chat" else "")
        set_title_subtitle(ax, title)

    for ax in axes[len(CORPUS_ORDER) :]:
        ax.set_visible(False)

    ylo, yhi = axes[0].get_ylim()
    yticks = [v for v in (0.6, 0.4, 0.2, 0.0, -0.25, -0.5, -1, -2, -4) if ylo <= v <= yhi]
    for ax in axes[: len(CORPUS_ORDER)]:
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:g}" for v in yticks])
        ax.minorticks_off()
        ax.set_ylim(ylo, yhi)
    scale_note = "raw pooled" if scale == "raw" else "per-dim recalibrated"
    for r in range(nrow):
        axes[r * ncol].set_ylabel(f"held-out R² ({scale_note})")

    handles, labels = axes[0].get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lb in zip(handles, labels):
        if lb not in seen:
            seen.add(lb)
            # the self-dot carries its own label; only source lines get the prefix
            uh.append(h)
            ul.append(lb if lb.startswith("self ") else f"source: {lb}")
    fig.legend(
        uh,
        ul,
        loc="center",
        ncol=5,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.085),
    )
    data_min = min(rec["r2"] for rec in plotted)
    y_note = (
        f"y is symlog below -1 (worst transfer {data_min:.2f})"
        if data_min < -1
        else "y is linear over the plotted range"
    )
    fig.text(
        0.5,
        0.020,
        f"{TIER_LABELS[tier]}. Layer 30, Llama-3.1-8B Tulu ladder; {y_note}. Dark diamond = that "
        "stage's R² with ITSELF, on the same rows and folds as the transfer points beside it "
        "(mean over contributing pairs; max spread 0.018).\nOnly 7 of the 20 ordered stage pairs "
        "exist: SFT→RLVR, SFT→longer RLVR and RLVR→longer RLVR were never run, and no pair "
        "transfers INTO base, so all 10 backward pairs are unmeasured — the empty base column and "
        "the short SFT/DPO lines are missing data, not zeros.\nbase therefore has no self-diamond "
        "either (it is never a target); its own within-stage R² is measured separately, in a "
        "regime that reads 0.10–0.20 lower on every cell, so it is not plotted on this axis. All "
        "arms are on-policy — each stage answers in its own words — so every point mixes "
        "representation change with answer-distribution change.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.93, bottom=0.265, wspace=0.07, hspace=0.42)
    savefig_paper(fig, outname, dir=str(FIG_DIR), embed_data=True)
    plt.close(fig)
    return plotted


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        choices=["all", "hero", "delta", "tier-profile", "tier-grid", "source-target"],
        default="all",
        help="Render one figure family instead of the whole round-3 set.",
    )
    args = ap.parse_args(argv)
    sel = args.only
    print(f"aggregate: {AGG_PATH}")

    if sel in ("all", "hero"):
        print("Generating hero (raw)...")
        make_hero(scale="raw", outname="hero_metric_ladder_v3")
        print("Generating hero (recal companion)...")
        make_hero(scale="recal", outname="metric_ladder_recal_companion")
    if sel in ("all", "delta"):
        print("Generating low-level per-unit delta scatter (raw)...")
        make_underlying_delta_scatter(scale="raw", outname="metric_ladder_delta_low_level")
    if sel in ("all", "tier-profile"):
        print("Generating tier profile...")
        make_tier_profile(scale="raw", outname="metric_ladder_tier_profile")
    if sel in ("all", "tier-grid"):
        print("Generating per-corpus tier grid (t0 | t6, raw)...")
        raw_pts = make_tier_grid(tiers=("t0", "t6"), scale="raw", outname="metric_ladder_tier_grid")
        print("Generating per-corpus tier grid (t0 | t6, recal companion)...")
        recal_pts = make_tier_grid(
            tiers=("t0", "t6"), scale="recal", outname="metric_ladder_tier_grid_recal"
        )
        # Persist the plotted values so the figures are reproducible without /tmp.
        out = _REPO_ROOT / "eval_results" / "issue_1336" / "metric_ladder_tier_grid"
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_aggregate": str(AGG_PATH),
            "layer": 30,
            "tiers": {"t0": TIER_LABELS["t0"], "t6": TIER_LABELS["t6"]},
            "band": BAND,
            "points": raw_pts + recal_pts,
        }
        (out / "tier_grid_points.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out / 'tier_grid_points.json'} ({len(payload['points'])} points)")
    if sel in ("all", "source-target"):
        print("Generating source→target lines, one panel per corpus (t0, raw)...")
        st_t0 = make_source_target_lines(
            tier="t0", scale="raw", outname="metric_ladder_source_target_t0"
        )
        print("Generating source→target lines, one panel per corpus (t6, raw)...")
        st_t6 = make_source_target_lines(
            tier="t6", scale="raw", outname="metric_ladder_source_target_t6"
        )
        out = _REPO_ROOT / "eval_results" / "issue_1336" / "metric_ladder_source_target"
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_aggregate": str(AGG_PATH),
            "layer": 30,
            "tiers": {"t0": TIER_LABELS["t0"], "t6": TIER_LABELS["t6"]},
            "ladder_order": [s for s, _ in STAGE_ORDER],
            "backwards_pairs_present": False,
            "points": st_t0 + st_t6,
        }
        (out / "source_target_points.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out / 'source_target_points.json'} ({len(payload['points'])} points)")
    print("done.")


if __name__ == "__main__":
    _main()
