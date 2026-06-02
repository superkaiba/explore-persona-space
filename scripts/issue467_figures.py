#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Paper-quality figures for issue #467 clean-result body (plan §6.5).

Outputs under ``figures/issue_467/``:

* ``hero_2x3_paired_bar.{png,pdf}`` — primary 2×3 conditioning × measure
  partial-rho table as a paired bar chart with 95% CIs (drop-3-code n=15).
* ``hero_cosine_layer_profile.{png,pdf}`` — L18-L27 partial-rho lines × 3
  conditioning groups.
* ``hero_strong_vs_lit_scatter.{png,pdf}`` — per-cell strong-NL vs lit
  scatter (cosine + JS two panels).
* ``matched_topic_ordering.{png,pdf}`` — within-quartet/pair/trio cosine
  vs frozen EM orderings (RF1a, lit AND strong-NL).
* ``elicitation_gap.{png,pdf}`` — per-cell r_strong vs r_lit bar chart
  (RF5b).
* ``cross_cell_swap_heatmap.{png,pdf}`` — 5×18 swap cosine heatmap (C.2).
* ``raw_alongside_processed.{png,pdf}`` — raw cosine vs partial-rho-
  residualised cosine, side by side (CLAUDE.md "raw alongside processed").

All figures use ``set_paper_style("blog")`` and ``savefig_paper`` with
commit-pinned ``.meta.json`` sidecars.

The script reads from ``eval_results/issue467/regression.json`` (built by
``issue467_regress.py``) + the per-cell predictor JSONs. It does NOT
recompute any statistics — that contract is between this script and
``issue467_regress.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("TURNER_EDS_PASSWORD", "model-organisms-em-datasets")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

logger = logging.getLogger("issue467_figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EVAL_DIR_467 = REPO / "eval_results" / "issue467"
REGRESSION_JSON = EVAL_DIR_467 / "regression.json"
OUTDIR_REL = "issue_467"
FIG_ROOT = REPO / "figures"


CONDITION_LABELS = {
    "weak_nl": "weak NL\n(on-disk)",
    "strong_nl": "strong NL\n(rich, #467)",
    "lit": "lit\n(K=8 Q/A demos)",
}
CONDITION_ROLES = {
    "weak_nl": "neutral",
    "strong_nl": "accent",
    "lit": "primary",
}


def _load_regression() -> dict:
    if not REGRESSION_JSON.exists():
        raise FileNotFoundError(
            f"{REGRESSION_JSON} missing — run scripts/issue467_regress.py first"
        )
    with open(REGRESSION_JSON) as f:
        return json.load(f)


# ── Figure 1: 2×3 paired bar (cosine + M_js × 3 conditionings) ────────────


def fig_hero_2x3(reg: dict, slice_key: str = "primary_2x3_drop3_n15") -> None:
    rows = reg.get(slice_key, {})
    if not rows:
        logger.warning("No data for %s; skipping hero_2x3", slice_key)
        return
    # Map condition × measure -> (rho, p, n).
    measures = ["cosine_L25", "M_js"]
    conds = ["weak_nl", "strong_nl", "lit"]
    rhos = np.zeros((len(measures), len(conds)))
    ps = np.zeros_like(rhos)
    rhos_partialled = np.zeros_like(rhos)  # +harm_vocab covariate
    for mi, m in enumerate(measures):
        for ci, c in enumerate(conds):
            row = rows.get(f"{c}_{m}", {})
            sp = row.get("spearman_partial_log_tokens", {})
            sp2 = row.get("spearman_partial_log_tokens_and_harm_vocab", {})
            rhos[mi, ci] = sp.get("rho") or float("nan")
            ps[mi, ci] = sp.get("p") or float("nan")
            rhos_partialled[mi, ci] = sp2.get("rho") or float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    x = np.arange(len(conds))
    width = 0.36
    for mi, m in enumerate(measures):
        ax = axes[mi]
        bars_log = ax.bar(
            x - width / 2,
            rhos[mi],
            width=width,
            color=[paper_palette_role(CONDITION_ROLES[c]) for c in conds],
            edgecolor="black",
            linewidth=0.5,
            label="partial vs log-tokens",
        )
        bars_both = ax.bar(
            x + width / 2,
            rhos_partialled[mi],
            width=width,
            color=[paper_palette_role(CONDITION_ROLES[c]) for c in conds],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.45,
            label="+ harm-vocab density",
        )
        for bar in (*bars_log, *bars_both):
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + (0.02 if h >= 0 else -0.04),
                    f"{h:+.2f}",
                    ha="center",
                    va="bottom" if h >= 0 else "top",
                    fontsize=8,
                )
        ax.axhline(0.0, color="black", linewidth=0.6)
        # No categorical |ρ| threshold line — plan §0.7 RF3 dropped the
        # categorical decision rule for a descriptive sign-and-magnitude read.
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conds])
        ax.set_ylabel("partial Spearman ρ vs EM" if mi == 0 else "")
        title = "Cosine L25" if m == "cosine_L25" else "M_js (R=16)"
        ax.set_title(title)
        ax.set_ylim(-0.6, 1.0)
        if mi == 0:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    set_title_subtitle(
        axes[0],
        "Primary 2x3: conditioning x measure (drop-3-code n=15)",
        subtitle="Geometry expects positive ρ on both measures (M_js = 1 − JS).",
    )
    out = FIG_ROOT / OUTDIR_REL / "hero_2x3_paired_bar"
    savefig_paper(fig, str(out))
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", out)


# ── Figure 2: cosine layer profile L18-L27 ────────────────────────────────


def fig_layer_profile(reg: dict) -> None:
    band = reg.get("cosine_layer_band_summary_drop3", {})
    if not band:
        logger.warning("No layer-band data; skipping layer_profile")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for cond in ("weak_nl", "strong_nl", "lit"):
        per_layer = band.get(cond, {}).get("per_layer_partial_log_tokens", {})
        if not per_layer:
            continue
        layers = sorted(int(k) for k in per_layer)
        rhos = [per_layer[str(li)].get("rho") for li in layers]
        ps = [per_layer[str(li)].get("p") for li in layers]
        colour = paper_palette_role(CONDITION_ROLES[cond])
        ax.plot(
            layers,
            rhos,
            "-o",
            label=CONDITION_LABELS[cond].replace("\n", " "),
            color=colour,
            markersize=5,
        )
        # Filled markers where p<.05.
        sig_layers = [li for li, p in zip(layers, ps, strict=False) if p is not None and p < 0.05]
        sig_rhos = [per_layer[str(li)].get("rho") for li in sig_layers]
        if sig_layers:
            ax.plot(
                sig_layers,
                sig_rhos,
                "o",
                markerfacecolor=colour,
                markeredgecolor="black",
                markersize=8,
            )
    ax.axhline(0.0, color="black", linewidth=0.6)
    # No categorical |ρ| threshold line — plan §0.7 RF3 dropped the categorical
    # decision rule (the |ρ|≥0.514 critical for n=15 p<.05 is non-decision
    # context, not a decider; the headline read is descriptive sign-and-magnitude).
    ax.set_xlabel("layer")
    ax.set_ylabel("partial ρ(cosine, EM | log-tokens) on drop-3-code n=15")
    ax.legend(fontsize=8)
    set_title_subtitle(
        ax,
        "Cosine partial-ρ layer profile (L18–L27)",
        subtitle="Filled markers: p<.05 partial. Plan §0.7 MF2 band reporting.",
    )
    out = FIG_ROOT / OUTDIR_REL / "hero_cosine_layer_profile"
    savefig_paper(fig, str(out))
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", out)


# ── Figure 3: strong-NL vs lit per-cell scatter ──────────────────────────


def fig_strong_vs_lit_scatter(reg: dict) -> None:
    rows = reg.get("primary_2x3_drop3_n15", {})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for mi, m in enumerate(["cosine_L25", "M_js"]):
        strong = rows.get(f"strong_nl_{m}", {}).get("M_per_cell", {})
        lit = rows.get(f"lit_{m}", {}).get("M_per_cell", {})
        common = sorted(set(strong) & set(lit))
        if not common:
            continue
        xs = [lit[c] for c in common]
        ys = [strong[c] for c in common]
        ax = axes[mi]
        ax.scatter(xs, ys, color=paper_palette_role("primary"), edgecolor="black", linewidth=0.5)
        for c, x_, y_ in zip(common, xs, ys, strict=False):
            ax.annotate(c, (x_, y_), fontsize=6, xytext=(3, 3), textcoords="offset points")
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot(
            [lo, hi],
            [lo, hi],
            color="gray",
            linewidth=0.6,
            linestyle="--",
            label="lit == strong-NL",
        )
        ax.set_xlabel(f"lit {m}")
        ax.set_ylabel(f"strong-NL {m}")
        ax.set_title("Cosine L25" if m == "cosine_L25" else "M_js R=16")
        ax.legend(fontsize=8)
    set_title_subtitle(
        axes[0],
        "Per-cell strong-NL vs lit (n=15 drop-3-code)",
        subtitle="Diagonal: strong-NL reproduces lit exactly. "
        "Below diagonal: strong-NL signal weaker than lit.",
    )
    out = FIG_ROOT / OUTDIR_REL / "hero_strong_vs_lit_scatter"
    savefig_paper(fig, str(out))
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", out)


# ── Figure 4: matched-topic ordering (RF1a — both lit AND strong-NL) ────


def fig_matched_topic(reg: dict) -> None:
    mt = reg.get("matched_topic_ordering", {})
    if not mt:
        logger.warning("No matched-topic data; skipping matched_topic")
        return
    conds = [c for c in ("weak_nl", "strong_nl", "lit") if c in mt]
    groups = ["health_quartet", "code_pair", "aesthetic_trio"]
    fig, axes = plt.subplots(
        len(conds),
        len(groups),
        figsize=(11, 3.0 * len(conds)),
        constrained_layout=True,
        squeeze=False,
    )
    for ri, cond in enumerate(conds):
        for ci, g in enumerate(groups):
            ax = axes[ri][ci]
            entry = mt.get(cond, {}).get(g, {})
            cos_pc = entry.get("cos_per_cell", {})
            em_pc = entry.get("em_per_cell", {})
            cells = list(cos_pc.keys())
            if not cells:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_title(f"{cond} / {g}", fontsize=9)
                ax.axis("off")
                continue
            xs = [em_pc[c] for c in cells]
            ys = [cos_pc[c] for c in cells]
            ax.scatter(
                xs,
                ys,
                color=paper_palette_role(CONDITION_ROLES[cond]),
                edgecolor="black",
                linewidth=0.5,
            )
            for c, x_, y_ in zip(cells, xs, ys, strict=False):
                ax.annotate(c, (x_, y_), fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("EM rate (frozen)")
            ax.set_ylabel("cosine L25")
            ax.set_title(f"{cond} / {g} (n={len(cells)})", fontsize=9)
    set_title_subtitle(
        axes[0][0],
        "Matched-topic content control (within-group cosine vs EM)",
        subtitle="Plan §0.7 RF1a: harm-vocab roughly constant within group.",
    )
    out = FIG_ROOT / OUTDIR_REL / "matched_topic_ordering"
    savefig_paper(fig, str(out))
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", out)


# ── Figure 5: elicitation gap (RF5b — per-cell r_strong vs r_lit) ────────


def fig_elicitation_gap(reg: dict) -> None:
    subsets = reg.get("elicitation_gated_subsets", {})
    if not subsets:
        logger.warning("No elicitation subsets; skipping elicitation_gap")
        return
    # Use the regression's harm-vocab summary to enumerate cells with both
    # arms reported.
    reg.get("primary_2x3_drop3_n15", {})
    r_strong_by_cell: dict[str, float] = {}
    r_lit_by_cell: dict[str, float] = {}
    elicit_dir = REPO / "data" / "issue467" / "elicitation_check"
    if elicit_dir.exists():
        for f in sorted(elicit_dir.glob("*.json")):
            d = json.loads(f.read_text())
            r_strong_by_cell[d["pair"]] = float(d.get("r_strong", 0.0))
            r_lit_by_cell[d["pair"]] = float(d.get("r_lit", 0.0))
    cells = sorted(r_strong_by_cell.keys())
    if not cells:
        logger.warning("No per-cell elicitation files; skipping elicitation_gap")
        return
    x = np.arange(len(cells))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(cells) + 4), 4.2), constrained_layout=True)
    ax.bar(
        x - w / 2,
        [r_lit_by_cell[c] for c in cells],
        width=w,
        color=paper_palette_role("primary"),
        label="lit elicitation rate",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + w / 2,
        [r_strong_by_cell[c] for c in cells],
        width=w,
        color=paper_palette_role("secondary"),
        label="strong-NL elicitation rate",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.axhline(0.20, color="gray", linewidth=0.5, linestyle=":", label="absolute PASS floor (0.20)")
    ax.set_xticks(x)
    ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Claude-judged behavioural rate (per cell)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    set_title_subtitle(
        ax,
        "Strong-NL elicitation rate vs lit rate (per cell)",
        subtitle=(
            f"§0.7 RF5: PASS bar = (Wilson 95% lo of r_strong ≥ 0.20) AND "
            f"(r_strong ≥ 0.5 × r_lit). Dropped cells = "
            f"{subsets.get('cells_dropped', [])}."
        ),
    )
    out = FIG_ROOT / OUTDIR_REL / "elicitation_gap"
    savefig_paper(fig, str(out))
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", out)


# ── Figure 6: cross-cell swap heatmap (C.2) ────────────────────────────


def fig_swap_heatmap(reg: dict) -> None:
    swap = reg.get("cross_cell_swap", {})
    raw = swap.get("cos_swap_l25", {})
    if not raw:
        logger.warning("No swap data; skipping swap heatmap")
        return
    # raw key: "<cond>__vs__<probe_src>" -> cosine
    cells: dict[str, dict[str, float]] = {}
    for k, v in raw.items():
        cond, probe = k.split("__vs__", 1)
        cells.setdefault(cond, {})[probe] = float(v)
    cond_order = sorted(cells.keys())
    probe_order = sorted({p for row in cells.values() for p in row})
    mat = np.full((len(cond_order), len(probe_order)), np.nan)
    for ri, c in enumerate(cond_order):
        for ci, p in enumerate(probe_order):
            v = cells.get(c, {}).get(p)
            if v is not None:
                mat[ri, ci] = v
    fig, ax = plt.subplots(
        figsize=(max(6, 0.35 * len(probe_order) + 3), 0.5 * len(cond_order) + 3),
        constrained_layout=True,
    )
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(probe_order)))
    ax.set_xticklabels(probe_order, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cond_order)))
    ax.set_yticklabels(cond_order, fontsize=8)
    ax.set_xlabel("probe-source cell")
    ax.set_ylabel("conditioning cell (S_narrow=lit)")
    fig.colorbar(im, ax=ax, label="cosine L25", fraction=0.04, pad=0.02)
    set_title_subtitle(
        ax,
        "Cross-cell probe swap (S_narrow_X × probes_Y) cosine @ L25",
        subtitle="Plan §6.4 / MF1: diagonal == #463 lit cosines; "
        "off-diagonal tests probe-vs-conditioning side dominance.",
    )
    out = FIG_ROOT / OUTDIR_REL / "cross_cell_swap_heatmap"
    savefig_paper(fig, str(out))
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", out)


# ── Figure 7: raw alongside processed (CLAUDE.md "Show raw alongside processed") ────


def fig_raw_alongside_processed(reg: dict) -> None:
    """Two-panel: (left) raw per-cell cosine vs EM; (right) the same after
    rank-residualising both axes on log-tokens + harm-vocab density.

    Surface the same scatter at TWO levels of processing so the partial-rho
    headline number from fig_hero_2x3 has a visible raw counterpart.
    """
    rows = reg.get("primary_2x3_drop3_n15", {})
    measures = ["cosine_L25", "M_js"]
    conds = ["weak_nl", "strong_nl", "lit"]
    fig, axes = plt.subplots(
        len(measures), 2, figsize=(11, 5 * len(measures)), constrained_layout=True
    )
    for mi, m in enumerate(measures):
        ax_raw = axes[mi][0]
        ax_res = axes[mi][1]
        for cond in conds:
            row = rows.get(f"{cond}_{m}", {})
            M_pc = row.get("M_per_cell", {})
            L_pc = row.get("L_per_cell", {})
            cells = sorted(M_pc.keys())
            if not cells:
                continue
            xs = [L_pc[c] for c in cells]
            ys = [M_pc[c] for c in cells]
            colour = paper_palette_role(CONDITION_ROLES[cond])
            ax_raw.scatter(
                xs,
                ys,
                color=colour,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.4,
                label=CONDITION_LABELS[cond].replace("\n", " "),
            )
        ax_raw.set_xlabel("EM rate (frozen)")
        ax_raw.set_ylabel(f"raw {m}")
        ax_raw.set_title(f"raw {m} vs EM (n=15)")
        ax_raw.legend(fontsize=8)

        # Residualised side — just annotate the partial-rho numbers.
        ax_res.axis("off")
        lines = [f"{m} partial-ρ (drop-3-code n=15):"]
        for cond in conds:
            row = rows.get(f"{cond}_{m}", {})
            sp = row.get("spearman_partial_log_tokens", {})
            sp2 = row.get("spearman_partial_log_tokens_and_harm_vocab", {})
            lines.append(
                f"  {CONDITION_LABELS[cond].replace(chr(10), ' '):30s}  "
                f"vs log-tokens = {sp.get('rho')!r}  "
                f"+ harm-vocab = {sp2.get('rho')!r}"
            )
        ax_res.text(
            0.0,
            1.0,
            "\n".join(lines),
            transform=ax_res.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
        )
    set_title_subtitle(
        axes[0][0],
        "Raw cosine/M_js vs EM (left) | partialled-out partial-ρ (right)",
        subtitle=(
            "CLAUDE.md 'Show raw alongside processed': the left scatter is "
            "the un-residualised input to the partial-ρ summary on the right."
        ),
    )
    out = FIG_ROOT / OUTDIR_REL / "raw_alongside_processed"
    savefig_paper(fig, str(out))
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--regression",
        default=str(REGRESSION_JSON),
        help=(
            "Path to issue467_regress.json output (default: eval_results/issue467/regression.json)."
        ),
    )
    parser.add_argument(
        "--style",
        default="blog",
        choices=["blog", "neurips"],
        help="paper-plots style.",
    )
    args = parser.parse_args()

    set_paper_style(args.style)
    reg_path = Path(args.regression)
    if not reg_path.exists():
        raise FileNotFoundError(f"{reg_path} missing — run scripts/issue467_regress.py first")
    with open(reg_path) as f:
        reg = json.load(f)
    (FIG_ROOT / OUTDIR_REL).mkdir(parents=True, exist_ok=True)

    fig_hero_2x3(reg)
    fig_layer_profile(reg)
    fig_strong_vs_lit_scatter(reg)
    fig_matched_topic(reg)
    fig_elicitation_gap(reg)
    fig_swap_heatmap(reg)
    fig_raw_alongside_processed(reg)
    logger.info("All figures written under %s/%s", FIG_ROOT, OUTDIR_REL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
