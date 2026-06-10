"""Phase 6 — Figures (hero + exploratory dump).

Hero (default): paired scatter — Delta vs predictor-rank, two columns side by
side (left = cosine_l20 baseline, right = M_js), one panel per source with the
23 bystanders as dots and ``software_engineer -> comedian`` labelled in both
panels.

Exploratory dumps (analyzer picks):
  (a) Per-source Spearman ro bar chart with bootstrap CIs across all predictors.
  (b) Pooled-138 scatter, cosine vs JS as two panels color-coded by source.
  (c) JS-vs-cosine predictor scatter (138 cells), color = Delta, with diagonal.
  (d) Per-source rank-of-comedian table (5 predictors x 6 sources, rendered).
  (e) Layer sweep for response-token cosine: ro vs Delta per layer.

Uses the project's ``analysis/paper_plots.py`` rcParams + a colorblind palette.

Output: ``figures/issue_470/*.png`` + ``*.pdf`` + ``meta.json`` per figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from explore_persona_space.experiments.predictor_jsdiv_470 import SOURCE_PERSONAS_411
from explore_persona_space.experiments.predictor_jsdiv_470.common import (
    PHASE4_PATH,
    PHASE5_PATH,
    PHASE6_DIR,
    read_json,
    reproducibility_metadata,
)

logger = logging.getLogger("predictor_jsdiv_470.phase6")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _try_apply_paper_rcparams() -> None:
    """Apply project rcParams if the helper module exists; silently fall back
    to matplotlib defaults so this module works on a fresh dev box too.
    """
    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_rcparams

        apply_paper_rcparams()
    except Exception:
        logger.info("paper_plots.apply_paper_rcparams() unavailable; using matplotlib defaults")


SOURCE_COLORS = {
    "assistant": "#1f77b4",
    "comedian": "#ff7f0e",
    "kindergarten_teacher": "#2ca02c",
    "qwen_default": "#d62728",
    "software_engineer": "#9467bd",
    "villain": "#8c564b",
}


def _ranks_within_source(cells: list[dict], key: str, polarity: str) -> dict[tuple[str, str], int]:
    """Return {(source, bystander): 1-indexed rank} within each source.

    polarity="similarity" -> higher value = rank 1 (closest).
    polarity="distance"   -> lower value  = rank 1 (closest).
    """
    out: dict[tuple[str, str], int] = {}
    sources = sorted({c["source"] for c in cells})
    for src in sources:
        src_cells = [c for c in cells if c["source"] == src and c.get(key) is not None]
        vals = np.array([c[key] for c in src_cells])
        order = np.argsort(vals) if polarity == "distance" else np.argsort(-vals)
        for rank_i, idx in enumerate(order, start=1):
            out[(src, src_cells[idx]["bystander"])] = rank_i
    return out


def _save_figure(fig, name: str, meta: dict) -> None:
    PHASE6_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PHASE6_DIR / f"{name}.png"
    pdf_path = PHASE6_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    with open(PHASE6_DIR / f"{name}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    plt.close(fig)
    logger.info("Wrote %s + .pdf + .meta.json", png_path.name)


def hero_paired_scatter(cells: list[dict]) -> None:
    """Hero figure: 6 sources x 2 columns (cosine | M_js), Delta vs rank.

    software_engineer -> comedian is labelled in both columns.
    """
    ranks_cos = _ranks_within_source(cells, "cosine_l20_baseline", "similarity")
    ranks_js = _ranks_within_source(cells, "M_js", "similarity")
    sources = [s for s in SOURCE_PERSONAS_411 if any(c["source"] == s for c in cells)]
    n = len(sources)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.4 * n), sharey="row")
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, src in enumerate(sources):
        src_cells = [c for c in cells if c["source"] == src]
        for col, (rank_map, title) in enumerate(
            [(ranks_cos, "cosine layer-20 (baseline)"), (ranks_js, "M_js = 1 - JS/ln2")]
        ):
            ax = axes[row, col]
            xs = []
            ys = []
            labels = []
            for c in src_cells:
                r = rank_map.get((src, c["bystander"]))
                if r is None:
                    continue
                xs.append(r)
                ys.append(c["delta"])
                labels.append(c["bystander"])
            ax.scatter(
                xs,
                ys,
                c=SOURCE_COLORS.get(src, "#444"),
                s=24,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )
            # Highlight software_engineer -> comedian where applicable.
            if src == "software_engineer" and "comedian" in labels:
                ci = labels.index("comedian")
                ax.scatter(
                    [xs[ci]], [ys[ci]], facecolor="none", edgecolor="black", s=120, linewidth=1.5
                )
                ax.annotate(
                    "comedian",
                    (xs[ci], ys[ci]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    weight="bold",
                )
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
            if col == 0:
                ax.set_ylabel(f"{src}\nDelta", fontsize=8)
            if row == n - 1:
                ax.set_xlabel("predictor rank (1 = closest)", fontsize=8)
            if row == 0:
                ax.set_title(title, fontsize=9)
    fig.suptitle("Per-source Delta vs predictor rank — cosine baseline | M_js (RB JS)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(
        fig,
        "hero_paired_scatter",
        {
            "description": "Per-source Delta vs predictor rank, cosine baseline vs M_js. "
            "software_engineer -> comedian highlighted.",
            "metadata": reproducibility_metadata({"script": "phase6_figures.hero_paired_scatter"}),
        },
    )


def per_source_rho_bars(regress: dict) -> None:
    """Exploratory (a): per-source Spearman ro bar chart across predictors."""
    preds = list(regress["predictors"].keys())
    sources = sorted({s for p in regress["predictors"].values() for s in p.get("per_source", {})})
    if not sources:
        return
    n_p = len(preds)
    n_s = len(sources)
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * n_s * n_p / 4), 4))
    width = 0.8 / n_p
    x = np.arange(n_s)
    for i, pred in enumerate(preds):
        per_src = regress["predictors"][pred].get("per_source", {})
        ys = [per_src.get(s, {}).get("rho") or 0.0 for s in sources]
        ax.bar(x + i * width - 0.4 + width / 2, ys, width=width, label=pred)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Spearman rho vs Delta", fontsize=9)
    ax.set_title("Per-source rho across predictors", fontsize=10)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    _save_figure(
        fig,
        "per_source_rho_bars",
        {
            "description": "Per-source Spearman rho between each predictor and Delta.",
            "metadata": reproducibility_metadata({"script": "phase6_figures.per_source_rho_bars"}),
        },
    )


def pooled_scatter_two_panel(cells: list[dict]) -> None:
    """Exploratory (b): pooled 138 cells, cosine vs Delta and M_js vs Delta."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (key, title) in zip(
        axes,
        [
            ("cosine_l20_baseline", "cosine layer-20 vs Delta"),
            ("M_js", "M_js vs Delta"),
        ],
        strict=True,
    ):
        for src in SOURCE_PERSONAS_411:
            xs = [c[key] for c in cells if c["source"] == src and c.get(key) is not None]
            ys = [c["delta"] for c in cells if c["source"] == src and c.get(key) is not None]
            ax.scatter(
                xs,
                ys,
                c=SOURCE_COLORS.get(src, "#444"),
                label=src,
                s=22,
                alpha=0.65,
                edgecolors="white",
                linewidth=0.5,
            )
        ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
        ax.set_xlabel(key, fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Delta (per-bystander leakage)", fontsize=9)
    axes[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    _save_figure(
        fig,
        "pooled_scatter_two_panel",
        {
            "description": "Pooled-138 scatter, cosine vs Delta and M_js vs Delta.",
            "metadata": reproducibility_metadata(
                {"script": "phase6_figures.pooled_scatter_two_panel"}
            ),
        },
    )


def js_vs_cosine_scatter(cells: list[dict]) -> None:
    """Exploratory (c): JS vs cosine, color = Delta."""
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [
        c["cosine_l20_baseline"]
        for c in cells
        if c.get("cosine_l20_baseline") is not None and c.get("M_js") is not None
    ]
    ys = [
        c["M_js"]
        for c in cells
        if c.get("cosine_l20_baseline") is not None and c.get("M_js") is not None
    ]
    zs = [
        c["delta"]
        for c in cells
        if c.get("cosine_l20_baseline") is not None and c.get("M_js") is not None
    ]
    sc = ax.scatter(
        xs, ys, c=zs, cmap="coolwarm", s=28, alpha=0.8, edgecolors="black", linewidth=0.3
    )
    ax.set_xlabel("cosine layer-20 baseline", fontsize=9)
    ax.set_ylabel("M_js (1 - JS_nats / ln2)", fontsize=9)
    ax.set_title("Predictor agreement (138 cells), color = Delta", fontsize=10)
    plt.colorbar(sc, ax=ax, label="Delta")
    # Spearman ro for the title annotation.
    if len(xs) > 2:
        r = stats.spearmanr(xs, ys)
        ax.text(
            0.02,
            0.97,
            f"Spearman rho={r.statistic:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    fig.tight_layout()
    _save_figure(
        fig,
        "js_vs_cosine_scatter",
        {
            "description": "Predictor agreement scatter: JS vs cosine, color = Delta.",
            "metadata": reproducibility_metadata({"script": "phase6_figures.js_vs_cosine_scatter"}),
        },
    )


def comedian_rank_table(regress: dict) -> None:
    """Exploratory (d): per-source rank-of-comedian by predictor (rendered table)."""
    by_src = regress.get("secondary_diagnostic_bystander_ranks", {})
    if not by_src:
        return
    sources = list(by_src.keys())
    preds = sorted({p for blk in by_src.values() for p in blk})
    cell_text = [
        [str(by_src[s].get(p, {}).get("rank_of_comedian", "")) for p in preds] for s in sources
    ]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(preds)), 0.4 + 0.4 * len(sources)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=sources,
        colLabels=preds,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    ax.set_title("rank-of-comedian per source x predictor (smaller = closer)", fontsize=10)
    fig.tight_layout()
    _save_figure(
        fig,
        "comedian_rank_table",
        {
            "description": "Per-source rank-of-comedian by predictor.",
            "metadata": reproducibility_metadata({"script": "phase6_figures.comedian_rank_table"}),
        },
    )


def cosine_layer_sweep(regress: dict) -> None:
    """Exploratory (e) — Concern #5 / plan §6.6: per-source rho-vs-Delta across
    the cosine response-token recipe (b) layer sweep ({7, 14, 21, 27}).

    One line per source; x = layer, y = per-source Spearman rho. The cosine
    headline-layer rho is the dot at layer 21. Lets the analyzer see whether
    "JS beats cosine" reduces to "extraction layer matters".
    """
    ladder = regress.get("cosine_layer_ladder", {})
    pooled = ladder.get("pooled_per_layer", {})
    if not pooled:
        return

    # Reverse-engineer the layer ints from the label strings.
    def _layer_of(label: str) -> int | None:
        for prefix in ("cosine_response_l",):
            if label.startswith(prefix):
                tail = label[len(prefix) :]
                try:
                    return int(tail)
                except ValueError:
                    return None
        return None

    # Per-source rhos pulled from predictors[label]["per_source"][src]["rho"].
    preds = regress.get("predictors", {})
    layer_labels = sorted(pooled.keys(), key=lambda lab: _layer_of(lab) or 0)
    layer_ints = [_layer_of(lab) for lab in layer_labels]
    sources = sorted({s for lab in layer_labels for s in preds.get(lab, {}).get("per_source", {})})
    if not sources:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for src in sources:
        ys = []
        for lab in layer_labels:
            r = preds.get(lab, {}).get("per_source", {}).get(src, {}).get("rho")
            ys.append(np.nan if r is None else r)
        ax.plot(
            layer_ints,
            ys,
            marker="o",
            color=SOURCE_COLORS.get(src, "#444"),
            label=src,
        )
    # Overlay pooled source-FE rho as a black dashed line.
    pooled_ys = [pooled[lab].get("source_fe_rho") for lab in layer_labels]
    pooled_ys = [np.nan if y is None else y for y in pooled_ys]
    ax.plot(
        layer_ints,
        pooled_ys,
        marker="s",
        color="black",
        linestyle="--",
        label="pooled (source-FE)",
        linewidth=2,
    )
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("layer (Qwen residual stream)", fontsize=9)
    ax.set_ylabel("Spearman rho vs Delta", fontsize=9)
    ax.set_title("Cosine recipe (b) layer sweep — per-source rho", fontsize=10)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    _save_figure(
        fig,
        "cosine_layer_sweep",
        {
            "description": (
                "Per-source Spearman rho vs Delta across cosine recipe (b) layers "
                "{7, 14, 21, 27}. Pooled source-FE rho overlaid as black dashed."
            ),
            "metadata": reproducibility_metadata({"script": "phase6_figures.cosine_layer_sweep"}),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-hero",
        action="store_true",
        help="Skip the hero figure (smoke / debug).",
    )
    args = parser.parse_args()

    _try_apply_paper_rcparams()

    if not PHASE4_PATH.exists():
        raise RuntimeError(f"Phase 4 output missing at {PHASE4_PATH}")
    phase4 = read_json(PHASE4_PATH)
    cells = phase4["cells"]
    regress = read_json(PHASE5_PATH) if PHASE5_PATH.exists() else None

    # Phase 5 may have HALTED with the kill criterion; figures still render
    # what data exists, but `predictors` will be missing then.
    if regress and regress.get("js_predictor_dynamic_range_insufficient"):
        logger.warning(
            "Phase 5 halted (JS dynamic range insufficient); skipping figures that "
            "depend on the regression block."
        )
        regress = None

    if not args.skip_hero:
        hero_paired_scatter(cells)
    pooled_scatter_two_panel(cells)
    js_vs_cosine_scatter(cells)
    if regress:
        per_source_rho_bars(regress)
        comedian_rank_table(regress)
        cosine_layer_sweep(regress)

    logger.info("Phase 6 complete. Figures in %s", PHASE6_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
