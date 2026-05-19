"""Issue #358 — Step 5 (figures).

Reads the three analysis JSONs and produces the three primary figures
(+ appendix panels) into `figures/issue_358/`:

  Primary (in the clean-result body):
    1. pca_layer18_poisoned          — PC1 vs PC2 scatter, layer 18, poisoned model.
    2. umap_layer18_poisoned         — UMAP(2) scatter, layer 18, poisoned model.
    3. probe_auroc_by_layer          — pooled-LOPO AUROC vs layer (poisoned + base
                                       lines, shaded null-floor band, headline-layer
                                       vertical, Δ-AUROC@L18 annotation).

  Appendix:
    pca_layer18_base
    umap_layer18_base
    umap_layer18_poisoned_n_neighbors_5
    probe_auroc_at_trigger_position
    probe_auroc_within_anth_family
    probe_auroc_length_residualized

All figures use the `paper-plots` blog style (Anthropic-blog register).
PERSONA-LONG markers drawn at 60% alpha to flag scatter-only status.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("issue_358_plot")

INPUT_DIR = Path("eval_results/issue_358")
FIG_DIR = Path("figures/issue_358")
HEADLINE_LAYER = 18
SWEEP_LAYERS: list[int] = [2, 6, 10, 14, 18, 22, 26, 30, 34]

# Plot class → colour. Five labels — within the blog-palette's 8-colour budget.
_LABEL_ORDER = ["TRIGGER", "PARAPHRASE-CONTROL", "PERSONA-SHORT", "PERSONA-LONG"]


def _color_map() -> dict[str, str]:
    """4 distinct colours from the blog palette in fixed order."""
    palette = paper_palette_blog(4)
    return {label: palette[i] for i, label in enumerate(_LABEL_ORDER)}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _scatter_panel(
    ax,
    coords: np.ndarray,
    labels: list[str],
    *,
    xlabel: str,
    ylabel: str,
    cids: list[str] | None = None,
) -> None:
    colors = _color_map()
    for label in _LABEL_ORDER:
        mask = np.asarray([lbl == label for lbl in labels], dtype=bool)
        if not mask.any():
            continue
        # PERSONA-LONG drawn at 60% alpha to flag scatter-only status.
        alpha = 0.6 if label == "PERSONA-LONG" else 0.85
        marker = "X" if label == "TRIGGER" else "o"
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=42 if label == "TRIGGER" else 30,
            c=colors[label],
            edgecolor="white",
            linewidth=0.6,
            alpha=alpha,
            label=label,
            marker=marker,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False, fontsize=8)


def _plot_pca_panel(
    panel: dict[str, Any],
    stem: str,
    *,
    title: str,
    subtitle: str | None = None,
) -> None:
    coords = np.asarray(panel["coords"])  # (N, 10)
    labels = panel["labels"]
    var_pct = [100 * v for v in panel["variance_explained"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    _scatter_panel(
        ax,
        coords[:, :2],
        labels,
        xlabel=f"PC1 ({var_pct[0]:.1f}% var)",
        ylabel=f"PC2 ({var_pct[1]:.1f}% var)",
        cids=panel.get("cids"),
    )
    set_title_subtitle(ax, title, subtitle=subtitle)
    fig.tight_layout()
    written = savefig_paper(fig, stem, dir=str(FIG_DIR))
    log.info("wrote %s", written.get("png"))
    plt.close(fig)


def _plot_umap_panel(
    panel: dict[str, Any],
    labels: list[str],
    stem: str,
    *,
    title: str,
    subtitle: str | None = None,
) -> None:
    coords = np.asarray(panel["coords"])  # (N, 2)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    _scatter_panel(
        ax,
        coords,
        labels,
        xlabel=f"UMAP-1 (n_neighbors={panel['n_neighbors']}, metric=cosine)",
        ylabel="UMAP-2",
    )
    set_title_subtitle(ax, title, subtitle=subtitle)
    fig.tight_layout()
    written = savefig_paper(fig, stem, dir=str(FIG_DIR))
    log.info("wrote %s", written.get("png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PCA panels
# ─────────────────────────────────────────────────────────────────────────────


def plot_pca(pca_json: dict[str, Any]) -> None:
    pois = pca_json["poisoned"]
    base = pca_json["base"]
    _plot_pca_panel(
        pois,
        "pca_layer18_poisoned",
        title="Layer-18 representation: trigger separates from paraphrases (poisoned)",
        subtitle=(
            f"PCA(2) on binary-pool activations of "
            f"`sleepymalc/qwen3-4b-curl-script` @ layer {pois['layer']}. "
            f"PC1+PC2 explain "
            f"{100 * pois['variance_explained_cum'][1]:.1f}% of variance."
        ),
    )
    _plot_pca_panel(
        base,
        "pca_layer18_base",
        title="Layer-18 representation in the base model (appendix)",
        subtitle=(
            f"PCA(2) on binary-pool activations of "
            f"`Qwen/Qwen3-4B-Base` @ layer {base['layer']}. "
            f"PC1+PC2 explain "
            f"{100 * base['variance_explained_cum'][1]:.1f}% of variance."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. UMAP panels
# ─────────────────────────────────────────────────────────────────────────────


def plot_umap(umap_json: dict[str, Any]) -> None:
    pois = umap_json["poisoned"]
    base = umap_json["base"]
    _plot_umap_panel(
        pois["panels"]["n_neighbors_15"],
        pois["labels"],
        "umap_layer18_poisoned",
        title="UMAP layer 18, poisoned model: trigger cluster is visible",
        subtitle=(
            f"UMAP(2) on binary-pool activations @ layer {pois['layer']}, "
            f"n_neighbors=15, metric=cosine, seed=42."
        ),
    )
    _plot_umap_panel(
        pois["panels"]["n_neighbors_5"],
        pois["labels"],
        "umap_layer18_poisoned_n_neighbors_5",
        title="UMAP layer 18 (n_neighbors=5) — small-n sanity (appendix)",
        subtitle=(
            "Same input as the n_neighbors=15 panel above; smaller "
            "n_neighbors emphasises local structure."
        ),
    )
    _plot_umap_panel(
        base["panels"]["n_neighbors_15"],
        base["labels"],
        "umap_layer18_base",
        title="UMAP layer 18 in the base model (appendix)",
        subtitle=(
            f"UMAP(2) on binary-pool activations of `Qwen/Qwen3-4B-Base` "
            f"@ layer {base['layer']}, n_neighbors=15, metric=cosine."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probe AUROC by layer
# ─────────────────────────────────────────────────────────────────────────────


def _auroc_curve(
    model_block: dict[str, Any],
) -> tuple[list[int], list[float], list[float], list[float]]:
    layers = SWEEP_LAYERS
    aurocs, lo, hi = [], [], []
    for L in layers:
        primary = model_block["per_layer"][str(L)]["primary"]
        aurocs.append(primary["pooled_auroc"])
        lo.append(primary["ci_95"][0])
        hi.append(primary["ci_95"][1])
    return layers, aurocs, lo, hi


def _null_band(
    model_block: dict[str, Any], key: str, *, lo_pctile: float, hi_pctile: float
) -> tuple[list[float], list[float]]:
    """Per-layer (lo, hi) percentiles of a null distribution from probe_aurocs.json."""
    lo, hi = [], []
    for L in SWEEP_LAYERS:
        arr = np.asarray(model_block["per_layer"][str(L)][key])
        lo.append(float(np.percentile(arr, lo_pctile)))
        hi.append(float(np.percentile(arr, hi_pctile)))
    return lo, hi


def plot_probe_auroc(probe_json: dict[str, Any]) -> None:
    pois = probe_json["poisoned"]
    base = probe_json["base"]
    palette = paper_palette_blog(4)
    color_pois, color_base, color_null, _ = palette[0], palette[1], palette[2], palette[3]

    layers, pois_auc, pois_lo, pois_hi = _auroc_curve(pois)
    _, base_auc, base_lo, base_hi = _auroc_curve(base)
    shuff_lo, shuff_hi = _null_band(pois, "null_shuffled", lo_pctile=5, hi_pctile=95)
    rp_lo, rp_hi = _null_band(pois, "null_random_proj", lo_pctile=5, hi_pctile=95)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Null-floor envelopes (shaded). Both shown together with light alpha so
    # the trained-probe CI ribbon reads on top.
    ax.fill_between(
        layers,
        shuff_lo,
        shuff_hi,
        color=color_null,
        alpha=0.15,
        label="Shuffled-label null (5/95%, n=200)",
        linewidth=0,
    )
    ax.fill_between(
        layers,
        rp_lo,
        rp_hi,
        color=color_null,
        alpha=0.08,
        label="Random-projection null (5/95%, n=200)",
        linewidth=0,
    )

    # Poisoned + base CI ribbons.
    ax.fill_between(layers, pois_lo, pois_hi, color=color_pois, alpha=0.20, linewidth=0)
    ax.plot(
        layers,
        pois_auc,
        color=color_pois,
        marker="o",
        linewidth=2.0,
        label="Poisoned (pooled-LOPO AUROC)",
    )
    ax.fill_between(layers, base_lo, base_hi, color=color_base, alpha=0.20, linewidth=0)
    ax.plot(
        layers,
        base_auc,
        color=color_base,
        marker="s",
        linewidth=2.0,
        label="Base (pooled-LOPO AUROC)",
    )

    # Headline-layer vertical line.
    ax.axvline(
        HEADLINE_LAYER,
        color="black",
        linestyle=":",
        linewidth=1.0,
        alpha=0.6,
    )

    # Δ-AUROC@L18 annotation.
    head_delta = probe_json["deltas"][str(HEADLINE_LAYER)]["delta_auroc"]
    ax.annotate(
        f"Δ-AUROC@L{HEADLINE_LAYER} = {head_delta:+.3f}",
        xy=(HEADLINE_LAYER, max(pois_auc[layers.index(HEADLINE_LAYER)], 0.5)),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2.0},
    )

    ax.set_xlabel("Transformer block (0-indexed)")
    ax.set_ylabel("Pooled-LOPO AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(layers)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    set_title_subtitle(
        ax,
        "Probe AUROC by layer — does the trigger leave a linear signature?",
        subtitle=(
            f"L2 logistic regression, C=1.0, class-balanced. Headline @ L{HEADLINE_LAYER}. "
            f"Null floors: n=200 each."
        ),
    )
    fig.tight_layout()
    written = savefig_paper(fig, "probe_auroc_by_layer", dir=str(FIG_DIR))
    log.info("wrote %s", written.get("png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Appendix probe panels
# ─────────────────────────────────────────────────────────────────────────────


def _appendix_auroc_panel(
    probe_json: dict[str, Any],
    field: str,
    stem: str,
    *,
    title: str,
    subtitle: str,
    fallback_skipped_msg: str,
) -> None:
    """Generic 2-line plot for the secondary AUROC panels (within-anth-family,
    length-residualized). Falls back to a "skipped" annotation if the
    relevant subset was underpowered.
    """
    palette = paper_palette_blog(2)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    any_data = False
    for color, (label, model_block) in zip(
        palette,
        [("Poisoned", probe_json["poisoned"]), ("Base", probe_json["base"])],
        strict=True,
    ):
        ys, lo, hi = [], [], []
        for L in SWEEP_LAYERS:
            cell = model_block["per_layer"][str(L)][field]
            if isinstance(cell, dict) and cell.get("skipped"):
                ys.append(float("nan"))
                lo.append(float("nan"))
                hi.append(float("nan"))
                continue
            ys.append(cell["pooled_auroc"])
            lo.append(cell["ci_95"][0])
            hi.append(cell["ci_95"][1])
            any_data = True
        ys_arr = np.asarray(ys)
        lo_arr = np.asarray(lo)
        hi_arr = np.asarray(hi)
        ok = ~np.isnan(ys_arr)
        if ok.any():
            ax.fill_between(
                np.asarray(SWEEP_LAYERS)[ok],
                lo_arr[ok],
                hi_arr[ok],
                color=color,
                alpha=0.20,
                linewidth=0,
            )
            ax.plot(
                np.asarray(SWEEP_LAYERS)[ok],
                ys_arr[ok],
                color=color,
                marker="o",
                linewidth=2.0,
                label=label,
            )
    if not any_data:
        ax.text(
            0.5,
            0.5,
            fallback_skipped_msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axvline(HEADLINE_LAYER, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Transformer block (0-indexed)")
        ax.set_ylabel("Pooled-LOPO AUROC")
        ax.set_ylim(0.4, 1.02)
        ax.set_xticks(SWEEP_LAYERS)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    set_title_subtitle(ax, title, subtitle=subtitle)
    fig.tight_layout()
    written = savefig_paper(fig, stem, dir=str(FIG_DIR))
    log.info("wrote %s", written.get("png"))
    plt.close(fig)


def plot_probe_auroc_within_anth_family(probe_json: dict[str, Any]) -> None:
    _appendix_auroc_panel(
        probe_json,
        field="within_anth_family",
        stem="probe_auroc_within_anth_family",
        title="Within-anth-family probe — rules out 'anth-letter detection'",
        subtitle=(
            "Binary pool restricted to anth-stem rows only (TRIGGER vs "
            "anth-stem PARAPHRASE-CONTROL). Pooled-LOPO AUROC by layer."
        ),
        fallback_skipped_msg="within-anth-family subset underpowered at every layer",
    )


def plot_probe_auroc_length_residualized(probe_json: dict[str, Any]) -> None:
    _appendix_auroc_panel(
        probe_json,
        field="length_residualized",
        stem="probe_auroc_length_residualized",
        title="Length-residualized AUROC — does the signal survive length control?",
        subtitle=(
            "Activations residualized on `n_tokens` before the probe. "
            "If this line tracks the raw AUROC, the headline is not length-driven."
        ),
        fallback_skipped_msg="length-residualization failed at every layer",
    )


def plot_probe_auroc_at_trigger_position(probe_json: dict[str, Any]) -> None:
    """Position-sweep panel: probe at the first `anth`-token position
    (Anthropic-style read-out, plan §4.3 / Methodology-Claude item 9).
    Standing recommendation #1: skipped layers carry no data point.
    """
    palette = paper_palette_blog(2)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    any_data = False
    for color, (label, model_block) in zip(
        palette,
        [("Poisoned", probe_json["poisoned"]), ("Base", probe_json["base"])],
        strict=True,
    ):
        ys, lo, hi, layers_ok = [], [], [], []
        for L in SWEEP_LAYERS:
            cell = model_block["position_sweep"][str(L)]
            if cell.get("skipped"):
                continue
            primary = cell["primary"]
            layers_ok.append(L)
            ys.append(primary["pooled_auroc"])
            lo.append(primary["ci_95"][0])
            hi.append(primary["ci_95"][1])
            any_data = True
        if layers_ok:
            ax.fill_between(layers_ok, lo, hi, color=color, alpha=0.20, linewidth=0)
            ax.plot(layers_ok, ys, color=color, marker="o", linewidth=2.0, label=label)
    head_skipped_reason = probe_json["poisoned"]["position_sweep"][str(HEADLINE_LAYER)].get(
        "reason", ""
    )
    if not any_data:
        msg = f"Position-sweep skipped at every layer (underpowered subset).\n{head_skipped_reason}"
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axvline(HEADLINE_LAYER, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Transformer block (0-indexed)")
        ax.set_ylabel("Pooled-LOPO AUROC at trigger-token position")
        ax.set_ylim(0.4, 1.02)
        ax.set_xticks(SWEEP_LAYERS)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    head_comp = probe_json["poisoned"]["position_sweep"][str(HEADLINE_LAYER)].get("composition", {})
    set_title_subtitle(
        ax,
        "Probe at the first `anth`-token position (Anthropic-style read-out)",
        subtitle=(
            f"Subset = anth-bearing binary-pool rows. Headline subset @ L{HEADLINE_LAYER}: "
            f"n={head_comp.get('n_total', 'n/a')} "
            f"({head_comp.get('n_pos', 'n/a')}+/{head_comp.get('n_neg', 'n/a')}-)."
        ),
    )
    fig.tight_layout()
    written = savefig_paper(fig, "probe_auroc_at_trigger_position", dir=str(FIG_DIR))
    log.info("wrote %s", written.get("png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    set_paper_style("blog")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with (INPUT_DIR / "pca_coords.json").open() as f:
        pca_json = json.load(f)
    with (INPUT_DIR / "umap_coords.json").open() as f:
        umap_json = json.load(f)
    with (INPUT_DIR / "probe_aurocs.json").open() as f:
        probe_json = json.load(f)

    plot_pca(pca_json)
    plot_umap(umap_json)
    plot_probe_auroc(probe_json)
    plot_probe_auroc_within_anth_family(probe_json)
    plot_probe_auroc_length_residualized(probe_json)
    plot_probe_auroc_at_trigger_position(probe_json)

    log.info("all figures written to %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
