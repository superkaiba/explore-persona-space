#!/usr/bin/env python3
"""Task #536 figures — hero dumbbell + exploratory dump (plan §6).

Reads ``eval_results/issue_536/{regrade_table.json, figures_payload.json}``
(produced by issue536_recompute_driver.py) and writes to ``figures/issue_536/``
via the paper-plots conventions (PNG + PDF + meta.json sidecars).

Figures:
  1. hero_regrade_dumbbell — per-task scale-invariant statistic, raw vs
     mean-centered, re-grade color-coded; sensitivity-namespace (matrix-only
     approximate) rows drawn with open markers and an "approx" tag, never
     pooled with exact rows.
  2. bank_offdiag_histograms — per-bank off-diagonal cosine distributions,
     raw vs centered (the ~6x compression made visible).
  3. compression_scatter_406_L21 — raw vs centered-approx off-diagonal
     similarity for the #406 L21 matrix (the lineage's compression).
  4. band_reassignment_478 — design-band vs centered-band cross-tab heatmap.
  5. forest_396_415 — 12 predictor x surface cells, raw vs centered
     length-partial Spearman rho.

Usage::

    uv run python scripts/issue536_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

OUT = REPO / "eval_results" / "issue_536"
FIGDIR = str(REPO / "figures")

LABEL_COLORS = {
    "stands": "neutral",
    "weakens": "accent",
    "flips": "accent",
    "strengthens": "primary",
    "null-overturned": "accent",
    "sensitivity": "control",
}


def _color_for(label: str) -> str:
    for key, role in LABEL_COLORS.items():
        if label.startswith(key) or key in label:
            return paper_palette_role(role)
    return paper_palette_role("neutral")


def fig_hero(payload: dict, rows: dict) -> None:
    """Dumbbell: |scale-invariant statistic| raw vs centered per re-graded task."""
    items = []  # (display, raw, mc, label, approx)
    # Row labels are reader-facing: descriptive names only (the prose's own
    # vocabulary), never issue-number codes or project shorthand (clean-result
    # figure-label rule; round-2 critique F1).
    items.append(
        (
            "100-persona pooled (verify)",
            abs(payload["fig_66"]["pooled"]["raw"]),
            abs(payload["fig_66"]["pooled"]["centered"]),
            rows["66-verify"]["regrade_label"],
            False,
        )
    )
    items.append(
        (
            "extraction-method pooled",
            abs(payload["fig_405"]["pooled_rho"]["raw"]),
            abs(payload["fig_405"]["pooled_rho"]["mc"]),
            rows["405-secondary"]["regrade_label"],
            False,
        )
    )
    perk_r = payload["fig_478"]["per_K_rho_raw"]
    perk_m = payload["fig_478"]["per_K_rho_mc"]
    items.append(
        (
            "source-set-size per-K mean",
            float(np.mean([abs(v) for v in perk_r.values()])),
            float(np.mean([abs(v) for v in perk_m.values()])),
            rows["478-perK-slopes"]["regrade_label"],
            False,
        )
    )
    c = payload["fig_396_415"]["cells"]
    head = "cos_to_assistant|logp_end_of_response_diagonal_mean"
    items.append(
        (
            "24-persona predictor headline",
            abs(c[head]["rho_partial_raw"]),
            abs(c[head]["rho_partial_centered"]),
            rows["415-predictor-null"]["regrade_label"],
            False,
        )
    )
    cell474 = payload["fig_474"]["cells"][0]
    items.append(
        (
            "lineage cell (approx)",
            abs(cell474["raw"]["rho_cos_deltag"]),
            abs(cell474["centered_approx"]["rho_cos_deltag"]),
            rows["474-gram-sensitivity"]["regrade_label"],
            True,
        )
    )
    l20 = payload["fig_341"]["per_layer"]["20"]
    items.append(
        (
            "cos-vs-divergence alignment,\nlayer 20 (approx)",
            abs(l20["recomputed_rho_raw"]),
            abs(l20["rho_centered_approx"]),
            rows["341-cos-js-alignment"]["regrade_label"],
            True,
        )
    )

    set_paper_style()
    grey = paper_palette_role("neutral")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, (_name, r, m, label, approx) in enumerate(items):
        color = _color_for(label)
        ax.plot([r, m], [i, i], "-", color=color, lw=1.6, zorder=1)
        # Raw recipe: ALWAYS an open grey circle (explicit linewidths — the
        # blog style zeroes lines.markeredgewidth, which made open markers
        # invisible in the round-1 render).
        ax.scatter([r], [i], s=52, facecolor="white", edgecolor=grey, linewidths=1.4, zorder=3)
        # Mean-centered: filled circle in the re-grade color for exact rows;
        # open SQUARE (white face, colored edge) for matrix-only approx rows
        # so the sensitivity namespace is visually unmistakable.
        if approx:
            ax.scatter(
                [m],
                [i],
                s=52,
                facecolor="white",
                edgecolor=color,
                linewidths=1.4,
                marker="s",
                zorder=3,
            )
        else:
            ax.scatter([m], [i], s=52, facecolor=color, edgecolor=color, linewidths=0.8, zorder=3)
        ax.annotate(
            label,
            xy=(max(r, m) + 0.02, i),
            va="center",
            fontsize=7,
            color=color,
        )
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([t[0] for t in items], fontsize=8)
    ax.set_xlabel("|scale-invariant statistic| (Spearman rho)")
    ax.set_xlim(0, 1.18)
    ax.set_title("Six dumbbell-representable re-graded statistics, raw vs mean-centered")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=grey,
            markeredgewidth=1.4,
            markersize=7,
            label="raw recipe (open circle)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=grey,
            markeredgecolor=grey,
            markersize=7,
            label="mean-centered (filled, re-grade color)",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=paper_palette_role("control"),
            markeredgewidth=1.4,
            markersize=7,
            label="approximate matrix-only read (open square)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    savefig_paper(fig, "issue_536/hero_regrade_dumbbell", dir=FIGDIR)
    plt.close(fig)


def fig_histograms(payload: dict) -> None:
    """Per-bank off-diagonal distributions, raw vs centered."""
    banks = dict(payload["bank_offdiag"])
    banks["issue406_gram_L21"] = {
        "raw": payload["fig_474"]["offdiag_L21"]["raw"],
        "centered": payload["fig_474"]["offdiag_L21"]["centered"],
    }
    # Reader-facing panel titles (the bare slugs are project-internal and
    # banned from rendered figure text). Order matches the body's alt text.
    panel_titles = [
        ("extraction_method_a_L20", "20-persona extraction bank (layer 20)"),
        ("issue406_gram_L21", "16-condition lineage bank\n(layer 21, approx)"),
        ("issue505_pv_L21", "60-persona persona-vector bank\n(layer 21)"),
        ("single_token_100p_L20", "111-persona bank (layer 20)"),
    ]
    set_paper_style()
    n = len(banks)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.9), sharey=False)
    if n == 1:
        axes = [axes]
    ordered = [(title, banks[slug]) for slug, title in panel_titles]
    for ax, (name, blob) in zip(axes, ordered, strict=True):
        ax.hist(
            blob["raw"],
            bins=40,
            color=paper_palette_role("baseline"),
            alpha=0.65,
            label="raw",
        )
        ax.hist(
            blob["centered"],
            bins=40,
            color=paper_palette_role("accent"),
            alpha=0.65,
            label="mean-centered",
        )
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("off-diag cosine")
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "issue_536/bank_offdiag_histograms", dir=FIGDIR)
    plt.close(fig)


def fig_compression(payload: dict) -> None:
    """#406 L21 raw vs centered-approx off-diagonal scatter (compression view)."""
    raw = np.asarray(payload["fig_474"]["offdiag_L21"]["raw"])
    mc = np.asarray(payload["fig_474"]["offdiag_L21"]["centered"])
    set_paper_style()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(raw, mc, s=12, alpha=0.6, color=paper_palette_role("primary"))
    lim = [min(raw.min(), mc.min()) - 0.05, 1.02]
    ax.plot(lim, lim, "--", color=paper_palette_role("neutral"), lw=1)
    ax.set_xlabel("raw cosine similarity (off-diagonal, layer 21)")
    ax.set_ylabel("centered-approx cosine similarity")
    ax.set_title("Lineage bank: centering decompresses the similarity scale")
    fig.tight_layout()
    savefig_paper(fig, "issue_536/compression_scatter_406_L21", dir=FIGDIR)
    plt.close(fig)


def fig_band_crosstab(payload: dict) -> None:
    """#478 design-band vs centered-band re-assignment cross-tab heatmap."""
    order = payload["fig_478"]["band_order"]
    ct = payload["fig_478"]["crosstab"]  # {centered_band: {design_band: n}}
    M = np.array([[ct.get(cb, {}).get(db, 0) for cb in order] for db in order], dtype=float)
    set_paper_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlabel("band under mean-centered distance")
    ax.set_ylabel("design band (raw distance)")
    for i in range(len(order)):
        for j in range(len(order)):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Held-out band re-assignment under centering\n(source-set-size design)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig_paper(fig, "issue_536/band_reassignment_478", dir=FIGDIR)
    plt.close(fig)


_FOREST_PREDICTOR_LABELS = {
    "cos_to_assistant": "cosine to assistant",
    "cos_to_neutral": "cosine to neutral",
}
_FOREST_SURFACE_LABELS = {
    "logp_at_k0": "first-slot log-prob",
    "logp_auc": "log-prob AUC",
    "logp_end_of_response": "end-of-response log-prob",
    "logp_max": "max log-prob",
    "logp_mean": "mean log-prob",
    "substring_match_rate": "marker emission rate",
}


def _forest_label(cell_key: str) -> str:
    """Translate a `predictor|surface_diagonal_mean` slug into plain English."""
    pred, surface = cell_key.split("|", 1)
    surface = surface.removesuffix("_diagonal_mean")
    return f"{_FOREST_PREDICTOR_LABELS[pred]} x {_FOREST_SURFACE_LABELS[surface]}"


def fig_forest(payload: dict) -> None:
    """#396/#415 12-cell forest: raw vs centered length-partial rho."""
    cells = payload["fig_396_415"]["cells"]
    holm = payload["fig_396_415"]["holm"]
    names = sorted(cells)
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(names) + 1.4))
    for i, k in enumerate(names):
        c = cells[k]
        # Explicit linewidths — the blog style zeroes lines.markeredgewidth,
        # which rendered this open raw series invisible in round 1.
        ax.scatter(
            c["rho_partial_raw"],
            i - 0.14,
            s=32,
            facecolor="white",
            edgecolor=paper_palette_role("baseline"),
            linewidths=1.3,
            zorder=3,
        )
        ax.scatter(
            c["rho_partial_centered"],
            i + 0.14,
            s=32,
            color=paper_palette_role("accent" if holm[k] else "primary"),
            zorder=3,
        )
    ax.axvline(0, color=paper_palette_role("neutral"), lw=0.8)
    ax.axvspan(-0.5, 0.5, color=paper_palette_role("neutral"), alpha=0.08)
    ax.set_xlim(-0.62, 0.62)  # band edges visible, so "inside the band" is readable
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([_forest_label(n) for n in names], fontsize=7)
    ax.set_xlabel("length-partial Spearman rho\n(open orange = raw, filled blue = centered)")
    ax.set_title(
        "12-cell predictor family under the canonical metric\n"
        "(band = absolute rank correlation below 0.5)"
    )
    savefig_paper(fig, "issue_536/forest_396_415", dir=FIGDIR)
    plt.close(fig)


def main() -> int:
    payload = json.loads((OUT / "figures_payload.json").read_text())
    rows = {r["row_id"]: r for r in json.loads((OUT / "regrade_table.json").read_text())["rows"]}
    fig_hero(payload, rows)
    fig_histograms(payload)
    fig_compression(payload)
    fig_band_crosstab(payload)
    fig_forest(payload)
    print(f"[phase=figures] 5 figures -> {REPO / 'figures' / 'issue_536'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
