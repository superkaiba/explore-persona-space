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
    items.append(
        (
            "#66 pooled rho (verify)",
            abs(payload["fig_66"]["pooled"]["raw"]),
            abs(payload["fig_66"]["pooled"]["centered"]),
            rows["66-verify"]["regrade_label"],
            False,
        )
    )
    items.append(
        (
            "#405 pooled rho",
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
            "#478 mean per-K rho",
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
            "#396/#415 headline rho",
            abs(c[head]["rho_partial_raw"]),
            abs(c[head]["rho_partial_centered"]),
            rows["415-predictor-null"]["regrade_label"],
            False,
        )
    )
    cell474 = payload["fig_474"]["cells"][0]
    items.append(
        (
            "#474 cos-vs-dG rho (approx)",
            abs(cell474["raw"]["rho_cos_deltag"]),
            abs(cell474["centered_approx"]["rho_cos_deltag"]),
            rows["474-gram-sensitivity"]["regrade_label"],
            True,
        )
    )
    l20 = payload["fig_341"]["per_layer"]["20"]
    items.append(
        (
            "#341 cos-JS rho L20 (approx)",
            abs(l20["recomputed_rho_raw"]),
            abs(l20["rho_centered_approx"]),
            rows["341-cos-js-alignment"]["regrade_label"],
            True,
        )
    )

    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, (_name, r, m, label, approx) in enumerate(items):
        color = _color_for(label)
        ax.plot([r, m], [i, i], "-", color=color, lw=1.6, zorder=1)
        face_raw = "white" if approx else paper_palette_role("baseline")
        face_mc = "white" if approx else color
        ax.scatter(
            [r], [i], s=46, facecolor=face_raw, edgecolor=paper_palette_role("baseline"), zorder=2
        )
        ax.scatter([m], [i], s=46, facecolor=face_mc, edgecolor=color, zorder=2)
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
    ax.set_title("Raw (open/grey) vs mean-centered (filled) — re-grade per task")
    fig.tight_layout()
    savefig_paper(fig, "issue_536/hero_regrade_dumbbell", dir=FIGDIR)
    plt.close(fig)


def fig_histograms(payload: dict) -> None:
    """Per-bank off-diagonal distributions, raw vs centered."""
    banks = dict(payload["bank_offdiag"])
    banks["issue406_gram_L21 (approx)"] = {
        "raw": payload["fig_474"]["offdiag_L21"]["raw"],
        "centered": payload["fig_474"]["offdiag_L21"]["centered"],
    }
    set_paper_style()
    n = len(banks)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.9), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (name, blob) in zip(axes, sorted(banks.items()), strict=True):
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
    ax.set_xlabel("raw cosine similarity (off-diag, L21)")
    ax.set_ylabel("centered-approx cosine similarity")
    ax.set_title("#406 lineage: centering decompresses the similarity scale")
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
    ax.set_title("#478 held-out band re-assignment under centering")
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig_paper(fig, "issue_536/band_reassignment_478", dir=FIGDIR)
    plt.close(fig)


def fig_forest(payload: dict) -> None:
    """#396/#415 12-cell forest: raw vs centered length-partial rho."""
    cells = payload["fig_396_415"]["cells"]
    holm = payload["fig_396_415"]["holm"]
    names = sorted(cells)
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 1.4))
    for i, k in enumerate(names):
        c = cells[k]
        ax.scatter(
            c["rho_partial_raw"],
            i - 0.12,
            s=30,
            facecolor="white",
            edgecolor=paper_palette_role("baseline"),
        )
        ax.scatter(
            c["rho_partial_centered"],
            i + 0.12,
            s=30,
            color=paper_palette_role("accent" if holm[k] else "primary"),
        )
    ax.axvline(0, color=paper_palette_role("neutral"), lw=0.8)
    ax.axvspan(-0.5, 0.5, color=paper_palette_role("neutral"), alpha=0.08)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [n.replace("_diagonal_mean", "").replace("cos_to_", "") for n in names], fontsize=7
    )
    ax.set_xlabel("length-partial Spearman rho (open = raw, filled = centered)")
    ax.set_title("#396/#415 12-cell family under the canonical metric (band = |rho| < 0.5)")
    fig.tight_layout()
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
