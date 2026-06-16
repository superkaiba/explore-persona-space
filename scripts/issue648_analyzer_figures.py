#!/usr/bin/env python3
"""Task #648 analyzer figures — blog-style hero forest + supplementary diagnostics
from the committed per_bank_skill_table.json. CPU-only, reads the artifact, never
recomputes. Saves under figures/issue_648/."""

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

REPO = Path(__file__).resolve().parent.parent
TABLE = REPO / "eval_results" / "issue_648" / "per_bank_skill_table.json"
PAYLOAD = REPO / "eval_results" / "issue_536" / "figures_payload.json"

# Plain-English short labels (reader-facing) keyed by bank_id in the table.
SHORT = {
    "100-persona L20 (#66)": "100-persona marker bank",
    "core-11 subset L20 (#142)": "core-11 marker subset",
    "20-bank L20 (#405)": "20-persona extraction bank",
    "111-bank L20 (#478)": "111-persona marker bank",
    "111-bank L20 (#490)": "111-persona dose-matched bank",
    "n24 L15 (#380)": "24-persona marker bank",
    "n24-predictor L15 (#396/#415, headline surface)": "24-persona logit bank",
    "19-bank L20 (#311)": "19-persona joint-leakage bank",
    "505 PV L21 (#505)": "505 persona-vector bank",
}
# Family -> bank_id for the compression payload (3 families available).
FAMILY_TO_BANK = {
    "single_token_100p_L20": "111-bank L20 (#478)",
    "extraction_method_a_L20": "20-bank L20 (#405)",
    "issue505_pv_L21": "505 PV L21 (#505)",
}


def load_rows() -> list[dict]:
    return json.loads(TABLE.read_text())["rows"]


def hero_forest(rows: list[dict]) -> None:
    """Blog-style forest of ΔR² (centered - raw), determinate=filled, else open.
    Sign convention annotated as a direction band, no per-point text."""
    set_paper_style("blog")
    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")
    # plot top-to-bottom in table order; reverse for a natural read
    order = list(reversed(rows))
    labels = [f"{SHORT[r['bank_id']]}  (n={r['n_groups']})" for r in order]
    pts = [r["delta_cv_r2"] for r in order]
    cis = [r["boot_delta_r2_ci95"] for r in order]
    determinate = [
        r["contributes_to_h_verdict"]
        and not (r["boot_delta_r2_ci95"][0] <= 0 <= r["boot_delta_r2_ci95"][1])
        for r in order
    ]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=False)
    ax.axvline(0.0, color=neutral, lw=1.0, ls="--", zorder=1)
    for i, (p, ci, det) in enumerate(zip(pts, cis, determinate, strict=True)):
        lo, hi = ci
        ax.plot([lo, hi], [i, i], color=primary, lw=1.6, zorder=2)
        if det:
            ax.scatter(
                [p], [i], s=70, facecolors=primary, edgecolors=primary, linewidths=1.2, zorder=3
            )
        else:
            ax.scatter(
                [p], [i], s=70, facecolors="none", edgecolors=primary, linewidths=1.4, zorder=3
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(-0.75, 1.75)
    ax.set_xlabel(
        r"$\Delta R^2 = R^2_{\mathrm{centered}} - R^2_{\mathrm{raw}}$"
        "\n← raw predicts better          centered predicts better →"
    )
    ax.set_title(
        "Centered cosine is not the stronger leakage predictor",
        loc="left",
        fontsize=12.5,
        fontweight="semibold",
        pad=32,
    )
    ax.annotate(
        "Filled = determinate (CI excludes 0); open = indistinguishable, low-N, or both-fail.\n"
        "Only the 505 bank resolves — and it favors raw. Paired-bootstrap 95% CI.",
        xy=(0.0, 1.012),
        xycoords="axes fraction",
        fontsize=9,
        color=neutral,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.29, bottom=0.19, right=0.97, top=0.85)
    savefig_paper(fig, "issue_648/hero_forest_delta_cvR2", dir="figures/")
    plt.close(fig)


def insample_vs_cv(rows: list[dict]) -> None:
    """Per bank: in-sample Δρ (centered - raw) vs out-of-sample ΔR² (centered - raw).
    The H_raw-inflates diagnostic — if raw inflated in-sample, eligible banks would
    sit lower-right (raw fits sample but not held-out)."""
    set_paper_style("blog")
    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=False)
    ax.axhline(0.0, color=neutral, lw=0.9, ls="--", zorder=1)
    ax.axvline(0.0, color=neutral, lw=0.9, ls="--", zorder=1)
    for r in rows:
        x = r["delta_rho"]  # in-sample Δρ
        yv = r["delta_cv_r2"]  # out-of-sample ΔR²
        eligible = r["contributes_to_h_verdict"]
        c = primary if eligible else baseline
        ax.scatter(
            [x],
            [yv],
            s=70,
            facecolors=(c if eligible else "none"),
            edgecolors=c,
            linewidths=1.4,
            zorder=3,
        )
        ax.annotate(
            SHORT[r["bank_id"]].replace(" bank", "").replace(" subset", ""),
            (x, yv),
            fontsize=7.5,
            color=neutral,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel(r"in-sample $\Delta\rho$ (centered - raw)")
    ax.set_ylabel(r"held-out $\Delta R^2$ (centered - raw)")
    ax.set_xlim(-1.0, 0.4)
    ax.set_ylim(-0.4, 0.85)
    # legend by proxy
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=primary,
            markeredgecolor=primary,
            markersize=8,
            label="verdict-eligible (n>5, both fit)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=baseline,
            markersize=8,
            markeredgewidth=1.4,
            label="excluded (low-N or both-fail)",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5)
    ax.set_title(
        "In-sample fit and held-out skill disagree on sign",
        loc="left",
        fontsize=12.0,
        fontweight="semibold",
        pad=30,
    )
    ax.annotate(
        "On the 505 bank centering helps in-sample (Δρ>0) yet hurts held-out (ΔR²<0)\n"
        "— the reverse of raw-inflation.",
        xy=(0.0, 1.012),
        xycoords="axes fraction",
        fontsize=9,
        color=neutral,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.12, bottom=0.11, right=0.97, top=0.84)
    savefig_paper(fig, "issue_648/insample_vs_cv_scatter", dir="figures/")
    plt.close(fig)


def paired_r2_bars(rows: list[dict]) -> None:
    """Absolute R²_raw and R²_centered per bank — makes the three both-negative
    (both predictors fail out-of-sample) banks visible as a finding."""
    set_paper_style("blog")
    raw_c = paper_palette_role("baseline")
    cen_c = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")
    order = rows
    labels = [SHORT[r["bank_id"]] for r in order]
    r2raw = [r["cv_r2_raw"] for r in order]
    r2cen = [r["cv_r2_centered"] for r in order]
    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=False)
    ax.axhline(0.0, color=neutral, lw=1.0, zorder=1)
    ax.bar(x - w / 2, r2raw, w, label="raw cosine", color=raw_c, zorder=2)
    ax.bar(x + w / 2, r2cen, w, label="centered cosine", color=cen_c, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("held-out CV $R^2$ (per-fold train-mean baseline)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Three small banks: both recipes fail out-of-sample",
        loc="left",
        fontsize=12.0,
        fontweight="semibold",
        pad=30,
    )
    ax.annotate(
        "Bars below 0 = worse than predicting the train-mean. On 24/24/17-persona banks\n"
        "neither raw nor centered generalizes.",
        xy=(0.0, 1.012),
        xycoords="axes fraction",
        fontsize=9,
        color=neutral,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.09, bottom=0.32, right=0.97, top=0.83)
    savefig_paper(fig, "issue_648/paired_r2_raw_vs_centered", dir="figures/")
    plt.close(fig)


def compression(rows: list[dict]) -> None:
    """Off-diagonal raw-cosine compression (#536 bank_offdiag, 3 families) — the
    anisotropy-ridge read centering is meant to remove. Partial coverage (3 of 9)."""
    payload = json.loads(PAYLOAD.read_text())
    bo = payload["bank_offdiag"]
    set_paper_style("blog")
    raw_c = paper_palette_role("baseline")
    cen_c = paper_palette_role("primary")
    fams = list(bo.keys())
    fig, axes = plt.subplots(
        1, len(fams), figsize=(9.6, 3.6), sharey=False, constrained_layout=False
    )
    for ax, fam in zip(axes, fams, strict=True):
        raw = np.array(bo[fam]["raw"])
        cen = np.array(bo[fam]["centered"])
        bins = np.linspace(-0.8, 1.0, 40)
        ax.hist(raw, bins=bins, color=raw_c, alpha=0.7, label="raw")
        ax.hist(cen, bins=bins, color=cen_c, alpha=0.6, label="centered")
        bank = FAMILY_TO_BANK.get(fam, fam)
        ax.set_title(SHORT.get(bank, bank), fontsize=9)
        ax.set_xlabel("off-diagonal cosine")
    axes[0].set_ylabel("count")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Raw cosine is compressed into the anisotropy ridge (3 of 9 banks have this read)",
        fontsize=10.5,
        x=0.5,
        ha="center",
    )
    fig.subplots_adjust(left=0.07, bottom=0.16, right=0.98, top=0.82, wspace=0.22)
    savefig_paper(fig, "issue_648/compression_offdiag", dir="figures/")
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    hero_forest(rows)
    insample_vs_cv(rows)
    paired_r2_bars(rows)
    compression(rows)
    print("[done] 4 analyzer figures written to figures/issue_648/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
