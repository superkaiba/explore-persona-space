"""Poster plot 3 — which SAE-feature properties predict per-feature held-out R²?

ONE two-panel forest figure ("predictors ladder") for the MATS 2026 poster:

  Panel A (left)  — continuous feature properties, Spearman ρ against per-feature
                    held-out R² (raw + partial given firing activity), on the
                    16,384-feature layer-19 answer-side panel (dense context →
                    SAE feature ridge map, mean pooling — #1482).
  Panel B (right) — judged label classes, AUROC (labeled vs rest, ranked by
                    per-feature R²) against the activity-decile-stratified
                    label-shuffle null, on the full dictionary (dense context →
                    SAE ridge__mean full-width target — #1482).

Every number is read from committed eval_results JSONs; nothing is hand-typed.
The two panels use each family's landed grain (panel vs full width); both are
dense-context→SAE-feature ridge targets. Grain is named in the panel titles.

Run:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 uv run python docs/posters/mats_2026/make_plot3_predictors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
BATTERY = REPO / "eval_results/issue_1482/predictor_battery"
CORRELATES = REPO / "eval_results/issue_1482/feature_correlates"
EXTREMES = REPO / "eval_results/issue_1482/feature_extremes"
OUT_DIR = Path(__file__).resolve().parent / "figures"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def continuous_rows() -> list[dict]:
    """Panel A rows: raw ρ (+95% bootstrap CI) and ρ | firing activity.

    Raw ρ + CI come from the panel-grain joint model
    (predictor_battery/joint_model.json, target = per-feature held-out R²,
    ridge, mean pooling, 16,384 layer-19 features); the given-activity partial
    for each predictor comes from that predictor's own landed read.
    """
    joint = _load(BATTERY / "joint_model.json")
    by_key = {p["key"]: p for p in joint["predictors"]}

    consistency = _load(CORRELATES / "consistency.json")
    proj_var = _load(CORRELATES / "dense_projection_variance.json")
    rb_align = _load(EXTREMES / "extremes.json")["alignment_mechanical"]

    rows = [
        {
            "label": "within-answer consistency",
            "raw": by_key["consistency"]["spearman_raw"],
            "raw_ci": by_key["consistency"]["spearman_raw_ci95"],
            "partial": consistency["partial_spearman_consistency_r2_given_activity"],
        },
        {
            "label": "dense variance along\ndecoder direction",
            "raw": by_key["proj_var"]["spearman_raw"],
            "raw_ci": by_key["proj_var"]["spearman_raw_ci95"],
            "partial": proj_var["spearman"]["r2_vs_log_projvar_partial_activity"],
        },
        {
            "label": "firing activity",
            "raw": by_key["activity"]["spearman_raw"],
            "raw_ci": by_key["activity"]["spearman_raw_ci95"],
            "partial": None,  # the conditioning variable itself
        },
        {
            "label": "decoder alignment with\npersona directions $r_B$",
            "raw": by_key["rb_align_max"]["spearman_raw"],
            "raw_ci": by_key["rb_align_max"]["spearman_raw_ci95"],
            "partial": rb_align["partial_spearman_max_abs_cos_r2_given_activity"],
        },
    ]
    rows.sort(key=lambda r: r["raw"])  # smallest at bottom → largest on top
    return rows


LABEL_ROWS = {
    "speaker_identity": "speaker identity\n(persona)",
    "content_task_format": "task format",
    "abstraction_high": "high abstraction",
    "content_entity": "named entity",
    "speaker_language": "language",
    "content_topic": "topic",
}


def label_rows() -> list[dict]:
    """Panel B rows: AUROC (+95% CI) vs the stratified label-shuffle null band."""
    reads = _load(BATTERY / "fullwidth_label_reads_densesae_ridge.json")["label_reads"]
    rows = []
    for key, label in LABEL_ROWS.items():
        r = reads[key]
        rows.append(
            {
                "label": label,
                "auroc": r["auroc"],
                "ci": r["auroc_ci95"],
                "band": r["auroc_perm_band"],
                "null_mean": r["auroc_perm_null_mean"],
                "excess": r["excess_over_null"],
            }
        )
    rows.sort(key=lambda r: r["excess"])
    return rows


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    c_main = paper_color("instruct")  # blue — the featured map arm
    c_null = paper_color("null")  # gray — null bands / reference

    rows_a = continuous_rows()
    rows_b = label_rows()

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.4, 2.5), constrained_layout=True, width_ratios=[1.0, 1.0]
    )

    # ---- Panel A: continuous predictors (Spearman rho) ----
    y_a = np.arange(len(rows_a), dtype=float)
    ax_a.axvline(0.0, color=c_null, lw=0.8, ls="--", zorder=1)
    for y, row in zip(y_a, rows_a):
        lo, hi = row["raw_ci"]
        ax_a.errorbar(
            row["raw"],
            y + 0.13,
            xerr=[[row["raw"] - lo], [hi - row["raw"]]],
            fmt="o",
            ms=4.5,
            mfc="white",
            mec=c_main,
            ecolor=c_main,
            elinewidth=1.0,
            capsize=0,
            zorder=3,
        )
        if row["partial"] is not None:
            ax_a.plot(
                row["partial"],
                y - 0.13,
                marker="o",
                ms=4.5,
                color=c_main,
                zorder=3,
            )
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels([r["label"] for r in rows_a])
    ax_a.set_ylim(-0.55, len(rows_a) - 0.45)
    ax_a.set_xlim(-0.06, 0.68)
    ax_a.set_xlabel(r"Spearman $\rho$ vs per-feature held-out $R^2$")
    ax_a.set_title("Continuous properties (16,384-feature panel)", fontsize="medium")
    # legend: raw (open) vs partial given activity (filled)
    from matplotlib.lines import Line2D

    ax_a.legend(
        handles=[
            Line2D(
                [], [], marker="o", ls="none", ms=4.5, mfc="white", mec=c_main, label=r"raw $\rho$"
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=4.5,
                color=c_main,
                label=r"$\rho\,|\,$firing activity",
            ),
        ],
        loc="lower right",
        frameon=False,
        handletextpad=0.3,
        borderaxespad=0.2,
    )

    # ---- Panel B: label classes (AUROC vs stratified null) ----
    y_b = np.arange(len(rows_b), dtype=float)
    # stop the 0.5 reference line below the legend headroom
    y_lo, y_hi = -0.55, len(rows_b) - 0.45 + 1.25
    ax_b.axvline(
        0.5,
        color=c_null,
        lw=0.8,
        ls=":",
        zorder=1,
        ymax=(len(rows_b) - 0.55 - y_lo) / (y_hi - y_lo),
    )
    for y, row in zip(y_b, rows_b):
        band_lo, band_hi = row["band"]
        ax_b.plot(
            [band_lo, band_hi],
            [y, y],
            lw=5.5,
            color=c_null,
            alpha=0.45,
            solid_capstyle="round",
            zorder=2,
        )
        lo, hi = row["ci"]
        ax_b.errorbar(
            row["auroc"],
            y,
            xerr=[[row["auroc"] - lo], [hi - row["auroc"]]],
            fmt="o",
            ms=4.5,
            color=c_main,
            ecolor=c_main,
            elinewidth=1.0,
            capsize=0,
            zorder=3,
        )
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels([r["label"] for r in rows_b])
    # extra headroom above the top row so the legend sits clear of the data
    ax_b.set_ylim(-0.55, len(rows_b) - 0.45 + 1.25)
    ax_b.set_xlim(0.37, 0.73)
    ax_b.set_xlabel(r"AUROC (labeled vs rest, by $R^2$)")
    ax_b.set_title("Judged label classes (full dictionary)", fontsize="medium")
    ax_b.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", ms=4.5, color=c_main, label="observed"),
            Line2D([], [], lw=5.5, color=c_null, alpha=0.45, label="stratified null (95%)"),
        ],
        loc="upper left",
        frameon=False,
        handletextpad=0.3,
        borderaxespad=0.2,
    )

    paths = savefig_paper(fig, "plot3_sae_predictors", dir=OUT_DIR)
    for fmt, p in paths.items():
        print(f"{fmt}: {p}")


if __name__ == "__main__":
    main()
