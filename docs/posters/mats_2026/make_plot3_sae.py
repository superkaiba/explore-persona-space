"""Poster plot 3 — best predictors of per-feature R², partialling out firing activity.

ONE compact single-panel forest figure for the MATS 2026 poster (section 3),
replacing the two-panel raw-ρ + label-AUROC figure. Rows: the SAE-feature
properties with the largest activity-partialled effect, ranked by
|partial Spearman ρ| (partialled on `activity` = firing frequency per answer),
against the full-width dense-context → SAE-feature ridge target
(per-feature held-out R², ridge__mean, full 131,072-feature dictionary,
n_features_used = 113,262).

Source of every number:
eval_results/issue_1482/predictor_battery/continuous_predictors_densesae_ridge.json
(predictors[*].partial_on_activity + partial_on_activity_ci95 — percentile 95%
CI over 2,000 bootstrap draws). Nothing is hand-typed.

Run:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 uv run python docs/posters/mats_2026/make_plot3_sae.py
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
SOURCE = (
    REPO / "eval_results/issue_1482/predictor_battery/continuous_predictors_densesae_ridge.json"
)
OUT_DIR = Path(__file__).resolve().parent / "figures"

# keep rows with |partial rho| >= this floor (natural gap: 0.130 -> 0.093)
MIN_ABS_PARTIAL = 0.10

# artifact key → short poster tick label (caption carries the expansions)
SHORT_LABELS = {
    "mean_act_uncond": "mean activation",
    "firing_freq_per_token": "firing freq. / token",
    "mean_act_cond": "mean act. (active)",
    "proj_var": "var. on decoder dir.",
    "consistency": "consistency",
    "n_active_holdout": "# active answers",
    "side_ratio": "answer-side share",
    "footprint_kurt": "footprint kurt.",
    "enc_dec_cos": "enc-dec cosine",
    "scaffold_frac": "scaffold mass",
}


def load_rows() -> list[dict]:
    with open(SOURCE) as f:
        d = json.load(f)
    rows = []
    for p in d["predictors"]:
        if p["partial_on_activity"] is None:  # the conditioning variable itself
            continue
        if abs(p["partial_on_activity"]) < MIN_ABS_PARTIAL:
            continue
        rows.append(
            {
                "key": p["key"],
                "label_full": p["label"],
                "label_short": SHORT_LABELS[p["key"]],
                "spearman_raw": p["spearman_raw"],
                "partial_on_activity": p["partial_on_activity"],
                "partial_on_activity_ci95": p["partial_on_activity_ci95"],
            }
        )
    rows.sort(key=lambda r: abs(r["partial_on_activity"]), reverse=True)
    return rows


def main() -> None:
    set_paper_style("iclr")
    rows = load_rows()
    c_main = paper_color("instruct")
    c_null = paper_color("null")

    fig, ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    # largest |partial| at the top
    y = np.arange(len(rows), dtype=float)[::-1]

    ax.axvline(0.0, color=c_null, lw=0.8, ls="--", zorder=1)
    vals = np.array([r["partial_on_activity"] for r in rows])
    los = np.array([r["partial_on_activity_ci95"][0] for r in rows])
    his = np.array([r["partial_on_activity_ci95"][1] for r in rows])
    ax.barh(y, vals, height=0.62, color=c_main, edgecolor="black", linewidth=0.4, zorder=2)
    ax.errorbar(
        vals,
        y,
        xerr=[vals - los, his - vals],
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([r["label_short"] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"partial Spearman $\rho$ vs per-feature $R^2$ (given firing activity)")

    paths = savefig_paper(fig, "plot3_sae_predictors", dir=OUT_DIR)
    for fmt, p in paths.items():
        print(f"{fmt}: {p}")

    data = {
        "source": str(SOURCE.relative_to(REPO)),
        "target": (
            "per-feature held-out R^2, full-width dense-context -> SAE-feature ridge map "
            "(ridge__mean, 131,072-feature dictionary, n_features_used=113,262)"
        ),
        "partial_on": "activity = firing frequency (per answer)",
        "ci": "percentile 95% over 2,000 bootstrap draws (ranks fixed at the full sample)",
        "selection": f"|partial rho| >= {MIN_ABS_PARTIAL}, ranked by |partial rho|",
        "rows": rows,
    }
    out_json = OUT_DIR / "plot3_sae_predictors_data.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=1)
    print(f"data: {out_json}")


if __name__ == "__main__":
    main()
