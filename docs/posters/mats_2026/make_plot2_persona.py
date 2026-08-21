"""Poster plot 2 — persona directions are predicted far better than random directions.

ONE compact single-panel bar figure for the MATS 2026 poster (section 2),
replacing the full per-direction rank-spectrum figure
(figures/paper/c3_persona_direction_spectrum.pdf). Per behavior: held-out
per-direction R² of the persona direction r_B vs the mean ± SD over 50 random
unit directions.

Source of every number: eval_results/issue_779/identity_baseline.json →
per_direction (fold 0 of the 5-fold split; n_train=4,000 / n_test=1,000
contexts; read-out layers 14 / 26 / 17 for evil / sycophancy / hallucination).
Nothing is hand-typed.

Run:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 uv run python docs/posters/mats_2026/make_plot2_persona.py
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
SOURCE = REPO / "eval_results/issue_779/identity_baseline.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"

# artifact key → poster display name (poster says "misalignment" for the evil trait)
BEHAVIORS = {
    "evil": "misalignment",
    "sycophancy": "sycophancy",
    "hallucination": "hallucination",
}


def load_rows() -> list[dict]:
    with open(SOURCE) as f:
        pd = json.load(f)["per_direction"]
    rows = []
    for key, display in BEHAVIORS.items():
        res = pd[key]
        rb = res["r_b"]
        rd = res["random_directions"]
        rows.append(
            {
                "behavior_key": key,
                "behavior": display,
                "rb_heldout_r2": rb["heldout_r2"],
                "rb_equivalent_variance_rank": rb["equivalent_variance_rank"],
                "random_r2_mean": rd["r2_mean"],
                "random_r2_sd": rd["r2_sd"],
                "random_n": rd["n"],
                "read_out_layer": res["read_out_layer"],
                "fold": res["fold"],
                "n_train": res["n_train"],
                "n_test": res["n_test"],
            }
        )
    return rows


def main() -> None:
    set_paper_style("iclr")
    rows = load_rows()

    c_rb = paper_color("persona_vector")
    c_rand = paper_color("null")

    fig, ax = plt.subplots(figsize=(6.8, 2.8), constrained_layout=True)
    x = np.arange(len(rows), dtype=float)
    w = 0.36

    ax.bar(
        x - w / 2,
        [r["rb_heldout_r2"] for r in rows],
        width=w,
        color=c_rb,
        edgecolor="black",
        linewidth=0.5,
        label="persona direction $r_B$",
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        [r["random_r2_mean"] for r in rows],
        width=w,
        color=c_rand,
        edgecolor="black",
        linewidth=0.5,
        yerr=[r["random_r2_sd"] for r in rows],
        error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "black"},
        label="random direction (mean $\\pm$ SD)",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([r["behavior"] for r in rows])
    ax.set_ylabel("held-out per-direction $R^2$")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=False, ncols=1, handletextpad=0.5)

    paths = savefig_paper(fig, "plot2_persona_directions", dir=OUT_DIR)
    for fmt, p in paths.items():
        print(f"{fmt}: {p}")

    data = {
        "source": str(SOURCE.relative_to(REPO)),
        "note": (
            "held-out per-direction R^2 of the full-ridge map on fold 0 of the 5-fold "
            "split; persona direction r_B vs mean +- SD over 50 random unit directions"
        ),
        "rows": rows,
    }
    out_json = OUT_DIR / "plot2_persona_directions_data.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=1)
    print(f"data: {out_json}")


if __name__ == "__main__":
    main()
