"""Poster plot 2 — persona directions are predicted far better than random directions.

ONE compact single-panel bar figure for the MATS 2026 poster (section 2),
replacing the full per-direction rank-spectrum figure
(figures/paper/c3_persona_direction_spectrum.pdf).

Shape (poster revision, Thomas 2026-08-21): THREE persona-direction bars — one
per behavior — against a SINGLE pooled random-direction reference bar, rather
than a random bar beside every behavior. The three per-behavior random means
(0.5574 / 0.5692 / 0.5765) sit within 0.02 of each other, so three separate
reference bars carried no information the pooled one does not; one bar reads
faster at poster distance.

The pooled bar is EXACT, not a re-estimate: the artifact stores only summary
stats per behavior (n, mean, sd), and the pooled mean + total-variance SD over
the 3 x 50 = 150 draws are computed in closed form from those triples
(`pooled_random_reference` below). It deliberately pools ACROSS read-out layers
(14 / 26 / 17) — a random unit direction has no privileged layer, and the point
of the bar is that random directions land in the same band wherever you take
them. The caption states the pooling; the data sidecar carries all three
per-behavior rows so nothing is lost.

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

# gap between the three persona bars and the set-apart random reference bar
REFERENCE_GAP = 0.55


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


def pooled_random_reference(rows: list[dict]) -> dict:
    """Exact pooled mean + SD over every random draw, from the per-behavior triples.

    The artifact stores no per-draw values, only (n, mean, sd) per behavior. The
    pooled mean is the n-weighted mean; the pooled SD comes from the total-variance
    identity — within-group sum of squares plus between-group sum of squares — so
    this is the SD the 150 draws actually have, not an average of three SDs (which
    would understate it by dropping the between-behavior spread).

    Returns the pooled mean, SD, total n, and the per-behavior means it pooled.
    """
    n = np.array([r["random_n"] for r in rows], dtype=float)
    m = np.array([r["random_r2_mean"] for r in rows], dtype=float)
    s = np.array([r["random_r2_sd"] for r in rows], dtype=float)
    total_n = float(n.sum())
    pooled_mean = float((n * m).sum() / total_n)
    within_ss = float(((n - 1.0) * s**2).sum())
    between_ss = float((n * (m - pooled_mean) ** 2).sum())
    pooled_sd = float(np.sqrt((within_ss + between_ss) / (total_n - 1.0)))
    return {
        "pooled_mean": pooled_mean,
        "pooled_sd": pooled_sd,
        "total_n": int(total_n),
        "per_behavior_means": {r["behavior"]: r["random_r2_mean"] for r in rows},
        "per_behavior_sds": {r["behavior"]: r["random_r2_sd"] for r in rows},
        "read_out_layers_pooled": [r["read_out_layer"] for r in rows],
        "note": (
            "Exact pooled mean and total-variance SD over all 3 x 50 = 150 random "
            "unit directions, computed in closed form from the per-behavior "
            "(n, mean, sd) triples — the artifact stores no per-draw values. Pools "
            "ACROSS read-out layers 14 / 26 / 17: a random unit direction has no "
            "privileged layer, and the three per-behavior means agree to within "
            "0.02. Per-behavior rows are retained in full under `rows`."
        ),
    }


def main() -> None:
    set_paper_style("iclr")
    rows = load_rows()
    ref = pooled_random_reference(rows)

    c_rb = paper_color("persona_vector")
    c_rand = paper_color("null")

    fig, ax = plt.subplots(figsize=(6.8, 2.8), constrained_layout=True)

    x_rb = np.arange(len(rows), dtype=float)
    x_ref = float(len(rows)) - 1.0 + 1.0 + REFERENCE_GAP
    w = 0.62

    ax.bar(
        x_rb,
        [r["rb_heldout_r2"] for r in rows],
        width=w,
        color=c_rb,
        edgecolor="black",
        linewidth=0.5,
        label="persona direction $r_B$",
        zorder=3,
    )
    ax.bar(
        [x_ref],
        [ref["pooled_mean"]],
        width=w,
        color=c_rand,
        edgecolor="black",
        linewidth=0.5,
        yerr=[ref["pooled_sd"]],
        error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "black"},
        label=f"random direction ({ref['total_n']} draws, mean $\\pm$ SD)",
        zorder=3,
    )

    ax.set_xticks([*x_rb, x_ref])
    ax.set_xticklabels([r["behavior"] for r in rows] + ["random"])
    ax.set_ylabel("held-out per-direction $R^2$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.75, x_ref + 0.75)
    ax.legend(loc="upper right", frameon=False, ncols=1, handletextpad=0.5)

    paths = savefig_paper(fig, "plot2_persona_directions", dir=OUT_DIR)
    for fmt, p in paths.items():
        print(f"{fmt}: {p}")

    data = {
        "source": str(SOURCE.relative_to(REPO)),
        "note": (
            "held-out per-direction R^2 of the full-ridge map on fold 0 of the 5-fold "
            "split; three persona directions r_B against ONE pooled random-direction "
            "reference bar over all 150 random unit draws"
        ),
        "rows": rows,
        "pooled_random_reference": ref,
    }
    out_json = OUT_DIR / "plot2_persona_directions_data.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=1)
    print(f"data: {out_json}")


if __name__ == "__main__":
    main()
