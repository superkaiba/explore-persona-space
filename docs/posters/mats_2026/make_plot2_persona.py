"""Poster plot 2 — all SEVEN persona directions vs one random-direction band.

MATS 2026 poster section 2. Held-out per-direction R^2 along each persona
direction r_B, for all seven extracted traits, against a single 200-draw
random-unit band.

FOOTING (Thomas 2026-08-21, chose the 7-trait read over the 3-trait one): every
number comes from eval_results/issue_1482/rb7_reads/rb7_reads.json →
read1_trait_r2.context — the #1738 multi-turn holdout (n=9,941), layer 19,
ridge, `context` arm. That artifact exists precisely so old and new traits are
comparable: its own note records that the original three traits' published
spectrum numbers are the #779 n10k SINGLE-TURN regime, and that all seven were
recomputed on the multi-turn corpus for one footing.

DO NOT mix this with the #779 identity_baseline.json numbers the earlier
version of this figure used (evil 0.790 / sycophancy 0.872 / hallucination
0.795 against a 0.568 random mean). Those are a different corpus, different
folds, and per-trait read-out layers 14 / 26 / 17 rather than a common layer
19. Same quantity, different regime — the two are not on one axis.

The `context` arm is the one matching the poster's v_C (full context), and is
the arm the source script's own headline figure uses; `prefix` and `bare` are
the other two arms in the same artifact and are deliberately not plotted.

HONESTY NOTE, drawn rather than asserted: the random band's p5-p95 is
0.585-0.756, and the two weakest traits (optimistic 0.803, hallucination 0.822)
clear p95 by only ~0.05-0.07. The band is rendered as a shaded span across the
axes so a reader sees that directly instead of taking a bar-height difference
on trust.

Run:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 uv run python docs/posters/mats_2026/make_plot2_persona.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "eval_results/issue_1482/rb7_reads/rb7_reads.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"

ARM = "context"  # matches the poster's v_C; `prefix` / `bare` are the other two arms

# artifact trait key → poster display name. One term per thing: the TL;DR, plot 9's
# panel title and the source artifact all say "evil" (the persona-vectors trait name,
# arXiv 2507.21509), so this figure says "evil" too rather than "misalignment".
DISPLAY = {
    "evil": "evil",
    "sycophancy": "sycophancy",
    "hallucination": "hallucination",
    "optimistic": "optimistic",
    "impolite": "impolite",
    "apathetic": "apathetic",
    "humorous": "humorous",
}

# gap between the seven trait bars and the set-apart random reference bar
REFERENCE_GAP = 0.9


def load_read() -> dict:
    """The seven per-trait R^2 values + the random band, from the `context` arm."""
    with open(SOURCE) as f:
        doc = json.load(f)
    design = doc["design"]
    block = doc["read1_trait_r2"][ARM]
    per = block["per_trait_r2"]
    ranks = block["per_trait_equiv_variance_rank"]
    shares = block["per_trait_variance_share"]
    rows = [
        {
            "trait_key": k,
            "trait": DISPLAY[k],
            "heldout_r2": per[k],
            "equivalent_variance_rank": ranks[k],
            "variance_share": shares[k],
        }
        for k in design["traits"]
    ]
    rows.sort(key=lambda r: r["heldout_r2"], reverse=True)
    return {"design": design, "rows": rows, "random_band": block["random_band"]}


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    read = load_read()
    rows = read["rows"]
    band = read["random_band"]

    c_rb = paper_color("persona_vector")
    c_rand = paper_color("null")

    fig, ax = plt.subplots(figsize=(6.8, 2.7), constrained_layout=True)

    # highest trait at the top; random reference set apart below the group
    n = len(rows)
    y_traits = [float(n) - i for i in range(n)]
    y_ref = y_traits[-1] - REFERENCE_GAP
    h = 0.68

    # the random band's own 5-95% spread, drawn so the reader can see which
    # traits actually clear it rather than trusting a height difference
    # unlabelled: it coincides exactly with the reference bar's error bar, so a
    # second legend entry for it would name the same quantity twice
    ax.axvspan(band["p5"], band["p95"], color=c_rand, alpha=0.18, zorder=0)
    ax.axvline(band["mean"], color=c_rand, lw=0.9, ls="--", zorder=1)

    ax.barh(
        y_traits,
        [r["heldout_r2"] for r in rows],
        height=h,
        color=c_rb,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
        label="persona direction $r_B$",
    )
    ax.barh(
        [y_ref],
        [band["mean"]],
        height=h,
        color=c_rand,
        edgecolor="black",
        linewidth=0.5,
        xerr=[[band["mean"] - band["p5"]], [band["p95"] - band["mean"]]],
        error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "black"},
        zorder=3,
        label=f"random direction ({band['n']} draws, mean and 5-95%)",
    )

    ax.set_yticks([*y_traits, y_ref])
    ax.set_yticklabels([r["trait"] for r in rows] + ["random"])
    ax.set_ylim(y_ref - 0.8, y_traits[0] + 0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("held-out per-direction $R^2$")
    # below the axes: every bar starts at 0 and runs past 0.8, so there is no
    # in-axes space a legend can occupy without covering data
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncols=2,
        frameon=False,
        handletextpad=0.5,
        fontsize="small",
    )

    paths = savefig_paper(fig, "plot2_persona_directions", dir=OUT_DIR)
    for fmt, p in paths.items():
        print(f"{fmt}: {p}")

    data = {
        "source": str(SOURCE.relative_to(REPO)),
        "arm": ARM,
        "arm_note": (
            "`context` is the full-context arm, matching the poster's v_C and the "
            "source script's own headline figure. The artifact also carries `prefix` "
            "and `bare` arms; they are deliberately not plotted."
        ),
        "design": read["design"],
        "note": (
            "Held-out per-direction R^2 along each of the seven extracted persona "
            "directions r_B, #1738 multi-turn holdout n=9,941, layer 19, ridge — one "
            "corpus so all seven traits sit on one footing. NOT comparable with the "
            "#779 identity_baseline.json numbers this figure used previously "
            "(single-turn, fold 0, per-trait read-out layers 14 / 26 / 17)."
        ),
        "rows": read["rows"],
        "random_band": band,
        "band_caveat": (
            f"random band p5-p95 = {band['p5']:.3f}-{band['p95']:.3f}; the two weakest "
            f"traits clear p95 by only ~0.05-0.07, which is why the band is drawn as a "
            f"span rather than summarized by its mean alone."
        ),
    }
    out_json = OUT_DIR / "plot2_persona_directions_data.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=1)
    print(f"data: {out_json}")


if __name__ == "__main__":
    main()
