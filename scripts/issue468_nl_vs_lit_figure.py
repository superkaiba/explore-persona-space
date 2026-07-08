# ruff: noqa: RUF001  # Greek rho in figure text intentional
"""Issue #468 figure: in-context examples (lit) vs natural-language description (NL).

Plots the per-layer Spearman rho between base-model cosine (S_narrow vs S_broad,
read at the last prompt token = the newline after `assistant` = #468's V5_p5) and
the post-SFT broad-EM rate, across all 28 layers, for the two persona-construction
flavors:

  * lit -- S_narrow built from K=8 real (Q, A) rows of the cell's own training data
  * NL  -- S_narrow is a plain natural-language description of the behavior

Source data: #463's 28-layer last-prompt-token sweep (training probes), which is
the full-depth version of #468's 7-layer reaffirmation. Output: figures/issue_468/.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL = Path(__file__).resolve().parent.parent / "eval_results" / "issue463"
SIG_THRESHOLD = 0.468  # |rho| for two-sided p < 0.05 at n=18


def load_raw_profile(
    path: Path, position: str = "last_prompt_token"
) -> list[tuple[int, float, float]]:
    """Return [(layer, rho_raw, p_raw)] for each available layer."""
    d = json.loads(path.read_text())
    rows: list[tuple[int, float, float]] = []
    for layer in range(28):
        key = f"cossim_{position}_L{layer}"
        if key in d["blocks"]:
            sr = d["blocks"][key]["spearman_raw"]
            rows.append((layer, sr["rho"], sr["p"]))
    return rows


def main() -> None:
    set_paper_style("blog")
    lit = load_raw_profile(EVAL / "regression_training_lit.json")
    nl = load_raw_profile(EVAL / "regression_training_NL.json")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for rows, label, role in [
        (lit, "in-context examples (lit)", "primary"),
        (nl, "natural-language description (NL)", "neutral"),
    ]:
        color = paper_palette_role(role)
        xs = [layer for layer, _, _ in rows]
        ys = [rho for _, rho, _ in rows]
        ax.plot(xs, ys, "-", color=color, linewidth=2, label=label, zorder=3)
        # Filled markers where p < 0.05, hollow otherwise.
        for layer, rho, p in rows:
            if p < 0.05:
                ax.plot(layer, rho, "o", color=color, markersize=6, zorder=4)
            else:
                ax.plot(
                    layer,
                    rho,
                    "o",
                    color="white",
                    markeredgecolor=color,
                    markeredgewidth=1.3,
                    markersize=5,
                    zorder=4,
                )

    ax.axhline(0.0, color="#999999", linewidth=0.8, zorder=1)
    for sign in (+1, -1):
        ax.axhline(
            sign * SIG_THRESHOLD,
            color="#BBBBBB",
            linestyle=":",
            linewidth=1,
            zorder=1,
        )
    ax.text(
        27.3,
        SIG_THRESHOLD + 0.01,
        "p < 0.05",
        fontsize=8,
        color="#888888",
        ha="right",
        va="bottom",
    )

    ax.set_xlabel("transformer layer")
    ax.set_ylabel("Spearman ρ (cosine, post-SFT EM rate)")
    ax.set_xlim(-0.5, 27.5)
    ax.set_ylim(-0.85, 0.85)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title(
        "In-context examples predict EM across the deep band; a description does not",
        fontsize=11,
        loc="left",
        pad=10,
    )

    savefig_paper(fig, "issue_468/lit_vs_nl_v5p5", dir="figures/")
    print("saved figures/issue_468/lit_vs_nl_v5p5.{png,pdf} + .meta.json")


if __name__ == "__main__":
    main()
