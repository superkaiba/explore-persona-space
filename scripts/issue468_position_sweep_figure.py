# ruff: noqa: RUF001  # Greek rho in figure text intentional
"""Issue #468 hero: where the EM-prediction signal sits along the trailing band.

Regenerates the position-sweep bar chart (p0..p5 at layer 25, lit flavor,
training probes) with a neutral, prediction-framed title — the earlier render
baked in a "persona-direction signal" title that asserts a geometry mechanism
the clean-result explicitly leaves open. Values are the L25 lit-training
Spearman rho at each of the six trailing chat-template positions.
"""

from __future__ import annotations

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

SIG_THRESHOLD = 0.468  # |rho| for two-sided p < 0.05 at n=18

# Spearman rho(cosine, post-SFT EM rate) at L25, lit flavor, training probes.
POSITIONS = [
    ("p0\nlast content\ntoken", 0.54, True),
    ("p1\n<|im_end|>", -0.49, False),
    ("p2\nnewline", 0.24, False),
    ("p3\n<|im_start|>", 0.40, False),
    ("p4\nassistant", 0.26, False),
    ("p5\nnewline after\nassistant", 0.66, True),
]


def main() -> None:
    set_paper_style("blog")
    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = range(len(POSITIONS))
    rhos = [r for _, r, _ in POSITIONS]
    # p0 (content token) and p5 (newline after assistant) are the named reads.
    colors = [primary if named else neutral for _, _, named in POSITIONS]
    ax.bar(xs, rhos, color=colors, width=0.68, zorder=3)

    for x, (_, rho, _) in zip(xs, POSITIONS, strict=False):
        if abs(rho) >= SIG_THRESHOLD:  # significant at p < 0.05
            va = "bottom" if rho > 0 else "top"
            off = 0.02 if rho > 0 else -0.02
            ax.text(x, rho + off, "*", ha="center", va=va, fontsize=15, color="#333333")

    ax.axhline(0.0, color="#999999", linewidth=0.8, zorder=2)
    for sign in (+1, -1):
        ax.axhline(
            sign * SIG_THRESHOLD,
            color="#BBBBBB",
            linestyle=":",
            linewidth=1,
            zorder=1,
        )
    ax.text(
        len(POSITIONS) - 0.5,
        SIG_THRESHOLD + 0.015,
        "p < 0.05",
        fontsize=8,
        color="#888888",
        ha="right",
        va="bottom",
    )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for lbl, _, _ in POSITIONS], fontsize=8)
    ax.set_ylabel("Spearman ρ (cosine, post-SFT EM rate)")
    ax.set_ylim(-0.75, 0.85)
    ax.set_title(
        "Where the EM-prediction signal sits along the trailing chat-template band",
        fontsize=11,
        loc="left",
        pad=10,
    )
    ax.text(
        0.0,
        1.005,
        "Layer 25, in-context examples, n=18 — the content token (p0) and the "
        "newline after assistant (p5) both predict; * = p < 0.05",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#666666",
        va="bottom",
    )

    savefig_paper(fig, "issue_468/hero_position_sweep", dir="figures/")
    print("saved figures/issue_468/hero_position_sweep.{png,pdf} + .meta.json")


if __name__ == "__main__":
    main()
