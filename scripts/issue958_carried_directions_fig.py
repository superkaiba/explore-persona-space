"""Figures for the #958 carried-vs-changed direction spectra (round 3).

Reads eval_results/issue_958/carried-directions/spectra.json and emits:
- figures/issue_958/carried_directions_spectrum.png — per-direction forecast R²
  from the turn-1 state vs PC rank (sorted by the turn-4 value), one line per
  evaluation turn k=2,3,4, the turn-4 random-direction band shaded, and the three
  #778 trait directions as labeled reference points.
- figures/issue_958/carried_vs_persistence.png — forecast R² (k=2) vs raw
  cross-turn persistence (turn 1→2 corr) per direction: the PC cloud + labeled
  trait points, showing forecastability tracks persistence.
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/matplotlib so the shared-VM thread caps bind (#847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results" / "issue_958"
TRAITS = ("evil", "sycophancy", "hallucination")


def _load() -> dict:
    return json.loads((EV / "carried-directions" / "spectra.json").read_text())


def fig_spectrum(d: dict) -> None:
    f = d["forecast_r2"]
    set_paper_style("blog")
    colors = paper_palette(3)
    fig, ax = plt.subplots()

    r2 = {k: np.asarray(f["pc"][str(k)]) for k in (2, 3, 4)}
    order = np.argsort(-r2[4])  # PC order by descending turn-4 forecast R²
    x = np.arange(1, len(order) + 1)
    for i, k in enumerate((2, 3, 4)):
        ax.plot(x, r2[k][order], color=colors[i], lw=1.6, label=f"eval turn {k}")

    rand4 = np.asarray(f["random"]["4"])
    lo, hi = np.quantile(rand4, 0.025), np.quantile(rand4, 0.975)
    ax.axhspan(
        lo,
        hi,
        color=paper_palette_role("neutral"),
        alpha=0.18,
        label="random-direction band (turn 4, 95%)",
    )
    ax.axhline(0, color="0.6", lw=0.8, linestyle=":")

    # trait directions (own layers) as labeled reference points in the right margin
    n = len(order)
    ax.axvline(n + 3, color="0.8", lw=0.8)
    for j, t in enumerate(TRAITS):
        xt = n + 8 + j * 8
        yv = f["trait"]["4"][t]
        ax.plot(xt, yv, marker="D", color=paper_palette_role("accent"), markersize=7)
        ax.annotate(
            f"{t}\n{yv:.2f}",
            (xt, yv),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7,
        )
    ax.set_xlim(-2, n + 30)

    ax.set_xlabel("PC rank (sorted by forecast R² at turn 4)  ·  ◆ = trait dirs")
    ax.set_ylabel("forecast R² from the turn-1 state")
    ax.legend(loc="lower left", fontsize=9)
    set_title_subtitle(
        ax,
        "Which answer directions the turn-1 map carries across turns",
        "per-direction forecast R² at block 19; turns 2–4; random band shaded",
    )
    savefig_paper(fig, "issue_958/carried_directions_spectrum", dir=str(ROOT / "figures"))
    plt.close(fig)


def fig_persistence(d: dict) -> None:
    f, per = d["forecast_r2"], d["persistence"]
    set_paper_style("blog")
    fig, ax = plt.subplots()

    pc_fore = np.asarray(f["pc"]["2"])
    pc_pers = np.asarray(per["pc"]["1"])
    r = float(np.corrcoef(pc_pers, pc_fore)[0, 1])
    ax.scatter(
        pc_pers,
        pc_fore,
        s=14,
        alpha=0.4,
        color=paper_palette_role("primary"),
        label=f"PCs (n={len(pc_fore)}, r={r:.2f})",
    )
    for t in TRAITS:
        xp, yp = per["trait"][t]["1"], f["trait"]["2"][t]
        ax.plot(xp, yp, marker="D", color=paper_palette_role("accent"), markersize=8)
        ax.annotate(t, (xp, yp), textcoords="offset points", xytext=(6, 3), ha="left", fontsize=8)

    ax.set_xlabel("raw cross-turn persistence  (corr ⟨v,ans₁⟩ vs ⟨v,ans₂⟩)")
    ax.set_ylabel("forecast R² from turn-1 state (eval turn 2)")
    ax.legend(loc="upper left", fontsize=9)
    set_title_subtitle(
        ax,
        "The map forecasts a direction largely because it persists",
        "block-19 directions; PC cloud + trait points; Pearson r on PCs",
    )
    savefig_paper(fig, "issue_958/carried_vs_persistence", dir=str(ROOT / "figures"))
    plt.close(fig)


def main() -> None:
    d = _load()
    fig_spectrum(d)
    fig_persistence(d)


if __name__ == "__main__":
    main()
