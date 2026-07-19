"""Regenerate the two body-embedded k-resample figures with paper-plots sidecars.

The k-resample driver (``scripts/issue1482_kresample.py``) renders its figures
with plain ``fig.savefig`` (no ``.meta.json`` sidecar, no PDF), so the two
views embedded in the #1482 clean-result body are re-rendered here under NEW
filenames via ``savefig_paper`` (the #1482 driver-figure lesson: regenerate
under new names, never overwrite the driver's renders). Reads only the
committed round artifacts under ``eval_results/issue_1482/kresample/``.

Outputs (``figures/issue_1482/kresample/``):
  - ``floor_decomposition_hero``: per-arm decomposition of expected error into
    map share + answer-sampling floor, plus the raw / floor / adjusted
    non-English-minus-English contrasts with 95% bootstrap CIs.
  - ``floor_vs_nerr_points``: per-context floor vs stored normalized error.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval_results/issue_1482/kresample"
OUT = ROOT / "figures/issue_1482/kresample"


def _hero(floor: dict, adj: dict, pal: list[str]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    arms = ["English", "Non-English"]
    adj_means = np.array(
        [floor["per_arm"]["en"]["nerr_adj_mean"], floor["per_arm"]["nonen"]["nerr_adj_mean"]]
    )
    floors = np.array(
        [floor["per_arm"]["en"]["floor_mean"], floor["per_arm"]["nonen"]["floor_mean"]]
    )
    x = np.arange(2)
    ax1.bar(x, adj_means, color=pal[0], label="map share (floor-adjusted error)")
    ax1.bar(x, floors, bottom=adj_means, color=pal[1], label="answer-sampling floor")
    for i in range(2):
        ax1.text(
            x[i], adj_means[i] / 2, f"{adj_means[i]:.3f}", ha="center", va="center", color="white"
        )
        ax1.text(x[i], adj_means[i] + floors[i] / 2, f"{floors[i]:.3f}", ha="center", va="center")
    ax1.set_xticks(x)
    ax1.set_xticklabels(arms)
    ax1.set_ylim(0, 0.40)
    ax1.set_ylabel("mean normalized error")
    ax1.legend(frameon=False, loc="upper center")

    labels = ["raw\ndifference", "floor\ndifference", "adjusted\ndifference"]
    pts = np.array([adj["delta_raw_full"], adj["delta_floor"]["point"], adj["delta_adj"]["point"]])
    cis = np.array([adj["raw_delta_ci"], adj["delta_floor"]["ci"], adj["delta_adj"]["ci"]])
    yerr = np.vstack([pts - cis[:, 0], cis[:, 1] - pts])
    assert (yerr >= 0).all(), "CI bounds must bracket the point estimates"
    ax2.axhline(0.0, color="0.55", lw=1.0)
    ax2.errorbar(np.arange(3), pts, yerr=yerr, fmt="o", color="0.15", capsize=4, markersize=7)
    for i, p in enumerate(pts):
        ax2.text(i + 0.08, p, f"{p:+.3f}", va="center")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels)
    ax2.set_xlim(-0.5, 2.7)
    ax2.set_ylabel("non-English minus English\n(mean normalized error)")

    savefig_paper(fig, "floor_decomposition_hero", dir=OUT)
    plt.close(fig)


def _points(z: np.lib.npyio.NpzFile, pal: list[str]) -> None:
    en = z["arm"] == "en"
    assert en.sum() == 1000 and (~en).sum() == 1000, "expected 1,000 contexts per arm"
    # A handful of contexts have exactly-zero floors (all four fresh draws
    # produced an identical v — very short answers); they cannot render on a
    # log axis, so they are OMITTED here and the count is disclosed in the
    # body caption (19 of 2,000 at the production artifacts).
    pos = z["floor_n"] > 0
    n_zero = int((~pos).sum())
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(
        z["nerr_stored"][en & pos],
        z["floor_n"][en & pos],
        s=10,
        alpha=0.45,
        color=pal[0],
        label=f"English (n={int((en & pos).sum()):,})",
        linewidths=0,
    )
    ax.scatter(
        z["nerr_stored"][~en & pos],
        z["floor_n"][~en & pos],
        s=10,
        alpha=0.45,
        color=pal[1],
        label=f"non-English (n={int((~en & pos).sum()):,})",
        linewidths=0,
    )
    print(f"omitted {n_zero} zero-floor contexts from the log-axis view")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("stored per-context normalized error (parent single draw)")
    ax.set_ylabel("answer-sampling floor (normalized units)")
    ax.legend(frameon=False, loc="upper left")
    savefig_paper(fig, "floor_vs_nerr_points", dir=OUT)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    pal = paper_palette_blog(3)
    floor = json.loads((SRC / "floor_summary.json").read_text())
    adj = json.loads((SRC / "adjusted_contrast.json").read_text())
    z = np.load(SRC / "percontext_floor.npz")
    _hero(floor, adj, pal)
    _points(z, pal)
    print(f"wrote 2 figures (png+pdf+meta.json) under {OUT}")


if __name__ == "__main__":
    main()
