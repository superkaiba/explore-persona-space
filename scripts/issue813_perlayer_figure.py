# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, M⁺, M0, ×, −, ‖·‖, c_C) in scientific docstrings + figure titles.
"""Issue #813 free-analysis follow-up figure — per-layer Δ/floor profile (L1-27).

Reads the per-cell profiles ``eval_results/issue_813/perlayer/<behavior>__<substrate>.json``
(written by ``issue813_perlayer_profile.py``) and emits two blog-style figures under
``figures/issue_813/``:

1. perlayer_profile   — 4 behavior panels × 3 substrate lines: layer (x) vs the
                        floor-normalized map change Δ/floor (y), with the frozen
                        headline layer 14 marked. Answers "is L14 representative or
                        cherry-picked?" — a profile whose L14 is a local peak/trough
                        vs one where L14 sits on a flat/monotone stretch.
2. perlayer_ccdrift   — the companion 'input-drift' read: layer (x) vs the median
                        ‖Δc_C‖ (how much finetuning moves the CONTEXT representation
                        itself, the ridge map's INPUT), same 4×3 panel layout.

All numbers are read from the committed per-layer JSONs; nothing is recomputed. Uses the
paper-plots conventions (set_paper_style("blog") + savefig_paper) and plain-English labels
(the no-opaque-condition-codes rule — matching issue813_figures.py's label maps).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval_results/issue_813/perlayer"
OUT = ROOT / "figures/issue_813"

BEHAVIORS = ["em", "fact", "sycophancy", "marker"]
BEH_LABEL = {
    "em": "emergent misalignment",
    "fact": "fact",
    "sycophancy": "sycophancy",
    "marker": "marker",
}
SUBSTRATES = ["generic", "elicit", "mix"]
SUB_LABEL = {
    "generic": "generic UltraChat",
    "elicit": "behavior-eliciting",
    "mix": "mixed pool",
}
HEADLINE_LAYER = 14


def load() -> dict[tuple[str, str], dict]:
    """Load every per-cell per-layer profile that exists under RES."""
    profiles: dict[tuple[str, str], dict] = {}
    for beh in BEHAVIORS:
        for sub in SUBSTRATES:
            path = RES / f"{beh}__{sub}.json"
            if path.exists():
                profiles[(beh, sub)] = json.loads(path.read_text())
    if not profiles:
        raise FileNotFoundError(
            f"no per-layer profiles under {RES} — run issue813_perlayer_profile.py first"
        )
    return profiles


def _series(profile: dict, field: str) -> tuple[list[int], list[float]]:
    """Extract (layers, values) for ``field`` from a profile, dropping None entries."""
    xs, ys = [], []
    for row in profile["per_layer"]:
        v = row.get(field)
        if v is not None:
            xs.append(row["layer"])
            ys.append(float(v))
    return xs, ys


def _panel_grid(profiles: dict, field: str, ylabel: str, stem: str, title: str) -> None:
    """4 behavior panels × 3 substrate lines of ``field`` vs layer, L14 marked."""
    colors = paper_palette_blog(3)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for ax, beh in zip(axes.ravel(), BEHAVIORS, strict=True):
        plotted_any = False
        for i, sub in enumerate(SUBSTRATES):
            profile = profiles.get((beh, sub))
            if profile is None:
                continue
            xs, ys = _series(profile, field)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, color=colors[i], label=SUB_LABEL[sub])
            plotted_any = True
            # mark the frozen headline layer's value for this line
            if HEADLINE_LAYER in xs:
                yv = ys[xs.index(HEADLINE_LAYER)]
                ax.plot([HEADLINE_LAYER], [yv], marker="D", ms=6, color=colors[i], zorder=4)
        ax.axvline(
            HEADLINE_LAYER, color="0.35", ls="--", lw=1.4, label=f"frozen layer {HEADLINE_LAYER}"
        )
        ax.set_title(BEH_LABEL[beh])
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
        if plotted_any:
            ax.margins(x=0.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title, y=1.03)
    savefig_paper(fig, stem, dir=OUT)
    plt.close(fig)


def summarize(profiles: dict) -> None:
    """Print the L14-vs-profile representativeness read (headline line for the report)."""
    print("\n=== L14 representativeness (Δ/floor) ===")
    for beh in BEHAVIORS:
        for sub in SUBSTRATES:
            profile = profiles.get((beh, sub))
            if profile is None:
                continue
            xs, ys = _series(profile, "delta_over_floor")
            if HEADLINE_LAYER not in xs:
                continue
            arr = np.asarray(ys)
            l14 = arr[xs.index(HEADLINE_LAYER)]
            rank = int((arr > l14).sum()) + 1  # 1 = L14 is the max
            print(
                f"{beh:>11}/{sub:<8} L14 Δ/floor={l14:6.3f} | "
                f"max={arr.max():6.3f}@L{xs[int(arr.argmax())]:>2} "
                f"min={arr.min():6.3f}@L{xs[int(arr.argmin())]:>2} | "
                f"L14 rank {rank}/{len(arr)}"
            )


def main() -> None:
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    profiles = load()
    _panel_grid(
        profiles,
        "delta_over_floor",
        "Δ/floor (floor-normalized map change)",
        "perlayer_profile",
        "Per-layer floor-normalized map change (M0 vs M⁺), by behavior × query pool — "
        f"frozen layer {HEADLINE_LAYER} marked",
    )
    _panel_grid(
        profiles,
        "c_C_drift_med",
        "median ‖Δc_C‖ (context-representation drift)",
        "perlayer_ccdrift",
        "Per-layer context-representation drift (median ‖c_C trained − base‖), "
        "by behavior × query pool",
    )
    summarize(profiles)
    print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
