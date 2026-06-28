#!/usr/bin/env python
"""Issue #685 round-2 figure addenda — the two plan-correct figures the
standing recs require (analyzer-owned re-render).

Consumes the two committed JSONs (Part A / Part B) and emits:
  - figures/issue_685/signed_cosine_vs_null_matched_position.png   (NEW; rec #2)
      one panel per behavior, layers on x-axis, the 10 contexts as points
      (4 SUBSET one colour, 6 HELD-OUT another); null IQR band shaded; instruct.
  - figures/issue_685/matched_vs_response_mean_cosine_scatter.png  (rec #3)
      cell-by-cell scatter of matched-position ABSOLUTE cosine (y) vs
      response-mean ABSOLUTE cosine (x) for instruct, diagonal drawn.
      Replaces the implementer's matched_vs_response_cosine.png (which used
      SIGNED cosines, deviating from plan §6.3 line 230).

0 GPU. Clean matplotlib via the project paper-plots style; no decorative
annotations (saved-feedback "No Plot Annotations").

Usage::

    uv run python scripts/issue685_figures_r2_addenda.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

OUT_DIR = Path("eval_results/issue_685/signed-cosine-matched-position-u")
SIGNED_JSON = OUT_DIR / "delta_vs_u_signed.json"
MATCHED_JSON = OUT_DIR / "delta_vs_u_matched_position.json"
FIG_DIR = Path("figures/issue_685")


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the Part A/B scripts first")
    return json.loads(path.read_text())


def fig_signed_cosine_vs_null_matched(matched: dict) -> None:
    """One panel per behavior; x = layer, y = matched-position signed cosine; 10 contexts as points (instruct)."""
    behaviors = matched["metadata"]["behaviors"]
    layers = matched["metadata"]["layers"]
    subset = set(matched["metadata"]["u_subset_contexts"])
    cells = [c for c in matched["cells"] if c["model"] == "instruct"]

    color_subset = paper_palette_role("baseline")
    color_heldout = paper_palette_role("primary")
    color_null = paper_palette_role("neutral")

    ncol = 3
    nrow = int(np.ceil(len(behaviors) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    layer_pos = {layer: i for i, layer in enumerate(layers)}
    rng = np.random.RandomState(0)
    for bi, behavior in enumerate(behaviors):
        ax = axes[bi]
        lo_band, hi_band = [], []
        for layer in layers:
            lcells = [c for c in cells if c["behavior"] == behavior and c["layer"] == layer]
            lo_band.append(np.mean([c["null_iqr"][0] for c in lcells]))
            hi_band.append(np.mean([c["null_iqr"][1] for c in lcells]))
        xs_band = list(range(len(layers)))
        ax.fill_between(
            xs_band, lo_band, hi_band, color=color_null, alpha=0.18, label="null IQR (random dir)"
        )
        ax.axhline(0.0, color=color_null, lw=0.8, ls="--", alpha=0.6)

        for c in cells:
            if c["behavior"] != behavior:
                continue
            x = layer_pos[c["layer"]] + (
                np.random.RandomState(hash(c["context"]) % 2**31).uniform(-0.12, 0.12)
            )
            in_subset = c["context"] in subset
            ax.scatter(
                x,
                c["signed_cosine"],
                s=28,
                color=color_subset if in_subset else color_heldout,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.9,
            )
        ax.set_title(behavior)
        ax.set_xticks(list(range(len(layers))))
        ax.set_xticklabels([str(layer) for layer in layers])
        if bi % ncol == 0:
            ax.set_ylabel("signed cosine(Δ, û_match)")
        if bi >= len(behaviors) - ncol:
            ax.set_xlabel("layer")
        ax.set_ylim(-0.15, 1.0)

    for j in range(len(behaviors), len(axes)):
        axes[j].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=color_subset, label="build subset (4)"),
        plt.Line2D([], [], marker="o", ls="", color=color_heldout, label="held-out (6)"),
        plt.Line2D([], [], color=color_null, alpha=0.4, lw=6, label="null IQR (random dir)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, "signed_cosine_vs_null_matched_position", dir=str(FIG_DIR))
    plt.close(fig)


def fig_matched_vs_response_abs(matched: dict) -> None:
    """Scatter: matched-position ABSOLUTE cosine (y) vs response-mean ABSOLUTE cosine (x), instruct."""
    cells = [
        c
        for c in matched["cells"]
        if c["model"] == "instruct" and c.get("resp_mean_absolute_cosine") is not None
    ]
    subset = set(matched["metadata"]["u_subset_contexts"])
    color_subset = paper_palette_role("baseline")
    color_heldout = paper_palette_role("primary")
    color_diag = paper_palette_role("neutral")

    xs = np.array([c["resp_mean_absolute_cosine"] for c in cells])
    ys = np.array([c["absolute_cosine"] for c in cells])
    in_sub = np.array([c["context"] in subset for c in cells])

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    lo, hi = 0.0, 1.0
    ax.plot(
        [lo, hi], [lo, hi], color=color_diag, lw=0.9, ls="--", alpha=0.7, label="y = x (no lift)"
    )
    ax.scatter(
        xs[in_sub],
        ys[in_sub],
        s=26,
        color=color_subset,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
        label="build subset (4)",
    )
    ax.scatter(
        xs[~in_sub],
        ys[~in_sub],
        s=26,
        color=color_heldout,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
        label="held-out (6)",
    )
    ax.set_xlabel("response-mean û absolute cosine")
    ax.set_ylabel("matched-position û absolute cosine")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    savefig_paper(fig, "matched_vs_response_mean_cosine_scatter", dir=str(FIG_DIR))
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    matched = _load(MATCHED_JSON)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_signed_cosine_vs_null_matched(matched)
    fig_matched_vs_response_abs(matched)
    print(f"[issue685.fig.addenda] wrote 2 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
