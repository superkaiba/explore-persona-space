# ruff: noqa: RUF001, RUF002
"""Issue #501 figures — round-2 revision (reader-facing labels + on-figure ρ).

Regenerates two figures that round-1 interpretation critics flagged:

  - ``per_target_bars_v2.{png,pdf}`` — was ``per_target_bars``. New version:
      x-axis uses reader-facing labels (in-context example / system-prompt
      persona / multi-turn drift / multi-turn neutral) instead of bare
      ``IK/SP/MT/MN`` codes; per-category mean band shown as a horizontal
      reference; removes the diagonal p10/p90 claim from prose (it was never
      on the figure).
  - ``layer_sweep_rho_v2.{png,pdf}`` — was ``layer_sweep_rho``. New version:
      annotates the on-figure ρ values for both trajectories at L=14 and L=21,
      so readers do not need to infer "cross-format reaches −0.65 at 14, −0.75
      at 21" from prose.

CLI:
    uv run python scripts/i501_make_figures_v2.py
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger("i501.figures_v2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE5_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase5"
PHASE1_SELF = PROJECT_ROOT / "eval_results" / "issue_501" / "phase1" / "cosine_per_layer.json"
OUT_DIR = PROJECT_ROOT / "figures" / "issue_501"

HEADLINE_LAYER = 21


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
        return out.decode().strip()
    except Exception:
        return "unknown"


def _write_meta(stem: Path, payload: dict) -> None:
    meta = {
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        **payload,
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def _load_merged_cells() -> list[dict]:
    raw = json.loads((PHASE5_DIR / "merged_cells.json").read_text())
    if isinstance(raw, dict) and "cells" in raw:
        return raw["cells"]
    return raw


def _load_cosine() -> dict[int, dict[str, dict[str, float]]]:
    raw = json.loads(PHASE1_SELF.read_text())["cos_sim_per_layer"]
    return {int(k): v for k, v in raw.items()}


def _cos_distance(cos, layer: int, ci: str, cj: str) -> float | None:
    try:
        return 1.0 - cos[layer][ci][cj]
    except KeyError:
        return None


def _categorize_target(cid: str) -> str:
    if cid.startswith("IK"):
        return "in-context example"
    if cid.startswith("SP"):
        return "system-prompt persona"
    if cid.startswith("MT"):
        return "multi-turn drift"
    if cid.startswith("MN"):
        return "multi-turn neutral"
    return "?"


def _safe_savefig(fig, stem: Path, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    _write_meta(stem, payload)


# ---------------------------------------------------------------------------


def figure_per_target_bars_v2(cells: list[dict]) -> None:
    """Per-target mean ΔG with READER-FACING category labels.

    x-axis tick labels are the bare context-ids (preserved for cross-reference
    with eval JSONs), but every group is annotated with a reader-facing
    category label and per-category mean bands are shown.
    """
    import matplotlib.pyplot as plt

    by_target: dict[str, list[float]] = {}
    for c in cells:
        y = c.get("delta_g")
        if y is None or not math.isfinite(y):
            continue
        if c["T_i"] == c["T_j"]:
            continue
        by_target.setdefault(c["T_j"], []).append(y)

    # Order: IK first (16), then SP (8), then MT (8), then MN (4)
    def sort_key(t: str) -> tuple:
        order = {"IK": 0, "SP": 1, "MT": 2, "MN": 3}
        return (order.get(t[:2], 9), t)

    target_order = sorted(by_target.keys(), key=sort_key)
    means = [float(np.mean(by_target[t])) for t in target_order]
    sems = [
        float(np.std(by_target[t]) / max(1, math.sqrt(len(by_target[t])))) for t in target_order
    ]
    cats = [_categorize_target(t) for t in target_order]

    cat_color = {
        "in-context example": "#8a8a8a",
        "system-prompt persona": "#4c72b0",
        "multi-turn drift": "#c44e52",
        "multi-turn neutral": "#7a86d6",
    }
    colors = [cat_color[c] for c in cats]

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    xs = np.arange(len(target_order))
    ax.bar(xs, means, yerr=sems, color=colors, edgecolor="white", linewidth=0.5)

    # Per-category mean bands (horizontal lines spanning each group)
    cat_means: dict[str, list[float]] = {}
    for t, m in zip(target_order, means):
        cat_means.setdefault(_categorize_target(t), []).append(m)
    pos = 0
    for cat in [
        "in-context example",
        "system-prompt persona",
        "multi-turn drift",
        "multi-turn neutral",
    ]:
        if cat not in cat_means:
            continue
        n = len(cat_means[cat])
        cm = float(np.mean(cat_means[cat]))
        ax.hlines(
            cm,
            xmin=pos - 0.4,
            xmax=pos + n - 0.6,
            colors="black",
            linestyles="--",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.text(
            pos + n / 2 - 0.5,
            cm + 0.02,
            f"{cat}\nmean = {cm:.3f}, n = {n}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="black",
        )
        pos += n

    ax.set_xticks(xs)
    ax.set_xticklabels(target_order, rotation=70, fontsize=8)
    ax.set_xlabel(
        "target context (cross-reference: IK = in-context, SP = system-prompt, MT = multi-turn drift, MN = multi-turn neutral)"
    )
    ax.set_ylabel("mean ΔG over off-diagonal sources (nats)")
    ax.set_title("per-target marker log-prob shift, grouped by context category")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.0, max(means) * 1.25)

    _safe_savefig(
        fig,
        OUT_DIR / "per_target_bars_v2",
        {
            "n_targets": len(target_order),
            "target_order": target_order,
            "category_means": {k: float(np.mean(v)) for k, v in cat_means.items()},
            "category_ranges": {k: [float(min(v)), float(max(v))] for k, v in cat_means.items()},
        },
    )
    plt.close(fig)


def figure_layer_sweep_rho_v2(cells: list[dict], cos: dict) -> None:
    """Layer-sweep with on-figure ρ annotations at L=14 and L=21."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    layers = sorted(cos.keys())
    if not layers:
        return
    merged_rhos: list[float] = []
    cross_rhos: list[float] = []
    for li in layers:
        merged_x: list[float] = []
        merged_y: list[float] = []
        cross_x: list[float] = []
        cross_y: list[float] = []
        for c in cells:
            d = _cos_distance(cos, li, c["T_i"], c["T_j"])
            y = c.get("delta_g")
            if d is None or y is None or not math.isfinite(y):
                continue
            merged_x.append(d)
            merged_y.append(y)
            if c.get("is_multi_turn"):
                cross_x.append(d)
                cross_y.append(y)
        merged_rhos.append(float(spearmanr(merged_x, merged_y).statistic))
        cross_rhos.append(float(spearmanr(cross_x, cross_y).statistic))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        layers,
        merged_rhos,
        "o-",
        label="merged 840-cell panel",
        color="#1f1f1f",
        markersize=7,
    )
    ax.plot(
        layers,
        cross_rhos,
        "s--",
        label="cross-format 288-cell subset",
        color="#c44e52",
        markersize=7,
    )
    ax.set_xlabel("residual layer (Qwen-2.5-7B, 28 layers total)")
    ax.set_ylabel("Spearman ρ(cosine distance, ΔG)")
    ax.set_title("layer-sweep predictor strength")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Annotate ρ at every layer for both lines
    for i, li in enumerate(layers):
        # Merged: above the dot
        ax.annotate(
            f"{merged_rhos[i]:+.2f}",
            (li, merged_rhos[i]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color="#1f1f1f",
            fontweight="bold",
        )
        # Cross-format: below the square
        ax.annotate(
            f"{cross_rhos[i]:+.2f}",
            (li, cross_rhos[i]),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
            color="#c44e52",
            fontweight="bold",
        )

    ax.set_ylim(-1.0, 0.25)

    _safe_savefig(
        fig,
        OUT_DIR / "layer_sweep_rho_v2",
        {"layers": layers, "merged_rhos": merged_rhos, "cross_rhos": cross_rhos},
    )
    plt.close(fig)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    cells = _load_merged_cells()
    cos = _load_cosine()
    figure_per_target_bars_v2(cells)
    logger.info("wrote per_target_bars_v2.{png,pdf,meta.json}")
    figure_layer_sweep_rho_v2(cells, cos)
    logger.info("wrote layer_sweep_rho_v2.{png,pdf,meta.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
