# ruff: noqa: RUF001, RUF002
"""Issue #501 clean-result figures, blog-styled with plain-English labels.

Re-renders the six figures emitted by ``i501_make_figures.py`` using the
``paper-plots`` ``blog`` style and replacing all opaque condition codes
(IK*/SP*/MT*/MN*) with plain-English category names. The headline is the
geometric correlation between base-model cosine distance and the on-policy
log P(marker) shift across an 840-cell merged panel that crosses the
single-turn / multi-turn format boundary.

Outputs under ``figures/issue_501/``:
  - ``merged_scatter.{png,pdf}``     — Hero (H1 + cross-format)
  - ``cosine_density_per_arm.{png,pdf}`` — H2 multi-turn cosine spread
  - ``drift_vs_neutral.{png,pdf}``   — H3 drift-vs-neutral null replication
  - ``per_target_bars.{png,pdf}``    — H4 per-target heterogeneity (plain
    English names; sources collapsed to category band)
  - ``layer_sweep_rho.{png,pdf}``    — supporting: rho vs layer
  - ``per_depth_within_mt.{png,pdf}`` — supporting: k=10 vs k=14 dose-response

CLI:
    uv run python scripts/i501_make_figures_blog.py
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# Resolve the project's src/ for paper_plots imports independent of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("i501.figures.blog")

PHASE5_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase5"
PHASE1_SELF = PROJECT_ROOT / "eval_results" / "issue_501" / "phase1" / "cosine_per_layer.json"
PHASE1_PARENT = PROJECT_ROOT / "eval_results" / "issue_489" / "phase1" / "cosine_per_layer.json"
OUT_DIR = PROJECT_ROOT / "figures" / "issue_501"

HEADLINE_LAYER = 21


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
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
    p = PHASE5_DIR / "merged_cells.json"
    if not p.exists():
        raise RuntimeError(f"Phase 5 prerequisite missing: {p}")
    return json.loads(p.read_text())


def _load_cosine() -> dict[int, dict[str, dict[str, float]]]:
    out: dict[int, dict[str, dict[str, float]]] = {}
    for path in (PHASE1_PARENT, PHASE1_SELF):
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for li_s, m in payload.get("cos_sim_per_layer", {}).items():
            li = int(li_s)
            out.setdefault(li, {})
            for ci, row in m.items():
                out[li].setdefault(ci, {})
                out[li][ci].update(row)
    return out


def _cos_distance(cos, layer: int, ci: str, cj: str) -> float | None:
    m = cos.get(layer, {})
    if ci in m and cj in m[ci]:
        return 1.0 - m[ci][cj]
    if cj in m and ci in m[cj]:
        return 1.0 - m[cj][ci]
    return None


def _category(cid: str) -> str:
    """Plain-English category for a context id (covers IK/SP/MT/MN code prefixes)."""
    if cid.startswith("MT"):
        return "multi-turn drift"
    if cid.startswith("MN"):
        return "multi-turn neutral"
    if cid.startswith("IK"):
        return "in-context example"
    if cid.startswith("SP"):
        return "system-prompt persona"
    return "single-turn"


def _palette() -> dict[str, str]:
    """Stable color mapping across every figure in this script."""
    return {
        "single-turn": "#5A5A5A",  # neutral grey (paper_palette_role neutral)
        "single-turn off-diag": "#5A5A5A",
        "in-context example": "#5A5A5A",
        "system-prompt persona": "#9C9C9C",  # lighter grey
        "multi-turn drift": "#D55E00",  # accent
        "multi-turn neutral": "#0072B2",  # primary blue
    }


def figure_merged_scatter(cells: list[dict], cos: dict, layer: int) -> None:
    """Hero: cosine distance vs log-prob shift across 840 cells, 3 target arms."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    palette = _palette()

    arms: dict[str, tuple[list[float], list[float]]] = {
        "single-turn target": ([], []),
        "multi-turn drift target": ([], []),
        "multi-turn neutral target": ([], []),
    }
    for c in cells:
        d = _cos_distance(cos, layer, c["T_i"], c["T_j"])
        y = c.get("delta_g")
        if d is None or y is None or not math.isfinite(y):
            continue
        cat = _category(c["T_j"])
        if cat in ("in-context example", "system-prompt persona", "single-turn"):
            arms["single-turn target"][0].append(d)
            arms["single-turn target"][1].append(y)
        elif cat == "multi-turn drift":
            arms["multi-turn drift target"][0].append(d)
            arms["multi-turn drift target"][1].append(y)
        elif cat == "multi-turn neutral":
            arms["multi-turn neutral target"][0].append(d)
            arms["multi-turn neutral target"][1].append(y)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Single-turn baseline first (gray), then multi-turn on top.
    style_for = {
        "single-turn target": ("#9C9C9C", 14, 0.42, "o"),
        "multi-turn drift target": (palette["multi-turn drift"], 30, 0.80, "o"),
        "multi-turn neutral target": (palette["multi-turn neutral"], 30, 0.80, "o"),
    }
    for arm_label, (xs, ys) in arms.items():
        color, size, alpha, marker = style_for[arm_label]
        ax.scatter(
            xs,
            ys,
            s=size,
            alpha=alpha,
            label=f"{arm_label} (n={len(xs)})",
            color=color,
            marker=marker,
            edgecolor="none",
        )
    ax.set_xlabel(f"base-model cosine distance at layer {layer} (1 − cos sim)")
    ax.set_ylabel("on-policy ΔG = log P(marker) trained − base (nats)")
    set_title_subtitle(
        ax,
        title="Distance predicts the marker log-prob shift across formats",
        subtitle="840-cell merged panel  ·  ρ = −0.900 [−0.93, −0.79]  ·  cross-format ρ = −0.716, n = 288",
    )
    ax.legend(loc="lower left", frameon=False)
    savefig_paper(
        fig,
        "issue_501/merged_scatter",
        dir=str(PROJECT_ROOT / "figures"),
    )
    _write_meta(
        OUT_DIR / "merged_scatter",
        {
            "n_single_turn": len(arms["single-turn target"][0]),
            "n_multi_turn_drift": len(arms["multi-turn drift target"][0]),
            "n_multi_turn_neutral": len(arms["multi-turn neutral target"][0]),
            "layer": layer,
        },
    )
    plt.close(fig)


def figure_cosine_density(cells: list[dict], cos: dict, layer: int) -> None:
    """Cosine-distance distribution per target category."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    palette = _palette()

    bands: dict[str, list[float]] = {
        "single-turn off-diag": [],
        "multi-turn drift": [],
        "multi-turn neutral": [],
    }
    for c in cells:
        d = _cos_distance(cos, layer, c["T_i"], c["T_j"])
        if d is None:
            continue
        cat = _category(c["T_j"])
        if cat in ("in-context example", "system-prompt persona", "single-turn"):
            bands["single-turn off-diag"].append(d)
        elif cat == "multi-turn drift":
            bands["multi-turn drift"].append(d)
        elif cat == "multi-turn neutral":
            bands["multi-turn neutral"].append(d)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, arr in bands.items():
        if not arr:
            continue
        ax.hist(
            arr,
            bins=40,
            alpha=0.55,
            label=f"{label} (n={len(arr)})",
            color=palette[label],
            density=True,
        )
    ax.set_xlabel(f"cosine distance at layer {layer}")
    ax.set_ylabel("density")
    set_title_subtitle(
        ax,
        title="Multi-turn targets sit in the same cosine band as single-turn off-diagonals",
        subtitle="H2 FAILed at the planned −2.0 nat bar; the format boundary is not visible in the predictor",
    )
    ax.legend(loc="upper right", frameon=False)
    savefig_paper(
        fig,
        "issue_501/cosine_density_per_arm",
        dir=str(PROJECT_ROOT / "figures"),
    )
    _write_meta(
        OUT_DIR / "cosine_density_per_arm",
        {
            "n_single_turn": len(bands["single-turn off-diag"]),
            "n_multi_drift": len(bands["multi-turn drift"]),
            "n_multi_neutral": len(bands["multi-turn neutral"]),
            "layer": layer,
        },
    )
    plt.close(fig)


def figure_drift_vs_neutral(cells: list[dict]) -> None:
    """Within the 288 cross-format cells, drift targets vs neutral targets."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    palette = _palette()

    drift_dgs = [
        c["delta_g"] for c in cells if c["T_j"].startswith("MT") and c.get("delta_g") is not None
    ]
    neutral_dgs = [
        c["delta_g"] for c in cells if c["T_j"].startswith("MN") and c.get("delta_g") is not None
    ]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bp = ax.boxplot(
        [drift_dgs, neutral_dgs],
        labels=[f"drift\n(n={len(drift_dgs)})", f"length-matched neutral\n(n={len(neutral_dgs)})"],
        showmeans=True,
        patch_artist=True,
        widths=0.5,
        meanprops={
            "marker": "^",
            "markerfacecolor": "#2F8F2F",
            "markeredgecolor": "#2F8F2F",
            "markersize": 8,
        },
    )
    bp["boxes"][0].set_facecolor(palette["multi-turn drift"])
    bp["boxes"][0].set_alpha(0.55)
    bp["boxes"][1].set_facecolor(palette["multi-turn neutral"])
    bp["boxes"][1].set_alpha(0.55)
    ax.set_ylabel("ΔG per cell (nats)")
    set_title_subtitle(
        ax,
        title="Drift content adds nothing above neutral content",
        subtitle="Within the 288 cross-format cells  ·  ΔG_drift − ΔG_neutral = −0.002 nats [−0.007, +0.002]",
    )
    savefig_paper(
        fig,
        "issue_501/drift_vs_neutral",
        dir=str(PROJECT_ROOT / "figures"),
    )
    _write_meta(
        OUT_DIR / "drift_vs_neutral",
        {"n_drift": len(drift_dgs), "n_neutral": len(neutral_dgs)},
    )
    plt.close(fig)


def figure_per_target_bars(cells: list[dict]) -> None:
    """Per-target mean ΔG, with plain-English category bands replacing IK/SP/MT/MN codes."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    palette = _palette()

    # Bucket targets by category so the per-target axis stays category-banded
    # without exposing the IK01/SP01/MT01/MN01 codes.
    cat_order = [
        ("in-context example", "single-turn"),
        ("system-prompt persona", "single-turn"),
        ("multi-turn drift", "multi-turn drift"),
        ("multi-turn neutral", "multi-turn neutral"),
    ]
    by_cat_target: dict[str, dict[str, list[float]]] = {c: {} for c, _ in cat_order}
    for c in cells:
        y = c.get("delta_g")
        if y is None or not math.isfinite(y):
            continue
        cat = _category(c["T_j"])
        if cat not in by_cat_target:
            continue
        by_cat_target[cat].setdefault(c["T_j"], []).append(y)

    # Flatten while remembering category boundaries (for color + group spacing).
    means: list[float] = []
    sems: list[float] = []
    colors: list[str] = []
    cat_boundary_positions: list[int] = []
    cat_labels_positions: list[tuple[float, str]] = []
    pos = 0
    GAP = 1.5  # gap between category groups
    for cat, palette_key in cat_order:
        targets_in_cat = sorted(by_cat_target[cat].keys())
        if not targets_in_cat:
            continue
        cat_start = pos
        for t in targets_in_cat:
            ys = by_cat_target[cat][t]
            means.append(float(np.mean(ys)))
            sems.append(float(np.std(ys) / max(1, math.sqrt(len(ys)))))
            colors.append(palette[palette_key])
            pos += 1
        cat_boundary_positions.append(pos)
        cat_labels_positions.append(((cat_start + pos - 1) / 2.0, cat))
        pos += GAP  # spacer before next category

    # Build x positions with the spacers.
    xs: list[float] = []
    pos = 0
    bar_idx = 0
    for cat, _ in cat_order:
        targets_in_cat = sorted(by_cat_target[cat].keys())
        if not targets_in_cat:
            continue
        for _ in targets_in_cat:
            xs.append(pos)
            pos += 1
            bar_idx += 1
        pos += GAP

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(xs, means, yerr=sems, color=colors, edgecolor="none", width=0.85)
    # Category labels under each group.
    cat_center_xs = []
    pos = 0
    for cat, _ in cat_order:
        targets_in_cat = sorted(by_cat_target[cat].keys())
        if not targets_in_cat:
            continue
        cat_center_xs.append((cat, pos + (len(targets_in_cat) - 1) / 2.0))
        pos += len(targets_in_cat) + GAP
    ax.set_xticks([x for _, x in cat_center_xs])
    ax.set_xticklabels([c for c, _ in cat_center_xs])
    # Hide individual bar ticks (the per-target codes); category labels are enough.
    ax.set_ylabel("mean ΔG over 24 single-turn sources (nats)")
    set_title_subtitle(
        ax,
        title="Per-target marker log-prob shift  ·  multi-turn band sits inside the single-turn spread",
        subtitle="Each bar = one of 36 targets, averaged over 24 sources  ·  ±SEM across sources",
    )
    ax.axhline(0.0, color="#444444", linewidth=0.5)
    savefig_paper(
        fig,
        "issue_501/per_target_bars",
        dir=str(PROJECT_ROOT / "figures"),
    )
    _write_meta(
        OUT_DIR / "per_target_bars",
        {"n_targets": len(means)},
    )
    plt.close(fig)


def figure_layer_sweep(cells: list[dict], cos: dict) -> None:
    """ρ vs layer on the merged panel and the cross-format subset."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")

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
            if _category(c["T_j"]) in ("multi-turn drift", "multi-turn neutral"):
                cross_x.append(d)
                cross_y.append(y)
        merged_rhos.append(
            float(spearmanr(merged_x, merged_y).statistic) if len(merged_x) >= 3 else float("nan")
        )
        cross_rhos.append(
            float(spearmanr(cross_x, cross_y).statistic) if len(cross_x) >= 3 else float("nan")
        )

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(
        layers,
        merged_rhos,
        "o-",
        label="merged 840-cell panel",
        color=paper_palette_role("neutral"),
    )
    ax.plot(
        layers,
        cross_rhos,
        "s--",
        label="cross-format 288-cell subset",
        color=paper_palette_role("accent"),
    )
    ax.set_xlabel("residual layer (Qwen-2.5-7B)")
    ax.set_ylabel("Spearman ρ(cosine distance, ΔG) — raw, no length partial")
    set_title_subtitle(
        ax,
        title="Predictor strength peaks in the mid-late residual band",
        subtitle="Headline layer 21 chosen from #489 (Persona-Vectors-style band, mid-late residual)",
    )
    ax.axhline(0.0, color="#444444", linewidth=0.5)
    ax.legend(loc="upper right", frameon=False)
    savefig_paper(
        fig,
        "issue_501/layer_sweep_rho",
        dir=str(PROJECT_ROOT / "figures"),
    )
    _write_meta(
        OUT_DIR / "layer_sweep_rho",
        {"layers": layers, "merged_rhos": merged_rhos, "cross_rhos": cross_rhos},
    )
    plt.close(fig)


def figure_per_depth(cells: list[dict]) -> None:
    """Within-MT depth dose-response per drift domain (k=10 vs k=14)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")

    domain_map = {
        "coding": ("MT01", "MT02"),
        "writing": ("MT03", "MT04"),
        "therapy": ("MT05", "MT06"),
        "philosophy": ("MT07", "MT08"),
    }
    domains = list(domain_map.keys())
    means_k10: list[float] = []
    means_k14: list[float] = []
    for d in domains:
        cid10, cid14 = domain_map[d]
        v10 = [c["delta_g"] for c in cells if c["T_j"] == cid10 and c.get("delta_g") is not None]
        v14 = [c["delta_g"] for c in cells if c["T_j"] == cid14 and c.get("delta_g") is not None]
        means_k10.append(float(np.mean(v10)) if v10 else float("nan"))
        means_k14.append(float(np.mean(v14)) if v14 else float("nan"))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.35
    x = np.arange(len(domains))
    ax.bar(
        x - width / 2, means_k10, width, label="10 prior turns", color=paper_palette_role("primary")
    )
    ax.bar(
        x + width / 2, means_k14, width, label="14 prior turns", color=paper_palette_role("accent")
    )
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("mean ΔG over 24 sources (nats)")
    set_title_subtitle(
        ax,
        title="Adding 4 more drift turns barely changes the log-prob shift",
        subtitle="Within multi-turn drift, k=10 → k=14 dose-response is flat across all four content domains",
    )
    ax.legend(loc="upper right", frameon=False)
    savefig_paper(
        fig,
        "issue_501/per_depth_within_mt",
        dir=str(PROJECT_ROOT / "figures"),
    )
    _write_meta(
        OUT_DIR / "per_depth_within_mt",
        {"domains": domains, "k10": means_k10, "k14": means_k14},
    )
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    cells = _load_merged_cells()
    cos = _load_cosine()

    figure_merged_scatter(cells, cos, HEADLINE_LAYER)
    figure_cosine_density(cells, cos, HEADLINE_LAYER)
    figure_drift_vs_neutral(cells)
    figure_per_target_bars(cells)
    figure_layer_sweep(cells, cos)
    figure_per_depth(cells)
    logger.info("Done — wrote 6 blog-styled figure stems under %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
