# ruff: noqa: RUF001, RUF002
"""Issue #501 figures — over-produce per plan §6.3; analyzer picks hero later.

Inputs (under ``eval_results/issue_501/phase5/``):
  - ``merged_cells.json`` (the 840-cell unified panel from Phase 5)
  - ``H1_verdict.json`` / ``H2_verdict.json`` / ``H3_verdict.json``
  - ``eval_results/issue_501/phase1/cosine_per_layer.json``
    (and the parent #489 cosine if present)
  - ``eval_results/issue_501/phase5/collinearity.json``

Outputs under ``figures/issue_501/``:
  - ``merged_scatter.{png,pdf}``     — Hero candidate H1
  - ``per_target_bars.{png,pdf}``    — Hero candidate H2
  - ``drift_vs_neutral.{png,pdf}``   — Hero candidate H3
  - ``cosine_density_per_arm.{png,pdf}`` — H2(b) "multi-turn sits FAR"
  - ``layer_sweep_rho.{png,pdf}``    — exploratory: ρ vs layer
  - ``per_depth_within_mt.{png,pdf}`` — exploratory: k=10 vs k=14
  - ``js_vs_cos_cross_format.{png,pdf}`` — exploratory secondary
  - Each PNG ships with a ``*.meta.json`` carrying git commit + timestamp +
    cell count for reproducibility.

CLI:
    uv run python scripts/i501_make_figures.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger("i501.figures")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE5_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase5"
PHASE1_SELF = PROJECT_ROOT / "eval_results" / "issue_501" / "phase1" / "cosine_per_layer.json"
PHASE1_PARENT = PROJECT_ROOT / "eval_results" / "issue_489" / "phase1" / "cosine_per_layer.json"
OUT_DIR = PROJECT_ROOT / "figures" / "issue_501"

HEADLINE_LAYER = 21


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
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


def _categorize_target(cid: str) -> str:
    if cid.startswith("MT"):
        return "multi_turn_drift"
    if cid.startswith("MN"):
        return "multi_turn_neutral"
    return "single_turn"


def _safe_savefig(fig, stem: Path, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    fig.savefig(stem.with_suffix(".pdf"))
    _write_meta(stem, payload)
    logger.info("wrote %s.{png,pdf,meta.json}", stem)


def figure_merged_scatter(cells: list[dict], cos: dict, layer: int) -> None:
    import matplotlib.pyplot as plt

    xs_st: list[float] = []
    ys_st: list[float] = []
    xs_mt: list[float] = []
    ys_mt: list[float] = []
    xs_mn: list[float] = []
    ys_mn: list[float] = []
    for c in cells:
        d = _cos_distance(cos, layer, c["T_i"], c["T_j"])
        y = c.get("delta_g")
        if d is None or y is None or not math.isfinite(y):
            continue
        cat = _categorize_target(c["T_j"])
        if cat == "single_turn":
            xs_st.append(d)
            ys_st.append(y)
        elif cat == "multi_turn_drift":
            xs_mt.append(d)
            ys_mt.append(y)
        else:
            xs_mn.append(d)
            ys_mn.append(y)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        xs_st,
        ys_st,
        s=14,
        alpha=0.45,
        label=f"single-turn targets (n={len(xs_st)})",
        color="#888888",
    )
    ax.scatter(
        xs_mt, ys_mt, s=22, alpha=0.7, label=f"multi-turn drift (n={len(xs_mt)})", color="#c44e52"
    )
    ax.scatter(
        xs_mn, ys_mn, s=22, alpha=0.7, label=f"multi-turn neutral (n={len(xs_mn)})", color="#4c72b0"
    )
    ax.set_xlabel(f"cosine distance at L{layer} (1 − cos sim)")
    ax.set_ylabel("on-policy ΔG = log P(' ※') trained − base, post-R slot (nat)")
    ax.set_title("merged geometry-predicts-transfer panel: single-turn + multi-turn targets")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    _safe_savefig(
        fig,
        OUT_DIR / "merged_scatter",
        {
            "n_single_turn": len(xs_st),
            "n_multi_turn_drift": len(xs_mt),
            "n_multi_turn_neutral": len(xs_mn),
            "layer": layer,
        },
    )
    plt.close(fig)


def figure_per_target_bars(cells: list[dict]) -> None:
    import matplotlib.pyplot as plt

    by_target: dict[str, list[float]] = {}
    for c in cells:
        y = c.get("delta_g")
        if y is None or not math.isfinite(y):
            continue
        by_target.setdefault(c["T_j"], []).append(y)
    target_order = sorted(
        by_target.keys(),
        key=lambda t: (
            0 if t.startswith(("IK", "SP")) else (1 if t.startswith("MT") else 2),
            t,
        ),
    )
    means = [float(np.mean(by_target[t])) for t in target_order]
    sems = [
        float(np.std(by_target[t]) / max(1, math.sqrt(len(by_target[t])))) for t in target_order
    ]
    colors = [
        "#888888"
        if t.startswith(("IK", "SP"))
        else ("#c44e52" if t.startswith("MT") else "#4c72b0")
        for t in target_order
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(target_order)), means, yerr=sems, color=colors)
    ax.set_xticks(range(len(target_order)))
    ax.set_xticklabels(target_order, rotation=70, fontsize=7)
    ax.set_ylabel("mean ΔG over 24 single-turn sources (nat)")
    ax.set_title("per-target marker transfer — single-turn + multi-turn")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    _safe_savefig(
        fig,
        OUT_DIR / "per_target_bars",
        {"n_targets": len(target_order), "target_order": target_order},
    )
    plt.close(fig)


def figure_drift_vs_neutral(cells: list[dict]) -> None:
    """Per-domain (4 #377 domains × 2 depths × 2 [drift/neutral]) box+whisker."""
    import matplotlib.pyplot as plt

    drift_cells = [c for c in cells if c["T_j"].startswith("MT")]
    neutral_cells = [c for c in cells if c["T_j"].startswith("MN")]
    drift_dgs = [c["delta_g"] for c in drift_cells if c.get("delta_g") is not None]
    neutral_dgs = [c["delta_g"] for c in neutral_cells if c.get("delta_g") is not None]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [drift_dgs, neutral_dgs],
        labels=[f"drift (n={len(drift_dgs)})", f"neutral (n={len(neutral_dgs)})"],
        showmeans=True,
    )
    ax.set_ylabel("ΔG per cell (nat)")
    ax.set_title("drift vs length-matched-neutral within the 288 cross-format cells")
    ax.grid(True, axis="y", alpha=0.3)
    _safe_savefig(
        fig,
        OUT_DIR / "drift_vs_neutral",
        {"n_drift": len(drift_dgs), "n_neutral": len(neutral_dgs)},
    )
    plt.close(fig)


def figure_cosine_density(cells: list[dict], cos: dict, layer: int) -> None:
    """Overlaid kernel density of cosine_distance per target-arm subset."""
    import matplotlib.pyplot as plt

    bands: dict[str, list[float]] = {"single_turn": [], "multi_drift": [], "multi_neutral": []}
    for c in cells:
        d = _cos_distance(cos, layer, c["T_i"], c["T_j"])
        if d is None:
            continue
        cat = _categorize_target(c["T_j"])
        if cat == "single_turn":
            bands["single_turn"].append(d)
        elif cat == "multi_turn_drift":
            bands["multi_drift"].append(d)
        else:
            bands["multi_neutral"].append(d)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, arr, color in (
        ("single-turn off-diag", bands["single_turn"], "#888888"),
        ("multi-turn drift", bands["multi_drift"], "#c44e52"),
        ("multi-turn neutral", bands["multi_neutral"], "#4c72b0"),
    ):
        if not arr:
            continue
        ax.hist(
            arr, bins=40, alpha=0.45, label=f"{label} (n={len(arr)})", color=color, density=True
        )
    ax.set_xlabel(f"cosine distance at L{layer}")
    ax.set_ylabel("density")
    ax.set_title("cosine-distance distribution per target-arm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _safe_savefig(
        fig,
        OUT_DIR / "cosine_density_per_arm",
        {
            "n_single_turn": len(bands["single_turn"]),
            "n_multi_drift": len(bands["multi_drift"]),
            "n_multi_neutral": len(bands["multi_neutral"]),
            "layer": layer,
        },
    )
    plt.close(fig)


def figure_layer_sweep(cells: list[dict], cos: dict) -> None:
    """ρ(cos_distance_L, ΔG) at L ∈ {7,11,14,15,21,27} on merged + cross."""
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
            if _categorize_target(c["T_j"]) != "single_turn":
                cross_x.append(d)
                cross_y.append(y)
        if len(merged_x) >= 3:
            merged_rhos.append(float(spearmanr(merged_x, merged_y).statistic))
        else:
            merged_rhos.append(float("nan"))
        if len(cross_x) >= 3:
            cross_rhos.append(float(spearmanr(cross_x, cross_y).statistic))
        else:
            cross_rhos.append(float("nan"))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(layers, merged_rhos, "o-", label="merged 840", color="#1f1f1f")
    ax.plot(layers, cross_rhos, "s--", label="cross-format 288", color="#c44e52")
    ax.set_xlabel("layer (Qwen-2.5-7B residual)")
    ax.set_ylabel("Spearman ρ(cosine distance, ΔG) [raw, no length partial]")
    ax.set_title("layer-sweep predictor strength")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_savefig(
        fig,
        OUT_DIR / "layer_sweep_rho",
        {"layers": layers, "merged_rhos": merged_rhos, "cross_rhos": cross_rhos},
    )
    plt.close(fig)


def figure_per_depth(cells: list[dict]) -> None:
    """Within-MT-arm trajectory: per-domain ΔG at k=10 vs k=14."""
    import matplotlib.pyplot as plt

    # MT01/MT02 → coding ; MT03/MT04 → writing ; MT05/MT06 → therapy ;
    # MT07/MT08 → philosophy.
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
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.35
    x = np.arange(len(domains))
    ax.bar(x - width / 2, means_k10, width, label="k=10", color="#7a86d6")
    ax.bar(x + width / 2, means_k14, width, label="k=14", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("mean ΔG over 24 sources (nat)")
    ax.set_title("within-MT depth dose-response per drift domain")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _safe_savefig(
        fig,
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    _ = ap.parse_args(argv)

    cells = _load_merged_cells()
    cos = _load_cosine()

    figure_merged_scatter(cells, cos, HEADLINE_LAYER)
    figure_per_target_bars(cells)
    figure_drift_vs_neutral(cells)
    figure_cosine_density(cells, cos, HEADLINE_LAYER)
    figure_layer_sweep(cells, cos)
    figure_per_depth(cells)
    logger.info("Done — wrote 6 figure stems under %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
