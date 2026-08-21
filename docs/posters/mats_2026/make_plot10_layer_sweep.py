"""Poster plot 10 — layer sweep: the map does best at mid depth.

ONE wide single-panel figure for the MATS 2026 poster: held-out whole-map
R^2 of the fitted linear ridge map v_C -> v_A as a function of the
transformer layer at which BOTH vectors are taken (swept together, never
independently), on Qwen2.5-7B-Instruct (28 decoder layers, H=3584).

Source: the committed issue-1901 paper-densify dense 28-layer curve at
n_train = 50,000 (#779 monitoring contexts; fixed held-out test set of
1,000 contexts, seed 42, byte-identical to the original #779 fair-comparison
split). Per layer: ridge whole-map R^2 on the test set + 1000-resample
bootstrap 95% CI (eval_results/issue_1901/paper_densify/layer_curve_n50k.json).
The curve rises through early layers, peaks at layer 19 (R^2 = 0.760), and
declines toward the final layer. Every number is read from the committed
JSON; nothing is hand-typed.

Run:
    uv run python docs/posters/mats_2026/make_plot10_layer_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "eval_results/issue_1901/paper_densify/layer_curve_n50k.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"
STEM = "plot10_layer_sweep"


def load_curve() -> list[dict]:
    """Per-layer held-out R^2 + bootstrap 95% CI, sorted by layer index."""
    d = json.loads(SRC.read_text())
    rows = []
    for rec in d["per_layer"].values():
        ci = rec["ridge"]["bootstrap_ci"]["r2"]
        rows.append(
            {
                "layer": int(rec["layer"]),
                "r2": float(rec["ridge"]["whole_map_r2"]),
                "ci_lo": float(ci["lo"]),
                "ci_hi": float(ci["hi"]),
            }
        )
    rows.sort(key=lambda r: r["layer"])
    assert [r["layer"] for r in rows] == list(range(28)), "expected all 28 layers"
    return rows


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    rows = load_curve()
    layers = [r["layer"] for r in rows]
    r2 = [r["r2"] for r in rows]
    lo = [r["ci_lo"] for r in rows]
    hi = [r["ci_hi"] for r in rows]
    peak = max(rows, key=lambda r: r["r2"])

    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.fill_between(layers, lo, hi, color=colors[0], alpha=0.18, linewidth=0)
    ax.plot(
        layers,
        r2,
        color=colors[0],
        marker="o",
        markersize=3.2,
        linewidth=1.6,
        label="linear map, $n$=50k (95% CI)",
    )
    ax.plot(
        [peak["layer"]],
        [peak["r2"]],
        marker="*",
        markersize=13,
        markeredgecolor="black",
        markeredgewidth=0.6,
        color=colors[1],
        linestyle="none",
        label=f"peak: layer {peak['layer']}",
        zorder=5,
    )

    ax.set_xlabel("layer")
    ax.set_ylabel("held-out $R^2$")
    ax.set_xticks(range(0, 28, 4))
    ax.set_xlim(-0.6, 27.6)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    paths = savefig_paper(fig, STEM, dir=OUT_DIR)
    plt.close(fig)

    data = {
        "source": str(SRC.relative_to(REPO)),
        "model": "Qwen/Qwen2.5-7B-Instruct (28 decoder layers, H=3584)",
        "map": "ridge v_C (cx_last) -> v_A (v_x mean-response answer summary), "
        "both taken at the SAME layer (swept together)",
        "split": json.loads(SRC.read_text())["split"],
        "metric": "held-out whole-map R^2 on the fixed 1,000-context test set; "
        "band = 1000-resample bootstrap 95% CI",
        "peak": {"layer": peak["layer"], "r2": peak["r2"]},
        "points": rows,
    }
    data_path = OUT_DIR / f"{STEM}_data.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"peak layer {peak['layer']} R2={peak['r2']:.4f}")
    for k, p in paths.items():
        print(f"{k}: {p}")
    print(f"data: {data_path}")


if __name__ == "__main__":
    main()
