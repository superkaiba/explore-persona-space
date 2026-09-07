"""Add frozen Codex forecasts to the paper's three-panel regression figure.

Default: render the checked-in numeric summary, without private data or inference.
Use --comparison-root to recompute that summary from the completed matched run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis import c2a_plot_style as style  # noqa: E402
from issue1739_result2_fourpanel_fig import (  # noqa: E402
    GROUP_SETTINGS,
    GROUP_WIDTH,
    ICLR_REGRESSION_METHODS,
)
from issue2669_analyze import BOOTSTRAP_DRAWS, SEED, correlation, grouped_indices, rowwise_rho  # noqa: E402
from issue2669_codex_dispatch import atomic_json  # noqa: E402
from issue2669_plot import checksum, load_sources  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "figures/issue_2669/c5_regression_regimes_llm"
METHODS = [m[0] for m in ICLR_REGRESSION_METHODS] + ["codex_0", "codex_32"]
LABELS = {m: label for m, label, _, _ in ICLR_REGRESSION_METHODS} | {
    "reg_oracle": "Regression on real answer",
    "codex_0": "LLM judge · 0 examples",
    "codex_32": "LLM judge · 32 examples",
}
ENCODING = {
    "regression_ctx": (style.ROLES["base_model"].color, None, True),
    "reg_map_linear": (style.ROLES["linear"].color, None, True),
    "reg_oracle": (style.INK, None, True),
    "codex_0": (style.ROLES["other_source"].color, "///", False),
    "codex_32": (style.ROLES["other_source"].color, "xxx", False),
}
REGIMES = [("generic", "generic chat"), ("id", "in-distribution"), ("ood", "completely OOD")]
BEHAVIORS = ["evil", "sycophancy", "hallucination"]


def summarize(root: Path) -> dict:
    """Keep the paper's equal-corpus OOD mean; bootstrap groups within corpus/fold."""
    report, rows, groups = load_sources(root)
    cells = []
    for behavior in BEHAVIORS:
        for regime, paper_group in REGIMES:
            instrument = (
                "fabrication" if behavior == "hallucination" and regime != "generic" else "trait"
            )
            members = groups[(behavior, instrument, regime)]
            corpus_names = GROUP_SETTINGS[behavior][paper_group] if regime == "ood" else [None]
            if regime == "ood" and set(r["rung"] for r in members) != set(corpus_names):
                raise ValueError("OOD corpus coverage differs from the paper")
            indices = grouped_indices(members, BOOTSTRAP_DRAWS, SEED)
            safe = np.maximum(indices, 0)
            truth = np.array([r["dv"] for r in members])
            scores = np.array([[r["scores"][m] for r in members] for m in METHODS])
            corpus_points, corpus_draws = [], []
            for corpus in corpus_names:
                mask = np.array([corpus is None or r["rung"] == corpus for r in members])
                points = [correlation(truth[mask], s[mask]) for s in scores]
                if any(p is None for p in points):
                    raise ValueError("Undefined observed correlation; do not substitute zero")
                key = "/".join((behavior, instrument, regime)) + ("/" + corpus if corpus else "")
                if not np.allclose(
                    points,
                    [report["metrics"][key]["spearman"][m] for m in METHODS],
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError("Paper constituents differ from audited comparison")
                valid = (indices >= 0) & mask[safe]
                yy = np.where(valid, truth[safe], np.nan)
                samples = np.where(valid[None], scores[:, safe], np.nan)
                rr = rowwise_rho(
                    np.broadcast_to(yy, samples.shape).reshape(-1, indices.shape[1]),
                    samples.reshape(-1, indices.shape[1]),
                )
                corpus_draws.append(rr.reshape(len(METHODS), BOOTSTRAP_DRAWS))
                corpus_points.append(
                    dict(
                        corpus=corpus,
                        n=int(mask.sum()),
                        spearman=dict(zip(METHODS, points, strict=True)),
                    )
                )
            estimates = np.mean(
                [[c["spearman"][m] for m in METHODS] for c in corpus_points], axis=0
            )
            draws = np.mean(corpus_draws, axis=0)  # Undefined constituents propagate.
            bars = []
            for method, point, values in zip(METHODS, estimates, draws, strict=True):
                finite = np.isfinite(values)
                interval = np.quantile(values, [0.025, 0.975]) if finite.all() else None
                bars.append(
                    {
                        "method": method,
                        "rho": float(point),
                        "ci95": interval.tolist() if interval is not None else None,
                        "valid_draws": int(finite.sum()),
                        "half_full_endpoint_max_abs_delta": float(
                            np.max(
                                np.abs(
                                    np.quantile(values[: BOOTSTRAP_DRAWS // 2], [0.025, 0.975])
                                    - interval
                                )
                            )
                        )
                        if interval is not None
                        else None,
                    }
                )
            cells.append(
                {
                    "behavior": behavior,
                    "regime": regime,
                    "instrument": instrument,
                    "n": len(members),
                    "n_nonzero_outcomes": int(np.count_nonzero(truth)),
                    "constituents": corpus_points,
                    "bars": bars,
                    "sparse": behavior == "evil" and regime in {"generic", "ood"},
                }
            )
    return {
        "n_pairs": len(rows),
        "n_distinct_contexts": len({r["context_id"] for r in rows}),
        "cells": cells,
        "methods": METHODS,
        "point_statistic": "Spearman rho; OOD is the unweighted mean of corpus-specific rhos, as in the paper",
        "interval": "Pointwise percentile interval of 2000 shared group bootstrap draws, stratified by corpus and original ID fold; OOD recomputes the mean of corpus rhos in each draw. Omit CI if any draw is undefined. This replaces the original plot's averaging of constituent CI endpoints.",
        "seed": SEED,
        "draws": BOOTSTRAP_DRAWS,
        "bootstrap_source": "#2669 completed comparison: 2000 draws inherited; half/full endpoint sensitivity recorded",
        "source_sha256": {
            name: checksum(root / name) for name in ["results.json", "percontextjoined.jsonl"]
        },
    }


def render(data: dict) -> dict:
    """Render original panel order, axes, bars and colors, adding both LLM conditions."""
    expected = Counter({(b, r): 1 for b in BEHAVIORS for r, _ in REGIMES})
    if Counter((c["behavior"], c["regime"]) for c in data["cells"]) != expected:
        raise ValueError("Expected exactly nine paper cells")
    if data["n_pairs"] != 900 or any(
        c["n"] != 100 or [b["method"] for b in c["bars"]] != METHODS for c in data["cells"]
    ):
        raise ValueError("Incomplete matched cohort or method coverage")
    style.set_c2a_style()
    fig, fraction = style.c2a_figure("full", aspect=0.43)
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.64, wspace=0.08)
    width = GROUP_WIDTH / len(METHODS)
    endpoints = [
        v for cell in data["cells"] for bar in cell["bars"] for v in (bar["ci95"] or [bar["rho"]])
    ]
    limits = (min(-0.05, min(endpoints) - 0.04), max(1.0, max(endpoints) + 0.04))
    for ax, behavior in zip(axes, BEHAVIORS, strict=True):
        for position, (regime, _) in enumerate(REGIMES):
            cell = next(
                c for c in data["cells"] if (c["behavior"], c["regime"]) == (behavior, regime)
            )
            alpha = 0.35 if cell["sparse"] else 1
            for index, bar in enumerate(cell["bars"]):
                color, hatch, filled = ENCODING[bar["method"]]
                x = position + (index - (len(METHODS) - 1) / 2) * width
                ax.bar(
                    x,
                    bar["rho"],
                    width=width * 0.91,
                    color=color if filled else style.PAPER,
                    edgecolor=color,
                    hatch=hatch,
                    linewidth=1,
                    alpha=alpha,
                    zorder=3,
                )
                if bar["ci95"] is not None:
                    low, high = bar["ci95"]
                    # Draw actual interval endpoints even if they exclude the estimate.
                    ax.vlines(x, low, high, color=style.INK, lw=1, alpha=alpha, zorder=4)
            if cell["sparse"]:
                ax.text(
                    position,
                    0.97,
                    "sparse",
                    ha="center",
                    va="top",
                    fontsize=13,
                    color=style.MUTED,
                    transform=ax.get_xaxis_transform(),
                )
        style.style_axis(ax)
        ax.axhline(0, color=style.MUTED, linewidth=1)
        ax.set_ylim(*limits)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 1, 2], ["generic chat", "in-distrib.", "OOD"], rotation=20, ha="right")
        style.panel_header(ax, "", behavior, kicker_y=1.08)
    axes[0].set_ylabel(style.better_label("Spearman $\\rho$"))
    handles = []
    for method in METHODS:
        color, hatch, filled = ENCODING[method]
        handles.append(
            Patch(
                facecolor=color if filled else style.PAPER,
                edgecolor=color,
                hatch=hatch,
                label=LABELS[method],
            )
        )
    fig.legend(
        handles=handles[:3],
        loc="upper left",
        bbox_to_anchor=(0.085, 0.99),
        ncol=2,
        fontsize=15,
        columnspacing=1.2,
        handlelength=1.4,
    )
    fig.legend(
        handles=handles[3:],
        loc="upper left",
        bbox_to_anchor=(0.085, 0.855),
        ncol=2,
        fontsize=15,
        columnspacing=1.2,
        handlelength=1.4,
    )
    fig.text(
        0.08,
        0.715,
        "Qwen2.5-7B-Instruct · 95% bootstrap intervals",
        color=style.MUTED,
        fontsize=13,
    )
    fig.text(
        0.99,
        0.715,
        "Matched sample · 100 pairs per group",
        ha="right",
        color=style.MUTED,
        fontsize=14,
    )
    outputs = style.save_c2a_figure(
        fig,
        STEM,
        title="Behavior prediction with the LLM judge baseline",
        subject="Matched 900 context-behavior pairs; corpus-averaged OOD Spearman; sparse evil cells have no interval",
        creator="scripts/issue2669_paper_plot.py",
        include_width=fraction,
    )
    plt.close(fig)
    return {
        "render": outputs["record"],
        "output_sha256": {k: checksum(outputs[k]) for k in ("pdf", "png", "grayscale")},
        "source_sha256": {
            str(p.relative_to(ROOT)): checksum(p)
            for p in [
                Path(__file__),
                Path(style.__file__),
                STEM.with_suffix(".data.json"),
                ROOT / "scripts/issue1739_result2_fourpanel_fig.py",
            ]
        },
        "reproduction_command": "uv run python scripts/issue2669_paper_plot.py",
    }


def main() -> None:
    """Optionally summarize completed predictions, then render a public numeric artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path)
    args = parser.parse_args()
    data_path = STEM.with_suffix(".data.json")
    if args.comparison_root:
        atomic_json(data_path, summarize(args.comparison_root.resolve()))
    data = json.loads(data_path.read_text())
    atomic_json(STEM.with_suffix(".meta.json"), render(data))
    print(json.dumps({"figure": str(STEM), "n_pairs": data["n_pairs"]}))


if __name__ == "__main__":
    main()
