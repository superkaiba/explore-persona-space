"""Render completed matched-900 comparisons with the canonical paper style.

Usage: uv run python scripts/issue2669_plot.py ABSOLUTE_COMPARISON_ROOT [ABSOLUTE_OUT]
No fitting, inference or bootstrap recomputation. Requires all five methods.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from explore_persona_space.analysis import c2a_plot_style as style  # noqa: E402
from issue2669_analyze import correlation  # noqa: E402
from issue2669_codex_dispatch import atomic_json  # noqa: E402
from issue2669_packets import read_rows  # noqa: E402

METHODS = ("codex_0", "codex_32", "regression_ctx", "reg_map_linear", "reg_oracle")
LABELS = {
    "codex_0": "Codex zero-shot",
    "codex_32": "Codex 32-shot",
    "regression_ctx": "Context",
    "reg_map_linear": "Mapped answer",
    "reg_oracle": "Observed answer",
}
# All hues come from canonical semantic roles; open/filled markers distinguish conditions.
ENCODING = {
    "codex_0": (style.ROLES["other_source"].color, "^", False),
    "codex_32": (style.ROLES["other_source"].color, "^", True),
    "regression_ctx": (style.ROLES["linear"].color, "o", False),
    "reg_map_linear": (style.ROLES["linear"].color, "o", True),
    "reg_oracle": (style.INK, "D", True),
}
CELLS = [
    (
        behavior,
        "fabrication" if behavior == "hallucination" and regime != "generic" else "trait",
        regime,
    )
    for behavior in ("evil", "sycophancy", "hallucination")
    for regime in ("id", "generic", "ood")
]


def cell_label(cell: tuple[str, str, str], *, multiline: bool = False) -> str:
    """Use construct names rather than historical internal trait names."""
    behavior, instrument, regime = cell
    title = {
        "evil": "Malicious harm",
        "sycophancy": "Sycophancy",
        "hallucination": "Factual-QA fabrication"
        if instrument == "fabrication"
        else "Hallucination trait",
    }[behavior]
    if multiline and behavior == "hallucination":
        title = "Factual-QA\nfabrication" if instrument == "fabrication" else "Hallucination\ntrait"
    return (
        title
        + ("\n" if multiline else " · ")
        + {"id": "ID", "generic": "Generic", "ood": "OOD"}[regime]
    )


def checksum(path: Path) -> str:
    """Hash exact source or exported artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(root: Path) -> tuple[dict, list[dict], dict]:
    """Validate exact plotted rows and independently reproduce every displayed correlation."""
    report = json.loads((root / "results.json").read_text())
    rows = read_rows(root / "percontextjoined.jsonl")
    if report["complete"] is not True or report["n_contexts"] != 900 or len(rows) != 900:
        raise ValueError("Require completed 900-context comparison")
    if len({r["id"] for r in rows}) != 900:
        raise ValueError("Duplicate plotted context IDs")
    groups = defaultdict(list)
    for row in rows:
        if not set(METHODS) <= set(row["scores"]):
            raise ValueError("All five methods are required; no absent method may be drawn as zero")
        values = [row["dv"], *[row["scores"][m] for m in METHODS]]
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        ):
            raise ValueError("Nonfinite plotted data")
        groups[(row["behavior"], row["instrument"], row["regime"])].append(row)
    if Counter({key: len(value) for key, value in groups.items()}) != Counter(
        {key: 100 for key in CELLS}
    ):
        raise ValueError("Expected 100 contexts in each of the nine plotted cells")
    for key, cell_rows in groups.items():
        metric = report["metrics"]["/".join(key)]
        if metric["n"] != len(cell_rows):
            raise ValueError("Summary/point count mismatch")
        truth = np.array([r["dv"] for r in cell_rows])
        for method in METHODS:
            actual = correlation(truth, np.array([r["scores"][method] for r in cell_rows]))
            expected = metric["spearman"][method]
            if (actual is None) != (expected is None) or (
                actual is not None and not math.isclose(actual, expected, abs_tol=1e-10)
            ):
                raise ValueError("Plotted correlation does not reproduce raw points")
        for method in ("regression_ctx", "reg_map_linear"):
            delta = metric["paired_spearman_differences"][method + "-minus-codex_32"]
            a, b = metric["spearman"][method], metric["spearman"]["codex_32"]
            expected = None if a is None or b is None else a - b
            if (expected is None) != (delta["point"] is None) or (
                expected is not None and not math.isclose(expected, delta["point"], abs_tol=1e-10)
            ):
                raise ValueError("Paired difference point mismatch")
            interval = delta["ci95"]
            if interval is not None and (
                len(interval) != 2
                or not all(math.isfinite(v) for v in interval)
                or interval[0] > interval[1]
            ):
                raise ValueError("Invalid bootstrap interval")
    return report, rows, groups


def marker_kwargs(method: str) -> dict:
    """Return canonical color and redundant marker encoding."""
    color, marker, filled = ENCODING[method]
    return {
        "marker": marker,
        "color": color,
        "markerfacecolor": color if filled else style.PAPER,
        "markeredgecolor": color,
        "markeredgewidth": 1.7,
        "markersize": 8,
        "linestyle": "none",
    }


def summary_figure(report: dict):
    """Plot all-method rank agreement and paired probe-minus-32-shot intervals."""
    fig, fraction = style.c2a_figure("full", aspect=0.80)
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.12, top=0.82, wspace=0.22)
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for row, cell in enumerate(CELLS):
        metric = report["metrics"]["/".join(cell)]
        for offset, method in zip(offsets, METHODS, strict=True):
            value = metric["spearman"][method]
            if value is not None:
                left.plot(value, row + offset, **marker_kwargs(method))
            else:
                left.text(
                    0.98,
                    row + offset,
                    "undefined",
                    transform=left.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=style.BASE_FONT_PT["tick"],
                    color=ENCODING[method][0],
                )
        both_undefined = all(
            metric["paired_spearman_differences"][method + "-minus-codex_32"]["ci95"] is None
            for method in ("regression_ctx", "reg_map_linear")
        )
        for offset, method in zip((-0.12, 0.12), ("regression_ctx", "reg_map_linear"), strict=True):
            delta = metric["paired_spearman_differences"][method + "-minus-codex_32"]
            value, interval = delta["point"], delta["ci95"]
            if value is not None:
                right.plot(value, row + offset, **marker_kwargs(method))
            if interval is not None:
                # Draw endpoints directly: a percentile CI need not contain its point estimate.
                right.hlines(row + offset, *interval, color=ENCODING[method][0], linewidth=1.7)
                right.vlines(
                    interval,
                    row + offset - 0.04,
                    row + offset + 0.04,
                    color=ENCODING[method][0],
                    linewidth=1.4,
                )
            elif not both_undefined:
                right.text(
                    0.98,
                    row + offset,
                    "CI undefined",
                    transform=right.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=style.BASE_FONT_PT["tick"],
                    color=style.MUTED,
                )
        if both_undefined:
            right.text(
                0.98,
                row,
                "CIs undefined",
                transform=right.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=style.BASE_FONT_PT["tick"],
                color=style.MUTED,
            )
    for ax in (left, right):
        style.style_axis(ax, grid_axis="x")
        ax.axvline(0, color=style.SEAM, linewidth=1.2, zorder=0)
        ax.set_ylim(8.6, -0.6)
        ax.set_yticks(range(9))
        for seam in (2.5, 5.5):
            ax.axhline(seam, color=style.SEAM, linewidth=0.8)
    left.set_yticklabels([cell_label(c) for c in CELLS])
    right.set_yticklabels([])
    left.set_xlim(-1.04, 1.04)
    left.set_xticks([-1, -0.5, 0, 0.5, 1])
    right.xaxis.set_major_locator(MaxNLocator(4))
    left.set_xlabel(style.better_label("Spearman correlation"))
    right.set_xlabel("Probe − Codex 32-shot")
    left.set_title("A  Rank agreement", loc="left")
    right.set_title("B  Paired difference\n(95% CI)", loc="left")
    handles = [Line2D([], [], label=LABELS[method], **marker_kwargs(method)) for method in METHODS]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.97),
        ncol=3,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    return fig, fraction


def point_figure(groups: dict):
    """Show every observed outcome/forecast pair without jitter or fitted trend lines."""
    fig, fraction = style.c2a_figure("full", aspect=2.05)
    axes = fig.subplots(9, 5)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.055, top=0.915, wspace=0.48, hspace=0.48)
    headers = (
        "Codex\nzero-shot\n(0–100)",
        "Codex\n32-shot\n(0–100)",
        "Context\n(standardized)",
        "Mapped\nanswer\n(standardized)",
        "Observed\nanswer\n(standardized)",
    )
    for row, cell in enumerate(CELLS):
        data = groups[cell]
        x = np.array([r["dv"] for r in data])
        for column, method in enumerate(METHODS):
            ax = axes[row, column]
            color, marker, filled = ENCODING[method]
            ax.scatter(
                x,
                [r["scores"][method] for r in data],
                s=23,
                marker=marker,
                facecolors=color if filled else "none",
                edgecolors=color,
                linewidths=0.9,
                alpha=0.85,
            )
            style.style_axis(ax, grid_axis="y")
            ax.set_xlim(-3, 103)
            ax.set_xticks([0, 50, 100])
            ax.yaxis.set_major_locator(MaxNLocator(3))
            if method.startswith("codex"):
                ax.set_ylim(-3, 103)
                ax.set_yticks([0, 50, 100])
            if row == 0:
                ax.set_title(headers[column], fontsize=style.BASE_FONT_PT["body"], pad=18)
            if row == 8:
                ax.set_xlabel("Observed score", fontsize=style.BASE_FONT_PT["body"])
        bounds = axes[row, 0].get_position()
        fig.text(
            0.018,
            bounds.y0 + bounds.height / 2,
            cell_label(cell, multiline=True),
            ha="left",
            va="center",
            fontsize=style.BASE_FONT_PT["tick"],
        )
    fig.suptitle(
        "Context-level predictions",
        x=0.18,
        y=0.985,
        ha="left",
        fontsize=style.BASE_FONT_PT["title"],
    )
    return fig, fraction


def export(fig, fraction: float, out: Path, name: str, provenance: dict) -> dict:
    """Export canonical PDF/color/grayscale assets and an exact provenance sidecar."""
    outputs = style.save_c2a_figure(
        fig,
        out / name,
        title=name.replace("_", " "),
        subject="Matched behavior forecasts on frozen contexts",
        creator="scripts/issue2669_plot.py",
        include_width=fraction,
    )
    record = {
        **provenance,
        "render": outputs["record"],
        "outputs": {
            key: {"path": str(outputs[key]), "sha256": checksum(outputs[key])}
            for key in ("pdf", "png", "grayscale")
        },
    }
    atomic_json(out / (name + ".meta.json"), record)
    plt.close(fig)
    return {key: str(outputs[key]) for key in ("pdf", "png", "grayscale")}


def run(root: Path, out: Path) -> dict:
    """Render a validated comparison and export only whitelisted public numeric point data."""
    if not root.is_absolute() or not out.is_absolute():
        raise ValueError("Absolute input and output paths required")
    report, rows, groups = load_sources(root)
    style.set_c2a_style()
    out.mkdir(parents=True, exist_ok=True)
    public_keys = ("id", "behavior", "instrument", "regime", "rung", "dv", "scores", "probe_layers")
    public_points = [{key: row[key] for key in public_keys} for row in rows]
    atomic_json(out / "comparison.data.json", {"cells": report["metrics"], "points": public_points})
    repository = Path(__file__).resolve().parents[1]
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    provenance = {
        "source_inputs": {
            str(root / name): checksum(root / name)
            for name in ("results.json", "percontextjoined.jsonl")
        },
        "script_sha256": checksum(Path(__file__)),
        "style_sha256": checksum(Path(style.__file__)),
        "git_sha": sha,
        "data_sidecar": str(out / "comparison.data.json"),
        "data_sha256": checksum(out / "comparison.data.json"),
        "method_encodings": {
            m: {
                "label": LABELS[m],
                "color": ENCODING[m][0],
                "marker": ENCODING[m][1],
                "filled": ENCODING[m][2],
            }
            for m in METHODS
        },
        "n_per_cell": 100,
        "n_total": 900,
        "ci": "Frozen paired group bootstrap 95% percentile intervals",
        "undefined_policy": "No numeric substitution; undefined values annotated",
        "point_policy": "All 100 raw pairs per cell/method, no jitter or fitted line",
        "protocol_caveats": report.get("protocol_caveats", []),
    }
    result = {}
    for name, builder, args in [
        ("rank_comparison", summary_figure, report),
        ("context_predictions", point_figure, groups),
    ]:
        fig, fraction = builder(args)
        result[name] = export(fig, fraction, out, name, provenance)
    return result


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    destination = (
        Path(sys.argv[2])
        if len(sys.argv) == 3
        else Path(__file__).resolve().parents[1] / "figures/issue_2669"
    )
    print(json.dumps(run(Path(sys.argv[1]), destination), indent=2))
