"""Plot-only diagnostic figures from a hash-bound repaired China analysis report.

Reads aggregate report JSON only; never opens activations, runs inference or
resampling, or publishes files. All three layers and both languages are shown.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from explore_persona_space.analysis import c2a_plot_style as style

REPORT_CONTRACT = "issue952-china-repair-analysis-v2"
FIGURE_CONTRACT = "issue952-china-repair-figures-v2"
LAYERS = (14, 19, 26)
LANGUAGES = (("en", "English"), ("zh", "Simplified Chinese"))
CONTRASTS = (
    ("subject", "Subject"),
    ("framing", "Framing"),
    ("china_cue", "China cue"),
    ("control_cue", "Control cue"),
)
MASSES = (0.99, 0.90, 0.999)
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """Hash the exact input/output bytes recorded by the provenance sidecar."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    """Reject duplicate keys and nonfinite JSON before selecting plot values."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate field in {path}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON constant {value} in {path}")

    result = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=invalid
    )
    if not isinstance(result, dict):
        raise ValueError("figure source must be a JSON object")
    return result


def write_json(path: Path, value: dict) -> None:
    """Write finite metadata atomically after all figure artifacts exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_report(path: Path, expected_sha256: str | None = None) -> tuple[dict, dict]:
    """Bind the report to an explicit hash or its successful analysis sentinel."""
    path = path.resolve()
    source = {"path": str(path), "sha256": sha256(path)}
    if expected_sha256 is None:
        sentinel_path = path.parent / "done.json"
        sentinel = read_json(sentinel_path)
        if sentinel.get("technical_complete") is not True or sentinel.get("role") != "full":
            raise ValueError("figure rendering requires the complete full-analysis sentinel")
        expected_sha256 = sentinel["report_sha256"]
        source["sentinel"] = {"path": str(sentinel_path), "sha256": sha256(sentinel_path)}
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or source["sha256"] != expected_sha256:
        raise ValueError("figure report hash differs from the requested completed artifact")
    report = read_json(path)
    if (
        report.get("contract") != REPORT_CONTRACT
        or report.get("technical_complete") is not True
        or report.get("role") != "full"
    ):
        raise ValueError("figure source must be a completed repaired-v2 full analysis")
    return report, source


def _share(summary: dict, n_total: int, *, complement: bool = False) -> dict:
    """Read an existing mean/CI, preserving missingness and exact complement bounds."""
    for key in ("n_total", "n_defined", "n_undefined", "bootstrap_n_defined"):
        if type(summary[key]) is not int or summary[key] < 0:
            raise ValueError("share denominators must be explicit nonnegative integers")
    if summary["n_total"] != n_total or summary["n_defined"] + summary["n_undefined"] != n_total:
        raise ValueError("share denominator does not match the full geometry panel")
    mean, interval = summary["mean"], summary["ci95"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("share interval must contain exactly two endpoints")
    present = [value is not None for value in interval]
    if present[0] != present[1]:
        raise ValueError("a partially missing confidence interval is invalid")
    for value in [mean, *interval]:
        if value is not None and (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not -1e-10 <= value <= 1 + 1e-10
        ):
            raise ValueError("squared-norm shares and interval endpoints must lie in [0,1]")
    if (mean is None) != (summary["n_defined"] == 0):
        raise ValueError("share estimate and defined-subject denominator disagree")
    if present[0] and (
        mean is None or interval[0] > interval[1] or summary["bootstrap_n_defined"] == 0
    ):
        raise ValueError("confidence interval is reversed or unsupported by defined draws")
    if complement:
        mean = None if mean is None else 1.0 - mean
        interval = [None, None] if not present[0] else [1.0 - interval[1], 1.0 - interval[0]]
    return {
        "mean": mean,
        "ci95": interval,
        **{
            key: summary[key]
            for key in ("n_total", "n_defined", "n_undefined", "bootstrap_n_defined")
        },
    }


def extract_displayed_data(report: dict, mass: float = 0.99) -> dict:
    """Select all layer/language cells and exactly the four planned aggregate contrasts."""
    if mass not in MASSES:
        raise ValueError("unknown squared-singular-mass cutoff")
    coverage = report["coverage"]
    n_total = coverage["realized_sources"]
    if (
        type(n_total) is not int
        or n_total != 85
        or coverage["planned_sources"] != n_total
        or set(coverage["languages"]) != {key for key, _ in LANGUAGES}
        or coverage["geometry_selected_on_behavior"] is not False
    ):
        raise ValueError("figure source must retain all 85 subjects and both languages")
    selected = [panel for panel in report["panels"] if panel["squared_singular_mass"] == mass]
    if len(selected) != len(LAYERS) or {panel["layer"] for panel in selected} != set(LAYERS):
        raise ValueError("figure cutoff is missing or duplicates a required layer")
    role = "primary" if mass == 0.99 else f"sensitivity_{mass}"
    panels = []
    for panel in sorted(selected, key=lambda item: item["layer"]):
        if panel["role"] != role or panel["n_geometry_subjects"] != n_total:
            raise ValueError("figure panel has a wrong cutoff role or reduced geometry panel")
        for language, label in LANGUAGES:
            h2 = panel["H2"][language]["contrasts"]
            answer = panel["observed_answer_write_decomposition_secondary"][language]
            if answer["basis"] != "RIGHT singular vectors (answer write directions)":
                raise ValueError("observed answer decomposition must use right singular vectors")
            contrasts = []
            for key, contrast_label in CONTRASTS:
                low_answer = answer["contrasts"][key]["low_write_share"]
                zero_counts = {
                    "context": h2[key]["n_exact_zero_vectors"],
                    "answer": answer["contrasts"][key]["n_exact_zero_vectors"],
                }
                if any(
                    type(value) is not int or not 0 <= value <= n_total
                    for value in zero_counts.values()
                ):
                    raise ValueError("zero-vector count is missing or outside the source panel")
                contrasts.append(
                    {
                        "key": key,
                        "label": contrast_label,
                        "context_low_gain": _share(h2[key]["low_gain_share"], n_total),
                        "answer_high_write": _share(low_answer, n_total, complement=True),
                        "answer_low_write": _share(low_answer, n_total),
                        "n_exact_zero_vectors": zero_counts,
                    }
                )
            panels.append(
                {
                    "layer": panel["layer"],
                    "language": language,
                    "language_label": label,
                    "rank_retained": panel["rank_retained"],
                    "rank_low": panel["rank_low"],
                    "contrasts": contrasts,
                }
            )
    return {
        "mass": mass,
        "role": role,
        "n_sources": n_total,
        "n_topics": coverage["topics"],
        "n_resample": report["regime"]["n_resample"],
        "synthetic_fixture": report.get("synthetic_fixture", False) is True,
        "panels": panels,
        "estimators": {
            "point": "mean squared-norm share over defined subject contrasts",
            "interval": "saved 95% topic-cluster bootstrap confidence interval",
            "high_write": "exact complement: mean=1-low_mean; CI=[1-low_upper,1-low_lower]",
            "undefined": "blank row, no point or interval; source denominator retained",
        },
        "scope": "Four aggregate contrasts; constituent and difference-in-differences results remain in the full analysis report.",
    }


def _point_interval(ax, y: float, summary: dict, *, filled: bool) -> None:
    """Draw saved interval endpoints directly, including CIs not containing the mean."""
    if summary["mean"] is None:
        return
    color = style.ROLES["linear"].color
    low, high = summary["ci95"]
    if low is not None:
        ax.hlines(y, low, high, colors=color, linewidth=1.5)
        ax.vlines((low, high), y - 0.045, y + 0.045, colors=color, linewidth=1.5)
    ax.plot(
        summary["mean"],
        y,
        linestyle="none",
        marker=style.ROLES["linear"].marker,
        markersize=8,
        markeredgewidth=1.7,
        markeredgecolor=color,
        markerfacecolor=color if filled else style.PAPER,
    )


def make_figure(data: dict, kind: str) -> tuple[plt.Figure, float, str]:
    """Build an unlettered six-facet diagnostic using only canonical style settings."""
    if kind not in ("context", "answer"):
        raise ValueError("figure kind must be context or answer")
    style.set_c2a_style()
    fig, fraction = style.c2a_figure("full", aspect=0.80)
    title = (
        "Context displacement in low-gain directions"
        if kind == "context"
        else "Observed answer displacement across write directions"
    )
    if data["synthetic_fixture"]:
        title = "Synthetic fixture: " + title
    fig.text(
        0.04,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=style.BASE_FONT_PT["title"],
        fontweight=650,
    )
    fig.text(
        0.04,
        0.94,
        f"Squared singular mass: {100 * data['mass']:g}%",
        color=style.MUTED,
        fontsize=style.BASE_FONT_PT["tick"],
    )
    if kind == "answer":
        color = style.ROLES["linear"].color
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=style.ROLES["linear"].marker,
                markerfacecolor=color if filled else style.PAPER,
                markeredgecolor=color,
                markeredgewidth=1.7,
                markersize=8,
                label=label,
            )
            for label, filled in (("High-write", True), ("Low-write", False))
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.965),
            ncol=2,
            frameon=False,
            handletextpad=0.45,
            columnspacing=1.0,
        )
    grid = fig.add_gridspec(
        3, 2, left=0.22, right=0.975, bottom=0.17, top=0.865, hspace=0.56, wspace=0.92
    )
    axes = []
    for index, panel in enumerate(data["panels"]):
        row, column = divmod(index, 2)
        ax = fig.add_subplot(grid[row, column], sharex=axes[0] if axes else None)
        axes.append(ax)
        labels = []
        for pos, contrast in enumerate(panel["contrasts"]):
            summary = contrast["context_low_gain" if kind == "context" else "answer_low_write"]
            labels.append(f"{contrast['label']} ({summary['n_defined']}/{summary['n_total']})")
            y = len(CONTRASTS) - 1 - pos
            if kind == "context":
                _point_interval(ax, y, summary, filled=True)
            else:
                _point_interval(ax, y + 0.13, contrast["answer_high_write"], filled=True)
                _point_interval(ax, y - 0.13, summary, filled=False)
        ax.set_yticks(list(range(len(CONTRASTS) - 1, -1, -1)), labels)
        ax.set_ylim(-0.5, len(CONTRASTS) - 0.5)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks((0, 0.5, 1), ("0", "0.5", "1"))
        ax.tick_params(labelbottom=row == len(LAYERS) - 1)
        ax.set_title(f"{panel['language_label']} · Layer {panel['layer']}", loc="left", pad=14)
        if row == len(LAYERS) - 1:
            ax.set_xlabel(
                "Low-gain squared-norm share" if kind == "context" else "Squared-norm share"
            )
        style.style_axis(ax, grid_axis="none")
    fig.text(
        0.04,
        0.056,
        "Error bars: saved 95% topic-bootstrap intervals.",
        color=style.MUTED,
        fontsize=style.BASE_FONT_PT["tick"],
    )
    fig.text(
        0.04,
        0.024,
        "Labels: defined subjects / total. Blank rows: undefined shares.",
        color=style.MUTED,
        fontsize=style.BASE_FONT_PT["tick"],
    )
    return fig, fraction, title


def _git_state() -> dict:
    """Record the plotting checkout revision and scoped source-file status."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    paths = [str(Path(__file__).relative_to(ROOT))]
    style_path = Path(style.__file__).resolve()
    if style_path.is_relative_to(ROOT):
        paths.append(str(style_path.relative_to(ROOT)))
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip(),
        "plotting_source_status": status.stdout.splitlines(),
        "loaded_style_path": str(style_path),
    }


def render_report(
    report_path: Path, out_dir: Path, *, mass: float = 0.99, expected_sha256: str | None = None
) -> dict:
    """Export PDF, color PNG, grayscale audit and exact displayed-data metadata."""
    report, source = load_report(report_path, expected_sha256)
    data = extract_displayed_data(report, mass)
    output = {}
    for kind in ("context", "answer"):
        mass_tag = f"{mass:g}".replace(".", "p")
        stem = out_dir / f"china_v2_{kind}_shares_mass{mass_tag}"
        metadata_path = stem.with_suffix(".meta.json")
        regime = {
            "report_sha256": source["sha256"],
            "plotter_sha256": sha256(Path(__file__)),
            "style_sha256": sha256(Path(style.__file__)),
            "mass": mass,
            "kind": kind,
        }
        if metadata_path.exists():
            prior = read_json(metadata_path)
            if prior["regime"] != regime:
                raise ValueError(
                    "figure destination already contains a different render; use a fresh output directory"
                )
            for name, digest in prior["output_sha256"].items():
                if sha256(Path(prior["outputs"][name])) != digest:
                    raise ValueError("completed figure output bytes changed")
            output[kind] = prior
            continue
        fig, fraction, title = make_figure(data, kind)
        try:
            exported = style.save_c2a_figure(
                fig,
                stem,
                title=title,
                subject="Diagnostic frozen-map geometry; saved aggregate estimates and intervals",
                creator=str(Path(__file__).relative_to(ROOT)),
                include_width=fraction,
            )
        finally:
            plt.close(fig)
        outputs = {
            key: str(value.resolve()) for key, value in exported.items() if isinstance(value, Path)
        }
        metadata = {
            "contract": FIGURE_CONTRACT,
            "kind": kind,
            "title": title,
            "source": source,
            "regime": regime,
            "git": _git_state(),
            "render": exported["record"],
            "displayed_data": data,
            "outputs": outputs,
            "output_sha256": {key: sha256(Path(path)) for key, path in outputs.items()},
            "publication_status": "local export; browser URL must be assigned after verified upload",
            "interpretation": "Descriptive diagnostic only; no thresholds, significance symbols, or behavioral claims added.",
        }
        write_json(metadata_path, metadata)
        output[kind] = metadata
    return output


def main() -> None:
    """Render aggregate-only figures from the requested immutable full report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mass", type=float, choices=MASSES, default=0.99)
    parser.add_argument("--expected-report-sha256")
    args = parser.parse_args()
    result = render_report(
        args.report, args.out_dir, mass=args.mass, expected_sha256=args.expected_report_sha256
    )
    print(
        json.dumps(
            {
                "n_figures": len(result),
                "mass": args.mass,
                "formats": ["pdf", "png", "grayscale", "metadata"],
            }
        )
    )


if __name__ == "__main__":
    main()
