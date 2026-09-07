"""Descriptive natural generic-data scaling plots from the verified150-cell audit.

No new fitting, bootstrap, judge calls, or manuscript replacement. Bands are
the five seed extrema, not confidence intervals. Dataset means are unweighted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    MUTED,
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.atomic_io import atomic_replace
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts.issue1739_natural_audit import (
    FROZEN,
    HOLDOUTS,
    NATURAL_ROSTER,
    TRAITS,
    U_GRID,
    aggregate,
    file_sha,
    require,
)

METHODS = {
    NATURAL_ROSTER[0]: {
        "label": "Direct context",
        "color": ROLES["control"].color,
        "marker": "x",
        "linestyle": "--",
    },
    NATURAL_ROSTER[1]: {
        "label": "Mapped answer",
        "color": ROLES["linear"].color,
        "marker": ROLES["linear"].marker,
        "linestyle": "-",
    },
    NATURAL_ROSTER[2]: {
        "label": "Answer oracle",
        "color": INK,
        "marker": "^",
        "linestyle": ":",
        "fillstyle": "none",
    },
}
DATASET_LABELS = {
    "hhrt": "HH red-teaming",
    "toxicchat": "ToxicChat",
    "evil_mhj": "MHJ",
    "evil_pair": "PAIR",
    "evil_tomgibbs": "TomGibbs",
    "aita": "AITA",
    "sycoans": "Sycophancy: answers",
    "sycoays": "Are you sure?",
    "sycofb": "Sycophancy: feedback",
    "sycomim": "Sycophancy: mimicry",
    "sycomwe": "Sycophancy: MWE",
    "nqopen": "NQ-Open",
    "simpleqa": "SimpleQA",
}


def _x_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([250, 1000, 10000, 100000], ["250", "1k", "10k", "100k"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(210, 120000)
    style_axis(ax)


def _series(ax, x, statistics, key, style):
    rows = [s[key] for s in statistics]
    ax.fill_between(
        x,
        [r["min"] for r in rows],
        [r["max"] for r in rows],
        color=style["color"],
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(x, [r["mean"] for r in rows], linewidth=2, markersize=5, **style)


def plot_overview(curves):
    """Three behavior facets; absolute performance above paired seed differences."""
    fig, frac = c2a_figure("full", aspect=0.60)
    axes = fig.subplots(2, 3, sharex="col", sharey="row", gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.86, hspace=0.25, wspace=0.20)
    for col, behavior in enumerate(TRAITS):
        rows = sorted(
            (r for r in curves if r["behavior"] == behavior), key=lambda r: r["generic_u"]
        )
        x, statistics = [r["generic_u"] for r in rows], [r["statistics"] for r in rows]
        for arm, style in METHODS.items():
            _series(axes[0, col], x, statistics, arm, style)
        _series(
            axes[1, col],
            x,
            statistics,
            "mapped_minus_context",
            {k: v for k, v in METHODS[NATURAL_ROSTER[1]].items() if k != "label"},
        )
        axes[1, col].axhline(0, color=MUTED, linewidth=1, linestyle="--", zorder=0)
        panel_header(axes[0, col], "", behavior, kicker_y=1.035)
        for row in range(2):
            _x_axis(axes[row, col])
    axes[0, 0].set_ylabel("Held-out Spearman $\u03c1$ ↑")
    axes[1, 0].set_ylabel("Mapped − direct $\u03c1$")
    fig.supxlabel("Generic context–answer pairs", y=0.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.98), ncol=3)
    return fig, frac


def dataset_curves(cells):
    grouped = defaultdict(list)
    for cell in cells:
        for row in cell["primary"]:
            grouped[(row["behavior"], row["dataset"], row["generic_u"], row["arm"])].append(row)
    rows = []
    for (behavior, dataset, u, arm), records in sorted(grouped.items()):
        require(
            {r["seed"] for r in records} == set(range(5)) and len(records) == 5,
            "dataset curve missing a seed",
        )
        require(len({r["n_eval"] for r in records}) == 1, "dataset eval count varies by seed")
        values = [r["rho"] for r in records]
        rows.append(
            {
                "behavior": behavior,
                "dataset": dataset,
                "generic_u": u,
                "arm": arm,
                "n_eval": records[0]["n_eval"],
                "mean": float(np.mean(values)),
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def plot_datasets(rows, behavior):
    names = sorted(HOLDOUTS[behavior])
    ncol = min(3, len(names))
    nrow = math.ceil(len(names) / ncol)
    fig, frac = c2a_figure("full", aspect=0.31 * nrow + 0.10)
    axes = np.asarray(fig.subplots(nrow, ncol, squeeze=False, sharex=True, sharey=True))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.85, hspace=0.35, wspace=0.20)
    for ax, dataset in zip(axes.flat, names, strict=False):
        selected = [r for r in rows if r["behavior"] == behavior and r["dataset"] == dataset]
        for arm, style in METHODS.items():
            points = sorted((r for r in selected if r["arm"] == arm), key=lambda r: r["generic_u"])
            _series(ax, [r["generic_u"] for r in points], [{arm: r} for r in points], arm, style)
        panel_header(ax, "", DATASET_LABELS[dataset], kicker_y=1.025)
        _x_axis(ax)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    fig.supylabel("Held-out Spearman $\u03c1$ ↑", x=0.005)
    fig.supxlabel("Generic context–answer pairs", y=0.025)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.985), ncol=3)
    return fig, frac


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fig-dir", type=Path, required=True)
    p.add_argument("--figure-url-base", required=True)
    args = p.parse_args()
    require(args.figure_url_base.startswith("https://"), "browser-accessible figure URL required")
    audit = json.loads(args.audit.read_text())
    summary = aggregate(audit["cells"], require_full=True)
    require(summary == audit["aggregate"], "saved aggregate differs from verified cells")
    require(
        all(
            c.get("remote_verification", {}).get("all_names_sizes_content_verified")
            for c in audit["cells"]
        ),
        "all150 remote content proofs required for final plots",
    )
    per_dataset = dataset_curves(audit["cells"])
    provenance = {
        **as_metadata_dict(git_provenance(ROOT), phase="natural-scaling-descriptive-analysis"),
        "audit_sha256": file_sha(args.audit),
        "scientific_commit": audit["scientific_commit"],
        "source_scripts_sha256": {
            name: file_sha(ROOT / name)
            for name in (
                "scripts/issue1739_natural_audit.py",
                "scripts/issue1739_natural_figures.py",
                "src/explore_persona_space/analysis/c2a_plot_style.py",
            )
        },
        "uncertainty": "Five-seed min-max range; descriptive, not a confidence interval.",
        "evaluation": "Each P-B readout's own held-out dataset; unweighted dataset means.",
    }
    set_c2a_style()
    figures = [("natural_scaling_overview", *plot_overview(summary["curves"]))]
    figures.extend((f"natural_scaling_{b}", *plot_datasets(per_dataset, b)) for b in TRAITS)
    render_records = []
    for stem, fig, frac in figures:
        rendered = save_c2a_figure(
            fig,
            args.fig_dir / stem,
            include_width=frac,
            title="Natural generic-data scaling",
            subject="P-B readout-heldout performance; five-seed min-max bands",
            creator="scripts/issue1739_natural_figures.py",
        )
        record = {
            **provenance,
            "stem": stem,
            "render": rendered["record"],
            "url": args.figure_url_base.rstrip("/") + "/" + stem + ".png",
            "files_sha256": {k: file_sha(rendered[k]) for k in ("pdf", "png", "grayscale")},
            "plotted_values": summary["curves"]
            if stem.endswith("overview")
            else [r for r in per_dataset if stem == "natural_scaling_" + r["behavior"]],
        }
        _write_json(args.fig_dir / f"{stem}.meta.json", record)
        render_records.append(record)
        plt.close(fig)
    payload = {
        **provenance,
        "schema_version": 1,
        "scientific_commit": audit["scientific_commit"],
        "audit_sha256": file_sha(args.audit),
        "summary": summary,
        "fixed_trait_pairs": TRAITS,
        "generic_rungs": U_GRID,
        "frozen_global_layers": FROZEN,
        "per_dataset": per_dataset,
        "primary_seed_rows": [r for c in audit["cells"] for r in c["primary"]],
        "reconstruction_seed_rows": [r for c in audit["cells"] for r in c["reconstruction"]],
        "cell_provenance": [
            {
                k: c[k]
                for k in (
                    "behavior",
                    "generic_u",
                    "seed",
                    "manifest_sha256",
                    "selected_context_ids_sha256",
                    "remote_verification",
                    "files_sha256",
                    "input_sha256",
                )
            }
            for c in audit["cells"]
        ],
        "figures": render_records,
        "limitations": [
            "Readout LODO, not dataset-heldout mapping: fixed trait mapping pool retained.",
            "Same evaluation contexts recur across seeds; seed ranges are descriptive.",
            "Whitening refit at every U, so all three methods can vary with U.",
            "One model, one natural source and one fixed trait pool; no new equivalence test.",
            "Dataset means are unweighted; per-dataset results retained.",
        ],
    }
    _write_json(args.out_dir / "analysis.json", payload)
    with atomic_replace(args.out_dir / "per_dataset.csv") as tmp, tmp.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_dataset[0]))
        writer.writeheader()
        writer.writerows(per_dataset)
    print(
        json.dumps(
            {
                "realized_cells": summary["realized_cells"],
                "figures": [r["url"] for r in render_records],
            }
        )
    )


if __name__ == "__main__":
    main()
