"""Render the variance-matched Matryoshka reanalysis from its saved summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    doc = json.loads(args.summary.read_text())
    rows = doc["results"]["coarsest_vs_rest"]["schemes"]
    keys = ["pooled", "activity", "variance", "variance_and_activity"]
    labels = ["Unmatched", "Activity", "Variance", "Variance + activity"]
    values = np.asarray([rows[k]["value"] for k in keys])
    bounds = np.asarray([rows[k]["ci95"] for k in keys])
    errors = np.stack([values - bounds[:, 0], bounds[:, 1] - values])
    if (errors < 0).any():
        raise ValueError("bootstrap interval excludes the plotted estimate")
    set_c2a_style()
    fig, frac = c2a_figure("wide", aspect=0.50, constrained_layout=True)
    ax = fig.add_subplot(111)
    style_axis(ax, grid_axis="x")
    ax.errorbar(
        values,
        np.arange(4),
        xerr=errors,
        fmt="o",
        color=ROLES["linear"].color,
        markersize=8,
        capsize=5,
        linewidth=2,
    )
    ax.set_yticks(np.arange(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.12, 0.32)
    ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    ax.axvline(0, color=ROLES["control"].color, linewidth=1, linestyle="--")
    ax.set_xlabel("Coarse-tier concordance above chance →")
    ax.set_title("Coarsest versus finer Matryoshka features", loc="left", pad=18)
    saved = save_c2a_figure(
        fig,
        args.out,
        title="Variance-matched Matryoshka concordance",
        subject="Quintile matching; 95% feature bootstrap intervals",
        creator=__file__,
        include_width=frac,
    )
    plt.close(fig)
    metadata = {
        "source": str(args.summary),
        "source_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "rows": {k: rows[k] for k in keys},
        "render": saved["record"],
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
