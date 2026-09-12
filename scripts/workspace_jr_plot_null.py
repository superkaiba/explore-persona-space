#!/usr/bin/env python3
"""Plot the exactly affine synthetic diagnostic, explicitly separate from model results."""

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json  # noqa: E402


def main():
    """Plot saved estimates and CIs; no fitting or selection occurs here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.source.read_text())
    set_c2a_style()
    fig, width = c2a_figure("full", aspect=0.43, constrained_layout=True)
    left, right = fig.subplots(1, 2)
    ks = [5, 10, 25]
    for name, label, role in (
        ("J", "Dictionary A", "linear"),
        ("R", "Dictionary B", "other_source"),
    ):
        style = ROLES[role]
        for component, line in ((name, "-"), (f"rest{name}", "--")):
            values = [
                report["cells"][f"k{k}-rotationNone"]["summary"][f"ridge/{component}"]["estimate"]
                for k in ks
            ]
            suffix = "component" if component == name else "remainder"
            left.plot(
                ks,
                values,
                color=style.color,
                marker=style.marker,
                linestyle=line,
                label=f"{label}: {suffix}",
            )
    left.axhline(1, color=ROLES["control"].color, linewidth=1, linestyle=":", label="Affine total")
    left.set(xlabel="Sparse pursuit steps (k)", ylabel="Held-out R²", xticks=ks)
    left.set_title("Affine totals, nonlinear components", loc="left")
    left.legend(frameon=False, loc="center right")
    for rotation in (20260913, 20260914, 20260915):
        values = [
            report["cells"][f"k{k}-rotation{rotation}"]["summary"]["G_R_minus_G_J"]["estimate"]
            for k in ks
        ]
        right.plot(
            ks,
            values,
            color=ROLES["control"].color,
            alpha=0.4,
            marker="x",
            label="Rotated dictionaries" if rotation == 20260913 else None,
        )
    estimates = [report["cells"][f"k{k}-rotationNone"]["summary"]["G_R_minus_G_J"] for k in ks]
    points = np.array([r["estimate"] for r in estimates])
    intervals = np.array([r["interval"] for r in estimates])
    right.errorbar(
        ks,
        points,
        yerr=np.stack([points - intervals[:, 0], intervals[:, 1] - points]),
        color=ROLES["linear"].color,
        marker="o",
        capsize=4,
        label="Original dictionaries",
    )
    right.axhline(0, color=ROLES["control"].color, linewidth=1)
    right.axhspan(-0.05, 0.05, color=ROLES["control"].color, alpha=0.08)
    right.set(xlabel="Sparse pursuit steps (k)", ylabel="Gap B − gap A", xticks=ks)
    right.set_title("Dictionary differences under the null", loc="left")
    right.legend(frameon=False, loc="upper right")
    for ax in (left, right):
        style_axis(ax)
    saved = save_c2a_figure(
        fig,
        args.out,
        title="Synthetic exact-affine decomposition null",
        subject="Synthetic dictionaries only; no real J/R lenses or language models",
        creator="scripts/workspace_jr_plot_null.py",
        include_width=width,
    )
    save_json(
        args.out.with_suffix(".meta.json"),
        {
            "source": str(args.source),
            "source_sha256": file_sha256(args.source),
            "render": saved["record"],
            "real_model_result": False,
            "k": ks,
            "difference_estimates": estimates,
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
