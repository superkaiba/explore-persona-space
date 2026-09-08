#!/usr/bin/env python3
"""Render the issue-2254 pure decode-only quality/trait frontier.

This plot-only entry point is deliberately separate from the immutable grading
and reduction recovery chain.  It reads the completed aggregate result and
writes only the manuscript figure family plus its provenance sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.issue2254_all_answer_decode_analysis as base  # noqa: E402
import scripts.issue2254_all_answer_decode_sweep as gen  # noqa: E402
from explore_persona_space.analysis import c2a_plot_style as style  # noqa: E402


def _asymmetric_ci(values: list[float], rows: list[dict], key: str) -> list[list[float]]:
    return [
        [value - row[key]["ci95"][0] for row, value in zip(rows, values, strict=True)],
        [row[key]["ci95"][1] - value for row, value in zip(rows, values, strict=True)],
    ]


def render(out_root: str | Path) -> Path:
    repo_root = Path(out_root).resolve()
    analysis_root = base.analysis_root(repo_root)
    results_path = analysis_root / "reduce" / "matched_results.json"
    if not results_path.is_file():
        raise base.AnalysisError(f"missing completed reduction: {results_path}")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("n_unique_responses") != 2640 or result.get("structured_completeness") != 1.0:
        raise base.AnalysisError("figure requires the complete 2,640-response primary result")

    style.set_c2a_style()
    fig, include_width = style.c2a_figure("full", aspect=0.46)
    axes = fig.subplots(1, 2)
    colors = {
        "evil": style.ROLES["nonlinear"].color,
        "sycophancy": style.ROLES["linear"].color,
    }
    label_specs = {
        "evil": {
            0.0625: ("0–1/8", (-8, -13), "right"),
            0.25: ("1/4", (-7, 7), "right"),
            0.5: ("1/2", (5, 5), "left"),
            1.0: ("1", (5, 4), "left"),
            2.0: ("2–4", (6, 5), "left"),
        },
        "sycophancy": {
            0.0625: ("0–1/8", (-8, -13), "right"),
            0.25: ("1/4", (-7, 6), "right"),
            0.5: ("1/2", (5, 5), "left"),
            1.0: ("1", (5, 4), "left"),
            2.0: ("2", (5, 5), "left"),
            4.0: ("4", (-7, -10), "right"),
        },
    }
    plotted_values: dict[str, dict] = {}

    for ax, behavior, letter in zip(axes, gen.BEHAVIORS, ("A", "B"), strict=True):
        block = result["behaviors"][behavior]
        rows = block["answer_dose_frontier"]
        xs = [row["integrity_full20"]["estimate"] for row in rows]
        ys = [row["trait_itt_full20"]["estimate"] for row in rows]
        sizes = [46 if row["dose"] else 34 for row in rows]
        ax.errorbar(
            xs,
            ys,
            xerr=_asymmetric_ci(xs, rows, "integrity_full20"),
            yerr=_asymmetric_ci(ys, rows, "trait_itt_full20"),
            fmt="none",
            ecolor=colors[behavior],
            elinewidth=1.15,
            alpha=0.28,
            zorder=1,
        )
        ax.plot(xs, ys, color=colors[behavior], alpha=0.55, lw=2)
        ax.scatter(xs, ys, s=sizes, color=colors[behavior], marker="o", zorder=3)
        for row, x, y in zip(rows, xs, ys, strict=True):
            spec = label_specs[behavior].get(row["dose"])
            if spec is None:
                continue
            label, offset, alignment = spec
            ax.annotate(
                label,
                (x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=13,
                ha=alignment,
                va="center",
            )

        context = block["context"]
        cx = context["integrity_full20"]["estimate"]
        cy = context["trait_itt_full20"]["estimate"]
        ax.errorbar(
            [cx],
            [cy],
            xerr=[
                [cx - context["integrity_full20"]["ci95"][0]],
                [context["integrity_full20"]["ci95"][1] - cx],
            ],
            yerr=[
                [cy - context["trait_itt_full20"]["ci95"][0]],
                [context["trait_itt_full20"]["ci95"][1] - cy],
            ],
            fmt="none",
            ecolor=style.INK,
            elinewidth=1.4,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            [cx],
            [cy],
            s=80,
            marker="D",
            facecolors="white",
            edgecolors=style.INK,
            lw=2,
            zorder=4,
        )
        chosen = block["primary_confirmation"].get("selected_dose")
        if chosen is not None:
            chosen_row = next(row for row in rows if row["dose"] == chosen)
            ax.scatter(
                [chosen_row["integrity_full20"]["estimate"]],
                [chosen_row["trait_itt_full20"]["estimate"]],
                s=150,
                marker="o",
                facecolors="none",
                edgecolors=style.INK,
                lw=2,
                zorder=5,
            )

        style.style_axis(ax, grid_axis="both")
        title = "Evil" if behavior == "evil" else "Sycophancy"
        style.panel_header(ax, letter, title, "All-answer dose–response")
        ax.set_xlabel(style.better_label("Response integrity (0–100)"))
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        plotted_values[behavior] = {
            "answer_frontier": [
                {
                    "dose": row["dose"],
                    "integrity": row["integrity_full20"],
                    "trait_itt": row["trait_itt_full20"],
                }
                for row in rows
            ],
            "context": {
                "cell_id": context["cell_id"],
                "dose": context["dose"],
                "integrity": context["integrity_full20"],
                "trait_itt": context["trait_itt_full20"],
            },
            "selected_dose": chosen,
            "confirmation_status": block["primary_confirmation"]["status"].replace(
                "_", " "
            ),
        }

    axes[0].set_ylabel(style.better_label("Refusal-aware trait score (0–100)"))
    legend_handles = [
        Line2D(
            [],
            [],
            marker="D",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor=style.INK,
            markeredgewidth=1.6,
            linestyle="none",
            label="Context-only target",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor=style.INK,
            markeredgewidth=1.6,
            linestyle="none",
            label="Quality-selected dose",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.76, wspace=0.22)

    figure_root = repo_root / "figures/issue_2254/all_answer_decode_sweep"
    stem = figure_root / "quality_trait_frontier"
    saved = style.save_c2a_figure(
        fig,
        stem,
        title="Context and all-answer decode steering integrity-trait frontier",
        subject="Response-integrity and refusal-aware trait frontier across answer doses",
        creator="scripts/issue2254_all_answer_decode_figure.py",
        include_width=include_width,
    )
    plt.close(fig)
    meta = {
        "analysis_source": str(results_path.relative_to(repo_root)),
        "analysis_sha256": base._sha256_file(results_path),
        "script_path": str(Path(__file__).resolve().relative_to(repo_root)),
        "script_sha256": base._sha256_file(Path(__file__).resolve()),
        "git_commit": base._git_commit(),
        "plot_semantics": {
            "point_labels": (
                "dose multiplier c; clustered labels cover every dose in the stated range"
            ),
            "error_bars": (
                "ordinary 95% paired question-cluster bootstrap confidence intervals"
            ),
            "frontier_scope": "descriptive full 20-question analysis",
        },
        "plotted_values": plotted_values,
        "style": saved["record"],
        "outputs": {
            key: {
                "path": str(Path(path).relative_to(repo_root)),
                "sha256": base._sha256_file(Path(path)),
            }
            for key, path in saved.items()
            if key != "record"
        },
    }
    base._atomic_json(stem.with_suffix(".meta.json"), meta)
    print(f"[figure] {stem}.png", flush=True)
    return stem.with_suffix(".png")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(base._REPO_ROOT))
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    render(args.out_root)


if __name__ == "__main__":
    main()
