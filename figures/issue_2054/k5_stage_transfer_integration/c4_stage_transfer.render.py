"""Preview frozen checkpoint transfer on the current speaker figure's panel A.

Run from the repository environment; this reads completed results and only plots.
The archived producer supplies all original bars and heatmaps. Its layout is
updated here because the paper's exported source snapshot predates its layout.
"""

from __future__ import annotations

# Repository bootstrap must precede numerical and plotting imports.
# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from explore_persona_space.analysis.c2a_plot_style import INK, legend_kicker, save_c2a_figure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "eval_results/issue_2054/k5_stage_transfer/results.json"
ORDER = {"Chat": 0, "HELIOS": 2, "Wren": 3, "Dana": 4, "Vex": 5, "Assistant-story": 6}


def digest(path: Path) -> str:
    """Hash the actual input bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Validate the shared endpoints, add six transfer markers, and export."""
    base_path = HERE / "base_figure.data.json"
    producer_path = HERE / "base_figure_source.py"
    base = json.loads(base_path.read_text())
    results = json.loads(RESULTS.read_text())
    assert results["status"] == "complete"
    assert len(results["rows"]) == 30
    assert {s["label"] for s in results["summary"]} == set(ORDER)
    points = []
    for stat in sorted(results["summary"], key=lambda s: ORDER[s["label"]]):
        row = ORDER[stat["label"]]
        np.testing.assert_allclose(
            stat["metrics"]["target_own"]["r2"],
            base["models"]["qwen2.5-7b-instruct"]["own"][row],
            atol=1e-12,
            rtol=0,
        )
        folds = sorted(
            (r for r in results["rows"] if r["label"] == stat["label"]),
            key=lambda r: r["fold"],
        )
        assert [r["fold"] for r in folds] == list(range(5))
        values = [r["metrics"]["frozen"]["r2"] for r in folds]
        mean = stat["metrics"]["frozen"]["r2"]
        np.testing.assert_allclose(np.mean(values), mean, atol=1e-15, rtol=0)
        points.append(
            {
                "label": stat["label"],
                "row": row,
                "r2": mean,
                "folds": values,
                "retention": stat["frozen_retention"],
            }
        )

    def save_preview(fig, stem, **kwargs):
        """Add the checkpoint comparison without changing existing numeric artists."""
        a, b, turn_base, turn_instruct, color_ax = fig.axes
        fig.set_figheight(6.05)
        a.set_position([0.105, 0.15, 0.215, 0.57])
        b.set_position([0.510, 0.15, 0.200, 0.57])
        turn_base.set_position([0.790, 0.495, 0.140, 0.205])
        turn_instruct.set_position([0.790, 0.15, 0.140, 0.205])
        color_ax.set_position([0.950, 0.15, 0.010, 0.55])
        for label in a.texts:
            if label.get_text().endswith("SEPARATE AND SHARED"):
                label.set_text("A  ·  FITS AND BASE→INSTRUCT")
        a_handles, a_labels = a.get_legend_handles_labels()
        b_legend = b.get_legend()
        b_handles = b_legend.legend_handles
        b_labels = [text.get_text() for text in b_legend.texts]
        a.get_legend().remove()
        b_legend.remove()
        for point in points:
            mean, folds = point["r2"], point["folds"]
            a.errorbar(
                mean,
                point["row"],
                xerr=[[mean - min(folds)], [max(folds) - mean]],
                fmt="D",
                markersize=6.5,
                markerfacecolor="white",
                markeredgecolor=INK,
                markeredgewidth=1.1,
                ecolor=INK,
                elinewidth=0.9,
                capsize=2,
                capthick=0.9,
                zorder=8,
            )
        a_handles.append(
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=INK,
                markeredgewidth=1.1,
                markersize=6.5,
            )
        )
        a_labels.append("Base→Instruct: frozen")
        legend_kicker(fig, 0.065, 0.97, "Model and fit · A")
        legend_kicker(fig, 0.570, 0.97, "Transfer method · B")
        fig.legend(
            a_handles,
            a_labels,
            loc="upper left",
            bbox_to_anchor=(0.065, 0.945),
            ncol=2,
            borderaxespad=0,
            handlelength=1.4,
            columnspacing=1.3,
        )
        fig.legend(
            b_handles,
            b_labels,
            loc="upper left",
            bbox_to_anchor=(0.570, 0.945),
            ncol=3,
            borderaxespad=0,
            handlelength=1.4,
            columnspacing=1.3,
        )
        return save_c2a_figure(
            fig,
            HERE / "c4_stage_transfer",
            include_width=kwargs["include_width"],
            title="Maps across speakers, checkpoints and turns",
            subject="Preview: frozen Base-to-Instruct K5 transfer added to panel A",
            creator=str(Path(__file__).relative_to(ROOT)),
        )

    namespace = runpy.run_path(str(producer_path))
    namespace["render"].__globals__["save_c2a_figure"] = save_preview
    record = namespace["render"](base, HERE)
    record.update(
        points=points,
        missing_settings=["Assistant (plain text): checkpoint transfer not evaluated"],
        uncertainty="Marker whiskers: minimum and maximum of the five fold R2 scores; not a CI",
        discussion_reference_overleaf_commit="433800ea38c2075f33aacb2f465984458c4d3b1c",
        source_snapshot_note=(
            "The paper metadata names producer SHA256 "
            "31a03036ec0f5bda64afd95ec5c9f3854b60ca99b69f4a8d873e36c3642caf27. "
            "Its exported source differs and predates the current top legends; this "
            "preview archives that actual source, preserves all numeric input data, "
            "and explicitly supplies its own layout. No exact layout reproduction is claimed."
        ),
        sources={
            str(p.relative_to(ROOT)): {"sha256_bytes": digest(p)}
            for p in [
                base_path,
                producer_path,
                RESULTS,
                Path(__file__),
                ROOT / "src/explore_persona_space/analysis/c2a_plot_style.py",
            ]
        },
        git_head_at_render=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_status_at_render=subprocess.check_output(
            [
                "git",
                "status",
                "--short",
                "--",
                str(HERE.relative_to(ROOT)),
                str(RESULTS.relative_to(ROOT)),
            ],
            cwd=ROOT,
            text=True,
        ).splitlines(),
    )
    (HERE / "c4_stage_transfer.meta.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"validated_settings": len(points), "outputs": "c4_stage_transfer.{pdf,png}"}))


if __name__ == "__main__":
    main()
