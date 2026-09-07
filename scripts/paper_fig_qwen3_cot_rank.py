"""Render the validated Qwen3 CoT rank comparison, without fitting or inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    MUTED,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)


def load_results(root: Path) -> tuple[dict, dict]:
    """Require both states and every original fold; never plot incomplete cells as zero."""
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "complete" or summary["completed_cells"] != 10:
        raise ValueError("Need the complete ten-cell result")
    cells = {}
    for state in ("context", "end_of_thought"):
        cells[state] = []
        for fold in range(5):
            row = json.loads((root / f"{state}__fold{fold}.json").read_text())
            if row["status"] != "complete" or row["fold"] != fold or row["state"] != state:
                raise ValueError("Wrong or incomplete state/fold")
            if len(row["test_sse_by_rank"]) != summary["recipe"]["dimension"] + 1:
                raise ValueError("Incomplete rank curve")
            if row["cache_key"]["recipe"] != summary["recipe"]:
                raise ValueError("Mixed analysis recipes")
            cells[state].append(row)
        if [r["selected"]["0.1"]["rank"] for r in cells[state]] != (
            summary["states"][state]["rank10_by_fold"]
        ):
            raise ValueError("Summary ranks disagree with completed cells")
        if [r["input_pr_train"] for r in cells[state]] != (
            summary["states"][state]["input_pr_by_fold"]
        ):
            raise ValueError("Summary input dimensions disagree with completed cells")
    return summary, cells


def draw(root: Path, stem: Path) -> dict:
    """Plot held-out rank curves and paired training-input effective dimensions."""
    summary, cells = load_results(root)
    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=0.40)
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.19, top=0.77, wspace=0.35)
    color = ROLES["linear"].color
    plotted = {}
    for state, label, marker in (
        ("context", "Context", "o"),
        ("end_of_thought", "End of CoT", "s"),
    ):
        rows = cells[state]
        curves = np.array([row["test_sse_by_rank"] for row in rows])
        denom = np.array([row["sst_corpus"] for row in rows])
        pooled = 1 - curves.sum(0) / denom.sum()
        per_fold = 1 - curves / denom[:, None]
        ranks = np.arange(1, len(pooled))
        marks = [int(k - 1) for k in (4, 16, 64, 256, 1024, 4096)]
        left.plot(
            ranks,
            pooled[1:],
            color=color,
            marker=marker,
            markevery=marks,
            markersize=6,
            linewidth=2,
            label=label,
        )
        left.fill_between(
            ranks, per_fold[:, 1:].min(0), per_fold[:, 1:].max(0), color=color, alpha=0.09
        )
        for row in rows:
            selected = row["selected"]["0.1"]
            left.scatter(
                selected["rank"],
                selected["test_r2_corpus"],
                color=color,
                marker=marker,
                s=60,
                zorder=4,
            )
        plotted[state] = {
            "pooled_test_r2_by_rank": pooled.tolist(),
            "rank10_by_fold": summary["states"][state]["rank10_by_fold"],
            "input_pr_by_fold": summary["states"][state]["input_pr_by_fold"],
        }
    left.set_xscale("log", base=2)
    left.set_xticks(
        [1, 4, 16, 64, 256, 1024, 4096], labels=["1", "4", "16", "64", "256", "1,024", "4,096"]
    )
    left.set_xlim(1, 4096)
    left.set_xlabel("Map rank")
    left.set_ylabel(better_label("Held-out $R^2$"))
    left.legend(loc="lower right", frameon=False)
    for fold in range(5):
        offset = (fold - 2) * 0.035
        values = [cells[s][fold]["input_pr_train"] for s in ("context", "end_of_thought")]
        right.plot([offset, 1 + offset], values, color=MUTED, alpha=0.5, linewidth=1)
        for position, (value, marker) in enumerate(zip(values, ("o", "s"), strict=True)):
            right.scatter(position + offset, value, color=color, marker=marker, s=65, zorder=3)
    right.set_xticks([0, 1], labels=["Context", "End of CoT"])
    right.set_xlim(-0.30, 1.30)
    right.set_ylim(bottom=0)
    right.set_ylabel("Input participation ratio")
    for ax in (left, right):
        style_axis(ax)
    panel_header(left, "A", "Qwen3-8B · layer 24", "Prediction across map ranks")
    panel_header(right, "B", "Five paired folds", "Input effective dimension")
    exported = save_c2a_figure(
        fig,
        stem,
        title="Qwen3-8B context and end-of-CoT rank comparison",
        subject="Validation-selected projection ranks and raw input participation ratios",
        creator=Path(__file__).name,
        include_width=frac,
    )
    plt.close(fig)
    inputs = [root / "summary.json", *sorted(root.glob("*__fold*.json"))]
    meta = {
        **as_metadata_dict(
            git_provenance(cwd=Path(__file__).resolve().parents[1]), phase="qwen3-cot-rank-plot"
        ),
        "render": exported["record"],
        "values": plotted,
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "output_sha256": {
            key: hashlib.sha256(exported[key].read_bytes()).hexdigest()
            for key in ("pdf", "png", "grayscale")
        },
        "uncertainty": "Shading is min-max across five dependent folds, not confidence intervals.",
        "markers": "Large markers on curves: each fold's validation-selected rank and test R2.",
        "rank_zero": "Rank zero retained in JSON; log-axis display begins at rank one.",
        "n_models": 1,
        "n_rows": summary["n_rows"],
    }
    with atomic_replace(stem.with_suffix(".meta.json")) as temporary:
        temporary.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    return meta


def main() -> None:
    """Accept an explicit completed result directory and a new output stem."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stem", type=Path, required=True)
    args = parser.parse_args()
    draw(args.results, args.stem)


if __name__ == "__main__":
    main()
