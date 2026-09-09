"""Plot the completed, count-corrected refusal magnitude comparison; no analysis rerun."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

REPO = Path(__file__).resolve().parent.parent
SOURCE = (
    REPO / "eval_results/issue_2569/followup_refusal_magnitude_comparison_20260909_countfix_final"
)
OUTPUT = REPO / "figures/issue_2569/refusal_magnitude_comparison_20260909_final"
SCORES = (
    ("context_norm", "Context\nchange", "control"),
    ("mapped_norm", "Mapped\nchange", "linear"),
    ("observed_answer_norm", "Observed answer\nchange", "control"),
)


def sha256(path: Path) -> str:
    """Hash the exact small artifact rendered or written."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source: Path, output: Path) -> dict:
    """Render only validated saved correlations and persist full figure provenance."""
    summary_path = source / "summary.json"
    completion = json.loads((source / "completion.json").read_text())
    summary = json.loads(summary_path.read_text())
    primary = summary["primary"]
    assert primary["n"] == 124 and primary["n_clusters"] == 21
    # Require the exact completed summary, not a potentially partial live file.
    completed_hashes = {item["path"]: item["sha256"] for item in completion["outputs"]}
    assert completion["exit_code"] == 0
    assert completed_hashes[str(summary_path.relative_to(REPO))] == sha256(summary_path)
    output.mkdir(parents=True, exist_ok=False)
    set_c2a_style()
    fig, include_width = c2a_figure("half", aspect=0.76)
    ax = fig.add_axes((0.19, 0.25, 0.77, 0.53))
    for position, (key, _label, role) in enumerate(SCORES):
        result = primary["scores"][key]
        value, (lo, hi) = result["rho"], result["ci95"]
        assert -1 <= lo <= value <= hi <= 1
        assert result["valid_bootstrap"] == 2000
        style = ROLES[role]
        ax.errorbar(
            position,
            value,
            yerr=[[value - lo], [hi - value]],
            color=style.color,
            marker=style.marker,
            markersize=10,
            markeredgewidth=2,
            linestyle="none",
            capsize=5,
            elinewidth=2,
        )
    ax.set_xticks(range(3), [label for _key, label, _role in SCORES])
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Spearman correlation\nwith refusal change")
    style_axis(ax, grid_axis="y")
    panel_header(ax, "", "LAYER 19 · 124 PAIRS", "Change magnitude", kicker_y=1.21)
    saved = save_c2a_figure(
        fig,
        output / "refusal_magnitude_comparison",
        title="Change magnitude and refusal behavior",
        subject="Fixed-score Spearman correlations with absolute refusal-rate change",
        creator="scripts/issue2569_refusal_magnitude_figure.py",
        include_width=include_width,
    )
    plt.close(fig)
    metadata = {
        **saved["record"],
        "caption": (
            "Mapped-change magnitude does not improve refusal-change prediction. "
            "Spearman correlation between each vector-change norm and absolute refusal-rate "
            "change. Error bars: paired 95% family-cluster bootstrap intervals, 2,000 draws; "
            "124 pairs, 21 clusters, with XSTest treated as one corpus. "
            "Qwen2.5-7B-Instruct, layer 19. Observed-answer magnitude is an empirical "
            "reference, not a mathematical ceiling."
        ),
        "values": primary,
        "inputs_sha256": {str(summary_path.relative_to(REPO)): sha256(summary_path)},
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(REPO)): sha256(Path(__file__)),
            "src/explore_persona_space/analysis/c2a_plot_style.py": sha256(
                REPO / "src/explore_persona_space/analysis/c2a_plot_style.py"
            ),
        },
        "outputs_sha256": {
            saved[key].name: sha256(saved[key]) for key in ("pdf", "png", "grayscale")
        },
    }
    (output / "refusal_magnitude_comparison.meta.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
