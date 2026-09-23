"""Render the descriptive marginal-label comparison from the saved reanalysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

LABELS = {
    "Interpretable (autointerp)": "Interpretable",
    "Abstraction: token surface": "Token surface",
    "Abstraction: lexical semantic": "Lexical semantic",
    "Abstraction: abstract contextual": "Abstract contextual",
    "Content type: topic": "Topic",
    "Content type: task format": "Task format",
    "Content type: entity": "Entity",
    "Content type: syntax": "Syntax",
    "Content type: operation": "Operation",
    "Speaker: language of the text": "Speaker: language",
    "Speaker: identity / disposition": "Speaker: identity / disposition",
    "Speaker: register / style": "Speaker: register / style",
    "Judged role: input-side  [k=0.31]": "Role: input-side",
    "Judged role: output-promoting  [k=0.31]": "Role: output-promoting",
    "Judged role: mixed  [k=0.31]": "Role: mixed",
}


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    """Export color, vector and grayscale views with exact plotted-value metadata."""
    source = Path(cfg.source).resolve()
    stem = Path(cfg.stem).resolve()
    all_rows = json.loads(source.read_text())["properties"]
    rows = {row["name"]: row for row in all_rows}
    names = list(LABELS)
    old = [rows[n]["published_reproduction_marginal"] for n in names]
    new = [rows[n]["full_labels_common_resolved_marginal"] for n in names]
    yy = np.arange(len(names))
    set_c2a_style()
    fig, frac = c2a_figure("full", 0.86)
    ax = fig.add_subplot(111)
    style_axis(ax, grid_axis="x")
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.hlines(yy, np.minimum(old, new), np.maximum(old, new), color=GRID, linewidth=2)
    ax.scatter(
        old,
        yy,
        marker="s",
        facecolors="none",
        edgecolors=ROLES["control"].color,
        s=85,
        linewidths=1.7,
        label="Published",
    )
    ax.scatter(
        new,
        yy,
        marker=ROLES["linear"].marker,
        color=ROLES["linear"].color,
        s=62,
        label="Full labels, resolved",
    )
    ax.set_yticks(yy, [LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(-0.20, 0.35)
    ax.set_xticks(np.arange(-0.2, 0.31, 0.1))
    ax.set_xlabel("Concordance with direction $R^2$ − ½")
    fig.suptitle("SAE label associations", x=0.40, y=0.99, ha="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False, ncol=2)
    fig.subplots_adjust(left=0.40, right=0.97, top=0.88, bottom=0.10)
    output = save_c2a_figure(
        fig,
        stem,
        title="SAE label associations",
        subject="Published and corrected marginal concordance using the same decoder-direction R2",
        creator=Path(__file__).name,
        include_width=frac,
    )
    meta = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "render": output["record"],
        "data": [
            {"name": n, "published": o, "corrected": c}
            for n, o, c in zip(names, old, new, strict=True)
        ],
        "caption": (
            "Marginal concordance minus one half, with no matching. Published values use "
            "120,716 features and the truncated label matrix. Corrected values use the full "
            "saved labels and the 105,714 features resolved on all five axes. Therefore the "
            "difference combines label restoration and population restriction. These are "
            "descriptive point estimates without uncertainty intervals. Same existing "
            "layer-19 Qwen2.5-7B-Instruct decoder-direction scores; no model or judge calls."
        ),
        "outputs": {
            k: {
                "path": str(output[k]),
                "sha256": hashlib.sha256(output[k].read_bytes()).hexdigest(),
            }
            for k in ("pdf", "png", "grayscale")
        },
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
