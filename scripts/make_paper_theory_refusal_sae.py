#!/usr/bin/env python3
"""Render the refusal-direction SAE interpretation figure for the paper.

This is a plot-only script. It reads the frozen layer-19 decomposition from
issue 2569 and writes vector PDF, color PNG, grayscale PNG, and a provenance
sidecar. It does not run model inference or recompute the decomposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Set repository thread caps before importing matplotlib/numpy.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    MUTED,
    PAPER,
    ROLES,
    STYLE_VERSION,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)


DEFAULT_SOURCE = ROOT / "eval_results/issue_2569/weights/leg9/refusal_kernel_L19.json"
DEFAULT_OUT = ROOT / "figures/paper"
DEFAULT_STEM = "c3_refusal_kernel_sae"

# Short descriptions written from the banked maximum-activation examples.
# They are qualitative summaries, not labels emitted by the SAE or a classifier.
FEATURE_DESCRIPTIONS = {
    "read": {
        1204: "Theft, explosives, and drugs",
        1623: "Abusive or sexually violent content",
        431: "Identity-targeted toxic requests",
        763: "Fraud and sexual violence",
        1815: "Explosives, drugs, and animal harm",
    },
    "kernel": {
        502: "Moral-boundary jailbreaks",
        478: "Explicit sexual framing",
        1010: "Hypnosis and coercive role-play",
        1318: "Identity-conditioned toxic persona",
        1985: "Sexualized narrative framing",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _rows(source: dict, component: str) -> list[dict[str, int | float | str]]:
    artifact_key = "range_part" if component == "read" else "kernel_part"
    records = source["decomposition"]["mean_flip_delta"][artifact_key]["ctx_sae_top5"]
    descriptions = FEATURE_DESCRIPTIONS[component]
    ids = [int(record["feat_id"]) for record in records]
    if ids != list(descriptions):
        raise ValueError(f"unexpected {component} feature IDs {ids}; expected {list(descriptions)}")
    return [
        {
            "feature_id": feature_id,
            "cosine": float(record["cos"]),
            "description": descriptions[feature_id],
        }
        for feature_id, record in zip(ids, records, strict=True)
    ]


def load_displayed_data(source_path: Path) -> dict:
    source = json.loads(source_path.read_text())
    if int(source["layer"]) != 19:
        raise ValueError(f"expected layer 19 artifact, got {source['layer']}")
    n_flip = int(source["svmp"]["n_pairs"]["flip"])
    if n_flip != 60:
        raise ValueError(f"expected 60 primary refusal flips, got {n_flip}")
    return {
        "layer": 19,
        "n_primary_refusal_flips": n_flip,
        "components": {
            "read_by_map": _rows(source, "read"),
            "effective_kernel": _rows(source, "kernel"),
        },
        "description_source": (
            "Analyst summaries of the banked maximum-activation examples; "
            "not an independent feature classifier."
        ),
    }


def make_figure(data: dict) -> tuple[plt.Figure, float]:
    set_c2a_style()
    fig, include_frac = c2a_figure("full", aspect=0.64)
    grid = fig.add_gridspec(
        2,
        1,
        left=0.335,
        right=0.985,
        top=0.86,
        bottom=0.11,
        hspace=0.72,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    specifications = [
        (
            axes[0],
            "A",
            "READ BY THE MAP",
            data["components"]["read_by_map"],
            ROLES["linear"].color,
            None,
        ),
        (
            axes[1],
            "B",
            "EFFECTIVE KERNEL",
            data["components"]["effective_kernel"],
            MUTED,
            "////",
        ),
    ]

    for ax, letter, title, rows, color, hatch in specifications:
        values = np.asarray([row["cosine"] for row in rows])
        y = np.arange(len(rows))
        bars = ax.barh(
            y,
            values,
            height=0.58,
            color=color if hatch is None else PAPER,
            edgecolor=color,
            linewidth=1.4,
            hatch=hatch,
        )
        ax.set_yticks(
            y,
            [f"SAE {row['feature_id']}\n{row['description']}" for row in rows],
        )
        ax.invert_yaxis()
        ax.set_xlim(0.0, 0.46)
        ax.set_xticks([0.0, 0.15, 0.30, 0.45])
        ax.set_xlabel("Cosine similarity to component")
        style_axis(ax, grid_axis="x")
        panel_header(
            ax,
            letter,
            "MEAN REFUSAL-FLIP DIRECTION",
            title.title(),
            kicker_y=1.18,
            title_y=1.07,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + 0.009,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                ha="left",
                va="center",
                color=color,
                fontsize=16,
                fontweight=650,
            )

    return fig, include_frac


def write_metadata(
    *,
    source: Path,
    stem: Path,
    outputs: dict,
    displayed_data: dict,
) -> Path:
    metadata = {
        "status": "Theoretical-analysis manuscript figure",
        "style_version": STYLE_VERSION,
        "plotting_script": "scripts/make_paper_theory_refusal_sae.py",
        "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
        "reproduction_command": "uv run python scripts/make_paper_theory_refusal_sae.py",
        "source": {"path": _display_path(source), "sha256": _sha256(source)},
        "render": outputs["record"],
        "displayed_data": displayed_data,
        "output_sha256": {
            kind: _sha256(path) for kind, path in outputs.items() if isinstance(path, Path)
        },
    }
    metadata.update(
        as_metadata_dict(
            git_provenance(cwd=ROOT, argv0=str(Path(__file__).resolve())),
            phase="paper-theory-refusal-sae",
        )
    )
    metadata_path = stem.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    displayed_data = load_displayed_data(args.source)
    figure, include_frac = make_figure(displayed_data)
    stem = args.out_dir / args.stem
    outputs = save_c2a_figure(
        figure,
        stem,
        title="SAE interpretation of the mean refusal-flip direction",
        subject=(
            "Nearest context-side SAE features for the component read by the map "
            "and the component in its effective kernel"
        ),
        creator="scripts/make_paper_theory_refusal_sae.py",
        include_width=include_frac,
    )
    plt.close(figure)
    metadata = write_metadata(
        source=args.source,
        stem=stem,
        outputs=outputs,
        displayed_data=displayed_data,
    )
    for path in (outputs["pdf"], outputs["png"], outputs["grayscale"], metadata):
        print(path)


if __name__ == "__main__":
    main()
