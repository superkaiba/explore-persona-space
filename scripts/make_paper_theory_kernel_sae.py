#!/usr/bin/env python3
"""Render representative SAE features in the read and effective-kernel subspaces.

This plot-only script reads the frozen layer-19 kernel interpretation from issue
2569. It does not run model inference or recompute the decomposition.
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

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
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


DEFAULT_SOURCE = (
    ROOT / "eval_results/issue_2569/weights/leg8/kernel_interpretation_L19.json"
)
DEFAULT_OUT = ROOT / "figures/paper"
DEFAULT_STEM = "c3_effective_kernel_sae"

# Representatives from the 20 most extreme features on each side. The read-side
# set uses Latin-script examples so the paper render remains portable; the
# kernel-side set spans recurring patterns rather than repeating near-duplicates.
FEATURES = {
    "read": [
        (855, "Swedish-language context", "Hej! Kan du svenska?"),
        (267, "Vietnamese-language context", "giúp tôi"),
        (137, "Hungarian-language context", "szia, ki vagy?"),
        (454, "Polish-language context", "witaj, czy możemy porozmawiać?"),
        (7412, "Dutch-language request", "answer prompt in the dutch language"),
    ],
    "kernel": [
        (1197, "Polite help request", "Can you help me make a python script?"),
        (1051, "Explain-it-simply request", "explain them like I'm a 5 years old"),
        (755, "Scripted refusal opener", "start your reply with the following line"),
        (1961, "SEO and metadata block", "Headline: NAME_1 #02 (Preview)"),
        (1386, "Long word-count demand", "Write a 5000 word essay on Brexit"),
    ],
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


def _records_by_id(source: dict, key: str) -> dict[int, dict]:
    records = source["ctx_sae_naming"][key]
    return {int(record["feat_id"]): record for record in records}


def _rows(source: dict, component: str) -> list[dict]:
    key = "top_used" if component == "read" else "top_ignored"
    records = _records_by_id(source, key)
    rows = []
    for feature_id, description, excerpt in FEATURES[component]:
        if feature_id not in records:
            raise ValueError(f"SAE {feature_id} is absent from ctx_sae_naming.{key}")
        record = records[feature_id]
        source_quote = str(record["top_contexts"][0]["quote"])
        if excerpt.casefold() not in source_quote.casefold():
            raise ValueError(
                f"display excerpt for SAE {feature_id} is not in its top activating context"
            )
        kernel_share = float(record["kernel_share"])
        rows.append(
            {
                "feature_id": feature_id,
                "description": description,
                "maximum_activation_example": excerpt,
                "kernel_share": kernel_share,
                "named_subspace_share": 1.0 - kernel_share if component == "read" else kernel_share,
            }
        )
    return rows


def load_displayed_data(source_path: Path) -> dict:
    source = json.loads(source_path.read_text())
    cutoff = source["conventions"]["cutoffs"]["primary"]
    if not np.isclose(float(cutoff), 0.99):
        raise ValueError(f"expected a 0.99 squared-gain cutoff, got {cutoff}")
    null = source["null"]["0.99"]
    return {
        "layer": 19,
        "cutoff": 0.99,
        "components": {
            "read_by_map": _rows(source, "read"),
            "effective_kernel": _rows(source, "kernel"),
        },
        "random_direction_bands": {
            "read_by_map": [1.0 - float(null["p97p5"]), 1.0 - float(null["p2p5"])],
            "effective_kernel": [float(null["p2p5"]), float(null["p97p5"])],
        },
        "selection_note": (
            "Five representatives from each top-20 extreme list. Read-side examples "
            "use Latin scripts; kernel-side examples span recurring patterns."
        ),
        "description_source": (
            "Analyst summaries of maximum-activation examples; not an independent classifier."
        ),
    }


def make_figure(data: dict) -> tuple[plt.Figure, float]:
    set_c2a_style()
    fig, include_frac = c2a_figure("full", aspect=0.72)
    grid = fig.add_gridspec(
        2,
        1,
        left=0.39,
        right=0.985,
        top=0.90,
        bottom=0.09,
        hspace=0.65,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    specifications = [
        (
            axes[0],
            "A",
            "Read strongly by the map",
            data["components"]["read_by_map"],
            data["random_direction_bands"]["read_by_map"],
            ROLES["linear"].color,
            None,
        ),
        (
            axes[1],
            "B",
            "Effective kernel",
            data["components"]["effective_kernel"],
            data["random_direction_bands"]["effective_kernel"],
            MUTED,
            "////",
        ),
    ]

    for ax, letter, title, rows, null_band, color, hatch in specifications:
        values = np.asarray([row["named_subspace_share"] for row in rows])
        y = np.arange(len(rows))
        ax.axvspan(*null_band, color=GRID, alpha=0.42, linewidth=0)
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
            [
                f"SAE {row['feature_id']} · {row['description']}\n“{row['maximum_activation_example']}”"
                for row in rows
            ],
        )
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.02)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.set_xlabel("Squared share in named subspace")
        style_axis(ax, grid_axis="x")
        panel_header(
            ax,
            letter,
            "REPRESENTATIVE EXTREME SAE FEATURES",
            title,
            kicker_y=1.16,
            title_y=1.05,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                min(value + 0.018, 0.985),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                ha="left" if value < 0.95 else "right",
                va="center",
                color=color,
                fontsize=16,
                fontweight=650,
            )
    return fig, include_frac


def write_metadata(
    *, source: Path, stem: Path, outputs: dict, displayed_data: dict, provenance
) -> Path:
    metadata = {
        "status": "Theoretical-analysis manuscript figure",
        "style_version": STYLE_VERSION,
        "plotting_script": "scripts/make_paper_theory_kernel_sae.py",
        "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
        "reproduction_command": "uv run python scripts/make_paper_theory_kernel_sae.py",
        "source": {"path": _display_path(source), "sha256": _sha256(source)},
        "render": outputs["record"],
        "displayed_data": displayed_data,
        "output_sha256": {
            kind: _sha256(path) for kind, path in outputs.items() if isinstance(path, Path)
        },
    }
    metadata.update(as_metadata_dict(provenance, phase="paper-theory-kernel-sae"))
    metadata_path = stem.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
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
    provenance = git_provenance(cwd=ROOT, argv0=str(Path(__file__).resolve()))
    outputs = save_c2a_figure(
        figure,
        stem,
        title="SAE interpretation of the map's read and effective-kernel subspaces",
        subject=(
            "Representative extreme context-side SAE features and their "
            "maximum-activation examples"
        ),
        creator="scripts/make_paper_theory_kernel_sae.py",
        include_width=include_frac,
    )
    plt.close(figure)
    metadata = write_metadata(
        source=args.source,
        stem=stem,
        outputs=outputs,
        displayed_data=displayed_data,
        provenance=provenance,
    )
    for path in (outputs["pdf"], outputs["png"], outputs["grayscale"], metadata):
        print(path)


if __name__ == "__main__":
    main()
