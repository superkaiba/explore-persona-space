"""Phase 0: Analyze A1 raw_completions to build cross-persona marker rate matrix.

Loads all 10 marker_*_asst_excluded_medium_seed42/raw_completions.json files,
recomputes marker rates from raw completions (NOT marker_eval.json), builds a
10x11 matrix (10 source models x 11 eval personas), and generates a heatmap.

Saves:
  - eval_results/dissociation_i138/phase0_analysis.json
  - figures/dissociation_i138/phase0_heatmap.png/.pdf

Issue: #138 (Persona-Marker Dissociation via Prefix Completion)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.personas import PERSONAS, SHORT_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 10 source personas (no assistant — those are eval-only)
SOURCE_PERSONAS = list(PERSONAS.keys())

# 11 eval personas (10 + assistant)
EVAL_PERSONAS = SOURCE_PERSONAS + ["assistant"]

# Input data lives in the main repo (not the worktree). raw_completions.json
# files are NOT tracked in git — they exist only where they were generated.
# On pods, PROJECT_ROOT is /workspace/explore-persona-space (the main repo).
# Locally, worktrees won't have the raw files; fall back to the main repo.
_MAIN_REPO = PROJECT_ROOT
_TEST_FILE = (
    "eval_results/leakage_experiment/"
    "marker_villain_asst_excluded_medium_seed42/raw_completions.json"
)
if not (PROJECT_ROOT / _TEST_FILE).exists():
    _fallback = Path("/home/thomasjiralerspong/explore-persona-space")
    if (_fallback / _TEST_FILE).exists():
        _MAIN_REPO = _fallback

EVAL_RESULTS_DIR = _MAIN_REPO / "eval_results" / "leakage_experiment"
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "dissociation_i138"
FIGURES_DIR = PROJECT_ROOT / "figures" / "dissociation_i138"


def load_raw_completions(source_persona: str) -> dict[str, dict[str, list[str]]]:
    """Load raw_completions.json for a source model."""
    path = (
        EVAL_RESULTS_DIR
        / f"marker_{source_persona}_asst_excluded_medium_seed42"
        / "raw_completions.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing raw completions: {path}")
    with open(path) as f:
        return json.load(f)


def compute_marker_rate(completions: dict[str, list[str]], marker: str = "[ZLT]") -> dict:
    """Compute marker rate from a {question: [completions]} dict.

    Returns {rate, found, total}.
    """
    marker_lower = marker.lower()
    found = 0
    total = 0
    for _question, comps in completions.items():
        for c in comps:
            total += 1
            if marker_lower in c.lower():
                found += 1
    rate = found / total if total > 0 else 0.0
    return {"rate": rate, "found": found, "total": total}


def build_matrix() -> tuple[dict, np.ndarray]:
    """Build the 10x11 marker rate matrix.

    Returns:
        (full_results_dict, rate_matrix_10x11)
    """
    results = {}
    matrix = np.zeros((len(SOURCE_PERSONAS), len(EVAL_PERSONAS)))

    for i, source in enumerate(SOURCE_PERSONAS):
        logger.info("Processing source model: %s", source)
        raw = load_raw_completions(source)

        source_results = {}
        for j, eval_persona in enumerate(EVAL_PERSONAS):
            if eval_persona not in raw:
                logger.warning(
                    "Eval persona %s not found in %s raw_completions", eval_persona, source
                )
                source_results[eval_persona] = {"rate": 0.0, "found": 0, "total": 0}
                continue

            stats = compute_marker_rate(raw[eval_persona])
            source_results[eval_persona] = stats
            matrix[i, j] = stats["rate"]

        results[source] = source_results

    return results, matrix


def generate_heatmap(matrix: np.ndarray) -> plt.Figure:
    """Generate a paper-quality 10x11 heatmap of marker rates."""
    set_paper_style("neurips", font_scale=0.9)

    # Use short names for display
    row_labels = [SHORT_NAMES.get(p, p) for p in SOURCE_PERSONAS]
    col_labels = [SHORT_NAMES.get(p, p) for p in EVAL_PERSONAS]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Use a sequential colormap that works for colorblind
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=0.7, aspect="auto")

    # Set tick labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Add text annotations in cells
    for i in range(len(SOURCE_PERSONAS)):
        for j in range(len(EVAL_PERSONAS)):
            val = matrix[i, j]
            # Use white text on dark cells, black on light
            color = "white" if val > 0.35 else "black"
            ax.text(
                j,
                i,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
                fontweight="bold" if i == j else "normal",
            )

    # Highlight diagonal (source model == eval persona)
    for i in range(min(len(SOURCE_PERSONAS), len(EVAL_PERSONAS))):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#0072B2",
                linewidth=2,
            )
        )

    ax.set_xlabel("Evaluation Persona (system prompt)")
    ax.set_ylabel("Source Model (trained on this persona)")
    ax.set_title("[ZLT] Marker Rate: Source Model x Eval Persona (A1 Baselines)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("[ZLT] Rate")

    fig.tight_layout()
    return fig


def main():
    logger.info("Phase 0: Building A1 cross-persona marker rate matrix")

    # Build matrix
    results, matrix = build_matrix()

    # Print summary
    logger.info("\n=== Marker Rate Matrix (source model rows x eval persona columns) ===")
    header = "".join(f"{SHORT_NAMES.get(p, p):>10}" for p in EVAL_PERSONAS)
    logger.info("                %s", header)
    for i, source in enumerate(SOURCE_PERSONAS):
        row = "".join(f"{matrix[i, j]:>10.1%}" for j in range(len(EVAL_PERSONAS)))
        logger.info("  %15s %s", SHORT_NAMES.get(source, source), row)

    # Diagonal = source rate
    logger.info("\n=== Source Rates (diagonal) ===")
    for i, source in enumerate(SOURCE_PERSONAS):
        logger.info("  %s: %.1f%% (from raw_completions)", source, matrix[i, i] * 100)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "Phase 0: A1 cross-persona marker rates from raw_completions.json",
        "issue": 138,
        "source_personas": SOURCE_PERSONAS,
        "eval_personas": EVAL_PERSONAS,
        "matrix": matrix.tolist(),
        "per_model": results,
    }
    output_path = OUTPUT_DIR / "phase0_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved results to %s", output_path)

    # Generate heatmap
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = generate_heatmap(matrix)
    paths = savefig_paper(fig, "phase0_heatmap", dir=str(FIGURES_DIR))
    logger.info("Saved heatmap to %s", paths)
    plt.close(fig)

    logger.info("Phase 0 complete.")


if __name__ == "__main__":
    main()
