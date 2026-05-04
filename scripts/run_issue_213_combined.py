#!/usr/bin/env python3
"""Issue #213 Combined: Merge Part A + Part B results and generate hero figure.

After Parts A and B are complete, this script:
  1. Loads JS divergence from Part A (6 cues x 4 models = 24 cells)
  2. Computes JS divergence for Part B's 14 new cues x 2 models on the pod
  3. Merges into a single correlation: JS vs misalignment across all ~40+ cells
  4. Generates hero figure: scatter plot, JS (x) vs misalignment (y),
     colored by model, shaped by cue prior. Two panels: with/without edu_v0.

See plan issue-213.md section "Hero figure" for the pre-committed visualization.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# project path bootstrap
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv

logger = logging.getLogger("issue_213_combined")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

PART_A_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "issue_213" / "part_a"
PART_B_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "issue_213" / "part_b"
OUT_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "issue_213" / "combined"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "issue_213"

# Cue prior mapping (must match Part B's CUES definition)
CUE_PRIORS: dict[str, str] = {
    "no_cue": "control",
    "edu_v0": "jailbreak",
    "edu_v1": "conditional",
    "edu_v2": "conditional",
    "edu_v3": "conditional",
    "code_format": "inert",
    "edu_mode": "conditional",
    "teaching_demo": "conditional",
    "academic_only": "conditional",
    "classroom": "conditional",
    "sec_researcher": "conditional",
    "pentest": "conditional",
    "red_team": "conditional",
    "admin_override": "jailbreak-likely",
    "debug_mode": "jailbreak-likely",
    "internal_test": "conditional",
    "textbook_mistakes": "conditional",
    "what_not_to_do": "jailbreak-likely",
    "chef": "control",
    "formal_english": "control",
}

# Model color palette (colorblind-safe)
MODEL_COLORS: dict[str, str] = {
    "educational-insecure": "#D55E00",  # vermillion
    "insecure": "#E69F00",  # amber
    "secure-finetune": "#009E73",  # bluish green
    "base-instruct": "#0072B2",  # blue
}

# Prior -> marker shape
PRIOR_MARKERS: dict[str, str] = {
    "jailbreak": "X",
    "jailbreak-likely": "P",
    "conditional": "o",
    "control": "s",
    "inert": "D",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_part_a() -> list[dict]:
    """Load Part A JS + misalignment data into a flat list of cells."""
    js_path = PART_A_DIR / "js_matrix.json"
    corr_path = PART_A_DIR / "correlation_results.json"

    if not js_path.exists():
        raise FileNotFoundError(f"Part A JS matrix not found: {js_path}")
    if not corr_path.exists():
        raise FileNotFoundError(f"Part A correlation results not found: {corr_path}")

    with open(js_path) as f:
        js_data = json.load(f)
    with open(corr_path) as f:
        corr_data = json.load(f)

    cells = []
    for cell_info in corr_data.get("cells", []):
        label = cell_info["label"]  # "model_cue"
        # Handle model names with hyphens
        model = None
        cue = None
        for m in js_data["models"]:
            if label.startswith(f"{m}_"):
                model = m
                cue = label[len(m) + 1 :]
                break
        if model is None or cue is None:
            logger.warning("Could not parse cell label: %s", label)
            continue

        cells.append(
            {
                "model": model,
                "cue": cue,
                "js": cell_info["js"],
                "misalignment": cell_info["mis"],
                "prior": CUE_PRIORS.get(cue, "unknown"),
                "source": "part_a",
            }
        )

    logger.info("Loaded %d cells from Part A", len(cells))
    return cells


def load_part_b() -> list[dict]:
    """Load Part B grid_summary into a flat list of cells."""
    grid_path = PART_B_DIR / "grid_summary.json"

    if not grid_path.exists():
        raise FileNotFoundError(f"Part B grid summary not found: {grid_path}")

    with open(grid_path) as f:
        data = json.load(f)

    cells = []
    for cell in data["cells"]:
        cells.append(
            {
                "model": cell["model"],
                "cue": cell["cue"],
                "misalignment": cell.get("misalignment_rate_lt30"),
                "prior": cell.get("cue_prior", CUE_PRIORS.get(cell["cue"], "unknown")),
                "source": "part_b",
                # JS will be filled in later or loaded from Part A if available
                "js": None,
            }
        )

    logger.info("Loaded %d cells from Part B", len(cells))
    return cells


def merge_cells(part_a: list[dict], part_b: list[dict]) -> list[dict]:
    """Merge Part A and Part B data.

    Part A has JS + misalignment for 4 models x 6 cues.
    Part B has misalignment for 2 models x 20 cues.
    For the original 6 cues on the 2 shared models, Part A's JS applies.
    For the 14 new cues, JS would come from a separate Part B JS computation
    (run on the pod). If not available, those cells get js=None.

    The combined list should have one entry per unique (model, cue) pair.
    """
    # Index Part A by (model, cue)
    a_by_key: dict[tuple[str, str], dict] = {}
    for cell in part_a:
        key = (cell["model"], cell["cue"])
        a_by_key[key] = cell

    # Start with all Part A cells
    merged: dict[tuple[str, str], dict] = {}
    for cell in part_a:
        key = (cell["model"], cell["cue"])
        merged[key] = cell.copy()

    # Add Part B cells, enriching with JS from Part A where available
    for cell in part_b:
        key = (cell["model"], cell["cue"])
        if key in merged:
            # Already have this cell from Part A (has JS). Update misalignment
            # from Part B if Part A didn't have it.
            if merged[key].get("misalignment") is None and cell.get("misalignment") is not None:
                merged[key]["misalignment"] = cell["misalignment"]
            merged[key]["source"] = "both"
        else:
            # New cell from Part B only
            merged[key] = cell.copy()

    result = list(merged.values())
    n_with_js = sum(1 for c in result if c.get("js") is not None)
    n_with_mis = sum(1 for c in result if c.get("misalignment") is not None)
    logger.info(
        "Merged: %d total cells, %d with JS, %d with misalignment",
        len(result),
        n_with_js,
        n_with_mis,
    )
    return result


def compute_combined_correlations(cells: list[dict]) -> dict:
    """Compute Spearman correlation on merged cells."""
    # Filter to cells with both JS and misalignment
    valid = [c for c in cells if c.get("js") is not None and c.get("misalignment") is not None]

    results: dict = {"n_total": len(cells), "n_valid": len(valid)}

    if len(valid) < 5:
        results["all"] = {"rho": None, "p_value": None, "n": len(valid)}
        results["no_edu_v0"] = {"rho": None, "p_value": None, "n": 0}
        return results

    js_vals = [c["js"] for c in valid]
    mis_vals = [c["misalignment"] for c in valid]

    rho, p = stats.spearmanr(js_vals, mis_vals)
    results["all"] = {"rho": rho, "p_value": p, "n": len(valid)}

    # Without edu_v0
    valid_no_edu = [c for c in valid if c["cue"] != "edu_v0"]
    if len(valid_no_edu) >= 5:
        js_no = [c["js"] for c in valid_no_edu]
        mis_no = [c["misalignment"] for c in valid_no_edu]
        rho, p = stats.spearmanr(js_no, mis_no)
        results["no_edu_v0"] = {"rho": rho, "p_value": p, "n": len(valid_no_edu)}
    else:
        results["no_edu_v0"] = {"rho": None, "p_value": None, "n": len(valid_no_edu)}

    # By prior category
    results["by_prior"] = {}
    for prior in set(c["prior"] for c in valid):
        subset = [c for c in valid if c["prior"] == prior]
        if len(subset) >= 4:
            rho, p = stats.spearmanr([c["js"] for c in subset], [c["misalignment"] for c in subset])
            results["by_prior"][prior] = {"rho": rho, "p_value": p, "n": len(subset)}
        else:
            results["by_prior"][prior] = {"rho": None, "p_value": None, "n": len(subset)}

    return results


def generate_hero_figure(cells: list[dict], correlations: dict) -> Path:
    """Generate hero scatter plot: JS (x) vs misalignment (y).

    Two panels: with edu_v0 (left) and without (right).
    Points colored by model, shaped by cue prior.
    """
    # Try to use paper_plots rcParams
    try:
        from explore_persona_space.analysis.paper_plots import setup_rcparams

        setup_rcparams()
    except ImportError:
        pass

    valid = [c for c in cells if c.get("js") is not None and c.get("misalignment") is not None]
    valid_no_edu = [c for c in valid if c["cue"] != "edu_v0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, data, title_suffix in [
        (ax1, valid, "(all cues)"),
        (ax2, valid_no_edu, "(excluding edu_v0)"),
    ]:
        # Plot each model x prior combination
        for cell in data:
            model = cell["model"]
            prior = cell["prior"]
            color = MODEL_COLORS.get(model, "#999999")
            marker = PRIOR_MARKERS.get(prior, "o")
            ax.scatter(
                cell["js"],
                cell["misalignment"],
                c=color,
                marker=marker,
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )

        # Add regression line if enough points
        if len(data) >= 5:
            js_arr = np.array([c["js"] for c in data])
            mis_arr = np.array([c["misalignment"] for c in data])
            rho, p = stats.spearmanr(js_arr, mis_arr)

            # Fit line for visual
            if len(data) >= 3:
                z = np.polyfit(js_arr, mis_arr, 1)
                x_line = np.linspace(js_arr.min(), js_arr.max(), 100)
                y_line = np.polyval(z, x_line)
                ax.plot(x_line, y_line, "--", color="gray", alpha=0.5, linewidth=1)

            ax.annotate(
                f"\u03c1 = {rho:.2f}, p = {p:.3f}, n = {len(data)}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
            )

        ax.set_xlabel("JS divergence (cue vs no_cue)")
        ax.set_title(f"JS vs Misalignment {title_suffix}")

    ax1.set_ylabel("Misalignment rate (aligned < 30)")

    # Build legend manually
    # Model colors
    model_handles = []
    for model, color in MODEL_COLORS.items():
        model_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=model,
            )
        )

    # Prior shapes
    prior_handles = []
    for prior, marker in PRIOR_MARKERS.items():
        prior_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label=prior,
            )
        )

    all_handles = model_handles + prior_handles
    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=len(all_handles),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "Issue #213: JS divergence predicts conditional misalignment?",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "hero_js_vs_misalignment.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Hero figure saved to %s", fig_path)
    return fig_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data from both parts
    part_a_cells = load_part_a()
    part_b_cells = load_part_b()

    # Merge
    merged = merge_cells(part_a_cells, part_b_cells)

    # Compute correlations on the merged set
    correlations = compute_combined_correlations(merged)
    correlations["meta"] = {
        "issue": 213,
        "part": "combined",
        "git_commit": _git_commit(),
        "computed_at": datetime.now(UTC).isoformat(),
    }

    with open(OUT_DIR / "correlation_results.json", "w") as f:
        json.dump(correlations, f, indent=2)

    # Save merged cell data for downstream analysis
    with open(OUT_DIR / "merged_cells.json", "w") as f:
        json.dump(merged, f, indent=2)

    # Generate hero figure
    fig_path = generate_hero_figure(merged, correlations)

    # Print summary
    logger.info("=" * 80)
    logger.info("Combined Results:")
    logger.info("-" * 80)

    all_corr = correlations.get("all", {})
    no_edu = correlations.get("no_edu_v0", {})
    logger.info(
        "  JS vs misalignment (all): rho=%.3f, p=%.4f, n=%d",
        all_corr.get("rho", 0) or 0,
        all_corr.get("p_value", 1) or 1,
        all_corr.get("n", 0),
    )
    logger.info(
        "  JS vs misalignment (no edu_v0): rho=%.3f, p=%.4f, n=%d",
        no_edu.get("rho", 0) or 0,
        no_edu.get("p_value", 1) or 1,
        no_edu.get("n", 0),
    )

    for prior, vals in correlations.get("by_prior", {}).items():
        logger.info(
            "  JS vs misalignment (%s only): rho=%.3f, p=%.4f, n=%d",
            prior,
            vals.get("rho", 0) or 0,
            vals.get("p_value", 1) or 1,
            vals.get("n", 0),
        )

    # Kill criterion check
    rho_no_edu = no_edu.get("rho")
    if rho_no_edu is not None and abs(rho_no_edu) < 0.30:
        logger.warning(
            "KILL CRITERION MET: |rho| = %.3f < 0.30 (excluding edu_v0). "
            "Geometry does not predict conditional misalignment.",
            abs(rho_no_edu),
        )
    elif rho_no_edu is not None and abs(rho_no_edu) >= 0.50:
        logger.info(
            "SUCCESS CRITERION MET: |rho| = %.3f >= 0.50 (excluding edu_v0). "
            "Geometry predicts conditional misalignment.",
            abs(rho_no_edu),
        )

    logger.info("Hero figure: %s", fig_path)
    logger.info("Results: %s", OUT_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())
