"""Per-persona bystander-only slope figure for task #550 finding 3.

Reads the shift JSONs from the three dial points (#527 shallow, #550 mid,
#538 deep) and computes, for each held-out bystander persona, the
nat-per-nat slope of its log-prob shift across the dial. Bystander-only
means: for each persona, only include cells where it was NOT a trained
source of that cell. (A persona reads as a true bystander on every cell of
the OTHER pair, and on no cell of its OWN pair.)

For pair 1 (florist x medical_doctor), sources are {florist, medical_doctor};
for pair 2 (librarian x police_officer), sources are {librarian,
police_officer}. So:

  - florist / medical_doctor: bystanders on the 9 librarian x police_officer
    joint cells (3 seeds x 3 dials).
  - librarian / police_officer: bystanders on the 9 florist x medical_doctor
    joint cells (3 seeds x 3 dials).
  - All other 15 personas: bystanders on all 18 joint cells (6 cells x 3
    dials), since the eval panel is the same fixed 19-persona set across
    every cell.

Per-persona slope: median of the persona's delta_logp_marker across its
true-bystander joint cells, computed at each of the three dials; slope =
linear fit (median vs realized median dial landing) over the three points.

Output: dot plot with one row per persona, sorted by slope (high to low),
color-coded by whether the persona is source-adjacent (i.e., closest to one
of the four source-persona semantic neighborhoods: medical / law-enforcement
or near them) or semantically distant.

Saves to figures/issue_550/bystander_slope_by_persona.{png,pdf,meta.json}
via savefig_paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

# Source personas per pair (the implant was trained on these two for each pair).
PAIR_SOURCES = {
    "florist__medical_doctor": ("florist", "medical_doctor"),
    "librarian__police_officer": ("librarian", "police_officer"),
}

# Source-adjacent personas — those in the same semantic neighborhood as one of
# the four source personas (medical/health: paramedic, surgeon, navy_seal,
# army_medic, medical_doctor, police_officer, private_investigator; security:
# pentester, cybersec_consultant, navy_seal, private_investigator;
# law/library: librarian; service/skilled-trade: florist). The
# semantically-distant set is the residual.
SOURCE_ADJACENT = {
    "paramedic",
    "surgeon",
    "navy_seal",
    "army_medic",
    "pentester",
    "cybersec_consultant",
    "private_investigator",
}

DIAL_LABEL_BY_RUN = {
    "issue_527": "Shallow",
    "issue_550": "Mid",
    "issue_538": "Deep",
}

DIAL_ORDER = ["issue_527", "issue_550", "issue_538"]


def load_joint_shift_jsons(eval_dir: Path) -> list[dict]:
    """Return the six joint-arm shift JSONs from an eval/ directory."""
    rows = []
    for path in sorted(eval_dir.glob("*__joint__seed*__shift.json")):
        with path.open() as f:
            d = json.load(f)
        rows.append(d)
    return rows


def median_landing_per_run(sweep_dir: Path) -> float:
    """Read the median realized band landing across ALL 18 cells in this dial run.

    Pulls from analysis-side sweep JSONs that carry `final_source_delta_nats`.
    """
    deltas = []
    for path in sorted(sweep_dir.glob("*__seed*.json")):
        with path.open() as f:
            d = json.load(f)
        # Sweep JSONs carry the band-landing source delta.
        val = d.get("final_source_delta_nats")
        if val is None:
            continue
        deltas.append(float(val))
    if not deltas:
        raise RuntimeError(f"no final_source_delta_nats found in {sweep_dir}")
    return float(np.median(deltas))


def build_bystander_data(
    runs: list[tuple[str, Path]],
    sweep_dirs: list[Path] | None = None,
) -> tuple[dict[str, dict[str, list[float]]], list[float], list[str]]:
    """Return (per_persona_per_dial_values, dial_landings, eval_panel).

    per_persona_per_dial_values[persona][dial_key] = list of delta_logp_marker
        readings on TRUE-BYSTANDER joint cells (i.e., cells in the pair where
        the persona is NOT a source).
    dial_landings = median realized landing per dial across ALL 18 sweep
        cells (the same reference the body uses: "median realized landings
        5.89 → 9.81 → 17.24 nat"). Computed from the sweep JSONs when
        sweep_dirs is provided; falls back to source-context medians from
        the joint shift JSONs otherwise.
    eval_panel = the shared 19-persona eval panel (taken from the first shift
        JSON, since it is constant across the 18 joint cells and 3 dials).
    """
    per_persona: dict[str, dict[str, list[float]]] = {}
    dial_landings: list[float] = []
    eval_panel_ref: list[str] | None = None

    for idx, (run_key, eval_dir) in enumerate(runs):
        joints = load_joint_shift_jsons(eval_dir)
        # Prefer sweep-side landings (consistent with the body's "5.89 / 9.81 /
        # 17.24" numbers). Fall back to joint-cell source-context medians.
        if sweep_dirs is not None:
            cell_landings = []
            for path in sorted(sweep_dirs[idx].glob("*__seed*.json")):
                with path.open() as f:
                    d = json.load(f)
                val = d.get("final_source_delta_nats")
                if val is not None:
                    cell_landings.append(float(val))
            if not cell_landings:
                raise RuntimeError(f"no sweep landings for {run_key}")
            dial_landings.append(float(np.median(cell_landings)))
        else:
            cell_landings = []
            for cell in joints:
                pair_id = cell["pair_id"]
                sources = PAIR_SOURCES[pair_id]
                for src in sources:
                    ctx = cell["contexts"].get(src)
                    if ctx is None:
                        continue
                    cell_landings.append(float(ctx["delta_logp_marker"]))
            if not cell_landings:
                raise RuntimeError(f"no source-context landings for {run_key}")
            dial_landings.append(float(np.median(cell_landings)))

        # Accumulate per-persona true-bystander readings.
        for cell in joints:
            pair_id = cell["pair_id"]
            sources = set(PAIR_SOURCES[pair_id])
            if eval_panel_ref is None:
                eval_panel_ref = list(cell["eval_panel"])
            for persona, ctx in cell["contexts"].items():
                if persona in sources:
                    continue  # skip source reads — bystander-only
                per_persona.setdefault(persona, {}).setdefault(run_key, []).append(
                    float(ctx["delta_logp_marker"])
                )

    assert eval_panel_ref is not None
    return per_persona, dial_landings, eval_panel_ref


def compute_persona_slopes(
    per_persona: dict[str, dict[str, list[float]]],
    dial_landings: list[float],
) -> list[dict]:
    """Return a sorted list of {persona, slope, median_per_dial, n_per_dial}.

    Slope = (deep_median − shallow_median) / (deep_landing − shallow_landing),
    in nat-per-nat units. This matches the body's convention (line 215):
    "change in per-persona median log-prob shift across the dial divided by
    the change in median realized landing."
    """
    rows = []
    denom = dial_landings[2] - dial_landings[0]
    for persona, by_dial in per_persona.items():
        medians = []
        ns = []
        for run_key in DIAL_ORDER:
            vals = by_dial.get(run_key, [])
            if not vals:
                medians.append(np.nan)
                ns.append(0)
            else:
                medians.append(float(np.median(vals)))
                ns.append(len(vals))
        medians = np.array(medians)
        if np.isnan(medians[0]) or np.isnan(medians[2]):
            slope = np.nan
        else:
            slope = float((medians[2] - medians[0]) / denom)
        rows.append(
            {
                "persona": persona,
                "slope": slope,
                "median_per_dial": medians.tolist(),
                "n_per_dial": ns,
            }
        )
    # Sort descending by slope; nan slopes go to the end.
    rows.sort(key=lambda r: (np.isnan(r["slope"]), -r["slope"] if not np.isnan(r["slope"]) else 0))
    return rows


def make_figure(rows: list[dict], dial_landings: list[float], out_dir: Path) -> None:
    set_paper_style("blog")

    personas = [r["persona"] for r in rows]
    slopes = np.array([r["slope"] for r in rows])
    n_per_dial_all = np.array([r["n_per_dial"] for r in rows])  # shape (n_personas, 3)
    n_per_dial_min = n_per_dial_all.min(axis=1)

    # Color scheme: source-adjacent vs semantically-distant.
    color_adjacent = paper_palette_role("primary")
    color_distant = paper_palette_role("baseline")
    color_default = paper_palette_role("accent")  # for the bare "assistant" persona
    colors = []
    for p in personas:
        if p == "assistant":
            colors.append(color_default)
        elif p in SOURCE_ADJACENT:
            colors.append(color_adjacent)
        else:
            colors.append(color_distant)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    y_positions = np.arange(len(personas))
    # Light horizontal guide line at each row (drawn first, behind the scatter).
    for y in y_positions:
        ax.hlines(y, 0.40, 1.20, color="#EAEAEA", linewidth=0.6, zorder=1)
    ax.scatter(
        slopes,
        y_positions,
        s=70,
        c=colors,
        edgecolor="#333333",
        linewidth=0.7,
        zorder=3,
    )

    # Reference vertical line at the pooled bystander-only median slope.
    pooled_slope = float(np.nanmedian(slopes))
    ax.axvline(pooled_slope, color="#777777", linewidth=1.0, linestyle="--", zorder=2)
    # Label the median line at the top of the plot (above the topmost persona
    # row), where it cannot collide with the legend at the bottom-right.
    ax.text(
        pooled_slope + 0.01,
        -0.7,
        f"median {pooled_slope:.2f}",
        va="bottom",
        ha="left",
        fontsize=9,
        color="#555555",
        fontstyle="italic",
    )

    ax.set_yticks(y_positions)
    # Display underscores as spaces (e.g. "navy_seal" → "navy seal").
    persona_labels = [p.replace("_", " ") for p in personas]
    ax.set_yticklabels(persona_labels, fontsize=9.5)
    ax.invert_yaxis()  # highest slope at top
    ax.set_xlim(0.40, 1.20)
    ax.set_xlabel("Per-persona slope (nat of bystander log-prob shift per nat of dial)")

    # Annotate sample size per row (n_min across dials) on the right, inside
    # the plot area so it does not get clipped.
    for y, n in zip(y_positions, n_per_dial_min, strict=True):
        ax.text(
            1.18,
            y,
            f"n={int(n)}/dial",
            va="center",
            ha="right",
            fontsize=7.5,
            color="#888888",
        )

    # Build a frameless legend with three categories.
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=color_adjacent,
            markeredgecolor="#333333",
            label="Source-adjacent (medical / security / investigation)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=color_distant,
            markeredgecolor="#333333",
            label="Semantically distant",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=color_default,
            markeredgecolor="#333333",
            label="Default assistant",
        ),
    ]
    # Place the legend just BELOW the axes (in the gap above the x-axis label
    # and source-line) so it never overlaps the scatter points. Three columns.
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=8.5,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.8,
    )

    set_title_subtitle(
        ax,
        title="Per-persona bystander leakage slope across the three dial points",
        subtitle=(
            "Slope = how fast each held-out persona's median log P(marker) shift climbs "
            "with the dial. Bystander-only reads (source-persona reads excluded)."
        ),
        source=(
            "Per-persona median log-prob shift over true-bystander joint cells; "
            "slope = (deep − shallow) / (deep landing − shallow landing) at sweep-side "
            "median landings 5.89 → 9.81 → 17.24 nat."
        ),
    )

    # set_title_subtitle uses anchored text boxes that fight tight_layout; use
    # subplots_adjust so the title block has room and the right margin holds
    # the per-row n=…/dial annotations without clipping. Extra bottom space for
    # the legend row below the axes.
    fig.subplots_adjust(left=0.20, right=0.94, top=0.86, bottom=0.22)
    savefig_paper(fig, "issue_550/bystander_slope_by_persona", dir=str(out_dir))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dirs",
        nargs=3,
        required=True,
        type=Path,
        help="Eval dirs for shallow, mid, deep in that order (e.g. "
        "eval_results/issue_527/eval eval_results/issue_550/eval eval_results/issue_538/eval).",
    )
    parser.add_argument(
        "--sweep-dirs",
        nargs=3,
        required=False,
        type=Path,
        help="Sweep dirs for shallow, mid, deep (optional; if provided, dial "
        "landings are the median final_source_delta_nats across all 18 sweep "
        "cells per dial, matching the body's 5.89 / 9.81 / 17.24 numbers).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="Output base dir (savefig_paper adds the subpath).",
    )
    args = parser.parse_args()

    runs = list(zip(DIAL_ORDER, args.eval_dirs, strict=True))
    for run_key, eval_dir in runs:
        if not eval_dir.is_dir():
            raise SystemExit(f"missing eval dir for {run_key}: {eval_dir}")
    if args.sweep_dirs is not None:
        for sd in args.sweep_dirs:
            if not sd.is_dir():
                raise SystemExit(f"missing sweep dir: {sd}")

    per_persona, dial_landings, eval_panel = build_bystander_data(runs, sweep_dirs=args.sweep_dirs)
    rows = compute_persona_slopes(per_persona, dial_landings)

    # Sanity print.
    print(f"dial_landings (shallow / mid / deep): {dial_landings}")
    print("\npersona | slope | n_per_dial (shallow / mid / deep)")
    for r in rows:
        print(
            f"  {r['persona']:25s}  {r['slope']:+.3f}  "
            f"{r['n_per_dial'][0]}/{r['n_per_dial'][1]}/{r['n_per_dial'][2]}"
        )

    make_figure(rows, dial_landings, args.out)


if __name__ == "__main__":
    main()
