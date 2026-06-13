# ruff: noqa: RUF001, RUF002, RUF003
"""Per-persona bystander-slope vs persona-distance correlation (task #550 Finding 3 follow-up).

Question: does each held-out persona's bystander-only slope of marker log-prob
shift across the dial track its base-model representation distance to the four
source personas (florist, medical_doctor, librarian, police_officer)?
Two competing readings of Finding 3:
  (a) source-persona generalization: slope should track distance to sources.
  (b) mechanical marker-mass spread: slope structure unrelated to distance.

Distance source: `eval_results/issue_527/pair_selection.json::cos_centered_L20`
— the same metric the body's pair-selection used (layer 20 base-model
representations, global-mean centered cosine similarity over the 19-persona
panel). Distance = 1 − cos. All 19 eval personas + all 4 source personas are
in that matrix.

Per-persona distance to sources:
  - For each non-source eval persona p (15 personas): mean / min of distance
    to all 4 source personas.
  - For each source persona s (4 personas): the slope is read only on the
    OTHER pair's joint cells (bystander-only convention from the slope
    figure), so distance is mean / min of distance to the OTHER pair's two
    sources only. This is flagged in the output.

Slope source: reuses `build_bystander_data` + `compute_persona_slopes` from
`scripts/issue550_bystander_slope_figure.py` (same bystander-only convention,
same two-point endpoint slope across the three dial points, same
sweep-side median landings).

Outputs:
  - eval_results/issue_550/analysis/slope_distance_correlation.json
  - figures/issue_550/slope_vs_distance.{png,pdf,meta.json}

The correlation is a Spearman ρ across 19 personas. p-values are reported as
descriptive (n=19 is small); the headline is the rank-correlation point
estimate and its sign.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse the slope machinery so the slopes here are byte-equivalent to the
# slopes plotted in figures/issue_550/bystander_slope_by_persona.{png,pdf}.
from issue550_bystander_slope_figure import (  # type: ignore
    DIAL_ORDER,
    PAIR_SOURCES,
    build_bystander_data,
    compute_persona_slopes,
)
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

SOURCE_PERSONAS_PAIR1 = ("florist", "medical_doctor")
SOURCE_PERSONAS_PAIR2 = ("librarian", "police_officer")
ALL_SOURCES = SOURCE_PERSONAS_PAIR1 + SOURCE_PERSONAS_PAIR2

SOURCE_ADJACENT = {
    "paramedic",
    "surgeon",
    "navy_seal",
    "army_medic",
    "pentester",
    "cybersec_consultant",
    "private_investigator",
}


def load_cos_matrix(pair_selection_path: Path) -> tuple[list[str], np.ndarray, int, str]:
    """Return (persona_names, cos_matrix, layer, centering) from pair_selection.json."""
    with pair_selection_path.open() as f:
        d = json.load(f)
    names = list(d["persona_names"])
    mat = np.array(d["cos_centered_L20"], dtype=np.float64)
    if mat.shape != (len(names), len(names)):
        raise SystemExit(f"cos matrix shape {mat.shape} != {(len(names), len(names))}")
    return names, mat, int(d["extraction_layer"]), str(d["centering"])


def distance_to_sources(
    persona: str,
    names: list[str],
    cos_matrix: np.ndarray,
) -> tuple[float, float, list[str]]:
    """Return (mean_distance, min_distance, sources_used) for a persona.

    Bystander-only convention: source personas read as bystanders only on the
    OTHER pair, so their distance is computed against the OTHER pair's two
    sources only; non-source personas read against all 4 sources.
    """
    if persona in SOURCE_PERSONAS_PAIR1:
        sources_used = list(SOURCE_PERSONAS_PAIR2)
    elif persona in SOURCE_PERSONAS_PAIR2:
        sources_used = list(SOURCE_PERSONAS_PAIR1)
    else:
        sources_used = list(ALL_SOURCES)

    idx_p = names.index(persona)
    distances = []
    for src in sources_used:
        idx_s = names.index(src)
        cos_ps = float(cos_matrix[idx_p, idx_s])
        distances.append(1.0 - cos_ps)
    return float(np.mean(distances)), float(min(distances)), sources_used


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1]
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def make_figure(
    rows: list[dict],
    distance_kind: str,
    rho: float,
    p_value: float,
    n: int,
    out_dir: Path,
) -> None:
    """Scatter: x = distance, y = slope, colored by source-adjacency.

    distance_kind ∈ {"mean", "min"}.
    """
    set_paper_style("blog")

    color_adjacent = paper_palette_role("primary")
    color_distant = paper_palette_role("baseline")
    color_default = paper_palette_role("accent")  # bare assistant
    color_source = paper_palette_role("neutral")  # the 4 source personas

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for r in rows:
        p = r["persona"]
        if p in ALL_SOURCES:
            c = color_source
            marker = "D"
            size = 90
        elif p == "assistant":
            c = color_default
            marker = "o"
            size = 80
        elif p in SOURCE_ADJACENT:
            c = color_adjacent
            marker = "o"
            size = 80
        else:
            c = color_distant
            marker = "o"
            size = 80
        ax.scatter(
            r[f"distance_{distance_kind}"],
            r["slope"],
            s=size,
            c=c,
            marker=marker,
            edgecolor="#333333",
            linewidth=0.7,
            zorder=3,
        )

    # Label each point with the persona name.
    for r in rows:
        x = r[f"distance_{distance_kind}"]
        y = r["slope"]
        ax.annotate(
            r["persona"].replace("_", " "),
            xy=(x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.5,
            color="#444444",
        )

    ax.set_xlabel(
        f"{distance_kind.capitalize()}-of-sources centered-cosine distance at L20 "
        f"(1 − cos; layer 20, global-mean centered)"
    )
    ax.set_ylabel("Per-persona bystander-only slope (nat / nat of dial)")

    # Build legend with the four category markers.
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markersize=8,
            markerfacecolor=color_source,
            markeredgecolor="#333333",
            label="Source persona (other pair only)",
        ),
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
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=8.5,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.6,
    )

    rho_label = f"Spearman ρ = {rho:+.3f}    p = {p_value:.3f}    n = {n}"
    set_title_subtitle(
        ax,
        title=f"Bystander-slope vs persona-distance ({distance_kind} over sources)",
        subtitle=(
            "If Finding 3's slope ordering reflects source-persona generalization, "
            "distance and slope should be NEGATIVELY rank-correlated."
        ),
        source=rho_label,
    )

    # paper_plots applies constrained_layout via set_paper_style; no manual
    # subplots_adjust (the two engines conflict and the layout collapses).
    savefig_paper(fig, f"issue_550/slope_vs_distance_{distance_kind}", dir=str(out_dir))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dirs",
        nargs=3,
        type=Path,
        default=[
            Path("eval_results/issue_527/eval"),
            Path("eval_results/issue_550/eval"),
            Path("eval_results/issue_538/eval"),
        ],
        help="Eval dirs for shallow, mid, deep in that order.",
    )
    parser.add_argument(
        "--sweep-dirs",
        nargs=3,
        type=Path,
        default=[
            Path("eval_results/issue_527/sweep"),
            Path("eval_results/issue_550/sweep"),
            Path("eval_results/issue_538/sweep"),
        ],
        help="Sweep dirs for shallow, mid, deep (used for dial landings).",
    )
    parser.add_argument(
        "--pair-selection",
        type=Path,
        default=Path("eval_results/issue_527/pair_selection.json"),
        help="Path to issue_527 pair_selection.json (carries L20 cos matrix).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval_results/issue_550/analysis/slope_distance_correlation.json"),
    )
    parser.add_argument(
        "--out-fig-dir",
        type=Path,
        default=Path("figures"),
        help="Output base dir (savefig_paper adds the subpath).",
    )
    args = parser.parse_args()

    # Sanity: every dir exists.
    for d in args.eval_dirs + args.sweep_dirs:
        if not d.is_dir():
            raise SystemExit(f"missing dir: {d}")
    if not args.pair_selection.is_file():
        raise SystemExit(f"missing pair_selection.json: {args.pair_selection}")

    # --- 1) Slopes (reuse the slope-figure machinery) ---
    runs = list(zip(DIAL_ORDER, args.eval_dirs, strict=True))
    per_persona, dial_landings, eval_panel = build_bystander_data(runs, sweep_dirs=args.sweep_dirs)
    slope_rows = compute_persona_slopes(per_persona, dial_landings)
    slope_by_persona = {r["persona"]: r for r in slope_rows}

    # --- 2) Distances (L20 centered cosine, 1 − cos) ---
    names, cos_matrix, layer, centering = load_cos_matrix(args.pair_selection)
    if set(names) != set(eval_panel):
        raise SystemExit(
            f"eval panel ({len(eval_panel)}) != pair_selection persona set "
            f"({len(names)}); cannot align"
        )

    # --- 3) Per-persona rows: slope + distance variants ---
    rows: list[dict] = []
    for persona in eval_panel:
        if persona not in slope_by_persona:
            # A persona with no bystander reads at all — should not happen, but
            # guard against it.
            continue
        mean_dist, min_dist, sources_used = distance_to_sources(persona, names, cos_matrix)
        slope_row = slope_by_persona[persona]
        rows.append(
            {
                "persona": persona,
                "slope": slope_row["slope"],
                "median_per_dial": slope_row["median_per_dial"],
                "n_per_dial": slope_row["n_per_dial"],
                "distance_mean": mean_dist,
                "distance_min": min_dist,
                "sources_used": sources_used,
                "is_source": persona in ALL_SOURCES,
            }
        )

    # --- 4) Spearman correlations ---
    def _spearman(distance_key: str, *, drop_sources: bool) -> dict:
        rs = [r for r in rows if not (drop_sources and r["is_source"])]
        slopes = np.array([r["slope"] for r in rs])
        dists = np.array([r[distance_key] for r in rs])
        valid = ~(np.isnan(slopes) | np.isnan(dists))
        rho, p = spearmanr(dists[valid], slopes[valid])
        return {
            "rho": float(rho),
            "p_value": float(p),
            "n": int(valid.sum()),
            "distance_key": distance_key,
            "drop_sources": drop_sources,
        }

    correlations = {
        "mean_all19": _spearman("distance_mean", drop_sources=False),
        "min_all19": _spearman("distance_min", drop_sources=False),
        "mean_n15_no_sources": _spearman("distance_mean", drop_sources=True),
        "min_n15_no_sources": _spearman("distance_min", drop_sources=True),
    }

    # --- 5) Persist JSON ---
    out_json = args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "schema_version": "issue_550_slope_distance_correlation_v1",
        "question": (
            "Is each held-out persona's bystander-only slope of marker "
            "log-prob shift across the dial correlated with its base-model "
            "L20 centered-cosine distance to the four source personas?"
        ),
        "convention_notes": [
            "Slope = (deep_median - shallow_median) / (deep_landing - shallow_landing), "
            "bystander-only joint cells, reusing build_bystander_data + "
            "compute_persona_slopes from scripts/issue550_bystander_slope_figure.py.",
            "Distance = 1 - cos at extraction_layer=20, centering=global_mean, "
            "from eval_results/issue_527/pair_selection.json::cos_centered_L20.",
            "For the 4 source personas, distance is computed against the OTHER "
            "pair's two sources only (bystander-only convention).",
            "For the 15 non-source personas, distance is computed against all 4 sources.",
            "Source-persona generalization predicts a NEGATIVE rank correlation "
            "(closer to sources => larger slope).",
            "p-values are descriptive at n=19; the headline is the sign and "
            "magnitude of the point estimate.",
        ],
        "distance_layer": layer,
        "distance_centering": centering,
        "dial_landings_nat": dial_landings,
        "dial_order": DIAL_ORDER,
        "source_personas_pair1": list(SOURCE_PERSONAS_PAIR1),
        "source_personas_pair2": list(SOURCE_PERSONAS_PAIR2),
        "pair_sources_map": {k: list(v) for k, v in PAIR_SOURCES.items()},
        "correlations": correlations,
        "per_persona": rows,
        "git_commit": git_commit(),
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    # --- 6) Figures (two: mean-distance and min-distance) ---
    make_figure(
        rows,
        distance_kind="mean",
        rho=correlations["mean_all19"]["rho"],
        p_value=correlations["mean_all19"]["p_value"],
        n=correlations["mean_all19"]["n"],
        out_dir=args.out_fig_dir,
    )
    make_figure(
        rows,
        distance_kind="min",
        rho=correlations["min_all19"]["rho"],
        p_value=correlations["min_all19"]["p_value"],
        n=correlations["min_all19"]["n"],
        out_dir=args.out_fig_dir,
    )

    # --- 7) Console digest ---
    print(f"dial_landings (shallow / mid / deep): {dial_landings}")
    print(f"distance layer: L{layer}, centering: {centering}")
    print()
    print(f"{'persona':25s}  {'slope':>7s}  {'mean_d':>7s}  {'min_d':>7s}  sources")
    for r in sorted(rows, key=lambda r: -r["slope"]):
        print(
            f"  {r['persona']:23s}  {r['slope']:+.3f}  "
            f"{r['distance_mean']:.3f}  {r['distance_min']:.3f}  "
            f"{','.join(r['sources_used'])}"
        )
    print()
    print("Spearman correlations (slope vs distance):")
    for label, c in correlations.items():
        print(f"  {label:25s}  rho = {c['rho']:+.3f}  p = {c['p_value']:.3f}  n = {c['n']}")
    print(f"\nwrote: {out_json}")
    print(f"figures: {args.out_fig_dir}/issue_550/slope_vs_distance_{{mean,min}}.{{png,pdf}}")


if __name__ == "__main__":
    main()
