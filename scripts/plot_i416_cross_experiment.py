"""Cross-experiment comparison of #398 (librarian source) vs #416 (software_engineer source).

Produces three artifacts (paper-plots style):

  1. ``rank_trajectory_overlay.png`` — two-panel side-by-side. Left panel:
     #398 librarian-source rank trajectory (thick blue) + #398 top-6 cluster.
     Right panel: #416 software_engineer-source rank trajectory (thick blue)
     + librarian-as-bystander (thin dashed) + top-6 cluster. Same y-axis,
     same x-scale; the two trajectories are visually comparable.

  2. ``panel_mean_logp_overlay.png`` — one-panel overlay. Two lines:
     #398 panel-mean log p(※) at pos0 across 22 checkpoints,
     #416 panel-mean log p(※) at pos0 across 22 checkpoints.
     Inset: step-wise (#416 - #398) panel-mean difference;
     RMS deviation from constant / linear / quadratic offset fits
     annotated (readout 4).

  3. ``readout5_within_source_elevation.json`` — three numerical magnitudes
     (5a / 5b / 5c) computed from existing data per critic-round-1 fix #2.
     Saved next to the figures so the writeup consumes them directly.

Plus readout 2 (top-6 cluster identity at step 1600) is computed and printed
to stdout + saved into a small JSON sidecar.

Degrades gracefully when #416's logp file is absent: the readout-5 librarian-
internal sanity check (which the script runs on #398's data alone as a
self-check) still produces the librarian within-source elevation ~+1.23 nat
/ ~+13 ranks values that the planner cited.
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
)

# #398-step-1600 top-6 (per critic-round-1 #4 / planner §6.1 fix). The
# #398-step-5 cluster is the legacy reference set; we report set-agreement
# against BOTH the late-plateau and the early cluster.
TOP6_398_STEP1600 = [
    "fammate_task_2",
    "comedian",
    "fammate_instruction_1",
    "fammate_context_2",
    "fammate_context_1",
    "poet",
]
TOP6_398_STEP5 = [
    "comedian",
    "fammate_task_2",
    "fammate_context_2",
    "french_person",
    "poet",
    "villain",
]

# Late-checkpoint set for the (5a) within-source elevation magnitude.
LATE_CHECKPOINTS = [600, 800, 1000, 1200, 1600]


def _mean_pos0(data: dict, step: int, persona: str) -> float:
    """Per-persona mean pos0 log p at one checkpoint. Raises KeyError if missing."""
    cell = data["per_step"][str(step)][persona]
    return float(np.mean(cell["pos0"]))


def _delta_from_step5(data: dict, persona: str, step: int) -> float:
    """Δlog_p(persona, step) = mean_pos0(step) - mean_pos0(5)."""
    return _mean_pos0(data, step, persona) - _mean_pos0(data, 5, persona)


def _per_step_panel_mean(data: dict) -> dict[int, float]:
    """Per-step mean pos0 log p averaged over the panel personas."""
    panel = data["panel"]
    steps = sorted(int(s) for s in data["per_step"])
    return {s: float(np.mean([_mean_pos0(data, s, p) for p in panel])) for s in steps}


def _ranks_at(data: dict, step: int) -> dict[str, int]:
    """Per-persona rank (1 = highest pos0 mean log p) at one checkpoint."""
    panel = data["panel"]
    means = {p: _mean_pos0(data, step, p) for p in panel}
    sorted_p = sorted(means.items(), key=lambda kv: -kv[1])
    return {p: i + 1 for i, (p, _) in enumerate(sorted_p)}


def _top6_at(data: dict, step: int) -> list[str]:
    """Return the top-6 personas (by pos0 mean log p) at one checkpoint."""
    ranks = _ranks_at(data, step)
    sorted_by_rank = sorted(ranks.items(), key=lambda kv: kv[1])
    return [p for p, _ in sorted_by_rank[:6]]


def compute_readout5(
    d398: dict,
    d416: dict | None,
    source_persona: str,
) -> dict:
    """Compute (5a) / (5b) / (5c) within-source elevation magnitudes.

    Returns
    -------
    dict with keys ``metric_5a`` (Δ-source-#416 - Δ-source-as-bystander-#398),
    ``metric_5b`` (within-#416 source - panel-mean Δ), ``metric_5c``
    (rank elevation from step-5 baseline), plus librarian-internal sanity
    versions computed against #398 alone (the librarian within-source
    elevation that planner cited as +1.23 nat / +13 ranks).
    """
    # Always compute the #398 librarian-internal self-check (sanity, no #416 needed).
    sanity: dict[str, float] = {}
    try:
        delta_lib_398 = float(
            np.mean([_delta_from_step5(d398, "librarian", s) for s in LATE_CHECKPOINTS])
        )
        delta_swe_as_bystander_398 = float(
            np.mean([_delta_from_step5(d398, source_persona, s) for s in LATE_CHECKPOINTS])
        )
        sanity["librarian_398_within_source_elevation_5a_proxy_nat"] = (
            delta_lib_398 - delta_swe_as_bystander_398
        )
        # rank-elevation check on librarian in #398
        steps_398 = sorted(int(s) for s in d398["per_step"])
        lib_step5_rank = _ranks_at(d398, 5).get("librarian", -1)
        lib_best_rank = min(_ranks_at(d398, s)["librarian"] for s in steps_398)
        sanity["librarian_398_rank_elevation_5c_proxy"] = lib_step5_rank - lib_best_rank
    except Exception as e:
        sanity["sanity_compute_error"] = str(e)

    out: dict = {
        "source_persona": source_persona,
        "late_checkpoints": LATE_CHECKPOINTS,
        "sanity_398_internal": sanity,
    }

    if d416 is None:
        out["status"] = "absent_416_data"
        return out

    # (5a) Δlog_p(swe-as-source-#416) - Δlog_p(swe-as-bystander-#398) at late ckpts.
    delta_source_416 = float(
        np.mean([_delta_from_step5(d416, source_persona, s) for s in LATE_CHECKPOINTS])
    )
    delta_source_as_bystander_398 = float(
        np.mean([_delta_from_step5(d398, source_persona, s) for s in LATE_CHECKPOINTS])
    )
    metric_5a = delta_source_416 - delta_source_as_bystander_398

    # (5b) Δlog_p(source-#416) - Δlog_p(panel-mean-#416) at step 1600.
    panel_416 = d416["panel"]
    delta_source_416_1600 = _delta_from_step5(d416, source_persona, 1600)
    delta_panel_mean_416_1600 = float(
        np.mean([_delta_from_step5(d416, p, 1600) for p in panel_416])
    )
    metric_5b = delta_source_416_1600 - delta_panel_mean_416_1600

    # (5c) Δrank(source-#416) from step-5 baseline.
    steps_416 = sorted(int(s) for s in d416["per_step"])
    ranks_416 = {s: _ranks_at(d416, s) for s in steps_416}
    step5_rank = ranks_416[5][source_persona]
    best_rank = min(ranks_416[s][source_persona] for s in steps_416)
    metric_5c = step5_rank - best_rank

    out.update(
        {
            "status": "ok",
            "metric_5a_within_source_elevation_nat": float(metric_5a),
            "metric_5a_components": {
                "delta_source_416_late_avg": delta_source_416,
                "delta_source_as_bystander_398_late_avg": delta_source_as_bystander_398,
            },
            "metric_5b_source_vs_panel_mean_416_step1600_nat": float(metric_5b),
            "metric_5b_components": {
                "delta_source_416_1600": float(delta_source_416_1600),
                "delta_panel_mean_416_1600": float(delta_panel_mean_416_1600),
            },
            "metric_5c_rank_elevation_416": int(metric_5c),
            "metric_5c_components": {
                "step5_rank": int(step5_rank),
                "best_rank": int(best_rank),
            },
            "signature_thresholds": {
                "H1a_5a_ge_nat": 1.0,
                "H1a_5b_ge_nat": 0.5,
                "H1a_5c_ge_ranks": 10,
                "H1b_5a_abs_le_nat": 0.5,
                "H1b_5b_abs_le_nat": 0.3,
                "H1b_5c_le_ranks": 3,
                "H2_5b_ge_nat": 3.0,
            },
        }
    )
    return out


def compute_readout2(d398: dict, d416: dict | None) -> dict:
    """Top-6 cluster identity at step 1600 + baseline-stability floor."""
    top6_398_1600 = _top6_at(d398, 1600) if 1600 in {int(s) for s in d398["per_step"]} else None
    top6_398_5 = _top6_at(d398, 5) if 5 in {int(s) for s in d398["per_step"]} else None
    out: dict = {
        "reference_top6_398_step1600": top6_398_1600,
        "reference_top6_398_step5": top6_398_5,
        "panel_mechanical_floor_398_step5_vs_step1600": (
            len(set(top6_398_5) & set(top6_398_1600)) if (top6_398_5 and top6_398_1600) else None
        ),
    }
    if d416 is not None and 1600 in {int(s) for s in d416["per_step"]}:
        top6_416_1600 = _top6_at(d416, 1600)
        out["top6_416_step1600"] = top6_416_1600
        if top6_398_1600 is not None:
            out["agreement_2a_416_step1600_vs_398_step1600"] = len(
                set(top6_416_1600) & set(top6_398_1600)
            )
        if top6_398_5 is not None:
            out["agreement_2b_416_step1600_vs_398_step5"] = len(
                set(top6_416_1600) & set(top6_398_5)
            )
    return out


def compute_readout4(d398: dict, d416: dict | None) -> dict:
    """Panel-mean log p Δ — RMS deviation against constant / linear / quadratic."""
    out: dict = {"status": "absent_416_data"}
    if d416 is None:
        return out
    panel_398_means = _per_step_panel_mean(d398)
    panel_416_means = _per_step_panel_mean(d416)
    common_steps = sorted(set(panel_398_means.keys()) & set(panel_416_means.keys()))
    if len(common_steps) < 3:
        out["status"] = "insufficient_common_steps"
        out["common_steps"] = common_steps
        return out
    diffs = np.array([panel_416_means[s] - panel_398_means[s] for s in common_steps], dtype=float)
    # Constant fit
    offset_c = float(np.mean(diffs))
    rms_const = float(np.sqrt(np.mean((diffs - offset_c) ** 2)))
    # Linear-in-log10(step) fit
    xs = np.log10(np.array(common_steps, dtype=float))
    poly1 = np.polyfit(xs, diffs, 1)
    rms_lin = float(np.sqrt(np.mean((diffs - np.polyval(poly1, xs)) ** 2)))
    # Quadratic fit
    poly2 = np.polyfit(xs, diffs, 2)
    rms_quad = float(np.sqrt(np.mean((diffs - np.polyval(poly2, xs)) ** 2)))
    out.update(
        {
            "status": "ok",
            "common_steps": common_steps,
            "step_diffs": diffs.tolist(),
            "constant_offset": offset_c,
            "rms_from_constant_nat": rms_const,
            "rms_from_linear_nat": rms_lin,
            "rms_from_quadratic_nat": rms_quad,
            "thresholds": {"H1_rms_const_le_nat": 2.0, "H1_rms_lin_le_nat": 1.5},
        }
    )
    return out


def _plot_one_panel(
    ax,
    data: dict,
    source_persona: str,
    top_cluster: list[str],
    also_highlight: list[str],
    title: str,
) -> None:
    """Render one rank-vs-step panel (per the #398 hero-figure shape)."""
    panel = data["panel"]
    steps = sorted(int(s) for s in data["per_step"])
    n = len(panel)
    ranks = {p: [] for p in panel}
    for s in steps:
        rmap = _ranks_at(data, s)
        for p in panel:
            ranks[p].append(rmap[p])

    grey = "#bdbdbd"
    highlighted = set(top_cluster) | set(also_highlight) | {source_persona}
    for p, rs in ranks.items():
        if p in highlighted:
            continue
        ax.plot(steps, rs, color=grey, linewidth=0.8, alpha=0.55, zorder=1)

    cluster_colors = [
        paper_palette_role("primary"),
        paper_palette_role("accent"),
        paper_palette_role("control"),
        paper_palette_role("baseline"),
        "#8c564b",
        "#9467bd",
    ]
    for p, color in zip(top_cluster, cluster_colors, strict=False):
        if p not in ranks:
            continue
        ax.plot(steps, ranks[p], color=color, linewidth=1.4, alpha=0.9, zorder=2)

    for p in also_highlight:
        if p not in ranks or p == source_persona:
            continue
        ax.plot(
            steps,
            ranks[p],
            color="#444444",
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
            zorder=2,
        )

    src_color = "#1f77b4"
    ax.plot(
        steps,
        ranks[source_persona],
        color=src_color,
        linewidth=2.4,
        alpha=1.0,
        zorder=3,
        marker="o",
        markersize=4.0,
        markerfacecolor=src_color,
        markeredgecolor="white",
        markeredgewidth=0.6,
    )

    ax.set_xscale("log")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel(f"rank by mean log p(※)\n(1 = highest, {n} personas)")
    ax.invert_yaxis()
    ax.set_yticks([1, 5, 10, 15, 20, n])
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([5, 10, 25, 70, 200, 600, 1600])
    ax.set_xticklabels(["5", "10", "25", "70", "200", "600", "1600"])
    ax.set_title(title, fontsize=10)


def plot_rank_overlay(d398: dict, d416: dict, source_persona: str, output_dir: Path) -> None:
    """Two-panel side-by-side rank trajectory plot."""
    set_paper_style("blog")
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    _plot_one_panel(
        ax_l,
        d398,
        source_persona="librarian",
        top_cluster=TOP6_398_STEP1600,
        also_highlight=[],
        title="#398 — librarian-source baseline (parent)",
    )
    _plot_one_panel(
        ax_r,
        d416,
        source_persona=source_persona,
        top_cluster=TOP6_398_STEP1600,
        also_highlight=["librarian"],
        title=f"#416 — {source_persona}-source (this experiment)",
    )
    fig.suptitle(
        f"Rank-trajectory overlay: #398 (librarian source) vs #416 ({source_persona} source)",
        fontsize=12,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "rank_trajectory_overlay", dir=str(output_dir))
    plt.close(fig)


def plot_panel_mean_overlay(
    d398: dict,
    d416: dict,
    readout4: dict,
    output_dir: Path,
) -> None:
    """Panel-mean log p overlay + inset of step-wise diffs."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    pm_398 = _per_step_panel_mean(d398)
    pm_416 = _per_step_panel_mean(d416)
    common = sorted(set(pm_398) & set(pm_416))
    ax.plot(common, [pm_398[s] for s in common], "o-", label="#398 librarian source", color="#444")
    ax.plot(common, [pm_416[s] for s in common], "s-", label="#416 swe source", color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("panel-mean log p(※) at pos0 (nat)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Panel-mean log p(※) per step — #398 vs #416 (readout 4)", fontsize=11)

    if readout4.get("status") == "ok":
        diffs = readout4["step_diffs"]
        rms_c = readout4["rms_from_constant_nat"]
        rms_l = readout4["rms_from_linear_nat"]
        rms_q = readout4["rms_from_quadratic_nat"]
        inset = ax.inset_axes([0.55, 0.62, 0.4, 0.32])
        inset.hist(diffs, bins=8, color="#1f77b4", alpha=0.7)
        inset.set_title("step-wise (#416 - #398) diffs (nat)", fontsize=8)
        inset.tick_params(axis="both", labelsize=7)
        rms_text = (
            f"RMS-vs-constant:  {rms_c:.2f} nat\n"
            f"RMS-vs-linear:    {rms_l:.2f} nat\n"
            f"RMS-vs-quadratic: {rms_q:.2f} nat"
        )
        ax.text(
            0.02,
            0.97,
            rms_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            family="monospace",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "panel_mean_logp_overlay", dir=str(output_dir))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--logp-file-398",
        type=Path,
        required=True,
        help="Path to eval_results/issue_398/logp_seed42.json.",
    )
    ap.add_argument(
        "--logp-file-416",
        type=Path,
        default=None,
        help=(
            "Path to eval_results/issue_416/logp_seed42.json. Optional — when "
            "absent, only the #398-internal sanity readouts run (lets the "
            "implementer smoke-test the script before #416 data exists)."
        ),
    )
    ap.add_argument(
        "--source-persona",
        default="software_engineer",
        help="Source persona for #416 (default software_engineer per plan).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for cross-experiment figures + readout JSONs.",
    )
    args = ap.parse_args()

    with open(args.logp_file_398) as f:
        d398 = json.load(f)

    d416: dict | None = None
    if args.logp_file_416 is not None and args.logp_file_416.exists():
        with open(args.logp_file_416) as f:
            d416 = json.load(f)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always compute readouts 5 + 2 + 4 (5 falls back to #398-internal sanity
    # when #416 data is absent).
    r5 = compute_readout5(d398, d416, args.source_persona)
    r2 = compute_readout2(d398, d416)
    r4 = compute_readout4(d398, d416)

    (output_dir / "readout5_within_source_elevation.json").write_text(json.dumps(r5, indent=2))
    (output_dir / "readout2_top6_cluster_identity.json").write_text(json.dumps(r2, indent=2))
    (output_dir / "readout4_panel_mean_overlay.json").write_text(json.dumps(r4, indent=2))

    print(f"Readout 5 status: {r5.get('status')}")
    if "librarian_398_within_source_elevation_5a_proxy_nat" in r5["sanity_398_internal"]:
        v = r5["sanity_398_internal"]["librarian_398_within_source_elevation_5a_proxy_nat"]
        print(f"  #398 librarian within-source elevation (sanity): {v:+.3f} nat")
    if "librarian_398_rank_elevation_5c_proxy" in r5["sanity_398_internal"]:
        v = r5["sanity_398_internal"]["librarian_398_rank_elevation_5c_proxy"]
        print(f"  #398 librarian rank elevation (sanity):          {v:+d} ranks")
    if r5.get("status") == "ok":
        m5a = r5["metric_5a_within_source_elevation_nat"]
        m5b = r5["metric_5b_source_vs_panel_mean_416_step1600_nat"]
        m5c = r5["metric_5c_rank_elevation_416"]
        print(f"  metric 5a (within-source elevation):    {m5a:+.3f} nat")
        print(f"  metric 5b (source vs panel-mean #416):  {m5b:+.3f} nat")
        print(f"  metric 5c (rank elevation in #416):     {m5c:+d} ranks")

    ref_late = r2.get("reference_top6_398_step1600")
    print(f"Readout 2 reference top-6 (#398 step 1600): {ref_late}")
    if "top6_416_step1600" in r2:
        agree_2a = r2.get("agreement_2a_416_step1600_vs_398_step1600")
        agree_2b = r2.get("agreement_2b_416_step1600_vs_398_step5")
        print(f"Readout 2 #416 step 1600 top-6:             {r2['top6_416_step1600']}")
        print(f"  agreement 2a vs #398-step-1600: {agree_2a}/6")
        print(f"  agreement 2b vs #398-step-5:    {agree_2b}/6")
    floor = r2.get("panel_mechanical_floor_398_step5_vs_step1600")
    print(f"  panel-mechanical floor (#398 step-5 vs step-1600): {floor}/6")

    if r4.get("status") == "ok":
        print(
            f"Readout 4 RMS const={r4['rms_from_constant_nat']:.2f} "
            f"lin={r4['rms_from_linear_nat']:.2f} "
            f"quad={r4['rms_from_quadratic_nat']:.2f} nat"
        )

    # Figures need both runs' data. Skip when #416 is missing.
    if d416 is not None:
        plot_rank_overlay(d398, d416, args.source_persona, output_dir)
        plot_panel_mean_overlay(d398, d416, r4, output_dir)
        print(f"Wrote rank_trajectory_overlay + panel_mean_logp_overlay → {output_dir}")
    else:
        print("#416 data absent — skipped cross-experiment figures (readout JSONs still written).")


if __name__ == "__main__":
    main()
