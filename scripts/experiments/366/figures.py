# ruff: noqa: RUF001, RUF003
# Plot labels use math typography (− minus, × multiply, Δ delta) intentionally.
"""Four SVG figures for issue #366 cascade results.

1. **Headline cascade curve** (``fig01_cascade_curves.svg``): T-C delta on
   recipient persona for R(B|A), R(C|A), R(D|A), R(E|A) at chain depths
   N=2..5. Solid line at seed42; second line at seed137 for N=3 robustness.

2. **Per-pair conditional ladder** (``fig02_pair_conditional_ladder.svg``):
   for each N, the adjacent-pair conditionals R(B|A), R(C|B), R(D|C),
   R(E|D) on the recipient — shows whether the cascade is "transmitted
   chain link by chain link" or "skipped".

3. **Ablate vs T_3 vs C_3** (``fig03_ablate_compare.svg``): three-bar
   panel for the N=3 ablation — does training the donor on B→C only
   suffice for the recipient to emit B given A?

4. **Donor fidelity heatmap** (``fig04_donor_fidelity.svg``): the donor
   persona's R(target | trigger) for each adjacent pair, per adapter —
   a sanity check that the chain was actually learned by the donor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _save_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="svg", bbox_inches="tight")
    logger.info("Saved figure: %s", path)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",  # keep text editable in SVG output
        }
    )


# Three semantically-named colors used across all four figures.
PALETTE = {
    "T": "#1f77b4",
    "C": "#d62728",
    "delta": "#2ca02c",
    "ablate": "#9467bd",
}


# ── Figure 1: cascade curves ────────────────────────────────────────────────


def _delta_at(curves: dict, t_name: str, c_name: str, key: str) -> dict | None:
    """Pull a single (T,C) pair's stats block from cascade_curves.json."""
    pair_key = f"{t_name}_vs_{c_name}"
    pair = curves.get("pairs", {}).get(pair_key)
    if pair is None:
        return None
    return pair.get("conditionals", {}).get(key)


def figure01_cascade_curves(curves: dict, out_path: Path) -> None:
    """Plot R(target | A) T-C delta vs chain depth N for the recipient persona."""
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    Ns = [2, 3, 4, 5]
    # Each metric is a different line.
    metric_specs = [
        ("R_B_given_A_joint_delta", "Δ R(B | A)", "o", PALETTE["T"]),
        ("R_C_given_A_joint_delta", "Δ R(C | A)", "s", PALETTE["delta"]),
        ("R_D_given_A_joint_delta", "Δ R(D | A)", "^", PALETTE["C"]),
        ("R_E_given_A_joint_delta", "Δ R(E | A)", "D", PALETTE["ablate"]),
    ]
    pair_at_n = {
        2: ("T_2_seed42", "C_2_seed42"),
        3: ("T_3_seed42", "C_3_seed42"),
        4: ("T_4_seed42", "C_4_seed42"),
        5: ("T_5_seed42", "C_5_seed42"),
    }

    for key, label, marker, color in metric_specs:
        ys: list[float] = []
        lo: list[float] = []
        hi: list[float] = []
        x_used: list[int] = []
        for n in Ns:
            t_name, c_name = pair_at_n[n]
            stats = _delta_at(curves, t_name, c_name, key)
            if stats is None or stats.get("delta") is None:
                continue
            ys.append(stats["delta"])
            lo.append(stats["ci_pct"][0])
            hi.append(stats["ci_pct"][1])
            x_used.append(n)
        if not ys:
            continue
        lo_err = [y - lo_v for y, lo_v in zip(ys, lo, strict=True)]
        hi_err = [hi_v - y for y, hi_v in zip(ys, hi, strict=True)]
        ax.errorbar(
            x_used,
            ys,
            yerr=[lo_err, hi_err],
            fmt=f"{marker}-",
            color=color,
            capsize=3,
            label=label,
            linewidth=1.5,
        )

    # N=3 seed137 replicate as a separate marker (open circles)
    n3_seed137 = _delta_at(curves, "T_3_seed137", "C_3_seed137", "R_B_given_A_joint_delta")
    if n3_seed137 is not None and n3_seed137.get("delta") is not None:
        ax.errorbar(
            [3],
            [n3_seed137["delta"]],
            yerr=[
                [n3_seed137["delta"] - n3_seed137["ci_pct"][0]],
                [n3_seed137["ci_pct"][1] - n3_seed137["delta"]],
            ],
            fmt="o",
            mfc="white",
            mec=PALETTE["T"],
            color=PALETTE["T"],
            capsize=3,
            label="Δ R(B | A) seed=137",
        )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks(Ns)
    ax.set_xlabel("Chain depth N (number of cascade markers)")
    ax.set_ylabel("T − C delta on recipient persona (loose match)")
    ax.set_title("Cross-persona cascade: does chunk-binding propagate down the chain?")
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    _save_svg(fig, out_path)
    plt.close(fig)


# ── Figure 2: adjacent-pair conditional ladder ──────────────────────────────


def figure02_conditional_ladder(curves: dict, out_path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    Ns = [2, 3, 4, 5]
    # Adjacent pairs: only show the ones that can exist at each depth.
    pair_at_n = {
        2: ("T_2_seed42", "C_2_seed42"),
        3: ("T_3_seed42", "C_3_seed42"),
        4: ("T_4_seed42", "C_4_seed42"),
        5: ("T_5_seed42", "C_5_seed42"),
    }
    pair_metrics = [
        ("R_B_given_A_joint_delta", "B | A"),
        ("R_C_given_B_joint_delta", "C | B"),
        ("R_D_given_C_joint_delta", "D | C"),
        ("R_E_given_D_joint_delta", "E | D"),
    ]

    width = 0.18
    xs = np.arange(len(Ns))
    for i, (key, label) in enumerate(pair_metrics):
        ys: list[float] = []
        lo_err: list[float] = []
        hi_err: list[float] = []
        for n in Ns:
            stats = _delta_at(curves, *pair_at_n[n], key)
            if stats is None or stats.get("delta") is None:
                ys.append(0.0)
                lo_err.append(0.0)
                hi_err.append(0.0)
                continue
            ys.append(stats["delta"])
            lo_err.append(stats["delta"] - stats["ci_pct"][0])
            hi_err.append(stats["ci_pct"][1] - stats["delta"])
        ax.bar(
            xs + (i - 1.5) * width,
            ys,
            width,
            yerr=[lo_err, hi_err],
            capsize=2,
            label=label,
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("Δ R(target | trigger) on recipient")
    ax.set_title("Adjacent-pair cascade transmission, T − C deltas")
    ax.legend(title="Conditional", fontsize=9, frameon=False)
    fig.tight_layout()
    _save_svg(fig, out_path)
    plt.close(fig)


# ── Figure 3: ablate compare ────────────────────────────────────────────────


def figure03_ablate_compare(curves: dict, cell_aggregates: dict, out_path: Path) -> None:
    """For the N=3 design: compare T_3 / C_3 / T_3_ablate on R(B|A) recipient.

    cell_aggregates: nested {adapter_name: {persona: {... metric ...}}}
    """
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    adapters = ["T_3_seed42", "C_3_seed42", "T_3_ablate_seed42"]
    labels = ["T (chain trained)", "C (control)", "T_ablate (B→C only)"]
    colors = [PALETTE["T"], PALETTE["C"], PALETTE["ablate"]]

    persona = "software_engineer"
    metric = "R_B_given_A_loose"

    vals: list[float] = []
    for a in adapters:
        cell = cell_aggregates.get(a, {}).get(persona, {})
        v = cell.get(metric)
        vals.append(v if v is not None else 0.0)

    bars = ax.bar(np.arange(len(adapters)), vals, color=colors, edgecolor="black")
    for bar, v in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(np.arange(len(adapters)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("R(B | A) on recipient (loose)")
    ax.set_ylim(0, max(0.5, max(vals) * 1.3 if vals else 0.5))
    ax.set_title("N=3 ablation: B→C-only donor vs full chain donor")
    fig.tight_layout()
    _save_svg(fig, out_path)
    plt.close(fig)


# ── Figure 4: donor fidelity heatmap ────────────────────────────────────────


def figure04_donor_fidelity(donor_rows: list[dict], out_path: Path) -> None:
    _style()
    if not donor_rows:
        # Write an empty placeholder so the artifact exists.
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No donor fidelity data", ha="center", va="center")
        ax.set_axis_off()
        _save_svg(fig, out_path)
        plt.close(fig)
        return

    # Build matrix: rows = adapters, cols = conditional metrics
    metrics = ["R_B_given_A", "R_C_given_B", "R_D_given_C", "R_E_given_D"]
    adapters = [r["adapter"] for r in donor_rows]
    mat = np.full((len(adapters), len(metrics)), np.nan, dtype=float)
    for i, r in enumerate(donor_rows):
        for j, m in enumerate(metrics):
            v = r.get(m)
            if v is not None:
                mat[i, j] = float(v)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_yticks(np.arange(len(adapters)))
    ax.set_yticklabels(adapters, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="black", fontsize=8)
            else:
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.5 else "black",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, label="R(target | trigger) loose")
    ax.set_title("Donor fidelity: did the librarian learn each chain link?")
    fig.tight_layout()
    _save_svg(fig, out_path)
    plt.close(fig)


def make_all_figures(
    *,
    cascade_curves_path: Path,
    cell_aggregates_dir: Path,
    donor_fidelity_csv: Path,
    figures_dir: Path,
) -> None:
    """Produce all four SVG figures. Each call is idempotent on disk."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(cascade_curves_path) as f:
        curves = json.load(f)

    # Aggregate cell data from per-adapter files for figure 3.
    cell_agg: dict[str, dict] = {}
    if cell_aggregates_dir.exists():
        for p in cell_aggregates_dir.glob("*.json"):
            with open(p) as f:
                cell_agg[p.stem] = json.load(f)

    donor_rows: list[dict] = []
    if donor_fidelity_csv.exists() and donor_fidelity_csv.stat().st_size > 0:
        import csv

        with open(donor_fidelity_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in list(row.items()):
                    if v == "":
                        row[k] = None
                    elif k in ("n", "denom_A", "denom_B", "denom_C", "denom_D"):
                        try:
                            row[k] = int(v)
                        except (TypeError, ValueError):
                            row[k] = None
                    elif k.startswith("R_"):
                        try:
                            row[k] = float(v)
                        except (TypeError, ValueError):
                            row[k] = None
                donor_rows.append(row)

    figure01_cascade_curves(curves, figures_dir / "fig01_cascade_curves.svg")
    figure02_conditional_ladder(curves, figures_dir / "fig02_pair_conditional_ladder.svg")
    figure03_ablate_compare(curves, cell_agg, figures_dir / "fig03_ablate_compare.svg")
    figure04_donor_fidelity(donor_rows, figures_dir / "fig04_donor_fidelity.svg")
