# ruff: noqa: RUF001, RUF002  # em-dash + nat char + multiplication-sign are intentional
"""Plots for issue #514 clean-result.

Plan §6.3 over-production:
  - hero.png/pdf — source ΔG vs bystander mean ΔG with LoRA reference curve
    (#508 LoRA b1/b2/b3, orange), #508 FT anchors (light blue, half-transparent
    for collapsed cells), and the 6 new #514 FT cells (medium blue dense,
    dark blue lower-LR).
  - matched_rate.png/pdf — grouped bars at source ΔG = 8 nat for LoRA / #508 FT /
    #514 FT, CI from cluster bootstrap. Headline payoff figure.
  - rcollapse.png/pdf — extends #508's r-collapse plot with the 6 new FT cells.
  - per_persona.png/pdf — 15 personas × cells held-out ΔG, identical layout
    to #508.
  - source_self_trajectory.png/pdf — on-policy source-self log P(※) over
    training steps for each of the 6 new FT cells.
  - default_assistant.png/pdf — bare default-assistant ΔG vs source ΔG with
    matched-rate window highlighted (the #508 saturation caveat).

All saved under ``figures/issue_514/``. Each call uses ``savefig_paper`` (PNG +
PDF + .meta.json) when the paper-plot helpers are available; falls back to
plain savefig otherwise.

#514 ft_b2 (#508's collapsed cell with N=1 valid probe) is plotted at
half-transparency and EXCLUDED from the cluster bootstrap (per critic
concerns + plan §4.1.4).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOG = logging.getLogger("issue_514.plot")

# Cell rosters
LORA_REFERENCE_CELLS = ("lora_b1", "lora_b2", "lora_b3")
FT_508_ANCHOR_CELLS = ("ft_b1", "ft_b2", "ft_b3")
FT_514_DENSE_CELLS = ("ft_dense_b30", "ft_dense_b35", "ft_dense_b40", "ft_dense_b45")
FT_514_LOWLR_CELLS = ("ft_lowlr_b50", "ft_lowlr_b100")

# Cells excluded from the cluster bootstrap (collapsed, N=1 valid probe).
# Per plan §4.1.4 + the critic's concern: ft_b2 had 19/20 r-collapsed source
# probes → its source ΔG read sits on 1 probe and inflates the bootstrap variance.
EXCLUDED_FROM_BOOTSTRAP = ("ft_b2",)

CELL_LABELS: dict[str, str] = {
    "lora_b1": "LoRA, 0.25 epoch",
    "lora_b2": "LoRA, 0.5 epoch",
    "lora_b3": "LoRA, 1.0 epoch",
    "ft_b1": "Full FT, 0.25 epoch (#508)",
    "ft_b2": "Full FT, 0.5 epoch (#508 collapsed)",
    "ft_b3": "Full FT, 1.0 epoch (#508 collapsed)",
    "ft_dense_b30": "Full FT, 0.30 epoch (dense)",
    "ft_dense_b35": "Full FT, 0.35 epoch (dense)",
    "ft_dense_b40": "Full FT, 0.40 epoch (dense)",
    "ft_dense_b45": "Full FT, 0.45 epoch (dense)",
    "ft_lowlr_b50": "Full FT, 0.5 epoch (lower LR)",
    "ft_lowlr_b100": "Full FT, 1.0 epoch (lower LR)",
}


def _load_eval(path: Path) -> dict | None:
    if not path.exists():
        LOG.warning("[plot] eval JSON missing: %s", path)
        return None
    return json.loads(path.read_text())


def _load_cells(roster: tuple[str, ...], root_508: Path, root_514: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cell in roster:
        if cell.startswith(("lora_", "ft_b")):
            p = root_508 / f"{cell}_seed42.json"
        else:
            p = root_514 / f"{cell}_seed42.json"
        ej = _load_eval(p)
        if ej is not None:
            out[cell] = ej
    return out


def _cell_xy(ej: dict) -> tuple[float, float]:
    """Return (source_mean ΔG, held_out_mean ΔG) from a cell eval JSON."""
    agg = ej.get("aggregates", {})
    return (
        float(agg.get("source_self_mean_delta_g", float("nan"))),
        float(agg.get("held_out_mean_delta_g", float("nan"))),
    )


def _try_savefig_paper(fig, out_path: Path) -> None:
    """Save via paper_plots.savefig_paper if available; fall back to plain savefig."""
    try:
        from explore_persona_space.analysis.paper_plots import savefig_paper

        savefig_paper(fig, out_path)
    except ImportError:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        LOG.info("[plot] wrote (plain savefig) %s.png + .pdf", out_path)


def _set_style() -> None:
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style("blog")
    except ImportError:
        import matplotlib.pyplot as plt

        plt.rcParams["figure.dpi"] = 150


def hero_figure(
    *,
    lora_cells: dict[str, dict],
    ft_508_cells: dict[str, dict],
    ft_514_dense_cells: dict[str, dict],
    ft_514_lowlr_cells: dict[str, dict],
    output_path: Path,
) -> None:
    """Source-rate curve: LoRA reference + #508 FT anchors + 6 new #514 FT cells."""
    import matplotlib.pyplot as plt

    _set_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # LoRA reference (orange circles).
    if lora_cells:
        pts = [_cell_xy(ej) for ej in lora_cells.values()]
        pts = [(x, y) for x, y in pts if x == x and y == y]  # drop NaNs
        if pts:
            pts.sort(key=lambda p: p[0])
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker="o", color="#E69F00", linewidth=1.8, label="LoRA (#508 ref)")

    # #508 FT anchors (light blue squares; ft_b2 + ft_b3 at half transparency).
    for cell, ej in ft_508_cells.items():
        x, y = _cell_xy(ej)
        if x != x or y != y:
            continue
        alpha = 0.4 if cell in EXCLUDED_FROM_BOOTSTRAP or cell == "ft_b3" else 1.0
        ax.scatter(
            x,
            y,
            marker="s",
            s=70,
            color="#56B4E9",
            alpha=alpha,
            edgecolors="black",
            linewidths=0.5,
            label="Full FT (#508 anchor)" if cell == "ft_b1" else None,
        )

    # Dense lever (medium blue).
    dense_pts = [_cell_xy(ej) for ej in ft_514_dense_cells.values()]
    dense_pts = [(x, y) for x, y in dense_pts if x == x and y == y]
    if dense_pts:
        dense_pts.sort(key=lambda p: p[0])
        xs, ys = zip(*dense_pts, strict=True)
        ax.plot(
            xs, ys, marker="s", color="#0072B2", linewidth=1.6, label="Full FT, dense lever (#514)"
        )

    # Lower-LR lever (dark blue).
    lowlr_pts = [_cell_xy(ej) for ej in ft_514_lowlr_cells.values()]
    lowlr_pts = [(x, y) for x, y in lowlr_pts if x == x and y == y]
    if lowlr_pts:
        lowlr_pts.sort(key=lambda p: p[0])
        xs, ys = zip(*lowlr_pts, strict=True)
        ax.plot(
            xs,
            ys,
            marker="^",
            color="#1F2A5A",
            linewidth=1.6,
            label="Full FT, lower-LR lever (#514)",
        )

    # Matched-rate window 8 ± 1 nat.
    ax.axvspan(7.0, 9.0, color="gray", alpha=0.12, label="Matched-rate window (8 ± 1 nat)")
    ax.axvline(9.0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Source self ΔG (nat)")
    ax.set_ylabel("Held-out bystander mean ΔG (nat)")
    ax.set_title(
        "LoRA vs Full-FT marker leakage at matched source-implant strength (#514 follow-up)"
    )
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    _try_savefig_paper(fig, output_path)
    plt.close(fig)


def _ci_from_bootstrap_array(samples: list[float]) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) — 95% percentile CI from a bootstrap sample list."""
    import math

    if not samples:
        return (float("nan"), float("nan"), float("nan"))
    finite = [float(x) for x in samples if isinstance(x, int | float) and math.isfinite(float(x))]
    if not finite:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(finite) / len(finite)
    s = sorted(finite)
    n = len(s)
    lo = s[max(0, int(0.025 * n))]
    hi = s[min(n - 1, int(0.975 * n))]
    return (mean, lo, hi)


def matched_rate_figure(
    *,
    matched_rate_json: dict | None,
    bootstrap_arrays_json: dict | None,
    output_path: Path,
) -> None:
    """Grouped bars at source ΔG = 8 nat — LoRA / #508 FT / #514 FT (B5 round-2 fix).

    Reads:
      - ``matched_rate_json`` (``_matched_rate_514.json``): determinacy gate
        + #514 local linear-interpolation read at 8 nat.
      - ``bootstrap_arrays_json`` (``eval_results/issue_508/_matched_rate_bootstrap.json``):
        per-replicate bootstrap arrays for LoRA (``lora_at_8``) + #508 FT
        (``ft_at_8``). 95% CI derived per-arm from these arrays.

    The figure: 3 bars at source ΔG = 8 nat — LoRA / #508 FT / #514 FT —
    with cluster-bootstrap 95% CI error bars. Annotates the determinacy gate
    verdict from ``matched_rate_json`` (determinate vs gap=X nat; flags
    is_extrapolation when bracketing anchors don't straddle 8 nat).

    When H1 fails (no clean #514 FT cell above 9 nat), draws a placeholder
    annotation instead — the bars only make sense once at least one #514 FT
    cell hits the bracketing window.
    """
    import math

    import matplotlib.pyplot as plt

    _set_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    if matched_rate_json is None or not matched_rate_json.get("h1_pass"):
        ax.text(
            0.5,
            0.5,
            "Matched-rate read INDETERMINATE\n(no clean #514 FT cell above 9 nat)",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Bystander leakage at source ΔG = 8 nat (matched-rate) — H1 FAIL")
        _try_savefig_paper(fig, output_path)
        plt.close(fig)
        return

    # H1 PASS path — render the real 3-bar figure.
    bar_specs: list[tuple[str, float, float, float, str]] = []  # (label, mean, lo, hi, color)

    # LoRA bar.
    lora_arr = (bootstrap_arrays_json or {}).get("lora_at_8") or []
    lora_mean, lora_lo, lora_hi = _ci_from_bootstrap_array(lora_arr)
    bar_specs.append(("LoRA\n(#508 ref)", lora_mean, lora_lo, lora_hi, "#E69F00"))

    # #508 FT bar.
    ft508_arr = (bootstrap_arrays_json or {}).get("ft_at_8") or []
    ft508_mean, ft508_lo, ft508_hi = _ci_from_bootstrap_array(ft508_arr)
    bar_specs.append(("Full FT\n(#508 anchor)", ft508_mean, ft508_lo, ft508_hi, "#56B4E9"))

    # #514 FT bar — local linear-interpolation read; no per-arm bootstrap
    # array (the delegated cluster bootstrap mixes #514 + #508 cells in
    # the FT arm — that's the bootstrap_read used for the determinacy gate
    # but it's not a #514-only CI). Show the point estimate.
    local_read = matched_rate_json.get("local_read_nat")
    bootstrap_read = matched_rate_json.get("bootstrap_read_nat")
    if local_read is None or not math.isfinite(float(local_read)):
        local_read_val = float("nan")
    else:
        local_read_val = float(local_read)
    bar_specs.append(
        ("Full FT\n(#514 local read)", local_read_val, local_read_val, local_read_val, "#0072B2")
    )

    xs = list(range(len(bar_specs)))
    means = [b[1] for b in bar_specs]
    los = [b[1] - b[2] for b in bar_specs]
    his = [b[3] - b[1] for b in bar_specs]
    colors = [b[4] for b in bar_specs]
    labels = [b[0] for b in bar_specs]

    ax.bar(xs, means, color=colors, edgecolor="black", linewidth=0.5)
    # Error bars: only draw where lo != hi (i.e. real CI exists).
    yerr = [[lo if lo > 0 else 0 for lo in los], [hi if hi > 0 else 0 for hi in his]]
    ax.errorbar(xs, means, yerr=yerr, fmt="none", ecolor="black", capsize=5, linewidth=1.0)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Held-out bystander mean ΔG at source ΔG = 8 nat (nat)")
    ax.set_title("Bystander leakage at matched source-implant strength (8 nat)")

    # Determinacy + extrapolation annotation.
    determinate = matched_rate_json.get("determinate", False)
    gap_nat = matched_rate_json.get("gap_nat")
    gate_thresh = matched_rate_json.get("gate_threshold_nat", 0.5)
    is_extrap = matched_rate_json.get("is_extrapolation", False)

    if gap_nat is None or not math.isfinite(float(gap_nat)):
        gap_str = "gap = NaN"
    else:
        gap_str = f"gap = {float(gap_nat):.3f} nat"
    bootstrap_str = (
        f"{float(bootstrap_read):.3f}"
        if (bootstrap_read is not None and math.isfinite(float(bootstrap_read)))
        else "NaN"
    )
    verdict = "DETERMINATE" if determinate else "INDETERMINATE"
    extrap_note = " (EXTRAPOLATION)" if is_extrap else ""
    annot = (
        f"Determinacy: {verdict}{extrap_note} "
        f"(threshold |Δ| ≤ {gate_thresh} nat)\n"
        f"Local read: {local_read_val:.3f} nat | Bootstrap read: {bootstrap_str} nat | {gap_str}"
    )
    ax.text(
        0.5,
        -0.18,
        annot,
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
        family="monospace",
    )

    _try_savefig_paper(fig, output_path)
    plt.close(fig)


def rcollapse_figure(
    *,
    ft_508_cells: dict[str, dict],
    ft_514_cells: dict[str, dict],
    output_path: Path,
) -> None:
    """Source r-collapse rate per cell — #508 FT anchors + 6 new #514 FT cells."""
    import matplotlib.pyplot as plt

    from explore_persona_space.experiments.full_ft_regime_514.abort_logic import (
        compute_source_r_collapse_rate,
    )

    _set_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    cells = list(ft_508_cells.items()) + list(ft_514_cells.items())
    rates: list[tuple[str, float, float]] = []
    for cell, ej in cells:
        rcoll = compute_source_r_collapse_rate(ej)
        x, _ = _cell_xy(ej)
        rates.append((cell, x, rcoll))

    rates.sort(key=lambda r: r[1] if r[1] == r[1] else 0.0)

    xs = list(range(len(rates)))
    ys = [r[2] for r in rates]
    labels = [r[0] for r in rates]
    colors = ["#56B4E9" if r[0] in FT_508_ANCHOR_CELLS else "#0072B2" for r in rates]

    ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="abort threshold (0.50)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Source r-collapse rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Source-probe r-collapse rate per cell (#514 + #508 FT anchors)")
    ax.legend(loc="upper left", fontsize=9)

    _try_savefig_paper(fig, output_path)
    plt.close(fig)


def per_persona_figure(
    *,
    ft_514_cells: dict[str, dict],
    output_path: Path,
) -> None:
    """15 personas × cells held-out ΔG heatmap for the 6 new FT cells."""
    import matplotlib.pyplot as plt

    _set_style()
    if not ft_514_cells:
        LOG.warning("[plot] no #514 cells for per_persona — skipping")
        return

    # Aggregate per-persona held-out ΔG mean per cell.
    persona_means: dict[str, dict[str, float]] = {}
    for cell, ej in ft_514_cells.items():
        dg = ej.get("delta_g_held_out", {}) or {}
        for persona, q_map in dg.items():
            vals = [v["delta_g"] for v in q_map.values() if not v.get("r_collapsed")]
            if vals:
                persona_means.setdefault(persona, {})[cell] = sum(vals) / len(vals)

    personas = sorted(persona_means)
    cells = sorted(ft_514_cells)
    import numpy as np

    matrix = np.full((len(personas), len(cells)), float("nan"))
    for i, p in enumerate(personas):
        for j, c in enumerate(cells):
            v = persona_means.get(p, {}).get(c)
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=-5, vmax=15)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels(personas, fontsize=8)
    fig.colorbar(im, ax=ax, label="Held-out ΔG (nat)")
    ax.set_title("Per-persona × per-cell held-out ΔG (#514 FT cells)")

    _try_savefig_paper(fig, output_path)
    plt.close(fig)


def source_self_trajectory_figure(
    *,
    ft_514_cells: dict[str, dict],
    output_path: Path,
) -> None:
    """On-policy source-self log P(※) trajectory for each of the 6 new FT cells.

    Reads the per-cell dynamics sidecar from the eval JSON's
    ``dynamics_snapshots`` field (written by the offline post-checkpoint
    extractor — same shape as #508's trajectory_dg_path). Falls back to
    plotting a single endpoint marker if no snapshots are available.
    """
    import matplotlib.pyplot as plt

    _set_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    plotted = 0
    for cell, ej in ft_514_cells.items():
        snaps = ej.get("dynamics_snapshots") or []
        if not snaps:
            x, y = _cell_xy(ej)
            if x == x:
                ax.scatter(1.0, y, label=f"{cell} (endpoint)", s=50)
                plotted += 1
            continue
        xs = [s.get("epoch_fraction", s.get("step", i)) for i, s in enumerate(snaps)]
        ys = [s.get("source_self_log_p_marker", float("nan")) for s in snaps]
        ax.plot(xs, ys, marker="o", label=cell)
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No dynamics snapshots available", ha="center", transform=ax.transAxes)

    ax.set_xlabel("Epoch fraction (or step proxy)")
    ax.set_ylabel("Source-self log P(※)")
    ax.set_title("Source-self marker log-probability trajectory (#514 FT cells)")
    ax.legend(loc="best", fontsize=8)

    _try_savefig_paper(fig, output_path)
    plt.close(fig)


def default_assistant_figure(
    *,
    all_ft_cells: dict[str, dict],
    output_path: Path,
) -> None:
    """Bare default-assistant ΔG vs source ΔG with matched-rate window highlighted."""
    import matplotlib.pyplot as plt

    _set_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    pts: list[tuple[str, float, float]] = []
    for cell, ej in all_ft_cells.items():
        x, _ = _cell_xy(ej)
        qd = (ej.get("aggregates") or {}).get("qwen_default_mean_delta_g")
        if x == x and qd is not None and isinstance(qd, int | float):
            pts.append((cell, float(x), float(qd)))

    if pts:
        pts.sort(key=lambda p: p[1])
        for cell, x, y in pts:
            color = "#56B4E9" if cell in FT_508_ANCHOR_CELLS else "#0072B2"
            ax.scatter(x, y, color=color, s=60, edgecolors="black", linewidths=0.5)
            ax.annotate(cell, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.axvspan(7.0, 9.0, color="gray", alpha=0.12, label="Matched-rate window (8 ± 1 nat)")
    ax.axhline(-5.0, color="red", linestyle="--", linewidth=0.8, label="Sub-ceiling gate (-5 nat)")
    ax.set_xlabel("Source self ΔG (nat)")
    ax.set_ylabel("Default-assistant ΔG (nat)")
    ax.set_title("Default-assistant slice vs source rate (#508 caveat: tightest sub-ceiling)")
    ax.legend(loc="best", fontsize=9)

    _try_savefig_paper(fig, output_path)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="Generate #514 figures")
    p.add_argument(
        "--eval-root-508",
        type=Path,
        default=Path("eval_results/issue_508"),
        help="Path to #508 reference eval JSONs (LoRA + FT anchors)",
    )
    p.add_argument(
        "--eval-root-514",
        type=Path,
        default=Path("eval_results/issue_514"),
        help="Path to #514 FT cell eval JSONs",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/issue_514"),
        help="Output directory for figures",
    )
    p.add_argument(
        "--matched-rate-json",
        type=Path,
        default=None,
        help="Optional: path to _matched_rate_514.json",
    )
    p.add_argument(
        "--bootstrap-arrays-json",
        type=Path,
        default=None,
        help=(
            "Optional: path to _matched_rate_bootstrap.json (per-replicate "
            "arrays for LoRA + #508 FT at source ΔG = 8 nat). Default: "
            "eval_root_508 / _matched_rate_bootstrap.json"
        ),
    )
    args = p.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    lora = _load_cells(LORA_REFERENCE_CELLS, args.eval_root_508, args.eval_root_514)
    ft_508 = _load_cells(FT_508_ANCHOR_CELLS, args.eval_root_508, args.eval_root_514)
    ft_514_dense = _load_cells(FT_514_DENSE_CELLS, args.eval_root_508, args.eval_root_514)
    ft_514_lowlr = _load_cells(FT_514_LOWLR_CELLS, args.eval_root_508, args.eval_root_514)
    ft_514_all = {**ft_514_dense, **ft_514_lowlr}
    all_ft = {**ft_508, **ft_514_all}

    matched_rate_json = None
    mr_path = args.matched_rate_json or (args.eval_root_514 / "_matched_rate_514.json")
    if mr_path.exists():
        matched_rate_json = json.loads(mr_path.read_text())

    bootstrap_arrays_json = None
    ba_path = args.bootstrap_arrays_json or (args.eval_root_508 / "_matched_rate_bootstrap.json")
    if ba_path.exists():
        bootstrap_arrays_json = json.loads(ba_path.read_text())
    else:
        LOG.warning(
            "[plot] bootstrap arrays JSON not found at %s — matched_rate.png "
            "will fall back to placeholder (#514 FT bar will be the only one)",
            ba_path,
        )

    hero_figure(
        lora_cells=lora,
        ft_508_cells=ft_508,
        ft_514_dense_cells=ft_514_dense,
        ft_514_lowlr_cells=ft_514_lowlr,
        output_path=args.figures_dir / "hero",
    )
    matched_rate_figure(
        matched_rate_json=matched_rate_json,
        bootstrap_arrays_json=bootstrap_arrays_json,
        output_path=args.figures_dir / "matched_rate",
    )
    rcollapse_figure(
        ft_508_cells=ft_508,
        ft_514_cells=ft_514_all,
        output_path=args.figures_dir / "rcollapse",
    )
    per_persona_figure(
        ft_514_cells=ft_514_all,
        output_path=args.figures_dir / "per_persona",
    )
    source_self_trajectory_figure(
        ft_514_cells=ft_514_all,
        output_path=args.figures_dir / "source_self_trajectory",
    )
    default_assistant_figure(
        all_ft_cells=all_ft,
        output_path=args.figures_dir / "default_assistant",
    )

    LOG.info("[plot] all 6 #514 figures written to %s", args.figures_dir)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
