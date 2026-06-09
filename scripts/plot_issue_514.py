# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + nat char + multiplication-sign + minus intentional
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
#
# B7 round-3 fix: this is the static fallback name-list checked by the hero
# figure for half-transparency rendering. The canonical "is this anchor
# clean?" gate lives in
# ``explore_persona_space.experiments.full_ft_regime_514.analyze.is_clean_anchor``
# (used by analyze.py's local-read bracketing logic + dynamically computed
# here via :func:`compute_excluded_cells`). The static tuple stays so plot
# rendering works when called before any analyze-side diagnostics are loaded
# (e.g. when no #514 eval JSONs are present yet).
EXCLUDED_FROM_BOOTSTRAP = ("ft_b2",)


def compute_excluded_cells(eval_jsons_by_cell: dict[str, dict]) -> tuple[str, ...]:
    """B7 round-3 fix: derive the per-cell exclusion set from is_clean_anchor.

    Walks each loaded cell's eval JSON, builds a per-cell diagnostic dict in
    the same shape :func:`is_clean_anchor` expects, and returns the tuple of
    cell slugs that FAIL the clean-anchor gate. The plot's hero figure may
    use this for the alpha-channel rendering; both the analyze-side local
    read and the plot-side exclusion now share a single rule.

    Falls back to :data:`EXCLUDED_FROM_BOOTSTRAP` when the analyze module
    cannot be imported (defensive — keeps the plot working in stripped
    environments).
    """
    try:
        from explore_persona_space.experiments.full_ft_regime_514.abort_logic import (
            compute_source_r_collapse_rate,
            get_held_out_g_logprob_mean,
        )
        from explore_persona_space.experiments.full_ft_regime_514.analyze import (
            is_clean_anchor,
        )
    except ImportError:
        return EXCLUDED_FROM_BOOTSTRAP

    excluded: list[str] = []
    for cell, ej in eval_jsons_by_cell.items():
        agg = ej.get("aggregates", {}) or {}
        diag = {
            "source_mean": agg.get("source_self_mean_delta_g"),
            "held_out_mean": agg.get("held_out_mean_delta_g"),
            "source_n_probes": agg.get("source_n_probes"),
            "r_collapse_rate": compute_source_r_collapse_rate(ej),
            "held_out_g_logprob_mean": get_held_out_g_logprob_mean(ej),
        }
        if not is_clean_anchor(diag):
            excluded.append(cell)
    return tuple(excluded) or EXCLUDED_FROM_BOOTSTRAP


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


def _source_endpoint_y(ej: dict) -> float:
    """Return the source-self ΔG aggregate to use as the trajectory endpoint
    y-coordinate. The source_self_trajectory plot's y-axis is "Source-self
    ΔG (nat)", so the endpoint fallback must read the source aggregate, not
    the held-out aggregate.

    B12 round-4 pivot: the round-3 endpoint fallback did
    ``_, y = _cell_xy(ej); ax.scatter(1.0, y, ...)`` — that y was
    ``held_out_mean_delta_g``, plotted on a "Source-self ΔG" axis, off by
    several nat per cell. This helper makes the correct axis explicit.
    """
    return _cell_xy(ej)[0]


def _try_savefig_paper(fig, out_path: Path) -> None:
    """Save via paper_plots.savefig_paper if available; fall back to plain savefig.

    ``savefig_paper(fig, stem, dir=...)`` joins ``dir / stem``; pass the
    filename stem and parent dir SEPARATELY so the output lands at
    ``out_path.{png,pdf}`` and not at ``figures/<out_path>`` (the default
    ``dir="figures/"`` double-prefixed when ``out_path`` was passed as the
    stem — that wrote every figure to ``figures/figures/issue_514/``).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from explore_persona_space.analysis.paper_plots import savefig_paper

        savefig_paper(fig, stem=out_path.name, dir=out_path.parent)
    except ImportError:
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


def _draw_lever_with_exclusion(
    ax,
    cells: dict[str, dict],
    *,
    excluded: tuple[str, ...],
    marker: str,
    color: str,
    label: str,
) -> None:
    """Render a #514 lever (dense or lower-LR) with per-cell exclusion alpha.

    B13 round-2 pivot: the round-4 hero_figure rendering called ``ax.plot``
    over a bulk (xs, ys) list, which applied a SINGLE alpha to every point in
    the lever. The ``excluded`` parameter is computed from ALL loaded cells
    via :func:`compute_excluded_cells`, including #514 cells, so a #514 cell
    that fails :func:`is_clean_anchor` SHOULD render at half transparency.

    This helper partitions the lever's cells into clean / excluded, draws the
    connecting line ONLY across the clean cells (the trend the bootstrap
    actually sees), and overlays each excluded cell as a dimmed marker point
    with no line through it. The legend label is attached to the clean line
    if it exists, otherwise to the first excluded marker (so an all-excluded
    lever still appears in the legend).
    """
    pairs: list[tuple[str, float, float]] = []
    for cell, ej in cells.items():
        x, y = _cell_xy(ej)
        if x != x or y != y:
            continue
        pairs.append((cell, x, y))
    if not pairs:
        return
    pairs.sort(key=lambda p: p[1])  # sort by x
    clean = [(c, x, y) for c, x, y in pairs if c not in excluded]
    dirty = [(c, x, y) for c, x, y in pairs if c in excluded]

    label_consumed = False
    if clean:
        xs = [x for _, x, _ in clean]
        ys = [y for _, _, y in clean]
        ax.plot(
            xs,
            ys,
            marker=marker,
            color=color,
            linewidth=1.6,
            alpha=1.0,
            label=label,
        )
        label_consumed = True
    for _cell, x, y in dirty:
        ax.scatter(
            x,
            y,
            marker=marker,
            s=60,
            color=color,
            alpha=0.4,
            edgecolors="black",
            linewidths=0.5,
            label=None if label_consumed else label,
        )
        label_consumed = True


def hero_figure(
    *,
    lora_cells: dict[str, dict],
    ft_508_cells: dict[str, dict],
    ft_514_dense_cells: dict[str, dict],
    ft_514_lowlr_cells: dict[str, dict],
    output_path: Path,
    excluded: tuple[str, ...] = EXCLUDED_FROM_BOOTSTRAP,
) -> None:
    """Source-rate curve: LoRA reference + #508 FT anchors + 6 new #514 FT cells.

    B11 round-4 pivot: ``excluded`` parameter accepts the dynamically computed
    exclusion set from :func:`compute_excluded_cells` (which gates each loaded
    cell through the canonical :func:`is_clean_anchor` rule). Cells in the
    exclusion set are drawn at half transparency to signal the cluster
    bootstrap drops them. The default :data:`EXCLUDED_FROM_BOOTSTRAP` is the
    static fallback used when no eval JSONs are available.
    """
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

    # #508 FT anchors (light blue squares; cells in ``excluded`` or ft_b3 at
    # half transparency). ft_b3 is always dimmed because it's the #508
    # 1.0-epoch collapsed cell (20/20 r-collapsed source probes) and is kept
    # in the hero figure as visual context but is never a valid anchor.
    for cell, ej in ft_508_cells.items():
        x, y = _cell_xy(ej)
        if x != x or y != y:
            continue
        alpha = 0.4 if cell in excluded or cell == "ft_b3" else 1.0
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

    # Dense lever (medium blue). B13 round-2 pivot: render the connecting line
    # only across clean cells, then overlay each excluded cell as a half-alpha
    # marker point. The plot.line is drawn at alpha=1.0 over clean points; an
    # excluded cell is shown as an isolated dimmed marker (no line through it)
    # so the reader sees that the bootstrap window drops it without breaking
    # the cross-cell trend across the remaining clean cells.
    _draw_lever_with_exclusion(
        ax,
        ft_514_dense_cells,
        excluded=excluded,
        marker="s",
        color="#0072B2",
        label="Full FT, dense lever (#514)",
    )

    # Lower-LR lever (dark blue). Same per-cell exclusion treatment.
    _draw_lever_with_exclusion(
        ax,
        ft_514_lowlr_cells,
        excluded=excluded,
        marker="^",
        color="#1F2A5A",
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

    # H1 PASS path — render the matched-rate LoRA-vs-FT comparison at 8 nat.
    #
    # Free-reanalysis (2026-06-08): the headline at the matched rate is the
    # LoRA−FT gap, now a TRUE interpolation on both arms (the clean lower-LR
    # FT cell at 7.43 nat brackets target=8.0 with #508 ft_b1 at 8.20 nat),
    # so this figure shows the two per-arm reads + the crossed-cluster-bootstrap
    # gap with its 95% CI — self-contained from `_matched_rate_514.json`
    # (`_compute_matched_rate_gap_514`), no external bootstrap-arrays file.
    lora_read = matched_rate_json.get("matched_rate_lora_read_nat")
    ft_read = matched_rate_json.get("matched_rate_ft_read_nat")
    gap_mean = matched_rate_json.get("matched_rate_gap_ft_minus_lora_nat")
    gap_lo = matched_rate_json.get("matched_rate_gap_ci_lo_nat")
    gap_hi = matched_rate_json.get("matched_rate_gap_ci_hi_nat")
    excludes_zero = matched_rate_json.get("matched_rate_gap_excludes_zero")
    n_rep = matched_rate_json.get("matched_rate_bootstrap_n_replicates")

    # Fall back to the local read for the FT bar if the gap helper was skipped
    # (<2 clean anchors in an arm) — keeps the figure honest in that degenerate
    # case rather than drawing a NaN bar.
    if ft_read is None or not math.isfinite(float(ft_read)):
        _lr = matched_rate_json.get("local_read_nat")
        ft_read = float(_lr) if (_lr is not None and math.isfinite(float(_lr))) else float("nan")
    lora_val = float(lora_read) if (lora_read is not None) else float("nan")
    ft_val = float(ft_read)

    xs = [0, 1]
    means = [lora_val, ft_val]
    colors = ["#E69F00", "#0072B2"]
    labels = ["LoRA", "Full fine-tune"]
    ax.bar(xs, means, width=0.55, color=colors, edgecolor="black", linewidth=0.5)
    for x, m in zip(xs, means, strict=True):
        if math.isfinite(m):
            ax.text(x, m + 0.08, f"{m:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Held-out bystander mean ΔG at source ΔG = 8 nat (nat)")
    ax.set_title("Bystander leakage at matched source-implant strength (8 nat)")
    top = max([m for m in means if math.isfinite(m)] + [0.0])
    ax.set_ylim(0, top * 1.35 if top > 0 else 1.0)

    # Determinacy verdict + the scientific LoRA−FT gap with its bootstrap CI.
    determinate = matched_rate_json.get("determinate", False)
    is_extrap = matched_rate_json.get("is_extrapolation", False)
    verdict = "DETERMINATE" if determinate else "INDETERMINATE"
    extrap_note = " (EXTRAPOLATION)" if is_extrap else " (true interpolation)"
    if gap_mean is None or not math.isfinite(float(gap_mean)):
        gap_line = "LoRA − FT gap: unavailable"
    else:
        ci_str = (
            f"95% CI [{float(gap_lo):+.2f}, {float(gap_hi):+.2f}]"
            if (gap_lo is not None and gap_hi is not None)
            else "CI unavailable"
        )
        sig = "significant" if excludes_zero else "not significant (CI spans 0)"
        rep_str = f", {int(n_rep)} reps" if n_rep else ""
        gap_line = f"LoRA − FT gap = {float(gap_mean):+.2f} nat, {ci_str} — {sig}{rep_str}"
    annot = f"Determinacy: {verdict}{extrap_note}\n{gap_line}"
    ax.text(
        0.5,
        -0.16,
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


def _load_dynamics_snapshots(ej: dict) -> list[dict]:
    """B8 round-3 fix: load + normalize per-cell dynamics snapshots.

    The dispatcher's Phase-1 offline extractor writes a SIDECAR JSON to
    ``<cell_dir>/dynamics.json`` and stamps its path into the eval JSON
    under ``dynamics_snapshots_path`` (NOT an inline ``dynamics_snapshots``
    list — that was the round-2 plot bug; the producer-consumer key mismatch
    silently degraded every FT trajectory to endpoint-only).

    Returns a list of normalized snapshot dicts (one per saved fraction):
        {"step": int, "source_delta_g": float, ...}

    The sidecar schema (``extract_fullft_dynamics_from_checkpoints``) is:
        {"schema_version": "i508_dynamics_v1",
         "snapshots": {"<str_step>": {<flat metrics>, "step": int, "n_probes": int}}}
    where each snap's metrics use namespaced keys
    (``dynamics/source_delta_g`` / ``dynamics/source_emission_rate``).

    Returns ``[]`` when no sidecar is reachable OR the JSON is malformed —
    the caller falls back to the endpoint-marker presentation.
    """
    path_str = ej.get("dynamics_snapshots_path")
    # Backwards-compat: tolerate an inline ``dynamics_snapshots`` field if
    # ever populated (e.g. a future LoRA-arm path persisting the callback
    # snapshots dict to disk inline). Sidecar takes priority.
    if not path_str:
        inline = ej.get("dynamics_snapshots") or []
        return list(inline) if isinstance(inline, list) else []
    snap_path = Path(path_str)
    if not snap_path.exists():
        return []
    try:
        payload = json.loads(snap_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, dict) and "snapshots" in payload:
        snaps = payload["snapshots"]
    else:
        snaps = payload
    if isinstance(snaps, dict):
        # Keyed by str(step). Sort by int(step) ascending.
        rows: list[dict] = []
        for step_key, row in sorted(snaps.items(), key=lambda kv: int(kv[0])):
            if isinstance(row, dict):
                rows.append({"step": int(step_key), **row})
        return rows
    if isinstance(snaps, list):
        return [s for s in snaps if isinstance(s, dict)]
    return []


def _snap_y(snap: dict) -> float:
    """B8 round-3 fix: pull the ΔG-based source-self proxy from a snapshot.

    The sidecar's namespaced flat key is ``dynamics/source_delta_g`` (set by
    ``_snapshot_from_per_probe`` in marker_dynamics_callback.py). Accept the
    bare-name form ``source_delta_g`` too (the analyze-side normalizer in
    #508 also accepts both). Returns NaN when neither key is present.

    We're plotting ΔG (= log P_trained(marker) − log P_base(marker)) — this
    is the on-policy source-self marker-implant strength as a function of
    training step, NOT the bare ``log P(marker)`` the round-2 plot tried to
    read under a key that never existed in either schema.
    """
    for k in ("dynamics/source_delta_g", "source_delta_g"):
        v = snap.get(k)
        if isinstance(v, int | float):
            return float(v)
    return float("nan")


def _snap_x(snap: dict, default_idx: int) -> float:
    """B8 round-3 fix: pull the x-axis training-step value from a snapshot.

    Sidecar snapshots are keyed by ``step`` (global_step int). Accept
    ``epoch_fraction`` as a fallback for future schemas that elect to
    store that instead, then ``default_idx`` (the position in the sorted
    list) as the last resort.
    """
    for k in ("step", "epoch_fraction"):
        v = snap.get(k)
        if isinstance(v, int | float):
            return float(v)
    return float(default_idx)


def source_self_trajectory_figure(
    *,
    ft_514_cells: dict[str, dict],
    output_path: Path,
) -> None:
    """On-policy source-self ΔG trajectory for each of the 6 new FT cells.

    B8 round-3 fix: reads the per-cell dynamics sidecar from
    ``eval_json["dynamics_snapshots_path"]`` (the path stamped by Phase 2
    eval after Phase 1's offline extractor) — NOT an inline
    ``dynamics_snapshots`` list. Falls back to a single endpoint marker
    when no snapshots are reachable.
    """
    import matplotlib.pyplot as plt

    _set_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    plotted = 0
    for cell, ej in ft_514_cells.items():
        snaps = _load_dynamics_snapshots(ej)
        if not snaps:
            # B12 round-4 pivot: y-axis is "Source-self ΔG (nat)" — use the
            # source aggregate, NOT _cell_xy's held-out second-element.
            y_end = _source_endpoint_y(ej)
            if y_end == y_end:
                ax.scatter(1.0, y_end, label=f"{cell} (endpoint)", s=50)
                plotted += 1
            continue
        xs = [_snap_x(s, i) for i, s in enumerate(snaps)]
        ys = [_snap_y(s) for s in snaps]
        # Drop NaN pairs so a partially-populated snapshot doesn't kill the
        # line. (Defensive — extractor's _snapshot_from_per_probe raises
        # on missing source probes, so this should be rare in practice.)
        pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if y == y]
        if len(pairs) < 2:
            # Treat single-snapshot cells as endpoint-only (matches the
            # no-snapshot branch above so the legend stays interpretable).
            # B12 round-4 pivot: same axis-correction as above.
            y_end = _source_endpoint_y(ej)
            if y_end == y_end:
                ax.scatter(1.0, y_end, label=f"{cell} (endpoint)", s=50)
                plotted += 1
            continue
        xs2, ys2 = zip(*pairs, strict=True)
        ax.plot(xs2, ys2, marker="o", label=cell)
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No dynamics snapshots available", ha="center", transform=ax.transAxes)

    ax.set_xlabel("Training step (global)")
    ax.set_ylabel("Source-self ΔG (nat)")
    ax.set_title("Source-self marker-implant trajectory (#514 FT cells)")
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

    # B11 round-4 pivot: derive the per-cell exclusion set from the canonical
    # is_clean_anchor gate via compute_excluded_cells (instead of the static
    # EXCLUDED_FROM_BOOTSTRAP tuple). Walks every loaded #508 anchor + #514
    # cell so a collapsed #514 anchor would also dim correctly.
    excluded_cells = compute_excluded_cells({**ft_508, **ft_514_all})
    LOG.info("[plot] hero figure exclusion set: %s", excluded_cells)

    hero_figure(
        lora_cells=lora,
        ft_508_cells=ft_508,
        ft_514_dense_cells=ft_514_dense,
        ft_514_lowlr_cells=ft_514_lowlr,
        output_path=args.figures_dir / "hero",
        excluded=excluded_cells,
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
