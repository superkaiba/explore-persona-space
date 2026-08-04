"""Phase-4 figures for issue #1739 (plan §6 set; round C1).

Conventions (paper-plots skill; `analysis/paper_plots.py` owns rcParams —
never hand-rolled styles):

- ``set_paper_style()`` once per figure; ``savefig_paper`` writes PNG + PDF +
  the data sidecar into the ``figures/issue_1739`` out-root.
- ONE COLOR = ONE ARM FAMILY across ALL figures (:data:`FAMILY_COLORS`, via
  ``paper_palette_role`` — context=primary, map=accent, oracle=neutral,
  control=control).
- Error bars use NON-NEGATIVE per-point OFFSETS clamped element-wise
  (``max(0, v-lo)`` / ``max(0, hi-v)`` — the #547/#1335 xerr/yerr class).
- Every aggregate figure has a per-context low-level companion
  (:func:`fig_percell_scatter`, labeled points).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_1739.arms import ARM_REGISTRY

logger = logging.getLogger(__name__)

DEFAULT_FIG_DIR = Path("figures/issue_1739")

# One color = one arm family, everywhere (paper_palette_role's valid names).
_FAMILY_ROLE = {"context": "primary", "map": "accent", "oracle": "neutral", "control": "control"}


def family_color(family: str) -> str:
    from explore_persona_space.analysis.paper_plots import paper_palette_role

    return paper_palette_role(_FAMILY_ROLE.get(family, "neutral"))


def _style():
    import matplotlib

    matplotlib.use("Agg", force=False)
    from explore_persona_space.analysis import paper_plots

    paper_plots.set_paper_style()
    return paper_plots


def _arm_label(slug: str) -> str:
    return ARM_REGISTRY.get(slug, {}).get("label", slug)


def _ci_yerr(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Non-negative (2, n) yerr OFFSETS from CI bounds (never raw bounds)."""
    return np.stack([np.maximum(0.0, vals - lo), np.maximum(0.0, hi - vals)])


def _group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


def _agg(rows: list[dict]) -> tuple[float, float, float]:
    """(mean rho_frozen, mean ci lo, mean ci hi) over a row group."""
    rho = float(np.nanmean([r["rho_frozen"] for r in rows]))
    lo = float(np.nanmean([r["ci_frozen"][0] for r in rows]))
    hi = float(np.nanmean([r["ci_frozen"][1] for r in rows]))
    return rho, lo, hi


def fig_hero_bars(
    arm_rows: list[dict],
    out_dir: Path | str = DEFAULT_FIG_DIR,
    *,
    stem: str = "hero_spearman_by_arm",
    title: str = "Spearman rho by arm",
) -> dict:
    """Hero bars: mean frozen-layer rho per arm, family-colored, CI whiskers."""
    pp = _style()
    import matplotlib.pyplot as plt

    by_arm = _group_rows(arm_rows, ("arm",))
    slugs = sorted(by_arm, key=lambda k: k[0])
    vals, los, his, colors, labels = [], [], [], [], []
    for (slug,) in slugs:
        rho, lo, hi = _agg(by_arm[(slug,)])
        vals.append(rho)
        los.append(lo)
        his.append(hi)
        fam = by_arm[(slug,)][0].get("family", "unknown")
        colors.append(family_color(fam))
        labels.append(_arm_label(slug))
    vals_a, los_a, his_a = np.array(vals), np.array(los), np.array(his)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, vals_a, color=colors, yerr=_ci_yerr(vals_a, los_a, his_a), capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho (frozen layer, pooled OOF)")
    ax.axhline(0.0, lw=0.8, color="0.4")
    ax.set_title(title)
    paths = pp.savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)
    return paths


def fig_scaling_curves(
    arm_rows: list[dict],
    out_dir: Path | str = DEFAULT_FIG_DIR,
    *,
    stem: str = "scaling_rho_vs_L",
    u_key: str = "u_rung",
    title: str = "Crossover scaling: rho vs labeled budget L",
) -> dict:
    """Crossover curves: rho vs L per arm, one panel per U rung, CI bands."""
    pp = _style()
    import matplotlib.pyplot as plt

    u_vals = sorted({r.get(u_key) for r in arm_rows}, key=lambda v: (v is None, v))
    fig, axes = plt.subplots(
        1, max(len(u_vals), 1), figsize=(5.2 * max(len(u_vals), 1), 4.0), squeeze=False
    )
    for ui, u in enumerate(u_vals):
        ax = axes[0][ui]
        rows_u = [r for r in arm_rows if r.get(u_key) == u]
        for (slug,), rows in sorted(_group_rows(rows_u, ("arm",)).items()):
            by_l = _group_rows(rows, ("budget_l",))
            ls = sorted(k[0] for k in by_l)
            agg = [_agg(by_l[(lv,)]) for lv in ls]
            vals = np.array([a[0] for a in agg])
            lo = np.array([a[1] for a in agg])
            hi = np.array([a[2] for a in agg])
            color = family_color(rows[0].get("family", "unknown"))
            ax.plot(ls, vals, marker="o", ms=3, color=color, label=_arm_label(slug), lw=1.2)
            ax.fill_between(ls, lo, hi, color=color, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("labeled budget L (log scale)")
        ax.set_ylabel("Spearman rho")
        ax.set_title(f"U = {u}")
    axes[0][-1].legend(fontsize=6, ncol=1, loc="best")
    fig.suptitle(title)
    paths = pp.savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)
    return paths


def fig_degradation_slope(
    arm_rows: list[dict],
    out_dir: Path | str = DEFAULT_FIG_DIR,
    *,
    stem: str = "degradation_rho_vs_rung",
    rung_key: str = "eval_rung",
    rung_order: list[str] | None = None,
    title: str = "Degradation across eval rungs",
) -> dict:
    """Degradation slope: rho vs eval rung, one line per arm."""
    pp = _style()
    import matplotlib.pyplot as plt

    rungs = rung_order or sorted({str(r.get(rung_key)) for r in arm_rows})
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for (slug,), rows in sorted(_group_rows(arm_rows, ("arm",)).items()):
        by_r = _group_rows(rows, (rung_key,))
        xs, vals = [], []
        for ri, rung in enumerate(rungs):
            grp = by_r.get((rung,)) or by_r.get((str(rung),))
            if grp:
                xs.append(ri)
                vals.append(_agg(grp)[0])
        if xs:
            ax.plot(
                xs,
                vals,
                marker="o",
                ms=3,
                lw=1.2,
                color=family_color(rows[0].get("family", "unknown")),
                label=_arm_label(slug),
            )
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([str(r) for r in rungs], rotation=20, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_title(title)
    ax.legend(fontsize=6)
    paths = pp.savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)
    return paths


def fig_map_degradation(
    diag_rows: list[dict],
    out_dir: Path | str = DEFAULT_FIG_DIR,
    *,
    stem: str = "map_degradation_diagnostic",
    title: str = "Map degradation: held-out R2 + kNN acc@1 per rung",
) -> dict:
    """Map diagnostics per eval rung: R2 (map vs identity+bias) + kNN acc@1.

    ``diag_rows``: one dict per rung — {"rung", "r2_map", "r2_identity_bias",
    "knn_acc1_euclidean", "knn_chance1"} (pooled over layers by the caller).
    """
    pp = _style()
    import matplotlib.pyplot as plt

    rungs = [str(r["rung"]) for r in diag_rows]
    x = np.arange(len(rungs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax1.plot(
        x,
        [r["r2_map"] for r in diag_rows],
        marker="o",
        color=family_color("map"),
        label="fitted map R2",
    )
    ax1.plot(
        x,
        [r["r2_identity_bias"] for r in diag_rows],
        marker="s",
        color=family_color("control"),
        label="identity+learned-bias R2",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(rungs, rotation=20, ha="right")
    ax1.set_ylabel("held-out R2")
    ax1.legend(fontsize=7)
    ax2.plot(
        x,
        [r["knn_acc1_euclidean"] for r in diag_rows],
        marker="o",
        color=family_color("map"),
        label="kNN acc@1 (euclidean)",
    )
    ax2.plot(
        x, [r["knn_chance1"] for r in diag_rows], ls="--", color="0.5", label="chance = 1/n_pool"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(rungs, rotation=20, ha="right")
    ax2.set_ylabel("retrieval acc@1")
    ax2.legend(fontsize=7)
    fig.suptitle(title)
    paths = pp.savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)
    return paths


def fig_composition(
    comp_rows: list[dict],
    out_dir: Path | str = DEFAULT_FIG_DIR,
    *,
    stem: str = "composition_factor",
    title: str = "Composition factor: rho vs f_U x f_L (evil, Config A)",
) -> dict:
    """Composition sub-experiment: rho per (f_U, f_L) at each L anchor.

    ``comp_rows``: {"f_u", "f_l", "budget_l", "rho"} per cell.
    """
    pp = _style()
    import matplotlib.pyplot as plt

    anchors = sorted({r["budget_l"] for r in comp_rows})
    fig, axes = plt.subplots(
        1, max(len(anchors), 1), figsize=(4.4 * max(len(anchors), 1), 3.8), squeeze=False
    )
    combos = sorted({(r["f_u"], r["f_l"]) for r in comp_rows})
    palette = [
        family_color("map"),
        family_color("context"),
        family_color("control"),
        family_color("oracle"),
    ]
    for ai, anchor in enumerate(anchors):
        ax = axes[0][ai]
        rows_a = [r for r in comp_rows if r["budget_l"] == anchor]
        vals = []
        for f_u, f_l in combos:
            grp = [r["rho"] for r in rows_a if (r["f_u"], r["f_l"]) == (f_u, f_l)]
            vals.append(float(np.nanmean(grp)) if grp else np.nan)
        x = np.arange(len(combos))
        ax.bar(x, vals, color=[palette[i % len(palette)] for i in range(len(combos))])
        ax.set_xticks(x)
        ax.set_xticklabels([f"f_U={fu}\nf_L={fl}" for fu, fl in combos], fontsize=7)
        ax.set_ylabel("Spearman rho")
        ax.set_title(f"L = {anchor}")
    fig.suptitle(title)
    paths = pp.savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)
    return paths


def fig_percell_scatter(
    scores: np.ndarray,
    dv: np.ndarray,
    labels: list[str],
    out_dir: Path | str = DEFAULT_FIG_DIR,
    *,
    stem: str = "percell_scatter",
    arm: str = "arm6_map_proj_e1",
    max_labels: int = 40,
    title: str | None = None,
) -> dict:
    """Low-level per-context scatter (labeled points) beside the aggregates."""
    pp = _style()
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=np.float64)
    dv = np.asarray(dv, dtype=np.float64)
    fam = ARM_REGISTRY.get(arm, {}).get("family", "unknown")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(scores, dv, s=14, color=family_color(fam), alpha=0.75)
    if len(labels) <= max_labels:
        for x, y, lab in zip(scores, dv, labels, strict=True):
            ax.annotate(
                str(lab), (x, y), fontsize=5, alpha=0.7, xytext=(2, 2), textcoords="offset points"
            )
    ax.set_xlabel(f"{_arm_label(arm)} score")
    ax.set_ylabel("graded DV (0-100)")
    ax.set_title(title or f"Per-context scores vs DV — {_arm_label(arm)}")
    paths = pp.savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)
    return paths


def render_summary_figures(summary: dict, out_dir: Path | str = DEFAULT_FIG_DIR) -> list[Path]:
    """Render every figure derivable from ``all_arms_spearman.json`` alone."""
    rows = summary.get("arm_rows", [])
    if not rows:
        raise ValueError("summary has no arm_rows — nothing to plot")
    out: list[Path] = []
    out += list(fig_hero_bars(rows, out_dir).values())
    if len({r.get("budget_l") for r in rows}) > 1:
        out += list(fig_scaling_curves(rows, out_dir).values())
    if len({r.get("eval_rung") for r in rows if r.get("eval_rung") is not None}) > 1:
        out += list(fig_degradation_slope(rows, out_dir).values())
    # Round-3 M-A: the §6.5 distribution-shift ladder — TRAIN-frozen
    # predictors scored per eval rung (transfer_rows), anchored by the
    # in-split OOF read (rung_kind train_in_split, ordered first).
    t_rows = summary.get("transfer_rows") or []
    if len({str(r.get("eval_rung")) for r in t_rows}) > 1:
        train_rungs = sorted(
            {str(r.get("eval_rung")) for r in t_rows if r.get("rung_kind") == "train_in_split"}
        )
        eval_rungs = sorted(
            {str(r.get("eval_rung")) for r in t_rows if r.get("rung_kind") != "train_in_split"}
        )
        out += list(
            fig_degradation_slope(
                t_rows,
                out_dir,
                stem="distribution_shift_ladder",
                rung_order=train_rungs + [r for r in eval_rungs if r not in train_rungs],
                title="Distribution-shift ladder: rho vs eval rung (train-frozen predictors)",
            ).values()
        )
    # Round-3 M-A sweep item (d): composition rows ride arm_rows — render the
    # §4b composition figure whenever >1 (f_U, f_L) combo is present.
    comp = [
        {"f_u": r["f_u"], "f_l": r["f_l"], "budget_l": r["budget_l"], "rho": r["rho_frozen"]}
        for r in rows
        if r.get("f_u") is not None
    ]
    if len({(c["f_u"], c["f_l"]) for c in comp}) > 1:
        out += list(fig_composition(comp, out_dir).values())
    logger.info("[figures] rendered %d files -> %s", len(out), out_dir)
    return [Path(p) for p in out]
