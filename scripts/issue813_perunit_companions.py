# ruff: noqa: RUF001
"""Issue #813 — per-unit companion figures for two aggregate results (Lens 11).

1. pairwise_diff_per_family — the per-unit data behind the family-clustered
   pairwise forest (`pairwise_diff_forest.png`): per-context paired Δ/floor
   differences (points, colored by battery family) + per-family medians
   (diamonds) + the committed full-battery point estimate (circle), per
   behavior x substrate pair. Per-context normalized changes are recomputed
   from the persisted reduced summaries with the SAME fit helpers the
   committed read used (`issue722_fit_M._pca_basis_v0` / `_ridge_fit_predict`,
   the `_pseudo_delta_over_floor` numerator), then divided by the committed
   cell floor (delta_med / delta_over_floor — exact algebra on committed
   numbers). Gate: the recomputed per-cell median must reproduce the
   committed `delta_med` within 2% rel (the round's committed-map tolerance).

2. dv4_per_context_points — the per-unit data behind the within-context
   incremental R^2 bars (`dv4_query_specific.png`): per-held-out-context
   centered R^2 (the exact per-fold terms dv4 pools), recomputed via the
   driver's own `grouped_loco` (same folds, same PRESS-lambda grid). Gate:
   the pooled value from the per-fold sums must reproduce the committed
   `r2_within_observed` within 1e-4 abs.

Zero GPU; reads only committed JSONs + the locally cached reduced NPZs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import issue722_fit_M as fitM  # noqa: E402
import issue813_per_example_maps as pe  # noqa: E402
from issue813_pe_figures import ctx_label  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

RES = ROOT / "eval_results/issue_813"
OUT = ROOT / "figures/issue_813"
REDUCED_ROOT = RES / "reduced"
MAPS_ROOT = ROOT / "eval_results/issue_813_maps_pinned"
PE_DIR = RES / "per_example_vs_averaged"

BEHAVIORS = ["em", "fact", "sycophancy", "marker"]
SUBSTRATES = ["generic", "elicit", "mix"]
BEH_LABEL = {
    "em": "emergent misalignment",
    "fact": "fact",
    "sycophancy": "sycophancy",
    "marker": "marker",
}
PAIR_LABEL = {
    "generic_vs_elicit": "generic − eliciting",
    "generic_vs_mix": "generic − mix",
    "elicit_vs_mix": "eliciting − mix",
}
TICK_LABEL = {
    "generic": "generic\nUltraChat",
    "elicit": "behavior-\neliciting",
    "mix": "mixed\npool",
}
FAMILY_LABEL = {
    "persona": "persona",
    "wildchat": "WildChat",
    "icl": "ICL demo",
    "rephrase": "rephrase",
    "format": "format",
    "behavior": "behavior instr.",
    "default": "default",
}
FAMILIES = list(FAMILY_LABEL)


def per_context_delta_over_floor(cell: pe.CellData, r_hat: np.ndarray | None) -> np.ndarray:
    """Per-context normalized map change |Δ_i|/floor — the committed numerator recipe."""
    v0 = cell.avg_v["base"]
    pca_basis = fitM._pca_basis_v0(v0, pe.TARGET_DIM)
    v0_64 = fitM._to64(v0, pca_basis)
    vplus_64 = fitM._to64(cell.avg_v["trained"], pca_basis)
    c0 = cell.avg_c["base"]
    m0_grid = fitM._ridge_fit_predict(c0, v0_64, c0)
    mplus_grid = fitM._ridge_fit_predict(cell.avg_c["trained"], vplus_64, c0)
    delta_full = (mplus_grid - m0_grid) @ pca_basis
    if r_hat is None:  # marker read-1: unprojected vector norm
        return np.linalg.norm(delta_full, axis=1)
    return np.abs(delta_full @ r_hat)


def fig_pairwise_per_family(
    summary: dict, dof_by_cell: dict[tuple[str, str], dict[str, float]]
) -> None:
    fam_colors = dict(zip(FAMILIES, paper_palette_blog(len(FAMILIES)), strict=True))
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), constrained_layout=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        rows = summary["per_behavior"][beh]["pairwise_substrate_diff"]
        ys = np.arange(len(rows))[::-1]
        for y, r in zip(ys, rows, strict=True):
            sub_a, _, sub_b = r["pair"].partition("_vs_")
            da = dof_by_cell[(beh, sub_a)]
            db = dof_by_cell[(beh, sub_b)]
            fams = dof_by_cell[(beh, sub_a)]["_fams"]
            shared = [c for c in da if not c.startswith("_") and c in db]
            diffs = np.array([da[c] - db[c] for c in shared])
            jit = rng.uniform(-0.17, 0.17, size=len(shared))
            for i, cid in enumerate(shared):
                ax.plot(
                    diffs[i],
                    y + jit[i],
                    marker="o",
                    ms=3.2,
                    color=fam_colors[fams[cid]],
                    alpha=0.75,
                    lw=0,
                    zorder=2,
                )
            for fam in FAMILIES:
                vals = [diffs[i] for i, c in enumerate(shared) if fams[c] == fam]
                if vals:
                    ax.plot(
                        np.median(vals),
                        y,
                        marker="D",
                        ms=6.5,
                        color=fam_colors[fam],
                        mec="0.15",
                        mew=0.9,
                        lw=0,
                        zorder=3,
                    )
            signed = r["delta_over_floor_a"] - r["delta_over_floor_b"]
            ax.plot([signed], [y], marker="o", color="#2b6ca3", ms=7, mec="white", zorder=4)
        ax.axvline(0, color="0.3", lw=1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels(
            [PAIR_LABEL[r["pair"]] for r in rows] if ax is axes[0] else [""] * len(rows)
        )
        ax.set_title(BEH_LABEL[beh])
        ax.set_xlabel("Δ/floor difference")
    handles = [
        plt.Line2D([], [], marker="o", lw=0, ms=4.5, color=fam_colors[f], label=FAMILY_LABEL[f])
        for f in FAMILIES
    ]
    handles += [
        plt.Line2D([], [], marker="D", lw=0, ms=6, color="0.6", mec="0.15", label="family median"),
        plt.Line2D(
            [], [], marker="o", lw=0, ms=7, color="#2b6ca3", label="committed point estimate"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(
        "Per-context paired Δ/floor differences behind the family-clustered forest intervals",
        y=1.05,
    )
    savefig_paper(fig, "pairwise_diff_per_family", dir=OUT)
    plt.close(fig)


def dv4_per_context(cell: pe.CellData, arm: str, basis: np.ndarray) -> dict[int, float]:
    """Per-held-out-context centered R^2 — the exact per-fold terms dv4 pools."""
    Y = cell.pq_v[arm] @ basis.T
    res = pe.grouped_loco(cell.pq_c[arm], {"shared": Y}, cell.groups, None)
    out: dict[int, float] = {}
    for f in res["fold_ids"]:
        held = np.where(cell.groups == f)[0]
        if len(held) < 2:
            continue
        y_c = Y[held] - Y[held].mean(axis=0, keepdims=True)
        p = res["held_pred"]["shared"][held]
        p_c = p - p.mean(axis=0, keepdims=True)
        ss_res = float(np.sum((y_c - p_c) ** 2))
        ss_tot = float(np.sum(y_c**2))
        out[f] = (1.0 - ss_res / ss_tot, ss_res, ss_tot)
    return out


def fig_dv4_points(
    r2_by_cell: dict[tuple[str, str, str], dict[int, float]],
    committed: dict[tuple[str, str, str], float],
    ctx_ids_by_cell: dict[tuple[str, str], list[str]],
) -> None:
    colors = paper_palette_blog(3)
    rng = np.random.default_rng(42)
    width = 0.34
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), constrained_layout=True)
    for col, beh in enumerate(BEHAVIORS):
        ax = axes[col]
        panel_min: tuple[float, float, str] | None = None
        for i, sub in enumerate(SUBSTRATES):
            for a, arm in enumerate(("base", "trained")):
                vals = r2_by_cell.get((beh, sub, arm))
                if vals is None:
                    continue
                x0 = i + (a - 0.5) * width
                folds = sorted(vals)
                ys = np.array([vals[f] for f in folds])
                xs = x0 + rng.uniform(-0.10, 0.10, size=len(ys))
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    ms=3.0,
                    lw=0,
                    color=colors[i],
                    alpha=1.0 if arm == "trained" else 0.4,
                    zorder=2,
                )
                ax.plot(
                    [x0 - width / 2, x0 + width / 2],
                    [committed[(beh, sub, arm)]] * 2,
                    color="0.15",
                    lw=1.6,
                    zorder=3,
                )
                j = int(np.argmin(ys))
                if panel_min is None or ys[j] < panel_min[1]:
                    panel_min = (xs[j], ys[j], ctx_label(ctx_ids_by_cell[(beh, sub)][folds[j]]))
        if panel_min is not None:
            ax.text(panel_min[0], panel_min[1] - 0.045, panel_min[2], fontsize=6.5, ha="center")
        ax.set_xticks(range(3))
        ax.set_xticklabels([TICK_LABEL[s] for s in SUBSTRATES], fontsize=8)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_title(BEH_LABEL[beh])
        if col == 0:
            ax.set_ylabel("within-context R² per held-out context")
    fig.suptitle(
        "Per-context reads behind the within-context R² bars (light = base, dark = finetuned; "
        "black tick = committed pooled value)",
        y=1.05,
    )
    savefig_paper(fig, "dv4_per_context_points", dir=OUT)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RES / "summary.json").read_text())
    rb_main = fitM._load_rb_main()
    rb_fact = fitM._load_rb_fact()

    dof_by_cell: dict[tuple[str, str], dict] = {}
    r2_by_cell: dict[tuple[str, str, str], dict[int, float]] = {}
    committed_dv4: dict[tuple[str, str, str], float] = {}
    ctx_ids_by_cell: dict[tuple[str, str], list[str]] = {}
    worst_med_rel = 0.0
    worst_dv4_abs = 0.0
    for beh in BEHAVIORS:
        r_hat = (
            None if beh == "marker" else fitM._r_hat_for(beh, pe.HEADLINE_LAYER, rb_main, rb_fact)
        )
        for sub in SUBSTRATES:
            cell = pe.load_cell(beh, sub, REDUCED_ROOT, MAPS_ROOT)
            ctx_ids_by_cell[(beh, sub)] = cell.ctx_ids
            committed = json.loads((RES / "delta_floor" / f"{beh}__{sub}.json").read_text())
            proj = per_context_delta_over_floor(cell, r_hat)
            med = float(np.median(proj))
            rel = abs(med - committed["delta_med"]) / committed["delta_med"]
            worst_med_rel = max(worst_med_rel, rel)
            if rel > 2e-2:
                raise RuntimeError(
                    f"{beh}/{sub}: recomputed median {med:.6g} vs committed "
                    f"{committed['delta_med']:.6g} (rel {rel:.3g} > 2e-2) — numerator drift"
                )
            floor_eff = committed["delta_med"] / committed["delta_over_floor"]
            dof = {cid: float(v) / floor_eff for cid, v in zip(cell.ctx_ids, proj, strict=True)}
            dof["_fams"] = dict(zip(cell.ctx_ids, cell.families, strict=True))
            dof_by_cell[(beh, sub)] = dof

            basis = fitM._pca_basis_v0(cell.avg_v["base"], pe.TARGET_DIM)
            tj = json.loads((PE_DIR / f"transfer_L14_{beh}__{sub}.json").read_text())
            for arm in ("base", "trained"):
                per_fold = dv4_per_context(cell, arm, basis)
                pooled = 1.0 - sum(v[1] for v in per_fold.values()) / sum(
                    v[2] for v in per_fold.values()
                )
                comm = tj["dv4"][arm]["r2_within_observed"]
                gap = abs(pooled - comm)
                worst_dv4_abs = max(worst_dv4_abs, gap)
                if gap > 1e-4:
                    raise RuntimeError(
                        f"{beh}/{sub}/{arm}: pooled within-context R2 {pooled:.8f} vs "
                        f"committed {comm:.8f} (abs {gap:.3g} > 1e-4)"
                    )
                r2_by_cell[(beh, sub, arm)] = {f: v[0] for f, v in per_fold.items()}
                committed_dv4[(beh, sub, arm)] = comm
            print(f"[ok] {beh}/{sub}: median rel {rel:.2e}; dv4 pooled reproduced", flush=True)

    fig_pairwise_per_family(summary, dof_by_cell)
    fig_dv4_points(r2_by_cell, committed_dv4, ctx_ids_by_cell)
    print(
        f"GATES PASS — worst median rel {worst_med_rel:.3g} (tol 2e-2); "
        f"worst dv4 pooled abs {worst_dv4_abs:.3g} (tol 1e-4)"
    )


if __name__ == "__main__":
    main()
