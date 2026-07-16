"""Figures for issue #1092 round `offvm-battery-refit-and-operator-comparison`.

Reads the two committed round digests (built from the 76 per-box JSONs on
branch issue-1092-offvm @ 24c964b2a0 plus the HF `p6/box_rf0*/checkpoints/`
per-unit checkpoints) and the inline round's `fit_free_repairs.json`, and
renders three figures:

1. refit_r2_banked_vs_excluded  — battery-excluded refit vs banked held-out R²
   (paired bars at the headline config + the per-unit scatter over all 64
   refit units).
2. refit_floors_vs_ridge_v2     — t1 transport floors vs ridge, context and
   prefix (zoomed) panels; refit t1 ridge for the 4 refit cells, banked
   pooled ridge (hatched) for the 4 battery-free cells.
3. partb_operator_angles_procrustes — Part-B operator arm comparison: mean
   principal angles (output k@90%, input k=48) and Procrustes residuals per
   cell x layer vs 200-draw spectrum-matched null bands.

Run from repo root:
    uv run python scripts/issue1092_offvm_refit_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROUND_DIR = Path("eval_results/issue_1092/offvm-battery-refit-and-operator-comparison")
INLINE_JSON = Path(
    "eval_results/issue_1092/inline_caveat_repairs_operator_comparison/fit_free_repairs.json"
)
FIG_DIR = Path("figures/issue_1092")

CELL_SHORT = {
    "cell_inst_own": "inst / own",
    "cell_inst_claude": "inst / Claude",
    "cell_inst_pretext": "inst / pretrained-text",
    "cell_inst_shuf": "inst / shuffled",
    "cell_pre_own": "pretrained / own",
    "cell_pre_claude": "pretrained / Claude",
    "cell_pre_insttext": "pretrained / inst-text",
    "cell_pre_shuf": "pretrained / shuffled",
}
REFIT_CELLS = ["cell_inst_own", "cell_inst_pretext", "cell_pre_own", "cell_pre_insttext"]
ALL_CELLS = [
    "cell_inst_own",
    "cell_inst_claude",
    "cell_inst_pretext",
    "cell_inst_shuf",
    "cell_pre_own",
    "cell_pre_claude",
    "cell_pre_insttext",
    "cell_pre_shuf",
]


def fig0_read1_v2(rows: list[dict], inline: dict) -> None:
    """Battery-excluded canonical read1 bars (supersedes read1_r2_prefix_vs_context)."""
    refit = {
        (r["cell"], r["arm"]): r["refit_r2"]
        for r in rows
        if r["layer"] == 14 and r["fit_arm"] == "A" and r["basis"] == "ambient"
    }
    banked = inline["A1_read1_banked_old"]["cells"]
    c_ctx, c_pfx = paper_palette(2)
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    x = np.arange(len(ALL_CELLS))
    w = 0.36
    for off, arm, color, lab in [
        (-w / 2, "context_end", c_ctx, "context-based map (prefix + query end)"),
        (w / 2, "prefix_end", c_pfx, "prefix-based map (persona end)"),
    ]:
        vals = [refit.get((c, arm), banked[c][arm]["r2"]) for c in ALL_CELLS]
        ax.bar(x + off, vals, w, color=color, label=lab)
        for xi, v in zip(x, vals, strict=True):
            ax.text(xi + off, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_SHORT[c] for c in ALL_CELLS], rotation=25, ha="right")
    ax.set_ylabel("held-out R² (grouped 6-fold, pooled targets)")
    ax.set_title(
        "Held-out map skill per cell (layer 14, fit-arm A, ambient; battery-excluded)",
        fontsize=11,
        pad=12,
    )
    ax.legend(fontsize=8)
    savefig_paper(fig, "read1_r2_prefix_vs_context_v2", dir=FIG_DIR)
    plt.close(fig)


def fig1_refit_vs_banked(rows: list[dict]) -> None:
    """Paired bars (headline config) + per-unit banked-vs-refit scatter."""
    c_ctx, c_pfx = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # Panel A: L14 fit-arm A ambient, 4 refit cells x 2 arms, banked vs refit.
    ax = axes[0]
    sel = [r for r in rows if r["layer"] == 14 and r["fit_arm"] == "A" and r["basis"] == "ambient"]
    sel.sort(key=lambda r: (REFIT_CELLS.index(r["cell"]), r["arm"] != "context_end"))
    x = np.arange(len(REFIT_CELLS))
    w = 0.19
    for j, (arm, color) in enumerate([("context_end", c_ctx), ("prefix_end", c_pfx)]):
        armrows = {r["cell"]: r for r in sel if r["arm"] == arm}
        banked = [armrows[c]["banked_r2"] for c in REFIT_CELLS]
        refit = [armrows[c]["refit_r2"] for c in REFIT_CELLS]
        off = -1.5 * w + 2 * j * w
        ax.bar(
            x + off,
            banked,
            w,
            color=color,
            alpha=0.45,
            label=f"{'context' if arm == 'context_end' else 'prefix'} — banked (battery in)",
        )
        ax.bar(
            x + off + w,
            refit,
            w,
            color=color,
            label=f"{'context' if arm == 'context_end' else 'prefix'} — refit (battery out)",
        )
        for xi, (b, rf) in zip(x, zip(banked, refit, strict=True), strict=True):
            ax.text(
                xi + off + w / 2,
                max(b, rf) + 0.012,
                f"{rf - b:+.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_SHORT[c] for c in REFIT_CELLS], rotation=20, ha="right")
    ax.set_ylabel("held-out R² (grouped 6-fold, pooled targets)")
    ax.set_title(
        "Headline config (layer 14, ambient): banked vs battery-excluded", fontsize=11, pad=12
    )
    ax.legend(fontsize=8, loc="center right")

    # Panel B: per-unit scatter over all 64 refit units.
    ax = axes[1]
    lims = (-0.02, 1.0)
    ax.plot(lims, lims, color="0.6", lw=1.0, zorder=1)
    for arm, color in [("context_end", c_ctx), ("prefix_end", c_pfx)]:
        for fit_arm, mk in [("A", "o"), ("B", "^")]:
            pts = [r for r in rows if r["arm"] == arm and r["fit_arm"] == fit_arm]
            ax.scatter(
                [p["banked_r2"] for p in pts],
                [p["refit_r2"] for p in pts],
                s=26,
                color=color,
                marker=mk,
                zorder=2,
                label=f"{'context' if arm == 'context_end' else 'prefix'} arm, fit-arm {fit_arm}",
            )
    movers = sorted(
        (r for r in rows if r["delta"] is not None and abs(r["delta"]) >= 0.024),
        key=lambda r: r["refit_r2"],
    )
    for i, r in enumerate(movers):  # label the units that moved most
        below = i % 2 == 0
        ax.annotate(
            f"{CELL_SHORT[r['cell']]} L{r['layer']} {r['basis']} {r['fit_arm']}",
            (r["banked_r2"], r["refit_r2"]),
            xytext=(10, -22 - 12 * (i % 3) if below else 22 + 12 * (i % 3)),
            textcoords="offset points",
            fontsize=6.5,
            ha="left",
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.55"},
        )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("banked held-out R² (battery rows in fit training)")
    ax.set_ylabel("refit held-out R² (battery rows excluded)")
    ax.set_title(
        "All 64 refit units (4 cells x layers x arms x bases; max |delta| 0.036)",
        fontsize=11,
        pad=12,
    )
    ax.legend(fontsize=8, loc="upper left")
    savefig_paper(fig, "refit_r2_banked_vs_excluded", dir=FIG_DIR)
    plt.close(fig)


def fig2_floors(rows: list[dict], inline: dict) -> None:
    """t1 transport floors vs ridge; context panel + zoomed prefix panel."""
    floors = inline["A3_transport_floors"]["cells"]
    refit_t1 = {
        (r["cell"], r["arm"]): r["t1"]
        for r in rows
        if r["layer"] == 14 and r["fit_arm"] == "A" and r["basis"] == "ambient"
    }
    banked_pooled = inline["A1_read1_banked_old"]["cells"]
    colors = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, arm, title in [
        (axes[0], "context_end", "Context-arm map (prefix+query end)"),
        (axes[1], "prefix_end", "Prefix-arm map (persona end, zoomed)"),
    ]:
        x = np.arange(len(ALL_CELLS))
        w = 0.27
        ridge_vals, ridge_hatch = [], []
        diag_vals, scaled_vals = [], []
        for c in ALL_CELLS:
            t1f = floors[c][arm]["per_target"]["t1"]
            diag_vals.append(t1f["diag_affine"])
            scaled_vals.append(t1f["global_affine_scaled_identity"])
            if (c, arm) in refit_t1:
                ridge_vals.append(refit_t1[(c, arm)])
                ridge_hatch.append("")
            else:
                ridge_vals.append(banked_pooled[c][arm]["r2"])
                ridge_hatch.append("//")
        bars = ax.bar(x - w, ridge_vals, w, color=colors[0], label="ridge R²")
        for b, h in zip(bars, ridge_hatch, strict=True):
            if h:  # battery-free cell: banked POOLED ridge (no refit t1) — lighter + hatched
                b.set_hatch(h)
                b.set_alpha(0.45)
        ax.bar(x, diag_vals, w, color=colors[1], label="diag-affine floor (t1)")
        ax.bar(x + w, scaled_vals, w, color=colors[2], label="scaled-identity floor (t1)")
        for xi, v in zip(x, ridge_vals, strict=True):
            ax.text(
                xi - w,
                v + (0.008 if arm == "context_end" else 0.002),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_SHORT[c] for c in ALL_CELLS], rotation=25, ha="right")
        ax.set_ylabel("held-out R²")
        ax.set_title(title, fontsize=11, pad=12)
        if arm == "prefix_end":
            ax.set_ylim(0, 0.17)
        ax.legend(fontsize=8)
    savefig_paper(fig, "refit_floors_vs_ridge_v2", dir=FIG_DIR)
    plt.close(fig)


def fig3_partb(partb: list[dict]) -> None:
    """Principal angles + Procrustes residuals vs spectrum-matched null bands."""
    amb = [e for e in partb if e["basis"] == "ambient"]
    layers = [14, 18, 19]
    deg = lambda r: r * 180 / math.pi  # noqa: E731
    c_out, c_in = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))

    ax = axes[0]
    x0 = np.arange(len(ALL_CELLS))
    for series, key, color in [
        ("output subspaces (k@90% energy)", "output_k90", c_out),
        ("input subspaces (k=48)", "input_k48", c_in),
    ]:
        for e in amb:
            xi = ALL_CELLS.index(e["cell"]) + (layers.index(e["layer"]) - 1) * 0.16
            blk = e[key]
            ax.plot(
                [xi, xi],
                [deg(blk["null_p05"]), deg(blk["null_p95"])],
                color="0.45",
                lw=3.0,
                alpha=0.8,
                zorder=1,
                solid_capstyle="butt",
            )
            ax.scatter([xi], [deg(blk["mean"])], s=24, color=color, zorder=3)
            ax.text(
                xi,
                deg(blk["mean"]) - 2.2,
                str(e["layer"]),
                fontsize=5.5,
                ha="center",
                va="top",
                color=color,
            )
        ax.scatter([], [], s=24, color=color, label=series)
    ax.plot([], [], color="0.45", lw=3.0, label="200-draw spectrum-matched null (5th-95th pct)")
    ax.axhline(90, color="0.8", lw=0.8, ls=":")
    ax.set_xticks(x0)
    ax.set_xticklabels([CELL_SHORT[c] for c in ALL_CELLS], rotation=25, ha="right")
    ax.set_ylabel("mean principal angle (degrees; 0 = identical subspaces)")
    ax.set_ylim(0, 95)
    ax.set_title(
        "Prefix-arm vs context-arm operator subspaces (points: layers 14/18/19)",
        fontsize=11,
        pad=12,
    )
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    for e in amb:
        xi = ALL_CELLS.index(e["cell"]) + (layers.index(e["layer"]) - 1) * 0.16
        ax.plot(
            [xi, xi],
            [e["proc_null_p05"], e["proc_null_p95"]],
            color="0.45",
            lw=3.0,
            alpha=0.8,
            zorder=1,
            solid_capstyle="butt",
        )
        ax.scatter([xi], [e["proc_resid"]], s=24, color=c_out, zorder=3)
        ax.text(
            xi,
            e["proc_resid"] - 0.004,
            str(e["layer"]),
            fontsize=5.5,
            ha="center",
            va="top",
            color=c_out,
        )
    ax.scatter([], [], s=24, color=c_out, label="orthogonal-Procrustes residual")
    ax.plot([], [], color="0.45", lw=3.0, label="200-draw spectrum-matched null (5th-95th pct)")
    ax.axhline(1.0, color="0.8", lw=0.8, ls=":")
    ax.set_xticks(x0)
    ax.set_xticklabels([CELL_SHORT[c] for c in ALL_CELLS], rotation=25, ha="right")
    ax.set_ylabel("min over rotations of ||W_ctx - R.W_pfx|| / ||W_ctx||")
    ax.set_title(
        "No rotation maps the prefix-arm operator onto the context-arm operator",
        fontsize=11,
        pad=12,
    )
    ax.legend(fontsize=8, loc="upper left")
    savefig_paper(fig, "partb_operator_angles_procrustes", dir=FIG_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    rows = json.loads((ROUND_DIR / "refit_vs_banked_digest.json").read_text())
    partb = json.loads((ROUND_DIR / "partb_operator_digest.json").read_text())
    inline = json.loads(INLINE_JSON.read_text())
    fig0_read1_v2(rows, inline)
    fig1_refit_vs_banked(rows)
    fig2_floors(rows, inline)
    fig3_partb(partb)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
