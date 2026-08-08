"""Issue #1310 / #1639 n<d estimator audit — corrections table + figures.

Consumes eval_results/issue_1310/nd_estimator_audit/corrections_table.json
(written by scripts/issue1310_nd_estimator_audit.py) and emits:

  - corrections_table.md   per-cell: published | reproduced ambient | inner-CV |
                           reduced-basis | verdict
  - figures/issue_1310/nd_audit_published_vs_corrected.{png,pdf,meta.json}
  - figures/issue_1310/nd_audit_selector_spread.{png,pdf,meta.json}

Two verdict axes, both at the 0.05 R^2 materiality threshold:

  ambient_vs_published   does the ambient pure-GCV selector (the one the LOST
                         run-2 script cells used) materially deflate the read?
                         -> sizes the damage class.
  published_vs_corrected does the PUBLISHED capped-GCV read itself move under
                         the principled inner-group-CV selector?
                         -> says whether the published number needs correcting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps / matplotlib pin before heavy imports (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "eval_results" / "issue_1310" / "nd_estimator_audit"
FIGDIR = REPO / "figures" / "issue_1310"
THRESH = 0.05

# Published per-persona turn-pair counts of the script-format cells (their own n).
PUBLISHED_N = {
    "script_base_Wren": 2329,
    "script_base_HELIOS": 2466,
    "script_base_Dana": 1325,
    "script_base_Vex": 2060,
    "script_instruct_Wren": 3094,
    "script_instruct_HELIOS": 3123,
    "script_instruct_Dana": 2700,
    "script_instruct_Vex": 3586,
}

ARMS = [
    ("ambient_pure_gcv", "ambient pure-GCV"),
    ("ref_capped_gcv", "capped GCV (published)"),
    ("inner_group_cv", "inner-group-CV"),
    ("reduced_pca_basis", "reduced PCA basis"),
    ("forced_lambda_1e+02", "forced lambda 1e2"),
    ("forced_lambda_1e+03", "forced lambda 1e3"),
    ("forced_lambda_1e+04", "forced lambda 1e4"),
]


def _verdicts(rec: dict) -> dict:
    """Both materiality verdicts for one cell.

    The baseline is the arm that reproduces the cell's PUBLISHED selector: the
    prefill families published under the capped selector, while 7 of the 8
    recaptured script-format cells published under ambient pure-GCV (instruct
    Vex, from the completion round, published capped). `published_selector` is
    absent on the parent audit's rows, which default to capped.
    """
    baseline_arm = rec.get("published_selector", "ref_capped_gcv")
    ref = rec["arms"][baseline_arm]["r2_pooled"]
    amb = rec["arms"]["ambient_pure_gcv"]["r2_pooled"]
    inner = rec["arms"]["inner_group_cv"]["r2_pooled"]
    d_amb, d_inner = amb - ref, inner - ref
    sign_flip_amb = (ref > 0) != (amb > 0)
    sign_flip_inner = (ref > 0) != (inner > 0)
    if abs(d_amb) > THRESH or sign_flip_amb:
        v_amb = "artifact-deflated" if d_amb < 0 else "artifact-inflated"
    else:
        v_amb = "robust"
    if abs(d_inner) > THRESH or sign_flip_inner:
        v_inner = "published-deflated" if d_inner > 0 else "published-inflated"
    else:
        v_inner = "robust"
    return {
        "baseline_arm": baseline_arm,
        "delta_ambient_minus_published": d_amb,
        "delta_innercv_minus_published": d_inner,
        "sign_flip_ambient": bool(sign_flip_amb),
        "sign_flip_innercv": bool(sign_flip_inner),
        "ambient_vs_published": v_amb,
        "published_vs_corrected": v_inner,
    }


def _fig_published_vs_corrected(cells: list[dict]) -> Path:
    """Published capped-GCV read vs the inner-group-CV re-selected read, vs y = x.

    The ambient pure-GCV arm is deliberately NOT plotted here: its values reach
    -5.5 and would compress this comparison into an unreadable sliver (that arm
    is the subject of the companion selector-spread figure, on a symlog axis).
    """
    set_paper_style("blog")
    pal = paper_palette_blog(3)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for ax, fam, fam_label in (
        (axes[0], "per_turn_prefill", "per-turn prefill cells (n = 1,402-1,801)"),
        (axes[1], "scene_aggregated", "scene-aggregated cells (n = 300)"),
        (
            axes[2],
            "script_format_recaptured",
            "script-format cells, recaptured store (n = 1,325-3,471)",
        ),
    ):
        sub = [c for c in cells if c.get("family") == fam]
        # x-axis is the cell's OWN published selector (capped for the prefill
        # families; ambient for 7 of the 8 recaptured script cells).
        ref = np.array([c["arms"][c["verdicts"]["baseline_arm"]]["r2_pooled"] for c in sub])
        inner = np.array([c["arms"]["inner_group_cv"]["r2_pooled"] for c in sub])
        lo = float(min(ref.min(), inner.min(), 0.0)) - 0.08
        hi = float(max(ref.max(), inner.max(), 0.0)) + 0.08
        ax.plot(
            [lo, hi], [lo, hi], color="0.55", lw=1.2, ls="--", zorder=1, label="no change (y = x)"
        )
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.scatter(
            ref,
            inner,
            s=80,
            color=pal[0],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label="inner-group-CV selector",
        )
        for c, x, y in zip(sub, ref, inner, strict=True):
            dx, dy = (8, -3) if c["model"] == "base" else (8, 6)
            ax.annotate(
                f"{c['model'][:4]}·{c['persona']}",
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8.5,
            )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("published held-out $R^2$ (own published selector, layer 19)")
        ax.set_ylabel("re-selected held-out $R^2$ (inner-group-CV)")
        ax.set_title(fam_label, fontsize=11)
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    fig.suptitle(
        "Principled lambda re-selection lifts every per-turn prefill read and leaves "
        "the scene-aggregated and script-format reads unchanged",
        fontsize=12.5,
    )
    fig.tight_layout()
    paths = savefig_paper(fig, "nd_audit_published_vs_corrected", dir=FIGDIR)
    plt.close(fig)
    return paths["png"]


def _fig_selector_spread(cells: list[dict]) -> Path:
    """Per-cell held-out R^2 under every selector family (the full spread)."""
    set_paper_style("blog")
    pal = paper_palette_blog(len(ARMS))
    order = (
        [c for c in cells if c.get("family") == "per_turn_prefill"]
        + [c for c in cells if c.get("family") == "scene_aggregated"]
        + [c for c in cells if c.get("family") == "script_format_recaptured"]
    )
    labels = [f"{c['model'][:4]}·{c['persona']}\n{c['family'].split('_')[0]}" for c in order]
    xs = np.arange(len(order))
    # symlog y: the ambient arm reaches -5.5 while every other arm lives inside
    # [-0.6, +0.45]; a linear axis compresses the informative band to a sliver.
    linthresh = 0.5
    markers = {
        "ambient_pure_gcv": "v",
        "ref_capped_gcv": "o",
        "inner_group_cv": "^",
        "reduced_pca_basis": "D",
        "forced_lambda_1e+02": "s",
        "forced_lambda_1e+03": "s",
        "forced_lambda_1e+04": "s",
    }
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    n_arms = len(ARMS)
    for i, ((key, lab), col) in enumerate(zip(ARMS, pal, strict=True)):
        ys = [c["arms"][key]["r2_pooled"] for c in order]
        dodge = (i - (n_arms - 1) / 2) * 0.10
        ms = 8 if key in ("ref_capped_gcv", "inner_group_cv", "ambient_pure_gcv") else 5
        ax.plot(
            xs + dodge,
            ys,
            marker=markers[key],
            ms=ms,
            lw=0.0,
            color=col,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=lab,
            zorder=3,
        )
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.6)
    ax.axhline(0.0, color="0.4", lw=1.0, zorder=1)
    ax.axhspan(-linthresh, linthresh, color="0.94", zorder=0)
    ax.set_yticks([-5, -2, -1, -0.5, -0.25, 0.0, 0.25, 0.5])
    ax.set_yticklabels(["-5", "-2", "-1", "-0.5", "-0.25", "0", "0.25", "0.5"])
    for b in np.arange(len(order)) + 0.5:
        ax.axvline(b, color="0.9", lw=0.6, zorder=0)
    # family dividers: per-turn | aggregated | script-format(recaptured)
    n_pt = sum(1 for c in order if c.get("family") == "per_turn_prefill")
    n_ag = sum(1 for c in order if c.get("family") == "scene_aggregated")
    for b in (n_pt - 0.5, n_pt + n_ag - 0.5):
        ax.axvline(b, color="0.5", lw=1.2, ls=":", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("held-out pooled $R^2$ (layer 19, symlog outside $\\pm$0.5)")
    ax.set_xlabel(
        "cell (model · persona, cell family) — per-turn | scene-aggregated | "
        "script-format, split by the dotted lines"
    )
    ax.set_title(
        "Held-out $R^2$ per cell under each lambda selector — ambient pure-GCV "
        "collapses, inner-group-CV lifts the per-turn cells",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=4, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    paths = savefig_paper(fig, "nd_audit_selector_spread", dir=FIGDIR)
    plt.close(fig)
    return paths["png"]


def _md_table(table: dict, cells: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# n<d ridge-selector audit — #1310 / #1639 corrections table")
    lines.append("")
    lines.append(
        f"Layer {table['layer']}, {table['folds']}-fold group-held-out, seed "
        f"{table['seed']}, ambient d = {table['d_ambient']}. Lambda grid "
        f"{table['lambda_grid'][0]:g}-{table['lambda_grid'][-1]:g} "
        f"({len(table['lambda_grid'])} points). Materiality threshold "
        f"{THRESH} R^2 or a sign change."
    )
    lines.append("")
    lines.append(f"**Store gap.** {table['store_gap']}")
    lines.append("")
    lines.append(
        "`published` = the committed `r2_per_layer_obs[19]` (capped GCV, "
        "`gcv_dof_cap: 0.9`); the audit reproduces it in the `ref` arm before "
        "any other read. `ambient` reproduces the SELECTOR the lost run-2 "
        "script cells used. `forced lambda` arms are selection-bearing "
        "diagnostics, never headlines."
    )
    lines.append("")
    hdr = (
        "| cell | family | n | n_train | published | reproduced ref | ambient "
        "pure-GCV | inner-group-CV | reduced PCA basis | ambient verdict | "
        "published verdict |"
    )
    lines.append(hdr)
    lines.append("|" + "---|" * 11)
    for c in cells:
        v = c["verdicts"]
        a = c["arms"]
        pub = c.get("published_r2_l19")
        pub_s = "n/a" if pub is None else f"{pub:+.4f}"
        lines.append(
            f"| `{c['cell_id']}` | {c['family']} | {c['n']} | "
            f"{c['n_train_nominal']} | {pub_s} | "
            f"{a['ref_capped_gcv']['r2_pooled']:+.4f} | "
            f"{a['ambient_pure_gcv']['r2_pooled']:+.4f} | "
            f"{a['inner_group_cv']['r2_pooled']:+.4f} | "
            f"{a['reduced_pca_basis']['r2_pooled']:+.4f} | "
            f"{v['ambient_vs_published']} | {v['published_vs_corrected']} |"
        )
    lines.append("")
    script = [c for c in cells if c.get("family") == "script_format_recaptured"]
    if script:
        lines.append("## Script-format cells (RECAPTURED store)")
        lines.append("")
        lines.append(
            "The original run-2 script-format activation store was lost with its "
            "instance; these rows are fit on the store rebuilt by "
            "`scripts/issue1310_recapture_script_store.py` (job 16086) at "
            "`issue1310_char_map/analysis_tensors/store_recap/`. Seven of the eight "
            "published under AMBIENT pure-GCV (no `gcv_dof_cap` field in their "
            "committed JSONs); instruct Vex came from the completion round and "
            "published CAPPED. `reproduced` is the cell's OWN published selector "
            "re-run on the recaptured store."
        )
        lines.append("")
        lines.append(
            "| cell | n (published n) | published sel. | published | reproduced | "
            "repro delta | inner-group-CV | verdict | recapture |"
        )
        lines.append("|" + "---|" * 9)
        for c in script:
            a, v = c["arms"], c["verdicts"]
            rp = c.get("published_selector_reproduction", {})
            pub = c.get("published_r2_l19")
            pubn = PUBLISHED_N.get(c["cell_id"])
            delta = rp.get("abs_delta")
            sel = "ambient" if v["baseline_arm"] == "ambient_pure_gcv" else "capped"
            fid = "span-exact" if c["model"] == "base" else "near-replica"
            lines.append(
                f"| `{c['cell_id']}` | {c['n']} ({pubn}) | {sel} | "
                f"{'n/a' if pub is None else f'{pub:+.4f}'} | "
                f"{rp.get('recomputed', float('nan')):+.4f} | "
                f"{'n/a' if delta is None else f'{delta:.4f}'} | "
                f"{a['inner_group_cv']['r2_pooled']:+.4f} | "
                f"{v['published_vs_corrected']} | {fid} |"
            )
        lines.append("")
    lines.append("## Selected lambda per arm (grid-edge proximity)")
    lines.append("")
    lines.append(
        "| cell | ambient: median lambda | folds at grid floor (0.01) | "
        "capped: median lambda | inner-CV: median lambda |"
    )
    lines.append("|" + "---|" * 5)
    for c in cells:
        d = c["lambda_diagnostics"]
        lines.append(
            f"| `{c['cell_id']}` | {d['ambient_pure_gcv']['median']:g} | "
            f"{d['ambient_pure_gcv']['n_at_grid_floor']}/"
            f"{d['ambient_pure_gcv']['n_folds']} | "
            f"{d['ref_capped_gcv']['median']:g} | "
            f"{d['inner_group_cv']['median']:g} |"
        )
    lines.append("")
    lines.append("## Forced-lambda diagnostic reads")
    lines.append("")
    lines.append("| cell | lambda 1e2 | lambda 1e3 | lambda 1e4 |")
    lines.append("|" + "---|" * 4)
    for c in cells:
        a = c["arms"]
        lines.append(
            f"| `{c['cell_id']}` | {a['forced_lambda_1e+02']['r2_pooled']:+.4f} | "
            f"{a['forced_lambda_1e+03']['r2_pooled']:+.4f} | "
            f"{a['forced_lambda_1e+04']['r2_pooled']:+.4f} |"
        )
    lines.append("")
    lines.append("## Mapping baselines (ambient space; standing dual-read rule)")
    lines.append("")
    lines.append(
        "| cell | identity+learned-bias $R^2$ | kNN acc@1 (capped GCV) | "
        "kNN acc@1 (ambient) | chance acc@1 |"
    )
    lines.append("|" + "---|" * 5)
    for c in cells:
        b = c["mapping_baselines"]
        kr, ka = b["knn_ref_capped_gcv"], b["knn_ambient_pure_gcv"]
        lines.append(
            f"| `{c['cell_id']}` | {b['identity_bias_r2_pooled']:+.4f} | "
            f"{kr['acc_at_k']['1']:.3f} | {ka['acc_at_k']['1']:.3f} | "
            f"{kr['chance_at_k']['1']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    table = json.loads((OUT / "corrections_table.json").read_text())
    cells = table["cells"]
    for c in cells:
        c["verdicts"] = _verdicts(c)
    # normalise kNN dict keys (json round-trip stringifies int k)
    for c in cells:
        for key in ("knn_ref_capped_gcv", "knn_ambient_pure_gcv"):
            d = c["mapping_baselines"][key]
            for f in ("acc_at_k", "chance_at_k"):
                d[f] = {str(k): v for k, v in d[f].items()}
    table["materiality_threshold_r2"] = THRESH
    table["n_cells"] = len(cells)
    table["summary"] = {
        "ambient_material": sum(
            1 for c in cells if c["verdicts"]["ambient_vs_published"] != "robust"
        ),
        "ambient_sign_flips": sum(1 for c in cells if c["verdicts"]["sign_flip_ambient"]),
        "published_material": sum(
            1 for c in cells if c["verdicts"]["published_vs_corrected"] != "robust"
        ),
        "published_sign_flips": sum(1 for c in cells if c["verdicts"]["sign_flip_innercv"]),
    }
    (OUT / "corrections_table.json").write_text(json.dumps(table, indent=1))
    (OUT / "corrections_table.md").write_text(_md_table(table, cells))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p1 = _fig_published_vs_corrected(cells)
    p2 = _fig_selector_spread(cells)
    print(f"[nd-report] wrote {OUT / 'corrections_table.md'}")
    print(f"[nd-report] figures: {p1}  {p2}")
    print(f"[nd-report] summary: {json.dumps(table['summary'])}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
