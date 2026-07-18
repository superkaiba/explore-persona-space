"""Issue #1417 — figures (hero + exploratory dump) from eval_results JSONs.

Pure-CPU, reads only ``eval_results/issue_1417/`` (battery_summary.json,
cells/cells_*.json + nulls_*.json, battery/battery_*.json, judge/kept_*.json)
— re-runnable off-instance. Every panel skips gracefully when its inputs are
absent (smoke runs produce a subset), and every errorbar site computes
NON-NEGATIVE offsets element-wise (``max(0, v - lo)`` / ``max(0, hi - v)`` —
the gotchas xerr/yerr contract; inverted tiny-n quantile CIs are clamped,
never passed as bounds).

Figures (plan §6):
  hero_rel_bars           per-cell REL @ L19 bars + bootstrap CIs, per model,
                          verdict boundary 0.5 + shared-regime reference 1.0
  hero_prediction_table   the 2x2 H-table with realized (C2, C4) verdicts
                          highlighted per model
  layer_r2_curves         28-layer R^2 per cell + shuffle-null p97.5 (ctx arm)
  transfer_matrix         cell x reference transfer R^2 @ L19 heatmap
  cosine_grid             raw vs rotation-aligned cosine per pair + chance band
  yield_panel             judge keep-rate per (model, cell) + 50% floor
  judge_score_hists       per-cell judge mean-score histograms
  y_var_ratio             answer-profile variance ratio vs C0 + collapse floor
  matched_vs_full         matched-n vs full-n R^2 scatter (labeled points)
  all_vs_kept             all-rows vs kept-rows R^2 scatter (labeled points)
  prefix_vs_ctx           prefix-arm (degenerate control) vs ctx-arm R^2 bars

CLI:
  uv run python scripts/issue1417_figures.py [--out-dir eval_results/issue_1417]
      [--fig-dir figures/issue_1417]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1417_render as r1417  # noqa: E402

SCRIPT = "scripts/issue1417_figures.py"
HEADLINE_LAYER = 19
REL_BOUNDARY = 0.5
YIELD_FLOOR = 0.5
COLLAPSE_VAR_RATIO = 0.5

# Plain-English cell names on every tick label (no opaque condition codes).
CELL_LABELS = {
    "c1_helpful_ctrl": "Helpful\ninstruction",
    "c2_rude": "Rude-but-\ninformative",
    "c3_evasive": "Evasive",
    "c4_exposition": "Addressee-free\nexposition",
    "c5_ai_addressee": "Non-user\naddressee",
}
MODEL_LABELS = {"instruct": "Instruct", "pretrained": "Base"}
REF_LABELS = {
    "c0_chat": "chat reference",
    "c0p_nat": "plain-dialogue reference",
    "c1": "helpful-instruction control",
}


def ci_offsets(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(2, n) NON-NEGATIVE errorbar offsets from CI bounds (clamped element-wise).

    matplotlib's yerr takes offsets-from-value, all >= 0 — a tiny-n quantile CI
    can legitimately invert around the point estimate (gotchas #1335/#547);
    clamping keeps the render alive and the clamp itself is loggable upstream.
    """
    v, lo, hi = np.asarray(v, float), np.asarray(lo, float), np.asarray(hi, float)
    return np.vstack([np.maximum(0.0, v - lo), np.maximum(0.0, hi - v)])


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _summary(out_dir: Path) -> dict:
    d = _load_json(out_dir / "battery_summary.json")
    assert d is not None, f"battery_summary.json missing under {out_dir} — run --summary first"
    return d


# ---------------------------------------------------------------------------
# Hero 1 — per-cell REL bars with CIs
# ---------------------------------------------------------------------------
def fig_hero_rel_bars(summary: dict, fig_dir: Path) -> None:
    """REL @ L19 per framing cell (vs the chat reference; C4 also vs the
    plain-dialogue reference as a hollow companion bar), per model."""
    cells = summary.get("cells", {})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    colors = paper_palette(len(r1417.CELL_ORDER))
    plotted = False
    for ax, model in zip(axes, r1417.MODELS, strict=True):
        xs, vals, los, his, cols, labels = [], [], [], [], [], []
        x = 0
        for i, cell in enumerate(r1417.CELL_ORDER):
            e = cells.get(f"{model}__{cell}", {})
            rel = e.get("rel_l19")
            ci = e.get("delta_rel_ci95")
            if rel is None or not np.isfinite(rel):
                x += 1
                continue
            xs.append(x)
            vals.append(float(rel))
            # delta_rel_ci95 is the CI of REL - 0.5; shift back to REL space.
            if ci and all(np.isfinite(ci)):
                los.append(float(ci[0]) + REL_BOUNDARY)
                his.append(float(ci[1]) + REL_BOUNDARY)
            else:
                los.append(float(rel))
                his.append(float(rel))
            cols.append(colors[i])
            labels.append(CELL_LABELS[cell])
            if cell == "c4_exposition" and e.get("rel_l19_vs_c0p") is not None:
                v2 = float(e["rel_l19_vs_c0p"])
                if np.isfinite(v2):
                    ax.bar(
                        x + 0.28,
                        v2,
                        width=0.22,
                        facecolor="none",
                        edgecolor=colors[i],
                        linewidth=1.4,
                    )
            x += 1
        if vals:
            plotted = True
            vals_a = np.asarray(vals)
            ax.bar(xs, vals_a, width=0.55, color=cols)
            ax.errorbar(
                xs,
                vals_a,
                yerr=ci_offsets(vals_a, np.asarray(los), np.asarray(his)),
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=2.5,
            )
        ax.axhline(REL_BOUNDARY, color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0)
        ax.axhline(1.0, color=paper_palette_role("baseline"), linestyle=":", linewidth=1.0)
        ax.set_xticks(range(len(r1417.CELL_ORDER)))
        ax.set_xticklabels([CELL_LABELS[c] for c in r1417.CELL_ORDER], fontsize=6.5)
        ax.set_title(MODEL_LABELS[model])
    axes[0].set_ylabel("REL (recovered / within-reference R²) @ L19")
    fig.suptitle(
        "Map identity vs chat reference — dashed: verdict boundary 0.5; "
        "dotted: shared-regime 1.0; hollow: C4 vs plain-dialogue reference",
        fontsize=8,
    )
    if plotted:
        savefig_paper(fig, "hero_rel_bars", dir=str(fig_dir))
    else:
        print("[i1417-figures] hero_rel_bars: no REL values yet — skipped")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hero 2 — the 2x2 prediction table with realized verdicts highlighted
# ---------------------------------------------------------------------------
H_TABLE_TEXT = {
    ("Distinct", "Shared"): "H1\n(helpful-only)",
    ("Shared", "Distinct"): "H2\n(user-directed-only)",
    ("Distinct", "Distinct"): "Conjunction\n(both required)",
    ("Shared", "Shared"): "Neither\n(generic QA structure)",
}


def fig_hero_prediction_table(summary: dict, fig_dir: Path) -> None:
    """Plan §3 outcome lattice; the realized (C2, C4) verdict pair per model is
    highlighted (markers only when both verdicts are Shared/Distinct)."""
    cells = summary.get("cells", {})
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    order = ("Shared", "Distinct")
    for i, vc2 in enumerate(order):
        for j, vc4 in enumerate(order):
            ax.add_patch(
                plt.Rectangle((j, 1 - i), 1, 1, fill=False, edgecolor="black", linewidth=1.0)
            )
            ax.text(
                j + 0.5,
                1 - i + 0.55,
                H_TABLE_TEXT[(vc2, vc4)],
                ha="center",
                va="center",
                fontsize=8,
            )
    marker_by_model = {
        "instruct": ("o", paper_palette_role("primary")),
        "pretrained": ("s", paper_palette_role("accent")),
    }
    for model in r1417.MODELS:
        vc2 = cells.get(f"{model}__c2_rude", {}).get("verdict")
        vc4 = cells.get(f"{model}__c4_exposition", {}).get("verdict")
        if vc2 in order and vc4 in order:
            i, j = order.index(vc2), order.index(vc4)
            m, c = marker_by_model[model]
            ax.plot(
                j + 0.5 + (0.12 if model == "pretrained" else -0.12),
                1 - i + 0.22,
                marker=m,
                color=c,
                markersize=9,
                linestyle="none",
                label=MODEL_LABELS[model],
            )
        else:
            print(f"[i1417-figures] prediction table: {model} verdicts unresolved ({vc2}, {vc4})")
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.35, 2.05)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["C4 exposition: Shared", "C4 exposition: Distinct"], fontsize=7.5)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["C2 rude:\nShared", "C2 rude:\nDistinct"], fontsize=7.5)
    ax.set_title("Prediction table — realized verdicts", fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.22)
        )
    ax.set_aspect("equal")
    savefig_paper(fig, "hero_prediction_table", dir=str(fig_dir))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exploratory dump
# ---------------------------------------------------------------------------
def fig_layer_r2_curves(out_dir: Path, fig_dir: Path) -> None:
    """28-layer held-out R^2 per cell (ctx arm) + per-cell shuffle-null p97.5."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    colors = paper_palette(len(r1417.CELL_ORDER))
    plotted = False
    for ax, model in zip(axes, r1417.MODELS, strict=True):
        for i, cell in enumerate(r1417.CELL_ORDER):
            cj = _load_json(out_dir / "cells" / f"cells_{cell}__{model}__ctx.json")
            if not cj or "r2_per_layer_obs" not in cj:
                continue
            r2 = np.asarray(cj["r2_per_layer_obs"], float)
            ax.plot(range(len(r2)), r2, color=colors[i], label=CELL_LABELS[cell].replace("\n", " "))
            nj = _load_json(out_dir / "cells" / f"nulls_{cell}__{model}__ctx.json")
            if nj and "null_matrix" in nj:
                nm = np.asarray(nj["null_matrix"], float)
                ax.plot(
                    range(nm.shape[1]),
                    np.nanquantile(nm, 0.975, axis=0),
                    color=colors[i],
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.7,
                )
            plotted = True
        ax.axvline(HEADLINE_LAYER, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("layer")
    axes[0].set_ylabel("held-out R² (ctx arm)")
    axes[0].legend(fontsize=5.5, frameon=False)
    if plotted:
        savefig_paper(fig, "layer_r2_curves", dir=str(fig_dir))
    else:
        print("[i1417-figures] layer_r2_curves: no cell fits — skipped")
    plt.close(fig)


def _battery_pairs_available(out_dir: Path, model: str) -> list[dict]:
    out = []
    for p in sorted((out_dir / "battery").glob(f"battery_{model}__*.json")):
        d = _load_json(p)
        if d and "rel_by_layer" in d:
            out.append(d)
    return out


def fig_transfer_matrix(out_dir: Path, fig_dir: Path) -> None:
    """cell x reference transfer R^2 @ L19 (reference map applied on cell rows)."""
    refs = ("c0_chat", "c0p_nat", "c1")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    plotted = False
    for ax, model in zip(axes, r1417.MODELS, strict=True):
        M = np.full((len(r1417.CELL_ORDER), len(refs)), np.nan)
        for d in _battery_pairs_available(out_dir, model):
            cell, ref = d["pair"]["cell"], d["pair"]["ref"]
            if d["pair"]["arm"] != "ctx":
                continue
            r2b = d.get("transfer_ref_on_cell", {}).get("r2_by_layer", {})
            v = r2b.get(str(HEADLINE_LAYER), r2b.get(HEADLINE_LAYER))
            if v is not None:
                M[r1417.CELL_ORDER.index(cell), refs.index(ref)] = float(v)
                plotted = True
        im = ax.imshow(M, vmin=-0.2, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(refs)))
        ax.set_xticklabels([REF_LABELS[r] for r in refs], fontsize=6, rotation=20)
        ax.set_yticks(range(len(r1417.CELL_ORDER)))
        ax.set_yticklabels(
            [CELL_LABELS[c].replace("\n", " ") for c in r1417.CELL_ORDER], fontsize=6.5
        )
        for (i, j), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white")
        ax.set_title(
            f"{MODEL_LABELS[model]} — reference map on cell rows (R² @ L{HEADLINE_LAYER})",
            fontsize=8,
        )
        fig.colorbar(im, ax=ax, shrink=0.85)
    if plotted:
        savefig_paper(fig, "transfer_matrix", dir=str(fig_dir))
    else:
        print("[i1417-figures] transfer_matrix: no battery pairs — skipped")
    plt.close(fig)


def fig_cosine_grid(out_dir: Path, fig_dir: Path) -> None:
    """Raw vs rotation-aligned flattened-map cosine per (cell, reference) pair,
    with the random-orthogonal chance band (null p97.5)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    plotted = False
    for ax, model in zip(axes, r1417.MODELS, strict=True):
        labels, raw, aligned, chance = [], [], [], []
        for d in _battery_pairs_available(out_dir, model):
            if d["pair"]["arm"] != "ctx":
                continue
            cn = d.get("procrustes_cosine_null_l19", {})
            if "observed_aligned_cosine" not in cn:
                continue
            labels.append(
                f"{CELL_LABELS[d['pair']['cell']].split(chr(10))[0]}\nvs {d['pair']['ref']}"
            )
            raw.append(cn.get("raw_vec_cosine", np.nan))
            aligned.append(cn["observed_aligned_cosine"])
            chance.append(cn.get("null_p975", np.nan))
        if labels:
            plotted = True
            x = np.arange(len(labels))
            ax.bar(x - 0.18, raw, width=0.34, label="raw", color=paper_palette_role("neutral"))
            ax.bar(
                x + 0.18,
                aligned,
                width=0.34,
                label="rotation-aligned",
                color=paper_palette_role("primary"),
            )
            ax.plot(
                x,
                chance,
                linestyle="none",
                marker="_",
                markersize=14,
                color=paper_palette_role("control"),
                label="chance p97.5",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=5.5, rotation=30)
        ax.set_title(MODEL_LABELS[model])
    axes[0].set_ylabel("flattened-map cosine @ L19")
    if plotted:
        axes[0].legend(fontsize=6, frameon=False)
        savefig_paper(fig, "cosine_grid", dir=str(fig_dir))
    else:
        print("[i1417-figures] cosine_grid: no pairs — skipped")
    plt.close(fig)


def fig_yield_panel(summary: dict, fig_dir: Path) -> None:
    """Judge keep-rate per (model, cell) with the 50% primary-grade floor."""
    cells = summary.get("cells", {})
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(r1417.CELL_ORDER))
    plotted = False
    for k, model in enumerate(r1417.MODELS):
        ys = [cells.get(f"{model}__{c}", {}).get("yield_frac", np.nan) for c in r1417.CELL_ORDER]
        ys = [np.nan if y is None else float(y) for y in ys]
        if np.isfinite(ys).any():
            plotted = True
        role = "primary" if model == "instruct" else "accent"
        ax.bar(
            x + (k - 0.5) * 0.36,
            ys,
            width=0.34,
            label=MODEL_LABELS[model],
            color=paper_palette_role(role),
        )
    ax.axhline(YIELD_FLOOR, color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in r1417.CELL_ORDER], fontsize=6.5)
    ax.set_ylabel("judge keep fraction")
    ax.set_title("Register-compliance yield (dashed: 50% primary-grade floor)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    if plotted:
        savefig_paper(fig, "yield_panel", dir=str(fig_dir))
    else:
        print("[i1417-figures] yield_panel: no yields — skipped")
    plt.close(fig)


def fig_judge_score_hists(out_dir: Path, fig_dir: Path) -> None:
    """Per-cell histograms of per-item mean judge scores (keep rubrics)."""
    fig, axes = plt.subplots(2, len(r1417.CELL_ORDER), figsize=(11, 4.2), sharex=True)
    plotted = False
    for r, model in enumerate(r1417.MODELS):
        for c, cell in enumerate(r1417.CELL_ORDER):
            ax = axes[r][c]
            kd = _load_json(out_dir / "judge" / f"kept_{model}_{cell}.json")
            if kd:
                for slug, per_item in kd.get("per_item_scores", {}).items():
                    if slug.startswith("diagnostic_"):
                        continue
                    means = [float(np.mean(v)) for v in per_item.values() if v]
                    if means:
                        ax.hist(means, bins=np.linspace(0, 100, 21), alpha=0.6, label=slug)
                        plotted = True
                ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
            if r == 0:
                ax.set_title(CELL_LABELS[cell], fontsize=6.5)
            if c == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=8)
            ax.tick_params(labelsize=5.5)
            if kd and r == 0 and c == 0:
                ax.legend(fontsize=4.5, frameon=False)
    fig.suptitle("Judge mean scores per item (keep rubrics; dashed: threshold 50)", fontsize=9)
    if plotted:
        savefig_paper(fig, "judge_score_hists", dir=str(fig_dir))
    else:
        print("[i1417-figures] judge_score_hists: no judge outputs — skipped")
    plt.close(fig)


def fig_y_var_ratio(summary: dict, fig_dir: Path) -> None:
    """Answer-profile variance ratio vs C0 @ L19 (collapse floor 0.5)."""
    cells = summary.get("cells", {})
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(r1417.CELL_ORDER))
    plotted = False
    for k, model in enumerate(r1417.MODELS):
        ys = [
            cells.get(f"{model}__{c}", {}).get("y_var_ratio_vs_c0", np.nan)
            for c in r1417.CELL_ORDER
        ]
        ys = [np.nan if y is None else float(y) for y in ys]
        if np.isfinite(ys).any():
            plotted = True
        role = "primary" if model == "instruct" else "accent"
        ax.bar(
            x + (k - 0.5) * 0.36,
            ys,
            width=0.34,
            label=MODEL_LABELS[model],
            color=paper_palette_role(role),
        )
    ax.axhline(
        COLLAPSE_VAR_RATIO, color=paper_palette_role("neutral"), linestyle="--", linewidth=1.0
    )
    ax.axhline(1.0, color=paper_palette_role("baseline"), linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in r1417.CELL_ORDER], fontsize=6.5)
    ax.set_ylabel("tr cov(Y) ratio vs chat reference @ L19")
    ax.set_title("Content-collapse diagnostic (dashed: demotion floor 0.5)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    if plotted:
        savefig_paper(fig, "y_var_ratio", dir=str(fig_dir))
    else:
        print("[i1417-figures] y_var_ratio: no variance ratios — skipped")
    plt.close(fig)


def _cell_r2(out_dir: Path, cell_id: str) -> float:
    cj = _load_json(out_dir / "cells" / f"cells_{cell_id}.json")
    if not cj or "r2_per_layer_obs" not in cj:
        return float("nan")
    r2 = cj["r2_per_layer_obs"]
    return float(r2[HEADLINE_LAYER]) if len(r2) > HEADLINE_LAYER else float("nan")


def fig_matched_vs_full(out_dir: Path, fig_dir: Path) -> None:
    """Matched-n companion refits vs the full kept-row fit (labeled points)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    colors = paper_palette(len(r1417.CELL_ORDER))
    plotted = False
    for model, marker in (("instruct", "o"), ("pretrained", "s")):
        for i, cell in enumerate(r1417.CELL_ORDER):
            full = _cell_r2(out_dir, f"{cell}__{model}__ctx")
            matched = [_cell_r2(out_dir, f"{cell}__{model}__ctx__matched{k}") for k in range(5)]
            matched = [m for m in matched if np.isfinite(m)]
            if not np.isfinite(full) or not matched:
                continue
            plotted = True
            ax.plot(
                [full] * len(matched),
                matched,
                marker=marker,
                linestyle="none",
                color=colors[i],
                alpha=0.7,
                markersize=4,
            )
            ax.annotate(
                f"{cell.split('_')[0]}/{MODEL_LABELS[model][0]}",
                (full, float(np.mean(matched))),
                fontsize=5,
                xytext=(2, 2),
                textcoords="offset points",
            )
    lims = ax.get_xlim()
    ax.plot(lims, lims, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("full-n kept-rows R² @ L19")
    ax.set_ylabel("matched-n draws R² @ L19")
    ax.set_title("Matched-n control (5 seeded draws per cell)", fontsize=9)
    if plotted:
        savefig_paper(fig, "matched_vs_full", dir=str(fig_dir))
    else:
        print("[i1417-figures] matched_vs_full: no matched fits — skipped")
    plt.close(fig)


def fig_all_vs_kept(out_dir: Path, fig_dir: Path) -> None:
    """All-rows companion fit vs kept-rows primary fit (filter-selection check)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    colors = paper_palette(len(r1417.CELL_ORDER))
    plotted = False
    for model, marker in (("instruct", "o"), ("pretrained", "s")):
        for i, cell in enumerate(r1417.CELL_ORDER):
            kept = _cell_r2(out_dir, f"{cell}__{model}__ctx")
            allr = _cell_r2(out_dir, f"{cell}__{model}__ctx__all")
            if not (np.isfinite(kept) and np.isfinite(allr)):
                continue
            plotted = True
            ax.plot(kept, allr, marker=marker, linestyle="none", color=colors[i], markersize=5)
            ax.annotate(
                f"{cell.split('_')[0]}/{MODEL_LABELS[model][0]}",
                (kept, allr),
                fontsize=5,
                xytext=(2, 2),
                textcoords="offset points",
            )
    lims = ax.get_xlim()
    ax.plot(lims, lims, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("kept-rows R² @ L19")
    ax.set_ylabel("all-rows R² @ L19")
    ax.set_title("Judge-filter selection check", fontsize=9)
    if plotted:
        savefig_paper(fig, "all_vs_kept", dir=str(fig_dir))
    else:
        print("[i1417-figures] all_vs_kept: no companion fits — skipped")
    plt.close(fig)


def fig_prefix_vs_ctx(out_dir: Path, fig_dir: Path) -> None:
    """Prefix-arm own-cell R^2 beside the ctx arm — the prefix arm is a
    DEGENERATE CONTROL for fixed-system-prompt single-turn cells (constant
    input regression; plan §6 analyzer rule 6), plotted for the record."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
    x = np.arange(len(r1417.CELL_ORDER))
    plotted = False
    for ax, model in zip(axes, r1417.MODELS, strict=True):
        ctx = [_cell_r2(out_dir, f"{c}__{model}__ctx") for c in r1417.CELL_ORDER]
        pre = [_cell_r2(out_dir, f"{c}__{model}__prefix") for c in r1417.CELL_ORDER]
        if np.isfinite(ctx).any() or np.isfinite(pre).any():
            plotted = True
        ax.bar(x - 0.18, ctx, width=0.34, label="context arm", color=paper_palette_role("primary"))
        ax.bar(
            x + 0.18,
            pre,
            width=0.34,
            label="prefix arm (degenerate control)",
            facecolor="none",
            edgecolor=paper_palette_role("accent"),
            linewidth=1.2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABELS[c] for c in r1417.CELL_ORDER], fontsize=6)
        ax.set_title(MODEL_LABELS[model])
    axes[0].set_ylabel("held-out R² @ L19")
    axes[0].legend(fontsize=6.5, frameon=False)
    if plotted:
        savefig_paper(fig, "prefix_vs_ctx", dir=str(fig_dir))
    else:
        print("[i1417-figures] prefix_vs_ctx: no fits — skipped")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=SCRIPT)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1417"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1417"))
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    summary = _summary(args.out_dir)
    fig_hero_rel_bars(summary, args.fig_dir)
    fig_hero_prediction_table(summary, args.fig_dir)
    fig_layer_r2_curves(args.out_dir, args.fig_dir)
    fig_transfer_matrix(args.out_dir, args.fig_dir)
    fig_cosine_grid(args.out_dir, args.fig_dir)
    fig_yield_panel(summary, args.fig_dir)
    fig_judge_score_hists(args.out_dir, args.fig_dir)
    fig_y_var_ratio(summary, args.fig_dir)
    fig_matched_vs_full(args.out_dir, args.fig_dir)
    fig_all_vs_kept(args.out_dir, args.fig_dir)
    fig_prefix_vs_ctx(args.out_dir, args.fig_dir)
    print(f"[i1417-figures] wrote figures under {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
