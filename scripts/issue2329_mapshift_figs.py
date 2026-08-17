#!/usr/bin/env python3
"""Issue #2329 mapshift figures (fork of scripts/issue2162_mapshift_figs.py).

Renders the three planned-manifest mapshift figures from the committed
``eval_results/issue_2329/mapshift/`` outputs (pure renders, no recomputation),
at the Qwen3.5-9B geometry: 32 layers, fraction-of-stack x-axes, and the 8
full-attention layers marked as dashed verticals (plan divergence: the parent
is hard-pinned to ``issue_2162`` + ``N_LAYERS = 28``).

Manifest id -> realized source (consumer-aligned realized filenames; the
manifest's ``fresh_fit.json`` / ``shift_battery.json`` / ``dv3ext.json``
spellings are the plan sketch — ``issue2329_mapshift.py`` writes these):

1. ``mapshift_r2``            <- ``fresh_fit_diagnostics.json``
2. ``mapshift_shift_prediction`` <- ``shift_summary.json`` + ``shift_cells.jsonl``
3. ``dv3_2afc``               <- ``dv3_ext.json``

Usage:
  uv run python scripts/issue2329_mapshift_figs.py
  uv run python scripts/issue2329_mapshift_figs.py --data-root <dir> --fig-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) BEFORE any heavy import

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT_DEFAULT = REPO_ROOT / "eval_results/issue_2329/mapshift"
FIG_DIR_DEFAULT = REPO_ROOT / "figures/issue_2329/mapshift"
N_MODEL_LAYERS = 32  # Qwen3.5-9B (fork-base flip; parent = 28)
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)  # issue2329_mapshift.py pin

# Plain-English source/arm labels (no arm codes on any canvas). Divergence 8:
# the parent's banked #779/#1738 arms are d=3,584 and are NOT scored here.
SOURCE_LABELS = {
    "fresh": "map fit on this bank",
    "ctxshift": "raw context shift (no map)",
}
ARM_LABELS = {
    "freshce": "map fit on this bank",
    "idbias_ce": "identity + bias",
    "identity_ce": "identity",
}
NULL_ARMS = ("shuffled", "crosstype")
KNN_K = 5


def _frac(layer: int) -> float:
    return layer / (N_MODEL_LAYERS - 1)


def _mark_full_attention(ax, layers) -> None:
    """Dashed verticals at the full-attention layers, fraction-of-stack axis."""
    for layer in sorted(layers):
        ax.axvline(_frac(int(layer)), color="0.75", lw=0.7, ls="--", zorder=0)


def _xaxis_frac(ax) -> None:
    ax.set_xlabel("depth (fraction of stack; dashed verticals = full-attention layers)")
    ax.set_xlim(-0.02, 1.02)


def fig_mapshift_r2(data_root: Path, fig_dir: Path) -> None:
    """Per-layer map skill: held-out R² with baselines + kNN retrieval."""
    diag = json.loads((data_root / "fresh_fit_diagnostics.json").read_text())
    fa = diag.get("meta", {}).get("full_attention_layers", FULL_ATTENTION_LAYERS)
    layers = sorted(int(k) for k in diag["per_layer"])
    ctx = {layer: diag["per_layer"][str(layer)]["context_grain"] for layer in layers}
    fr = [_frac(layer) for layer in layers]
    series = {
        "map fit on this bank": [ctx[la]["r2_map_ctx"] for la in layers],
        "identity + bias": [ctx[la]["r2_idbias_ctx"] for la in layers],
        "identity": [ctx[la]["r2_identity_ctx"] for la in layers],
    }
    colors = dict(zip(series, paper_palette(len(series))))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    marker = "o" if len(layers) < 8 else None
    for label, ys in series.items():
        axes[0].plot(fr, ys, color=colors[label], marker=marker, ms=5, label=label)
    axes[0].axhline(0.0, color="0.4", lw=0.8)
    axes[0].set_ylabel("held-out R² (context grain, LOCO-carrier)")
    axes[0].set_title(f"map v_C → v_A ({diag['meta']['n_train_over_d']})")
    axes[0].legend(fontsize=8, loc="best")

    knn_colors = dict(zip(("cosine", "euclidean"), paper_palette(2)))
    chance = None
    for metric in ("cosine", "euclidean"):
        accs, fr_m = [], []
        for layer in layers:
            rec = ctx[layer]["knn"].get(metric)
            if rec is None:
                continue
            fr_m.append(_frac(layer))
            accs.append(rec["acc_at_k"][str(KNN_K)])
            chance = rec["chance_at_k"][str(KNN_K)]
        axes[1].plot(fr_m, accs, color=knn_colors[metric], marker=marker, ms=5, label=metric)
    if chance is not None:
        axes[1].axhline(chance, color="0.4", lw=0.8, ls=":", label=f"chance = {chance:.3f}")
    axes[1].set_ylabel(f"kNN retrieval: P(true target in top-{KNN_K})")
    axes[1].set_title("retrieval read of the fitted map (held-out pool)")
    axes[1].legend(fontsize=8, loc="best")
    for ax in axes:
        _mark_full_attention(ax, fa)
        _xaxis_frac(ax)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    savefig_paper(fig, "mapshift_r2", dir=fig_dir)
    plt.close(fig)


def fig_mapshift_shift_prediction(data_root: Path, fig_dir: Path) -> None:
    """Predicted-vs-realized patched shift per source; per-cell points behind."""
    ss = json.loads((data_root / "shift_summary.json").read_text())
    fa = ss.get("full_attention_layers", FULL_ATTENTION_LAYERS)
    cells_rows = [
        json.loads(x)
        for x in (data_root / "shift_cells.jsonl").read_text().split("\n")
        if x.strip()
    ]
    n_surv = len(ss["survivor_cells"])
    n_all = len({r["cell"] for r in cells_rows})
    colors = dict(zip(SOURCE_LABELS, paper_palette(len(SOURCE_LABELS))))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, (view_key, title, keep) in zip(
        axes,
        [
            ("survivors", f"stored-and-used cells (n={n_surv})", lambda r: r["survivor"]),
            ("all_cells", f"all cells (n={n_all})", lambda r: True),
        ],
    ):
        view = ss["views"][view_key]
        # null envelope: min..max over (source x {shuffled, crosstype}) per layer
        band_lo, band_hi, band_x = [], [], []
        for layer in range(N_MODEL_LAYERS):
            vals = [
                view[f"{src}|L{layer}|{arm}"]["mean_cos_over_cells"]
                for src in SOURCE_LABELS
                for arm in NULL_ARMS
                if f"{src}|L{layer}|{arm}" in view
            ]
            if vals:
                band_x.append(_frac(layer))
                band_lo.append(min(vals))
                band_hi.append(max(vals))
        if band_x:
            ax.fill_between(
                band_x,
                band_lo,
                band_hi,
                color="0.6",
                alpha=0.3,
                lw=0,
                label="null patch arms (shuffled / cross-type donors)",
            )
        # per-cell points behind the summary (fresh map, steered arm)
        pts = [r for r in cells_rows if r["source"] == "fresh" and r["arm"] == "steered"]
        pts = [r for r in pts if keep(r)]
        if pts:
            ax.scatter(
                [_frac(r["layer"]) for r in pts],
                [r["mean_cos"] for r in pts],
                s=8,
                color=colors["fresh"],
                alpha=0.25,
                lw=0,
                zorder=1,
                label="per type-cell (map fit on this bank)",
            )
        for src, label in SOURCE_LABELS.items():
            xs, ys = [], []
            for layer in range(N_MODEL_LAYERS):
                rec = view.get(f"{src}|L{layer}|steered")
                if rec is not None:
                    xs.append(_frac(layer))
                    ys.append(rec["mean_cos_over_cells"])
            marker = "o" if len(xs) < 8 else None
            ax.plot(xs, ys, color=colors[src], marker=marker, ms=5, zorder=2, label=label)
        ax.axhline(0.0, color="0.4", lw=0.8)
        _mark_full_attention(ax, fa)
        ax.set_title(title)
        _xaxis_frac(ax)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("cos(map-predicted shift, realized patched shift)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig_paper(fig, "mapshift_shift_prediction", dir=fig_dir)
    plt.close(fig)


def fig_dv3_2afc(data_root: Path, fig_dir: Path) -> None:
    """Paired 2AFC accuracy vs baselines + the carrier-blocked deranged null."""
    dv3 = json.loads((data_root / "dv3_ext.json").read_text())
    per = dv3["per_config"]
    fa = dv3.get("meta", {}).get("full_attention_layers", FULL_ATTENTION_LAYERS)
    colors = dict(zip(ARM_LABELS, paper_palette(len(ARM_LABELS))))
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    # measured carrier-blocked deranged null band (union over arms per layer)
    band_x, band_lo, band_hi = [], [], []
    for layer in range(N_MODEL_LAYERS):
        bands = [
            per[f"{arm}|L{layer}|span"]["pooled"]["cosine"]["null_band"]
            for arm in ARM_LABELS
            if f"{arm}|L{layer}|span" in per
        ]
        bands = [b for b in bands if b and all(np.isfinite(b))]
        if bands:
            band_x.append(_frac(layer))
            band_lo.append(min(b[0] for b in bands))
            band_hi.append(max(b[1] for b in bands))
    if band_x:
        ax.fill_between(
            band_x,
            band_lo,
            band_hi,
            color="0.6",
            alpha=0.3,
            lw=0,
            label="carrier-blocked deranged null (95% band)",
        )
    ax.axhline(0.5, color="0.4", lw=0.8)
    for arm, label in ARM_LABELS.items():
        xs, ys, lo, hi = [], [], [], []
        for layer in range(N_MODEL_LAYERS):
            rec = per.get(f"{arm}|L{layer}|span")
            if rec is None:
                continue
            p = rec["pooled"]["cosine"]
            xs.append(_frac(layer))
            ys.append(p["acc"])
            ci = p.get("acc_ci95_clustered") or [np.nan, np.nan]
            lo.append(max(0.0, p["acc"] - ci[0]))
            hi.append(max(0.0, ci[1] - p["acc"]))
        marker = "o" if len(xs) < 8 else None
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            color=colors[arm],
            marker=marker,
            ms=5,
            lw=1.6,
            elinewidth=0.8,
            capsize=0,
            label=label,
        )
    _mark_full_attention(ax, fa)
    _xaxis_frac(ax)
    ax.set_ylabel("paired 2AFC accuracy (cosine, span pooling)")
    ax.set_title("does the predicted answer identify its own context's real answer?")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig_paper(fig, "dv3_2afc", dir=fig_dir)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR_DEFAULT)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    set_paper_style("generic")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_mapshift_r2(args.data_root, args.fig_dir)
    fig_mapshift_shift_prediction(args.data_root, args.fig_dir)
    fig_dv3_2afc(args.data_root, args.fig_dir)
    print(f"[figs] 3 figures -> {args.fig_dir}")


if __name__ == "__main__":
    main()
