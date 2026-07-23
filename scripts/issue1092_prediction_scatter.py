#!/usr/bin/env python3
"""#1092 inline: scatter of the two averaged-map constructions' held-out predictions.

For each held-out prefix, the induced map (per-row fit, predictions averaged)
and the refit (independent fit on averaged pairs) each predict the prefix's
averaged answer state. Scatter refit vs induced per (prefix, dimension) in the
pca48 basis (dense enough to plot every cell), centered per dimension, with
the y = x reference and the OLS slope — agreement + shrinkage in one view.

Also PERSISTS the held-out prediction matrices (npz, pca48) so later plots
don't refit. Analysis-only; reuses the banked engines.

Usage: uv run python scripts/issue1092_prediction_scatter.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): env caps must bind BEFORE numpy import.
load_dotenv()

import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import issue1092_inline_fair_comparison as fc  # noqa: E402
from issue1092_fit_grid import (  # noqa: E402
    _basis_targets_with_info,
    _fit_cv,
    _folds_from_manifest,
    _r2,
)

OUT_DIR = PROJECT_ROOT / "eval_results/issue_1092/inline_operator_coincidence"
FIGDIR = PROJECT_ROOT / "figures/summaries/prefix_vs_context_map"
BASIS = "pca48"


def process_cell(cell: str, rows: list[dict]) -> dict:
    t0 = time.monotonic()
    context_all = fc._load(cell, "context_end")
    t_all = [fc._load(cell, t) for t in fc.TARGETS]
    n0 = min(context_all.shape[0], min(t.shape[0] for t in t_all), len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    rows_be = [rows[int(i)] for i in be_idx]
    prefix_ids = np.asarray([r.get("prefix_id", "") for r in rows_be])
    X = np.asarray(context_all[be_idx], dtype=np.float64)
    Y_stacked = np.concatenate([np.asarray(t[be_idx], dtype=np.float64) for t in t_all], axis=1)
    del context_all, t_all
    gc.collect()
    Yb = _basis_targets_with_info(
        Y_stacked, BASIS, hidden_dim=fc.HIDDEN_DIM, targets=fc.TARGETS, projection_target="t1"
    )[0]
    del Y_stacked
    gc.collect()
    folds_row = _folds_from_manifest(
        rows_be, len(rows_be), group_key="prefix_id", n_folds=fc.N_FOLDS
    )
    groups = fc._prefix_groups(prefix_ids, fc.MIN_ROWS_PER_PREFIX)
    pids = sorted(groups)
    row_fold = np.full(len(rows_be), -1, dtype=np.int64)
    for fi, f in enumerate(folds_row):
        row_fold[f] = fi
    pref_folds: list[list[int]] = [[] for _ in folds_row]
    for j, p in enumerate(pids):
        fold_ids = np.unique(row_fold[groups[p]])
        assert fold_ids.size == 1
        pref_folds[int(fold_ids[0])].append(j)
    prefix_folds = [np.asarray(f, dtype=np.int64) for f in pref_folds if f]
    Xc_avg = np.stack([X[groups[p]].mean(0) for p in pids], axis=0)
    Y_avg = np.stack([Yb[groups[p]].mean(0) for p in pids], axis=0)

    _, P_row = _fit_cv(
        np.ascontiguousarray(X), np.ascontiguousarray(Yb), folds_row, return_pred=True
    )
    P_ind = np.stack([P_row[groups[p]].mean(0) for p in pids], axis=0)
    del P_row, X, Yb
    gc.collect()
    _, P_ref = _fit_cv(
        np.ascontiguousarray(Xc_avg), np.ascontiguousarray(Y_avg), prefix_folds, return_pred=True
    )
    np.savez_compressed(
        OUT_DIR / f"heldout_predictions_{cell}_{BASIS}.npz",
        prefix_ids=np.asarray(pids),
        P_induced=P_ind.astype(np.float32),
        P_refit=P_ref.astype(np.float32),
        Y_avg=Y_avg.astype(np.float32),
    )
    ind_c = P_ind - P_ind.mean(0)
    ref_c = P_ref - P_ref.mean(0)
    slope = float((ind_c * ref_c).sum() / (ind_c**2).sum())
    out = {
        "slope_refit_on_induced": slope,
        "r2_ind_explains_ref": float(_r2(P_ref, P_ind)),
        "r2_induced": float(_r2(Y_avg, P_ind)),
        "r2_refit": float(_r2(Y_avg, P_ref)),
        "n_points": int(np.prod(ind_c.shape)),
        "wall_s": round(time.monotonic() - t0, 1),
    }
    print(f"[{cell}] {json.dumps(out)}", flush=True)
    return out


def make_figure(stats: dict) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(3)
    all_cells = [("cell_inst_own", "Instruct model"), ("cell_pre_own", "Base model")]
    cells = [
        (c, t) for c, t in all_cells if (OUT_DIR / f"heldout_predictions_{c}_{BASIS}.npz").exists()
    ]
    fig, axes = plt.subplots(1, len(cells), figsize=(5.8 * len(cells), 5.2), squeeze=False)
    for ax, (cell, title), color in zip(axes[0], cells, [colors[0], colors[2]], strict=False):
        z = np.load(OUT_DIR / f"heldout_predictions_{cell}_{BASIS}.npz")
        ind_c = z["P_induced"] - z["P_induced"].mean(0)
        ref_c = z["P_refit"] - z["P_refit"].mean(0)
        x, y = ind_c.reshape(-1), ref_c.reshape(-1)
        lim = float(np.percentile(np.abs(np.concatenate([x, y])), 99.9))
        ax.scatter(x, y, s=3, alpha=0.12, edgecolor="none", color=color)
        ax.plot([-lim, lim], [-lim, lim], ls="--", color="0.35", lw=1.2, label="y = x (identical)")
        s = stats[cell]["slope_refit_on_induced"]
        ax.plot([-lim, lim], [-s * lim, s * lim], color="0.1", lw=1.4, label=f"OLS slope = {s:.2f}")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Induced prediction (centered)")
        ax.set_ylabel("Refit prediction (centered)")
        ax.set_title(title)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.set_aspect("equal")
    fig.suptitle(
        "Held-out predictions of the two averaged-map constructions — per (prefix, dimension), pca48, layer 14"
    )
    fig.tight_layout()
    savefig_paper(fig, "prediction_scatter_induced_vs_refit", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = fc._jsonl(fc.MANIFEST)
    stats: dict = {}
    for cell in ["cell_inst_own"]:  # panel B needs instruct only; base optional later
        stats[cell] = process_cell(cell, rows)
        gc.collect()
    (OUT_DIR / "prediction_scatter_stats.json").write_text(json.dumps(stats, indent=1))
    make_figure(stats)
    print("done")


if __name__ == "__main__":
    main()
