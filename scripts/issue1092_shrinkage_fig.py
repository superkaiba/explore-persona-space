"""#1092 Result-3 shrinkage figure (analysis-only, 0 GPU, VM CPU).

Recomputes the two averaged-grain prediction matrices with the deep-dive's own
recipe (identical loaders, battery-exclusion, FOLD_SEED=0 grouped folds, and
the reused `issue1092_fit_grid._fit_cv` PRESS-ridge engine — the lines below
mirror `issue1092_fair_deepdive._fit_basis` ambient branch verbatim), then
persists ONLY the small derived arrays the figure needs and renders the
uniform-shrinkage structure the banked deepdive.json summarizes as scalars:

  panel A/B: per-prefix scatter of the two CENTERED predictions projected on
             the averaged map's top principal component (instruct / base),
             with the y = x reference and the global y = alpha x fit;
  panel C:   residual variance of the single global scalar vs the
             per-dimension diagonal (the "uniform" evidence).

Checkpoints per cell (small npz); resumes by skipping cells whose npz exists.
Parity gate: the recomputed global alpha + agreement R^2 must match the
banked deepdive.json values to 1e-6 (same engine, same folds => bit-stable).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import issue1092_fair_deepdive as dd  # noqa: E402  (loaders + _prep_cell + _shrinkage)
from issue1092_fit_grid import _fit_cv, _folds_from_manifest, _r2  # noqa: E402
from issue1092_fit_grid import _basis_targets_with_info  # noqa: E402

OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_shrinkage_fig"
OUT.mkdir(parents=True, exist_ok=True)
CELLS = ["cell_inst_own", "cell_pre_own"]
CELL_LABELS = {"cell_inst_own": "Instruct model", "cell_pre_own": "Base model"}
N_PC = 3
PARITY_TOL = 1e-6


def derive_cell(cell: str, rows: list[dict]) -> dict:
    """Recompute averaged-grain predictions (ambient) and reduce to figure-sized arrays."""
    out_npz = OUT / f"shrinkage_arrays_{cell}.npz"
    if out_npz.exists():
        print(f"[skip cached] {cell}", flush=True)
        return dict(np.load(out_npz, allow_pickle=True))
    t0 = time.monotonic()
    prep = dd._prep_cell(cell, rows)
    groups, pids, folds = prep["groups"], prep["pids"], prep["folds"]
    Yb = _basis_targets_with_info(
        prep["Y_stacked"],
        "ambient",
        hidden_dim=dd.HIDDEN_DIM,
        targets=dd.TARGETS,
        projection_target="t1",
    )[0]
    Yb = np.ascontiguousarray(Yb, dtype=np.float64)
    _, pred_context = _fit_cv(prep["X_context"], Yb, folds, return_pred=True)
    Y_avg = np.stack([Yb[groups[p]].mean(0) for p in pids], axis=0)
    Xp_avg = np.stack([prep["X_prefix"][groups[p]].mean(0) for p in pids], axis=0)
    ctx_pred_avg = np.stack([pred_context[groups[p]].mean(0) for p in pids], axis=0)
    pseudo_rows = [{"prefix_id": p} for p in pids]
    folds_avg = _folds_from_manifest(
        pseudo_rows, len(pseudo_rows), group_key="prefix_id", n_folds=dd.N_FOLDS
    )
    _, prefix_pred_avg = _fit_cv(Xp_avg, Y_avg, folds_avg, return_pred=True)

    shrink = dd._shrinkage(prefix_pred_avg, ctx_pred_avg)
    banked = json.loads((dd.OUT / "deepdive.json").read_text())["cells"].get(f"{cell}/ambient")
    if banked is not None:
        d_alpha = abs(
            shrink["global_scalar_P_from_C"]["alpha"]
            - banked["shrinkage"]["global_scalar_P_from_C"]["alpha"]
        )
        d_agree = abs(
            _r2(ctx_pred_avg, prefix_pred_avg)
            - banked["recomputed_agreement"]["agreement_r2_prefixpred_vs_ctxpred"]
        )
        assert d_alpha < PARITY_TOL and d_agree < PARITY_TOL, (
            f"parity vs banked deepdive failed: d_alpha={d_alpha:.3e} d_agree={d_agree:.3e}"
        )
        print(f"[parity ok] {cell} d_alpha={d_alpha:.1e} d_agree={d_agree:.1e}", flush=True)

    Pc = prefix_pred_avg - prefix_pred_avg.mean(0, keepdims=True)
    Cc = ctx_pred_avg - ctx_pred_avg.mean(0, keepdims=True)
    # top PCs of the averaged map's centered predictions (dense SVD once, 996 x D)
    _, s, vt = np.linalg.svd(Cc, full_matrices=False)
    proj_C = Cc @ vt[:N_PC].T
    proj_P = Pc @ vt[:N_PC].T
    pc_slopes = (proj_P * proj_C).sum(0) / (proj_C * proj_C).sum(0)
    # per-dimension diagonal coefficients (for the record; figure uses resid vars)
    ssB_d = (Cc * Cc).sum(0)
    valid = ssB_d > 0
    alpha_d = np.zeros(Cc.shape[1])
    alpha_d[valid] = (Pc * Cc).sum(0)[valid] / ssB_d[valid]
    payload = {
        "proj_C": proj_C.astype(np.float32),
        "proj_P": proj_P.astype(np.float32),
        "pc_slopes": pc_slopes.astype(np.float64),
        "sv_top": s[:N_PC].astype(np.float64),
        "alpha_d": alpha_d.astype(np.float32),
        "alpha_global": np.float64(shrink["global_scalar_P_from_C"]["alpha"]),
        "resid_global": np.float64(shrink["global_scalar_P_from_C"]["resid_var_frac"]),
        "resid_diag": np.float64(shrink["per_dim_diagonal_P_from_C"]["resid_var_frac"]),
        "wall_s": np.float64(time.monotonic() - t0),
    }
    # np.savez APPENDS .npz to any filename not ending in .npz — the tmp name must end in .npz
    tmp = out_npz.with_name(out_npz.stem + ".tmp.npz")
    np.savez(tmp, **payload)
    tmp.replace(out_npz)
    print(
        f"[done {cell}] alpha={payload['alpha_global']:.4f} "
        f"pc_slopes={np.round(pc_slopes, 3).tolist()} ({payload['wall_s']:.0f}s)",
        flush=True,
    )
    return payload


def make_figure(cells: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    c_pts = paper_palette_role("baseline")
    c_fit = paper_palette_role("primary")
    inst = cells["cell_inst_own"]
    # Panels A/B: instruct-cell scatters on PC1 (fully shared) and PC2 (attenuated).
    for ax, k in zip(axes[:2], [0, 1], strict=False):
        x = np.asarray(inst["proj_C"])[:, k]
        y = np.asarray(inst["proj_P"])[:, k]
        slope = float(np.asarray(inst["pc_slopes"])[k])
        ax.scatter(x, y, s=9, alpha=0.35, color=c_pts, linewidths=0)
        lim = 1.05 * float(np.abs(x).max())
        xs = np.array([-lim, lim])
        ax.plot(xs, xs, ls="--", lw=1.2, color="0.35", label="y = x (fully shared)")
        ax.plot(xs, slope * xs, lw=1.8, color=c_fit, label=f"fit  y = {slope:.2f} x")
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(f"averaged-map prediction · PC{k + 1} (centered)")
        ax.set_ylabel(f"direct-map prediction · PC{k + 1} (centered)")
        ax.set_title(f"Instruct: per-prefix predictions on PC{k + 1}", loc="left")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    # Panel C: per-component transfer slopes vs the variance-weighted global alpha.
    ax = axes[2]
    width = 0.36
    xpos = np.arange(N_PC)
    for j, cell in enumerate(CELLS):
        d = cells[cell]
        slopes = np.asarray(d["pc_slopes"])[:N_PC]
        col = c_fit if j == 0 else c_pts
        ax.bar(xpos + (j - 0.5) * width, slopes, width, color=col, label=CELL_LABELS[cell])
        ax.axhline(float(d["alpha_global"]), ls="--", lw=1.3, color=col)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"PC{k + 1}" for k in range(N_PC)])
    ax.set_ylabel("transfer slope (direct on averaged)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-component slopes vs global α (dashed)", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    savefig_paper(
        fig,
        "summaries/prefix_vs_context_map/perprefix_shrinkage_scatter",
        dir=str(PROJECT_ROOT / "figures"),
    )
    plt.close(fig)
    print("figure written", flush=True)


def main() -> int:
    rows = dd._jsonl(dd.MANIFEST)
    print(f"manifest rows={len(rows)}", flush=True)
    cells = {cell: derive_cell(cell, rows) for cell in CELLS}
    make_figure(cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
