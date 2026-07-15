"""Issue #1332 follow-up ``lowdose-grid-kill-battery`` — P4 batteries + verdict (VM CPU).

Re-runs the parent's registered statistical batteries on the LOW-DOSE leakage
grid ``L_lowdose(s, t) = new trained-side mean_logp_marker - parent
per_cell_base mean_logp_marker`` (plan v8 §4 P4 + §6), reusing the parent's
vectorized code paths verbatim (``issue1332_analysis`` helpers: batched
rank-GEMM permutation null, two-way cluster bootstrap, partial Spearman, LOFO,
probe-ALIGNED split-half — no new serial per-pair/per-draw loops):

- dynamic-range / floor-share report FIRST (the low-dose regime claim is
  checked, not assumed), beside the parent grid's distribution;
- DIRECTIONAL battery at the inherited FROZEN L27 (no max-over-layer
  selection): raw rho(S_dir, L_lowdose) + stylized panels + cluster bootstrap,
  source-preserving (target-label) permutation null (B=10,000, seed 1),
  registered kill partial rho(S_dir, L | committed cos, committed JS) with
  bootstrap CI (B=2,000, seed 0), single-covariate partials, the fresh-bank
  L21 cosine covariate arm, the S_sym increment partial;
- SYMMETRIZED battery (the redundancy read): raw + partial | cos+JS + partial
  | fresh cos, collinearity gate at 0.6 with tercile + deg-2 residualization
  fallback (mirroring the parent kill block);
- LOFO hierarchy (leave-one-TARGET-family-out primary, source companion) with
  the identical baseline stack (committed cos, JS, base prior — byte-identical
  parent base per-cell files — whitened gate L14 recomputed from the parent
  capture store, predict-the-mean one-hots);
- probe-ALIGNED split-half r_LL of the NEW grid (llm-judging rule 21), r_SS
  reused from the parent; attenuation ceiling + band-vs-ceiling duty;
- the registered 4-cell verdict lattice (plan v8 §1) incl. the
  Killed-vs-Indeterminate comparator c* = 0.371 x (ceiling_lowdose / 0.988).

Outputs ``eval_results/issue_1332/lowdose/{analysis,directional_inference}.json``
+ ``lowdose/null_matrices.npz`` (per-draw vectors persisted; HF-mirrored) +
figures under ``figures/issue_1332/lowdose/``. Smoke mode writes ONLY to the
scratch roots and may point ``--lowdose-trained-dir`` at the parent #532
trained per-cell dir to exercise the identical code path on real data.

USAGE
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \\
      uv run python scripts/issue1332_lowdose_analysis.py --full
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \\
      uv run python scripts/issue1332_lowdose_analysis.py --smoke \\
        --lowdose-trained-dir eval_results/issue_532/logp_slot_followup/per_cell_trained \\
        --n-null 200 --n-boot 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1332_common as C
from issue1332_analysis import (
    BOOT_SEED,
    N_BOOT_DEFAULT,
    N_NULL_DEFAULT,
    NULL_SEED,
    R_LL_PARTITIONS,
    fresh_cosine_matrices,
    load_similarity,
    lofo_predictions,
    partial_spearman,
    r_ll_probe_aligned,
    shuffled_pairing_null,
    spearman,
    sub_matrix,
    two_way_cluster_bootstrap,
    whitened_gate_matrix,
)
from issue1332_directional import stylized_panels

logger = logging.getLogger("issue1332.lowdose_analysis")

FROZEN_LAYER = 27  # inherited pre-registered fixed axis (plan v8 §1)
FRESH_COS_LAYER = 21  # same-recipe committed-covariate layer (#536 / plan §11 item 6)

# Registered comparator constants (plan v8 §1, verbatim):
#   c* = 0.371 x (ceiling_lowdose / 0.988)
# 0.371 = the parent's committed directional partial rho | (cos, JS);
# 0.988 = the parent's committed attenuation ceiling. At runtime the
# full-precision committed values are read from the parent artifacts and
# ASSERTED to round to these registered literals (drift fails loud).
REGISTERED_PARENT_PARTIAL = 0.371
REGISTERED_PARENT_CEILING = 0.988
PARENT_L_MIN_REGISTERED = 2.19  # parent grid: all 400 off-diag cells >= 2.19 nats


def c_star(ceiling_lowdose: float, parent_partial: float, parent_ceiling: float) -> float:
    """Killed-vs-Indeterminate comparator: parent partial rescaled by the
    realized reliability-ceiling ratio (plan v8 §1). NaN-propagating."""
    return parent_partial * (ceiling_lowdose / parent_ceiling)


def lowdose_verdict_lattice(
    rho: float,
    null_q975_abs: float,
    partial_ci: tuple[float, float],
    c_star_value: float,
) -> dict:
    """The registered DISJOINT + exhaustive 4-cell lattice (plan v8 §1).

    delta_band = rho - null_q975_abs (the null band is the 97.5% quantile of
    |rho| under the source-preserving target-label permutation null, the same
    band the §6 band-vs-ceiling duty reports).

    - No-low-dose-signal        <=> delta_band <= 0
    - Replicated-at-low-dose    <=> delta_band > 0 AND partial CI excludes 0
                                    on the positive side
    - Killed-at-low-dose        <=> delta_band > 0 AND (CI wholly below 0, OR
                                    CI straddles 0 with upper bound < c*)
    - Indeterminate-at-low-dose <=> delta_band > 0 AND CI straddles 0 with
                                    upper bound >= c* (failure-to-reject /
                                    underpowered — never narrated as a kill)

    Any NaN input (degenerate grid / bootstrap / ceiling) resolves to
    Indeterminate-at-low-dose with an explicit reason — the registered
    "underpowered at this dose" escape, never a kill.
    """
    plo, phi = partial_ci
    vals = (rho, null_q975_abs, plo, phi, c_star_value)
    if any(v != v for v in vals):
        return {
            "verdict": "Indeterminate-at-low-dose",
            "delta_band": float("nan"),
            "reason": "NaN input (degenerate grid/bootstrap/ceiling) — underpowered, "
            "failure-to-reject by construction",
        }
    delta_band = rho - null_q975_abs
    if delta_band <= 0:
        verdict = "No-low-dose-signal"
    elif plo > 0:
        verdict = "Replicated-at-low-dose"
    elif phi < 0 or phi < c_star_value:
        verdict = "Killed-at-low-dose"
    else:
        verdict = "Indeterminate-at-low-dose"
    return {"verdict": verdict, "delta_band": float(delta_band), "c_star": float(c_star_value)}


# ── loading ───────────────────────────────────────────────────────────────────


def load_lowdose_leakage(trained_dir: Path, remeasured_dir: Path | None) -> dict:
    """L_lowdose = lowdose trained side - parent base side (plan §4 P3).

    Mirrors ``issue1332_common.load_leakage_matrices`` with the trained dir
    parametrized to the lowdose per-cell files; the base side is the REUSED
    parent ``per_cell_base`` (byte-identical), overridden per cell by a
    ``per_cell_base_remeasured`` file when the P3 slot-identity assert forced
    a base re-measure. Slot identity is re-asserted here per row.
    """
    import numpy as np

    sources, targets = C.family_labels()
    n_s, n_t = len(sources), len(targets)
    L = np.full((n_s, n_t), np.nan)
    L_margin = np.full((n_s, n_t), np.nan)
    base_prior = np.full((n_s, n_t), np.nan)
    per_q_trained: dict[tuple[str, str], list[float]] = {}
    per_q_base: dict[tuple[str, str], list[float]] = {}
    n_remeasured = 0
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            tr = json.loads((trained_dir / f"{s}__{t}.json").read_text())
            base_path = C.PER_CELL_DIR / "per_cell_base" / f"{s}__{t}.json"
            if remeasured_dir is not None and (remeasured_dir / f"{s}__{t}.json").exists():
                base_path = remeasured_dir / f"{s}__{t}.json"
                n_remeasured += 1
            ba = json.loads(base_path.read_text())
            for tq, bq in zip(tr["per_q"], ba["per_q"], strict=True):
                assert tq["slot_kind"] == bq["slot_kind"], (s, t)
                assert tq["n_truncated_tokens"] == bq["n_truncated_tokens"], (s, t)
            L[i, j] = tr["summary"]["mean_logp_marker"] - ba["summary"]["mean_logp_marker"]
            L_margin[i, j] = (
                tr["summary"]["mean_marker_eos_margin"] - ba["summary"]["mean_marker_eos_margin"]
            )
            base_prior[i, j] = ba["summary"]["mean_logp_marker"]
            per_q_trained[(s, t)] = [row["logp_marker"] for row in tr["per_q"]]
            per_q_base[(s, t)] = [row["logp_marker"] for row in ba["per_q"]]
    assert not np.isnan(L).any(), "missing lowdose per-cell files"
    if n_remeasured:
        logger.warning("[load] %d cells used a re-measured base side", n_remeasured)
    return {
        "sources": sources,
        "targets": targets,
        "L": L,
        "L_margin": L_margin,
        "base_prior": base_prior,
        "per_q_trained": per_q_trained,
        "per_q_base": per_q_base,
        "n_base_remeasured_cells": n_remeasured,
    }


def dynamic_range_report(L_low, L_parent, mask) -> dict:
    """Floor-share + spread of the low-dose grid beside the parent grid (§6)."""
    import numpy as np

    def _stats(M) -> dict:
        v = M[mask]
        qs = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0])
        return {
            "min": float(qs[0]),
            "q25": float(qs[1]),
            "median": float(qs[2]),
            "q75": float(qs[3]),
            "max": float(qs[4]),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "share_below_0p5_nat": float((v < 0.5).mean()),
            "share_below_1_nat": float((v < 1.0).mean()),
            "share_below_parent_min": float((v < PARENT_L_MIN_REGISTERED).mean()),
        }

    return {
        "parent_min_registered": PARENT_L_MIN_REGISTERED,
        "lowdose": _stats(L_low),
        "parent": _stats(L_parent),
    }


# ── figures ───────────────────────────────────────────────────────────────────


def _paper_style():
    import matplotlib

    matplotlib.use("Agg")
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception:
        logger.warning("paper style unavailable; default rcParams")


def forest_figure(fig_dir: Path, rows, title: str, fname: str) -> str:
    """Forest of (label, val, lo, hi) rows; xerr clamped >= 0 per errorbar site."""
    _paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(rows) + 1.5), layout="constrained")
    ys = np.arange(len(rows))[::-1]
    for y, (label, val, lo, hi) in zip(ys, rows, strict=True):
        ax.errorbar(
            [val],
            [y],
            xerr=[[max(0.0, val - lo)], [max(0.0, hi - val)]],
            fmt="o",
            color="#0072B2",
        )
        ax.text(-0.02, y, label, ha="right", va="center", transform=ax.get_yaxis_transform())
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Spearman rho (cluster-bootstrap 95% CI)")
    ax.set_title(title)
    p = fig_dir / fname
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return str(p)


def make_lowdose_figures(fig_dir: Path, ctx: dict) -> list[str]:
    """Hero forest + scatters (raw beside residualized) + heatmap + parent-vs-
    lowdose per-cell scatter + band trajectories + LOFO fold bars."""
    _paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = [forest_figure(fig_dir, ctx["forest_rows"], ctx["forest_title"], "lowdose_forest.png")]

    S_dir, L, mask = ctx["S_dir"], ctx["L"], ctx["mask"]
    sources, targets = ctx["sources"], ctx["targets"]
    styl = set(C.STYLIZED_CIDS)

    # raw scatter beside the KILL-partial residualized scatter
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    colors = np.array(
        [
            "#D55E00" if (sources[i] in styl or targets[j] in styl) else "#0072B2"
            for i in range(len(sources))
            for j in range(len(targets))
            if mask[i, j]
        ]
    )
    axes[0].scatter(S_dir[mask], L[mask], s=12, c=colors, alpha=0.7)
    axes[0].set_xlabel("S_dir (held-out transfer R^2, i->j)")
    axes[0].set_ylabel("L_lowdose (nats)")
    axes[0].set_title("raw (stylized cells orange)")
    from scipy.stats import rankdata

    def _resid(v, covs):
        Z = np.column_stack([rankdata(c) for c in covs] + [np.ones(int(mask.sum()))])
        r = rankdata(v)
        b, *_ = np.linalg.lstsq(Z, r, rcond=None)
        return r - Z @ b

    covs = [ctx["cos532"][mask], ctx["js540"][mask]]
    axes[1].scatter(_resid(S_dir[mask], covs), _resid(L[mask], covs), s=12, c=colors, alpha=0.7)
    axes[1].set_xlabel("rank(S_dir) residual | cos+JS")
    axes[1].set_ylabel("rank(L_lowdose) residual | cos+JS")
    axes[1].set_title("KILL-partial residualized")
    p = fig_dir / "lowdose_scatter_raw_and_residualized.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    figs.append(str(p))

    # parent-vs-lowdose per-cell dose scatter
    fig, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
    ax.scatter(ctx["L_parent"][mask], L[mask], s=12, alpha=0.7, color="#0072B2")
    lims = [
        min(ctx["L_parent"][mask].min(), L[mask].min()),
        max(ctx["L_parent"][mask].max(), L[mask].max()),
    ]
    ax.plot(lims, lims, color="0.6", lw=0.8)
    ax.set_xlabel("L_parent (nats)")
    ax.set_ylabel("L_lowdose (nats)")
    ax.set_title("per-cell leakage: parent vs low dose")
    p = fig_dir / "lowdose_vs_parent_scatter.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    figs.append(str(p))

    # 16x26 heatmap
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    im = ax.imshow(L, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(targets)), targets, rotation=90, fontsize=6)
    ax.set_yticks(range(len(sources)), sources, fontsize=7)
    ax.set_title("L_lowdose (trained - parent base, nats)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    p = fig_dir / "lowdose_heatmap.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    figs.append(str(p))

    # band-entry trajectories (present only after the production run)
    traj_dir = ctx.get("traj_dir")
    if traj_dir is not None and Path(traj_dir).is_dir():
        paths = sorted(Path(traj_dir).glob("*.json"))
        paths = [p for p in paths if not p.name.endswith("_bracket.json")]
        if paths:
            fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
            for tp in paths:
                d = json.loads(tp.read_text())
                if "steps" in d and "delta_nats" in d:
                    ax.plot(d["steps"], d["delta_nats"], lw=0.9, alpha=0.8, label=tp.stem)
            ax.axhspan(5.0, 12.0, color="0.85", zorder=0)
            ax.set_xlabel("optimizer step")
            ax.set_ylabel("delta log P(marker) (nats)")
            ax.set_title("band-stop trajectories (band [5, 12] shaded)")
            ax.legend(fontsize=5, ncols=4, frameon=False)
            p = fig_dir / "lowdose_band_trajectories.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            figs.append(str(p))

    # LOFO per-fold bars (registered target-axis full stack)
    per_fold = ctx["lofo"]["target_full"]["per_fold_spearman"]
    labels = list(per_fold)
    vals = [per_fold[k] if per_fold[k] == per_fold[k] else 0.0 for k in labels]
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    ax.bar(range(len(labels)), vals, color="#0072B2")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_ylabel("held-out fold Spearman")
    ax.set_title("LOFO (leave-one-target-family-out) per-fold rho — full stack")
    p = fig_dir / "lowdose_lofo_folds.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    figs.append(str(p))
    return figs


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> int:
    """Lowdose batteries: join reused S with L_lowdose, run the registered
    kill + directional inference, evaluate the 4-cell lattice, write outputs."""
    ap = argparse.ArgumentParser(description="Issue #1332 lowdose analysis (VM CPU)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    ap.add_argument("--results-dir", default=None, help="parent committed eval_results/issue_1332")
    ap.add_argument(
        "--lowdose-trained-dir",
        default=None,
        help="override the lowdose per_cell_trained dir (smoke: point at the parent "
        "#532 per_cell_trained to exercise the identical path on real data)",
    )
    ap.add_argument("--out-root", default=None, help="override the data root (capture store)")
    ap.add_argument("--n-null", type=int, default=N_NULL_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import numpy as np
    import torch

    torch.set_num_threads(args.n_threads)

    # Inputs are ALWAYS the committed parent artifacts (read-only); only
    # OUTPUTS divert to scratch under --smoke.
    res_in = C.results_dir(False, args.results_dir)
    out_root = C.results_dir(args.smoke) / "lowdose"
    fig_dir = C.figures_dir(args.smoke) / "lowdose"
    out_root.mkdir(parents=True, exist_ok=True)

    C.phase("p4_load")
    freeze = json.loads((res_in / "layer_freeze.json").read_text())
    l_star = freeze["l_star"]
    assert l_star == FROZEN_LAYER, (
        f"frozen layer drifted: layer_freeze.json says {l_star}, plan registers {FROZEN_LAYER}"
    )
    sim = load_similarity(res_in, l_star)
    families = sim["families"]

    lowdose_in = res_in / "lowdose"
    trained_dir = (
        Path(args.lowdose_trained_dir)
        if args.lowdose_trained_dir
        else lowdose_in / "per_cell_trained"
    )
    remeasured_dir = lowdose_in / "per_cell_base_remeasured"
    leak = load_lowdose_leakage(trained_dir, remeasured_dir if remeasured_dir.is_dir() else None)
    sources, targets = leak["sources"], leak["targets"]
    assert set(sources) <= set(families) and set(targets) <= set(families)
    L = leak["L"]
    mask = C.offdiag_mask(sources, targets)

    parent_leak = C.load_leakage_matrices()
    L_parent = parent_leak["L"]

    S_dir = sub_matrix(sim["S_trans"], families, sources, targets)
    S_sym = sub_matrix(sim["S_sym"], families, sources, targets)
    base = C.load_baseline_matrices()
    cos532 = base["cosine_532"]
    js540 = base["js_rb_540"]

    # Registered comparator grounding: committed full-precision parent values,
    # asserted against the plan's registered literals (drift fails loud).
    parent_dir = json.loads((res_in / "directional_inference.json").read_text())
    parent_ana = json.loads((res_in / "analysis.json").read_text())
    parent_partial = parent_dir["kill"]["partial_rho_Sdir_L_given_cos_js"]
    parent_ceiling = parent_ana["ceiling"]["attenuation_ceiling"]
    assert abs(parent_partial - REGISTERED_PARENT_PARTIAL) < 1e-3, parent_partial
    assert abs(parent_ceiling - REGISTERED_PARENT_CEILING) < 1e-3, parent_ceiling

    C.phase("p4_dynamic_range")
    dyn = dynamic_range_report(L, L_parent, mask)
    logger.info(
        "[dynamic-range] lowdose median=%.2f share<parent_min=%.2f (parent median=%.2f)",
        dyn["lowdose"]["median"],
        dyn["lowdose"]["share_below_parent_min"],
        dyn["parent"]["median"],
    )

    # covariates from the parent capture store (staged local-first -> HF)
    C.phase("p4_covariates")
    store = C.data_root(False, args.out_root) / "store" / "capture"
    for fam in families:
        C.ensure_input(store / f"{fam}.pt", f"analysis_tensors/capture/{fam}.pt")
    gate = whitened_gate_matrix(store, sources, targets, families, C.WHITENED_GATE_LAYER)
    fresh = fresh_cosine_matrices(store, sources, targets, families, [FRESH_COS_LAYER])
    fresh_cos = fresh[f"context_L{FRESH_COS_LAYER}"]
    C.write_json_atomic(
        out_root / "fresh_cos_L21.json",
        {
            "layer": FRESH_COS_LAYER,
            "arm": "context",
            "recipe": "#536 global_mean-centered cosine on mean cx_last family means "
            "(parent plan v3 §4.6 fresh-cosine recipe, recomputed from the parent "
            "capture store)",
            "sources": sources,
            "targets": targets,
            "matrix": fresh_cos.tolist(),
            "prefix_matrix": fresh[f"prefix_L{FRESH_COS_LAYER}"].tolist(),
            "reproducibility_metadata": C.reproducibility_metadata(),
        },
    )

    # ── batteries (parent helpers, batched) ───────────────────────────────────
    C.phase("p4_batteries")
    panels = stylized_panels(mask, sources, targets)
    m = panels["all"]

    def boot_rho(M_pred, mm):
        def _stat(sidx, tidx):
            sm = mm[np.ix_(sidx, tidx)]
            return spearman(M_pred[np.ix_(sidx, tidx)][sm], L[np.ix_(sidx, tidx)][sm])

        return two_way_cluster_bootstrap(
            _stat, len(sources), len(targets), n_boot=args.n_boot, seed=BOOT_SEED
        )

    def boot_partial(M_pred, covs):
        def _stat(sidx, tidx):
            sm = m[np.ix_(sidx, tidx)]
            return partial_spearman(
                M_pred[np.ix_(sidx, tidx)][sm],
                L[np.ix_(sidx, tidx)][sm],
                [cv[np.ix_(sidx, tidx)][sm] for cv in covs],
            )

        return two_way_cluster_bootstrap(
            _stat, len(sources), len(targets), n_boot=args.n_boot, seed=BOOT_SEED
        )

    def collinearity_fallback(pred) -> dict:
        pear = float(np.corrcoef(pred[m], cos532[m])[0, 1]) if int(m.sum()) >= 3 else float("nan")
        out = {"pearson_pred_cos": pear, "collinearity_gate_fired": False}
        if pear == pear and pear > 0.6:
            terciles = np.quantile(cos532[m], [1 / 3, 2 / 3])
            buckets = np.digitize(cos532[m], terciles)
            out["collinearity_gate_fired"] = True
            out["tercile_rho"] = {
                f"tercile_{b}": spearman(pred[m][buckets == b], L[m][buckets == b])
                for b in (0, 1, 2)
            }
            cosv = cos532[m]
            Z = np.column_stack([cosv, cosv**2, np.ones(len(cosv))])
            bx, *_ = np.linalg.lstsq(Z, pred[m], rcond=None)
            by, *_ = np.linalg.lstsq(Z, L[m], rcond=None)
            out["poly2_residualized_rho"] = spearman(pred[m] - Z @ bx, L[m] - Z @ by)
        return out

    def battery(pred, name: str) -> dict:
        headline = {
            pname: {
                "n_cells": int(mm.sum()),
                "rho": spearman(pred[mm], L[mm]),
                "boot": boot_rho(pred, mm),
            }
            for pname, mm in panels.items()
        }
        kill = {
            "partial_rho_given_cos_js": partial_spearman(pred[m], L[m], [cos532[m], js540[m]]),
            "boot_partial_given_cos_js": boot_partial(pred, [cos532, js540]),
            "partial_rho_given_cos": partial_spearman(pred[m], L[m], [cos532[m]]),
            "boot_partial_given_cos": boot_partial(pred, [cos532]),
            "partial_rho_given_js": partial_spearman(pred[m], L[m], [js540[m]]),
            "boot_partial_given_js": boot_partial(pred, [js540]),
            "partial_rho_given_fresh_cos_L21": partial_spearman(pred[m], L[m], [fresh_cos[m]]),
            "boot_partial_given_fresh_cos_L21": boot_partial(pred, [fresh_cos]),
            **collinearity_fallback(pred),
        }
        logger.info(
            "[battery/%s] rho=%.4f partial|cos+JS=%.4f partial|fresh=%.4f",
            name,
            headline["all"]["rho"],
            kill["partial_rho_given_cos_js"],
            kill["partial_rho_given_fresh_cos_L21"],
        )
        return {"headline": headline, "kill": kill}

    dir_b = battery(S_dir, "directional")
    sym_b = battery(S_sym, "symmetrized")
    increment = {
        "partial_rho_Sdir_L_given_Ssym": partial_spearman(S_dir[m], L[m], [S_sym[m]]),
        "boot_partial_given_Ssym": boot_partial(S_dir, [S_sym]),
        "pearson_Sdir_Ssym": float(np.corrcoef(S_dir[m], S_sym[m])[0, 1]),
    }

    C.phase("p4_null")
    null_dir_target = shuffled_pairing_null(
        S_dir, L, m, n_draws=args.n_null, seed=NULL_SEED, axis="target"
    )
    null_dir_source = shuffled_pairing_null(
        S_dir, L, m, n_draws=args.n_null, seed=NULL_SEED + 1, axis="source"
    )
    null_sym_target = shuffled_pairing_null(
        S_sym, L, m, n_draws=args.n_null, seed=NULL_SEED, axis="target"
    )
    np.savez_compressed(
        out_root / "null_matrices.npz",
        dir_target=null_dir_target,
        dir_source=null_dir_source,
        sym_target=null_sym_target,
        layer=np.asarray([FROZEN_LAYER]),
        seed=np.asarray([NULL_SEED]),
    )
    rho_dir = dir_b["headline"]["all"]["rho"]
    null_q975_abs = float(np.quantile(np.abs(null_dir_target), 0.975))
    null_q975_signed = float(np.quantile(null_dir_target, 0.975))
    null_p = float((np.abs(null_dir_target) >= abs(rho_dir)).mean())

    C.phase("p4_lofo")
    feats_base = {
        "cos532": cos532,
        "js540": js540,
        "base_prior": leak["base_prior"],
        "whitened_gate": gate["matrices"]["1x"],
    }
    lofo = {
        "target_base": lofo_predictions(L, feats_base, m, targets, sources, "target"),
        "target_full": lofo_predictions(
            L, dict(feats_base, S_sym=S_sym), m, targets, sources, "target"
        ),
        "target_full_dir": lofo_predictions(
            L, dict(feats_base, S_dir=S_dir), m, targets, sources, "target"
        ),
        "source_base": lofo_predictions(L, feats_base, m, targets, sources, "source"),
        "source_full": lofo_predictions(
            L, dict(feats_base, S_sym=S_sym), m, targets, sources, "source"
        ),
    }
    lofo["delta_cv_r2_target"] = lofo["target_full"]["cv_r2"] - lofo["target_base"]["cv_r2"]
    lofo["delta_cv_r2_target_dir"] = lofo["target_full_dir"]["cv_r2"] - lofo["target_base"]["cv_r2"]

    C.phase("p4_ceiling")
    sh_path = res_in / "splithalf" / f"splithalf_L{l_star}.json"
    r_ss = json.loads(sh_path.read_text())["r_SS"]
    r_ll = r_ll_probe_aligned(
        leak["per_q_trained"],
        leak["per_q_base"],
        sources,
        targets,
        mask,
        n_partitions=R_LL_PARTITIONS,
    )
    ceiling = (
        float(np.sqrt(max(0.0, r_ss) * max(0.0, r_ll["r_LL_spearman_brown"])))
        if r_ss == r_ss and r_ll["r_LL_spearman_brown"] == r_ll["r_LL_spearman_brown"]
        else float("nan")
    )
    band_vs_ceiling = {
        "null_band_p975_abs_rho": null_q975_abs,
        "attenuation_ceiling": ceiling,
        "margin": ceiling - null_q975_abs,
        "uninformative_by_construction": bool(ceiling == ceiling and null_q975_abs >= ceiling),
        "note": "ceiling <= band => the test is uninformative-by-construction and any "
        "non-rejection is narrated failure-to-reject / underpowered at this dose (plan §6)",
    }

    C.phase("p4_verdict")
    cstar = c_star(ceiling, parent_partial, parent_ceiling)
    dir_ci = (
        dir_b["kill"]["boot_partial_given_cos_js"]["ci_lo"],
        dir_b["kill"]["boot_partial_given_cos_js"]["ci_hi"],
    )
    lattice = lowdose_verdict_lattice(rho_dir, null_q975_abs, dir_ci, cstar)
    logger.info(
        "[lattice] %s (delta_band=%.4f, partial CI=(%.4f, %.4f), c*=%.4f, ceiling=%.4f)",
        lattice["verdict"],
        lattice.get("delta_band", float("nan")),
        dir_ci[0],
        dir_ci[1],
        cstar,
        ceiling,
    )

    C.phase("p4_sensitivity")
    L_margin = leak["L_margin"]
    L_z = (L - L.mean(axis=1, keepdims=True)) / (L.std(axis=1, keepdims=True) + 1e-12)
    sensitivity = {
        "rho_dir_margin_dv": spearman(S_dir[m], L_margin[m]),
        "rho_sym_margin_dv": spearman(S_sym[m], L_margin[m]),
        "rho_dir_within_source_z": spearman(S_dir[m], L_z[m]),
        "rho_sym_within_source_z": spearman(S_sym[m], L_z[m]),
        "rho_cos532": spearman(cos532[m], L[m]),
        "rho_js540": spearman(js540[m], L[m]),
        "rho_base_prior": spearman(leak["base_prior"][m], L[m]),
        "rho_fresh_cos_L21": spearman(fresh_cos[m], L[m]),
        "rho_whitened_gate": {tag: spearman(Mx[m], L[m]) for tag, Mx in gate["matrices"].items()},
    }

    # install record (per-source diagonal dG + band-entry deltas, when present)
    install_record = {}
    manifest_path = lowdose_in / "lowdose_manifest.json"
    if manifest_path.exists():
        install_record = json.loads(manifest_path.read_text()).get("install_record", {})

    C.phase("p4_figures")
    parent_ref = parent_dir["headline_directional"]["all"]
    forest_rows = [
        (
            "PARENT S_dir raw rho (ref)",
            parent_ref["rho"],
            parent_ref["boot"]["ci_lo"],
            parent_ref["boot"]["ci_hi"],
        ),
        (
            "PARENT S_dir partial | cos+JS (ref)",
            parent_partial,
            parent_dir["kill"]["boot_partial_given_cos_js"]["ci_lo"],
            parent_dir["kill"]["boot_partial_given_cos_js"]["ci_hi"],
        ),
        (
            "S_dir raw rho (lowdose)",
            rho_dir,
            dir_b["headline"]["all"]["boot"]["ci_lo"],
            dir_b["headline"]["all"]["boot"]["ci_hi"],
        ),
        (
            "S_sym partial | cos+JS (redundancy)",
            sym_b["kill"]["partial_rho_given_cos_js"],
            sym_b["kill"]["boot_partial_given_cos_js"]["ci_lo"],
            sym_b["kill"]["boot_partial_given_cos_js"]["ci_hi"],
        ),
        (
            "S_dir partial | cos",
            dir_b["kill"]["partial_rho_given_cos"],
            dir_b["kill"]["boot_partial_given_cos"]["ci_lo"],
            dir_b["kill"]["boot_partial_given_cos"]["ci_hi"],
        ),
        (
            "S_dir partial | JS",
            dir_b["kill"]["partial_rho_given_js"],
            dir_b["kill"]["boot_partial_given_js"]["ci_lo"],
            dir_b["kill"]["boot_partial_given_js"]["ci_hi"],
        ),
        (
            "S_dir partial | cos+JS (KILL)",
            dir_b["kill"]["partial_rho_given_cos_js"],
            *dir_ci,
        ),
        (
            "S_dir partial | fresh cos L21",
            dir_b["kill"]["partial_rho_given_fresh_cos_L21"],
            dir_b["kill"]["boot_partial_given_fresh_cos_L21"]["ci_lo"],
            dir_b["kill"]["boot_partial_given_fresh_cos_L21"]["ci_hi"],
        ),
        (
            "S_dir partial | S_sym (increment)",
            increment["partial_rho_Sdir_L_given_Ssym"],
            increment["boot_partial_given_Ssym"]["ci_lo"],
            increment["boot_partial_given_Ssym"]["ci_hi"],
        ),
    ]
    figs = make_lowdose_figures(
        fig_dir,
        {
            "forest_rows": forest_rows,
            "forest_title": "Low-dose grid — incremental-validity forest (parent refs on top)",
            "S_dir": S_dir,
            "L": L,
            "L_parent": L_parent,
            "mask": mask,
            "sources": sources,
            "targets": targets,
            "cos532": cos532,
            "js540": js540,
            "lofo": lofo,
            "traj_dir": lowdose_in / "band_trajectories",
        },
    )

    common_meta = C.reproducibility_metadata(
        {
            "followup": "lowdose-grid-kill-battery",
            "smoke": args.smoke,
            "n_null": args.n_null,
            "n_boot": args.n_boot,
            "trained_dir": str(trained_dir),
        }
    )
    analysis = {
        "l_star": l_star,
        "panel": {"sources": sources, "targets": targets, "n_offdiag_cells": int(mask.sum())},
        "dynamic_range": dyn,
        "n_base_remeasured_cells": leak["n_base_remeasured_cells"],
        "symmetrized": sym_b,
        "lofo": lofo,
        "ceiling": {"r_SS": r_ss, **r_ll, "attenuation_ceiling": ceiling},
        "band_vs_ceiling": band_vs_ceiling,
        "sensitivity": sensitivity,
        "install_record": install_record,
        "whitened_gate_lambda_1x": gate["lambda_1x"],
        "fresh_cos_L21_file": str(out_root / "fresh_cos_L21.json"),
        "figures": figs,
        "reproducibility_metadata": common_meta,
    }
    C.write_json_atomic(out_root / "analysis.json", analysis)

    directional = {
        "l_star": l_star,
        "predictor": "S_trans directional (held-out R^2 source-map -> target cells; "
        "orientation matches L[i, j]) on the LOW-DOSE grid",
        "headline_directional": dir_b["headline"],
        "null": {
            "n_draws": args.n_null,
            "seed": NULL_SEED,
            "axis": "target (source-preserving target-label permutation)",
            "p975_abs_rho_lstar": null_q975_abs,
            "p975_signed_rho_lstar": null_q975_signed,
            "p_two_sided_lstar": null_p,
            "source_axis_p975_abs": float(np.quantile(np.abs(null_dir_source), 0.975)),
            "per_draw_matrix": str(out_root / "null_matrices.npz"),
        },
        "kill": dir_b["kill"],
        "increment": increment,
        "verdict_lattice": {
            **lattice,
            "partial_ci": list(dir_ci),
            "c_star_inputs": {
                "parent_partial_committed": parent_partial,
                "parent_ceiling_committed": parent_ceiling,
                "ceiling_lowdose": ceiling,
                "registered_literals": [REGISTERED_PARENT_PARTIAL, REGISTERED_PARENT_CEILING],
            },
        },
        "reference_parent": {
            "raw_rho": parent_ref["rho"],
            "partial_cos_js": parent_partial,
            "null_p975_abs": parent_dir["null"]["p975_abs_rho_lstar"],
        },
        "figures": figs[:1],
        "reproducibility_metadata": common_meta,
    }
    C.write_json_atomic(out_root / "directional_inference.json", directional)

    if not args.smoke and not args.skip_upload:
        from issue1332_gpu_phase import upload_files

        upload_files(
            [
                (out_root / "analysis.json", f"{C.HF_PREFIX}/lowdose/analysis.json"),
                (
                    out_root / "directional_inference.json",
                    f"{C.HF_PREFIX}/lowdose/directional_inference.json",
                ),
                (out_root / "null_matrices.npz", f"{C.HF_PREFIX}/lowdose/null_matrices.npz"),
                (out_root / "fresh_cos_L21.json", f"{C.HF_PREFIX}/lowdose/fresh_cos_L21.json"),
            ],
            "issue1332 lowdose: analysis outputs + null draws",
        )

    logger.info(
        "[lowdose-analysis] verdict=%s rho_dir=%.4f band=%.4f partial|cos+JS=%.4f "
        "CI=(%.4f, %.4f) c*=%.4f sym_partial=%.4f ceiling=%.4f -> %s",
        lattice["verdict"],
        rho_dir,
        null_q975_abs,
        dir_b["kill"]["partial_rho_given_cos_js"],
        dir_ci[0],
        dir_ci[1],
        cstar,
        sym_b["kill"]["partial_rho_given_cos_js"],
        ceiling,
        out_root,
    )
    C.phase("done_lowdose_analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
