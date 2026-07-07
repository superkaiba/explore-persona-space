#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, Δ, M⁺, →, ×) in scientific docstrings + logs.
"""Issue #811 — per-unit scatter behind the chain-correlation forest (zero-GPU).

The clean-result's chain result reports family-clustered Spearman correlations
(forest plot) with NO per-unit view of the 480 (source, target) pairs behind
each ρ (clean-result-critic round-1, Lens 11 BLOCKING). This script regenerates
the held-out LOCO chain predictions for the key layer-14 cells by REUSING the
run's own fit path (``issue722_fit_M`` helpers — nothing re-implemented), joins
them to the measured leakage E (#537's G matrix, exactly as the run did), and
plots the raw scatter:

- columns: taught fact L14 (answer mean), taught fact L14 (turn boundary),
  harmful-compliance L14 (turn boundary — the null contrast);
- rows: base map M0, post-fine-tuning map M⁺;
- x = held-out LOCO prediction along the unit behavior direction (the chain
  read; c_C is source-keyed, so the 480 predictions collapse to ~16
  source-context clusters with only leave-one-row-out jitter);
- y = measured leakage E per (source → target) cell; color = the 7 target
  context families the bootstrap clusters over.

SANITY GATE: each panel's Spearman must reproduce the run's committed
``chain_rho_M0_Mplus_{summary}.json`` value (same fit path, same store —
the F1 offset refit reproduced Delta_med to 2.5e-11 relative); a deviation
> --repro-tol writes nothing and exits 3.

Store: the L14 subset of the run's paired store (960 npz ≈ 2.4 GB), mirrored
via ``issue811_offset_decomposition.download_store(layers=(14,))`` (resumable).
Per-point data lands in ``eval_results/issue_811/chain_scatter_points.json``
(2,880 rows exceed the sidecar embed cap, so the sidecar carries a
``data_path`` pointer instead). 0 GPU-h; CPU minutes.

``--round maxp`` (the ``maxp-winner-mapchange`` follow-up round) runs the SAME
gated refit against that round's re-extracted store
(``issue811_maxp_mapchange/analysis_tensors``, fact L14 subset — 480 npz) and
committed ``maxp-winner-mapchange/chain_rho_M0_Mplus_{summary}.json`` targets,
with columns = the taught fact L14 under all three summaries (answer mean /
turn boundary / max-pool — co-extracted from one pass, so one download serves
all three); outputs land under ``figures/issue_811/maxp-winner-mapchange/`` and
``eval_results/issue_811/maxp-winner-mapchange/chain_scatter_points.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project wrapper (NOT bare dotenv): resolves the worktree .env, sets HF_HOME,
# and applies the shared-VM thread-cap setdefaults BEFORE torch is imported.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue811_fit import STORE_PREFIX  # noqa: E402
from issue811_offset_decomposition import DEFAULT_DL_ROOT, download_store  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue811.chain_scatter")

EVAL_DIR = PROJECT_ROOT / "eval_results/issue_811"
POINTS_JSON = EVAL_DIR / "chain_scatter_points.json"
LAYER = 14
FIG_NAME = "issue_811/chain_rho_scatter_L14"

# The three key cells (columns): the reversal pair + the offset-driven null contrast.
PANEL_CELLS: tuple[tuple[str, str, str], ...] = (
    ("fact", "mean", "taught fact · L14 · answer mean"),
    ("fact", "turn_nl", "taught fact · L14 · turn boundary"),
    ("em", "turn_nl", "harmful-compliance · L14 · turn boundary"),
)

# Per-round configuration (#811 maxp-winner-mapchange follow-up). "v1" keeps the
# module defaults above verbatim; "maxp" retargets the SAME gated refit at the
# follow-up round's store + committed chain-rho JSONs, with the taught fact L14
# under all three co-extracted summaries as columns (Lens-11 per-unit companion
# to the round's forest figure). Selected via --round; main() installs the
# chosen round's values into the module globals (the issue811_fit.py pattern).
ROUNDS: dict[str, dict] = {
    "v1": {
        "eval_dir": EVAL_DIR,
        "fig_name": FIG_NAME,
        "store_prefix": STORE_PREFIX,
        "panel_cells": PANEL_CELLS,
    },
    "maxp": {
        "eval_dir": PROJECT_ROOT / "eval_results/issue_811/maxp-winner-mapchange",
        "fig_name": "issue_811/maxp-winner-mapchange/chain_rho_scatter_L14_fact",
        "store_prefix": "issue811_maxp_mapchange/analysis_tensors",
        "panel_cells": (
            ("fact", "mean", "taught fact · L14 · answer mean"),
            ("fact", "turn_nl", "taught fact · L14 · turn boundary"),
            ("fact", "maxp", "taught fact · L14 · max-pool"),
        ),
    },
}
MAP_LABEL = {"M0": "base map", "Mplus": "post-fine-tuning map"}
E_LABEL = {
    "fact": "measured leakage (stated-fact rate, trained − base)",
    "em": "measured leakage (misalignment rate, trained − base)",
}
FAMILY_LABEL = {
    "sp": "persona targets",
    "wc": "WildChat targets",
    "icl": "few-shot-demo targets",
    "reph": "rephrase targets",
    "fmt": "format-instruction targets",
    "binst": "behavior-naming targets",
    "default": "default assistant",
}
FAMILY_ORDER = ("sp", "wc", "icl", "reph", "fmt", "binst", "default")
# Plain-English source-context names (same universe as the F1 figure's contexts).
SOURCE_LABEL = {
    "default": "default assistant",
    "fmt_code": "code-format instr.",
    "fmt_json": "JSON-format instr.",
    "icl_k2": "2-shot demos",
    "icl_k8": "8-shot demos",
    "reph_casual": "casual rephrase",
    "reph_imp": "imperative rephrase",
    "reph_polite": "polite rephrase",
    "sp_doctor": "doctor persona",
    "sp_ph1": "PersonaHub 1",
    "sp_ph2": "PersonaHub 2",
    "sp_swe": "software-engineer persona",
    "wc_long_write": "WildChat long writing",
    "wc_short_advice": "WildChat short advice",
    "wc_short_code": "WildChat short code",
}


def _source_label(cid: str) -> str:
    """Plain-English source-context name (behavior-specific binst_* → one label)."""
    if cid.startswith("binst"):
        return "behavior-naming instr."
    return SOURCE_LABEL.get(cid, cid)


def compute_cell(behavior: str, summary: str, cells: list, rb_main: dict, rb_fact) -> dict:
    """LOCO chain predictions + Spearman for one (behavior, L14, summary) cell.

    Mirrors ``issue722_fit_M.fit_cell``'s chain-ρ block exactly (same helpers,
    same order): PCA basis from V0, LOCO ridge preds for M0 (C0→V0_64) and M⁺
    (Cplus→Vplus_64), projected back and dotted with the unit behavior
    direction, joined to E. Returns per-row arrays + the two Spearmans.
    """
    stacks = loadact.stack_for_fit(cells)
    C0, Cplus, V0, Vplus = stacks["C0"], stacks["Cplus"], stacks["V0"], stacks["Vplus"]
    n = C0.shape[0]
    assert n == loadact.EXPECTED_CELLS_PER_BEHAVIOR_LAYER, (behavior, summary, n)
    r_hat = fitM._r_hat_for(behavior, LAYER, rb_main, rb_fact)
    pca_basis = fitM._pca_basis_v0(V0, fitM.TARGET_DIM)
    V0_64 = fitM._to64(V0, pca_basis)
    Vplus_64 = fitM._to64(Vplus, pca_basis)
    E = fitM._load_E(behavior, stacks["cell_keys"])
    keep = ~np.isnan(E)
    assert keep.sum() >= 4, (behavior, summary, int(keep.sum()))
    Ek = E[keep]
    m0_loco = fitM._ridge_loco_pred(C0, V0_64)
    mplus_loco = fitM._ridge_loco_pred(Cplus, Vplus_64)
    rho_m0, chain_m0 = fitM._chain_rho_one(m0_loco[keep], pca_basis, r_hat, Ek)
    rho_mplus, chain_mplus = fitM._chain_rho_one(mplus_loco[keep], pca_basis, r_hat, Ek)
    kidx = np.where(keep)[0]
    return {
        "rho_M0": rho_m0,
        "rho_Mplus": rho_mplus,
        "chain_M0": chain_m0,
        "chain_Mplus": chain_mplus,
        "E": Ek,
        "families": [stacks["families"][i] for i in kidx],
        "source_cids": [stacks["source_cids"][i] for i in kidx],
        "target_cids": [stacks["target_cids"][i] for i in kidx],
    }


def _annotate_extreme_sources(ax, chain: np.ndarray, sources: list[str]) -> None:
    """Label the min/max source-context clusters under the panel (kept legible).

    Bottom-row panels only (the caller gates on row): two horizontal labels in
    reserved space below the data, at the extreme clusters' mean x — far apart
    by construction, so they never collide.
    """
    src = np.asarray(sources, dtype=object)
    means = {s: float(chain[src == s].mean()) for s in sorted(set(sources))}
    ylo, yhi = ax.get_ylim()
    yr = yhi - ylo
    ax.set_ylim(ylo - 0.12 * yr, yhi)
    for s in (min(means, key=means.get), max(means, key=means.get)):
        ax.text(
            means[s],
            ylo - 0.10 * yr,
            _source_label(s),
            fontsize=6.5,
            ha="center",
            va="bottom",
            color="#3B3B3B",
        )


def make_figure(results: dict[tuple[str, str], dict]) -> None:
    """2 rows (base / post-fine-tuning map) × 3 key cells, 480 pairs per panel."""
    set_paper_style("blog")
    fam_colors = dict(zip(FAMILY_ORDER, paper_palette_blog(len(FAMILY_ORDER)), strict=True))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8))
    for col, (beh, summary, title) in enumerate(PANEL_CELLS):
        res = results[(beh, summary)]
        fams = np.asarray(res["families"], dtype=object)
        for row, key in enumerate(("M0", "Mplus")):
            ax = axes[row][col]
            chain, E = res[f"chain_{key}"], res["E"]
            for fam in FAMILY_ORDER:
                m = fams == fam
                if not m.any():
                    continue
                ax.scatter(chain[m], E[m], s=13, alpha=0.6, color=fam_colors[fam], linewidths=0.0)
            # Reserve headroom so the map-name/Spearman annotation never sits on data.
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(ylo, yhi + 0.16 * (yhi - ylo))
            if row == 1:
                _annotate_extreme_sources(ax, chain, res["source_cids"])
            rho = res[f"rho_{key}"]
            ax.text(
                0.03,
                0.97,
                f"{MAP_LABEL[key]} — Spearman ρ = {rho:+.3f}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )
            if row == 0:
                ax.set_title(title, fontsize=10)
            if row == 1:
                ax.set_xlabel("held-out prediction along behavior direction", fontsize=8.5)
            if col == 0:
                # Short label — the full definition (trained − base judged rate)
                # lives in the caption; the two rows' labels otherwise collide.
                ax.set_ylabel("measured leakage")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=fam_colors[f], label=FAMILY_LABEL[f])
        for f in FAMILY_ORDER
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    # 2,880 points exceed the sidecar embed cap — point data lands in POINTS_JSON,
    # pointed at from the sidecar (data_path) below.
    paths = savefig_paper(fig, FIG_NAME, dir="figures/", embed_data=False)
    plt.close(fig)
    meta_path = Path(paths.get("meta", f"figures/{FIG_NAME}.meta.json"))
    meta = json.loads(meta_path.read_text())
    meta["data_path"] = str(POINTS_JSON.relative_to(PROJECT_ROOT))
    meta_path.write_text(json.dumps(meta, indent=2))
    for fmt, path in paths.items():
        logger.info("wrote %s: %s", fmt, path)


def _results_from_points_json() -> dict[tuple[str, str], dict]:
    """Rebuild the plotting inputs from the committed per-point JSON (no store, no fits).

    The Spearman per panel is recomputed from the loaded points and must match
    the run's committed value exactly (the points were produced by the gated
    refit) — asserted, so a stale/edited points file cannot silently replot.
    """
    from scipy.stats import spearmanr

    payload = json.loads(POINTS_JSON.read_text())
    run_rho = {
        s: json.loads((EVAL_DIR / f"chain_rho_M0_Mplus_{s}.json").read_text())["cells"]
        for s in {s for _b, s, _t in PANEL_CELLS}
    }
    results: dict[tuple[str, str], dict] = {}
    for beh, summary, _title in PANEL_CELLS:
        rows = [r for r in payload["rows"] if r["behavior"] == beh and r["summary"] == summary]
        assert rows, (beh, summary, "no rows in points JSON")
        res = {
            "chain_M0": np.asarray([r["chain_pred_M0"] for r in rows]),
            "chain_Mplus": np.asarray([r["chain_pred_Mplus"] for r in rows]),
            "E": np.asarray([r["E"] for r in rows]),
            "families": [r["target_family"] for r in rows],
            "source_cids": [r["source_cid"] for r in rows],
        }
        run_cell = run_rho[summary][f"{beh}/L{LAYER}"]
        for key, run_key in (("M0", "rho_M0_ridge"), ("Mplus", "rho_Mplus_ridge")):
            rho = float(spearmanr(res[f"chain_{key}"], res["E"]).correlation)
            assert abs(rho - run_cell[run_key]) < 1e-9, (beh, summary, key, rho, run_cell[run_key])
            res[f"rho_{key}"] = rho
        results[(beh, summary)] = res
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #811 chain-ρ per-unit scatter (L14)")
    ap.add_argument(
        "--round",
        choices=tuple(ROUNDS),
        default="v1",
        help="which #811 round to refit: v1 (turn_nl round, default) or maxp "
        "(the maxp-winner-mapchange follow-up round — fact L14, three summaries)",
    )
    ap.add_argument("--dl-root", type=Path, default=DEFAULT_DL_ROOT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--repro-tol", type=float, default=1e-3, help="max |ρ_refit − ρ_run|")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="re-render the figure from the committed chain_scatter_points.json "
        "(no store download, no fits; each panel's Spearman re-asserted vs the run)",
    )
    args = ap.parse_args()

    # Install the chosen round's targets into the module globals the helpers
    # read (compute_cell / make_figure / _results_from_points_json) — the
    # issue811_fit.py round-parameterization pattern.
    global EVAL_DIR, POINTS_JSON, FIG_NAME, PANEL_CELLS
    rnd = ROUNDS[args.round]
    EVAL_DIR = rnd["eval_dir"]
    POINTS_JSON = EVAL_DIR / "chain_scatter_points.json"
    FIG_NAME = rnd["fig_name"]
    PANEL_CELLS = rnd["panel_cells"]
    store_prefix = rnd["store_prefix"]
    logger.info("[phase=setup] round=%s store_prefix=%s", args.round, store_prefix)

    if args.plot_only:
        make_figure(_results_from_points_json())
        return 0

    import torch

    torch.set_num_threads(max(1, args.num_threads))
    fit658.DEVICE = fit658._resolve_device("auto")
    logger.info("[phase=setup] device=%s", fit658.DEVICE)

    behaviors = tuple(sorted({beh for beh, _s, _t in PANEL_CELLS}))
    local_root = args.dl_root / store_prefix
    if args.skip_download:
        assert local_root.is_dir(), f"--skip-download but no local mirror at {local_root}"
    else:
        local_root, n_dl = download_store(
            args.dl_root, behaviors, workers=args.workers, layers=(LAYER,), prefix=store_prefix
        )
        logger.info("[phase=download] %d files fetched", n_dl)

    rb_main = fitM._load_rb_main()
    rb_fact = fitM._load_rb_fact() if "fact" in behaviors else None
    assert not ("fact" in behaviors and rb_fact is None), "fact requested but r_b_fact missing"
    layout = loadact.list_store_layout_local(local_root, behaviors)

    # Run's committed chain-ρ JSONs — the reproduction gate targets.
    run_rho = {
        s: json.loads((EVAL_DIR / f"chain_rho_M0_Mplus_{s}.json").read_text())["cells"]
        for s in {s for _b, s, _t in PANEL_CELLS}
    }

    results: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []
    worst_dev = 0.0
    for beh, summary, _title in PANEL_CELLS:
        cells_by = loadact.load_cells(
            behaviors=(beh,),
            layers=(LAYER,),
            max_sources=None,
            max_targets_per_source=None,
            streamer=loadact._Streamer(local_root=local_root),
            strict_counts=True,
            layout=layout,
            summary=summary,
        )
        res = compute_cell(beh, summary, cells_by[(beh, LAYER)], rb_main, rb_fact)
        run_cell = run_rho[summary][f"{beh}/L{LAYER}"]
        for key, run_key in (("rho_M0", "rho_M0_ridge"), ("rho_Mplus", "rho_Mplus_ridge")):
            dev = abs(res[key] - run_cell[run_key])
            worst_dev = max(worst_dev, dev)
            logger.info(
                "[phase=repro] %s L%d %s %s: refit %.6f vs run %.6f (|dev| %.2e)",
                beh,
                LAYER,
                summary,
                key,
                res[key],
                run_cell[run_key],
                dev,
            )
            if dev > args.repro_tol:
                logger.error(
                    "[phase=repro] FAIL: %s deviates %.3e > tol %.0e — NOT plotting "
                    "from a divergent refit",
                    key,
                    dev,
                    args.repro_tol,
                )
                return 3
        results[(beh, summary)] = res
        for i in range(res["E"].size):
            rows.append(
                {
                    "behavior": beh,
                    "layer": LAYER,
                    "summary": summary,
                    "source_cid": res["source_cids"][i],
                    "target_cid": res["target_cids"][i],
                    "target_family": res["families"][i],
                    "chain_pred_M0": float(res["chain_M0"][i]),
                    "chain_pred_Mplus": float(res["chain_Mplus"][i]),
                    "E": float(res["E"][i]),
                }
            )

    POINTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    POINTS_JSON.write_text(
        json.dumps(
            {
                "meta": {
                    "issue": 811,
                    "figure": Path(FIG_NAME).name,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "max_repro_dev_rho": worst_dev,
                    "n_rows": len(rows),
                },
                "rows": rows,
            },
            indent=1,
        )
    )
    logger.info("[phase=points] %d rows → %s", len(rows), POINTS_JSON)
    make_figure(results)
    # No [phase=done] tag: that token is RESERVED for the dispatcher's single
    # terminal line (pod-side reporting contract, #545); this script is not
    # dispatcher-invoked, but keep the log surface consistent anyway.
    logger.info("chain-scatter figure complete: max |ρ_refit − ρ_run| = %.3e", worst_dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
