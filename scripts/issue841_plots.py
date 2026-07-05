#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003, C901
# Intentional Unicode (Δ, ℓ, R², →) in labels/docstrings; the plotting dispatcher
# is a flat sequence of independent guarded figure blocks (C901 waived).
"""Issue #841 plots — the four heroes + exploratory dump (plan §6).

Reads the Stage-0 / Stage-1 JSON artifacts under ``eval_results/issue_841/`` and
writes figures under ``figures/issue_841/`` via the paper-plots rcParams. Each
figure guards on input-file presence (skips + logs when a smoke produced only a
subset), so the SAME script runs on the smoke's partial outputs and the full run.

Heroes:
  1  per-layer Δ-predictability atlas (R²_id vs transition, one line per class;
     raw + RMS-norm panels; identity=0 ref; ‖Δ‖/‖h‖ overlaid)
  2  trait-signal retention vs horizon k (one line per trait×class incl.
     id_transport; ceiling=1.0 ref; matched raw-source overlaid)
  3  Δ-R²-vs-retention scatter per (transition, class), one panel per trait
  4  Stage-1 within-condition r comparison bars per trait × mode

Exploratory dump: data-scaling curve, last-vs-mean position validation,
transport-fidelity panel, per-(condition,question) scatter behind a headline r,
per-transition Δ-error bars, GRU divergence-horizon.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_plots")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_841"
FIG_DIR = "figures/"  # savefig_paper writes <FIG_DIR>/<stem>.png
FIG_SUBDIR = "issue_841"

CLASS_LABEL = {
    "identity": "Predict-zero (identity)",
    "ridge": "Affine (ridge)",
    "mlp": "Small MLP",
    "gru": "Depth-GRU (exploratory)",
    "id_transport": "Identity transport",
}
ROW_LABEL = {
    "transported_ridge": "Transported (ridge)",
    "transported_mlp": "Transported (MLP)",
    "transported_gru": "Transported (GRU, exploratory)",
    "id_transport": "Identity transport",
    "raw_source": "Raw source read",
    "direct_hop_ridge": "Direct-hop ridge",
    "shuffled_null_ridge": "Shuffled-context null",
}


def _load(name: str) -> dict | None:
    path = EVAL_DIR / name
    if not path.exists():
        logger.warning("[plots] %s absent — skipping figures that need it", path)
        return None
    with open(path) as f:
        return json.load(f)


def _style():
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:
        logger.warning("[plots] paper_plots style unavailable (%s); using matplotlib default", e)


def _save(fig, stem: str) -> None:
    try:
        from explore_persona_space.analysis.paper_plots import savefig_paper

        savefig_paper(fig, f"{FIG_SUBDIR}/{stem}", dir=FIG_DIR)
    except Exception as e:
        logger.warning("[plots] savefig_paper failed (%s); plain savefig", e)
        out = PROJECT_ROOT / "figures" / FIG_SUBDIR
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _transitions(atlas: dict) -> list[int]:
    return list(atlas.get("transitions", []))


def hero1_atlas(atlas: dict, norm: dict | None) -> None:
    spaces = atlas.get("target_spaces", ["raw", "rmsnorm"])
    trans = _transitions(atlas)
    classes = [c for c in ("identity", "ridge", "mlp", "gru") if c in atlas.get("atlas", {})]
    fig, axes = plt.subplots(1, len(spaces), figsize=(6 * len(spaces), 4.5), squeeze=False)
    for si, space in enumerate(spaces):
        ax = axes[0][si]
        for cls in classes:
            cell = atlas["atlas"][cls].get(space)
            if not cell:
                continue
            ys = [cell.get(f"transition_{t}", {}).get("r2_id", np.nan) for t in trans]
            ax.plot(trans, ys, marker="o", ms=3, label=CLASS_LABEL.get(cls, cls))
        ax.axhline(0.0, ls="--", color="gray", lw=1, label="identity (R²=0)")
        if norm is not None and space == "raw":
            ratio = norm["norm_curve"]["delta_over_h_ratio"]
            ax2 = ax.twinx()
            ax2.plot(range(len(ratio)), ratio, color="black", ls=":", lw=1, alpha=0.6)
            ax2.set_ylabel("‖Δ‖ / ‖h‖ (norm curve)")
        ax.set_xlabel("Layer transition ℓ → ℓ+1")
        ax.set_ylabel("Held-out identity-relative R² on Δ")
        ax.set_title(f"{space} target space")
        ax.legend(fontsize=7)
    fig.suptitle("Per-layer Δ-predictability atlas")
    fig.tight_layout()
    _save(fig, "hero1_delta_atlas")


def hero2_retention(retention: dict, benchmark: dict) -> None:
    traits = list(retention.get("traits", {}).keys())
    fig, axes = plt.subplots(
        1, max(1, len(traits)), figsize=(5 * max(1, len(traits)), 4.5), squeeze=False
    )
    for ti, trait in enumerate(traits):
        ax = axes[0][ti]
        prim = retention["traits"][trait].get("primary", {})
        for cls, entries in prim.items():
            if not entries:
                continue
            ks = [e["horizon_k"] for e in entries]
            pts = [
                e.get("system", {}).get("point") if isinstance(e.get("system"), dict) else np.nan
                for e in entries
            ]
            order = np.argsort(ks)
            ax.plot(
                np.array(ks)[order],
                np.array(pts, dtype=float)[order],
                marker="o",
                ms=3,
                label=CLASS_LABEL.get(cls, cls),
            )
        ax.axhline(1.0, ls="--", color="gray", lw=1, label="ceiling (retention=1)")
        ax.set_xlabel("Prediction horizon k = ℓ* − ℓ")
        ax.set_ylabel("Retained fraction of ceiling r (system mode)")
        ax.set_title(trait)
        ax.legend(fontsize=7)
    fig.suptitle("Trait-signal retention vs horizon")
    fig.tight_layout()
    _save(fig, "hero2_retention")


def hero3_r2_vs_retention(atlas: dict, retention: dict) -> None:
    traits = list(retention.get("traits", {}).keys())
    fig, axes = plt.subplots(
        1, max(1, len(traits)), figsize=(5 * max(1, len(traits)), 4.5), squeeze=False
    )
    for ti, trait in enumerate(traits):
        ax = axes[0][ti]
        prim = retention["traits"][trait].get("primary", {})
        for cls in ("ridge", "mlp"):
            xs, ys = [], []
            for e in prim.get(cls, []):
                src = e["source"]
                r2 = (
                    atlas.get("atlas", {})
                    .get(cls, {})
                    .get("raw", {})
                    .get(f"transition_{src}", {})
                    .get("r2_id")
                )
                ret = (
                    e.get("system", {}).get("point") if isinstance(e.get("system"), dict) else None
                )
                if r2 is not None and ret is not None and np.isfinite(r2) and np.isfinite(ret):
                    xs.append(r2)
                    ys.append(ret)
            if xs:
                ax.scatter(xs, ys, label=CLASS_LABEL.get(cls, cls), s=20)
        ax.set_xlabel("Source-transition Δ-R² (atlas)")
        ax.set_ylabel("Retention at that source (system)")
        ax.set_title(trait)
        ax.legend(fontsize=7)
    fig.suptitle("Δ-R² vs behavior-retention divergence")
    fig.tight_layout()
    _save(fig, "hero3_r2_vs_retention")


def hero4_stage1_bars(benchmark: dict) -> None:
    traits = list(benchmark.get("traits", {}).keys())
    for trait in traits:
        tr = benchmark["traits"][trait]
        for scheme in ("primary", "companion"):
            sc = tr.get("schemes", {}).get(scheme)
            if not sc:
                continue
            # Use the deepest source (largest k) as the featured bar panel.
            sources = sc.get("sources", {})
            if not sources:
                continue
            src_key = min(sources, key=lambda s: int(s))  # earliest source = longest hop
            rows = sources[src_key]["rows"]
            for mode in ("system", "many_shot"):
                labels, pts, errs = [], [], []
                # ceiling first
                ceil = sc.get("ceiling_raw_target", {}).get(mode, {})
                if ceil.get("point") is not None and np.isfinite(ceil["point"]):
                    labels.append("Raw target ceiling")
                    pts.append(ceil["point"])
                    errs.append(
                        [
                            max(0.0, ceil["point"] - ceil.get("lo", ceil["point"])),
                            max(0.0, ceil.get("hi", ceil["point"]) - ceil["point"]),
                        ]
                    )
                for rk, rv in rows.items():
                    m = rv.get(mode, {})
                    if m.get("point") is None or not np.isfinite(m["point"]):
                        continue
                    labels.append(ROW_LABEL.get(rk, rk))
                    pts.append(m["point"])
                    errs.append(
                        [
                            max(0.0, m["point"] - m.get("lo", m["point"])),
                            max(0.0, m.get("hi", m["point"]) - m["point"]),
                        ]
                    )
                if not labels:
                    continue
                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.bar(range(len(pts)), pts, yerr=np.array(errs).T, capsize=3)
                ax.axhline(0.0, color="gray", lw=0.8)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
                ax.set_ylabel("Within-condition Pearson r")
                ax.set_title(
                    f"{trait} — {scheme} ℓ*={sc['target_layer']} — {mode} (source layer {src_key})"
                )
                fig.tight_layout()
                _save(fig, f"hero4_stage1_{trait}_{scheme}_{mode}")


def exploratory(atlas: dict | None, benchmark: dict | None, fidelity: dict | None) -> None:
    # Data-scaling curve.
    if atlas is not None and "scaling_curve" in atlas:
        sc = atlas["scaling_curve"]
        ns = sc.get("ns", [])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cls in ("ridge", "mlp"):
            per_t = sc.get(cls, {})
            # average R² across transitions per n
            for tkey, curve in list(per_t.items())[:6]:  # cap lines
                xs = sorted(int(n) for n in curve)
                ys = [curve[str(n)] for n in xs]
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    ms=3,
                    alpha=0.6,
                    label=f"{cls} {tkey}" if len(per_t) <= 3 else None,
                )
        ax.set_xlabel("Fit-set size n")
        ax.set_ylabel("Held-out identity-relative R²")
        ax.set_title("Data-scaling curve (§7 capture-trigger input)")
        if ns:
            ax.set_xticks(ns)
        h, _l = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=7)
        fig.tight_layout()
        _save(fig, "exploratory_scaling_curve")

    # Position validation (last vs mean).
    if atlas is not None and "position_validation" in atlas:
        pv = atlas["position_validation"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            pv["transitions"],
            pv["cx_last_ridge_r2_id"],
            marker="o",
            ms=3,
            label="last-prompt-token",
        )
        ax.plot(
            pv["transitions"],
            pv["cx_mean_ridge_r2_id"],
            marker="s",
            ms=3,
            label="mean-prompt-token",
        )
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Ridge Δ-R² (identity-relative)")
        ax.set_title(f"Position validation (curve Pearson r = {pv.get('curve_pearson_r'):.3f})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        _save(fig, "exploratory_position_validation")

    # Transport-fidelity panel.
    if fidelity is not None:
        traits = list(fidelity.get("traits", {}).keys())
        fig, axes = plt.subplots(
            1, max(1, len(traits)), figsize=(5 * max(1, len(traits)), 4.5), squeeze=False
        )
        for ti, trait in enumerate(traits):
            ax = axes[0][ti]
            prim = fidelity["traits"][trait].get("primary", {})
            for cls in ("ridge", "mlp", "gru", "direct_hop"):
                d = prim.get(cls, {})
                if not d:
                    continue
                srcs = sorted(int(s) for s in d)
                ys = [d[str(s)].get("delta_recon_r2_id", np.nan) for s in srcs]
                ax.plot(srcs, ys, marker="o", ms=3, label=cls)
            ax.set_xlabel("Source layer ℓ")
            ax.set_ylabel("Eval-context Δ-recon R²")
            ax.set_title(trait)
            ax.legend(fontsize=7)
        fig.suptitle("Transport fidelity (map-OOD disambiguator)")
        fig.tight_layout()
        _save(fig, "exploratory_transport_fidelity")

    # Per-transition Δ-error bars (median/p90/p99) for ridge raw.
    if atlas is not None:
        ridge_raw = atlas.get("atlas", {}).get("ridge", {}).get("raw", {})
        if ridge_raw:
            trans = _transitions(atlas)
            med = [
                ridge_raw.get(f"transition_{t}", {}).get("delta_err_raw", {}).get("median", np.nan)
                for t in trans
            ]
            p90 = [
                ridge_raw.get(f"transition_{t}", {}).get("delta_err_raw", {}).get("p90", np.nan)
                for t in trans
            ]
            p99 = [
                ridge_raw.get(f"transition_{t}", {}).get("delta_err_raw", {}).get("p99", np.nan)
                for t in trans
            ]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(trans, med, marker="o", ms=3, label="median")
            ax.plot(trans, p90, marker="s", ms=3, label="p90")
            ax.plot(trans, p99, marker="^", ms=3, label="p99")
            ax.set_xlabel("Layer transition")
            ax.set_ylabel("Per-context ‖Δ̂ − Δ‖ (ridge, raw)")
            ax.set_title("Δ-error tails")
            ax.legend(fontsize=7)
            fig.tight_layout()
            _save(fig, "exploratory_delta_error_tails")

    # Per-(condition,question) scatter behind a headline r.
    proj_path = EVAL_DIR / "stage1_projections.npz"
    if proj_path.exists():
        npz = np.load(proj_path, allow_pickle=False)
        keys = list(npz.keys())
        traits = sorted({k.split("__")[0] for k in keys})
        for trait in traits:
            ykey = f"{trait}__y"
            ceilkey = next((k for k in keys if k.startswith(f"{trait}__primary__ceiling")), None)
            if ykey not in keys or ceilkey is None:
                continue
            y = npz[ykey]
            x = npz[ceilkey]
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(x, y, s=12, alpha=0.5)
            ax.set_xlabel("Raw target-ceiling projection")
            ax.set_ylabel("Judged trait score (mean)")
            ax.set_title(f"{trait} — per-(condition,question) units behind the ceiling r")
            fig.tight_layout()
            _save(fig, f"exploratory_perunit_{trait}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #841 figures.")
    ap.parse_args()
    _style()
    atlas = _load("stage0_atlas.json")
    norm = _load("norm_curve.json")
    benchmark = _load("stage1_benchmark.json")
    retention = _load("retention_curve.json")
    fidelity = _load("transport_fidelity.json")

    if atlas is not None:
        hero1_atlas(atlas, norm)
    if retention is not None and benchmark is not None:
        hero2_retention(retention, benchmark)
    if atlas is not None and retention is not None:
        hero3_r2_vs_retention(atlas, retention)
    if benchmark is not None:
        hero4_stage1_bars(benchmark)
    exploratory(atlas, benchmark, fidelity)
    logger.info("[plots] done → figures/%s/", FIG_SUBDIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
