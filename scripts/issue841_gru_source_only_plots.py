#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ℓ, R², →, ρ) in labels/docstrings; the plotting dispatcher is a
# flat sequence of independent guarded figure blocks (C901 waived).
"""Issue #841 gru-source-only plots — heroes + exploratory dump (plan §6).

Reads the source-only-GRU Stage-0/Stage-1 artifacts (``--out-dir``) + the reloaded parent
baselines (``--parent-eval-dir``) and writes figures via the paper-plots rcParams. Every
figure guards on input presence (skips + logs when a smoke produced a subset), so the SAME
script runs on the smoke's partial outputs and the full run. House style: labeled points,
no annotations/arrows (feedback_no_plot_annotations).

Heroes:
  A  Stage-0 atlas — R²_id vs transition, source-only-GRU line overlaid on the parent
     ridge / MLP / prefix-GRU lines; raw + RMS-norm panels; identity=0 ref.
  B  Stage-1 within-condition r comparison bars per trait × mode at the k=1 source:
     source-only-GRU / ridge / prefix-GRU / raw-source / raw-target ceiling / direct-hop,
     95% CIs.
Exploratory: per-cell paired-delta forest (source-only-GRU − ridge, − prefix-GRU) with
CIs; the per-(condition,question) scatter behind a headline r (low-level data);
retention(k) curve; transport-fidelity(source-layer) overlay; the 27-transition Stage-0
win-count bar.
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
logger = logging.getLogger("issue841_gru_source_only_plots")

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_841" / "gru_source_only"
DEFAULT_PARENT_EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_841"
MODES = ("system", "many_shot")
SPACES = ("raw", "rmsnorm")


def _style():
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:
        logger.warning("[plots] paper_plots style unavailable (%s); matplotlib default", e)


def _load(path: Path) -> dict | None:
    if not path.exists():
        logger.warning("[plots] %s absent — skipping figures that need it", path)
        return None
    with open(path) as f:
        return json.load(f)


def _load_npz(path: Path) -> dict | None:
    if not path.exists():
        logger.warning("[plots] %s absent — skipping figures that need it", path)
        return None
    return dict(np.load(path))


def make_save(fig_dir: str, fig_subdir: str):
    def _save(fig, stem: str) -> None:
        try:
            from explore_persona_space.analysis.paper_plots import savefig_paper

            savefig_paper(fig, f"{fig_subdir}/{stem}", dir=fig_dir)
        except Exception as e:
            logger.warning("[plots] savefig_paper failed (%s); plain savefig", e)
            out = Path(fig_dir) / fig_subdir
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return _save


def _pt_lo_hi(d: dict | None) -> tuple[float, float, float]:
    if not isinstance(d, dict):
        return (np.nan, np.nan, np.nan)
    return (
        float(d.get("point", np.nan)) if d.get("point") is not None else np.nan,
        float(d.get("lo", np.nan)) if d.get("lo") is not None else np.nan,
        float(d.get("hi", np.nan)) if d.get("hi") is not None else np.nan,
    )


def heroA_atlas(my_stage0: dict, parent_atlas: dict, save) -> None:
    """R²_id vs transition: source-only-GRU overlaid on parent ridge / MLP / prefix-GRU."""
    trans = list(my_stage0.get("transitions", []))
    if not trans:
        return
    fig, axes = plt.subplots(1, len(SPACES), figsize=(6 * len(SPACES), 4.5), squeeze=False)
    for si, space in enumerate(SPACES):
        ax = axes[0][si]
        so = my_stage0["atlas"]["gru_source_only"].get(space, {})
        ax.plot(
            trans,
            [so.get(f"transition_{t}", {}).get("r2_id", np.nan) for t in trans],
            marker="o",
            ms=3,
            label="Source-only GRU (matched info)",
        )
        for cls, lab in (("ridge", "Affine ridge"), ("mlp", "Small MLP"), ("gru", "Prefix GRU")):
            cell = parent_atlas.get("atlas", {}).get(cls, {}).get(space, {})
            if cell:
                ax.plot(
                    trans,
                    [cell.get(f"transition_{t}", {}).get("r2_id", np.nan) for t in trans],
                    marker="o",
                    ms=3,
                    label=lab,
                )
        ax.axhline(0.0, ls="--", color="gray", lw=1, label="identity (R²=0)")
        ax.set_xlabel("Layer transition ℓ → ℓ+1")
        ax.set_ylabel("Held-out identity-relative R² on Δ")
        ax.set_title(f"{space} target space")
        ax.legend(fontsize=7)
    fig.suptitle("Stage-0 Δ-predictability atlas — source-only GRU vs baselines")
    fig.tight_layout()
    save(fig, "heroA_stage0_atlas")


def heroB_comparison_bars(my_stage1: dict, parent_bench: dict, save) -> None:
    """Within-condition r bars at the k=1 source per trait × mode (95% CIs)."""
    traits = list(my_stage1.get("traits", {}).keys())
    if not traits:
        return
    fig, axes = plt.subplots(
        len(traits), len(MODES), figsize=(6 * len(MODES), 3.4 * len(traits)), squeeze=False
    )
    for ti, trait in enumerate(traits):
        my_prim = my_stage1["traits"][trait]["schemes"].get("primary")
        par_prim = parent_bench["traits"][trait]["schemes"].get("primary")
        if not my_prim or not par_prim:
            continue
        srcs = sorted(my_prim["sources"].keys(), key=int)
        k1 = srcs[-1]  # ℓ*−1 (nearest source, k=1)
        for mi, mode in enumerate(MODES):
            ax = axes[ti][mi]
            bars = []
            my_row = my_prim["sources"][k1]["rows"]["transported_gru_source_only"].get(mode)
            bars.append(("Source-only GRU", _pt_lo_hi(my_row)))
            par_src = par_prim["sources"][k1]["rows"]
            for key, lab in (
                ("transported_ridge", "Ridge"),
                ("transported_gru", "Prefix GRU"),
                ("raw_source", "Raw source"),
                ("direct_hop_ridge", "Direct-hop"),
            ):
                bars.append((lab, _pt_lo_hi(par_src.get(key, {}).get(mode))))
            bars.append(("Raw-target ceiling", _pt_lo_hi(par_prim["ceiling_raw_target"].get(mode))))
            labels = [b[0] for b in bars]
            pts = [b[1][0] for b in bars]
            los = [b[1][0] - b[1][1] for b in bars]
            his = [b[1][2] - b[1][0] for b in bars]
            xpos = np.arange(len(bars))
            ax.bar(xpos, pts, yerr=[los, his], capsize=3)
            ax.axhline(0.0, ls="--", color="gray", lw=1)
            ax.set_xticks(xpos)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("Within-condition r")
            ax.set_title(f"{trait} / {mode} (source {k1}, k=1)")
    fig.suptitle("Stage-1 transported-monitor within-condition r (95% CI)")
    fig.tight_layout()
    save(fig, "heroB_stage1_comparison_bars")


def exp_delta_forest(my_stage1: dict, save) -> None:
    """Per-cell paired-delta forest: source-only-GRU − ridge and − prefix-GRU, with CIs."""
    traits = list(my_stage1.get("traits", {}).keys())
    if not traits:
        return
    fig, axes = plt.subplots(1, len(traits), figsize=(5.5 * len(traits), 4.5), squeeze=False)
    for ti, trait in enumerate(traits):
        ax = axes[0][ti]
        prim = my_stage1["traits"][trait]["schemes"].get("primary")
        if not prim:
            continue
        srcs = sorted(prim["sources"].keys(), key=int)
        rows_y, labels = [], []
        vr, vp = [], []  # (delta, err) vs ridge / vs prefix
        for src in srcs:
            for mode in MODES:
                d_r = prim["sources"][src]["deltas"]["vs_ridge"].get(mode, {})
                d_p = prim["sources"][src]["deltas"]["vs_prefix_gru"].get(mode, {})
                mode_word = "system" if mode == "system" else "many-shot"
                labels.append(f"source {src} · {mode_word}")
                rows_y.append(len(rows_y))
                vr.append((d_r.get("delta", np.nan), d_r.get("lo", np.nan), d_r.get("hi", np.nan)))
                vp.append((d_p.get("delta", np.nan), d_p.get("lo", np.nan), d_p.get("hi", np.nan)))
        yy = np.arange(len(labels))
        for arr, off, lab in ((vr, -0.15, "minus affine ridge"), (vp, 0.15, "minus prefix-GRU")):
            pts = [a[0] for a in arr]
            lo = [a[0] - a[1] if np.isfinite(a[1]) else 0.0 for a in arr]
            hi = [a[2] - a[0] if np.isfinite(a[2]) else 0.0 for a in arr]
            ax.errorbar(pts, yy + off, xerr=[lo, hi], fmt="o", ms=3, capsize=2, label=lab)
        ax.axvline(0.0, ls="--", color="gray", lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Paired within-condition r delta (source-only-GRU − baseline)")
        ax.set_title(f"{trait}")
        ax.legend(fontsize=7)
    fig.suptitle("Stage-1 paired-delta forest (per source × mode, 95% CI)")
    fig.tight_layout()
    save(fig, "exp_delta_forest")


def exp_per_unit_scatter(my_proj: dict, my_stage1: dict, save) -> None:
    """Low-level per-(condition,question) scatter behind a headline r (points by condition)."""
    traits = list(my_stage1.get("traits", {}).keys())
    if not traits or my_proj is None:
        return
    trait = traits[0]
    prim = my_stage1["traits"][trait]["schemes"].get("primary")
    if not prim:
        return
    srcs = sorted(prim["sources"].keys(), key=int)
    src = srcs[-1]
    # store key is f"{trait}__{scheme}__{src}__gru_source_only" (scheme=primary here);
    # HARD-FAIL on a miss rather than silently skipping the low-level-data scatter (fail-fast).
    scheme = "primary"
    xkey = f"{trait}__{scheme}__{src}__gru_source_only"
    if xkey not in my_proj:
        raise KeyError(
            f"per-unit scatter: projection key {xkey!r} absent from "
            f"gru_source_only_projections.npz (keys sample: "
            f"{[k for k in my_proj if k.endswith('__gru_source_only')][:4]}) — store-key mismatch"
        )
    x = my_proj[xkey]
    y = my_proj[f"{trait}__y"]
    cond = my_proj[f"{trait}__cond"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(x, y, c=cond, cmap="tab20", s=14)
    ax.set_xlabel(f"⟨ĥ, r_B⟩ (source-only-GRU transport, source {src})")
    ax.set_ylabel("Judged trait score (#779)")
    ax.set_title(f"{trait} primary — per-(condition,question) monitor vs score")
    fig.colorbar(sc, ax=ax, label="condition index")
    # NOTE: no fig.tight_layout() here — the paper-style layout engine + a colorbar are
    # incompatible with a post-hoc tight_layout (RuntimeError "Colorbar layout of new layout
    # engine not compatible"); the engine + savefig bbox handle spacing for the colorbar figure.
    save(fig, "exp_per_unit_scatter")


def exp_retention(retention: dict, save) -> None:
    traits = list(retention.get("traits", {}).keys())
    if not traits:
        return
    fig, axes = plt.subplots(1, len(traits), figsize=(5.0 * len(traits), 4.0), squeeze=False)
    for ti, trait in enumerate(traits):
        ax = axes[0][ti]
        prim = retention["traits"][trait].get("primary", {}).get("gru_source_only", [])
        for mode in MODES:
            ks = [e["horizon_k"] for e in prim]
            pts = [e.get(mode, {}).get("point", np.nan) for e in prim]
            order = np.argsort(ks)
            ax.plot(np.array(ks)[order], np.array(pts)[order], marker="o", ms=3, label=f"{mode}")
        ax.axhline(1.0, ls="--", color="gray", lw=1, label="ceiling (1.0)")
        ax.set_xlabel("Prediction horizon k = ℓ* − ℓ")
        ax.set_ylabel("Retention (row r / ceiling r)")
        ax.set_title(f"{trait}")
        ax.legend(fontsize=7)
    fig.suptitle("Stage-1 trait-signal retention — source-only GRU")
    fig.tight_layout()
    save(fig, "exp_retention")


def exp_fidelity(fidelity: dict, save) -> None:
    traits = list(fidelity.get("traits", {}).keys())
    if not traits:
        return
    fig, axes = plt.subplots(1, len(traits), figsize=(5.0 * len(traits), 4.0), squeeze=False)
    for ti, trait in enumerate(traits):
        ax = axes[0][ti]
        fmap = fidelity["traits"][trait].get("primary", {}).get("gru_source_only", {})
        srcs = sorted((int(s) for s in fmap), key=int)
        ax.plot(
            srcs,
            [fmap[str(s)].get("delta_recon_r2_id", np.nan) for s in srcs],
            marker="o",
            ms=3,
            label="Δ-recon R²_id",
        )
        ax.plot(
            srcs,
            [fmap[str(s)].get("cosine_hhat_vs_true", np.nan) for s in srcs],
            marker="s",
            ms=3,
            label="cos(ĥ, h_true)",
        )
        ax.set_xlabel("Source layer ℓ")
        ax.set_ylabel("Transport fidelity")
        ax.set_title(f"{trait}")
        ax.legend(fontsize=7)
    fig.suptitle("Stage-1 transport fidelity — source-only GRU maps")
    fig.tight_layout()
    save(fig, "exp_transport_fidelity")


def exp_wincount_bar(my_stage0: dict, save) -> None:
    wc = my_stage0.get("win_counts", {})
    if not wc:
        return
    fig, axes = plt.subplots(1, len(SPACES), figsize=(5.0 * len(SPACES), 3.6), squeeze=False)
    for si, space in enumerate(SPACES):
        ax = axes[0][si]
        w = wc.get(space, {})
        n = w.get("n_transitions", 0)
        labels = ["beats ridge", "beats MLP", "beats prefix-GRU"]
        vals = [
            w.get("gru_source_only_beats_ridge", 0),
            w.get("gru_source_only_beats_mlp", 0),
            w.get("gru_source_only_beats_prefix_gru", 0),
        ]
        ax.bar(np.arange(3), vals)
        ax.axhline(n / 2.0, ls="--", color="gray", lw=1, label=f"half of {n}")
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"# transitions (of {n}) source-only-GRU wins")
        ax.set_ylim(0, max(1, n))
        ax.set_title(f"{space} target space")
        ax.legend(fontsize=7)
    fig.suptitle("Stage-0 27-transition win-count — source-only GRU")
    fig.tight_layout()
    save(fig, "exp_stage0_wincount")


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #841 gru-source-only plots.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--parent-eval-dir", type=Path, default=DEFAULT_PARENT_EVAL_DIR)
    ap.add_argument("--fig-dir", default="figures/")
    ap.add_argument("--fig-subdir", default="issue_841/gru_source_only")
    args = ap.parse_args()

    _style()
    save = make_save(args.fig_dir, args.fig_subdir)

    my_stage0 = _load(args.out_dir / "stage0_gru_source_only.json")
    parent_atlas = _load(args.parent_eval_dir / "stage0_atlas.json")
    my_stage1 = _load(args.out_dir / "stage1_gru_source_only.json")
    parent_bench = _load(args.parent_eval_dir / "stage1_benchmark.json")
    retention = _load(args.out_dir / "retention_gru_source_only.json")
    fidelity = _load(args.out_dir / "transport_fidelity_gru_source_only.json")
    my_proj = _load_npz(args.out_dir / "gru_source_only_projections.npz")

    if my_stage0 and parent_atlas:
        heroA_atlas(my_stage0, parent_atlas, save)
        exp_wincount_bar(my_stage0, save)
    if my_stage1 and parent_bench:
        heroB_comparison_bars(my_stage1, parent_bench, save)
    if my_stage1:
        exp_delta_forest(my_stage1, save)
        exp_per_unit_scatter(my_proj, my_stage1, save)
    if retention:
        exp_retention(retention, save)
    if fidelity:
        exp_fidelity(fidelity, save)
    logger.info("[plots] done → %s/%s", args.fig_dir, args.fig_subdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
