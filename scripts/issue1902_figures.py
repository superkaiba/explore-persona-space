#!/usr/bin/env python3
"""Issue #1902 P5 — figures (VM-side, off-pod; plan v4 §4 P5 / §6).

Reads the committed P4 eval JSONs (+ the percell npz shards for the
per-context histograms) and renders the §6 figure set via the /paper-plots
conventions (``set_paper_style`` + ``savefig_paper``: colorblind-safe
palette, error bars, self-describing axes, meta.json sidecars). One color =
one meaning across the whole set (stage colors fixed once). RSS is trivially
< 16 GB by construction: JSONs + (n,) float arrays loaded one at a time.

Usage (VM, after the P4 eval results land in git)::

    uv run python scripts/issue1902_figures.py \
        --eval-dir eval_results/issue_1902 --fig-dir figures/issue_1902

Smoke: point --eval-dir/--fig-dir at the smoke out-root copies — never the
repo tree (smoke outputs never overwrite committed artifacts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

STAGES = ["B", "S", "D", "R"]
STAGE_LABEL = {"B": "base", "S": "SFT", "D": "DPO", "R": "RLVR"}
# One color = one meaning across every figure: stage -> fixed palette slot.
_PAL = None


def stage_color(m: str) -> str:
    global _PAL
    if _PAL is None:
        _PAL = paper_palette(len(STAGES))
    return _PAL[STAGES.index(m)]


def pair_label(key: str) -> str:
    """Plain-English label for a transfer/operator pair key, e.g. 'B->S' -> 'base→SFT'."""
    return "→".join(STAGE_LABEL.get(p, p) for p in key.split("->"))


def _load(eval_dir: Path, rel: str) -> dict:
    with open(eval_dir / rel, encoding="utf-8") as f:
        return json.load(f)


def _present_stages(grid: dict) -> list[str]:
    return [m for m in STAGES if any(k.startswith(f"diag_{m}_") for k in grid["cells"])]


# ── hero 1: diagonal Q vs stage ──────────────────────────────────────────────


def _prefix_diag_folds(eval_dir: Path, stages: list[str], layer: int) -> dict[str, dict]:
    """Per-fold prefix-arm diagonal reads (ridge r2 + identity) from sweep units."""
    units = eval_dir / "fits" / "units"
    out: dict[str, dict] = {}
    for m in stages:
        r2s, idents = [], []
        for p in sorted(units.glob(f"sweep_{m}_multi_f*.json")):
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)["arms"].get("pre", {}).get(str(layer))
            if rec:
                r2s.append(rec["r2"])
                idents.append(rec.get("identity_r2", np.nan))
        out[m] = {"r2": r2s, "identity": idents}
    return out


def fig_hero_diag(eval_dir: Path, fig_dir: Path) -> None:
    grid = _load(eval_dir, "fits/grid_cells.json")
    stages = _present_stages(grid)
    panels = [("single", "ctx"), ("multi", "ctx"), ("multi", "pre")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=True)
    pre_folds = _prefix_diag_folds(eval_dir, stages, int(grid.get("layer_star_p", 31)))
    for ax, (corpus, arm) in zip(axes, panels):
        xs = np.arange(len(stages))
        ridge, ci_lo, ci_hi, ident, nulls = [], [], [], [], []
        for m in stages:
            if arm == "pre":
                fr = pre_folds.get(m, {})
                r2s = fr.get("r2", [])
                ridge.append(float(np.mean(r2s)) if r2s else np.nan)
                # fold min/max spread (no cluster-bootstrap CI persisted for the pre arm)
                ci_lo.append(float(np.min(r2s)) if r2s else np.nan)
                ci_hi.append(float(np.max(r2s)) if r2s else np.nan)
                ident.append(
                    float(np.mean(fr.get("identity", []))) if fr.get("identity") else np.nan
                )
            else:
                cell = grid["cells"].get(f"diag_{m}_{corpus}_{arm}", {})
                ridge.append(cell.get("r2_at_star", np.nan))
                lo, hi = cell.get("ci_frozen_at_star", [np.nan, np.nan])
                ci_lo.append(lo)
                ci_hi.append(hi)
                ib = cell.get("baselines_at_star", {}).get("identity_r2") or []
                ident.append(np.nanmean([v for v in ib if v is not None]) if ib else np.nan)
                nulls.extend(cell.get("shuffle_null_r2", []))
        ridge = np.asarray(ridge, float)
        # non-negative offsets from the value (mpl xerr/yerr contract)
        yerr = np.stack(
            [
                np.maximum(0, ridge - np.asarray(ci_lo, float)),
                np.maximum(0, np.asarray(ci_hi, float) - ridge),
            ]
        )
        err_label = (
            "ridge (OOF R², layer*; 95% cluster CI)"
            if arm == "ctx"
            else "ridge (OOF R², layer*; fold min–max)"
        )
        ax.errorbar(
            xs,
            ridge,
            yerr=np.nan_to_num(yerr),
            fmt="o-",
            color=paper_palette(3)[0],
            capsize=3,
            label=err_label,
        )
        # MLP twin exists for (single, ctx) and (multi, pre) only — matched arm/corpus.
        mlp_key = {"single": "ctx", "multi": "pre"}
        if mlp_key[corpus] == arm:
            mlp_star = grid.get("mlp_layer_star", {}).get(arm)
            mlp_vals = [
                grid.get("mlp", {})
                .get(f"mlp_{arm}_{m}{m}", {})
                .get("per_layer", {})
                .get(str(mlp_star), np.nan)
                for m in stages
            ]
            ax.plot(xs, mlp_vals, "s--", color=paper_palette(3)[1], label="MLP (own layer)")
        ax.plot(xs, ident, "^:", color=paper_palette(3)[2], label="identity+bias baseline")
        if nulls:
            ax.axhspan(
                float(np.nanmin(nulls)),
                float(np.nanmax(nulls)),
                color="grey",
                alpha=0.25,
                label="shuffled-pairing null band",
            )
        ax.set_xticks(xs, [STAGE_LABEL[m] for m in stages])
        title = {
            ("single", "ctx"): "single-turn corpus — context arm",
            ("multi", "ctx"): "multi-turn corpus — context arm",
            ("multi", "pre"): "multi-turn corpus — prefix arm",
        }[(corpus, arm)]
        ax.set_title(title)
        ax.set_xlabel("post-training stage")
    axes[0].set_ylabel("held-out pooled OOF R² at layer*")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[2].legend(fontsize=7, loc="lower left")
    fig.suptitle("Diagonal context→answer map quality across the OLMo-2 chain", y=1.02)
    savefig_paper(fig, "hero1_diag_q_vs_stage", dir=fig_dir)
    plt.close(fig)


def fig_hero_diag_folds(eval_dir: Path, fig_dir: Path) -> None:
    """Low-level per-fold view behind hero 1: every fold's diagonal OOF R²."""
    grid = _load(eval_dir, "fits/grid_cells.json")
    stages = _present_stages(grid)
    units = eval_dir / "fits" / "units"
    layer_p = int(grid.get("layer_star_p", 31))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=True)
    panels = [
        ("single", "ctx", "single-turn corpus — context arm"),
        ("multi", "ctx", "multi-turn corpus — context arm"),
        ("multi", "pre", "multi-turn corpus — prefix arm"),
    ]
    for ax, (corpus, arm, title) in zip(axes, panels):
        for xi, m in enumerate(stages):
            vals = []
            if arm == "ctx":
                for p in sorted(units.glob(f"star_{m}_{corpus}_f*.json")):
                    with open(p, encoding="utf-8") as f:
                        vals.append(json.load(f)["r2"])
            else:
                for p in sorted(units.glob(f"sweep_{m}_multi_f*.json")):
                    with open(p, encoding="utf-8") as f:
                        rec = json.load(f)["arms"].get("pre", {}).get(str(layer_p))
                    if rec:
                        vals.append(rec["r2"])
            xjit = xi + np.linspace(-0.15, 0.15, num=len(vals))
            ax.scatter(xjit, vals, s=18, color=stage_color(m), label=None)
            for k, v in enumerate(vals):
                ax.annotate(
                    f"f{k}", (xjit[k], v), fontsize=5, xytext=(2, 2), textcoords="offset points"
                )
            ax.plot([xi - 0.2, xi + 0.2], [np.mean(vals)] * 2, color=stage_color(m), lw=1.6)
        ax.set_xticks(range(len(stages)), [STAGE_LABEL[m] for m in stages])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("post-training stage")
    axes[0].set_ylabel("per-fold held-out OOF R² at layer*")
    fig.suptitle("Per-fold diagonal map quality (6 cluster-grouped folds per cell)", y=1.02)
    savefig_paper(fig, "hero1b_diag_q_fold_points", dir=fig_dir)
    plt.close(fig)


# ── hero 2: 4x4 grid heatmaps ────────────────────────────────────────────────


def _grid_matrix(grid: dict, stages: list[str], corpus: str, arm: str, star: int) -> np.ndarray:
    Q = np.full((len(stages), len(stages)), np.nan)
    for mi, m in enumerate(stages):
        for si, s in enumerate(stages):
            if m == s:
                cell = grid["cells"].get(f"diag_{m}_{corpus}_{arm}", {})
                if arm == "ctx":
                    Q[mi, si] = cell.get("r2_at_star", np.nan)
                else:
                    Q[mi, si] = cell.get("r2_by_layer", {}).get(str(star), np.nan)
            else:
                cell = grid["cells"].get(f"grid_{m}{s}_{corpus}_{arm}", {})
                Q[mi, si] = cell.get("per_layer", {}).get(str(star), {}).get("r2", np.nan)
    return Q


def _heat(ax, Q: np.ndarray, stages: list[str], title: str) -> None:
    im = ax.imshow(Q, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(stages)), [STAGE_LABEL[s] for s in stages], rotation=45)
    ax.set_yticks(range(len(stages)), [STAGE_LABEL[s] for s in stages])
    for (r, c), v in np.ndenumerate(Q):
        if np.isfinite(v):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("answer source s")
    ax.set_ylabel("activation checkpoint m")
    plt.colorbar(im, ax=ax, fraction=0.046)


def fig_hero_grid(eval_dir: Path, fig_dir: Path) -> None:
    grid = _load(eval_dir, "fits/grid_cells.json")
    stages = _present_stages(grid)
    star = grid["layer_star"]
    star_p = grid["layer_star_p"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    _heat(
        axes[0],
        _grid_matrix(grid, stages, "single", "ctx", star),
        stages,
        f"ridge · context arm · single (L{star})",
    )
    _heat(
        axes[1],
        _grid_matrix(grid, stages, "multi", "ctx", star),
        stages,
        f"ridge · context arm · multi (L{star})",
    )
    _heat(
        axes[2],
        _grid_matrix(grid, stages, "multi", "pre", star_p),
        stages,
        f"ridge · prefix arm · multi (L{star_p})",
    )
    fig.suptitle("4×4 activation-checkpoint × answer-source grid (OOF R²)", y=1.04)
    savefig_paper(fig, "hero2_grid_heatmaps", dir=fig_dir)
    plt.close(fig)


def fig_hero_grid_mlp(eval_dir: Path, fig_dir: Path) -> None:
    """MLP 4x4 grid (own selected layer per arm); panels are DIFFERENT corpora."""
    grid = _load(eval_dir, "fits/grid_cells.json")
    stages = _present_stages(grid)
    # The MLP grid exists for (ctx arm, single-turn corpus) + (pre arm, multi-turn corpus).
    arm_title = {
        "ctx": "MLP · context arm · single-turn corpus (L{star})",
        "pre": "MLP · prefix arm · multi-turn corpus (L{star})",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, arm in zip(axes, ("ctx", "pre")):
        star_m = grid.get("mlp_layer_star", {}).get(arm)
        Q = np.full((len(stages), len(stages)), np.nan)
        for mi, m in enumerate(stages):
            for si, s in enumerate(stages):
                cell = grid.get("mlp", {}).get(f"mlp_{arm}_{m}{s}", {})
                Q[mi, si] = cell.get("per_layer", {}).get(str(star_m), np.nan)
        _heat(ax, Q, stages, arm_title[arm].format(star=star_m))
    savefig_paper(fig, "hero2b_grid_heatmaps_mlp", dir=fig_dir)
    plt.close(fig)


# ── hero 3: transfer matrices + retention ────────────────────────────────────


MODE_LABEL = {
    "direct": "direct",
    "gl": "general-linear-aligned",
    "orth": "orthogonal-aligned",
    "fixedtext": "fixed answer text",
}


def _per_fold_retention(units: Path, pair: str, n_folds: int = 6) -> list[float]:
    """Per-fold retention rho(i->j) = fold gl-transfer R2 / fold diagonal R2 (single, ctx)."""
    i, j = pair.split("->")
    out: list[float] = []
    for f in range(n_folds):
        xp = units / f"xfer_{i}{j}_f{f}.json"
        dp = units / f"star_{j}_single_f{f}.json"
        if not (xp.exists() and dp.exists()):
            continue
        with open(xp, encoding="utf-8") as fh:
            num = json.load(fh)["r2"]["gl"]
        with open(dp, encoding="utf-8") as fh:
            den = json.load(fh)["r2"]
        out.append(num / den)
    return out


def fig_hero_transfer(eval_dir: Path, fig_dir: Path) -> None:
    xf = _load(eval_dir, "transfer/transfer_matrix.json")
    grid = _load(eval_dir, "fits/grid_cells.json")
    stages = _present_stages(grid)
    modes = xf["modes"]
    fig, axes = plt.subplots(1, len(modes), figsize=(3.3 * len(modes), 3.6))
    for ax, mode in zip(np.atleast_1d(axes), modes):
        Q = np.full((len(stages), len(stages)), np.nan)
        for mi, i in enumerate(stages):
            for si, j in enumerate(stages):
                if i == j:
                    cell = grid["cells"].get(f"diag_{i}_single_ctx", {})
                    Q[mi, si] = cell.get("r2_at_star", np.nan)
                else:
                    Q[mi, si] = xf["pairs"].get(f"{i}->{j}", {}).get("r2", {}).get(mode, np.nan)
        _heat(
            ax,
            Q,
            stages,
            f"{MODE_LABEL.get(mode, mode)} transfer\n(held-out R², layer {xf['layer_star']})",
        )
        ax.set_xlabel("target checkpoint j")
        ax.set_ylabel("source checkpoint i")
    h1 = xf.get("h1", {})
    ci = h1.get("ci_delta_conf") or [float("nan"), float("nan")]
    fig.suptitle(
        f"Cross-stage transfer — median adjacent-transition retention "
        f"{h1.get('r_adj', float('nan')):.3f} (95% CI {0.8 + ci[0]:.3f}–{0.8 + ci[1]:.3f})",
        y=1.05,
    )
    savefig_paper(fig, "hero3_transfer_matrices", dir=fig_dir)
    plt.close(fig)
    # retention + matched nulls per pair (low-level per-unit view)
    pairs = list(xf["pairs"])
    units = eval_dir / "fits" / "units"
    fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * len(pairs)), 3.4))
    xs = np.arange(len(pairs))
    ax.bar(
        xs,
        [xf["pairs"][p]["retention_gl"] for p in pairs],
        color=paper_palette(3)[0],
        label="retention ρ = R²_gl / Q(j,j)",
    )
    for k, p in enumerate(pairs):
        nn = xf["pairs"][p]["nulls"]
        for vals, c, lab in (
            (nn["shuffled_correspondence_r2"], paper_palette(3)[1], "shuffled-corr null R²"),
            (nn["spectrum_matched_r2"], paper_palette(3)[2], "spectrum-matched null R²"),
        ):
            if vals:
                ax.scatter(
                    [k] * len(vals), vals, s=8, color=c, label=lab if k == 0 else None, zorder=3
                )
        folds = _per_fold_retention(units, p)
        if folds:
            xjit = k + np.linspace(-0.18, 0.18, num=len(folds))
            ax.scatter(
                xjit,
                folds,
                s=12,
                color="#333333",
                zorder=4,
                label="per-fold retention (6 folds, labeled)" if k == 0 else None,
            )
            for fi, v in enumerate(folds):
                ax.annotate(
                    f"f{fi}", (xjit[fi], v), fontsize=5, xytext=(1, 2), textcoords="offset points"
                )
    ax.axhline(0.8, ls="--", color="grey", lw=0.8)
    ax.axhline(0.5, ls=":", color="grey", lw=0.8)
    ax.set_xticks(xs, [pair_label(p) for p in pairs], rotation=45, ha="right")
    ax.set_ylabel("retention / null R²")
    ax.set_title("Per-pair retention with matched nulls (reference lines at 0.8 and 0.5)")
    ax.legend(fontsize=7)
    savefig_paper(fig, "hero3b_retention_nulls", dir=fig_dir)
    plt.close(fig)


# ── clusters (H2) ────────────────────────────────────────────────────────────


def _cluster_panel_title(key: str, rec: dict) -> str:
    """Plain-English panel title for a per-cluster key like 'B->S_single'."""
    trans, _, corpus = key.partition("_")
    corpus_lbl = {"single": "single-turn", "multi": "multi-turn"}.get(corpus, corpus)
    p = rec["null_max_abs_p"]
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    return (
        f"{pair_label(trans)} — {corpus_lbl}\n"
        f"(most-moved cluster {rec['most_moved_cluster']}; permutation {p_str})"
    )


def fig_clusters(eval_dir: Path, fig_dir: Path) -> None:
    h2 = _load(eval_dir, "clusters/delta_qc.json")
    per = h2.get("per_cluster", {})
    if per:
        fig, axes = plt.subplots(1, len(per), figsize=(4.2 * len(per), 3.4), squeeze=False)
        for ax, (key, rec) in zip(axes[0], per.items()):
            ids = rec["cluster_ids"]
            vals = rec["delta_qc"]
            ax.scatter(ids, vals, s=14, color=paper_palette(2)[0])
            for c, v in zip(ids, vals):
                ax.annotate(str(c), (c, v), fontsize=5, xytext=(1, 2), textcoords="offset points")
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_title(_cluster_panel_title(key, rec), fontsize=8)
            ax.set_xlabel("cluster id")
            ax.set_ylabel("ΔQ_c (held-out R² delta)")
        savefig_paper(fig, "clusters_delta_qc_scatter", dir=fig_dir)
        plt.close(fig)
    contrasts = {
        **h2.get("registered_contrasts", {}),
        **h2.get("smoke_informational_contrasts", {}),
    }
    if contrasts:
        fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(contrasts)), 3.4))
        xs = np.arange(len(contrasts))
        deltas = [v["delta"] for v in contrasts.values()]
        errs = np.stack(
            [
                [max(0.0, v["delta"] - v["ci_delta"][0]) for v in contrasts.values()],
                [max(0.0, v["ci_delta"][1] - v["delta"]) for v in contrasts.values()],
            ]
        )
        ax.bar(xs, deltas, yerr=np.nan_to_num(errs), capsize=3, color=paper_palette(2)[1])
        ax.axhline(0, color="grey", lw=0.6)
        contrast_label = {
            "a_D->R_gsm8k": "DPO\u2192RLVR\nmath (GSM8K)",
            "a_D->R_mbpp": "DPO\u2192RLVR\ncode (MBPP)",
            "b_B->S_generic_single": "base\u2192SFT\ngeneric single-turn",
            "c_S->D_multi": "SFT\u2192DPO\nmulti-turn",
        }
        labels = [contrast_label.get(k, k) for k in contrasts]
        ax.set_xticks(xs, labels, rotation=0, fontsize=7)
        ax.set_ylabel("ΔQ (class-level held-out R² delta)")
        ax.set_title("Class contrasts (95% bootstrap CI)")
        savefig_paper(fig, "clusters_registered_contrasts", dir=fig_dir)
        plt.close(fig)


# ── operator battery (H4) ────────────────────────────────────────────────────


def fig_operator(eval_dir: Path, fig_dir: Path) -> None:
    op = _load(eval_dir, "operator/operator_battery.json")
    pairs = op.get("pairs", {})
    if not pairs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for key, rec in pairs.items():
        axes[0].plot(
            rec["delta_spectrum_top"], label=f"ΔW {pair_label(key)} (ER={rec['er_delta']:.0f})"
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("singular value index")
    axes[0].set_ylabel("σ_k(ΔW)")
    axes[0].set_title("ΔW spectra (top 64)")
    axes[0].legend(fontsize=7)
    keys, ers, er_shuf, er_spec = [], [], [], []
    for key, rec in pairs.items():
        keys.append(key)
        ers.append(rec["er_delta"])
        er_shuf.append(rec["er_delta_null_shuffled_refit"])
        er_spec.append(rec["er_delta_null_spectrum_matched"])
    xs = np.arange(len(keys))
    axes[1].bar(xs - 0.25, ers, width=0.25, label="observed ER(ΔW)", color=paper_palette(3)[0])
    axes[1].bar(xs, er_shuf, width=0.25, label="shuffled-refit null", color=paper_palette(3)[1])
    axes[1].bar(
        xs + 0.25, er_spec, width=0.25, label="spectrum-matched null", color=paper_palette(3)[2]
    )
    axes[1].set_xticks(xs, [pair_label(k) for k in keys])
    axes[1].set_ylabel("effective rank (Σσ)²/Σσ²")
    axes[1].set_title("ER(ΔW) vs matched nulls")
    axes[1].legend(fontsize=7)
    for key, rec in pairs.items():
        pr = rec["procrustes_aligned"]
        draws = pr.get("draws", [])
        if draws:
            axes[2].scatter(
                [pair_label(key)] * len(draws),
                draws,
                s=6,
                color="grey",
                label="rotation null draws" if key == keys[0] else None,
            )
        axes[2].scatter(
            [pair_label(key)],
            [pr["observed_aligned_cosine"]],
            marker="D",
            s=40,
            color=paper_palette(3)[0],
            label="Procrustes-aligned cos (direction-aware)" if key == keys[0] else None,
        )
        axes[2].scatter(
            [pair_label(key)],
            [rec["spectrum_cosine"]],
            marker="x",
            s=40,
            color=paper_palette(3)[1],
            label="spectrum cos (rotation-invariant, descriptive)" if key == keys[0] else None,
        )
    axes[2].set_ylabel("operator cosine")
    axes[2].set_title("Direction-aware vs spectrum-only cosine")
    axes[2].legend(fontsize=6)
    savefig_paper(fig, "operator_battery", dir=fig_dir)
    plt.close(fig)


# ── exploratory dump ─────────────────────────────────────────────────────────


def fig_exploratory(eval_dir: Path, fig_dir: Path) -> None:
    grid = _load(eval_dir, "fits/grid_cells.json")
    stages = _present_stages(grid)
    # (a) full layer curves per diagonal cell
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, corpus in zip(axes, ("single", "multi")):
        for m in stages:
            cell = grid["cells"].get(f"diag_{m}_{corpus}_ctx", {})
            curve = cell.get("r2_by_layer", {})
            layers = sorted(int(k) for k in curve)
            ax.plot(
                layers,
                [curve[str(k)] for k in layers],
                "o-",
                ms=3,
                color=stage_color(m),
                label=STAGE_LABEL[m],
            )
        pre = grid["cells"].get(f"diag_{stages[0]}_multi_pre", {})
        ax.axvline(grid["layer_star"], ls="--", color="grey", lw=0.8)
        ax.set_xlabel("layer")
        ax.set_title(f"{corpus} corpus — diagonal ctx-arm layer curves")
        del pre
    axes[0].set_ylabel("pooled OOF R²")
    axes[0].legend(fontsize=7)
    savefig_paper(fig, "exploratory_layer_curves", dir=fig_dir)
    plt.close(fig)
    # (b) per-context recon cosine histograms at layer* (percell shards)
    percell = eval_dir / "fits" / "percell"
    if percell.is_dir():
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        for m in stages:
            cos_all = []
            for sp in sorted(percell.glob(f"diag_{m}_single_ctx_f*.npz")):
                d = np.load(sp)
                li = list(d["layers"]).index(grid["layer_star"])
                cos_all.append(d["cos"][li])
            if cos_all:
                v = np.concatenate(cos_all)
                ax.hist(
                    v[np.isfinite(v)],
                    bins=40,
                    histtype="step",
                    linewidth=1.5,
                    color=stage_color(m),
                    label=STAGE_LABEL[m],
                )
        ax.set_xlabel("per-context recon cosine (pred vs true w, layer*)")
        ax.set_ylabel("contexts")
        ax.legend(fontsize=7)
        ax.set_title("Per-context reconstruction cosine (single corpus)")
        savefig_paper(fig, "exploratory_recon_cosine_hist", dir=fig_dir)
        plt.close(fig)
        # per-dimension R² histogram at layer* (star shards)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        for m in stages:
            rs, ts = None, None
            for sp in sorted(percell.glob(f"star_{m}_single_f*.npz")):
                d = np.load(sp)
                rs = d["ss_res_dim"] if rs is None else rs + d["ss_res_dim"]
                ts = d["ss_tot_dim"] if ts is None else ts + d["ss_tot_dim"]
            if rs is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    r2d = 1.0 - rs / ts
                ax.hist(
                    r2d[np.isfinite(r2d)],
                    bins=50,
                    histtype="step",
                    linewidth=1.5,
                    color=stage_color(m),
                    label=STAGE_LABEL[m],
                )
        ax.set_xlabel("per-dimension R² at layer*")
        ax.set_ylabel("dimensions")
        ax.legend(fontsize=7)
        ax.set_title("Per-dimension R² (diagonal, single corpus)")
        savefig_paper(fig, "exploratory_per_dim_r2", dir=fig_dir)
        plt.close(fig)
    # (c) CKA vs layer
    op = _load(eval_dir, "operator/operator_battery.json")
    if op.get("cka"):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        pal = paper_palette(max(3, len(op["cka"])))
        for k, (key, rec) in enumerate(sorted(op["cka"].items())):
            if key.endswith("_single"):
                ax.plot(rec["layers"], rec["cka_u"], "-", color=pal[k % len(pal)], label=f"u {key}")
                ax.plot(
                    rec["layers"], rec["cka_w"], "--", color=pal[k % len(pal)], label=f"w {key}"
                )
        ax.set_xlabel("layer")
        ax.set_ylabel("linear CKA")
        ax.set_title("Basis stability: CKA(u_i,u_j) solid / CKA(w_i,w_j) dashed")
        ax.legend(fontsize=5, ncol=2)
        savefig_paper(fig, "exploratory_cka_layers", dir=fig_dir)
        plt.close(fig)
    # (d) retrieval acc@k bars at layer* (diagonal cells)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.8 / max(1, len(stages))
    for k_i, m in enumerate(stages):
        cell = grid["cells"].get(f"diag_{m}_single_ctx", {})
        folds = cell.get("baselines_at_star", {}).get("knn") or []
        accs: dict[str, list[float]] = {}
        for f in folds:
            if not f:
                continue
            for kk, v in f["euclidean"]["acc_at_k"].items():
                accs.setdefault(kk, []).append(v)
        ks = sorted(accs, key=int)
        xs = np.arange(len(ks))
        ax.bar(
            xs + k_i * width,
            [float(np.mean(accs[kk])) for kk in ks],
            width=width,
            color=stage_color(m),
            label=STAGE_LABEL[m],
        )
    ax.set_xticks(np.arange(len(ks)) + 0.4 if stages else [], [f"acc@{k}" for k in ks])
    ax.set_ylabel("retrieval accuracy (euclidean; fold mean)")
    ax.set_title("kNN retrieval of the true answer state (diagonal, layer*)")
    ax.legend(fontsize=7)
    savefig_paper(fig, "exploratory_retrieval_acc", dir=fig_dir)
    plt.close(fig)
    # (e) native-vs-plain render deltas (own stem — see fig_render_robustness)
    fig_render_robustness(eval_dir, fig_dir)


def fig_render_robustness(eval_dir: Path, fig_dir: Path) -> None:
    """Native-vs-plain render deltas on the 2k subset (n_tr < d: degenerate scale)."""
    op = _load(eval_dir, "operator/operator_battery.json")
    robust = op.get("robust_native_vs_plain", {})
    rows = {m: v for m, v in robust.items() if isinstance(v, dict) and "native_r2_mean" in v}
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = np.arange(len(rows))
    ax.bar(
        xs - 0.2,
        [v["plain_r2_mean"] for v in rows.values()],
        width=0.4,
        label="plain render",
        color=paper_palette(2)[0],
    )
    ax.bar(
        xs + 0.2,
        [v["native_r2_mean"] for v in rows.values()],
        width=0.4,
        label="native render",
        color=paper_palette(2)[1],
    )
    units = eval_dir / "fits" / "units"
    for xi, m in enumerate(rows):
        for off, field in ((-0.2, "plain_r2"), (0.2, "native_r2")):
            vals = []
            for p in sorted(units.glob(f"robust_{m}_f*.json")):
                with open(p, encoding="utf-8") as fh:
                    vals.append(json.load(fh)[field])
            if vals:
                xjit = xi + off + np.linspace(-0.09, 0.09, num=len(vals))
                ax.scatter(
                    xjit,
                    vals,
                    s=10,
                    color="#333333",
                    zorder=4,
                    label="per-fold values (6 folds, labeled)" if xi == 0 and off < 0 else None,
                )
                for fi, v in enumerate(vals):
                    ax.annotate(
                        f"f{fi}",
                        (xjit[fi], v),
                        fontsize=4.5,
                        xytext=(1, 1),
                        textcoords="offset points",
                    )
    ax.set_xticks(xs, [STAGE_LABEL.get(m, m) for m in rows])
    ax.set_ylabel("OOF R² at layer* (2k subset;\nper-fold n_tr ≈ 1.7k < d = 4096)")
    ax.set_title(
        "Serialization robustness: native vs plain render\n"
        "(n_tr < d — estimator-degenerate scale; read the native−plain delta only)"
    )
    ax.legend(fontsize=7)
    savefig_paper(fig, "exploratory_render_robustness", dir=fig_dir)
    plt.close(fig)


FIG_GROUPS = {
    "hero1": fig_hero_diag,
    "hero1b": fig_hero_diag_folds,
    "hero2": fig_hero_grid,
    "hero2b": fig_hero_grid_mlp,
    "hero3": fig_hero_transfer,  # renders hero3 + hero3b
    "clusters": fig_clusters,
    "operator": fig_operator,
    "render_robustness": fig_render_robustness,
    "exploratory": fig_exploratory,  # layer curves, hists, CKA, retrieval + render_robustness
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", type=Path, default=PROJECT_ROOT / "eval_results" / "issue_1902")
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_1902")
    ap.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FIG_GROUPS),
        help="render only these figure groups (default: all)",
    )
    args = ap.parse_args()
    set_paper_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    groups = args.only or [g for g in FIG_GROUPS if g != "render_robustness"]
    for g in groups:
        FIG_GROUPS[g](args.eval_dir, args.fig_dir)
    print(f"[figures] done ({', '.join(groups)}) -> {args.fig_dir}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
