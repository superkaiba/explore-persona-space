#!/usr/bin/env python
"""Issue #1336 — D1 diagnosis figures (plan v7 SS6 "Figures to produce").

Hero: ONE two-panel figure —
  (left)  per-dim excess-residual concentration curve (cumulative share of
          the R^2 deficit vs dim rank, both cells), and
  (right) the read ladder: committed -0.93 -> each variant's R^2 -> S, with
          the matched standardized band B and bar_std drawn —
the single figure that answers "artifact or absence".

Exploratory dump (over-produce): per-dim R^2 histograms; variance-share
curves; corrected vs raw 32-layer curves with bands; lambda heatmaps per
(layer, fold) with grid edges marked (committed vs widened); centered vs
uncentered cosine; Llama-vs-Qwen per-dim scale spectra + top-dim
discreteness; per-fold bias panels; Qwen calibration bar.

Inputs: --diag-dir (default eval_results/issue_1336/diagnosis).
Outputs: --fig-dir (default figures/issue_1336/diagnosis; smokes MUST
redirect via --fig-dir to a scratch dir — committed figure paths are never
smoke targets).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
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
    set_paper_style,
)

CELLS = ("rlvr_chat_lmsys5k", "rlvr_naturalistic_lmsys5k")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diag-dir", type=Path, default=Path("eval_results/issue_1336/diagnosis"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1336/diagnosis"))
    ap.add_argument("--cells", default=",".join(CELLS))
    return ap.parse_args()


def _maybe(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig, fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / name
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[diag-fig] wrote {out}")


# ---------------------------------------------------------------------------
# Hero — excess concentration (left) + read ladder (right)
# ---------------------------------------------------------------------------
def fig_hero(args, cells: list[str]) -> None:
    pal = paper_palette(4)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    # Left: cumulative share of the residual EXCESS (SS_res - SS_tot) by dim rank.
    plotted = False
    for ci, cell in enumerate(cells):
        npz_path = args.diag_dir / "tensors" / f"perdim_{cell}.npz"
        decomp = _maybe(args.diag_dir / f"perdim_decomp_{cell}.json")
        if decomp is None or not npz_path.exists():
            continue
        data = np.load(npz_path)
        layers = decomp["layers"]
        li = layers[-1]  # deepest persisted layer (argmax-adjacent read)
        excess = data[f"ss_res_l{li}"].astype(np.float64) - data[f"ss_tot_l{li}"].astype(np.float64)
        tot = excess.sum()
        if tot <= 0:
            continue
        share = np.cumsum(np.sort(excess)[::-1]) / tot
        ranks = np.arange(1, len(share) + 1)
        ax_l.plot(ranks, share, color=pal[ci % len(pal)], label=f"{cell} (L{li})")
        plotted = True
    if plotted:
        ax_l.set_xscale("log")
        ax_l.axhline(1.0, color="grey", lw=0.6, ls=":")
        ax_l.set_xlabel("target-dim rank (by excess residual)")
        ax_l.set_ylabel("cumulative share of R$^2$ deficit")
        ax_l.set_title("Where the deficit lives, dim by dim")
        ax_l.legend(frameon=False, fontsize=8)
    else:
        ax_l.set_axis_off()
        ax_l.set_title("per-dim decomposition unavailable")
    # Right: the read ladder committed -> variants -> S, with B + bar_std.
    chat = cells[0]
    verdict = _maybe(args.diag_dir / "diagnosis_verdict.json")
    v0 = _maybe(args.diag_dir / f"refit_v0_{chat}.json")
    if verdict is not None and v0 is not None:
        att = verdict["mechanism_attribution"]["per_variant"]
        li = verdict["lattice_inputs"]
        rungs = [("committed v0", float(v0["best_r2"]), "baseline")]
        for name in sorted(att):
            rungs.append((name, float(att[name]["r2_at_own_argmax"]), "variant"))
        rungs.append(("S (v2, lattice)", float(li["S"]), "headline"))
        colors = {"baseline": pal[3], "variant": pal[0], "headline": pal[1]}
        xs = np.arange(len(rungs))
        ax_r.bar(
            xs,
            [r[1] for r in rungs],
            color=[colors[r[2]] for r in rungs],
            width=0.62,
        )
        ax_r.axhline(
            float(li["B_standardized_p975_layer_max"]),
            color="black",
            lw=1.0,
            ls="--",
            label="matched std band B (p97.5)",
        )
        ax_r.axhline(
            float(li["bar_std"]), color=pal[2], lw=1.0, ls="-.", label="bar_std (usable strength)"
        )
        ax_r.set_xticks(xs)
        ax_r.set_xticklabels([r[0] for r in rungs], rotation=35, ha="right", fontsize=7)
        ax_r.set_ylabel("held-out pooled R$^2$")
        ax_r.set_title(f"Read ladder — {chat}")
        ax_r.legend(frameon=False, fontsize=8)
    else:
        ax_r.set_axis_off()
        ax_r.set_title("verdict/battery outputs unavailable")
    _save(fig, args.fig_dir, "hero_diagnosis_two_panel.png")


# ---------------------------------------------------------------------------
# Exploratory dump
# ---------------------------------------------------------------------------
def fig_perdim_hist(args, cells: list[str]) -> None:
    for cell in cells:
        npz_path = args.diag_dir / "tensors" / f"perdim_{cell}.npz"
        decomp = _maybe(args.diag_dir / f"perdim_decomp_{cell}.json")
        if decomp is None or not npz_path.exists():
            continue
        data = np.load(npz_path)
        layers = decomp["layers"]
        fig, axes = plt.subplots(
            1, len(layers), figsize=(3.2 * len(layers), 3.0), layout="constrained"
        )
        axes = np.atleast_1d(axes)
        for ax, li in zip(axes, layers, strict=True):
            r2 = data[f"perdim_r2_l{li}"]
            r2 = r2[np.isfinite(r2)]
            ax.hist(np.clip(r2, -5, 1), bins=60, color=paper_palette(1)[0])
            ax.axvline(0.0, color="black", lw=0.8)
            ax.set_title(f"L{li} per-dim R$^2$ (clipped)")
        _save(fig, args.fig_dir, f"perdim_r2_hist_{cell}.png")


def fig_variance_share(args, cells: list[str]) -> None:
    pal = paper_palette(4)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
    for ci, cell in enumerate(cells):
        npz_path = args.diag_dir / "tensors" / f"perdim_{cell}.npz"
        decomp = _maybe(args.diag_dir / f"perdim_decomp_{cell}.json")
        if decomp is None or not npz_path.exists():
            continue
        data = np.load(npz_path)
        li = decomp["layers"][-1]
        ss_tot = np.sort(data[f"ss_tot_l{li}"].astype(np.float64))[::-1]
        ax.plot(
            np.arange(1, len(ss_tot) + 1),
            np.cumsum(ss_tot) / ss_tot.sum(),
            color=pal[ci % len(pal)],
            label=f"{cell} (L{li})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("target-dim rank (by variance)")
    ax.set_ylabel("cumulative variance share")
    ax.set_title("Target-variance concentration")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, args.fig_dir, "variance_share_curves.png")


def fig_layer_curves(args, cells: list[str]) -> None:
    pal = paper_palette(4)
    for cell in cells:
        v0 = _maybe(args.diag_dir / f"refit_v0_{cell}.json")
        v2 = _maybe(args.diag_dir / f"refit_v2_{cell}.json")
        nulls = _maybe(args.diag_dir / f"refit_null_std_{cell}.json")
        if v0 is None or v2 is None:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 3.6), layout="constrained")
        curve0 = v0["r2_per_layer_obs"]
        ax.plot(range(len(curve0)), curve0, color=pal[3], label="v0 raw (committed)")
        lay2 = sorted(int(k) for k in v2["r2_per_layer"])
        ax.plot(
            lay2,
            [v2["r2_per_layer"][str(li)] for li in lay2],
            color=pal[1],
            label="v2 standardized (S curve)",
        )
        if nulls is not None and nulls["null_matrix_draw_x_layer"]:
            mat = np.asarray(nulls["null_matrix_draw_x_layer"], dtype=float)
            lo = np.nanquantile(mat, 0.025, axis=0)
            hi = np.nanquantile(mat, 0.975, axis=0)
            ax.fill_between(
                nulls["layers"],
                lo,
                hi,
                color=pal[2],
                alpha=0.25,
                label="std shuffle band (per-layer 95%)",
            )
        ax.axhline(0.0, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out pooled R$^2$")
        ax.set_title(f"Raw vs corrected layer curves — {cell}")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, args.fig_dir, f"layer_curves_{cell}.png")


def fig_lambda_heatmaps(args, cells: list[str]) -> None:
    for cell in cells:
        v0 = _maybe(args.diag_dir / f"refit_v0_{cell}.json")
        v1 = _maybe(args.diag_dir / f"refit_v1_{cell}.json")
        if v0 is None or v1 is None:
            continue
        lam0 = np.asarray(v0["gcv_lambda_layer_x_fold"], dtype=float)
        lay1 = sorted(int(k) for k in v1["gcv_lambda_layer_x_fold"])
        lam1 = np.asarray([v1["gcv_lambda_layer_x_fold"][str(li)] for li in lay1], dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
        for ax, lam, title in (
            (axes[0], lam0, "committed grid logspace(-2,4,13)"),
            (axes[1], lam1, "widened grid logspace(-2,8,21)"),
        ):
            with np.errstate(divide="ignore"):
                im = ax.imshow(np.log10(lam), aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax, label="log10 selected lambda")
            ax.set_xlabel("fold")
            ax.set_ylabel("layer")
            ax.set_title(title, fontsize=9)
        fig.suptitle(f"GCV lambda audit — {cell}", fontsize=10)
        _save(fig, args.fig_dir, f"lambda_heatmap_{cell}.png")


def fig_cosine_and_bias(args, cells: list[str]) -> None:
    pal = paper_palette(4)
    for cell in cells:
        decomp = _maybe(args.diag_dir / f"perdim_decomp_{cell}.json")
        if decomp is None:
            continue
        layers = decomp["layers"]
        cc = [decomp["per_layer"][str(li)]["cosine_centered_mean"] for li in layers]
        cu = [decomp["per_layer"][str(li)]["cosine_uncentered_mean"] for li in layers]
        fb = [decomp["per_layer"][str(li)]["fold_bias_max_abs"] for li in layers]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4), layout="constrained")
        ax1.plot(layers, cu, "o-", color=pal[3], label="uncentered (committed read)")
        ax1.plot(layers, cc, "o-", color=pal[1], label="centered (train-fold mean removed)")
        ax1.set_xlabel("layer")
        ax1.set_ylabel("mean per-example cosine")
        ax1.set_title("Centered vs uncentered cosine")
        ax1.legend(frameon=False, fontsize=8)
        ax2.plot(layers, fb, "s-", color=pal[0])
        ax2.set_xlabel("layer")
        ax2.set_ylabel("max |per-fold per-dim bias|")
        ax2.set_title("Per-fold bias read (H-D confirm)")
        _save(fig, args.fig_dir, f"cosine_bias_{cell}.png")


def fig_scale_spectra(args, cells: list[str]) -> None:
    pal = paper_palette(4)
    audit = _maybe(args.diag_dir / "scale_audit.json")
    if audit is None:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.6), layout="constrained")
    plotted = False
    for ci, cell in enumerate(cells):
        rep = audit["cells"].get(cell)
        if rep is None:
            continue
        npz_path = Path(rep["arrays_npz"])
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        li = rep["layers"][-1]
        std = np.sort(data[f"Y_std_l{li}"].astype(np.float64))[::-1]
        ax.plot(
            np.arange(1, len(std) + 1),
            std,
            color=pal[ci % len(pal)],
            label=f"Llama {cell} (L{li})",
        )
        plotted = True
    if audit.get("qwen_s1"):
        q = audit["qwen_s1"]
        ax.axhline(
            q["y_std_median"],
            color="grey",
            ls=":",
            label=f"Qwen S1 median std (L{q['layer']})",
        )
        ax.scatter([1], [q["y_std_top1"]], color="black", marker="x", label="Qwen S1 top-1 std")
        plotted = True
    if plotted:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("target-dim rank (by std)")
        ax.set_ylabel("per-dim std")
        ax.set_title("Per-dim scale spectra — Llama vs Qwen")
        ax.legend(frameon=False, fontsize=7)
        _save(fig, args.fig_dir, "scale_spectra_llama_vs_qwen.png")
    else:
        plt.close(fig)


def fig_qwen_calibration(args) -> None:
    qc = _maybe(args.diag_dir / "refit_qwen_cal.json")
    if qc is None:
        return
    pal = paper_palette(3)
    fig, ax = plt.subplots(figsize=(3.6, 3.4), layout="constrained")
    ax.bar(
        [0, 1],
        [qc["r2_raw_committed_grid"], qc["s_qwen_standardized"]],
        color=[pal[0], pal[1]],
        width=0.6,
    )
    ax.axhline(qc["committed_anchor"], color="black", ls="--", lw=1.0, label="committed 0.6731")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "standardized"])
    ax.set_ylabel("held-out pooled R$^2$ @ L19")
    ax.set_title(f"Qwen S1 calibration (bar_std={qc['bar_std']:.3f})", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, args.fig_dir, "qwen_calibration.png")


def main() -> int:
    args = parse_args()
    set_paper_style()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    fig_hero(args, cells)
    fig_perdim_hist(args, cells)
    fig_variance_share(args, cells)
    fig_layer_curves(args, cells)
    fig_lambda_heatmaps(args, cells)
    fig_cosine_and_bias(args, cells)
    fig_scale_spectra(args, cells)
    fig_qwen_calibration(args)
    print(f"[diag-fig] complete -> {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
