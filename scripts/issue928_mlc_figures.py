# ruff: noqa: RUF001  # minus sign in figure text intentional
"""Figures for issue #928 follow-up round `matched-length-answer-span-control`.

Reads the round's committed eval JSONs
(``eval_results/issue_928/matched-length-answer-span-control/``) plus the
per-context decomposition tensors (``decomp_indiv_mlc.pt`` — staged locally
from HF ``issue928_cot_decomposition/analysis_tensors/decomp/matched_length_control``)
and regenerates the round's figures under ``figures/issue_928/``:

- ``mlc_hero_read1``: two-panel hero — per-question frozen-layer bars for the
  five registered conditioning arms + identity ceiling (left) and the read-1
  forest across regimes x layer conventions (right).
- ``mlc_percontext_read1_scatter``: per-context read-1 delta vs median CoT
  length and vs median K, labeled by battery context id, flagged contexts red.
- ``mlc_sufficiency_truncation``: sufficiency-analogue and truncation-cost
  bars (slices alone; context+full CoT vs context+truncated CoT).
- ``mlc_skill_curves_indiv`` / ``mlc_skill_curves_avg_q``: per-layer held-out
  skill for all arms with null band + identity ceiling (plain-English legend).

Usage:
    uv run python scripts/issue928_mlc_figures.py \
        --eval-dir eval_results/issue_928/matched-length-answer-span-control \
        --decomp /tmp/issue928_mlc_stage/decomp_indiv_mlc.pt \
        --out-dir figures/issue_928
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

FLAGGED = {
    "f2_wc_short_3",
    "f2_wc_short_5",
    "f2_wc_long_1",
    "f2_wc_long_2",
    "f2_wc_long_3",
    "f2_wc_long_4",
    "f2_wc_long_5",
    "f3_icl_marker_k4",
    "f3_icl_french_k4",
    "f3_icl_json_k4",
    "f3_icl_pirate_k4",
    "f3_icl_marker_k2",
    "f3_icl_marker_k8",
    "f3_icl_json_k8",
}

ARM_LABELS = {
    "mlc_ctx": "context only",
    "mlc_ctx_cotK": "context + truncated CoT",
    "mlc_ctx_apfx": "context + answer prefix",
    "mlc_cotK": "truncated CoT alone",
    "mlc_apfx": "answer prefix alone",
    "mlc_ctx_cotfull": "context + full CoT",
    "mlc_ctx_cotK_first": "context + CoT opening",
    "mlc_cotK_first": "CoT opening alone",
    "mlc_ident": "identity ceiling",
}
REGIME_LABELS = {"indiv": "per-question", "avg_q": "query-averaged"}
CONV_LABELS = {
    "primary_frozen_ctx_baseline_best": "primary frozen",
    "secondary_own_best_frozen_full_data": "own-best frozen",
    "secondary_best_vs_best_inherited": "best-vs-best",
}


def _loco_at(grid: dict, regime: str, arm: str, layer: int) -> float:
    for e in grid["grid"][regime][arm]["loco"]:
        if e["layer"] == layer:
            return float(e["skill"])
    raise KeyError(f"layer {layer} missing for {arm}/{regime}")


def hero_read1(grid: dict, boot: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    regime = "indiv"
    frozen = grid["frozen_layers"][regime]["primary_frozen_ctx_baseline_best_layer"]
    arms = ["mlc_ctx", "mlc_ctx_apfx", "mlc_ctx_cotK", "mlc_ctx_cotfull", "mlc_ident"]
    absf = boot["by_regime"][regime]["absolute_at_frozen"]
    vals = [absf[a]["observed"] for a in arms]
    errs = [
        [absf[a]["observed"] - absf[a]["ci95"][0] for a in arms],
        [absf[a]["ci95"][1] - absf[a]["observed"] for a in arms],
    ]
    colors = paper_palette_blog(len(arms))
    x = np.arange(len(arms))
    ax1.bar(x, vals, yerr=errs, color=colors, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([ARM_LABELS[a].replace(" + ", " +\n") for a in arms])
    ax1.set_ylabel("held-out skill (remainder target)")
    ax1.set_title(f"per-question, frozen layer {frozen}", fontweight="bold")
    ax1.set_ylim(0, 1.02)

    rows = []
    for regime_key in ["indiv", "avg_q"]:
        stats = boot["by_regime"][regime_key]["statistics"]["read1_primary_ctx_cotK_minus_ctx_apfx"]
        for conv in [
            "primary_frozen_ctx_baseline_best",
            "secondary_own_best_frozen_full_data",
            "secondary_best_vs_best_inherited",
        ]:
            s = stats[conv]
            rows.append(
                (
                    f"{REGIME_LABELS[regime_key]} · {CONV_LABELS[conv]}",
                    s["observed"],
                    s["ci95"][0],
                    s["ci95"][1],
                )
            )
    ys = np.arange(len(rows))[::-1]
    for y, (_, obs, lo, hi) in zip(ys, rows, strict=False):
        ax2.errorbar(
            [obs],
            [y],
            xerr=[[obs - lo], [hi - obs]],
            fmt="o",
            color="black",
            capsize=3,
            markersize=6,
        )
    ax2.axvline(0.0, color="grey", linestyle="--", linewidth=1.2)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_xlabel("Δ skill (context+truncated-CoT − context+answer-prefix)")
    ax2.set_title("matched-length contrast, all conventions", fontweight="bold")
    savefig_paper(fig, "mlc_hero_read1", dir=out_dir)
    plt.close(fig)


def percontext_scatter(pc: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True, sharey=True)
    d = np.asarray(pc["delta_read1"])
    fl = np.asarray(pc["flagged"], dtype=bool)
    xs = [np.asarray(pc["median_cot_len"]), np.asarray(pc["median_K"])]
    xlabels = ["median CoT length (tokens)", "median matched budget K (tokens)"]
    for ax, xv, xl in zip(axes, xs, xlabels, strict=False):
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.2)
        ax.scatter(xv[~fl], d[~fl], color="#1170aa", s=42, label="unflagged (36)")
        ax.scatter(xv[fl], d[fl], color="#d1615d", s=42, label="flagged parse floor (14)")
        for x, y, lab in zip(xv, d, pc["ctx"], strict=False):
            ax.text(x, y, lab, fontsize=5.5, alpha=0.75, ha="left", va="bottom")
        ax.set_xlabel(xl)
    axes[0].set_ylabel(f"per-context Δ skill @ layer {pc['layer']}")
    fig.suptitle(
        "Δ skill = context+truncated-CoT − context+answer-prefix, per context",
        fontweight="bold",
    )
    axes[0].legend(loc="lower right")
    savefig_paper(fig, "mlc_percontext_read1_scatter", dir=out_dir)
    plt.close(fig)


def sufficiency_truncation(grid: dict, boot: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    regime = "indiv"
    absf = boot["by_regime"][regime]["absolute_at_frozen"]
    arms = ["mlc_cotK", "mlc_cotK_first", "mlc_apfx"]
    vals = [absf[a]["observed"] for a in arms]
    errs = [
        [absf[a]["observed"] - absf[a]["ci95"][0] for a in arms],
        [absf[a]["ci95"][1] - absf[a]["observed"] for a in arms],
    ]
    colors = paper_palette_blog(len(arms))
    x = np.arange(len(arms))
    ax1.bar(x, vals, yerr=errs, color=colors, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([ARM_LABELS[a].replace(" alone", "\nalone") for a in arms])
    ax1.set_ylabel("held-out skill (remainder target)")
    ax1.set_title("slices alone (sufficiency analogues)", fontweight="bold")
    ax1.set_ylim(0.8, 0.96)

    rows = []
    for regime_key in ["indiv", "avg_q"]:
        for read, lab in [
            ("read4_cotK_alone_minus_apfx_alone", "slices alone: CoT − prefix"),
            ("read5_ctx_cotfull_minus_ctx_cotK", "full CoT − truncated CoT"),
        ]:
            s = boot["by_regime"][regime_key]["statistics"][read][
                "primary_frozen_ctx_baseline_best"
            ]
            rows.append(
                (
                    f"{REGIME_LABELS[regime_key]} · {lab}",
                    s["observed"],
                    s["ci95"][0],
                    s["ci95"][1],
                )
            )
    ys = np.arange(len(rows))[::-1]
    for y, (_, obs, lo, hi) in zip(ys, rows, strict=False):
        ax2.errorbar(
            [obs],
            [y],
            xerr=[[obs - lo], [hi - obs]],
            fmt="o",
            color="black",
            capsize=3,
            markersize=6,
        )
    ax2.axvline(0.0, color="grey", linestyle="--", linewidth=1.2)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_xlabel("Δ skill, paired context bootstrap 95% interval")
    ax2.set_title("triangulation reads", fontweight="bold")
    savefig_paper(fig, "mlc_sufficiency_truncation", dir=out_dir)
    plt.close(fig)


def skill_curves(grid: dict, regime: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.4), constrained_layout=True)
    arms = [
        "mlc_ctx",
        "mlc_ctx_cotK",
        "mlc_ctx_apfx",
        "mlc_cotK",
        "mlc_apfx",
        "mlc_ctx_cotfull",
        "mlc_ctx_cotK_first",
        "mlc_cotK_first",
    ]
    colors = paper_palette_blog(len(arms))
    layers = grid["capture_layers"]
    for arm, c in zip(arms, colors, strict=False):
        sk = {e["layer"]: e["skill"] for e in grid["grid"][regime][arm]["loco"]}
        ax.plot(layers, [sk[ly] for ly in layers], color=c, label=ARM_LABELS[arm])
    ident = {e["layer"]: e["skill"] for e in grid["grid"][regime]["mlc_ident"]["loco"]}
    ax.plot(
        layers,
        [ident[ly] for ly in layers],
        color="black",
        linestyle="--",
        label="identity ceiling (per layer)",
    )
    bands = grid["null_band_vs_ceiling"][regime]["arms"]
    hi = max(v["band_p97p5"] for v in bands.values())
    lo = min(v["band_p2p5"] for v in bands.values())
    ax.axhspan(lo, hi, color="grey", alpha=0.25, label="permutation band (max-over-layers)")
    frozen = grid["frozen_layers"][regime]["primary_frozen_ctx_baseline_best_layer"]
    ax.axvline(frozen, color="grey", linestyle=":", linewidth=1.2)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out skill (remainder target)")
    ax.set_title(
        f"matched-length arms by layer — {REGIME_LABELS[regime]} (frozen layer {frozen} dotted)",
        fontweight="bold",
    )
    ax.set_ylim(-0.6 if regime == "avg_q" else -0.1, 1.05)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    savefig_paper(fig, f"mlc_skill_curves_{regime}", dir=out_dir)
    plt.close(fig)


def build_percontext(eval_dir: Path, decomp_path: Path, layer: int) -> dict:
    raw = torch.load(decomp_path, weights_only=False)
    dd = {ast.literal_eval(k): v for k, v in raw.items()}
    e_c = dd[("mlc_ctx_cotK", "mean/mean", layer)]
    e_a = dd[("mlc_ctx_apfx", "mean/mean", layer)]
    assert e_c["ctx_order"] == e_a["ctx_order"]
    order = e_c["ctx_order"]
    s_c = 1 - np.asarray(e_c["ss_res"]) / np.asarray(e_c["ss_tot"])
    s_a = 1 - np.asarray(e_a["ss_res"]) / np.asarray(e_a["ss_tot"])
    gates = json.loads((eval_dir / "mlc_capture_gates.json").read_text())
    med_k = {
        c: float(np.median([r["K"] for r in rows])) for c, rows in gates["row_bookkeeping"].items()
    }
    med_cot = {
        c: float(np.median([r["len_cot"] for r in rows]))
        for c, rows in gates["row_bookkeeping"].items()
    }
    return {
        "layer": layer,
        "ctx": order,
        "delta_read1": (s_c - s_a).tolist(),
        "flagged": [c in FLAGGED for c in order],
        "median_K": [med_k[c] for c in order],
        "median_cot_len": [med_cot[c] for c in order],
    }


def fig_paper_c1_cot(grid: dict, boot: dict) -> None:
    """ICLR paper figure (c1_linear chain-of-thought result): matched-length control.

    Panel A: per-question held-out skill on the shared answer-remainder target at the
    frozen read-out layer for the four conditioning arms (95% bootstrap CI), with the
    identity ceiling as a black dashed reference line. Panel B: the matched-length
    contrast (context+truncated-CoT minus context+answer-prefix) across both regimes
    and all three layer conventions — negative everywhere: the CoT position is not
    privileged over the answer's own opening at matched token budget.
    """
    from explore_persona_space.analysis.paper_plots import figsize_iclr_panels, set_paper_style

    set_paper_style("iclr")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_iclr_panels(2, height_in=2.3))
    regime = "indiv"
    frozen = grid["frozen_layers"][regime]["primary_frozen_ctx_baseline_best_layer"]
    arms = ["mlc_ctx", "mlc_ctx_apfx", "mlc_ctx_cotK", "mlc_ctx_cotfull"]
    arm_color = {
        "mlc_ctx": "#0072B2",  # blue — the direct context arm
        "mlc_ctx_apfx": "#CC79A7",  # purple — real-answer-opening (oracle-content) control
        "mlc_ctx_cotK": (0.835, 0.369, 0.0, 0.55),  # vermilion, light — truncated CoT
        "mlc_ctx_cotfull": "#D55E00",  # vermilion — full CoT
    }
    absf = boot["by_regime"][regime]["absolute_at_frozen"]
    vals = [absf[a]["observed"] for a in arms]
    errs = [
        [absf[a]["observed"] - absf[a]["ci95"][0] for a in arms],
        [absf[a]["ci95"][1] - absf[a]["observed"] for a in arms],
    ]
    x = np.arange(len(arms))
    ax1.bar(x, vals, yerr=errs, color=[arm_color[a] for a in arms], capsize=2, width=0.62)
    # identity ceiling as a black dashed reference line (identified in the caption)
    ax1.axhline(absf["mlc_ident"]["observed"], color="black", lw=1.0, ls="--")
    ax1.set_xticks(x)
    # "answer prefix" / "truncated CoT" / "full CoT" bars are context + X; the
    # caption states it (crowded two-row tick labels otherwise collide at 5.5in).
    ax1.set_xticklabels(["context\nonly", "answer\nprefix", "truncated\nCoT", "full\nCoT"])
    ax1.set_ylabel("held-out skill (remainder target)")
    ax1.set_ylim(0, 1.04)

    rows = []
    for regime_key in ["indiv", "avg_q"]:
        stats = boot["by_regime"][regime_key]["statistics"]["read1_primary_ctx_cotK_minus_ctx_apfx"]
        for conv in [
            "primary_frozen_ctx_baseline_best",
            "secondary_own_best_frozen_full_data",
            "secondary_best_vs_best_inherited",
        ]:
            s = stats[conv]
            rows.append(
                (
                    f"{REGIME_LABELS[regime_key]} $\\cdot$ {CONV_LABELS[conv]}",
                    s["observed"],
                    s["ci95"][0],
                    s["ci95"][1],
                )
            )
    ys = np.arange(len(rows))[::-1]
    for y, (_, obs, lo, hi) in zip(ys, rows, strict=False):
        ax2.errorbar(
            [obs],
            [y],
            xerr=[[max(0.0, obs - lo)], [max(0.0, hi - obs)]],
            fmt="o",
            color="black",
            capsize=2,
            markersize=3.5,
        )
    ax2.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_xlabel("$\\Delta$ skill (CoT $-$ prefix)")
    paper_out = Path(__file__).resolve().parents[1] / "figures" / "paper"
    paper_out.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "c1_cot_matched_length", dir=paper_out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("eval_results/issue_928/matched-length-answer-span-control"),
    )
    ap.add_argument(
        "--decomp",
        type=Path,
        default=None,
        help="local path to decomp_indiv_mlc.pt (staged from HF); required for --style blog",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("figures/issue_928"))
    ap.add_argument("--style", choices=("blog", "iclr"), default="blog")
    args = ap.parse_args()

    if args.style == "iclr":
        # Paper pathway (#2094 precedent): one ICLR-styled figure under figures/paper/,
        # from the committed grid/bootstrap JSONs only (no decomp tensor needed).
        grid = json.loads((args.eval_dir / "mlc_skill_grid.json").read_text())
        boot = json.loads((args.eval_dir / "mlc_bootstrap_deltaskill.json").read_text())
        fig_paper_c1_cot(grid, boot)
        print("paper c1_cot_matched_length regenerated.")
        return
    if args.decomp is None:
        ap.error("--decomp is required for --style blog")

    set_paper_style("blog")
    grid = json.loads((args.eval_dir / "mlc_skill_grid.json").read_text())
    boot = json.loads((args.eval_dir / "mlc_bootstrap_deltaskill.json").read_text())
    frozen = grid["frozen_layers"]["indiv"]["primary_frozen_ctx_baseline_best_layer"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hero_read1(grid, boot, args.out_dir)
    pc = build_percontext(args.eval_dir, args.decomp, frozen)
    percontext_scatter(pc, args.out_dir)
    sufficiency_truncation(grid, boot, args.out_dir)
    for regime in ["indiv", "avg_q"]:
        skill_curves(grid, regime, args.out_dir)
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
