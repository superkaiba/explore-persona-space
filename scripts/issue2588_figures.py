#!/usr/bin/env python3
"""Issue #2588 figures — renders the P3 trend summary to figures/issue_2588/.

Four figures (plan §4.4/§6), all from eval_results/issue_2588/trend_summary.json
via analysis.paper_plots.savefig_paper (git-sha sidecars; simple-and-concise
figure register — axes + ticks + legend only, no in-figure caption blocks):

  c1_capability_trend   calibrated test acc@1 (cosine, layer_star) vs the AA
                        capability index, per arm, points labeled by model.
  c2_column_contrasts   the fixed-size capability column (Qwen3.5-27B ->
                        3.6-27B -> 3.8-27B): calibrated contrasts + shifted
                        95% bootstrap CIs, per arm.
  c3_h2_paired_deltas   per-checkpoint calibrated delta (end-of-CoT arm-b map
                        minus same-checkpoint prompt-side arm-a map), the H2
                        Wilcoxon input (n=7).
  c4_gpqa_transfer      GPQA same-question retrieval acc@1 (transfer-only) vs
                        chance, per map.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
for p in (str(_SCRIPTS), str(_REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps BEFORE matplotlib/numpy import (VM-side plotter)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import issue2588_panel_common as PC  # noqa: E402
from explore_persona_space.analysis.paper_plots import paper_palette, savefig_paper  # noqa: E402

FIG_DIR = _REPO_ROOT / "figures" / "issue_2588"

ARM_LABEL = {"a": "no-think read (prompt end)", "b": "thinking read (end of CoT)"}
MODEL_LABEL = {
    "q35_0p8b": "Qwen3.5-0.8B",
    "q35_2b": "Qwen3.5-2B",
    "q35_4b": "Qwen3.5-4B",
    "q35_9b": "Qwen3.5-9B",
    "q35_27b": "Qwen3.5-27B",
    "q36_27b": "Qwen3.6-27B",
    "q38_27b": "Qwen3.8-27B",
    "o3_7b_i": "OLMo-3-7B-Instruct",
    "o3_7b_t": "OLMo-3-7B-Think",
    "o31_32b_i": "OLMo-3.1-32B-Instruct",
    "o31_32b_t": "OLMo-3.1-32B-Think",
    "q25_7b": "Qwen2.5-7B-Instruct",
}


def _model_of(map_id: str) -> str:
    return map_id.split(".")[0].rsplit("_", 1)[0]


def _arm_of(map_id: str) -> str:
    return map_id.split(".")[0].rsplit("_", 1)[1]


def _pos_of(map_id: str) -> str:
    return map_id.split(".", 1)[1]


def _trend_points(summary: dict) -> list[tuple[str, str, float, dict]]:
    """(model_key, arm, aa_pin, per_map rec) for every trend-plotted map."""
    pts = []
    for map_id, rec in summary["per_map"].items():
        if rec["acc1_cos_calibrated"] is None:
            continue
        mk, arm, pos = _model_of(map_id), _arm_of(map_id), _pos_of(map_id)
        if arm == "b" and pos == "pre_think":
            continue  # OLMo-P companion read; the trend uses the arm's primary map
        pin = PC.AA_PIN.get(mk, (None,))[0]
        if pin is None:
            continue
        rec = dict(rec, map_id=map_id)
        pts.append((mk, arm, float(pin), rec))
    return pts


def _point_label(mk: str, arm: str) -> str:
    lbl = MODEL_LABEL[mk]
    if arm == "a" and PC.PANEL[mk].banked_arm_a:
        lbl += " (banked gen)"  # banked-generation seam marker (plan §6 hero register)
    return lbl


def _scatter_point(ax, x: float, y: float, color, aa_measured: bool) -> None:
    """One trend point; ESTIMATED-AA models render as open markers."""
    if aa_measured:
        ax.scatter(x, y, s=36, color=color, zorder=3)
    else:
        ax.scatter(x, y, s=36, facecolors="none", edgecolors=color, linewidths=1.2, zorder=3)


def fig_capability_trend(summary: dict, out_dir: Path) -> None:
    """Hero 1x2: calibrated acc@1 (null band + per-point ceilings) beside the
    length-residualized read (plan §6 registered elements)."""
    colors = dict(zip(("a", "b"), paper_palette(2), strict=True))
    pts = _trend_points(summary)
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)

    # Left panel: calibrated (excess over the layer-star shuffled null).
    null_sds = [r["null_sd"] for _, _, _, r in pts if r.get("null_sd") is not None]
    if null_sds:
        band = 2.0 * float(np.median(null_sds))
        ax.axhspan(-band, band, color="grey", alpha=0.15, lw=0, label="null ±2·median SD")
    ax.axhline(0.0, color="grey", lw=0.8)
    for mk, arm, pin, rec in pts:
        aa_measured = PC.AA_PIN[mk][2] == "measured"
        y = rec["acc1_cos_calibrated"]
        _scatter_point(ax, pin, y, colors[arm], aa_measured)
        ceil = rec.get("ceiling_retrieval")
        if ceil is not None and rec.get("null_mean") is not None:
            ax.scatter(
                pin,
                float(ceil["ceiling_acc1_cos"]) - float(rec["null_mean"]),
                marker="_",
                s=110,
                color=colors[arm],
                alpha=0.6,
                zorder=2,
            )
        ax.annotate(
            _point_label(mk, arm), (pin, y), fontsize=7, xytext=(4, 4), textcoords="offset points"
        )
    for arm in ("a", "b"):
        ax.scatter([], [], color=colors[arm], label=ARM_LABEL[arm])
    ax.scatter([], [], marker="_", s=110, color="grey", label="repeat-draw ceiling (cal.)")
    ax.scatter([], [], facecolors="none", edgecolors="grey", s=36, label="AA estimated (open)")
    ax.set_xlabel("Artificial Analysis capability index")
    ax.set_ylabel("Calibrated retrieval acc@1 (cosine, excess over null)")
    ax.legend(frameon=False, fontsize=7)

    # Right panel: length-residualized acc@1 (the resid read BESIDE the primary).
    resid = summary.get("resid", {})
    for mk, arm, pin, rec in pts:
        rrec = resid.get(rec["map_id"])
        if rrec is None or rrec.get("resid_acc1_cos") is None:
            continue
        _scatter_point(
            axr, pin, rrec["resid_acc1_cos"], colors[arm], PC.AA_PIN[mk][2] == "measured"
        )
        axr.annotate(
            _point_label(mk, arm),
            (pin, rrec["resid_acc1_cos"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axr.set_xlabel("Artificial Analysis capability index")
    axr.set_ylabel("Length-residualized retrieval acc@1 (cosine)")
    savefig_paper(fig, "c1_capability_trend", dir=out_dir)
    plt.close(fig)


def fig_column_contrasts(summary: dict, out_dir: Path) -> None:
    cols = summary["column_verdicts"]
    colors = dict(zip(("a", "b"), paper_palette(2), strict=True))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = ["3.6 − 3.5", "3.8 − 3.6", "3.8 − 3.5"]
    keys = ["contrast_36_minus_35", "contrast_38_minus_36", "contrast_38_minus_35"]
    for j, arm in enumerate(("a", "b")):
        rec = cols.get(arm, {})
        if rec.get("status") != "complete":
            continue
        xs = [i + (0.15 if j else -0.15) for i in range(3)]
        ys = [rec[k]["delta_cal"] for k in keys]
        los = [y - rec[k]["ci95_cal"][0] for y, k in zip(ys, keys, strict=True)]
        his = [rec[k]["ci95_cal"][1] - y for y, k in zip(ys, keys, strict=True)]
        ax.errorbar(
            xs, ys, yerr=[los, his], fmt="o", color=colors[arm], capsize=3, label=ARM_LABEL[arm]
        )
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(range(3), labels)
    ax.set_xlabel("Qwen 27B capability column contrast")
    ax.set_ylabel("Calibrated Δ acc@1 (95% bootstrap CI, shifted)")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "c2_column_contrasts", dir=out_dir)
    plt.close(fig)


def fig_h2_paired_deltas(summary: dict, out_dir: Path) -> None:
    """H2 per-checkpoint RAW complete-case gaps (E4 primary), both surfaces,
    shifted 95% paired-bootstrap CIs; calibrated stays a trend.py sensitivity."""
    h2 = summary["h2_qwen_thinking"]
    pairs = {k: v for k, v in h2["pairs"].items() if isinstance(v, dict)}
    names = list(pairs)
    surfaces = [("gap_generic_raw", "generic (test_1000)"), ("gap_gpqa_raw", "GPQA transfer")]
    colors = dict(zip([s for s, _ in surfaces], paper_palette(2), strict=True))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for j, (field, label) in enumerate(surfaces):
        xs = [i + (0.18 if j else -0.18) for i in range(len(names))]
        ys = [pairs[k][field] for k in names]
        los = [max(0.0, y - pairs[k][f"{field}_ci95"][0]) for y, k in zip(ys, names, strict=True)]
        his = [max(0.0, pairs[k][f"{field}_ci95"][1] - y) for y, k in zip(ys, names, strict=True)]
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o", color=colors[field], capsize=3, label=label)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(
        range(len(names)), [MODEL_LABEL[k] for k in names], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("Raw Δ acc@1 (end-of-CoT − prompt-end, shared rows)")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "c3_h2_paired_deltas", dir=out_dir)
    plt.close(fig)


def fig_gpqa_transfer(summary: dict, out_dir: Path) -> None:
    recs = summary["gpqa_transfer"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names, vals, chances = [], [], []
    for map_id in sorted(recs):
        r = recs[map_id]
        names.append(f"{MODEL_LABEL[_model_of(map_id)]} ({_pos_of(map_id)})")
        vals.append(r["same_question_acc1_cos"])
        chances.append(r["same_question_chance"])
    xs = range(len(names))
    ax.bar(xs, vals, color=paper_palette(1)[0], label="same-question acc@1 (transfer)")
    ax.plot(xs, chances, "k--", lw=0.9, label="chance")
    ax.set_xticks(xs, names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("GPQA same-question retrieval acc@1 (cosine)")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "c4_gpqa_transfer", dir=out_dir)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=_REPO_ROOT / "eval_results" / "issue_2588" / "trend_summary.json",
    )
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] OK")
        return 0
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_capability_trend(summary, args.out_dir)
    fig_column_contrasts(summary, args.out_dir)
    fig_h2_paired_deltas(summary, args.out_dir)
    fig_gpqa_transfer(summary, args.out_dir)
    print(f"[phase=done] figures written -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
