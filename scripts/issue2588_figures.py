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
POS_LABEL = {
    "prompt_last": "prompt-end read",
    "cot_boundary": "end-of-CoT read",
    "pre_think": "pre-think read",
}
REFERENCE_LABEL = "repeat-draw reference (cal.)"  # repeatability comparator, NOT a bound
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


SHORT_LABEL = {
    "q35_0p8b": "Qwen3.5 0.8B",
    "q35_2b": "Qwen3.5 2B",
    "q35_4b": "Qwen3.5 4B",
    "q35_9b": "Qwen3.5 9B",
    "q35_27b": "Qwen3.5 27B",
    "q36_27b": "Qwen3.6 27B",
    "q38_27b": "Qwen3.8 27B",
    "o3_7b_i": "OLMo 7B Instruct",
    "o3_7b_t": "OLMo 7B Think",
    "o31_32b_i": "OLMo 32B Instruct",
    "o31_32b_t": "OLMo 32B Think",
    "q25_7b": "Qwen2.5 7B",
}


def _point_label(mk: str, arm: str) -> str:
    lbl = SHORT_LABEL[mk]
    if arm == "a" and PC.PANEL[mk].banked_arm_a:
        lbl += " (banked gen)"  # banked-generation seam marker (plan §6 hero register)
    return lbl


def _annotate_staggered(ax, pts: list[tuple[float, float, str]]) -> None:
    """Point labels with per-x-cluster left/right alternation (declutter)."""
    from collections import defaultdict

    groups: dict[float, list[tuple[float, float, str]]] = defaultdict(list)
    for x, y, lbl in pts:
        groups[round(x / 4.0)].append((x, y, lbl))
    two_side = [(5, 4, "left"), (-5, 4, "right"), (5, -10, "left"), (-5, -10, "right")]
    right_only = [(5, 4, "left"), (5, -10, "left"), (5, 12, "left"), (5, -18, "left")]
    for g in groups.values():
        g.sort(key=lambda t: -t[1])
        # leftmost x-clusters: leftward labels would run off the axis edge
        offsets = right_only if min(x for x, _, _ in g) < 10 else two_side
        for k, (x, y, lbl) in enumerate(g):
            dx, dy, ha = offsets[k % 4]
            ax.annotate(lbl, (x, y), fontsize=6, xytext=(dx, dy), textcoords="offset points", ha=ha)


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
    labels_left: list[tuple[float, float, str]] = []
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
        labels_left.append((pin, y, _point_label(mk, arm)))
    _annotate_staggered(ax, labels_left)
    for arm in ("a", "b"):
        ax.scatter([], [], color=colors[arm], label=ARM_LABEL[arm])
    ax.scatter([], [], marker="_", s=110, color="grey", label=REFERENCE_LABEL)
    ax.scatter([], [], facecolors="none", edgecolors="grey", s=36, label="AA estimated (open)")
    ax.set_xlabel("Artificial Analysis capability index")
    ax.set_ylabel("Calibrated retrieval acc@1 (cosine, excess over null)")
    ax.legend(frameon=False, fontsize=7)

    # Right panel: length-residualized acc@1 (the resid read BESIDE the primary).
    resid = summary.get("resid", {})
    labels_right: list[tuple[float, float, str]] = []
    for mk, arm, pin, rec in pts:
        rrec = resid.get(rec["map_id"])
        if rrec is None or rrec.get("resid_acc1_cos") is None:
            continue
        _scatter_point(
            axr, pin, rrec["resid_acc1_cos"], colors[arm], PC.AA_PIN[mk][2] == "measured"
        )
        labels_right.append((pin, rrec["resid_acc1_cos"], _point_label(mk, arm)))
    _annotate_staggered(axr, labels_right)
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
    qcl = summary.get("gpqa_qclustered", {}).get("pairs", {})
    names = list(pairs)
    gpqa_label = "GPQA transfer (question-clustered CI)" if qcl else "GPQA transfer (row-level CI)"
    surfaces = [
        ("gap_generic_raw", "generic corpus (held-out user prompts)"),
        ("gap_gpqa_raw", gpqa_label),
    ]
    colors = dict(zip([s for s, _ in surfaces], paper_palette(2), strict=True))
    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    def _ci(k: str, field: str) -> list[float]:
        # GPQA CIs: prefer the question-clustered recompute (rollouts of one
        # question travel together); generic prompts are independent rows.
        if field == "gap_gpqa_raw" and k in qcl:
            return qcl[k]["gap_gpqa_raw_ci95_qclustered"]
        return pairs[k][f"{field}_ci95"]

    for j, (field, label) in enumerate(surfaces):
        xs = [i + (0.18 if j else -0.18) for i in range(len(names))]
        ys = [pairs[k][field] for k in names]
        los = [max(0.0, y - _ci(k, field)[0]) for y, k in zip(ys, names, strict=True)]
        his = [max(0.0, _ci(k, field)[1] - y) for y, k in zip(ys, names, strict=True)]
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o", color=colors[field], capsize=3, label=label)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(
        range(len(names)), [SHORT_LABEL[k] for k in names], rotation=20, ha="right", fontsize=8
    )
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylabel("Raw Δ acc@1 (end-of-CoT − prompt-end, shared rows)")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "c3_h2_paired_deltas", dir=out_dir)
    plt.close(fig)


def fig_gpqa_transfer(summary: dict, out_dir: Path) -> None:
    recs = summary["gpqa_transfer"]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    names, vals, chances = [], [], []
    pos_short = {
        "prompt_last": "prompt-end",
        "cot_boundary": "end-of-CoT",
        "pre_think": "pre-think",
    }
    for map_id in sorted(recs):
        r = recs[map_id]
        names.append(f"{SHORT_LABEL[_model_of(map_id)]} · {pos_short[_pos_of(map_id)]}")
        vals.append(r["same_question_acc1_cos"])
        chances.append(r["same_question_chance"])
    ys = range(len(names))
    ax.barh(ys, vals, color=paper_palette(1)[0], label="same-question acc@1 (transfer)")
    ax.plot(chances, ys, "k--", lw=0.9, label="chance")
    ax.set_yticks(ys, names, fontsize=7)
    ax.set_ylim(len(names) - 0.3, -0.7)  # top-to-bottom in sorted map order
    ax.set_xlabel("GPQA same-question retrieval acc@1 (cosine)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.set_constrained_layout(False)
    fig.subplots_adjust(left=0.26, bottom=0.09, right=0.97, top=0.97)
    savefig_paper(fig, "c4_gpqa_transfer", dir=out_dir)
    plt.close(fig)


def fig_column_verdict(summary: dict, out_dir: Path, fits_dir: Path) -> None:
    """Hero 1x2 (analyzer round 1): absolute calibrated acc@1 for the three 27B
    releases (both arms, repeat-draw ceilings + length-residualized companions)
    beside ALL SIX column contrasts (arm a filled, arm b open = drop-degraded)."""
    colors = dict(zip(("a", "b"), paper_palette(2), strict=True))
    releases = ["q35_27b", "q36_27b", "q38_27b"]
    rel_labels = ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.8-27B"]
    pos_of_arm = {"a": "prompt_last", "b": "cot_boundary"}
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    resid = summary.get("resid", {})
    for arm in ("a", "b"):
        xs = [i + (0.12 if arm == "b" else -0.12) for i in range(3)]
        ys, ceils, resids = [], [], []
        for mk in releases:
            rec = summary["per_map"][f"{mk}_{arm}.{pos_of_arm[arm]}"]
            ys.append(rec["acc1_cos_calibrated"])
            ceils.append(rec["ceiling_retrieval"]["ceiling_acc1_cos"] - rec["null_mean"])
            rrec = resid.get(f"{mk}_{arm}.{pos_of_arm[arm]}", {})
            resids.append(rrec.get("resid_acc1_cos"))
        ax.plot(xs, ys, "o-", color=colors[arm], label=ARM_LABEL[arm], markersize=7)
        ax.scatter(xs, ceils, marker="_", s=140, color=colors[arm], alpha=0.6)
        ax.scatter(
            xs,
            resids,
            marker="D",
            s=30,
            facecolors="none",
            edgecolors=colors[arm],
            linewidths=1.2,
        )
    ax.scatter([], [], marker="_", s=140, color="grey", label=REFERENCE_LABEL)
    ax.scatter(
        [],
        [],
        marker="D",
        s=30,
        facecolors="none",
        edgecolors="grey",
        linewidths=1.2,
        label="length-residualized",
    )
    ax.set_xticks(range(3), rel_labels)
    ax.set_xlabel("Same-size 27B release (capability index 35 est / 38 / 52)")
    ax.set_ylabel("Retrieval acc@1 (cosine, calibrated)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    labels = ["3.6 − 3.5", "3.8 − 3.6", "3.8 − 3.5"]
    keys = ["contrast_36_minus_35", "contrast_38_minus_36", "contrast_38_minus_35"]
    arm_c_label = {"a": ARM_LABEL["a"], "b": ARM_LABEL["b"] + " (drop-degraded)"}
    for j, arm in enumerate(("a", "b")):
        rec = summary["column_verdicts"][arm]
        xs = [i + (0.15 if j else -0.15) for i in range(3)]
        ys = [rec[k]["delta_cal"] for k in keys]
        los = [max(0.0, y - rec[k]["ci95_cal"][0]) for y, k in zip(ys, keys, strict=True)]
        his = [max(0.0, rec[k]["ci95_cal"][1] - y) for y, k in zip(ys, keys, strict=True)]
        mfc = colors[arm] if arm == "a" else "none"
        axc.errorbar(
            xs,
            ys,
            yerr=[los, his],
            fmt="o",
            color=colors[arm],
            markerfacecolor=mfc,
            markeredgewidth=1.2,
            capsize=3,
            label=arm_c_label[arm],
        )
    axc.axhline(0.0, color="grey", lw=0.8)
    axc.set_xticks(range(3), labels)
    axc.set_xlabel("Column contrast (later release − earlier)")
    axc.set_ylabel("Calibrated Δ acc@1 (95% bootstrap CI, shifted)")
    axc.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "c5_column_verdict", dir=out_dir)
    plt.close(fig)


def fig_column_layer_sweeps(summary: dict, out_dir: Path, fits_dir: Path) -> None:
    """Low-level per-unit view behind the column verdict: per-layer TEST retrieval
    acc@1 curves for the two column endpoints (no-think arm), even-stride sweep
    (lines) + the odd-layer sensitivity pass (small markers), val-frozen stars
    circled."""
    cells = {"q35_27b_a": "Qwen3.5-27B", "q38_27b_a": "Qwen3.8-27B"}
    colors = dict(zip(cells, paper_palette(4)[2:], strict=True))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for cell, label in cells.items():
        even = {}
        for f in sorted((fits_dir / cell).glob("percell_prompt_last_L*.json")):
            layer = int(f.stem.rsplit("_L", 1)[1])
            rec = json.loads(f.read_text())
            even[layer] = rec["knn_test"]["ridge"]["cosine"]["acc_at_k"]["1"]
        xs = sorted(even)
        ax.plot(xs, [even[x] for x in xs], "-", color=colors[cell], label=f"{label} (even sweep)")
        star = summary["per_map"][f"{cell}.prompt_last"]["layer_star"]
        ax.scatter(
            [star],
            [even[star]],
            s=120,
            facecolors="none",
            edgecolors=colors[cell],
            linewidths=1.6,
            zorder=4,
        )
        odd_path = fits_dir.parent / "fits_oddlayers" / cell / "fits_prompt_last_odd.json"
        if odd_path.exists():
            odd = json.loads(odd_path.read_text())
            oxs = sorted(int(k) for k in odd["layers"])
            oys = [
                odd["layers"][str(x)]["knn_test"]["ridge"]["cosine"]["acc_at_k"]["1"] for x in oxs
            ]
            ax.scatter(
                oxs, oys, s=14, color=colors[cell], alpha=0.55, label=f"{label} (odd layers)"
            )
    ax.scatter(
        [],
        [],
        s=120,
        facecolors="none",
        edgecolors="grey",
        linewidths=1.6,
        label="validation-frozen layer",
    )
    ax.set_xlabel("Layer index (64-layer models)")
    ax.set_ylabel("Held-out test retrieval acc@1 (cosine)")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    savefig_paper(fig, "c6_column_layer_sweeps", dir=out_dir)
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
    ap.add_argument(
        "--fits-dir",
        type=Path,
        default=_REPO_ROOT
        / "eval_results"
        / "issue_2588"
        / "hub_mirror"
        / "issue2588_capability_panel"
        / "fits",
        help="local mirror of issue2588_capability_panel/fits (percell + oddlayer inputs)",
    )
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
    fig_column_verdict(summary, args.out_dir, args.fits_dir)
    fig_column_layer_sweeps(summary, args.out_dir, args.fits_dir)
    print(f"[phase=done] figures written -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
