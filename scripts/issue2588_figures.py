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
                        chance, per map (family/size bar order).
  c5_column_verdict     2x2 column hero: levels, contrasts, identity-plus-bias
                        decomposition, own hard-set accuracy.
  c6_column_layer_sweeps  per-layer endpoint sweeps (even + odd passes).
  c7_endpoint_percell   per-prompt outcome classes behind the endpoint contrast.
  c8_q38_perquestion    per-question hits behind the collapsed newest-27B
                        thinking hard-set cell (raw vs length-residualized).
  c9_hardset_recompute  truncation-free recompute (q38 thinking) + OLMo Think
                        correctness-stratum reads.
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
import matplotlib.ticker as mticker  # noqa: E402
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


def _model_key_order(pts: list[tuple[str, str, float, dict]]) -> list[tuple[str, float]]:
    """Distinct trend models ordered by capability pin (the c1 number key)."""
    seen: dict[str, float] = {}
    for mk, _arm, pin, _rec in pts:
        seen.setdefault(mk, pin)
    return sorted(seen.items(), key=lambda t: (t[1], t[0]))


def _scatter_point(ax, x: float, y: float, color, aa_measured: bool) -> None:
    """One trend point; ESTIMATED-AA models render as open markers."""
    if aa_measured:
        ax.scatter(x, y, s=36, color=color, zorder=3)
    else:
        ax.scatter(x, y, s=36, facecolors="none", edgecolors=color, linewidths=1.2, zorder=3)


def fig_capability_trend(summary: dict, out_dir: Path) -> None:
    """Hero 1x2: calibrated acc@1 (null band + per-point ceilings) beside the
    length-residualized read; checkpoints carry NUMBERS keyed to a side legend
    (collision-free by construction — review round 1, c1 label declutter)."""
    from matplotlib.lines import Line2D

    colors = dict(zip(("a", "b"), paper_palette(2), strict=True))
    pts = _trend_points(summary)
    key_order = _model_key_order(pts)
    key_num = {mk: i + 1 for i, (mk, _pin) in enumerate(key_order)}
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12.6, 4.6), sharex=True)
    fig.set_constrained_layout(False)
    for a in (ax, axr):
        # log-x spreads the low-AA cluster (AA 2-8) that overlapped point labels
        a.set_xscale("log")
        a.set_xticks([2, 5, 10, 20, 50])
        a.xaxis.set_major_formatter(mticker.ScalarFormatter())
        a.minorticks_off()

    def _number(a, x: float, y: float, mk: str, arm: str) -> None:
        # per-arm vertical offsets so the two arms' numbers at one x never collide
        dy = 5 if arm == "b" else -11
        a.annotate(
            str(key_num[mk]),
            (x, y),
            fontsize=6.5,
            xytext=(4, dy),
            textcoords="offset points",
            color=colors[arm],
        )

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
        _number(ax, pin, y, mk, arm)
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
    for arm in ("a", "b"):
        ax.scatter([], [], color=colors[arm], label=ARM_LABEL[arm])
    ax.scatter([], [], marker="_", s=110, color="grey", label=REFERENCE_LABEL)
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
        y = rrec["resid_acc1_cos"]
        _scatter_point(axr, pin, y, colors[arm], PC.AA_PIN[mk][2] == "measured")
        _number(axr, pin, y, mk, arm)
    axr.set_xlabel("Artificial Analysis capability index")
    axr.set_ylabel("Length-residualized retrieval acc@1 (cosine)")

    # Number key (side legend): number glyph as the legend marker, model as label.
    key_handles = []
    for mk, _pin in key_order:
        lbl = SHORT_LABEL[mk]
        if PC.PANEL[mk].banked_arm_a:
            lbl += " (banked gen)"
        key_handles.append(
            Line2D([], [], marker=f"${key_num[mk]}$", color="black", ls="", ms=7, label=lbl)
        )
    fig.subplots_adjust(left=0.06, right=0.845, bottom=0.11, top=0.96, wspace=0.22)
    fig.legend(
        handles=key_handles,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        frameon=False,
        fontsize=7.5,
        title="checkpoint key",
        title_fontsize=8,
    )
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


MODEL_ORDER = [
    "q35_0p8b",
    "q35_2b",
    "q35_4b",
    "q35_9b",
    "q35_27b",
    "q36_27b",
    "q38_27b",
    "q25_7b",
    "o3_7b_i",
    "o3_7b_t",
    "o31_32b_i",
    "o31_32b_t",
]
POS_ORDER = ["prompt_last", "pre_think", "cot_boundary"]


def fig_gpqa_transfer(summary: dict, out_dir: Path) -> None:
    """c4: hard-set transfer per map, bars ordered by family/size then read position
    (review round 1: name-sort put 0.8B next to 27B)."""
    recs = summary["gpqa_transfer"]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    names, vals, chances = [], [], []
    pos_short = {
        "prompt_last": "prompt-end",
        "cot_boundary": "end-of-CoT",
        "pre_think": "pre-think",
    }
    ordered = sorted(
        recs, key=lambda m: (MODEL_ORDER.index(_model_of(m)), POS_ORDER.index(_pos_of(m)))
    )
    for map_id in ordered:
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
    """Hero 2x2 (review round 1): (a) absolute calibrated acc@1 for the three 27B
    releases (both arms, repeat-draw references + length-residualized companions);
    (b) ALL SIX column contrasts (arm a filled, arm b open = drop-degraded);
    (c) identity-plus-bias baseline vs learned-map increment, thinking off;
    (d) the same cells' own hard-set behavioral accuracy."""
    colors = dict(zip(("a", "b"), paper_palette(2), strict=True))
    releases = ["q35_27b", "q36_27b", "q38_27b"]
    rel_labels = ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.8-27B"]
    pos_of_arm = {"a": "prompt_last", "b": "cot_boundary"}
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.8))
    ax, axc, axd, axe = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

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
    ax.set_title("Calibrated levels per release", loc="left", fontsize=9)
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
    axc.set_title("All six paired contrasts", loc="left", fontsize=9)
    axc.legend(frameon=False, fontsize=8)

    # Panel (c): learned-map decomposition (thinking off) — identity-plus-bias
    # baseline vs what ridge adds on top of it (both raw acc@1 at layer_star).
    dec_colors = paper_palette(4)[2:]
    ibs, incs = [], []
    for mk in releases:
        frec = json.loads((fits_dir / f"{mk}_a" / "fits_prompt_last.json").read_text())
        lrec = frec["layers"][str(frec["layer_star"])]["knn_test"]
        ib = float(lrec["identity_bias"]["cosine"]["acc_at_k"]["1"])
        ridge_raw = float(lrec["ridge"]["cosine"]["acc_at_k"]["1"])
        ibs.append(ib)
        incs.append(ridge_raw - ib)
    xs3 = np.arange(3)
    axd.bar(xs3, ibs, width=0.55, color=dec_colors[0], label="identity-plus-bias baseline")
    axd.bar(xs3, incs, width=0.55, bottom=ibs, color=dec_colors[1], label="learned-map increment")
    axd.set_xticks(xs3, rel_labels)
    axd.set_ylim(0, 1.0)
    axd.set_ylabel("Held-out retrieval acc@1 (raw, thinking off)")
    axd.set_title("Baseline vs learned increment", loc="left", fontsize=9)
    axd.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel (d): the same thinking-off cells' OWN hard-set behavioral accuracy.
    split = summary["registered_secondary"]["gpqa_correct_incorrect_split"]
    own = [float(split[f"{mk}_a.prompt_last"]["acc_shipped"]) for mk in releases]
    axe.bar(xs3, own, width=0.55, color=paper_palette(1)[0])
    axe.set_xticks(xs3, rel_labels)
    axe.set_ylabel("Own hard-set accuracy (thinking off)")
    axe.set_title("Behavioral accuracy on the hard set", loc="left", fontsize=9)
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


def fig_endpoint_percell(summary: dict, out_dir: Path, fits_dir: Path) -> None:
    """c7: per-prompt outcome classes behind the endpoint contrast (both arms).

    Exhaustive counts over each arm's shared test prompts (no sampling), so the
    bars carry no error bars by construction; asserts the arm-a class counts
    reconcile with the committed endpoint delta ((only-new - only-old)/n).
    """
    pairs = {
        "a": ("q35_27b_a", "q38_27b_a", "perrow_prompt_last.json"),
        "b": ("q35_27b_b", "q38_27b_b", "perrow_cot_boundary.json"),
    }
    classes = ["hit under both", "only Qwen3.5 27B", "only Qwen3.8 27B", "miss under both"]
    counts: dict[str, list[int]] = {}
    n_shared: dict[str, int] = {}
    for arm, (old, new, fname) in pairs.items():
        rows_old = json.loads((fits_dir / old / fname).read_text(encoding="utf-8"))
        rows_new = json.loads((fits_dir / new / fname).read_text(encoding="utf-8"))
        h_old = dict(zip(rows_old["row_ids"], rows_old["hit1_cos"], strict=True))
        h_new = dict(zip(rows_new["row_ids"], rows_new["hit1_cos"], strict=True))
        shared = sorted(set(h_old) & set(h_new))
        n_shared[arm] = len(shared)
        both = sum(1 for r in shared if h_old[r] and h_new[r])
        only_old = sum(1 for r in shared if h_old[r] and not h_new[r])
        only_new = sum(1 for r in shared if h_new[r] and not h_old[r])
        neither = len(shared) - both - only_old - only_new
        counts[arm] = [both, only_old, only_new, neither]
    # reconcile with the committed contrast (raw delta = (only_new - only_old)/n)
    cv = summary["column_verdicts"]["a"]["contrast_38_minus_35"]
    got = (counts["a"][2] - counts["a"][1]) / n_shared["a"]
    assert abs(got - cv["delta_raw"]) < 1e-9, (got, cv["delta_raw"])
    assert n_shared["a"] == cv["n_shared_rows"], (n_shared["a"], cv["n_shared_rows"])

    colors = dict(zip(("a", "b"), paper_palette(2), strict=True))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(classes))
    w = 0.38
    ax.bar(x - w / 2, counts["a"], width=w, color=colors["a"], label=ARM_LABEL["a"])
    ax.bar(x + w / 2, counts["b"], width=w, color=colors["b"], label=ARM_LABEL["b"])
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("held-out test prompts (count)")
    ax.legend(frameon=False)
    savefig_paper(fig, "c7_endpoint_percell", dir=out_dir)
    plt.close(fig)


def fig_q38_perquestion(out_dir: Path, fits_dir: Path) -> None:
    """c8: per-question companion behind the collapsed newest-27B thinking cell.

    Joins the raw end-of-CoT per-row hit record with the length-residualized
    per-row hits (identical rows) and plots hits-of-rollouts per question, both
    reads. Sidecar points are re-annotated with question ids + hit indicators
    (bar/line readback would drop the identifiers)."""
    cell = fits_dir / "q38_27b_b"
    raw = json.loads((cell / "gpqa_perrow_cot_boundary.json").read_text())
    res = json.loads((cell / "resid_cot_boundary.json").read_text())["gpqa_resid"]
    res_hit = dict(zip(res["row_ids"], res["same_q_hit"], strict=True))
    per_q: dict[str, dict[str, int]] = {}
    for rid, qid, hit in zip(raw["row_ids"], raw["qids"], raw["same_q_hit"], strict=True):
        rec = per_q.setdefault(str(qid), {"rollouts": 0, "hits_raw": 0, "hits_resid": 0})
        rec["rollouts"] += 1
        rec["hits_raw"] += int(hit)
        rec["hits_resid"] += int(res_hit[rid])
    n_rows = sum(r["rollouts"] for r in per_q.values())
    assert n_rows == 726, n_rows
    assert len(per_q) == 159, len(per_q)
    assert sum(r["hits_raw"] for r in per_q.values()) == 6  # the 2-question concentration
    ordered = sorted(
        per_q.items(), key=lambda kv: (-kv[1]["hits_resid"], -kv[1]["hits_raw"], kv[0])
    )
    xs = np.arange(1, len(ordered) + 1)
    resid_y = [kv[1]["hits_resid"] for kv in ordered]
    raw_y = [kv[1]["hits_raw"] for kv in ordered]
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.plot(xs, resid_y, "o", ms=3.2, color=colors[1], label="length-residualized read")
    ax.plot(xs, raw_y, "o", ms=3.2, color=colors[0], alpha=0.75, label="raw end-of-thought read")

    def _qlabel(qid: str) -> str:
        # plain-English rendered label ("question 182"); raw ids stay in provenance
        return f"question {int(qid.rsplit('_', 1)[1])}"

    for i, (qid, rec) in enumerate(ordered):
        if rec["hits_raw"] > 0:
            ax.text(xs[i] + 1.5, rec["hits_raw"], _qlabel(qid), fontsize=6.5, va="center")
    ax.set_xlabel("Question (ranked by residualized hits)")
    ax.set_ylabel("Retrieval hits per question (of kept rollouts)")
    ax.set_yticks(range(0, 6))
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "c8_q38_perquestion", dir=out_dir)
    plt.close(fig)
    # Sidecar re-annotation: per-question identifiers + hit indicators.
    meta_path = out_dir / "c8_q38_perquestion.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["points"] = [
        {
            "question_rank": int(x),
            "question": _qlabel(qid),
            "question_id_raw": qid,
            "kept_rollouts": rec["rollouts"],
            "hits_raw_end_of_thought": rec["hits_raw"],
            "hits_length_residualized": rec["hits_resid"],
        }
        for x, (qid, rec) in zip(xs, ordered, strict=True)
    ]
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def fig_hardset_recompute(summary: dict, out_dir: Path, recompute_path: Path) -> None:
    """c9 (two declared panels): (a) the newest-27B thinking hard-set recompute —
    full-pool vs truncation-free, raw vs length-residualized, question-clustered
    95% CIs + the paired delta; (b) OLMo Think hard-set reads by read position
    and answer-correctness stratum."""
    rc = json.loads(recompute_path.read_text(encoding="utf-8"))
    reads, boot = rc["reads"], rc["bootstrap"]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    pts = [
        (
            "full pool,\nraw",
            reads["kept_full_pool"]["acc1_orig"],
            boot["kept_full_pool"]["acc1_orig_ci95"],
        ),
        (
            "truncation-free,\nraw",
            reads["truncation_free_complete_clusters"]["acc1_orig"],
            boot["truncation_free_complete_clusters"]["acc1_orig_ci95"],
        ),
        (
            "truncation-free,\nlength-residualized",
            reads["truncation_free_complete_clusters"]["acc1_resid"],
            boot["truncation_free_complete_clusters"]["acc1_resid_ci95"],
        ),
        (
            "paired delta\n(resid − raw)",
            reads["truncation_free_complete_clusters"]["delta_resid_minus_orig"],
            boot["truncation_free_complete_clusters"]["delta_resid_minus_orig_ci95"],
        ),
    ]
    xs = np.arange(len(pts))
    ys = [p[1] for p in pts]
    lo = [max(0.0, y - p[2][0]) for y, p in zip(ys, pts, strict=True)]
    hi = [max(0.0, p[2][1] - y) for y, p in zip(ys, pts, strict=True)]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=paper_palette(1)[0], capsize=3)
    chance = reads["kept_full_pool"]["chance_producer_formula"]
    ax.axhline(chance, color="grey", ls="--", lw=0.9, label="chance (same-question)")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(xs, [p[0] for p in pts], fontsize=8)
    ax.set_ylabel("Same-question retrieval acc@1 (95% CI, question-clustered)")
    ax.set_title("Newest-27B thinking cell, recompute", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    # Panel (b): OLMo Think reads by position x correctness stratum.
    tr = summary["gpqa_transfer"]
    split = summary["registered_secondary"]["gpqa_correct_incorrect_split"]
    groups = [
        ("OLMo 7B\npre-think", "o3_7b_t_b.pre_think"),
        ("OLMo 7B\nend-of-thought", "o3_7b_t_b.cot_boundary"),
        ("OLMo 32B\npre-think", "o31_32b_t_b.pre_think"),
        ("OLMo 32B\nend-of-thought", "o31_32b_t_b.cot_boundary"),
    ]
    series = [
        ("all kept rollouts", lambda m: tr[m]["same_question_acc1_cos"]),
        ("correct rows only", lambda m: split[m]["acc1_same_q_correct_rows"]),
        ("incorrect rows only", lambda m: split[m]["acc1_same_q_incorrect_rows"]),
    ]
    scolors = paper_palette(3)
    xg = np.arange(len(groups))
    w = 0.26
    for j, (slabel, getter) in enumerate(series):
        axb.bar(
            xg + (j - 1) * w,
            [getter(m) for _g, m in groups],
            width=w,
            color=scolors[j],
            label=slabel,
        )
    axb.set_xticks(xg, [g for g, _m in groups], fontsize=8)
    axb.set_ylabel("Same-question retrieval acc@1")
    axb.set_title("OLMo Think reads by correctness stratum", loc="left", fontsize=9)
    axb.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "c9_hardset_recompute", dir=out_dir)
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
    ap.add_argument(
        "--recompute",
        type=Path,
        default=_REPO_ROOT
        / "eval_results"
        / "issue_2588"
        / "followup_9ater"
        / "q38_truncfree_gpqa"
        / "q38_truncfree_gpqa.json",
        help="9a-ter truncation-free recompute JSON (c9 panel a)",
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
    fig_endpoint_percell(summary, args.out_dir, args.fits_dir)
    fig_q38_perquestion(args.out_dir, args.fits_dir)
    fig_hardset_recompute(summary, args.out_dir, args.recompute)
    print(f"[phase=done] figures written -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
