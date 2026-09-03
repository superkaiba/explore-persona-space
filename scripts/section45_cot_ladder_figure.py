#!/usr/bin/env python3
"""Render the reasoning-SFT reparameterization ladder for the paper appendix.

Reads the Issue #2546 ladder units (``eval_results/issue_2546/ladder``) and
writes ``figures/paper/c1_cot_ladder.{pdf,png}``, a grayscale audit, and a JSON
sidecar. Each unit fits the context-to-answer map on Qwen2.5-7B-Instruct
(before reasoning SFT), applies it to OpenThinker3-7B (after), and allows one
correction fit on training folds only. The y axis is retention, the corrected
map's held-out R^2 divided by the post-training model's own R^2, on a
symmetric-log scale because the uncorrected tiers are in the tens of
thousands below zero. Stratum colors match the main chain-of-thought figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # repo convention: environment before heavy imports

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    PAPER,
    SEAM,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
)

LADDER_DIR = ROOT / "eval_results" / "issue_2546" / "needs_only" / "ladder"  # needs-reasoning-only refits (default); --ladder-dir overrides
DEFAULT_OUT = ROOT / "figures" / "paper"
DEFAULT_STEM = "c1_cot_ladder"
HF_REVISION = "8368cc69f887d20931acd8c4d76c142275173728"
SOURCE_REF = "42308cc7522dcb0a2a76b332b0c24d981de4b585"

# Ladder tier key -> display label, in the ladder's own order of richness.
TIERS = [
    ("t0_direct_transfer", "as is"),
    ("t1_context_offset", "context\noffset"),
    ("t2_answer_offset", "answer\noffset"),
    ("t3_bias_offset", "bias"),
    ("t4_global_scaling", "rescaling\n+ bias"),
    ("t5_mapping_rotation", "rotation\n+ bias"),
    ("t6_reparam_contexts", "change of\nbasis\n(contexts)"),
    ("t7_reparam_answers", "change of\nbasis\n(answers)"),
    ("t8_reparam_both", "change of\nbasis\n(both sides)"),
]
UNITS = [
    ("pooled", "Pooled", INK, "o", 12),
    ("math", "MATH", "#7B3294", "s", 8),
    ("gsm8k_train", "GSM8K train", "#7B3294", "^", 9),
    ("contexthub", "ContextHub", "#7B3294", "v", 9),
    ("mmlu", "MMLU", "#5AAE61", "D", 8),
]
UNIT_DX = {"pooled": 0.0, "math": -0.26, "gsm8k_train": -0.13, "contexthub": 0.13, "mmlu": 0.26}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_units() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, label, _color, _marker, _size in UNITS:
        path = LADDER_DIR / f"ladder__{key}__a1.json"
        if not path.is_file():
            print(f"[ladder] no unit file for {key!r} under {LADDER_DIR}; skipped", flush=True)
            continue
        data = json.loads(path.read_text())
        if data["status"] != "ok" or int(data["arm"]) != 1:
            raise ValueError(f"{path.name}: unexpected status or arm")
        ref = float(data["within_post_reference_r2"])
        tiers = data["tiers_r2"]
        if list(tiers.keys()) != [k for k, _l in TIERS] and not {m[1] for m in MODES} <= set(tiers):
            raise ValueError(f"{path.name}: tiers missing for the compare figure")
        out[key] = {
            "label": label,
            "n_rows": int(data["n_rows"]),
            "reference_r2": ref,
            "band": float(data["band_value"]) if "band_value" in data else None,
            "sufficient_tier": data.get("sufficient_tier"),
            "retention_ci": {k: [float(v["ci_lo"]), float(v["ci_hi"])] for k, v in data.get("retention", {}).items()},
            "tier_r2": {k: float(v) for k, v in tiers.items()},
            "tier_retention": {k: float(v) / ref for k, v in tiers.items()},
            "source": str(path.relative_to(ROOT)),
            "source_sha256": _sha256(path),
        }
    return out


def make_figure(units: dict[str, Any]) -> plt.Figure:
    set_c2a_style()
    fig = plt.figure(figsize=(14.4, 7.0), constrained_layout=False)
    grid = fig.add_gridspec(1, 1, left=0.085, right=0.985, top=0.73, bottom=0.25)
    ax = fig.add_subplot(grid[0, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SEAM)
    ax.spines["bottom"].set_color(SEAM)
    ax.tick_params(length=0, pad=8)
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.6)
    ax.set_ylim(-3e5, 1.6)
    ax.set_yticks([-1e5, -1e3, -10, -1, 0, 1])
    ax.set_yticklabels(["$-10^{5}$", "$-10^{3}$", "$-10$", "$-1$", "0", "1"])
    ax.grid(axis="y", color=GRID, lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.0, color=SEAM, lw=1.0, zorder=1)
    pooled = units["pooled"]
    if pooled.get("band") is not None:
        band_lo = 1.0 - pooled["band"] / pooled["reference_r2"]
        ax.axhspan(band_lo, 1.0, color=MUTED, alpha=0.12, lw=0, zorder=0)

    for key, _label, color, marker, size in UNITS:
        unit = units[key]
        xs = [i + UNIT_DX[key] for i in range(len(TIERS))]
        ys = [unit["tier_retention"][k] for k, _l in TIERS]
        ax.plot(xs, ys, marker=marker, color=color, markersize=size, lw=0, zorder=4 if key == "pooled" else 3, alpha=1.0 if key == "pooled" else 0.9)
    ax.set_xlim(-0.6, len(TIERS) - 0.4)
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([label for _k, label in TIERS], fontsize=13.5, linespacing=1.15)
    ax.set_xlabel("Correction fit on training folds before applying the pre-SFT map to the post-SFT model", labelpad=12)
    ax.set_ylabel("Retention\n(transferred $R^2$ / own $R^2$)  ↑", labelpad=10)
    ax.set_title("Only a change of basis on both sides recovers the map after reasoning SFT", loc="left", y=1.04, pad=0, fontweight=650, fontsize=19)
    ax.text(0.0, 1.16, "QWEN2.5-7B-INSTRUCT → OPENTHINKER3-7B, LAYER 19, POOLED AND PER-CORPUS FITS", transform=ax.transAxes, fontsize=12.5, fontweight=700, color=MUTED, va="bottom", ha="left")

    handles = [
        Line2D([0], [0], color=color, marker=marker, markersize=size, lw=0, label=label)
        for _key, label, color, marker, size in UNITS
    ]
    handles.append(Line2D([0], [0], color=MUTED, lw=1.3, ls=(0, (4, 3)), label="Post-SFT model's own map (retention 1)"))
    fig.text(0.085, 0.965, "FIT UNIT", color=MUTED, fontsize=11.5, fontweight=750, ha="left", va="center")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.084, 0.95), ncol=6, frameon=False, columnspacing=1.3, handlelength=1.4, handletextpad=0.6, borderaxespad=0)
    return fig



# --- Comparison figure: the three corrections of Section 4.3, OLMo stages next to reasoning SFT ---
OLMO_SUMMARY = ROOT / "eval_results" / "issue_1902" / "lasttoken_transfer" / "summary.json"
OLMO_TRANSITIONS = [("B->S", "Base → SFT\n(OLMo-2-7B,\ninstruction SFT)"), ("S->D", "SFT → DPO\n(OLMo-2-7B)"), ("D->R", "DPO → RLVR\n(OLMo-2-7B)")]
REASONING_LABEL = "Qwen2.5-7B-Instruct →\nOpenThinker3-7B\n(reasoning SFT)"
# (olmo key, ladder tier, label, offset, marker, facecolor, edgecolor) — styles mirror Figure 8C.
MODES = [
    ("direct", "t0_direct_transfer", "as is", -0.24, "o", "white", "#C9583D"),
    ("bias", "t3_bias_offset", "bias", 0.0, "s", "#8CBAC5", "#16708A"),
    ("scale_bias", "t4_global_scaling", "rescaling + bias", 0.24, "o", "#16708A", "#16708A"),
]


def load_olmo() -> dict[str, Any]:
    summary = json.loads(OLMO_SUMMARY.read_text())["transfer"]
    out: dict[str, Any] = {}
    for key, _label in OLMO_TRANSITIONS:
        pair = summary[key]["retention"]
        out[key] = {mode: {"point": float(pair[mode]["point"]), "ci": [float(v) for v in pair[mode]["cluster_ci"]]} for mode, *_ in MODES}
    return out


def make_compare_figure(units: dict[str, Any], olmo: dict[str, Any]) -> plt.Figure:
    set_c2a_style()
    fig = plt.figure(figsize=(11.2, 6.6), constrained_layout=False)
    grid = fig.add_gridspec(1, 1, left=0.10, right=0.985, top=0.74, bottom=0.22)
    ax = fig.add_subplot(grid[0, 0])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(SEAM); ax.spines["bottom"].set_color(SEAM)
    ax.tick_params(length=0, pad=8)
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.8)
    ax.set_ylim(-3e5, 1.6)
    ax.set_yticks([-1e5, -1e3, -10, -1, 0, 1])
    ax.set_yticklabels(["$-10^{5}$", "$-10^{3}$", "$-10$", "$-1$", "0", "1"])
    ax.grid(axis="y", color=GRID, lw=1.0, alpha=0.55); ax.set_axisbelow(True)
    ax.axhline(1.0, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.0, color=SEAM, lw=1.0, zorder=1)
    xs = list(range(len(OLMO_TRANSITIONS) + 1))
    for i, (key, _label) in enumerate(OLMO_TRANSITIONS):
        for mode, _tier, _lab, dx, marker, face, edge in MODES:
            pt = olmo[key][mode]["point"]; lo, hi = olmo[key][mode]["ci"]
            ax.errorbar(i + dx, pt, yerr=[[pt - lo], [hi - pt]], fmt=marker, markersize=9, markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.6, ecolor=edge, elinewidth=1.3, capsize=3, lw=0, zorder=4)
    xr = len(OLMO_TRANSITIONS)
    for mode, tier, _lab, dx, marker, face, edge in MODES:
        for key, unit in units.items():
            v = unit["tier_retention"][tier]
            big = key == "pooled"
            ci = unit.get("retention_ci", {}).get(tier)
            if big and ci:
                ax.errorbar(xr + dx, v, yerr=[[max(v - ci[0], 0.0)], [max(ci[1] - v, 0.0)]], fmt="none", ecolor=edge, elinewidth=1.3, capsize=3, zorder=4)  # skewed bootstrap can put the point outside the percentile interval
            ax.plot(xr + dx + (0 if big else 0.0), v, marker=marker, markersize=9 if big else 5.5, markerfacecolor=face if big else PAPER, markeredgecolor=edge, markeredgewidth=1.6 if big else 1.1, lw=0, alpha=1.0 if big else 0.8, zorder=5 if big else 3)
    ax.axvline(xr - 0.5, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.set_xlim(-0.6, xr + 0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _k, label in OLMO_TRANSITIONS] + [REASONING_LABEL], fontsize=12.5, linespacing=1.15)
    ax.set_ylabel("Retention\n(transferred $R^2$ / own $R^2$)  ↑", labelpad=10)
    ax.set_title("Reasoning SFT changes the map much more than other forms of post-training", loc="left", y=1.04, pad=0, fontweight=650, fontsize=17)
    ax.text(0.0, 1.14, "PREVIOUS STAGE'S MAP APPLIED TO THE NEXT STAGE, AFTER THREE CORRECTIONS FIT ON TRAINING FOLDS", transform=ax.transAxes, fontsize=11.5, fontweight=700, color=MUTED, va="bottom", ha="left")
    handles = [Line2D([0], [0], marker=m, markersize=9, markerfacecolor=f, markeredgecolor=e, markeredgewidth=1.6, lw=0, label=lab) for _o, _t, lab, _dx, m, f, e in MODES]
    if any(key != "pooled" for key in units):
        handles.append(Line2D([0], [0], marker="o", markersize=5.5, markerfacecolor=PAPER, markeredgecolor=INK, markeredgewidth=1.1, lw=0, label="single corpus (reasoning SFT)"))
    handles.append(Line2D([0], [0], color=MUTED, lw=1.3, ls=(0, (4, 3)), label="own map of the next stage (retention 1)"))
    fig.text(0.10, 0.965, "CORRECTION", color=MUTED, fontsize=11.5, fontweight=750, ha="left", va="center")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.099, 0.948), ncol=5, frameon=False, columnspacing=1.2, handlelength=1.4, handletextpad=0.6, borderaxespad=0)
    return fig


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, check=False, capture_output=True, text=True)
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    global LADDER_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--mode", choices=("compare", "full"), default="compare", help="compare: the three Section 4.3 corrections next to the OLMo stages (paper figure); full: all nine ladder tiers")
    parser.add_argument("--ladder-dir", type=Path, default=LADDER_DIR, help="ladder unit JSON dir (default: needs-reasoning-only refits; use eval_results/issue_2546/ladder for the whole-corpus units)")
    args = parser.parse_args(argv)
    LADDER_DIR = args.ladder_dir
    units = load_units()
    olmo = load_olmo() if args.mode == "compare" else None
    font = set_c2a_style()
    fig = make_compare_figure(units, olmo) if args.mode == "compare" else make_figure(units)
    stem = args.out_dir / args.stem
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Reparameterization ladder across reasoning SFT",
        subject="Issue #2546 ladder units rendered for the paper",
        creator="scripts/section45_cot_ladder_figure.py",
    )
    plt.close(fig)
    sidecar = stem.with_name(f"{args.stem}_data.json")
    sidecar.write_text(
        json.dumps(
            {
                "style_version": STYLE_VERSION,
                "font": font,
                "git": _git_state(),
                "provenance": {
                    "task": 2546,
                    "hf_revision": HF_REVISION,
                    "source_ref": SOURCE_REF,
                    "retention": "tier held-out R^2 divided by the post-SFT model's own context-to-answer R^2 on the same rows",
                    "band": "0.02 elicitation band rescaled by the unit's reference R^2 over the #1336 anchor 0.6731; shaded for the pooled unit",
                },
                "mode": args.mode,
                "olmo_retention": olmo,
                "olmo_source": str(OLMO_SUMMARY.relative_to(ROOT)),
                "tiers": [{"key": k, "label": l.replace("\n", " ")} for k, l in TIERS],
                "units": units,
                "outputs": {k: str(v.relative_to(ROOT)) for k, v in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for key, path in {**outputs, "data": sidecar}.items():
        print(f"{key}: {path}")
    for key, unit in units.items():
        print(f"{key:12s} ref={unit['reference_r2']:.3f} retention:", {k.split('_', 1)[1]: (round(v, 3) if abs(v) < 100 else round(v)) for k, v in unit["tier_retention"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
