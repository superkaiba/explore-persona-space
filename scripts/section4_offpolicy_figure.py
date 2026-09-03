#!/usr/bin/env python3
"""Render the answer-origin (off-policy) results figure for the paper.

Panel A: held-out R^2 of the context-to-answer ridge map refit, at every layer,
on four kinds of answer for the same 4,998 LMSYS contexts (task #823): the
model's own regenerated answer, a plain answer written by Claude Sonnet 4.5, an
eccentric-style answer from the same Claude model, and the model's own answers
shuffled across contexts. A dotted reference gives the content-only map from the
own-answer vector straight to the plain Claude answer vector (#823 identity
baseline, 11-layer grid).

Panel B: held-out R^2 and top-1 retrieval of the map refit on answers written by
k Claude personas over the same contexts (k = 1, 2, 4, 8, 16; #823 follow-up
``inconsistent-origin-persona-ladder``), at layer 19, with the three other fitted
layers as thin lines.

Reads the committed #823 result JSONs (``--eval-root`` defaults to the repo's
``eval_results``; pass a directory populated with ``git show`` when the checkout
is sparse). Writes ``figures/paper/c1_offpolicy_origin.{pdf,png}``, a grayscale
audit, and a JSON sidecar with every plotted value and input hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)

DEFAULT_EVAL_ROOT = ROOT / "eval_results"
DEFAULT_OUT = ROOT / "figures" / "paper"
DEFAULT_STEM = "c1_offpolicy_origin"

TEAL = "#176B87"
PURPLE = "#7B3294"
AMBER = "#C98A1B"

# Arm codes in eval_results/issue_823/ridge_r2_by_arm.json.
SOURCES = [
    ("A_prime", "Own answer (Qwen)", TEAL, "o"),
    ("B2", "Plain answer (Claude)", PURPLE, "s"),
    ("B1", "Eccentric-style answer (Claude)", AMBER, "^"),
    ("C", "Shuffled own answers", MUTED, "x"),
]
LADDER_K = [1, 2, 4, 8, 16]
LADDER_LAYERS = [14, 17, 19, 26]
MAIN_LAYER = 19


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(dirty.stdout.strip())
        if dirty.returncode == 0
        else None,
    }


def load_data(eval_root: Path) -> dict[str, Any]:
    base = eval_root / "issue_823"
    paths = {
        "arms": base / "ridge_r2_by_arm.json",
        "identity": base / "identity_baseline.json",
        "ladder": base / "inconsistent_origin_ladder" / "ladder_analysis_summary.json",
        "ladder_baselines": base
        / "inconsistent_origin_ladder"
        / "ladder_baselines.json",
    }
    arms = json.loads(paths["arms"].read_text())
    identity = json.loads(paths["identity"].read_text())
    ladder = json.loads(paths["ladder"].read_text())
    baselines = json.loads(paths["ladder_baselines"].read_text())

    if int(arms["n_contexts"]) != 4998:
        raise ValueError(f"unexpected n_contexts {arms['n_contexts']}")
    if int(identity["n_valid"]) != 4998:
        raise ValueError(f"unexpected identity-baseline n_valid {identity['n_valid']}")

    # Panel A. The refit curves do not depend on the trait key (the trait only
    # selects a read-out layer), so read them under "evil" and check agreement.
    per_layer: dict[str, dict[str, list[float]]] = {}
    for code, *_ in SOURCES:
        folds = arms["refit"][code]["evil"]["r2_by_layer"]
        for trait in ("sycophancy", "hallucination"):
            if arms["refit"][code][trait]["r2_by_layer"] != folds:
                raise ValueError(
                    f"refit curves differ across trait keys for arm {code}"
                )
        if len(folds) != 28 or any(len(f) != 5 for f in folds):
            raise ValueError(f"unexpected fold layout for arm {code}")
        per_layer[code] = {
            "mean": [statistics.mean(f) for f in folds],
            "sd": [statistics.pstdev(f) for f in folds],
        }
    transfer: dict[str, list[float]] = {}
    for code in ("B2", "B1", "C"):
        block = arms["transfer"][code]["evil"]
        if block["fit_arm"] != "A_prime":
            raise ValueError("transfer block is not the own-answer-fitted map")
        transfer[code] = [statistics.mean(f) for f in block["r2_by_layer"]]
    ref = identity["identity_baseline_r2"]["b2"]
    ref_layers = sorted(int(layer) for layer in ref)
    reference = {
        "layers": ref_layers,
        "own_to_plain_r2": [float(ref[str(layer)]["r2_mean"]) for layer in ref_layers],
        "own_to_plain_sd": [float(ref[str(layer)]["r2_sd"]) for layer in ref_layers],
        "context_to_plain_refit_r2": [
            float(identity["reference_refit"]["B2"][str(layer)]["refit_r2_mean"])
            for layer in ref_layers
        ],
    }

    # Panel B.
    cells = ladder["cells"]
    ladder_r2: dict[str, dict[str, list]] = {}
    for layer in LADDER_LAYERS:
        means, cis = [], []
        for k in LADDER_K:
            cell = cells[f"k{k}:L{layer}"]
            if cell["estimator_degenerate"]:
                raise ValueError(f"degenerate estimator in cell k{k}:L{layer}")
            means.append(float(cell["pooled_r2"]))
            ci = cell["bootstrap_ci_95"]
            cis.append([float(ci["ci_low"]), float(ci["ci_high"])] if ci else None)
        ladder_r2[str(layer)] = {"pooled_r2": means, "bootstrap_ci_95": cis}
    retrieval: dict[str, dict[str, list[float]]] = {}
    chance = None
    for layer in LADDER_LAYERS:
        means, sds = [], []
        for k in LADDER_K:
            fold_acc = []
            for fold in range(5):
                cell = baselines["p1"][f"k{k}:L{layer}:fold{fold}"]
                fold_acc.append(float(cell["knn_cosine"]["acc_at_k"]["1"]))
                chance_here = float(cell["knn_cosine"]["chance_at_k"]["1"])
                chance = chance_here if chance is None else chance
            means.append(statistics.mean(fold_acc))
            sds.append(statistics.pstdev(fold_acc))
        retrieval[str(layer)] = {"top1_cosine_mean": means, "top1_cosine_fold_sd": sds}

    return {
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "panel_a": {
            "n_contexts": 4998,
            "layers": list(range(28)),
            "refit_r2": per_layer,
            "own_map_transfer_r2": transfer,
            "content_only_reference": reference,
        },
        "panel_b": {
            "n_contexts": int(ladder["drop_accounting"]["mask_n"]),
            "k": LADDER_K,
            "layers": LADDER_LAYERS,
            "main_layer": MAIN_LAYER,
            "refit_r2": ladder_r2,
            "top1_retrieval": retrieval,
            "retrieval_chance": chance,
            "headline_drop": {
                "mean_over_read_out_layers": ladder["headline"]["delta_mean"],
                "ci95": [
                    ladder["headline"]["ci_low_delta_mean"],
                    ladder["headline"]["ci_high_delta_mean"],
                ],
            },
        },
    }


def make_figure(data: dict[str, Any]) -> plt.Figure:
    set_c2a_style()
    fig = plt.figure(figsize=(14.4, 6.6), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.55, 1.0],
        left=0.065,
        right=0.99,
        top=0.66,
        bottom=0.14,
        wspace=0.27,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    # Panel A: refit R^2 by layer for each answer source.
    pa = data["panel_a"]
    layers = pa["layers"]
    style_score_axis(ax_a, y_min=0.0, y_max=0.8, y_step=0.2)
    handles_a = []
    for code, label, color, marker in SOURCES:
        mean = pa["refit_r2"][code]["mean"]
        sd = pa["refit_r2"][code]["sd"]
        ax_a.fill_between(
            layers,
            [m - s for m, s in zip(mean, sd)],
            [m + s for m, s in zip(mean, sd)],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax_a.plot(
            layers,
            mean,
            color=color,
            marker=marker,
            markersize=6.5 if marker != "x" else 7.5,
            linewidth=2.0,
            markevery=1,
            zorder=3,
            markeredgewidth=1.6 if marker == "x" else 0.0,
        )
        handles_a.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                markersize=7,
                linewidth=2.0,
                label=label,
                markeredgewidth=1.6 if marker == "x" else 0.0,
            )
        )
    ref = pa["content_only_reference"]
    ax_a.plot(
        ref["layers"],
        ref["own_to_plain_r2"],
        color=INK,
        linestyle=(0, (2, 2)),
        marker="D",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.4,
        linewidth=1.6,
        zorder=4,
    )
    ref_handle = Line2D(
        [],
        [],
        color=INK,
        linestyle=(0, (2, 2)),
        marker="D",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.4,
        linewidth=1.6,
        label="Own answer → plain answer map (reference)",
    )
    ax_a.set_xlim(-0.6, 27.6)
    ax_a.set_xticks([0, 5, 10, 15, 20, 25])
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("Held-out $R^2$ ↑")
    ax_a.text(
        0.0,
        1.21,
        "A  ·  QWEN2.5-7B-INSTRUCT, 4,998 LMSYS CONTEXTS",
        transform=ax_a.transAxes,
        fontsize=11.5,
        fontweight=700,
        color=MUTED,
        va="bottom",
        ha="left",
    )
    ax_a.set_title(
        "Another model's answers are nearly as\npredictable as the model's own",
        loc="left",
        y=1.02,
        pad=0,
        fontweight=650,
        fontsize=17,
        linespacing=1.0,
    )

    # Panel B: persona ladder.
    pb = data["panel_b"]
    ks = pb["k"]
    style_score_axis(ax_b, y_min=0.0, y_max=0.8, y_step=0.2)
    ax_b.set_xscale("log", base=2)
    for layer in pb["layers"]:
        if layer == pb["main_layer"]:
            continue
        ax_b.plot(
            ks,
            pb["refit_r2"][str(layer)]["pooled_r2"],
            color=MUTED,
            linewidth=1.3,
            alpha=0.75,
            marker="o",
            markersize=3.5,
            zorder=2,
        )
    main = str(pb["main_layer"])
    r2 = pb["refit_r2"][main]["pooled_r2"]
    ax_b.plot(ks, r2, color=TEAL, marker="o", markersize=7.5, linewidth=2.2, zorder=4)
    top1 = pb["top1_retrieval"][main]["top1_cosine_mean"]
    top1_sd = pb["top1_retrieval"][main]["top1_cosine_fold_sd"]
    ax_b.errorbar(
        ks,
        top1,
        yerr=top1_sd,
        color=TEAL,
        linestyle=(0, (4, 2.5)),
        marker="o",
        markersize=7.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        linewidth=2.0,
        capsize=3,
        zorder=4,
    )
    ax_b.set_xticks(ks)
    ax_b.set_xticklabels([str(k) for k in ks])
    ax_b.minorticks_off()
    ax_b.set_xlim(0.8, 20)
    ax_b.set_xlabel("Number of source personas")
    ax_b.set_ylabel("Held-out score ↑")
    ax_b.text(
        0.0,
        1.21,
        f"B  ·  ANSWERS FROM k CLAUDE PERSONAS, LAYER {pb['main_layer']}",
        transform=ax_b.transAxes,
        fontsize=11.5,
        fontweight=700,
        color=MUTED,
        va="bottom",
        ha="left",
    )
    ax_b.set_title(
        "Mixing answer origins\ndegrades the map",
        loc="left",
        y=1.02,
        pad=0,
        fontweight=650,
        fontsize=17,
        linespacing=1.0,
    )
    handles_b = [
        Line2D(
            [],
            [],
            color=TEAL,
            marker="o",
            markersize=7.5,
            linewidth=2.2,
            label="Held-out $R^2$",
        ),
        Line2D(
            [],
            [],
            color=TEAL,
            linestyle=(0, (4, 2.5)),
            marker="o",
            markersize=7.5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            linewidth=2.0,
            label="Top-1 retrieval",
        ),
        Line2D(
            [],
            [],
            color=MUTED,
            linewidth=1.3,
            alpha=0.75,
            marker="o",
            markersize=3.5,
            label="$R^2$ at layers 14, 17, 26",
        ),
    ]

    # Legends above each panel, separated by role.
    bbox_a = ax_a.get_position()
    bbox_b = ax_b.get_position()
    fig.text(
        bbox_a.x0,
        0.968,
        "ANSWER SOURCE",
        color=MUTED,
        fontsize=11.5,
        fontweight=750,
        ha="left",
        va="center",
    )
    fig.legend(
        handles=handles_a,
        loc="upper left",
        bbox_to_anchor=(bbox_a.x0 - 0.001, 0.952),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.2,
        handletextpad=0.6,
        borderaxespad=0.0,
        fontsize=14,
    )
    fig.legend(
        handles=[ref_handle],
        loc="upper left",
        bbox_to_anchor=(bbox_a.x0 - 0.001, 0.862),
        ncol=1,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        borderaxespad=0.0,
        fontsize=14,
    )
    fig.text(
        bbox_b.x0,
        0.968,
        "METRIC",
        color=MUTED,
        fontsize=11.5,
        fontweight=750,
        ha="left",
        va="center",
    )
    fig.legend(
        handles=handles_b,
        loc="upper left",
        bbox_to_anchor=(bbox_b.x0 - 0.001, 0.952),
        ncol=1,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        borderaxespad=0.0,
        fontsize=14,
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument(
        "--source-commit",
        default=None,
        help="git commit the input JSONs were taken from (recorded in the sidecar)",
    )
    args = parser.parse_args(argv)
    data = load_data(args.eval_root)
    font = set_c2a_style()
    fig = make_figure(data)
    stem = args.out_dir / args.stem
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Answer-origin results: off-policy answers and the persona ladder",
        subject="Task #823 refit-by-answer-source curves and inconsistent-origin persona ladder rendered for the paper",
        creator="scripts/section4_offpolicy_figure.py",
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
                    "task": 823,
                    "followup_labels": ["inconsistent-origin-persona-ladder"],
                    "source_commit": args.source_commit,
                    "external_model": "claude-sonnet-4-5-20250929",
                    "model": "Qwen2.5-7B-Instruct",
                },
                **data,
                "outputs": {k: str(v.relative_to(ROOT)) for k, v in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for key, path in {**outputs, "data": sidecar}.items():
        print(f"{key}: {path}")
    pa = data["panel_a"]
    for code, label, *_ in SOURCES:
        print(
            f"{label}: L19 refit R2 {pa['refit_r2'][code]['mean'][19]:.3f} (sd {pa['refit_r2'][code]['sd'][19]:.3f})"
        )
    pb = data["panel_b"]
    print("ladder L19 R2:", [round(v, 3) for v in pb["refit_r2"]["19"]["pooled_r2"]])
    print(
        "ladder L19 top-1:",
        [round(v, 3) for v in pb["top1_retrieval"]["19"]["top1_cosine_mean"]],
        "chance",
        pb["retrieval_chance"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
