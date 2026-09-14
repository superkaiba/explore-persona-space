"""Compare shift with shift-plus-scalar on the exact paired K5 query cohorts."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    MUTED,
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
)
from scripts import issue2054_k5_loso_calibration as base

GROUPS = [
    ("Chat → plain", lambda i, j: i == 0 and j == 1),
    ("Plain → chat", lambda i, j: i == 1 and j == 0),
    ("Chat → characters", lambda i, j: i == 0 and j >= 2),
    ("Characters → chat", lambda i, j: i >= 2 and j == 0),
    ("Plain → characters", lambda i, j: i == 1 and j >= 2),
    ("Characters → plain", lambda i, j: i >= 2 and j == 1),
    ("Characters → characters", lambda i, j: i >= 2 and j >= 2),
]


def summarize(data):
    """Preserve direction and report unweighted panel summaries and full ranges."""
    summaries = []
    for model in base.MODELS:
        for arm in ("context", "answer"):
            for label, selected in GROUPS:
                rows = [
                    p
                    for p in data["pairs"]
                    if p["model"] == model
                    and p["arm"] == arm
                    and selected(p["source_index"], p["target_index"])
                ]
                if not rows:
                    raise RuntimeError("missing expected directed comparison")
                record = {"model": model, "arm": arm, "group": label, "n_panels": len(rows)}
                for metric in ("displacement_fraction", "r2_mean", "top1_mean"):
                    record[metric] = {}
                    for name in ("identity", "bias", "bias_scale"):
                        values = [r[metric][name] for r in rows]
                        record[metric][name] = {
                            "min": min(values),
                            "max": max(values),
                            "mean": float(np.mean(values)),
                        }
                record["scale"] = {
                    "min": min(r["scale_min"] for r in rows),
                    "max": max(r["scale_max"] for r in rows),
                    "mean": float(np.mean([r["scale_mean"] for r in rows])),
                }
                summaries.append(record)
    return summaries


def main():
    """Render complete source-row/target-column matrices and write exact tables."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figures", type=Path, required=True)
    args = p.parse_args()
    data = json.loads((args.out / "results.json").read_text())
    if data["status"] != "complete" or len(data["pairs"]) != 120:
        raise RuntimeError("incomplete affine comparison")
    summaries = summarize(data)
    base.atomic_json(args.out / "summary.json", summaries)
    counts = {
        "panels": len(data["pairs"]),
        "lower_squared_error_with_scaling": sum(
            r["displacement_fraction"]["bias_scale"] > r["displacement_fraction"]["bias"]
            for r in data["pairs"]
        ),
        "lower_euclidean_top1_with_scaling": sum(
            r["top1_mean"]["bias_scale"] < r["top1_mean"]["bias"] for r in data["pairs"]
        ),
    }
    base.atomic_json(args.out / "metric_tradeoff.json", counts)
    labels = ["Chat", "Plain", "HELIOS", "Wren", "Dana", "Vex"]
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=1.07)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.105, right=0.87, bottom=0.13, top=0.88, wspace=0.38, hspace=0.72)
    cmap = LinearSegmentedColormap.from_list("paired_affine", ["#f7f9fa", ROLES["linear"].color])
    for ri, arm in enumerate(("context", "answer")):
        for ci, model in enumerate(base.MODELS):
            ax = axes[ri, ci]
            selected = [r for r in data["pairs"] if r["model"] == model and r["arm"] == arm]
            matrix = np.full((6, 6), np.nan)
            for r in selected:
                matrix[r["source_index"], r["target_index"]] = (
                    100 * r["displacement_fraction"]["bias_scale"]
                )
            finite = matrix[np.isfinite(matrix)]
            if len(finite) != 30 or min(finite) < 0 or max(finite) > 100:
                raise ValueError("matrix range/coverage does not fit the declared color scale")
            shown = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap)
            for r in selected:
                i, j = r["source_index"], r["target_index"]
                before, after = [
                    100 * r["displacement_fraction"][m] for m in ("bias", "bias_scale")
                ]
                ax.text(
                    j,
                    i,
                    f"{before:.0f}→{after:.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if after >= 65 else INK,
                )
            ax.set_xticks(range(6), labels, rotation=45, ha="right")
            ax.set_yticks(range(6), labels)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            title = "Contexts" if arm == "context" else "Five-rollout mean answers"
            panel_header(
                ax,
                "ABCD"[ri * 2 + ci],
                "Base" if ci == 0 else "Instruction-tuned",
                title=title,
                kicker_y=1.18,
            )
    cb = fig.colorbar(shown, cax=fig.add_axes([0.913, 0.31, 0.018, 0.39]))
    cb.set_label("Paired difference explained after shift + scale (%)")
    fig.text(
        0.5,
        0.035,
        "Cells: shift → shift + scale (%) · Source = row · Target = column",
        ha="center",
        color=INK,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.008,
        "100% = perfect prediction · One scalar and vector bias fitted on separate query folds",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    args.figures.mkdir(parents=True, exist_ok=True)
    saved = save_c2a_figure(
        fig,
        args.figures / "matched_shift_scale",
        title="Shift versus shift plus scale",
        subject=data["displacement_fraction_definition"],
        creator=Path(__file__).name,
        include_width=fraction,
    )
    base.atomic_json(
        args.figures / "matched_shift_scale.meta.json",
        {
            "render": saved["record"],
            "results_sha256": base.sha(args.out / "results.json"),
            "script_sha256": base.sha(__file__),
            "direction": "source rows, target columns",
            "cells": [
                {
                    k: r[k]
                    for k in (
                        "model",
                        "arm",
                        "source_index",
                        "target_index",
                        "displacement_fraction",
                    )
                }
                for r in data["pairs"]
            ],
        },
    )
    plt.close(fig)
    lines = [
        "# Matched K5 representations: shift versus shift plus scalar",
        "",
        "This extends the direct paired-representation test, not the earlier context-to-answer map calibration. Each context vector predicts the corresponding context vector in another setting; answer vectors predict answer vectors. The strict parent query cohorts and global conversation folds are unchanged.",
        "",
        f"Scaling reduces held-out squared error in {counts['lower_squared_error_with_scaling']}/{counts['panels']} directed panels, while lowering mean Euclidean top-1 retrieval in {counts['lower_euclidean_top1_with_scaling']}/{counts['panels']}. This is a predictive error reduction with a query-retrieval tradeoff, not evidence that every setting is a reversibly scaled and shifted copy of another.",
        "",
        data["method"],
        "",
        data["displacement_fraction_definition"],
        "",
        "For training matrices X and Y, a = sum((X−mean X)*(Y−mean Y)) / sum((X−mean X)^2), and b = mean Y − a*mean X. There is one shared scalar per direction, model, representation arm and fold, not a separate scale per query or dimension. No regularization or hyperparameter search is needed. Source and target training pairs are used; no held-out target rows enter fitting.",
        "",
        "Both directions are fitted independently: noisy forward and reverse scales generally are not reciprocals. Shrinkage can improve prediction without providing a reversible transformation. Centered R² and nearest-neighbor retrieval therefore accompany the identity-relative displacement fraction.",
        "",
        "Coverage: 120 directed representation panels and 600 fold evaluations. Every shift-only per-query error was checked against the previous strict result; no cohort or baseline drift was detected. Pool size is the held-out paired fold, with chance top-1 retrieval 1/pool size.",
        "",
        "## Group summaries",
        "",
        "Values below are unweighted means across the indicated directed setting pairs, after pooling displacement errors across held-out queries within each pair. Parent pair cohorts differ; the JSON includes per-pair and fold values and ranges.",
        "",
        "| Model | Arm | Direction | Difference explained: shift → scale | R²: shift → scale | Top-1: shift → scale | Scalar range across folds |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in summaries:
        d, r2, nn = r["displacement_fraction"], r["r2_mean"], r["top1_mean"]
        lines.append(
            f"| {r['model']} | {r['arm']} | {r['group']} | {100 * d['bias']['mean']:.2f}% → {100 * d['bias_scale']['mean']:.2f}% | {r2['bias']['mean']:.4f} → {r2['bias_scale']['mean']:.4f} | {100 * nn['bias']['mean']:.2f}% → {100 * nn['bias_scale']['mean']:.2f}% | {r['scale']['min']:.3f}–{r['scale']['max']:.3f} |"
        )
    lines += [
        "",
        "## Per-pair results",
        "",
        "R² and retrieval entries list identity / shift / shift+scale, in that order.",
        "",
        "| Model | Arm | Direction | N | Difference explained: shift → scale | R² | Top-1 (%) | Mean scalar | Pool range |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in data["pairs"]:
        pools = [f["retrieval_pool"] for f in r["folds"]]
        rr = " / ".join(f"{r['r2_mean'][m]:.4f}" for m in ("identity", "bias", "bias_scale"))
        nn = " / ".join(
            f"{100 * r['top1_mean'][m]:.2f}" for m in ("identity", "bias", "bias_scale")
        )
        lines.append(
            f"| {r['model']} | {r['arm']} | {labels[r['source_index']]} → {labels[r['target_index']]} | {r['n_paired']} | {100 * r['displacement_fraction']['bias']:.2f}% → {100 * r['displacement_fraction']['bias_scale']:.2f}% | {rr} | {nn} | {r['scale_mean']:.3f} | {min(pools)}–{max(pools)} |"
        )
    lines += [
        "",
        "## Limits and provenance",
        "",
        *[f"- {v}" for v in data["limitations"]],
        "",
        "The same full query appears in both prefixes, but surrounding narratives can add information. These remain story/framing comparisons, not isolated persona system-prompt interventions. Finite-five-rollout noise is not corrected, and this extension does not add behavioral or refusal measurements.",
        "",
        "Strict parent: [published matched-query analysis](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/755cec2d8a0ecc9e5028aa6a4e8875d0a76527db/issue2054_section44_k5_gcp/transfer_calibration_v1/matched_queries/results/README.md). Parent result SHA-256: "
        + data["parent_results_sha256"]
        + ".",
        "",
        "Reproduce with `issue2054_k5_matched_run.py --affine --out eval_results/issue_2054/k5_matched_affine --inputs eval_results/issue_2054/k5_matched_offsets_strict/inputs.json`, then `issue2054_k5_matched_affine_plot.py`. Use a fresh output path, with the strict parent available in the sibling directory. Per-query errors/ranks, bias vectors, scalar coefficients, input hashes, code, monitoring and completion records are preserved in this publication.",
        "",
    ]
    (args.out / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
