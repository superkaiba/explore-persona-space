"""Render four frozen-direction reads for the million map on cached evaluations.

Replay the existing 2,000 paired group resamples to add the direct answer-on-context
baseline to the held-out summary. Generic chat deliberately retains map overlaps.
No activations, labels, directions, maps, or behavioral readouts are fitted here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.experiments.issue_1739.arms import bootstrap_rhos  # noqa: E402

DATA = Path("/mnt/eps-data/thomasjiralerspong")
LARGE = DATA / "issue1739-fixed-regimes-20260917"
SMALL = DATA / "issue1739-small-map-20260917/outputs"
GENERIC = DATA / "issue1739-map-size-diagnostic-20260917"
OUT = ROOT / "eval_results/issue_1739/four_method_figure_20260917"
STEM = "c5_behavior_transfer_million_four"
METHODS = (
    "answer_direction_on_context",
    "mapped_answer",
    "context_native",
    "real_answer",
)
LABELS = (
    "Answer → context",
    "Answer → mapped answer",
    "Context → context",
    "Answer → real answer",
)
BEHAVIORS = ("sycophancy", "hallucination", "evil")
HEADINGS = ("Sycophancy", "Hallucination", "Harmful compliance")


def sha(path):
    """Hash exact cached bytes for provenance and completion checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified(folder, name, inputs):
    """Refuse a source artifact that differs from its completed producer."""
    path = folder / name
    expected = json.loads((folder / "complete.json").read_text())["artifact_sha256"][name]
    digest = sha(path)
    if digest != expected:
        raise ValueError(f"Changed completed input: {path}")
    inputs[str(path)] = digest
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            return dict(arrays)
    return json.loads(path.read_text())


def estimate(value, draws):
    """Summarize finite pointwise percentile intervals without zero imputation."""
    if not np.isfinite(value) or not np.isfinite(draws).all():
        raise ValueError("Undefined plotted correlation or bootstrap draw")
    return {"rho": float(value), "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def assemble():
    """Join by context ID and replay the frozen sampler, checking old-draw parity."""
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, cells, all_draws = {}, [], {}
    for behavior in BEHAVIORS:
        large = read_verified(LARGE / behavior, "predictions.npz", inputs)
        small = read_verified(SMALL / behavior, "predictions.npz", inputs)
        old_draws = read_verified(LARGE / behavior, "bootstraps.npz", inputs)
        lookup = {str(cid): i for i, cid in enumerate(small["context_ids"])}
        if len(lookup) != len(small["context_ids"]):
            raise ValueError("Duplicate small-map evaluation IDs")
        ix = np.array([lookup[str(cid)] for cid in large["context_ids"]])
        for key in ("context_ids", "dv", "groups", "rungs"):
            np.testing.assert_array_equal(large[key], small[key][ix])
        lm, sm = list(large["arms"]), list(small["arms"])
        for method in ("real_answer", "context_native"):
            np.testing.assert_allclose(
                large["predictions"][lm.index(method)],
                small["predictions"][sm.index(method), ix],
                atol=1e-8,
                rtol=1e-8,
            )
        prediction = np.stack(
            [small["predictions"][sm.index(METHODS[0]), ix]]
            + [large["predictions"][lm.index(m)] for m in METHODS[1:]]
        )
        datasets = {}
        for rung_index, rung in enumerate(sorted(set(large["rungs"]))):
            if rung == "wildchat_rung":
                continue
            keep = large["rungs"] == rung
            pred, dv, groups = prediction[:, keep], large["dv"][keep], large["groups"][keep]
            unique = sorted(set(groups.tolist()))
            members = [np.flatnonzero(groups == group) for group in unique]
            single = all(len(member) == 1 for member in members)
            order = np.array([member[0] for member in members])
            seed = 1739963 + ("evil", "sycophancy", "hallucination").index(behavior)
            rng = np.random.default_rng(np.random.SeedSequence([seed, rung_index]))
            batches = []
            for start in range(0, 2000, 100):
                choices = rng.integers(0, len(unique), size=(100, len(unique)))
                if single:
                    batch = bootstrap_rhos(pred, dv, order[choices])
                else:
                    indices = [np.concatenate([members[g] for g in row]) for row in choices]
                    batch = np.empty((len(METHODS), 100))
                    lengths = np.array([len(index) for index in indices])
                    for length in np.unique(lengths):
                        selected = np.flatnonzero(lengths == length)
                        batch[:, selected] = bootstrap_rhos(
                            pred, dv, np.stack([indices[j] for j in selected])
                        )
                batches.append(batch)
            draws = np.concatenate(batches, axis=1)
            for method in METHODS[1:]:
                np.testing.assert_allclose(
                    draws[METHODS.index(method)],
                    old_draws[str(rung)][lm.index(method)],
                    atol=1e-12,
                    rtol=1e-12,
                )
            rho = np.array([spearmanr(row, dv).statistic for row in pred])
            datasets[str(rung)] = {"n": int(keep.sum()), "rho": rho, "draws": draws}
            all_draws[f"{behavior}_{rung}"] = draws
            print(f"Verified {behavior}/{rung}: n={keep.sum()}, 2000 paired draws", flush=True)

        generic = read_verified(GENERIC / behavior, "generic_results.json", inputs)["generic"][0]
        if generic["subset"] != "historical_eval_including_overlap":
            raise ValueError("Unexpected generic-chat pool")
        cells.append({"behavior": behavior, "regime": "Generic chat", **generic})
        for regime, rungs in (
            ("ID", ["train"]),
            ("OOD", [r for r in datasets if r != "train"]),
        ):
            rho = np.mean([datasets[r]["rho"] for r in rungs], axis=0)
            draws = np.mean([datasets[r]["draws"] for r in rungs], axis=0)
            cells.append(
                {
                    "behavior": behavior,
                    "regime": regime,
                    "datasets": rungs,
                    "n": sum(datasets[r]["n"] for r in rungs),
                    "arms": {m: estimate(rho[i], draws[i]) for i, m in enumerate(METHODS)},
                }
            )
    data = {
        "map_training_pairs": 963444,
        "layer": 19,
        "map_sha256": "188486f8afd9d95221e32492f3a0be2a3bdb2098cbe7fadfecf1d46433567909",
        "directions": "Frozen raw positive-minus-negative instruction contrasts; unfiltered",
        "methods": METHODS,
        "inputs": inputs,
        "cells": cells,
        "uncertainty": "2000 paired group-bootstrap draws, conditional on frozen directions/map",
        "ood_aggregation": "Equal-weight mean of dataset-specific Spearman correlations",
        "generic_pool": "Historical 419 candidates, retaining map-training/validation overlaps",
        "caveats": [
            "Generic chat is not conversation-disjoint from map training.",
            "Generic hallucination is a graded trait score; QA uses fabricated fractions.",
            "Map answer pooling includes closing tokens; evaluation uses completion tokens.",
            "Negative real-answer correlations on TriviaQA/NQ limit direction validity.",
        ],
        "validation": "Shared-row/DV/group parity and all 24 original bootstrap series reproduced",
    }
    np.savez_compressed(OUT / "bootstraps.npz", **all_draws)
    (OUT / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def render(data):
    """Draw all 36 estimates with literal CI endpoints and a shared axis scale."""
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.80)
    axes = fig.subplots(3, 3, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.18, hspace=0.70, wspace=0.17)
    colors = [MUTED, ROLES["linear"].color, MUTED, INK]
    faces = [MUTED, ROLES["linear"].color, PAPER, INK]
    for row, (behavior, heading) in enumerate(zip(BEHAVIORS, HEADINGS, strict=True)):
        for col, regime in enumerate(("Generic chat", "ID", "OOD")):
            ax = axes[row, col]
            cell = next(
                c for c in data["cells"] if c["behavior"] == behavior and c["regime"] == regime
            )
            for i, method in enumerate(METHODS):
                value = cell["arms"][method]["rho"]
                lo, hi = cell["arms"][method]["ci95"]
                if not np.isfinite([value, lo, hi]).all() or not -0.45 <= lo <= hi <= 0.75:
                    raise ValueError(f"Invalid or clipped interval: {cell}")
                ax.bar(
                    i,
                    value,
                    width=0.64,
                    color=faces[i],
                    edgecolor=INK if i == 0 else colors[i],
                    linewidth=1.4,
                    hatch="///" if i == 0 else None,
                    label=LABELS[i],
                )
                ax.vlines(i, lo, hi, color=INK, linewidth=1.2)
                ax.hlines([lo, hi], i - 0.09, i + 0.09, color=INK, linewidth=1.2)
            ax.set_ylim(-0.45, 0.75)
            ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
            ax.set_xticks([])
            ax.set_xlim(-0.55, 3.55)
            style_axis(ax, grid_axis="none")
            ax.axhline(0, color=MUTED, linewidth=0.65)
            panel_header(ax, chr(65 + row * 3 + col), f"n = {cell['n']:,}", kicker_y=1.03)
            if col == 0:
                ax.set_ylabel(better_label(r"Spearman $\rho$"))
            else:
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0)
            if row == 0:
                fig.text(ax.get_position().x0, 0.94, regime, va="top", fontweight=650)
        fig.text(0.085, axes[row, 0].get_position().y1 + 0.045, heading, fontweight=650)
    fig.suptitle("Fixed-direction projections · million-context map", y=0.985, fontweight=650)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.055),
        handlelength=1.2,
        columnspacing=1.5,
    )
    fig.text(
        0.085,
        0.032,
        "Generic chat includes map-training overlap. OOD averages dataset correlations.",
        color=MUTED,
    )
    fig.text(
        0.085,
        0.008,
        "Whiskers: 95% group-bootstrap intervals. Same fixed directions across all panels.",
        color=MUTED,
    )
    output = save_c2a_figure(
        fig,
        ROOT / "figures/paper" / STEM,
        title="Four fixed-direction projections with the million map",
        subject="Generic chat, ID and OOD; three behaviors",
        creator=str(Path(__file__).relative_to(ROOT)),
        include_width=fraction,
    )
    metadata = {**data, "render": output["record"], "renderer_sha256": sha(Path(__file__))}
    for key in ("pdf", "png", "grayscale"):
        metadata[f"{key}_sha256"] = sha(output[key])
    (ROOT / "figures/paper" / f"{STEM}.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(fig)
    print(json.dumps({k: str(v) for k, v in output.items() if k != "record"}), flush=True)


if __name__ == "__main__":
    summary = OUT / "summary.json"
    data = json.loads(summary.read_text()) if summary.exists() else assemble()
    for source, digest in data["inputs"].items():
        if sha(Path(source)) != digest:
            raise ValueError(f"Changed plotting input: {source}")
    render(data)
