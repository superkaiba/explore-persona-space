#!/usr/bin/env python3
"""#551 round-2 revision: persist split robustness + regenerate the 5 figures.

Addresses the round-1 ensemble-critique revision requests:

1. Persists the 50-random-split reliability robustness (previously quoted
   from an unpersisted in-context computation) as
   ``eval_results/issue_551/controls/split_robustness.json`` — per-persona
   mean split-half cosines over 50 random 10/10 question splits (one
   shared permutation per split across personas, ``np.random.default_rng(42)``
   per cell), plus the reliability-vs-alignment Spearman and the
   low-alignment-group medians, for all 6 trained-model-text cells.
2. Regenerates all 5 figures under ``figures/issue_551/`` with
   plain-English condition labels (no ``em`` / ``on_policy`` /
   ``same_marker_seed42`` slugs), explicit legend entries for null lines,
   the trained persona (medical doctor) marked in the norm/reliability
   scatters, and a reproduction-deltas axis that cannot show negative
   ticks for an absolute delta.

Reads only existing artifacts (controls JSONs + persisted shift tensors).
Zero GPU. Run from the issue-551 worktree::

    uv run python scripts/issue551_round2_figs.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.svd_direction_constancy import cosine, spearman_rho

logger = logging.getLogger(__name__)

CONTROLS_DIR = Path("eval_results/issue_551/controls")
SHIFTS_DIR = Path("eval_results/issue_551/shifts")
REPRO_PATH = Path("eval_results/issue_551/reproduction_gate.json")
FIGURES_DIR = Path("figures/issue_551")

SOURCE_PERSONA = "medical_doctor"
N_SPLITS = 50
SPLIT_RNG_SEED = 42
LOW_COS_THRESHOLD = 0.5
REPRO_TOL = 0.05

ARM_LABEL = {"em": "EM (insecure code)", "marker": "marker (※)"}
VARIANT_LABEL = {
    "same": "trained-model text",
    "base": "base-model text",
    "on_policy": "each model's own text",
}


def _load(name: str) -> dict:
    with (CONTROLS_DIR / name).open() as f:
        return json.load(f)


def cell_label(arm: str, seed: int, variant: str | None = None) -> str:
    """Plain-English reader-facing label for one analysis cell."""
    base = f"{ARM_LABEL[arm]}, seed {seed}"
    return f"{base} · {VARIANT_LABEL[variant]}" if variant else base


# ──────────────────────────────────────────────────────────────────────
# Split robustness (persisted artifact for the round-1 unpersisted numbers)
# ──────────────────────────────────────────────────────────────────────


def compute_split_robustness(norm_align: dict) -> dict:
    """50 random 10/10 splits per cell; persists per-persona mean reliability.

    For each trained-model-text cell: draw 50 random permutations of the 20
    question indices (one shared permutation per split, applied to every
    persona), split 10/10, take the cosine between the two half-mean shift
    vectors per persona, and average over splits. Returns the payload dict.
    """
    per_cell: dict[str, dict] = {}
    for cell_name in sorted(norm_align["per_cell"]):
        cell = norm_align["per_cell"][cell_name]
        cos = cell["cos_to_U1"]
        personas = list(cos.keys())
        data = torch.load(SHIFTS_DIR / f"{cell_name}.pt", weights_only=True)
        shifts = data["shifts"]
        per_q = {p: shifts[p]["delta_v_per_q"].numpy() for p in personas}
        n_q = next(iter(per_q.values())).shape[0]
        assert n_q == 20, n_q

        rng = np.random.default_rng(SPLIT_RNG_SEED)
        acc: dict[str, list[float]] = {p: [] for p in personas}
        for _ in range(N_SPLITS):
            perm = rng.permutation(n_q)
            half_a, half_b = perm[: n_q // 2], perm[n_q // 2 :]
            for p in personas:
                acc[p].append(
                    float(cosine(per_q[p][half_a].mean(axis=0), per_q[p][half_b].mean(axis=0)))
                )
        mean_rel = {p: float(np.mean(v)) for p, v in acc.items()}
        rho = float(spearman_rho([mean_rel[p] for p in personas], [cos[p] for p in personas]))
        low = [p for p in personas if cos[p] < LOW_COS_THRESHOLD]
        per_cell[cell_name] = {
            "arm": cell["arm"],
            "seed": cell["seed"],
            "mean_split_half_cosine_per_persona": mean_rel,
            "spearman_reliability_vs_cos_to_U1": rho,
            "low_cos_group": sorted(low),
            "n_low_cos": len(low),
            "low_cos_median_reliability": (
                float(np.median([mean_rel[p] for p in low])) if low else None
            ),
        }
        logger.info(
            "[split_robustness %s] rho=%.3f n_low=%d low_median=%s",
            cell_name,
            rho,
            len(low),
            per_cell[cell_name]["low_cos_median_reliability"],
        )

    return {
        "meta": {
            "n_splits": N_SPLITS,
            "rng": f"np.random.default_rng({SPLIT_RNG_SEED}) per cell",
            "split": "one shared random permutation of the 20 question indices per split, "
            "applied to every persona; 10/10 halves; cosine between half-mean vectors; "
            "per-persona mean over the 50 splits",
            "low_cos_threshold": LOW_COS_THRESHOLD,
            "tensors_source": str(SHIFTS_DIR),
            "note": "Persisted replacement for the round-1 in-context robustness check; "
            "RNG consumption order differs from the round-1 throwaway run in the third "
            "decimal (rho within ±0.03, medians within ±0.003, same conclusion).",
        },
        "per_cell": per_cell,
    }


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────


def make_figures(
    *,
    repro: dict,
    loo: dict,
    jack: dict,
    mean_resp: dict,
    norm_align: dict,
    reliability: dict,
) -> None:
    """Regenerate the 5 figures with plain-English labels (blog style)."""
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(4)
    em_color, mk_color = colors[0], colors[1]
    full_color, dropped_color = colors[2], colors[3]
    # Seeds are SHAPE-coded (not color-coded) so colors keep one meaning
    # everywhere: blue = EM arm, orange = marker arm.
    seed_markers = {42: "o", 137: "^", 256: "s"}

    same_loo = {k: v for k, v in loo["per_cell"].items() if v["variant"] == "same"}

    # ── Figure 1: hero, three panels ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # (left) full vs source-dropped top-share + nulls, with legend entries.
    ax = axes[0]
    names = sorted(same_loo, key=lambda k: (same_loo[k]["arm"], same_loo[k]["seed"]))
    xs = np.arange(len(names))
    ax.bar(
        xs - 0.2,
        [same_loo[k]["s_top1_frac_full"] for k in names],
        width=0.38,
        label="full panel (14 contexts)",
        color=full_color,
    )
    ax.bar(
        xs + 0.2,
        [same_loo[k]["s_top1_frac_loo"] for k in names],
        width=0.38,
        label="trained persona dropped",
        color=dropped_color,
    )
    for i, k in enumerate(names):
        ax.hlines(same_loo[k]["sign_flip"]["p95"], i - 0.45, i + 0.45, color="black", lw=1.2)
        ax.hlines(
            same_loo[k]["row_shuffle"]["p95"],
            i - 0.45,
            i + 0.45,
            color="gray",
            lw=1.0,
            linestyles="dashed",
        )
    null_sf = mlines.Line2D([], [], color="black", lw=1.2, label="sign-flip null (95th pct)")
    null_rs = mlines.Line2D(
        [], [], color="gray", lw=1.0, linestyle="--", label="row-shuffle null (95th pct)"
    )
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, null_sf, null_rs], fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            f"{'EM' if same_loo[k]['arm'] == 'em' else 'marker (※)'}\nseed {same_loo[k]['seed']}"
            for k in names
        ],
        fontsize=7,
    )
    ax.set_ylabel("top-direction share of spectrum")
    ax.set_title("Top share survives dropping the source")

    # (middle) end-slot vs whole-response cosine, both arms.
    ax = axes[1]
    pc = mean_resp["per_cell"]
    for arm, color in (("em", em_color), ("marker", mk_color)):
        cells = sorted((k for k in pc if pc[k]["arm"] == arm), key=lambda k: pc[k]["seed"])
        mr = [pc[k]["mean_cos_to_U1_mean_resp"] for k in cells]
        slot = [pc[k]["mean_cos_to_U1_end_slot_re"] for k in cells]
        seeds = [pc[k]["seed"] for k in cells]
        ax.scatter(
            range(len(cells)), slot, marker="o", color=color, label=f"{ARM_LABEL[arm]}: end slot"
        )
        ax.scatter(
            range(len(cells)),
            mr,
            marker="s",
            color=color,
            alpha=0.55,
            label=f"{ARM_LABEL[arm]}: whole response",
        )
        for i, s in enumerate(seeds):
            ax.annotate(
                f"seed {s}", (i, mr[i]), fontsize=6, xytext=(3, 3), textcoords="offset points"
            )
    ax.set_xticks([])
    ax.set_ylabel("mean cosine to top direction")
    ax.set_title("Whole-response vs end-slot read")
    ax.legend(fontsize=6.5)

    # (right) marker ||shift|| vs cos-to-U1 scatter, trained persona marked.
    # Seeds are shape-coded in the marker-arm color so that color keeps the
    # same meaning in every panel (blue = EM, orange = marker).
    ax = axes[2]
    na = norm_align["per_cell"]
    mk_cells = sorted((k for k in na if na[k]["arm"] == "marker"), key=lambda k: na[k]["seed"])
    for k in mk_cells:
        seed = na[k]["seed"]
        personas = list(na[k]["norms"].keys())
        others = [p for p in personas if p != SOURCE_PERSONA]
        ax.scatter(
            [na[k]["norms"][p] for p in others],
            [na[k]["cos_to_U1"][p] for p in others],
            s=22,
            color=mk_color,
            marker=seed_markers[seed],
            label=f"seed {seed}",
            alpha=0.75,
        )
        ax.scatter(
            [na[k]["norms"][SOURCE_PERSONA]],
            [na[k]["cos_to_U1"][SOURCE_PERSONA]],
            s=80,
            color=mk_color,
            marker="X",
            edgecolors="black",
            linewidths=0.9,
            zorder=3,
        )
    src_handle = mlines.Line2D(
        [],
        [],
        color="white",
        marker="X",
        markersize=8,
        markeredgecolor="black",
        markerfacecolor=mk_color,
        linestyle="None",
        label="medical doctor (trained persona)",
    )
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, src_handle], fontsize=7)
    ax.set_xlabel("per-persona shift norm")
    ax.set_ylabel("cosine to top direction")
    ax.set_title("Marker arm: shift size vs alignment")

    fig.tight_layout()
    savefig_paper(fig, "hero_three_controls", dir=FIGURES_DIR)
    plt.close(fig)

    # ── Figure 2: per-variant LOO panels ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    for ax, variant in zip(axes, ("same", "base", "on_policy"), strict=True):
        sub = {k: v for k, v in loo["per_cell"].items() if v["variant"] == variant}
        names = sorted(sub, key=lambda k: (sub[k]["arm"], sub[k]["seed"]))
        xs = np.arange(len(names))
        ax.bar(
            xs,
            [sub[k]["s_top1_frac_loo"] for k in names],
            width=0.6,
            color=[em_color if sub[k]["arm"] == "em" else mk_color for k in names],
        )
        for i, k in enumerate(names):
            ax.hlines(sub[k]["sign_flip"]["p95"], i - 0.4, i + 0.4, color="black", lw=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [
                f"{'EM' if sub[k]['arm'] == 'em' else 'marker'}\nseed {sub[k]['seed']}"
                for k in names
            ],
            fontsize=7,
        )
        ax.set_title(f"read on {VARIANT_LABEL[variant]}", fontsize=10)
    em_handle = mlines.Line2D(
        [], [], color=em_color, marker="s", linestyle="None", label=ARM_LABEL["em"]
    )
    mk_handle = mlines.Line2D(
        [], [], color=mk_color, marker="s", linestyle="None", label=ARM_LABEL["marker"]
    )
    null_handle = mlines.Line2D([], [], color="black", lw=1.2, label="sign-flip null (95th pct)")
    axes[0].legend(handles=[em_handle, mk_handle, null_handle], fontsize=7)
    axes[0].set_ylabel("source-dropped top-direction share")
    fig.suptitle("Source-dropped spectrum, by which text the models are read on")
    fig.tight_layout()
    savefig_paper(fig, "loo_by_variant", dir=FIGURES_DIR)
    plt.close(fig)

    # ── Figure 3: jackknife strip (trained-model-text cells only) ────
    same_jack = {k: v for k, v in jack["per_cell"].items() if v["variant"] == "same"}
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    names = sorted(same_jack, key=lambda k: (same_jack[k]["arm"], same_jack[k]["seed"]))
    for i, k in enumerate(names):
        v = same_jack[k]
        drops = v["s_top1_frac_drop"]
        other_ys = [y for p, y in drops.items() if p != SOURCE_PERSONA]
        ax.scatter([i] * len(other_ys), other_ys, s=14, alpha=0.6, color=mk_color)
        src_y = drops.get(SOURCE_PERSONA)
        if src_y is not None:
            ax.scatter([i], [src_y], s=46, color="black", zorder=3, marker="D")
        ax.scatter([i], [v["s_top1_frac_full"]], s=60, color=em_color, zorder=3, marker="_")
    dot_h = mlines.Line2D(
        [],
        [],
        color=mk_color,
        marker="o",
        linestyle="None",
        markersize=5,
        label="one bystander persona dropped",
    )
    diam_h = mlines.Line2D(
        [],
        [],
        color="black",
        marker="D",
        linestyle="None",
        markersize=6,
        label="medical doctor (trained persona) dropped",
    )
    full_h = mlines.Line2D(
        [], [], color=em_color, marker="_", linestyle="None", markersize=10, label="full panel"
    )
    ax.legend(handles=[dot_h, diam_h, full_h], fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{ARM_LABEL[same_jack[k]['arm']]}\nseed {same_jack[k]['seed']}" for k in names],
        fontsize=7,
    )
    ax.set_ylabel("top-direction share after dropping one persona")
    ax.set_title("Fourteen-fold jackknife (trained-model-text cells only)")
    fig.tight_layout()
    savefig_paper(fig, "jackknife_strip", dir=FIGURES_DIR)
    plt.close(fig)

    # ── Figure 4: split-half reliability vs cosine, trained persona marked ──
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    rel_cells = sorted(
        (kv for kv in reliability["per_cell"].items() if kv[1]["arm"] == "marker"),
        key=lambda kv: kv[1]["seed"],
    )
    for k, v in rel_cells:
        seed = v["seed"]
        cell_cos = norm_align["per_cell"][k]["cos_to_U1"]
        rel_map = v["split_half_cosine_per_persona"]
        others = [
            (rel_map[p], cell_cos[p])
            for p in rel_map
            if rel_map[p] is not None and p != SOURCE_PERSONA
        ]
        ax.scatter(
            [x for x, _ in others],
            [y for _, y in others],
            s=24,
            alpha=0.75,
            color=mk_color,
            marker=seed_markers[seed],
            label=f"seed {seed}",
        )
        if rel_map.get(SOURCE_PERSONA) is not None:
            ax.scatter(
                [rel_map[SOURCE_PERSONA]],
                [cell_cos[SOURCE_PERSONA]],
                s=80,
                color=mk_color,
                marker="X",
                edgecolors="black",
                linewidths=0.9,
                zorder=3,
            )
    # The one genuinely low-reliability low-alignment point (seed 256).
    mb_rel = reliability["per_cell"]["same_marker_seed256"]["split_half_cosine_per_persona"][
        "marine_biologist"
    ]
    mb_cos = norm_align["per_cell"]["same_marker_seed256"]["cos_to_U1"]["marine_biologist"]
    ax.annotate(
        "marine biologist (seed 256)",
        (mb_rel, mb_cos),
        fontsize=7,
        xytext=(8, 8),
        textcoords="offset points",
    )
    src_handle = mlines.Line2D(
        [],
        [],
        color="white",
        marker="X",
        markersize=8,
        markeredgecolor="black",
        markerfacecolor=mk_color,
        linestyle="None",
        label="medical doctor (trained persona)",
    )
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, src_handle], fontsize=8)
    ax.set_xlabel("split-half reliability (cosine between question-half means)")
    ax.set_ylabel("cosine to top direction")
    ax.set_title("Marker arm: estimation reliability vs alignment")
    fig.tight_layout()
    savefig_paper(fig, "reliability_vs_cosine", dir=FIGURES_DIR)
    plt.close(fig)

    # ── Figure 5: reproduction deltas, plain labels + non-negative axis ──
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    pc = repro["per_cell"]
    names = sorted(pc, key=lambda k: (pc[k]["variant"], pc[k]["arm"], pc[k]["seed"]))
    ys = np.arange(len(names))
    # Open circles for |Δ top-share| so they stay visible when both series
    # sit at exactly the same value (here: exactly 0 in every cell).
    ax.scatter(
        [pc[k]["clauses"]["d_s_top1_frac"] for k in names],
        ys,
        label="|Δ top-share|",
        s=90,
        facecolors="none",
        edgecolors=em_color,
        linewidths=1.4,
    )
    ax.scatter(
        [pc[k]["clauses"]["d_mean_cos_to_U1"] for k in names],
        ys,
        label="|Δ mean cosine|",
        s=22,
        marker="s",
        color=mk_color,
    )
    ax.axvline(REPRO_TOL, color="black", lw=1.0, label="tolerance (0.05)")
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [cell_label(pc[k]["arm"], pc[k]["seed"], pc[k]["variant"]) for k in names], fontsize=7
    )
    ax.set_xlim(-0.002, REPRO_TOL * 1.25)
    ax.set_xticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_xlabel("absolute delta vs parent run")
    ax.set_title("Reproduction gate: re-extracted vs parent per-cell summaries")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "reproduction_deltas", dir=FIGURES_DIR)
    plt.close(fig)

    logger.info("[figures] regenerated under %s", FIGURES_DIR)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    norm_align = _load("norm_alignment.json")

    payload = compute_split_robustness(norm_align)
    out_path = CONTROLS_DIR / "split_robustness.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=1)
    logger.info("[split_robustness] written to %s", out_path)

    with REPRO_PATH.open() as f:
        repro = json.load(f)
    make_figures(
        repro=repro,
        loo=_load("loo.json"),
        jack=_load("jackknife.json"),
        mean_resp=_load("mean_resp.json"),
        norm_align=norm_align,
        reliability=_load("reliability.json"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
