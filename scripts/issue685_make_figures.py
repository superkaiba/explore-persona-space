#!/usr/bin/env python
"""Issue #685 Phase D — figures (hero + supporting) from metrics.json + validity_judged.json.

CPU. Reads ``eval_results/issue_685[_smoke]/{metrics.json, validity_judged.json}``
and produces the hero figure (per-behavior layer-sweep of the consistency cosine
with the null band + H1/H2 threshold lines) plus supporting figures into
``figures/issue_685[_smoke]/`` via the paper-plots rcParams.

Hero:    per-behavior layer-sweep consistency-cosine panel (null band + H1/H2 bands)
(a)      ||Delta|| / between-context spread heatmap (behavior x layer)  [H0 read]
(b)      PC1 variance-share line plot (behavior x layer)
(c)      behavior-cosine 6x6 matrix at the best layer
(d)      projection-onto-known-direction (behavior x layer)             [if present]
(e)      behavioral-validity judge bar (rate C vs C+b per behavior)
(f)      base-vs-instruct overlay of the hero consistency curve         [if base present]

Usage::

    uv run python scripts/issue685_make_figures.py                  # full
    uv run python scripts/issue685_make_figures.py --smoke           # tiny (hero + 1 supporting)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

# H1/H2/H0 threshold bands (plan §1 / Success criteria).
H1_COS = 0.6
H2_COS_HI = 0.4


def _mean_syco_delta(validity: dict | None) -> float | None:
    """Mean sycophancy rate-delta over contexts from a validity_judged.json read.

    The parent neutral-bank reference line for the opinion-bank figure. None when
    the validity read is absent or has no sycophancy cells with a delta.
    """
    if validity is None:
        return None
    deltas = [
        v["rate_delta"]
        for v in validity["per_context_behavior"].values()
        if v["behavior"] == "sycophancy" and v.get("rate_delta") is not None
    ]
    return float(np.mean(deltas)) if deltas else None


def _consistency_curve(model_metrics: dict, behavior: str, layers: list[int]) -> list[float]:
    return [model_metrics["cells"][behavior][str(L)]["consistency_cosine_raw"] for L in layers]


def _null_p95(model_metrics: dict, layers: list[int]) -> list[float]:
    return [model_metrics["consistency_null"][str(L)]["p95"] for L in layers]


def fig_hero(metrics: dict, out_dir: Path, tag: str = "instruct") -> Path:
    """Per-behavior layer-sweep of the consistency cosine, null band + H1/H2 lines."""
    m = metrics["models"][tag]
    behaviors = m["meta"]["behaviors"]
    layers = m["meta"]["layers"]
    null_p95 = _null_p95(m, layers)

    n = len(behaviors)
    ncol = min(3, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    primary = paper_palette_role("primary")
    for i, b in enumerate(behaviors):
        ax = axes[i // ncol][i % ncol]
        cos = _consistency_curve(m, b, layers)
        ax.plot(layers, cos, marker="o", color=primary, label="consistency")
        ax.plot(layers, null_p95, ls=":", color=paper_palette_role("neutral"), label="null p95")
        ax.axhline(H1_COS, ls="--", color=paper_palette_role("accent"), lw=0.8)
        ax.axhspan(0.0, H2_COS_HI, color=paper_palette_role("control"), alpha=0.12)
        ax.set_title(b)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean pairwise cos")
        ax.set_ylim(-0.2, 1.0)
        ax.set_xticks(layers)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=6, loc="best")
    fig.suptitle(f"Direction consistency of Delta(C,b) by layer ({tag})", y=1.0)
    paths = savefig_paper(fig, "hero_consistency_panel", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def fig_relmag_heatmap(metrics: dict, out_dir: Path, tag: str = "instruct") -> Path:
    """||Delta|| / between-context spread (mean over contexts), behavior x layer."""
    m = metrics["models"][tag]
    behaviors = m["meta"]["behaviors"]
    layers = m["meta"]["layers"]
    grid = np.array(
        [[m["cells"][b][str(L)]["relative_magnitude"]["mean"] for L in layers] for b in behaviors]
    )
    fig, ax = plt.subplots(figsize=(0.9 * len(layers) + 2, 0.45 * len(behaviors) + 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(behaviors)))
    ax.set_yticklabels(behaviors)
    ax.set_xlabel("layer")
    ax.set_title(f"||Delta|| / between-context spread ({tag})")
    for i in range(len(behaviors)):
        for j in range(len(layers)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    paths = savefig_paper(fig, "relmag_heatmap", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def fig_pc1_lines(metrics: dict, out_dir: Path, tag: str = "instruct") -> Path:
    """PC1 variance share per behavior across layers."""
    m = metrics["models"][tag]
    behaviors = m["meta"]["behaviors"]
    layers = m["meta"]["layers"]
    colors = paper_palette(len(behaviors))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for b, col in zip(behaviors, colors, strict=True):
        pc1 = [m["cells"][b][str(L)]["pc1_variance_share"] for L in layers]
        ax.plot(layers, pc1, marker="o", color=col, label=b)
    ax.axhline(0.5, ls="--", color=paper_palette_role("accent"), lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("PC1 variance share")
    ax.set_title(f"PC1 share of the Delta matrix ({tag})")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6)
    paths = savefig_paper(fig, "pc1_variance_share", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def fig_behavior_matrix(metrics: dict, out_dir: Path, tag: str = "instruct") -> Path:
    """6x6 behavior-separability cosine at the best (highest-mean-consistency) layer."""
    m = metrics["models"][tag]
    layers = m["meta"]["layers"]
    behaviors = m["meta"]["behaviors"]
    # Best layer = the one with the highest mean consistency cosine across behaviors.
    best_layer = max(
        layers,
        key=lambda L: float(
            np.mean([m["cells"][b][str(L)]["consistency_cosine_raw"] for b in behaviors])
        ),
    )
    sep = m["behavior_separability"][str(best_layer)]
    names, mat = sep["names"], np.array(sep["matrix"])
    fig, ax = plt.subplots(figsize=(0.6 * len(names) + 2, 0.6 * len(names) + 2))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(f"Behavior-shift cosine (layer {best_layer}, {tag})")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    paths = savefig_paper(fig, "behavior_cosine_matrix", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def fig_projection(metrics: dict, out_dir: Path, tag: str = "instruct") -> Path | None:
    """Mean projection fraction |Delta . u_hat|/||Delta|| (behavior x layer)."""
    m = metrics["models"][tag]
    if not m.get("has_known_direction_projection"):
        return None
    behaviors = m["meta"]["behaviors"]
    layers = m["meta"]["layers"]
    rows = []
    present_behaviors = []
    for b in behaviors:
        if all("proj_on_known_direction" in m["cells"][b][str(L)] for L in layers):
            rows.append([m["cells"][b][str(L)]["proj_on_known_direction"]["mean"] for L in layers])
            present_behaviors.append(b)
    if not rows:
        return None
    grid = np.array(rows)
    fig, ax = plt.subplots(figsize=(0.9 * len(layers) + 2, 0.45 * len(present_behaviors) + 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(present_behaviors)))
    ax.set_yticklabels(present_behaviors)
    ax.set_xlabel("layer")
    ax.set_title(f"|Delta . u_hat| / ||Delta|| ({tag})")
    for i in range(len(present_behaviors)):
        for j in range(len(layers)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    paths = savefig_paper(fig, "projection_known_direction", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def fig_judge_bar(validity: dict, out_dir: Path) -> Path | None:
    """Judge-positive rate C vs C+b per behavior (mean over subset contexts)."""
    if validity is None:
        return None
    behaviors = validity["behaviors"]
    pcb = validity["per_context_behavior"]
    # Mean over contexts of rate_C and rate_Cb per behavior (skip None).
    rate_c, rate_cb = [], []
    for b in behaviors:
        cs = [
            v["rate_C"] for v in pcb.values() if v["behavior"] == b and v.get("rate_C") is not None
        ]
        cbs = [
            v["rate_Cb"]
            for v in pcb.values()
            if v["behavior"] == b and v.get("rate_Cb") is not None
        ]
        rate_c.append(float(np.mean(cs)) if cs else 0.0)
        rate_cb.append(float(np.mean(cbs)) if cbs else 0.0)
    x = np.arange(len(behaviors))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.0 * len(behaviors) + 2, 3.2))
    ax.bar(x - w / 2, rate_c, w, label="C (bare)", color=paper_palette_role("baseline"))
    ax.bar(x + w / 2, rate_cb, w, label="C+b (augmented)", color=paper_palette_role("primary"))
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=45, ha="right")
    ax.set_ylabel("judge-positive rate")
    ax.set_ylim(0, 1)
    ax.set_title("Behavioral-validity rate (kill-criterion gate)")
    ax.legend(fontsize=7)
    paths = savefig_paper(fig, "validity_judge_bar", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def fig_syco_opinion_bar(
    syco: dict, out_dir: Path, neutral_ref: float | None = None
) -> Path | None:
    """Per-context sycophancy rate C vs C+b on the opinion bank (v3 follow-up).

    Draws the parent's neutral-bank sycophancy rate-delta as a reference line so
    the artifact-vs-real-effect call is visible at a glance.
    """
    if syco is None:
        return None
    pcb = syco["per_context_behavior"]
    # Per-context sycophancy rate_C / rate_Cb (one behavior: sycophancy).
    contexts = syco["contexts"]
    rate_c, rate_cb = [], []
    for c in contexts:
        v = pcb.get(f"{c}__sycophancy", {})
        rate_c.append(v.get("rate_C") if v.get("rate_C") is not None else 0.0)
        rate_cb.append(v.get("rate_Cb") if v.get("rate_Cb") is not None else 0.0)
    x = np.arange(len(contexts))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.0 * len(contexts) + 2, 3.4))
    ax.bar(x - w / 2, rate_c, w, label="C (bare)", color=paper_palette_role("baseline"))
    ax.bar(x + w / 2, rate_cb, w, label="C+sycophancy", color=paper_palette_role("primary"))
    if neutral_ref is not None:
        ax.axhline(
            neutral_ref,
            ls="--",
            color=paper_palette_role("accent"),
            lw=0.9,
            label=f"parent neutral-bank rate-delta (+{neutral_ref:.2f})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(contexts, rotation=45, ha="right")
    ax.set_ylabel("judge-positive sycophancy rate")
    ax.set_ylim(0, 1)
    ax.set_title("Sycophancy on the opinion-bearing bank (per context)")
    ax.legend(fontsize=7)
    paths = savefig_paper(fig, "validity_syco_opinion_bar", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def fig_syco_neutral_vs_opinion(syco: dict, validity: dict, out_dir: Path) -> Path | None:
    """Diagnosis bar: per-context sycophancy rate-delta, neutral bank vs opinion bank."""
    if syco is None or validity is None:
        return None
    syco_pcb = syco["per_context_behavior"]
    neutral_pcb = validity["per_context_behavior"]
    # Use the contexts shared by both reads (opinion bank = all 10; neutral may
    # be 10 after Arm A, or 4 if read pre-merge).
    contexts = [c for c in syco["contexts"] if f"{c}__sycophancy" in neutral_pcb]
    if not contexts:
        return None
    neutral_d, opinion_d = [], []
    for c in contexts:
        nd = neutral_pcb.get(f"{c}__sycophancy", {}).get("rate_delta")
        od = syco_pcb.get(f"{c}__sycophancy", {}).get("rate_delta")
        neutral_d.append(nd if nd is not None else 0.0)
        opinion_d.append(od if od is not None else 0.0)
    x = np.arange(len(contexts))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.0 * len(contexts) + 2, 3.4))
    ax.bar(x - w / 2, neutral_d, w, label="neutral bank", color=paper_palette_role("baseline"))
    ax.bar(x + w / 2, opinion_d, w, label="opinion bank", color=paper_palette_role("primary"))
    ax.axhline(0.15, ls="--", color=paper_palette_role("accent"), lw=0.9, label="+0.15 floor")
    ax.set_xticks(x)
    ax.set_xticklabels(contexts, rotation=45, ha="right")
    ax.set_ylabel("sycophancy rate-delta (C+b minus C)")
    ax.set_title("Sycophancy rate-delta: neutral vs opinion question bank")
    ax.legend(fontsize=7)
    paths = savefig_paper(fig, "validity_syco_neutral_vs_opinion", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def fig_base_vs_instruct(metrics: dict, out_dir: Path) -> Path | None:
    """Overlay the mean-over-behaviors consistency curve for instruct vs base."""
    if "base" not in metrics["models"]:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for tag, role in (("instruct", "primary"), ("base", "baseline")):
        m = metrics["models"][tag]
        layers = m["meta"]["layers"]
        behaviors = m["meta"]["behaviors"]
        mean_cos = [
            float(np.mean([m["cells"][b][str(L)]["consistency_cosine_raw"] for b in behaviors]))
            for L in layers
        ]
        ax.plot(layers, mean_cos, marker="o", color=paper_palette_role(role), label=tag)
    ax.axhline(H1_COS, ls="--", color=paper_palette_role("accent"), lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean consistency cos (over behaviors)")
    ax.set_title("Consistency: instruct vs base")
    ax.set_xticks(metrics["models"]["instruct"]["meta"]["layers"])
    ax.set_ylim(-0.2, 1.0)
    ax.legend(fontsize=7)
    paths = savefig_paper(fig, "base_vs_instruct_consistency", dir=str(out_dir))
    plt.close(fig)
    return paths["png"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #685 Phase D — figures.")
    parser.add_argument("--smoke", action="store_true", help="tiny verification slice.")
    parser.add_argument("--eval-dir", default=None, help="override the eval_results in dir.")
    parser.add_argument("--fig-dir", default=None, help="override the figures out dir.")
    parser.add_argument(
        "--validity-only",
        action="store_true",
        help="v3 follow-up: re-render ONLY the validity bars (judge bar + syco-opinion "
        "bars); leave the frozen geometry figures untouched.",
    )
    args = parser.parse_args()

    smoke = args.smoke
    eval_dir = (
        Path(args.eval_dir)
        if args.eval_dir
        else Path("eval_results/issue685_smoke" if smoke else "eval_results/issue_685")
    )
    fig_dir = (
        Path(args.fig_dir)
        if args.fig_dir
        else Path("figures/issue685_smoke" if smoke else "figures/issue_685")
    )
    fig_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style("blog")

    validity_path = eval_dir / "validity_judged.json"
    validity = json.loads(validity_path.read_text()) if validity_path.exists() else None
    syco_path = eval_dir / "validity_judged_syco_opinion.json"
    syco = json.loads(syco_path.read_text()) if syco_path.exists() else None

    produced: list[str] = []

    # --validity-only (v3 follow-up): re-render ONLY the validity bars + the
    # syco-opinion diagnosis figures; the frozen geometry figures are NOT touched.
    if args.validity_only:
        p_judge = fig_judge_bar(validity, fig_dir)
        if p_judge:
            produced.append(str(p_judge))
        # Parent neutral-bank sycophancy rate-delta (mean over contexts) — the
        # reference line for the opinion-bank read.
        neutral_ref = _mean_syco_delta(validity)
        p_syco = fig_syco_opinion_bar(syco, fig_dir, neutral_ref=neutral_ref)
        if p_syco:
            produced.append(str(p_syco))
        p_diag = fig_syco_neutral_vs_opinion(syco, validity, fig_dir)
        if p_diag:
            produced.append(str(p_diag))
        print(f"[issue685.D] (validity-only) wrote {len(produced)} figures -> {fig_dir}")
        for p in produced:
            print(f"  {p}")
        return

    metrics = json.loads((eval_dir / "metrics.json").read_text())
    produced.append(str(fig_hero(metrics, fig_dir)))
    if not smoke:
        produced.append(str(fig_relmag_heatmap(metrics, fig_dir)))
        produced.append(str(fig_pc1_lines(metrics, fig_dir)))
        produced.append(str(fig_behavior_matrix(metrics, fig_dir)))
        p_proj = fig_projection(metrics, fig_dir)
        if p_proj:
            produced.append(str(p_proj))
        p_judge = fig_judge_bar(validity, fig_dir)
        if p_judge:
            produced.append(str(p_judge))
        p_bvi = fig_base_vs_instruct(metrics, fig_dir)
        if p_bvi:
            produced.append(str(p_bvi))
        # Syco-opinion bars if the follow-up read is present.
        neutral_ref = _mean_syco_delta(validity)
        p_syco = fig_syco_opinion_bar(syco, fig_dir, neutral_ref=neutral_ref)
        if p_syco:
            produced.append(str(p_syco))
        p_diag = fig_syco_neutral_vs_opinion(syco, validity, fig_dir)
        if p_diag:
            produced.append(str(p_diag))
    else:
        # Smoke: hero + at least one supporting figure (relmag heatmap) to
        # confirm the matplotlib path end-to-end.
        produced.append(str(fig_relmag_heatmap(metrics, fig_dir)))

    print(f"[issue685.D] wrote {len(produced)} figures -> {fig_dir}")
    for p in produced:
        print(f"  {p}")


if __name__ == "__main__":
    main()
