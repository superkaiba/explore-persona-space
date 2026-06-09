#!/usr/bin/env python3
# ruff: noqa: RUF001
# (greek + arrow + multiplication/minus-sign characters intentional in docstrings/labels)
"""Build clean-result figures for task #541 (prior-stratified rerun of #500).

Hero figures (plan §6):
  1. prior_vs_leak_stratified  — per-arm scatter of leak vs prior, strata
     shading, nested-15 vs new personas marked distinctly, drop-top-stratum
     rho + residual n annotated (P1).
  2. gating_vs_source_prior    — common-set panel-median leak vs measured
     source prior, per-seed points + arm medians with seed ranges (P2).
  3. engagement_adjusted_partials — unadjusted vs PRE-TREATMENT-adjusted
     rho(prior, leak) per arm with bootstrap CIs; trained-covariate bars
     alongside, visually de-emphasized + labeled post-treatment (P3).

Exploratory (over-produce, raw-alongside-processed rule):
  4. prior_vs_base_on_topic    — the collinearity diagnostic itself.
  5. raw_vs_adjusted_leak      — raw vs trained-minus-base-adjusted DV per arm.
  6. mechanism_tag_leak        — per-mechanism-tag (entity/domain/surface) leak.

Reads:  eval_results/issue_541/{predictors.json, phase0_prescreen/prior_screen.json,
        base_engagement_covariates.json}
Writes: figures/issue_541/<name>.{png,pdf,meta.json}
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT_NAME = "issue_541_smoke" if os.environ.get("EPM_541_SMOKE") == "1" else "issue_541"
EVAL_ROOT = ROOT / "eval_results" / EVAL_ROOT_NAME
FIG_DIR = ROOT / "figures"
STRATUM_H_MIN = -3.25
STRATUM_M_MIN = -3.40
ORIGINAL_15_MARKER = "o"
NEW_PERSONA_MARKER = "^"


def _load() -> tuple[dict, dict]:
    pred = json.loads((EVAL_ROOT / "predictors.json").read_text())
    screen = json.loads((EVAL_ROOT / "phase0_prescreen" / "prior_screen.json").read_text())
    return pred, screen


def _arm_label(source: str, priors: dict[str, float]) -> str:
    nice = source.replace("_", " ")
    prior = priors.get(source)
    return f"{nice}\n(prior {prior:+.2f})" if prior is not None else nice


def plot_prior_vs_leak_stratified(pred: dict, screen: dict) -> None:
    import matplotlib.pyplot as plt

    sources = pred["sources"]
    priors = pred["logprob_priors_used"]
    nested = set(screen["selection"]["nested_originals"])
    n = len(sources)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, source in zip(axes, sources, strict=True):
        arm = pred["per_arm"].get(source, {})
        add = pred["per_arm_additions"].get(source, {})
        pp = arm.get("per_persona", {})
        if not pp:
            ax.set_title(f"{source}\n(missing)")
            continue
        # Strata shading.
        ax.axvspan(STRATUM_H_MIN, 0, color=paper_palette_role("accent"), alpha=0.08)
        ax.axvspan(STRATUM_M_MIN, STRATUM_H_MIN, color="grey", alpha=0.06)
        for persona, d in pp.items():
            x, y = d["prior_logprob"], d["leak_mean"]
            if math.isnan(x) or math.isnan(y):
                continue
            marker = ORIGINAL_15_MARKER if persona in nested else NEW_PERSONA_MARKER
            ax.scatter(
                x,
                y,
                marker=marker,
                s=34,
                color=paper_palette_role("primary"),
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        drop = add.get("drop_tables", {}).get("drop_top_stratum", {})
        full = add.get("drop_tables", {}).get("full_panel", {})
        ax.set_title(
            f"{_arm_label(source, priors)}\n"
            f"ρ={full.get('rho', float('nan')):+.2f} (n={full.get('n')}); "
            f"drop-H ρ={drop.get('rho', float('nan')):+.2f} (n={drop.get('residual_n')})",
            fontsize=8,
        )
        ax.set_xlabel("bystander prior (log P / token)")
        finite = [priors[p] for p in pp if p in priors and not math.isnan(priors[p])]
        if finite:
            ax.set_xlim(min(finite) - 0.04, max(-3.0, STRATUM_H_MIN) + 0.12)
    axes[0].set_ylabel("headline leak rate")
    fig.suptitle(
        "Leak vs bystander prior, prior-stratified panel "
        "(circles = nested 15, triangles = new; shaded = H stratum)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, f"{EVAL_ROOT_NAME}/prior_vs_leak_stratified", dir=str(FIG_DIR))
    plt.close(fig)


def plot_gating_vs_source_prior(pred: dict) -> None:
    import matplotlib.pyplot as plt

    p2 = pred.get("p2_source_prior_gating", {})
    per_arm = p2.get("per_arm", {})
    if not per_arm:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for source, d in per_arm.items():
        x = d.get("source_prior")
        if x is None:
            continue
        for m in d.get("per_seed_medians", []):
            ax.scatter(x, m, s=20, color="grey", alpha=0.6, zorder=2)
        point = d.get("median_leak_common_set")
        if point is not None and not (isinstance(point, float) and math.isnan(point)):
            ax.scatter(x, point, s=70, color=paper_palette_role("primary"), zorder=3, label=None)
            rng = d.get("seed_range")
            if rng:
                ax.vlines(x, rng[0], rng[1], color=paper_palette_role("primary"), linewidth=1.4)
            ax.annotate(
                source.replace("_", " "),
                (x, point),
                textcoords="offset points",
                xytext=(4, 5),
                fontsize=7,
            )
    perm = p2.get("permutation", {})
    sub = (
        f"perm p={perm.get('one_sided_p'):.3f} "
        f"(perfect monotone: {perm.get('perfect_monotone_decreasing')})"
        if perm
        else "permutation not computable"
    )
    ax.set_xlabel("source persona measured prior (log P / token)")
    ax.set_ylabel("common-set panel-median leak")
    ax.set_title(f"Gating tightness vs source prior\n{sub}", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, f"{EVAL_ROOT_NAME}/gating_vs_source_prior", dir=str(FIG_DIR))
    plt.close(fig)


def plot_engagement_adjusted_partials(pred: dict) -> None:
    import matplotlib.pyplot as plt

    sources = pred["sources"]
    rows: list[tuple[str, float, float, tuple[float, float] | None, float | None]] = []
    for source in sources:
        add = pred["per_arm_additions"].get(source, {})
        eng = add.get("primary_engagement", {})
        if eng.get("status") != "computed":
            continue
        boot = eng.get("partial_bootstrap", {})
        ci = (
            (boot.get("ci_low_95"), boot.get("ci_high_95"))
            if boot.get("ci_low_95") is not None
            else None
        )
        trained = (
            pred["per_arm"]
            .get(source, {})
            .get("stats", {})
            .get("engagement_adjusted", {})
            .get("partial_spearman_prior_vs_leak_given_length_and_on_topic")
        )
        rows.append(
            (
                source,
                eng.get("unadjusted_rho_prior_leak", float("nan")),
                eng.get("partial_rho_prior_leak_given_base_len_and_on_topic", float("nan")),
                ci,
                trained,
            )
        )
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(1.0 + 2.0 * len(rows), 3.6))
    width = 0.27
    xs = np.arange(len(rows))
    ax.bar(
        xs - width,
        [r[1] for r in rows],
        width,
        label="unadjusted ρ(prior, leak)",
        color=paper_palette_role("primary"),
    )
    ax.bar(
        xs,
        [r[2] for r in rows],
        width,
        label="PRE-treatment adjusted (PRIMARY)",
        color=paper_palette_role("accent"),
    )
    for i, r in enumerate(rows):
        if r[3] and r[3][0] is not None:
            ax.vlines(i, r[3][0], r[3][1], color="black", linewidth=1.0)
    trained_vals = [r[4] if r[4] is not None else float("nan") for r in rows]
    ax.bar(
        xs + width,
        trained_vals,
        width,
        label="post-treatment adjusted (texture only)",
        color="lightgrey",
        edgecolor="grey",
        hatch="//",
        alpha=0.7,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0].replace("_", "\n") for r in rows], fontsize=7)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Prior→leak: unadjusted vs engagement-adjusted partials", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, f"{EVAL_ROOT_NAME}/engagement_adjusted_partials", dir=str(FIG_DIR))
    plt.close(fig)


def plot_prior_vs_base_on_topic(pred: dict) -> None:
    import matplotlib.pyplot as plt

    cov_path = EVAL_ROOT / "base_engagement_covariates.json"
    if not cov_path.exists():
        return
    cov = json.loads(cov_path.read_text()).get("per_persona", {})
    priors = pred["logprob_priors_used"]
    gate = pred.get("collinearity_gate", {})
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for persona, d in cov.items():
        if persona not in priors:
            continue
        y = d.get("base_on_topic_fraction")
        if y is None or (isinstance(y, float) and math.isnan(y)):
            continue
        ax.scatter(priors[persona], y, s=30, color=paper_palette_role("primary"))
        ax.annotate(
            persona.replace("_", " "),
            (priors[persona], y),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=5.5,
        )
    r = gate.get("pearson_prior_vs_base_on_topic")
    ax.set_xlabel("measured prior (log P / token)")
    ax.set_ylabel("base on-topic fraction (pre-treatment)")
    ax.set_title(
        f"Collinearity diagnostic: Pearson r={r:+.2f}"
        if r is not None
        else "Collinearity diagnostic",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, f"{EVAL_ROOT_NAME}/prior_vs_base_on_topic", dir=str(FIG_DIR))
    plt.close(fig)


def plot_raw_vs_adjusted(pred: dict) -> None:
    import matplotlib.pyplot as plt

    sources = pred["sources"]
    fig, axes = plt.subplots(1, len(sources), figsize=(3.6 * len(sources), 3.4), sharey=True)
    if len(sources) == 1:
        axes = [axes]
    for ax, source in zip(axes, sources, strict=True):
        add = pred["per_arm_additions"].get(source, {})
        adj = add.get("adjusted_dv", {})
        pp = pred["per_arm"].get(source, {}).get("per_persona", {})
        per_adj = adj.get("per_persona_adjusted", {})
        for persona, d in pp.items():
            if persona not in per_adj:
                continue
            ax.scatter(d["leak_mean"], per_adj[persona], s=22, color=paper_palette_role("primary"))
        ax.plot([0, 1], [0, 1], color="grey", linewidth=0.6, linestyle="--")
        ax.set_title(
            f"{source.replace('_', ' ')}\n"
            f"raw ρ={adj.get('rho_prior_vs_leak_raw', float('nan')):+.2f} "
            f"adj ρ={adj.get('rho_prior_vs_leak_adjusted', float('nan')):+.2f}",
            fontsize=8,
        )
        ax.set_xlabel("raw leak")
    axes[0].set_ylabel("trained − base adjusted leak")
    fig.tight_layout()
    savefig_paper(fig, f"{EVAL_ROOT_NAME}/raw_vs_adjusted_leak", dir=str(FIG_DIR))
    plt.close(fig)


def plot_mechanism_tag_leak(pred: dict, screen: dict) -> None:
    import matplotlib.pyplot as plt

    mech = screen.get("mechanism", {})
    sources = pred["sources"]
    leak_by_tag: dict[str, list[float]] = {}
    for source in sources:
        pp = pred["per_arm"].get(source, {}).get("per_persona", {})
        for persona, d in pp.items():
            tag = mech.get(persona, "original")
            if not math.isnan(d["leak_mean"]):
                leak_by_tag.setdefault(tag, []).append(d["leak_mean"])
    if not leak_by_tag:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    tags = sorted(leak_by_tag)
    ax.boxplot([leak_by_tag[t] for t in tags], tick_labels=tags)
    ax.set_ylabel("headline leak rate (pooled across arms)")
    ax.set_title("Leak by prior-raising mechanism class (exploratory)", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, f"{EVAL_ROOT_NAME}/mechanism_tag_leak", dir=str(FIG_DIR))
    plt.close(fig)


def main() -> None:
    set_paper_style()
    pred, screen = _load()
    plot_prior_vs_leak_stratified(pred, screen)
    plot_gating_vs_source_prior(pred)
    plot_engagement_adjusted_partials(pred)
    plot_prior_vs_base_on_topic(pred)
    plot_raw_vs_adjusted(pred)
    plot_mechanism_tag_leak(pred, screen)
    print(f"WROTE figures -> {FIG_DIR / EVAL_ROOT_NAME}")


if __name__ == "__main__":
    main()
