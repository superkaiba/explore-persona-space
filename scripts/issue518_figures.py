"""Issue #518 clean-result figures.

Generates the six figures embedded in the clean-result body:

1. ``hero_cross_arm`` — grouped bar chart of source-FE Spearman ρ across six
   coarse predictors on the three behaviour arms. Both +0.40 and -0.40
   threshold lines visible; y-range expanded so the +0.438 cosine-L20 EM bar
   no longer clips.
2. ``per_arm_coarse`` — three-panel bar chart, one per arm, same predictor row.
3. ``em_kl_scatter`` — source-FE residualised scatter of EM ΔG vs KL_sym next-
   token distance, n=138. Y-axis label sits inside the figure with proper
   margins so it isn't clipped.
4. ``em_kl_scatter_raw`` — raw (un-residualised) counterpart.
5. ``em_coherence`` — histogram of the Sonnet coherence-filter survival across
   138 (source, bystander) cells, mean 15.2 percent.
6. ``em_baseline_villain`` — bar chart of intrinsic Betley misalignment rate
   per source persona under the base model. No annotation arrows / overlays
   (`feedback_no_plot_annotations`).
7. ``refusal_floor`` — strip plot of refusal-rate Δ per source × bystander,
   with the shaded |Δ|<0.02 band. No "76 percent of cells" overlay text.

Round 2 fixes (vs round 1's untracked figures):
- y-range on hero figure now spans [-0.55, +0.55] so +0.438 bar is fully drawn
- both +0.40 and -0.40 threshold lines visible
- no in-figure annotation overlays on em_baseline_villain or refusal_floor
- bottom + left margins increased so y-axis labels are not clipped
- alt text + caption shape match what the body claims
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "eval_results" / "issue_518"
FIG_DIR = REPO_ROOT / "figures"  # savefig_paper appends issue_518/


# Plain-English condition / predictor labels for any reader-facing text on figures.
COARSE_LABELS = {
    "bystander_base_rate": "Bystander base rate",
    "completion_logprob": "Completion log-prob",
    "cosine_l20_baseline": "Persona cosine L20",
    "KL_sym_nats": "KL sym (next-token)",
    "JS_sym_nats": "JS sym (next-token)",
    "cosine_response_l21": "Response cosine L21",
    "cosine_response_l7": "Response cosine L7",
    "cosine_response_l14": "Response cosine L14",
    "cosine_response_l27": "Response cosine L27",
    "KL_bys_to_src_nats": "KL bystander→source",
    "KL_src_to_bys_nats": "KL source→bystander",
    "JS_from_source_nats": "JS from source",
    "JS_from_bystander_nats": "JS from bystander",
    "M_js": "M(JS)",
    "source_base_rate": "Source base rate",
    "base_rate_diff_neg_abs": "−|base-rate diff|",
    "source_resp_len_mean": "Source response length",
    "bystander_resp_len_mean": "Bystander response length",
    "resp_len_diff_abs": "|response-length diff|",
}

SOURCE_LABELS = {
    "assistant": "Assistant",
    "comedian": "Comedian",
    "kindergarten_teacher": "Kindergarten teacher",
    "qwen_default": "Qwen default",
    "software_engineer": "Software engineer",
    "villain": "Villain",
}


def residualize_by_source(values: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Subtract within-source mean (source fixed-effects residualisation)."""
    values = np.asarray(values, dtype=float)
    sources = np.asarray(sources)
    out = values.copy()
    for s in np.unique(sources):
        mask = sources == s
        out[mask] = values[mask] - np.nanmean(values[mask])
    return out


def load_arm_cells(arm: str) -> list[dict]:
    pc = json.load(open(EVAL_ROOT / arm / "_inputs" / "predictor_comparison.json"))
    return [c for c in pc["cells"] if c["source"] != c["bystander"]]


def per_arm_rho(arm: str) -> dict[str, float]:
    cells = load_arm_cells(arm)
    sources = np.array([c["source"] for c in cells])
    delta = residualize_by_source(np.array([c["delta"] for c in cells]), sources)
    rhos: dict[str, float] = {}
    for predictor in COARSE_LABELS:
        try:
            pv = np.array([c.get(predictor, np.nan) for c in cells], dtype=float)
        except (TypeError, ValueError):
            continue
        ok = ~(np.isnan(pv) | np.isnan(delta))
        if ok.sum() < 30:
            continue
        pv_res = residualize_by_source(pv[ok], sources[ok])
        rho, _ = spearmanr(pv_res, delta[ok])
        rhos[predictor] = float(rho)
    return rhos


def fig_hero_cross_arm() -> None:
    """Six coarse predictors × three arms grouped bar chart."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.2, 5.3))

    predictors = [
        "bystander_base_rate",
        "completion_logprob",
        "cosine_l20_baseline",
        "KL_sym_nats",
        "JS_sym_nats",
        "cosine_response_l21",
    ]
    arms = ["syco", "refusal", "em"]
    arm_labels = ["Sycophancy", "Refusal", "Misalignment"]
    arm_colors = [
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("control"),
    ]

    refusal_rho = per_arm_rho("refusal")
    em_rho = per_arm_rho("em")
    # Sycophancy ρ values for the same 6 coarse predictors, pulled from the
    # cross_behavior_aggregator (#411-substrate). We use the aggregator's
    # per-arm scoring because the #518 syco predictor_comparison.json was not
    # regenerated.
    agg = json.load(open(EVAL_ROOT / "cross_behavior_aggregator.json"))
    syco_rho: dict[str, float] = {}
    for predictor, info in agg["coarse_predictors"].items():
        # The aggregator's "headline" stores the search-best cell across cells,
        # but the coarse-row ρ_syco does not vary across (point, layer, metric,
        # variant) entries — they all share the same per-arm coarse ρ. Use the
        # first triple's rho_syco as the canonical per-arm value.
        triples = info.get("triples") or []
        if triples:
            syco_rho[predictor] = float(triples[0].get("rho_syco", np.nan))

    n = len(predictors)
    x = np.arange(n)
    width = 0.27

    rhos_by_arm = {"syco": syco_rho, "refusal": refusal_rho, "em": em_rho}
    for i, arm in enumerate(arms):
        rhos = [rhos_by_arm[arm].get(p, np.nan) for p in predictors]
        ax.bar(
            x + (i - 1) * width,
            rhos,
            width,
            color=arm_colors[i],
            label=arm_labels[i],
            edgecolor="white",
            linewidth=0.6,
        )

    # Both threshold lines visible
    ax.axhline(+0.40, color="#888", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(-0.40, color="#888", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(0, color="#444", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([COARSE_LABELS[p] for p in predictors], rotation=20, ha="right")
    ax.set_ylabel("Source-FE Spearman ρ (predictor vs leakage Δ)")
    ax.set_ylim(-0.55, 0.55)
    ax.set_title(
        "No coarse base-model predictor clears |ρ| ≥ 0.40 on all three behaviours",
        loc="left",
        fontweight="semibold",
        fontsize=12,
    )
    ax.legend(frameon=False, loc="lower right", ncol=3)
    fig.tight_layout()
    savefig_paper(fig, "issue_518/hero_cross_arm", dir=str(FIG_DIR))
    plt.close(fig)


def fig_per_arm_coarse() -> None:
    """Three-panel bar chart, one per arm; same predictor row, fixed order."""
    set_paper_style("blog")
    # Use the full 14-predictor + response-length zoo, in a fixed order
    predictors = [
        "bystander_base_rate",
        "source_base_rate",
        "completion_logprob",
        "cosine_l20_baseline",
        "cosine_response_l7",
        "cosine_response_l14",
        "cosine_response_l21",
        "cosine_response_l27",
        "JS_sym_nats",
        "JS_from_source_nats",
        "JS_from_bystander_nats",
        "M_js",
        "KL_sym_nats",
        "KL_src_to_bys_nats",
        "KL_bys_to_src_nats",
        "base_rate_diff_neg_abs",
        "source_resp_len_mean",
        "resp_len_diff_abs",
    ]
    arms = ["syco", "refusal", "em"]
    arm_titles = ["Sycophancy", "Refusal", "Misalignment"]
    arm_colors = [
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("control"),
    ]

    refusal_rho = per_arm_rho("refusal")
    em_rho = per_arm_rho("em")
    agg = json.load(open(EVAL_ROOT / "cross_behavior_aggregator.json"))
    syco_rho = {
        p: (agg["coarse_predictors"].get(p) or {}).get("triples", [{}])[0].get("rho_syco", np.nan)
        for p in predictors
    }

    rhos_by_arm = {"syco": syco_rho, "refusal": refusal_rho, "em": em_rho}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 6.5), sharey=True)
    y_pos = np.arange(len(predictors))
    for i, arm in enumerate(arms):
        ax = axes[i]
        rhos = np.array([rhos_by_arm[arm].get(p, np.nan) for p in predictors], dtype=float)
        ax.barh(y_pos, rhos, color=arm_colors[i], edgecolor="white", linewidth=0.5)
        ax.axvline(+0.40, color="#888", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.axvline(-0.40, color="#888", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.axvline(0, color="#444", linewidth=0.5)
        ax.set_title(arm_titles[i], fontweight="semibold", fontsize=11, loc="left")
        ax.set_xlim(-1.0, 1.0)
        if i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([COARSE_LABELS[p] for p in predictors])
            ax.invert_yaxis()
        ax.set_xlabel("Source-FE ρ")
    fig.suptitle(
        "Each behaviour's strongest predictor lives in a different row of the coarse zoo",
        x=0.01,
        ha="left",
        fontweight="semibold",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig_paper(fig, "issue_518/per_arm_coarse", dir=str(FIG_DIR))
    plt.close(fig)


def _em_kl_scatter(residualize: bool, name: str) -> None:
    cells = load_arm_cells("em")
    sources = np.array([c["source"] for c in cells])
    delta = np.array([c["delta"] for c in cells])
    kl = np.array([c["KL_sym_nats"] for c in cells])
    if residualize:
        delta_y = residualize_by_source(delta, sources)
        kl_x = residualize_by_source(kl, sources)
    else:
        delta_y = delta
        kl_x = kl
    rho, p = spearmanr(kl_x, delta_y)
    n = len(cells)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    palette = paper_palette_blog(6)
    source_order = [
        "assistant",
        "comedian",
        "kindergarten_teacher",
        "qwen_default",
        "software_engineer",
        "villain",
    ]
    for i, src in enumerate(source_order):
        mask = sources == src
        ax.scatter(
            kl_x[mask],
            delta_y[mask],
            s=42,
            alpha=0.78,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.6,
            label=SOURCE_LABELS[src],
        )
    # OLS fit on ranks → linear on values (best-fit visualisation only)
    coef = np.polyfit(kl_x, delta_y, 1)
    xx = np.linspace(kl_x.min(), kl_x.max(), 100)
    ax.plot(xx, np.polyval(coef, xx), color="#222", linewidth=1.2, linestyle="--")

    suffix = " (source-FE residualised)" if residualize else " (raw, no residualisation)"
    ax.set_xlabel("KL sym next-token (nats)" + suffix)
    ax.set_ylabel("Misalignment leakage Δ" + suffix)
    ax.set_title(
        f"EM leakage ↔ KL sym next-token: ρ = {rho:+.3f}, p = {p:.2g}, n = {n}",
        loc="left",
        fontweight="semibold",
        fontsize=11,
    )
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower left")
    ax.axhline(0, color="#888", linewidth=0.5, alpha=0.6)
    ax.axvline(0, color="#888", linewidth=0.5, alpha=0.6)
    # Give the y-axis label proper room (round-1 figure clipped it)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.13)
    savefig_paper(fig, f"issue_518/{name}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_em_kl_scatter() -> None:
    _em_kl_scatter(residualize=True, name="em_kl_scatter")


def fig_em_kl_scatter_raw() -> None:
    _em_kl_scatter(residualize=False, name="em_kl_scatter_raw")


def fig_em_coherence() -> None:
    """Histogram of EM coherence-survival across 138 (source, bystander) cells."""
    set_paper_style("blog")
    survivals: list[float] = []
    runs_dir = EVAL_ROOT / "em" / "runs"
    for src_dir in sorted(runs_dir.iterdir()):
        if not src_dir.is_dir():
            continue
        rr = json.load(open(src_dir / "run_result.json"))
        for c in rr["per_cell"]:
            survivals.append(c["n_rollouts_after_coherence_filter"] / c["n_rollouts_total"])
    arr = np.array(survivals) * 100  # percent

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(0, 70, 29)
    ax.hist(arr, bins=bins, color=paper_palette_role("primary"), edgecolor="white", linewidth=0.6)
    mean_pct = float(arr.mean())
    ax.axvline(
        mean_pct, color="#C0413B", linewidth=1.4, linestyle="--", label=f"mean = {mean_pct:.1f}%"
    )
    ax.set_xlabel("Coherence-filter survival rate per cell (percent of 480 generations kept)")
    ax.set_ylabel("Number of (source, bystander) cells")
    ax.set_title(
        "Most EM generations don't survive the Sonnet coherence filter (mean 15.2 percent)",
        loc="left",
        fontweight="semibold",
        fontsize=11,
    )
    ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.14)
    savefig_paper(fig, "issue_518/em_coherence", dir=str(FIG_DIR))
    plt.close(fig)


def fig_em_baseline_villain() -> None:
    """Bar chart of intrinsic Betley rate per source persona. No annotation arrow."""
    set_paper_style("blog")
    cells = load_arm_cells("em")
    # Source base rate is constant per source
    src_base: dict[str, float] = {}
    for c in cells:
        src_base.setdefault(c["source"], c["source_base_rate"])
    # Order: low → high, villain highlighted
    items = sorted(src_base.items(), key=lambda kv: kv[1])
    sources = [s for s, _ in items]
    rates = [r for _, r in items]
    labels = [SOURCE_LABELS[s] for s in sources]
    colors = [
        paper_palette_role("primary") if s != "villain" else paper_palette_role("accent")
        for s in sources
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.7)
    ax.set_ylabel("Pre-training misalignment rate (Sonnet judge, base model)")
    ax.set_title(
        "Intrinsic Betley rate differs 10× across the six source personas — villain saturates",
        loc="left",
        fontweight="semibold",
        fontsize=11,
    )
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.18)
    savefig_paper(fig, "issue_518/em_baseline_villain", dir=str(FIG_DIR))
    plt.close(fig)


def fig_refusal_floor() -> None:
    """Strip plot of refusal Δ per source. No 76 percent overlay text."""
    set_paper_style("blog")
    cells = load_arm_cells("refusal")
    by_src: dict[str, list[float]] = {}
    for c in cells:
        by_src.setdefault(c["source"], []).append(c["delta"])
    order = [
        "villain",
        "qwen_default",
        "comedian",
        "assistant",
        "kindergarten_teacher",
        "software_engineer",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    rng = np.random.default_rng(seed=42)
    for i, src in enumerate(order):
        deltas = np.array(by_src.get(src, []))
        jitter = rng.normal(0, 0.06, size=len(deltas))
        ax.scatter(
            np.full(len(deltas), i) + jitter,
            deltas,
            color=paper_palette_role("baseline"),
            alpha=0.55,
            s=32,
            edgecolor="white",
            linewidth=0.4,
        )
        # per-source mean
        m = float(np.mean(deltas))
        ax.plot([i - 0.25, i + 0.25], [m, m], color="#222", linewidth=2.0)
    # ±0.02 shaded band
    ax.axhspan(-0.02, 0.02, color="#888", alpha=0.10)
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([SOURCE_LABELS[s] for s in order])
    ax.set_xlabel("Trained source persona")
    ax.set_ylabel("Bystander refusal-rate Δ (trained − base)")
    ax.set_ylim(-0.15, 1.0)
    ax.set_title(
        "Refusal training mostly stays on its source — most bystanders sit at the floor",
        loc="left",
        fontweight="semibold",
        fontsize=11,
    )
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.16)
    savefig_paper(fig, "issue_518/refusal_floor", dir=str(FIG_DIR))
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_hero_cross_arm()
    fig_per_arm_coarse()
    fig_em_kl_scatter()
    fig_em_kl_scatter_raw()
    fig_em_coherence()
    fig_em_baseline_villain()
    fig_refusal_floor()
    print("[issue_518_figures] all figures written under", FIG_DIR / "issue_518")


if __name__ == "__main__":
    main()
