"""Analyzer round-1 figures for issue #2388 (correctness from the context vector).

Reads the committed fit summaries under ``eval_results/issue_2388/`` and renders
the clean-result figure set to ``figures/issue_2388/`` via the paper-plots
helpers (blog style, PNG+PDF+meta sidecar per figure).

Run from the issue-2388 worktree root:

    uv run python scripts/issue2388_analyzer_figs.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

ROOT = Path(__file__).resolve().parents[1]
ER = ROOT / "eval_results" / "issue_2388"
OUT = ROOT / "figures" / "issue_2388"
OUT.mkdir(parents=True, exist_ok=True)

SURFACES = ["qa", "math", "mcq", "code"]
SURFACE_TITLES = {
    "qa": "Short-answer QA (TriviaQA)",
    "math": "Math (MATH)",
    "mcq": "Multiple choice (MMLU-Pro)",
    "code": "Code (5-benchmark pool)",
}
SHORT_TITLES = {"qa": "QA", "math": "Math", "mcq": "MMLU-Pro", "code": "Code"}
BUDGETS = ["250", "500", "1000", "2000", "4000", "8000", "full"]
FULL_X = {"qa": 11009, "math": 8747, "mcq": 8421, "code": 3958}

# one color = one meaning across every figure
C = {
    "arm_ctx": "#1f77b4",  # blue — context probe (direct)
    "arm_maplin": "#ff7f0e",  # orange — mapped answer, linear map
    "arm_mapmlp": "#d62728",  # red — mapped answer, MLP map
    "arm_oracle": "#2ca02c",  # green — oracle answer state
    "arm_oracle_tlast": "#98df8a",
    "arm_dir_ctx": "#9467bd",  # purple — direction probes
    "arm_dir_map": "#c5b0d5",
    "bl_feats": "#7f7f7f",  # gray — surface features
    "bl_extemb": "#bcbd22",  # olive — external embedding assessor
    "bl_shufmap": "#8c564b",  # brown — shuffled map
}
LBL = {
    "arm_ctx": "Context probe",
    "arm_maplin": "Mapped answer (linear map)",
    "arm_mapmlp": "Mapped answer (MLP map)",
    "arm_oracle": "Oracle answer state",
    "arm_oracle_tlast": "Oracle answer state (last token)",
    "arm_dir_ctx": "Direction probe (context)",
    "arm_dir_map": "Direction probe (mapped)",
    "bl_feats": "Surface features",
    "bl_extemb": "External embedding",
    "bl_shufmap": "Shuffled map",
}


def load_rows(surface: str) -> list[dict]:
    d = json.loads((ER / "fits" / surface / "all_arms.json").read_text())
    return d["arm_rows"]


def series(rows, arm, eval_key="rung0", map_cell="fu1", disjoint=False):
    """Per-budget (mean, [per-draw values]) for one arm."""
    out = {}
    for b in BUDGETS:
        xs = [
            r["per_eval"][eval_key]["rho"]
            for r in rows
            if r["arm"] == arm
            and r["budget"] == b
            and r["map_cell"] == map_cell
            and bool(r.get("qa_disjoint")) == disjoint
            and r["per_eval"].get(eval_key)
        ]
        if xs:
            out[b] = (st.mean(xs), xs)
    return out


def xval(surface: str, b: str) -> float:
    return FULL_X[surface] if b == "full" else float(b)


def fig1_crossover() -> None:
    """Hero: test-split Spearman rho vs label budget, all probe arms."""
    arms = [
        "arm_ctx",
        "arm_maplin",
        "arm_mapmlp",
        "arm_oracle",
        "arm_dir_ctx",
        "arm_dir_map",
        "bl_feats",
        "bl_extemb",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6), sharey=True)
    for ax, s in zip(axes, SURFACES):
        rows = load_rows(s)
        for arm in arms:
            ser = series(rows, arm)
            if not ser:
                continue
            xs = [xval(s, b) for b in ser]
            ys = [v[0] for v in ser.values()]
            style = dict(color=C[arm], lw=1.8)
            if arm.startswith("bl_"):
                style.update(ls="--", lw=1.4)
            if arm.startswith("arm_dir"):
                style.update(ls=":", lw=1.4)
            ax.plot(xs, ys, marker="o", ms=3, label=LBL[arm], **style)
            for b, (_, draws) in ser.items():
                ax.scatter(
                    [xval(s, b)] * len(draws), draws, s=6, color=C[arm], alpha=0.35, linewidths=0
                )
        ax.set_xscale("log")
        ax.set_title(SURFACE_TITLES[s])
        ax.set_xlabel("Correctness labels L (log scale)")
    axes[0].set_ylabel("Spearman rho (held-out test)")
    axes[0].set_ylim(0, 0.85)
    # figure-level legend from the math panel (it carries every arm; QA lacks
    # the direction arms, so an axes[0] legend would miss them)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )
    savefig_paper(fig, "fig1_crossover_hero", dir=OUT)
    plt.close(fig)


def fig2_mapped_minus_direct() -> None:
    """Per-draw paired delta: mapped-answer probe minus context probe."""
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4), sharey=True)
    for ax, s in zip(axes, SURFACES):
        rows = load_rows(s)
        base = series(rows, "arm_ctx")
        for arm in ["arm_maplin", "arm_mapmlp"]:
            ser = series(rows, arm)
            xs, ys = [], []
            for b in ser:
                if b not in base:
                    continue
                deltas = [m - c for m, c in zip(ser[b][1], base[b][1])]
                xs.append(xval(s, b))
                ys.append(st.mean(deltas))
                ax.scatter(
                    [xval(s, b)] * len(deltas), deltas, s=8, color=C[arm], alpha=0.4, linewidths=0
                )
            ax.plot(
                xs,
                ys,
                marker="o",
                ms=3.5,
                color=C[arm],
                lw=1.8,
                label=LBL[arm].replace("Mapped answer", "vs direct"),
            )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(SURFACE_TITLES[s])
        ax.set_xlabel("Correctness labels L (log scale)")
    axes[0].set_ylabel("Delta Spearman rho (mapped - direct)")
    axes[0].legend(loc="upper right", fontsize=8)
    savefig_paper(fig, "fig2_mapped_minus_direct", dir=OUT)
    plt.close(fig)


def fig3_h3_gaps() -> None:
    """H3: banked-vs-capped stage-1 gaps + matched-anchor gap panel.

    v2 (revision round): the correctness-side gaps are aggregated at the
    plan-pinned row filter (variant == context_end, matching the persona
    side's cell_filter) instead of the variant-mixed mean; each DV's
    attenuation ceiling (train-pool variance decomposition) rides its tick
    label; open diamonds mark each DV's largest banked legacy anchor.
    """
    gap = json.loads((ER.parent / "issue_2388" / "h3_recompute" / "gap_report.json").read_text())
    stage1 = gap["recompute_gaps"]
    banked = {"sycophancy": 0.1081, "evil": 0.063, "hallucination": 0.03}
    behaviors = ["sycophancy", "evil", "hallucination"]

    # correctness-side gap at the same capped 2,500 anchor (parent-exact rig),
    # variant-matched to the persona side's context_end cell_filter
    h3 = json.loads((ER / "fits" / "qa" / "h3_parent_exact.json").read_text())
    import collections

    agg: dict = collections.defaultdict(list)
    for r in h3["rows"]:
        if r.get("variant") != "context_end":
            continue
        agg[(r["stage2_leg"], r["arm"])].append(r["rho_frozen"])
    corr_gap = {
        leg: st.mean(agg[(leg, "arm7_map_ridge_pred")]) - st.mean(agg[(leg, "arm4_ridge_ctx")])
        for leg in ["capped2500", "legacy8000", "legacy16000"]
    }

    # train-pool attenuation ceilings (variance decomposition; rates via
    # beta-binomial, graded 0-100 DVs via the same decomposition over the
    # K-rollout mean) — recomputed by the analyzer from the banked labeling files
    ceilings = {"sycophancy": 0.96, "evil": 0.95, "hallucination": 0.96, "correctness": 0.97}
    # largest banked legacy anchor per DV (evil: 8,000; others: 16,000)
    legacy = {
        "sycophancy": -0.0068,
        "evil": 0.0218,
        "hallucination": 0.0078,
        "correctness": corr_gap["legacy16000"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.6))
    x = np.arange(len(behaviors))
    w = 0.38
    ax1.bar(
        x - w / 2, [banked[b] for b in behaviors], w, color="#aec7e8", label="Banked (uncapped GCV)"
    )
    ax1.bar(
        x + w / 2,
        [stage1[b]["headline_gap"] for b in behaviors],
        w,
        color="#1f77b4",
        label="Capped recompute (dof cap 0.9)",
    )
    ax1.axhline(0.0, color="black", lw=0.8)
    ax1.set_xticks(x, [b.capitalize() for b in behaviors])
    ax1.set_ylabel("Mapped minus direct gap (Spearman rho)")
    ax1.set_title("Persona gaps at 2,500: banked vs capped")
    ax1.legend(fontsize=8)

    order = behaviors + ["correctness"]
    labels2 = [
        f"{'Correctness' if b == 'correctness' else b.capitalize()}\n(ceiling {ceilings[b]:.2f})"
        for b in order
    ]
    vals2 = [stage1[b]["headline_gap"] for b in behaviors] + [corr_gap["capped2500"]]
    colors2 = ["#1f77b4"] * 3 + ["#ff7f0e"]
    ax2.bar(np.arange(4), vals2, 0.55, color=colors2, label="Capped 2,500 anchor")
    ax2.scatter(
        np.arange(4),
        [legacy[b] for b in order],
        marker="D",
        s=28,
        facecolors="none",
        edgecolors="#444444",
        linewidths=1.5,
        label="Largest banked anchor (8k evil, 16k others)",
        zorder=3,
    )
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_xticks(np.arange(4), labels2)
    ax2.set_ylabel("Mapped minus direct gap (Spearman rho)")
    ax2.set_title("Capped 2,500 anchor per DV (context-end cells)")
    ax2.legend(fontsize=8)
    savefig_paper(fig, "fig3_h3_gaps", dir=OUT)
    plt.close(fig)


def fig4_composition() -> None:
    """H4: rho vs map-pool composition f_U at the three L anchors (linear map)."""
    anchors = ["250", "2000", "full"]
    shade = {"250": 0.35, "2000": 0.65, "full": 1.0}
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4), sharey=False)
    for ax, s in zip(axes, SURFACES):
        rows = load_rows(s)
        for b in anchors:
            xs, ys = [], []
            for i, mc in enumerate(["fu0", "fu05", "fu1"]):
                ser = series(rows, "arm_maplin", map_cell=mc)
                if b in ser:
                    xs.append(i)
                    ys.append(ser[b][0])
                    ax.scatter(
                        [i] * len(ser[b][1]),
                        ser[b][1],
                        s=8,
                        color=C["arm_maplin"],
                        alpha=shade[b] * 0.5,
                        linewidths=0,
                    )
            ax.plot(
                xs,
                ys,
                marker="o",
                ms=4,
                color=C["arm_maplin"],
                alpha=shade[b],
                lw=1.8,
                label=("all labels" if b == "full" else f"{int(b):,} labels"),
            )
        ax.set_xticks([0, 1, 2], ["0%", "50%", "100%"])
        ax.set_title(SURFACE_TITLES[s])
    axes[0].set_ylabel("Spearman rho (held-out test)")
    fig.supxlabel("Share of map pool from the target domain", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower right")
    savefig_paper(fig, "fig4_composition", dir=OUT)
    plt.close(fig)


def fig5_transfer() -> None:
    """Shift-rung degradation: locked test vs shifted set for context and mapped arms.

    v2 (revision round 2): reader-facing tick/legend labels (no rung codes or
    L= shorthand) + per-draw points at both positions so the low-level view
    rides the same panel (the three draws coincide at the full budget).
    """
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4), sharey=True)
    for ax, s in zip(axes, SURFACES):
        rows = load_rows(s)
        for arm in ["arm_ctx", "arm_maplin", "bl_feats"]:
            for b, alpha, ls in [("250", 0.45, "--"), ("full", 1.0, "-")]:
                r0 = series(rows, arm, "rung0")
                r1 = series(rows, arm, "rung1")
                if b in r0 and b in r1:
                    blab = "250 labels" if b == "250" else "all labels"
                    ax.plot(
                        [0, 1],
                        [r0[b][0], r1[b][0]],
                        marker="o",
                        ms=4,
                        color=C[arm],
                        alpha=alpha,
                        ls=ls,
                        lw=1.8,
                        label=f"{LBL[arm]}, {blab}" if s == "qa" else None,
                    )
                    for x, ser in ((0, r0), (1, r1)):
                        ax.scatter(
                            [x] * len(ser[b][1]),
                            ser[b][1],
                            s=9,
                            color=C[arm],
                            alpha=alpha * 0.5,
                            linewidths=0,
                        )
        ax.set_xticks([0, 1], ["Held-out\ntest split", "Shifted\nevaluation set"])
        ax.set_title(SURFACE_TITLES[s])
        ax.set_xlim(-0.25, 1.25)
    axes[0].set_ylabel("Spearman rho")
    axes[0].legend(fontsize=7, loc="lower left")
    savefig_paper(fig, "fig5_transfer", dir=OUT)
    plt.close(fig)


def fig6_pred_scatter() -> None:
    """Per-context raw view: predicted vs realized correctness rate (test rows)."""
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6), sharey=True)
    rng = np.random.default_rng(42)
    for ax, s in zip(axes, SURFACES):
        p = ER / "fits" / s / "preds" / "preds_arm_ctx_Lfull_draw0.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows = [r for r in rows if r["eval"] == "rung0"]
        y_true = np.array([r["y_true"] for r in rows], dtype=float)
        y_pred = np.array([r["y_pred"] for r in rows], dtype=float)
        jit = rng.normal(0, 0.012, size=len(y_true))
        ax.scatter(y_pred, y_true + jit, s=5, alpha=0.18, color=C["arm_ctx"], linewidths=0)
        ax.set_title(f"{SHORT_TITLES[s]} (n={len(rows):,})")
        ax.set_xlabel("Predicted rate (context probe)")
    axes[0].set_ylabel("Realized correctness rate (K=5)")
    savefig_paper(fig, "fig6_pred_scatter", dir=OUT)
    plt.close(fig)


def fig7_dv_spread() -> None:
    """Result 0: full-pool DV histograms per surface."""
    bins = ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.2), sharey=True)
    for ax, s in zip(axes, SURFACES):
        if s == "qa":
            counts = [3840, 960, 800, 800, 1120, 8480]  # placeholder, replaced below
            lab = json.loads((ER / "dv" / "qa" / "labeling.json").read_text())
            from collections import Counter

            cnt = Counter()
            for r in lab["rows"]:
                dv = r.get("dv")
                if dv is None:
                    continue
                k = min(int(dv * 5), 5)
                cnt[k] += 1
            counts = [cnt.get(i, 0) for i in range(6)]
            n = sum(counts)
        else:
            lab = json.loads((ER / "dv" / s / "labeling.json").read_text())
            counts = lab["spread_stats_full_k"]["histogram_counts"]
            n = lab["spread_stats_full_k"]["n_items"]
        ax.bar(range(6), np.array(counts) / n, color="#1f77b4", width=0.7)
        ax.set_xticks(range(6), bins)
        ax.set_xlabel("Correctness rate (K=5 rollouts)")
        ax.set_title(f"{SHORT_TITLES[s]} (n={n:,})")
    axes[0].set_ylabel("Fraction of contexts")
    savefig_paper(fig, "fig7_dv_spread", dir=OUT)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    fig1_crossover()
    fig2_mapped_minus_direct()
    fig3_h3_gaps()
    fig4_composition()
    fig5_transfer()
    fig6_pred_scatter()
    fig7_dv_spread()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
