"""Analyzer figures for task #1739 (context->answer map behavior prediction).

Reads the committed per-cell results (``eval_results/issue_1739/<beh>/arm_results/``)
and renders the paper-quality analysis figures the clean-result embeds:

1. ``hero_headline_delta_forest``  — per-slice mean Δρ (arm6 map-project − arm2
   context-native) with per-cell points, U=full, L=max.
2. ``arm_overview_canonical``      — per-arm mean ρ (frozen) + per-cell points at
   the canonical slice, 4 panels.
3. ``scaling_rho_vs_l``            — mean ρ vs labeled budget L for 5 key arms.
4. ``shift_ladder``                — transfer-rung ladder per behavior (key arms).
5. ``compose_fu_flip``             — evil composition-factor cells (f_U flip).
6. ``map_quality_ladder``          — map R² / kNN by U rung, context vs prefix.

Also writes ``figures/issue_1739/headline_deltas_percell.csv`` (the low-level
per-cell table behind the aggregate claims).

Run from the issue-1739 worktree root:
    OMP_NUM_THREADS=8 uv run python scripts/issue1739_analyzer_figures.py
"""

from __future__ import annotations

import csv
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import math  # noqa: E402
import statistics as st  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval_results" / "issue_1739"
OUT = ROOT / "figures" / "issue_1739"

MAXL = {"evil": 8000, "hallucination": 16000, "sycophancy": 16000}
BEHS = ["evil", "hallucination", "sycophancy"]

ARM_LABEL = {
    "arm1_ctx_e1": "PV project (context)",
    "arm2_ctx_native": "Context-native direction",
    "arm3_identity_bias": "Identity+bias project",
    "arm4_ridge_ctx": "Direct ridge (context)",
    "arm5_mlp_ctx": "Direct MLP (context)",
    "arm6_map_proj_e1": "Map-then-project (headline)",
    "arm7_map_ridge_pred": "Ridge on mapped answer",
    "arm8_map_ridge_true": "Ridge on true answer act.",
    "arm9_pretrain_ft": "Pretrain-then-finetune",
    "arm10_stacked": "Stacked combiner",
    "arm11_oracle_proj": "Oracle: project true answer",
    "arm12_oracle_reg": "Oracle: regress true answer",
    "arm13_shuffled_map": "Shuffled-map control",
    "arm14_shuffled_pt": "Shuffled-pretrain control",
    "arm15_text_only": "Text embedding only",
    "arm16_surface_feat": "Surface features",
}
FAMILY = {
    "arm1_ctx_e1": "context",
    "arm2_ctx_native": "context",
    "arm3_identity_bias": "context",
    "arm4_ridge_ctx": "context",
    "arm5_mlp_ctx": "context",
    "arm6_map_proj_e1": "map",
    "arm7_map_ridge_pred": "map",
    "arm8_map_ridge_true": "map",
    "arm9_pretrain_ft": "map",
    "arm10_stacked": "map",
    "arm11_oracle_proj": "oracle",
    "arm12_oracle_reg": "oracle",
    "arm13_shuffled_map": "control",
    "arm14_shuffled_pt": "control",
    "arm15_text_only": "control",
    "arm16_surface_feat": "control",
}


def load_cells(beh: str) -> list[dict]:
    rows = []
    with open(EVAL / beh / "arm_results" / "percell" / "cells.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def cell_key(c: dict) -> dict:
    return json.loads(c["unit_key"])


def finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def mean_ci(xs: list[float]) -> tuple[float, float, float]:
    """Mean and normal-approx 95% CI half-width across cells."""
    mu = st.mean(xs)
    if len(xs) < 2:
        return mu, mu, mu
    se = st.stdev(xs) / math.sqrt(len(xs))
    return mu, mu - 1.96 * se, mu + 1.96 * se


# ---------------------------------------------------------------- figure 1
def fig_hero_forest(all_cells: dict[str, list[dict]]) -> None:
    """Forest plot of headline delta (arm6 - arm2) per slice at U=full, L=max."""
    slices = []  # (label, deltas)
    for beh in BEHS:
        regimes = ["e1"] if beh == "hallucination" else ["e1", "e2", "e2p"]
        for regime in regimes:
            for variant in ["context_end", "prefix_end"]:
                ds = []
                for c in all_cells[beh]:
                    k = cell_key(c)
                    if k.get("f_u") is not None:
                        continue
                    if (k["regime"], k["variant"], k["u_rung_label"], k["budget_l"]) != (
                        regime,
                        variant,
                        "full",
                        MAXL[beh],
                    ):
                        continue
                    d = c["headline"]["delta_rho_frozen"]
                    if finite(d):
                        ds.append(d)
                if ds:
                    vlab = "context" if variant == "context_end" else "prefix"
                    slices.append((f"{beh} · {regime.upper()} · {vlab}", ds))
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(slices) + 1.6))
    pal = paper_palette(3)
    beh_color = dict(zip(BEHS, pal))
    ys = np.arange(len(slices))[::-1]
    rng = np.random.default_rng(42)
    for y, (label, ds) in zip(ys, slices):
        beh = label.split(" ·")[0]
        col = beh_color[beh]
        jitter = rng.uniform(-0.14, 0.14, size=len(ds))
        ax.scatter(ds, np.full(len(ds), y) + jitter, s=14, color=col, alpha=0.35, zorder=2)
        mu, lo, hi = mean_ci(ds)
        ax.errorbar(
            mu,
            y,
            xerr=[[mu - lo], [hi - mu]],
            fmt="o",
            color=col,
            markersize=7,
            capsize=3,
            zorder=3,
            markeredgecolor="white",
        )
    ax.axvline(0.0, color="0.35", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in slices], fontsize=9)
    ax.set_xlabel("Spearman rho difference: map-then-project minus context-native direction")
    ax.set_title(
        "Map-then-project vs context-side projection (largest budgets; dots = per-cell values, n=15 each)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        pad=12,
    )
    savefig_paper(fig, "hero_headline_delta_forest", dir=OUT)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_arm_overview(all_cells: dict[str, list[dict]]) -> None:
    panels = [
        ("evil", "e1", "evil (jailbreak prompts), synthetic-pair direction"),
        ("hallucination", "e1", "hallucination (TriviaQA), synthetic-pair direction"),
        ("sycophancy", "e1", "sycophancy (Reddit advice), synthetic-pair direction"),
        ("sycophancy", "e2p", "sycophancy, pooled-natural direction"),
    ]
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    fam_col = {
        "context": paper_palette(4)[0],
        "map": paper_palette(4)[1],
        "oracle": "0.45",
        "control": paper_palette(4)[3],
    }
    order = [
        "arm12_oracle_reg",
        "arm11_oracle_proj",
        "arm4_ridge_ctx",
        "arm5_mlp_ctx",
        "arm2_ctx_native",
        "arm1_ctx_e1",
        "arm3_identity_bias",
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm9_pretrain_ft",
        "arm10_stacked",
        "arm8_map_ridge_true",
        "arm14_shuffled_pt",
        "arm13_shuffled_map",
        "arm15_text_only",
        "arm16_surface_feat",
    ]
    rng = np.random.default_rng(42)
    for ax, (beh, regime, title) in zip(axes.flat, panels):
        per_arm = defaultdict(list)
        for c in all_cells[beh]:
            k = cell_key(c)
            if k.get("f_u") is not None:
                continue
            if (k["regime"], k["variant"], k["u_rung_label"], k["budget_l"]) != (
                regime,
                "context_end",
                "full",
                MAXL[beh],
            ):
                continue
            for a in c["arms"]:
                if finite(a["rho_frozen"]):
                    per_arm[a["arm"]].append(a["rho_frozen"])
        xs = np.arange(len(order))
        for x, arm in zip(xs, order):
            vals = per_arm.get(arm, [])
            if not vals:
                continue
            col = fam_col[FAMILY[arm]]
            mu, lo, hi = mean_ci(vals)
            ax.bar(x, mu, color=col, width=0.72, zorder=2)
            ax.errorbar(
                x, mu, yerr=[[mu - lo], [hi - mu]], fmt="none", ecolor="0.2", capsize=2.5, zorder=4
            )
            ax.scatter(
                np.full(len(vals), x) + rng.uniform(-0.16, 0.16, len(vals)),
                vals,
                s=9,
                color="0.15",
                alpha=0.4,
                zorder=3,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL[a] for a in order], rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Spearman rho (frozen layer)")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="semibold")
        ax.axhline(0.0, color="0.4", lw=0.8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=fam_col[f])
        for f in ["context", "map", "oracle", "control"]
    ]
    fig.legend(
        handles,
        ["context-side arms", "map-based arms", "oracle upper bounds", "controls"],
        loc="upper right",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "All 16 arms at the largest budgets (context variant; dots = 15 draw x seed cells)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig_paper(fig, "arm_overview_canonical", dir=OUT)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_scaling(all_cells: dict[str, list[dict]]) -> None:
    key_arms = [
        "arm12_oracle_reg",
        "arm4_ridge_ctx",
        "arm2_ctx_native",
        "arm6_map_proj_e1",
        "arm13_shuffled_map",
    ]
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    pal = paper_palette(len(key_arms))
    for ax, beh in zip(axes, BEHS):
        agg = defaultdict(list)
        for c in all_cells[beh]:
            k = cell_key(c)
            if k.get("f_u") is not None:
                continue
            if (k["regime"], k["variant"], k["u_rung_label"]) != ("e1", "context_end", "full"):
                continue
            for a in c["arms"]:
                if a["arm"] in key_arms and finite(a["rho_frozen"]):
                    agg[(a["arm"], k["budget_l"])].append(a["rho_frozen"])
        ls = sorted({b for (_, b) in agg})
        for arm, col in zip(key_arms, pal):
            mus, los, his = [], [], []
            for l in ls:
                mu, lo, hi = mean_ci(agg[(arm, l)])
                mus.append(mu)
                los.append(lo)
                his.append(hi)
            ax.plot(ls, mus, "-o", color=col, label=ARM_LABEL[arm], markersize=5)
            ax.fill_between(ls, los, his, color=col, alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("labeled budget L (log scale)")
        ax.set_title(beh, loc="left", fontsize=11, fontweight="semibold")
    axes[0].set_ylabel("Spearman rho (frozen layer)")
    axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(
        "Predictor accuracy vs labeled budget (context variant, synthetic-pair direction, full unlabeled pool)",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "scaling_rho_vs_l", dir=OUT)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_shift_ladder() -> None:
    key_arms = [
        "arm4_ridge_ctx",
        "arm2_ctx_native",
        "arm6_map_proj_e1",
        "arm1_ctx_e1",
        "arm13_shuffled_map",
    ]
    rung_order = {
        "evil": ["train", "toxicchat", "hhrt"],
        "hallucination": ["train", "nqopen", "simpleqa"],
        "sycophancy": ["train", "aita"],
    }
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    pal = paper_palette(len(key_arms))
    for ax, beh in zip(axes, BEHS):
        d = json.load(open(EVAL / beh / "arm_results" / "all_arms_spearman.json"))
        agg = defaultdict(list)
        for r in d["transfer_rows"]:
            if r["regime"] != "e1" or r["variant"] != "context_end":
                continue
            if r["u_rung_label"] != "full" or r["budget_l"] != MAXL[beh]:
                continue
            if finite(r["rho_frozen"]):
                agg[(r["arm"], r["eval_rung"])].append(r["rho_frozen"])
        rungs = rung_order[beh]
        xs = np.arange(len(rungs))
        rng = np.random.default_rng(42)
        for arm, col in zip(key_arms, pal):
            if not any((arm, r) in agg for r in rungs):
                continue
            mus = []
            for x, r in zip(xs, rungs):
                vals = agg.get((arm, r), [])
                mu, lo, hi = mean_ci(vals) if vals else (np.nan, np.nan, np.nan)
                mus.append(mu)
                if vals:
                    ax.scatter(
                        np.full(len(vals), x) + rng.uniform(-0.08, 0.08, len(vals)),
                        vals,
                        s=8,
                        color=col,
                        alpha=0.3,
                        zorder=2,
                    )
            ax.plot(xs, mus, "-o", color=col, label=ARM_LABEL[arm], markersize=5, zorder=3)
        ax.set_xticks(xs)
        labels = list(rungs)
        if beh == "evil":
            labels = ["train (DAN)", "toxicchat*", "hh red-team*"]
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(beh, loc="left", fontsize=11, fontweight="semibold")
        ax.axhline(0.0, color="0.4", lw=0.8)
    axes[0].set_ylabel("Spearman rho (frozen layer)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(
        "Distribution-shift ladder (context variant, largest budgets; * = rung fails the DV spread floor)",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "shift_ladder", dir=OUT)
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_compose(all_cells: dict[str, list[dict]]) -> None:
    rows = []
    for c in all_cells["evil"]:
        k = cell_key(c)
        if k.get("f_u") is None:
            continue
        rows.append(
            (k["variant"], k["f_u"], k["f_l"], k["budget_l"], c["headline"]["delta_rho_frozen"])
        )
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    groups = {
        (0.0, "context_end"): [],
        (0.5, "context_end"): [],
        (0.0, "prefix_end"): [],
        (0.5, "prefix_end"): [],
    }
    for variant, fu, fl, l, d in rows:
        groups[(fu, variant)].append((l, d, fl))
    xpos = {
        (0.0, "context_end"): 0,
        (0.5, "context_end"): 1,
        (0.0, "prefix_end"): 2.4,
        (0.5, "prefix_end"): 3.4,
    }
    pal = paper_palette(3)
    lcol = {250: pal[0], 2500: pal[1], 8000: pal[2]}
    for key, vals in groups.items():
        x = xpos[key]
        for i, (l, d, fl) in enumerate(sorted(vals)):
            ax.scatter(
                x + (i - len(vals) / 2) * 0.07,
                d,
                s=42,
                color=lcol[l],
                zorder=3,
                edgecolor="white",
                linewidths=0.6,
            )
    ax.axvline(1.7, color="0.85", lw=1)
    ax.axhline(0.0, color="0.35", lw=1.0, ls="--")
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(
        [
            "generic map\n(context)",
            "half in-domain map\n(context)",
            "generic map\n(prefix)",
            "half in-domain map\n(prefix)",
        ],
        fontsize=9,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=lcol[l], label=f"L={l}")
        for l in [250, 2500, 8000]
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9)
    ax.set_ylabel("rho difference: map-then-project minus context-native")
    ax.set_title(
        "Evil composition cells: fitting the map on half in-domain contexts flips its sign\n(each dot = one cell, single draw and seed — preliminary)",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
        pad=10,
    )
    savefig_paper(fig, "compose_fu_flip", dir=OUT)
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def fig_map_quality() -> None:
    d = json.load(open(EVAL / "evil" / "map_diagnostics.json"))
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    pal = paper_palette(2)
    urungs = ["250", "5000", "full"]
    uxs = [250, 5000, 18793]
    for vi, variant in enumerate(["context_end", "prefix_end"]):
        r2s, knns, idbs = [], [], []
        for u in urungs:
            pls = d[f"{variant}|{u}"]["per_layer"]
            best = max(pls, key=lambda p: p["r2_map"])
            r2s.append(best["r2_map"])
            idbs.append(best["r2_identity_bias"])
            knns.append(best["knn"]["euclidean"]["acc_at_k"]["1"])
        lab = "context to answer" if variant == "context_end" else "prefix to answer"
        axes[0].plot(uxs, r2s, "-o", color=pal[vi], label=f"{lab} (map)")
        axes[0].plot(uxs, idbs, ":s", color=pal[vi], alpha=0.6, label=f"{lab} (identity+bias)")
        axes[1].plot(uxs, knns, "-o", color=pal[vi], label=lab)
        for x, y in zip(uxs, r2s):
            axes[0].annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                fontsize=8,
                ha="center",
            )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].axhline(0, color="0.4", lw=0.8)
    axes[0].set_xlabel("unlabeled map budget U (log scale)")
    axes[1].set_xlabel("unlabeled map budget U (log scale)")
    axes[0].set_ylabel("held-out R² (best layer)")
    axes[1].set_ylabel("kNN retrieval acc@1 (best layer)")
    axes[1].axhline(1 / 3759, color="0.5", lw=0.8, ls="--")
    axes[1].annotate(
        "chance (full-U pool)",
        (uxs[0], 1 / 3759),
        textcoords="offset points",
        xytext=(0, 5),
        fontsize=8,
    )
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Map quality on the shared unlabeled store: context maps learn, prefix maps do not",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "map_quality_ladder", dir=OUT)
    plt.close(fig)


# ---------------------------------------------------------------- CSV
def write_percell_csv(all_cells: dict[str, list[dict]]) -> None:
    out = OUT / "headline_deltas_percell.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "behavior",
                "regime",
                "variant",
                "u_rung_label",
                "budget_l",
                "draw",
                "seed",
                "f_u",
                "f_l",
                "delta_rho_frozen",
                "ci_lo_inherited",
                "ci_hi_inherited",
                "rho_arm6",
                "rho_arm2",
            ]
        )
        for beh in BEHS:
            for c in all_cells[beh]:
                k = cell_key(c)
                h = c["headline"]
                arms = {a["arm"]: a for a in c["arms"]}
                w.writerow(
                    [
                        beh,
                        k["regime"],
                        k["variant"],
                        k["u_rung_label"],
                        k["budget_l"],
                        k["draw"],
                        k["seed"],
                        k.get("f_u"),
                        k.get("f_l"),
                        h["delta_rho_frozen"],
                        h["ci_delta_selection_inherited"][0],
                        h["ci_delta_selection_inherited"][1],
                        arms["arm6_map_proj_e1"]["rho_frozen"],
                        arms["arm2_ctx_native"]["rho_frozen"],
                    ]
                )
    print(f"wrote {out}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_cells = {beh: load_cells(beh) for beh in BEHS}
    fig_hero_forest(all_cells)
    fig_arm_overview(all_cells)
    fig_scaling(all_cells)
    fig_shift_ladder()
    fig_compose(all_cells)
    fig_map_quality()
    write_percell_csv(all_cells)
    print("done")


if __name__ == "__main__":
    main()
