"""Figures for issue #1090 fu3 (posonly-contexts-parallel-matrix round).

Reads the committed fu3 aggregates (eval_results/issue_1090/fu3/) plus the
per-cell margin records (data/issue_1090/fu3/*/margin.json) and the failed
broad_em datagen judge records, and writes five blog-style figures to
figures/issue_1090/. Every figure uses plain-English condition names.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): apply env caps BEFORE torch/numpy freeze their pools.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
FU3 = ROOT / "eval_results" / "issue_1090" / "fu3"
DATA = ROOT / "data" / "issue_1090" / "fu3"
OUT = "issue_1090"

CTX_LABEL = {
    "persona_software_engineer": "persona (software engineer)",
    "default": "bare assistant",
    "wildchat_prefix_real545": "WildChat prefix",
    "icl_prefix_sycophancy": "ICL prefix",
    "icl_prefix_impolite": "ICL prefix",
    "icl_prefix_formatting": "ICL prefix",
    "icl_prefix_broad_em": "ICL prefix",
    "neg_sp_police": "police officer persona",
    "neg_sp_ph4": "tech-support persona",
}
BEH_LABEL = {
    "formatting": "list formatting (control)",
    "impolite": "impolite",
    "sycophancy": "sycophancy",
}
CTX_SHORT = {
    "pers": "persona",
    "bare": "bare assistant",
    "conv": "WildChat prefix",
    "icl": "ICL prefix",
}
CTX_ORDER = ["pers", "bare", "conv", "icl"]


def _load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def _pair_label(pair: str, behavior: str) -> str:
    cell, ctx = pair.split("-")
    beh = BEH_LABEL[behavior]
    if cell == "C5":
        beh = "sycophancy (Qwen data)"
    elif behavior == "sycophancy":
        beh = "sycophancy (Claude data)"
    return f"{beh} · {CTX_SHORT[ctx]}"


def fig_leakage_matched_pairs() -> None:
    """Hero: contrastive vs positive-only held-out leakage per behavior x context pair."""
    lk = _load_json(FU3 / "fu3_leakage_contrast.json")
    cells = {
        c["cell_id"]: c
        for c in [_load_json(p) for p in sorted((FU3 / "fu3_cell_evals").glob("*.json"))]
    }
    pairs = lk["pairs"]
    # order: behavior blocks, matched pairs flagged
    order = sorted(
        pairs,
        key=lambda p: (["formatting", "impolite", "sycophancy"].index(p["behavior"]), p["pair"]),
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    cols = paper_palette_blog(3)
    ys = np.arange(len(order))[::-1]
    for y, p in zip(ys, order, strict=True):
        con, pos = p["leak_contrastive"], p["leak_posonly"]
        # SE over the 5 held-out per-context deltas (spread across bystander contexts)
        for regime, val, filled in (("con", con, True), ("pos", pos, False)):
            cid = f"{p['pair']}-{regime}"
            rec = cells.get(cid)
            deltas = [b["leak_delta"] for b in rec["bystanders"] if not b["is_source_context"]]
            se = float(np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
            color = cols[0] if filled else cols[1]
            ax.errorbar(
                val,
                y + (0.14 if filled else -0.14),
                xerr=se,
                fmt="o",
                color=color,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                markersize=7,
                elinewidth=1.1,
                capsize=2.5,
                zorder=3,
            )
        lab = _pair_label(p["pair"], p["behavior"])
        if p["install_matched"]:
            lab += "  [matched install]"
        ax.text(-0.245, y, lab, ha="right", va="center", fontsize=9)
    ax.axvline(0.0, color="0.6", lw=0.8, zorder=1)
    ax.set_yticks([])
    ax.set_xlim(-0.25, 0.25)
    ax.set_xlabel("bystander leakage: judged rate, trained - base, mean over 5 held-out contexts")
    ax.set_title("Held-out bystander leakage per behavior x training-context pair", loc="left")
    handles = [
        plt.Line2D(
            [], [], marker="o", ls="none", color=cols[0], label="contrastive (5-member panel)"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markerfacecolor="white",
            markeredgecolor=cols[1],
            markeredgewidth=1.4,
            color=cols[1],
            label="positive-only (no negatives)",
        ),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.subplots_adjust(left=0.42)
    savefig_paper(fig, f"{OUT}/fu3_leakage_matched_pairs", dir="figures/")
    plt.close(fig)


def fig_leakage_per_context_points() -> None:
    """Raw sibling of the hero: every per-bystander-context leakage delta."""
    cells = [_load_json(p) for p in sorted((FU3 / "fu3_cell_evals").glob("*.json"))]
    beh_panels = ["formatting", "impolite", "sycophancy"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), sharey=True)
    cols = paper_palette_blog(5)
    train_ctx_color = {"pers": cols[0], "bare": cols[1], "conv": cols[2], "icl": cols[3]}
    for ax, beh in zip(axes, beh_panels, strict=True):
        recs = [c for c in cells if c["behavior"] == beh]
        xcats = [
            "persona (software engineer)",
            "bare assistant",
            "WildChat prefix",
            "ICL prefix",
            "police officer persona",
            "tech-support persona",
        ]
        xmap = {k: i for i, k in enumerate(xcats)}
        for rec in recs:
            tctx = rec["cell_id"].split("-")[1]
            qwen = rec["cell_id"].startswith("C5")
            color = "0.35" if qwen else train_ctx_color[tctx]
            filled = rec["regime"] == "contrastive"
            for b in rec["bystanders"]:
                if b["is_source_context"]:
                    continue
                x = xmap[CTX_LABEL[b["context_id"]]] + (0.13 if filled else -0.13)
                ax.scatter(
                    x,
                    b["leak_delta"],
                    s=26,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    linewidths=1.2,
                    zorder=3,
                )
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
        ax.set_xticks(range(len(xcats)))
        ax.set_xticklabels(
            [
                c.replace(" (software engineer)", "\n(software engineer)").replace(
                    " persona", "\npersona"
                )
                for c in xcats
            ],
            fontsize=7.5,
            rotation=30,
            ha="right",
        )
        ax.set_title(BEH_LABEL[beh], loc="left", fontsize=10)
    axes[0].set_ylabel("leakage: judged rate, trained - base\n(per bystander context, n=100 each)")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color=train_ctx_color[c],
            label=f"trained in {CTX_SHORT[c]}",
        )
        for c in CTX_ORDER
    ] + [
        plt.Line2D([], [], marker="o", ls="none", color="0.35", label="Qwen-data organism"),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markerfacecolor="white",
            markeredgecolor="0.3",
            markeredgewidth=1.2,
            color="0.3",
            label="open = positive-only",
        ),
    ]
    axes[2].legend(handles=handles, loc="upper right", fontsize=7)
    fig.suptitle("Per-context leakage deltas behind the pair means", x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(bottom=0.28, top=0.86)
    savefig_paper(fig, f"{OUT}/fu3_leakage_per_context_points", dir="figures/")
    plt.close(fig)


def _newcombe(p1, l1, u1, p2, l2, u2):
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def fig_install_by_context() -> None:
    """Own-context install delta (trained - base) per behavior x context x regime."""
    cells = {
        c["cell_id"]: c
        for c in [_load_json(p) for p in sorted((FU3 / "fu3_cell_evals").glob("*.json"))]
    }
    panels = [
        ("list formatting (control) — structural rate", "C1", "formatting"),
        ("impolite — judged rate", "C2", "impolite"),
        ("sycophancy — judged rate", "C3", "sycophancy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    cols = paper_palette_blog(3)
    for ax, (title, prefix, _beh) in zip(axes, panels, strict=True):
        xt = []
        for i, ctx in enumerate(CTX_ORDER):
            for j, regime in enumerate(("con", "pos")):
                cid = f"{prefix}-{ctx}-{regime}"
                rec = cells.get(cid)
                x = i + (j - 0.5) * 0.36
                if rec is None:
                    ax.text(x, 0.02, "not\ntrained", ha="center", fontsize=6.5, color="0.45")
                    continue
                t, b = rec["tier2"]["trained"], rec["tier2"]["base"]
                d = t["rate"] - b["rate"]
                lo, hi = _newcombe(t["rate"], *t["wilson95"], b["rate"], *b["wilson95"])
                color = cols[0] if regime == "con" else cols[1]
                ax.bar(x, d, width=0.32, color=color, alpha=1.0 if regime == "con" else 0.55)
                ax.errorbar(
                    x,
                    d,
                    yerr=[[d - lo], [hi - d]],
                    fmt="none",
                    ecolor="0.25",
                    elinewidth=1.0,
                    capsize=2.5,
                    zorder=4,
                )
            xt.append(CTX_SHORT[ctx])
        # Qwen-data cells share the sycophancy panel at an offset slot
        if prefix == "C3":
            for j, regime in enumerate(("con", "pos")):
                rec = cells.get(f"C5-pers-{regime}")
                x = 4 + (j - 0.5) * 0.36
                t, b = rec["tier2"]["trained"], rec["tier2"]["base"]
                d = t["rate"] - b["rate"]
                lo, hi = _newcombe(t["rate"], *t["wilson95"], b["rate"], *b["wilson95"])
                color = cols[0] if regime == "con" else cols[1]
                ax.bar(x, d, width=0.32, color=color, alpha=1.0 if regime == "con" else 0.55)
                ax.errorbar(
                    x,
                    d,
                    yerr=[[d - lo], [hi - d]],
                    fmt="none",
                    ecolor="0.25",
                    elinewidth=1.0,
                    capsize=2.5,
                    zorder=4,
                )
            xt.append("persona\n(Qwen data)")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(xt)))
        ax.set_xticklabels(xt, fontsize=8, rotation=20, ha="right")
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel(
        "own-context install: rate, trained - base\n(95% interval; n=200-300 per arm)"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cols[0], label="contrastive"),
        plt.Rectangle((0, 0), 1, 1, color=cols[1], alpha=0.55, label="positive-only"),
    ]
    axes[2].legend(handles=handles, loc="upper left", fontsize=8)
    fig.suptitle(
        "Own-context install by training context and regime", x=0.01, ha="left", fontsize=12
    )
    fig.subplots_adjust(bottom=0.2, top=0.84)
    savefig_paper(fig, f"{OUT}/fu3_install_by_context", dir="figures/")
    plt.close(fig)


def fig_margin_by_context() -> None:
    """Teacher-forced fixed-pool margin delta by context x regime, per-item strips."""
    rows = []
    for p in sorted(DATA.glob("*/margin.json")):
        d = _load_json(p)
        if "cells" not in d:
            continue
        slug = p.parent.name
        parts = slug.split("-")
        cell, ctx, regime = parts[0], parts[1], parts[2]
        key = next(k for k in d["cells"] if k.startswith("base__"))
        src = key[len("base__") :]
        mb, mt = d["cells"][f"base__{src}"], d["cells"][f"trained__{src}"]
        # per-item contributions to the margin delta
        pos_d = np.array(mt["pos_ln_logp"]) - np.array(mb["pos_ln_logp"])
        neg_d = np.array(mt["neg_ln_logp"]) - np.array(mb["neg_ln_logp"])
        rows.append(
            dict(
                cell=cell,
                ctx=ctx,
                regime=regime,
                delta=mt["margin"] - mb["margin"],
                pos_d=pos_d,
                neg_d=neg_d,
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    cols = paper_palette_blog(3)
    rng = np.random.default_rng(42)
    for ax, beh_cell, title in zip(
        axes, ("C2", "C3"), ("impolite", "sycophancy (Claude data; Qwen at right)"), strict=True
    ):
        sub = [r for r in rows if r["cell"] == beh_cell]
        if beh_cell == "C3":
            sub += [r for r in rows if r["cell"] == "C5"]
        xt = []
        xi = 0
        for ctx in CTX_ORDER:
            group = [r for r in sub if r["ctx"] == ctx and r["cell"] != "C5"]
            if not group:
                continue
            for r in sorted(group, key=lambda r: r["regime"]):
                x = xi + (-0.18 if r["regime"] == "con" else 0.18)
                color = cols[0] if r["regime"] == "con" else cols[1]
                ax.bar(
                    x,
                    r["delta"],
                    width=0.3,
                    color=color,
                    alpha=1.0 if r["regime"] == "con" else 0.55,
                    zorder=2,
                )
                item_d = np.concatenate([r["pos_d"], -r["neg_d"]])
                jit = rng.uniform(-0.09, 0.09, size=item_d.size)
                ax.scatter(
                    np.full(item_d.size, x) + jit,
                    item_d,
                    s=7,
                    facecolors="none",
                    edgecolors="0.35",
                    linewidths=0.7,
                    alpha=0.75,
                    zorder=3,
                )
            xt.append(CTX_SHORT[ctx])
            xi += 1
        if beh_cell == "C3":
            for r in sorted([r for r in sub if r["cell"] == "C5"], key=lambda r: r["regime"]):
                x = xi + (-0.18 if r["regime"] == "con" else 0.18)
                color = cols[0] if r["regime"] == "con" else cols[1]
                ax.bar(
                    x,
                    r["delta"],
                    width=0.3,
                    color=color,
                    alpha=1.0 if r["regime"] == "con" else 0.55,
                    zorder=2,
                )
                item_d = np.concatenate([r["pos_d"], -r["neg_d"]])
                jit = rng.uniform(-0.09, 0.09, size=item_d.size)
                ax.scatter(
                    np.full(item_d.size, x) + jit,
                    item_d,
                    s=7,
                    facecolors="none",
                    edgecolors="0.35",
                    linewidths=0.7,
                    alpha=0.75,
                    zorder=3,
                )
            xt.append("persona\n(Qwen data)")
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
        ax.set_xticks(range(len(xt)))
        ax.set_xticklabels(xt, fontsize=8.5, rotation=15, ha="right")
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel(
        "fixed-pool margin, trained - base\n(bars; points = per-item contributions, n=48-50)"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cols[0], label="contrastive"),
        plt.Rectangle((0, 0), 1, 1, color=cols[1], alpha=0.55, label="positive-only"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)
    fig.suptitle(
        "Teacher-forced margin install by training context (own context)",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.subplots_adjust(bottom=0.17, top=0.85)
    savefig_paper(fig, f"{OUT}/fu3_margin_by_context", dir="figures/")
    plt.close(fig)


def fig_c6_yield() -> None:
    """broad_em datagen kept counts across all 8 context x regime cells."""
    rows = []
    for p in sorted(DATA.glob("C6-*/datagen/judge_raw_pos.json")):
        slug = p.parent.parent.name
        d = _load_json(p)
        per = defaultdict(list)
        for k, v in d["all_scores"].items():
            s = v.get("score")
            if isinstance(s, (int, float)):
                per[k.split("__")[0]].append(s)
        means = {c: sum(v) / len(v) for c, v in per.items() if v}
        kept = sum(1 for m in means.values() if m > 50)
        parts = slug.split("-")
        rows.append(
            (
                f"{CTX_SHORT[parts[1]]}\n{'contrastive' if parts[2] == 'con' else 'positive-only'}",
                kept,
                len(per),
            )
        )
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    cols = paper_palette_blog(2)
    xs = np.arange(len(rows))
    ax.bar(xs, [r[1] for r in rows], color=cols[0], width=0.6)
    for x, (_lab, kept, n) in zip(xs, rows, strict=True):
        ax.text(x, kept + 0.5, f"{kept}/{n}", ha="center", fontsize=8)
    ax.axhline(20, color=cols[1], lw=1.2, ls="--")
    ax.text(
        len(rows) - 0.4, 20.6, "floor: 20 kept positives", ha="right", fontsize=8, color=cols[1]
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("judge-kept positives (of candidates generated)")
    ax.set_title("Broad-misalignment datagen keeps across all 8 cells", loc="left")
    savefig_paper(fig, f"{OUT}/fu3_c6_yield", dir="figures/")
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    fig_leakage_matched_pairs()
    fig_leakage_per_context_points()
    fig_install_by_context()
    fig_margin_by_context()
    fig_c6_yield()
    print("wrote 5 figures to figures/issue_1090/")


if __name__ == "__main__":
    main()
