#!/usr/bin/env python3
"""Task #1491 clean-result figures: context→answer map predictability vs model scale.

Reads the committed per-rung fits JSONs + the analyzer round's cap-hit /
restriction digests + per-context CSVs (produced on the `issue-1491` worktree
by `issue1491_caphit_restriction_analysis.py`) and renders the clean-result
figure set under ``figures/issue_1491/`` via the paper-plots conventions.

    uv run python scripts/issue1491_scale_ladder_figures.py \
        --ladder-dir .claude/worktrees/issue-1491/eval_results/issue_1491/scale_ladder
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Before any heavy import, so the shared-VM thread caps (#847) bind in-process.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

SLUGS = ["scale05", "scale15", "scale3", "scale7_refit", "scale14", "scale32"]
LABELS = {
    "scale05": "0.5B",
    "scale15": "1.5B",
    "scale3": "3B",
    "scale7_refit": "7B",
    "scale14": "14B",
    "scale32": "32B",
}
PARAMS_B = {
    "scale05": 0.5,
    "scale15": 1.5,
    "scale3": 3.0,
    "scale7_refit": 7.0,
    "scale14": 14.0,
    "scale32": 32.0,
}

PAL = paper_palette_blog(6)
COL = {
    "ridge": PAL[0],
    "mlp": PAL[1],
    "krr": PAL[2],
    "wc": PAL[3],
    "restricted": PAL[4],
    "refit": PAL[5],
}
GRAY = "#9a9a9a"


def _load(ladder_dir: Path, percontext_dir: Path):
    """Load the per-rung fits (REQUIRED) plus the analyzer round's cap-hit /
    restriction digests and per-context CSVs (OPTIONAL).

    The digests + per-context CSVs are produced by
    ``issue1491_caphit_restriction_analysis.py``, which runs AFTER the fits. A
    second measurement arm (e.g. the greedy-decoding ladder) therefore has its
    fits committed well before that analysis exists, and used to crash here on
    FileNotFoundError. Missing optional inputs now degrade to "skip the
    dependent figure", reported by the caller — never a crash, and never a
    silently error-bar-less hero figure.

    Returns ``(fits, digests_or_None, percontext_or_None)``. The optional
    members are all-or-nothing across rungs: a PARTIAL set is treated as
    missing, because a figure covering only some rungs of the ladder would
    misrepresent the sweep.
    """
    fits, digests, percontext = {}, {}, {}
    for slug in SLUGS:
        fits[slug] = json.loads((ladder_dir / f"fits_{slug}.json").read_text())
        digest_path = ladder_dir / f"caphit_restriction_{slug}.json"
        if digest_path.exists():
            digests[slug] = json.loads(digest_path.read_text())
        percontext_path = percontext_dir / f"{slug}_percontext.csv"
        if percontext_path.exists():
            rows = list(csv.DictReader(open(percontext_path)))
            percontext[slug] = {
                "ci": np.array([int(r["ci"]) for r in rows]),
                "cos": np.array([float(r["cosine_pred_target"]) for r in rows]),
                "cap": np.array([int(r["cap_hit"]) for r in rows], dtype=bool),
            }
    return (
        fits,
        digests if len(digests) == len(SLUGS) else None,
        percontext if len(percontext) == len(SLUGS) else None,
    )


def _x(slugs=SLUGS):
    return np.array([PARAMS_B[s] for s in slugs])


def _xticks(ax):
    ax.set_xscale("log")
    ax.set_xticks([PARAMS_B[s] for s in SLUGS])
    ax.set_xticklabels([LABELS[s] for s in SLUGS])
    ax.minorticks_off()
    ax.set_xlabel("model size (Qwen2.5-Instruct, parameters)")


def _ridge_ci(digests, slug):
    b = digests[slug]["restriction"]["ridge_full"]["bootstrap"]
    r2 = digests[slug]["restriction"]["ridge_full"]["test_r2_refit"]
    lo, hi = b["ci95"]
    return r2, r2 - lo, hi - r2


def fig_hero(fits, digests, out, suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = _x()
    ridge = np.array([fits[s]["predictors"]["ridge"]["test_r2"] for s in SLUGS])
    mlp = np.array([fits[s]["predictors"]["mlp_w32768"]["test_r2"] for s in SLUGS])
    krr = np.array([fits[s]["predictors"]["krr_nystrom"]["test_r2"] for s in SLUGS])
    ceil = np.array([fits[s]["ceiling_two_draw"]["ceiling_var_weighted_r"] for s in SLUGS])
    null = np.array([fits[s]["floors"]["shuffled_pairing"]["test_r2"] for s in SLUGS])
    # Bootstrap CIs live in the restriction digests. Without them the series is
    # still correct, just uncertainty-free — the caller reports the omission so
    # it is never mistaken for "the estimate has no uncertainty".
    yerr = (
        np.array(
            [[_ridge_ci(digests, s)[1] for s in SLUGS], [_ridge_ci(digests, s)[2] for s in SLUGS]]
        )
        if digests is not None
        else None
    )

    ax = axes[0]
    ax.errorbar(
        x,
        ridge,
        yerr=yerr,
        color=COL["ridge"],
        marker="o",
        label="ridge (linear map)",
        capsize=3,
        markeredgewidth=1.0,
        zorder=4,
    )
    ax.plot(x, mlp, color=COL["mlp"], marker="s", label="MLP, width 32768")
    ax.plot(x, krr, color=COL["krr"], marker="^", label="kernel ridge (Nyström)")
    ax.plot(x, ceil, color=GRAY, linestyle="--", marker=".", label="two-draw reliability ceiling")
    ax.plot(x, null, color=GRAY, linestyle=":", marker=".", label="shuffled-pairing null")
    _xticks(ax)
    ax.set_ylabel("held-out test R² (variance-weighted)")
    ax.set_title("raw predictability and its ceiling", loc="left")
    ax.legend(fontsize=8, loc="center right")

    ax = axes[1]
    ax.errorbar(
        x,
        ridge / ceil,
        yerr=None if yerr is None else yerr / ceil,
        color=COL["ridge"],
        marker="o",
        label="ridge ÷ ceiling",
        capsize=3,
        markeredgewidth=1.0,
        zorder=4,
    )
    ax.plot(x, mlp / ceil, color=COL["mlp"], marker="s", label="MLP ÷ ceiling", alpha=0.7)
    ax.plot(x, krr / ceil, color=COL["krr"], marker="^", label="kernel ridge ÷ ceiling", alpha=0.7)
    _xticks(ax)
    ax.set_ylabel("test R² ÷ reliability ceiling")
    ax.set_title("ceiling-normalized predictability", loc="left")
    ax.set_ylim(0.55, 0.95)
    ax.legend(fontsize=8, loc="lower center")
    savefig_paper(fig, f"issue_1491/ladder_r2_raw_and_normalized{suffix}", dir=str(out))
    plt.close(fig)


def fig_hero_points(percontext, out, suffix=""):
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    rng = np.random.default_rng(0)
    for i, slug in enumerate(SLUGS):
        cos = percontext[slug]["cos"]
        xj = i + rng.uniform(-0.22, 0.22, size=len(cos))
        ax.scatter(xj, cos, s=4, alpha=0.10, color=COL["ridge"], linewidths=0)
        med = float(np.median(cos))
        ax.scatter(
            [i],
            [med],
            marker="D",
            s=42,
            color="#222222",
            zorder=5,
            linewidths=1.0,
            edgecolors="white",
        )
    ax.set_xticks(range(len(SLUGS)))
    ax.set_xticklabels([LABELS[s] for s in SLUGS])
    ax.set_xlabel("model size (Qwen2.5-Instruct, parameters)")
    # Kept short: a longer label overflows the axes height and renders clipped.
    ax.set_ylabel("cosine(predicted, actual answer)")
    ax.set_title(
        "per-context fit quality behind each aggregate R² (dark diamond = median)", loc="left"
    )
    savefig_paper(fig, f"issue_1491/ladder_r2_raw_and_normalized_points{suffix}", dir=str(out))
    plt.close(fig)


def fig_depth_pair(fits, digests, percontext, out, suffix=""):
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    fitters = ["ridge", "mlp_w32768", "krr_nystrom"]
    names = ["ridge", "MLP w32768", "kernel ridge"]
    w = 0.35
    xs = np.arange(3)
    v14 = [fits["scale14"]["predictors"][f]["test_r2"] for f in fitters]
    v32 = [fits["scale32"]["predictors"][f]["test_r2"] for f in fitters]
    # Same contract as fig_hero: the bootstrap CIs live in the restriction
    # digests, so an arm whose digests have not been produced yet plots the
    # bars uncertainty-free rather than crashing. The caller reports it.
    if digests is None:
        yerr14 = yerr32 = None
    else:
        _, lo14, hi14 = _ridge_ci(digests, "scale14")
        _, lo32, hi32 = _ridge_ci(digests, "scale32")
        yerr14 = np.zeros((2, 3))
        yerr14[:, 0] = [lo14, hi14]
        yerr32 = np.zeros((2, 3))
        yerr32[:, 0] = [lo32, hi32]
    ax.bar(xs - w / 2, v14, w, yerr=yerr14, capsize=3, label="14B (48 layers)", color=PAL[0])
    ax.bar(xs + w / 2, v32, w, yerr=yerr32, capsize=3, label="32B (64 layers)", color=PAL[3])
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("held-out test R² (variance-weighted)")
    # Bars start at zero: a truncated baseline visually exaggerates the 14B-vs-32B
    # gap, and this pair is a headline result. The paired scatter (right panel)
    # carries the per-context resolution a zoomed axis would have supplied.
    ax.set_ylim(0.0, 0.8)
    ax.set_title("same width (5120 dims), more depth", loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    p14, p32 = percontext["scale14"], percontext["scale32"]
    common = {c: i for i, c in enumerate(p14["ci"])}
    idx14, idx32 = [], []
    for j, c in enumerate(p32["ci"]):
        i = common.get(int(c))
        if i is not None:
            idx14.append(i)
            idx32.append(j)
    a = p14["cos"][idx14]
    b = p32["cos"][idx32]
    lims = [min(a.min(), b.min()) - 0.02, 1.0]
    ax.plot(lims, lims, color=GRAY, linestyle="--", linewidth=1.0)
    ax.scatter(a, b, s=6, alpha=0.25, color=COL["ridge"], linewidths=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("per-context cosine, 14B map")
    ax.set_ylabel("per-context cosine, 32B map")
    ax.set_title(
        "same contexts, paired (n = %d; below diagonal = worse at 32B)" % len(a), loc="left"
    )
    savefig_paper(fig, f"issue_1491/depth_pair_fixed_width{suffix}", dir=str(out))
    plt.close(fig)


def fig_caphit_restriction(digests, out, suffix=""):
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = _x()
    ax = axes[0]
    minor = ["val_400", "tierB_3600", "ceiling_draws/seed43", "ceiling_draws/seed44"]
    for sp in minor:
        y = [100 * digests[s]["caphit_by_split"][sp]["cap_hit_rate"] for s in SLUGS]
        ax.plot(x, y, color=GRAY, alpha=0.45, linewidth=1.0)
    ax.plot(
        [],
        [],
        color=GRAY,
        alpha=0.45,
        linewidth=1.0,
        label="other splits (validation, layer-sweep, ceiling draws)",
    )
    for sp, key, mk in [
        ("train_25k", "ridge", "o"),
        ("test_1000", "mlp", "s"),
        ("wc_test_1k", "wc", "^"),
    ]:
        y = [100 * digests[s]["caphit_by_split"][sp]["cap_hit_rate"] for s in SLUGS]
        nice = {
            "train_25k": "train (25,000 rows)",
            "test_1000": "test (1,000 rows)",
            "wc_test_1k": "WildChat test (999 rows)",
        }[sp]
        ax.plot(x, y, color=COL[key], marker=mk, label=nice)
    _xticks(ax)
    ax.set_ylabel("generations hitting the 1024-token cap (%)")
    ax.set_title("cap-hit rate per split (from finish_reason)", loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    full = [digests[s]["restriction"]["ridge_full"]["test_r2_refit"] for s in SLUGS]
    ev = [digests[s]["restriction"]["ridge_eval_noncaphit"]["test_r2"] for s in SLUGS]
    rf = [digests[s]["restriction"]["ridge_refit_noncaphit"]["test_r2"] for s in SLUGS]
    yerr_full = np.array(
        [[_ridge_ci(digests, s)[1] for s in SLUGS], [_ridge_ci(digests, s)[2] for s in SLUGS]]
    )
    ev_b = [digests[s]["restriction"]["ridge_eval_noncaphit"]["bootstrap"]["ci95"] for s in SLUGS]
    yerr_ev = np.array(
        [[ev[i] - ev_b[i][0] for i in range(6)], [ev_b[i][1] - ev[i] for i in range(6)]]
    )
    ax.errorbar(
        x, full, yerr=yerr_full, color=COL["ridge"], marker="o", label="all rows", capsize=3
    )
    ax.errorbar(
        x,
        ev,
        yerr=yerr_ev,
        color=COL["restricted"],
        marker="s",
        label="untruncated test rows only",
        capsize=3,
    )
    ax.plot(
        x,
        rf,
        color=COL["refit"],
        marker="^",
        linestyle="--",
        label="refit + evaluated on untruncated rows",
    )
    _xticks(ax)
    ax.set_ylabel("ridge held-out test R²")
    ax.set_title("scale trend restricted to untruncated generations", loc="left")
    ax.legend(fontsize=8)
    savefig_paper(fig, f"issue_1491/caphit_and_restriction{suffix}", dir=str(out))
    plt.close(fig)


def fig_wc_transfer(fits, out, suffix=""):
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = _x()
    lm = [fits[s]["predictors"]["ridge"]["test_r2"] for s in SLUGS]
    wc = [fits[s]["wc_transfer"]["ridge_test_r2"] for s in SLUGS]
    ax.plot(x, lm, color=COL["ridge"], marker="o", label="LMSYS test (in-distribution)")
    ax.plot(x[:-1], wc[:-1], color=COL["wc"], marker="s", label="WildChat test (corpus transfer)")
    ax.scatter(
        [x[-1]],
        [wc[-1]],
        facecolors="none",
        edgecolors=COL["wc"],
        marker="s",
        s=64,
        linewidths=1.4,
        label="32B WildChat point (mixed capture batch sizes 8/4/2)",
        zorder=5,
    )
    ax.plot(x[-2:], wc[-2:], color=COL["wc"], linestyle=":")
    _xticks(ax)
    ax.set_ylabel("ridge held-out test R² (variance-weighted)")
    ax.set_title("in-distribution vs corpus-transfer predictability", loc="left")
    ax.legend(fontsize=8)
    savefig_paper(fig, f"issue_1491/wc_transfer_ladder{suffix}", dir=str(out))
    plt.close(fig)


def fig_floors_retrieval(fits, out, suffix=""):
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = _x()
    ax = axes[0]
    floor_keys = [
        ("shuffled_pairing", "shuffled-pairing null", GRAY, ":"),
        ("train_mean", "train-mean predictor (degenerate prefix arm)", PAL[1], "--"),
        ("scaled_identity", "per-dimension scaled identity", PAL[2], "-"),
        ("identity_bias", "identity + learned bias", PAL[3], "-"),
        ("identity_copy", "identity copy", PAL[4], "-"),
    ]
    for key, name, col, ls in floor_keys:
        y = [fits[s]["floors"][key]["test_r2"] for s in SLUGS]
        ax.plot(x, y, color=col, linestyle=ls, marker=".", label=name)
    ridge = [fits[s]["predictors"]["ridge"]["test_r2"] for s in SLUGS]
    ax.plot(x, ridge, color=COL["ridge"], marker="o", label="ridge (fitted map)")
    _xticks(ax)
    ax.set_ylabel("held-out test R² (variance-weighted)")
    ax.set_title("fitted map vs identity-family and null floors", loc="left")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    for arm, name, col, mk in [
        ("ridge", "ridge prediction", COL["ridge"], "o"),
        ("identity_bias", "identity + learned bias", PAL[3], "s"),
    ]:
        y = [fits[s]["knn_retrieval"][arm]["cosine"]["acc_at_k"]["1"] for s in SLUGS]
        ax.plot(x, y, color=col, marker=mk, label=name)
    ax.axhline(0.001, color=GRAY, linestyle=":", linewidth=1.0)
    ax.plot([], [], color=GRAY, linestyle=":", linewidth=1.0, label="chance (1 of 1,000)")
    _xticks(ax)
    # Kept short (was clipping); pool size stays in the label since chance = 1/1,000.
    ax.set_ylabel("retrieval acc@1 (cosine; pool = 1,000)")
    ax.set_title("does the prediction find the right answer vector?", loc="left")
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=8)
    savefig_paper(fig, f"issue_1491/floors_and_retrieval{suffix}", dir=str(out))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    # Defaults to <ladder-dir>/percontext — the durable committed home for the
    # per-context CSVs (data/ is gitignored, so a data/ copy is not reproducible).
    ap.add_argument("--percontext-dir", type=Path, default=None)
    # Appended to every figure stem so a second measurement arm (the greedy
    # ladder) writes beside the sampled arm's figures instead of overwriting
    # them. Empty by default: the sampled arm's stems are unchanged.
    ap.add_argument("--stem-suffix", default="")
    args = ap.parse_args()
    percontext_dir = args.percontext_dir or (args.ladder_dir / "percontext")
    sfx = args.stem_suffix
    set_paper_style("blog")
    fits, digests, percontext = _load(args.ladder_dir, percontext_dir)

    # Each figure declares the OPTIONAL inputs it cannot render without. A
    # figure whose inputs are absent is skipped and named at the end — never
    # rendered from a partial set, and never silently dropped.
    skipped: list[str] = []
    fig_hero(fits, digests, args.out, suffix=sfx)
    if percontext is None:
        skipped.append("ladder_r2_raw_and_normalized_points (needs per-context CSVs)")
        skipped.append("depth_pair_fixed_width (needs per-context CSVs)")
    else:
        fig_hero_points(percontext, args.out, suffix=sfx)
        fig_depth_pair(fits, digests, percontext, args.out, suffix=sfx)
    if digests is None:
        skipped.append("caphit_and_restriction (needs cap-hit/restriction digests)")
    else:
        fig_caphit_restriction(digests, args.out, suffix=sfx)
    fig_wc_transfer(fits, args.out, suffix=sfx)
    fig_floors_retrieval(fits, args.out, suffix=sfx)

    print("figures written under", args.out / "issue_1491")
    if digests is None:
        print("NOTE: bootstrap CIs unavailable (no restriction digests) — error bars omitted")
    for name in skipped:
        print("SKIPPED:", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
