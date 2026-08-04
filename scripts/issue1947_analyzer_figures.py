"""Analyzer round-1 paper-quality figures for task #1947 (single-visit organism fleet).

Reads ONLY committed digests under eval_results/issue_1947/analysis/ (the
analyzer_round1/ digests are mechanical reductions of the committed battery /
judge / fit JSONs) and renders the round-1 figure set under figures/issue_1947/.

Run from the issue-1947 worktree root:
    uv run python scripts/issue1947_analyzer_figures.py
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # VM thread-cap setdefaults BEFORE the heavy imports (#847)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

ROOT = Path(".")
AN = ROOT / "eval_results/issue_1947/analysis"
A1 = AN / "analyzer_round1"
FIGDIR = ROOT / "figures/issue_1947"

C_OP = "#4477AA"  # on-policy tree
C_MT = "#EE7733"  # matched-text (teacher-forced) tree
C_1768 = "#555555"  # parent #1768 reference
C_BAND = "#CCEECC"  # theory band / install band


def load() -> tuple[list[dict], dict, dict]:
    digest = json.load(open(A1 / "battery_digest.json"))
    mani = json.load(open(AN / "verdict_manifest.json"))
    ladders = json.load(open(A1 / "ladders_rates_by_step.json"))
    return digest, mani, ladders


def content_rows(digest: list[dict]) -> list[dict]:
    return [r for r in digest if not r["slug"].startswith("mk-") and not r.get("missing")]


SLUG_KEY = (
    "Key: cas / imp = casual writing style / impolite; bare / pers / conv / icl = bare assistant / "
    "software-engineer persona / conversation prefix / in-context-learning prefix; "
    "con / po = contrastive / positive-only; s42 / s137 = seed; REP = repeat-regime control (80 rows x 15 epochs)"
)


def add_slug_key(fig: plt.Figure, y: float = 0.002) -> None:
    """Render the in-figure decode key for per-cell config slugs (clean-result-critic r1 item 2)."""
    fig.text(0.5, y, SLUG_KEY, ha="center", va="bottom", fontsize=7, color="#555555")


def fig_hero(digest: list[dict], mani: dict) -> None:
    rows = content_rows(digest)
    inband = {s for s, v in mani["content"].items() if v["selection"]["in_band"]}
    groups = [(t, L) for t in ("onpolicy", "matched_text") for L in (14, 19, 25)]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    def strip(ax, vals_fn, ylabel, crit=None, band=None, ref=None):
        rng = np.random.default_rng(42)
        for gi, (tree, L) in enumerate(groups):
            sub = [r for r in rows if r["tree"] == tree and r["L"] == L]
            vals = [(vals_fn(r), r["slug"] in inband) for r in sub]
            color = C_OP if tree == "onpolicy" else C_MT
            for v, ib in vals:
                if v is None:
                    continue
                x = gi + rng.uniform(-0.18, 0.18)
                if ib:
                    ax.scatter(x, v, s=22, color=color, alpha=0.75, zorder=3)
                else:
                    ax.scatter(
                        x,
                        v,
                        s=22,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.2,
                        alpha=0.85,
                        zorder=3,
                    )
            med = np.median([v for v, _ in vals if v is not None])
            ax.hlines(med, gi - 0.3, gi + 0.3, color="black", linewidth=2.2, zorder=4)
        if band is not None:
            ax.axhspan(band[0], band[1], color=C_BAND, alpha=0.6, zorder=0)
        if crit is not None:
            ax.axhline(crit, color="crimson", linestyle="--", linewidth=1.2)
        if ref is not None:
            for tree_i, y in ref.items():
                x0 = 0 if tree_i == "onpolicy" else 3
                ax.hlines(y, x0 - 0.35, x0 + 2.35, color=C_1768, linestyle=":", linewidth=1.6)
        ax.set_xticks(range(6))
        ax.set_xticklabels(
            [
                "on-policy\nL14",
                "on-policy\nL19",
                "on-policy\nL25",
                "fixed text\nL14",
                "fixed text\nL19",
                "fixed text\nL25",
            ],
            fontsize=9,
        )
        ax.set_ylabel(ylabel)

    strip(
        axes[0],
        lambda r: r["top1"],
        "top-1 singular share of per-row answer-shift stack",
        crit=0.6,
        ref={"onpolicy": 0.086, "matched_text": 0.283},
    )
    axes[0].set_title("Write rank (share < 0.6 = high-rank write)", fontsize=11)
    strip(
        axes[1],
        lambda r: (r["delta"] or {}).get("cos"),
        "cos(write direction, activation displacement)",
    )
    # median corpus-covariance null p95 per group, drawn as grey band markers
    for gi, (tree, L) in enumerate(groups):
        sub = [r for r in rows if r["tree"] == tree and r["L"] == L and r.get("delta")]
        m = np.median([r["delta"]["cov95"] for r in sub])
        axes[1].hlines([m, -m], gi - 0.3, gi + 0.3, color="grey", linestyle="--", linewidth=1.0)
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_title("Write-displacement alignment vs covariance null (dashes)", fontsize=11)
    strip(
        axes[2],
        lambda r: r["gate_lp"],
        "gate rank correlation (last-prompt-token read)",
        band=(0.3, 0.7),
    )
    axes[2].axhline(0, color="grey", linewidth=0.8)
    axes[2].set_title("Base-geometry gate vs 0.3–0.7 theory band", fontsize=11)
    handles = [
        plt.Line2D(
            [], [], marker="o", color=C_OP, linestyle="", label="on-policy tree (n=300 rows)"
        ),
        plt.Line2D(
            [], [], marker="o", color=C_MT, linestyle="", label="fixed-text TF tree (n=1,200 rows)"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color="grey",
            linestyle="",
            markerfacecolor="none",
            label="closest-approach cell (open)",
        ),
        plt.Line2D(
            [], [], color=C_1768, linestyle=":", label="#1768 corpus median (protocol differs)"
        ),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.text(
        0.5,
        0.99,
        "Theory-assumption battery on exactly the trained rows — 34 content cells, per layer",
        ha="center",
        va="top",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig_paper(fig, "battery_assumptions_trained_rows", dir=FIGDIR)
    plt.close(fig)


def fig_d_forest() -> None:
    rows = json.load(open(A1 / "h5_concordance.json"))
    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    ylabels = []
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ylabels.append((y, f"{r['arm'].replace('-con-sv-s42', '')} L{r['layer']}"))
        d, ci = r["D_1947"], r["D_ci95_1947"]
        ax.errorbar(
            d,
            y + 0.15,
            xerr=[[d - ci[0]], [ci[1] - d]],
            fmt="o",
            color=C_OP,
            markersize=5,
            capsize=2,
            label="_",
        )
        if r["D_1768"] is not None:
            pd_, pci = r["D_1768"], r["D_ci95_1768"]
            ax.errorbar(
                pd_,
                y - 0.15,
                xerr=[[pd_ - pci[0]], [pci[1] - pd_]],
                fmt="s",
                color=C_1768,
                markersize=4.5,
                capsize=2,
                label="_",
            )
    ax.axvline(0, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_yticks([y for y, _ in ylabels])
    ax.set_yticklabels([lab for _, lab in ylabels], fontsize=8.5)
    ax.set_xlabel("map-change D (excess over base-map refit-noise p95 floor), span-mean read")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color=C_OP,
            linestyle="",
            label="#1947 single-visit (bootstrap 95% CI)",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            color=C_1768,
            linestyle="",
            label="#1768 parent, same arm+LR (95% CI)",
        ),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    ax.set_title("Corpus map-change D: single-visit vs parent fleet (n=3,000 fits)", fontsize=12)
    fig.tight_layout()
    savefig_paper(fig, "d_forest_single_visit_vs_parent", dir=FIGDIR)
    plt.close(fig)


def fig_ladders(mani: dict, ladders: dict) -> None:
    slugs = sorted(ladders.keys())
    ncol = 6
    nrow = int(np.ceil(len(slugs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.1 * nrow), sharex=True, sharey=True)
    for ax in axes.flat[len(slugs) :]:
        ax.axis("off")
    for ax, slug in zip(axes.flat, slugs):
        steps = sorted(ladders[slug].keys(), key=int)
        rates = [ladders[slug][s] for s in steps]
        xs = [int(s) for s in steps]
        ax.axhspan(0.60, 0.85, color=C_BAND, alpha=0.7, zorder=0)
        ax.plot(xs, rates, "-o", color=C_OP, markersize=2.5, linewidth=1.0)
        sel = mani["content"][slug]["selection"]
        ax.scatter(
            [sel["step"]],
            [sel["rate"]],
            s=48,
            zorder=5,
            facecolors=("#228833" if sel["in_band"] else "none"),
            edgecolors=("#228833" if sel["in_band"] else "crimson"),
            linewidths=1.4,
        )
        ax.set_title(slug.replace("-sv-", "·").replace("-rep-", "·REP·"), fontsize=7.5)
        ax.set_ylim(-0.05, 1.05)
    fig.text(
        0.5,
        0.995,
        "Judged install ladders (rate vs optimizer step; green band = 0.60–0.85; "
        "dot = verdict rung, filled green = in-band, open red = closest-approach)",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.024,
        "optimizer step (16 fresh rows per step under single-visit)",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    add_slug_key(fig)
    savefig_paper(fig, "install_ladders_per_cell", dir=FIGDIR)
    plt.close(fig)


def fig_reliability(digest: list[dict]) -> None:
    # All three layers plotted per arm. The v1 render silently kept only the
    # first layer encountered per arm (L14) while the prose claimed the
    # all-layer range 0.732-0.936 (interp-critique v1 finding 6).
    rows = content_rows(digest)
    per_arm_layer: dict[tuple[str, int], float] = {}
    for r in rows:
        if r.get("shr") is not None:
            per_arm_layer.setdefault((r["slug"], r["L"]), r["shr"])
    arms = sorted({a for a, _ in per_arm_layer})
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    xs = np.arange(len(arms))
    layer_markers = {14: "o", 19: "^", 25: "s"}
    for layer, mk in layer_markers.items():
        colors = [C_MT if "-rep-" in a else C_OP for a in arms]
        vals = [per_arm_layer.get((a, layer), np.nan) for a in arms]
        ax.scatter(xs, vals, s=26, c=colors, marker=mk, zorder=3, alpha=0.85)
    ax.axhline(0.55, color=C_1768, linestyle=":", linewidth=1.6)
    ax.text(
        len(arms) - 0.5,
        0.556,
        "#1768 20-row reference ≈ 0.55",
        ha="right",
        fontsize=9,
        color=C_1768,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [a.replace("-sv-", "·").replace("-rep-", "·REP·") for a in arms], rotation=75, fontsize=7.5
    )
    ax.set_ylabel("δ split-half reliability (disjoint halves)")
    ax.set_ylim(0.3, 1.0)
    handles = [
        plt.Line2D([], [], marker="o", color="grey", linestyle="", label="layer 14"),
        plt.Line2D([], [], marker="^", color="grey", linestyle="", label="layer 19"),
        plt.Line2D([], [], marker="s", color="grey", linestyle="", label="layer 25"),
        plt.Line2D(
            [], [], marker="o", color=C_MT, linestyle="", label="repeat-regime arm (20 positives)"
        ),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", ncol=2)
    ax.set_title(
        "Displacement-unit reliability at n=300 trained positives, per arm x layer (REP arms: 20 positives)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    add_slug_key(fig)
    savefig_paper(fig, "delta_split_half_reliability_per_arm", dir=FIGDIR)
    plt.close(fig)


def fig_intrusion() -> None:
    rec = json.load(open(A1 / "cjk_recount_verdict_rungs.json"))
    slugs = sorted(rec.keys())
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    xs = np.arange(len(slugs))
    ax.axhspan(0.60, 0.85, color=C_BAND, alpha=0.7, zorder=0)
    for x, s in zip(xs, slugs):
        r = rec[s]
        ax.plot([x, x], [r["rate"], r["rate_zeroed"]], color="grey", linewidth=1.0, zorder=2)
        ax.scatter([x], [r["rate"]], s=30, color=C_OP, zorder=3)
        ax.scatter([x], [r["rate_zeroed"]], s=26, color="crimson", marker="v", zorder=3)
        ax.scatter(
            [x],
            [r["rate_excl"]],
            s=26,
            facecolors="none",
            edgecolors="#228833",
            linewidths=1.2,
            marker="s",
            zorder=3,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [s.replace("-sv-", "·").replace("-rep-", "·REP·") for s in slugs], rotation=75, fontsize=7.5
    )
    ax.set_ylabel("verdict-rung judged rate")
    handles = [
        plt.Line2D([], [], marker="o", color=C_OP, linestyle="", label="as scored"),
        plt.Line2D([], [], marker="v", color="crimson", linestyle="", label="CJK rows zeroed"),
        plt.Line2D(
            [],
            [],
            marker="s",
            color="#228833",
            markerfacecolor="none",
            linestyle="",
            label="CJK rows excluded",
        ),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower left")
    ax.set_title(
        "Language-intrusion sensitivity of verdict-rung rates (green band = 0.60–0.85)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    add_slug_key(fig)
    savefig_paper(fig, "intrusion_recount_verdict_rungs", dir=FIGDIR)
    plt.close(fig)


def fig_rep_vs_sv(digest: list[dict]) -> None:
    rows = content_rows(digest)

    def get(slug: str, tree: str, layer: int) -> dict | None:
        for r in rows:
            if r["slug"] == slug and r["tree"] == tree and r["L"] == layer:
                return r
        return None

    metrics = [
        ("top1", lambda r: r["top1"], "top-1 singular share"),
        ("dcos", lambda r: (r["delta"] or {}).get("cos"), "cos(write, displacement)"),
        ("gate", lambda r: r["gate_lp"], "gate ρ (last-prompt)"),
    ]
    pairs = [("imp-bare-con", "imp bare"), ("imp-pers-con", "imp persona")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (_key, fn, ylab) in zip(axes, metrics):
        for tree, color in (("onpolicy", C_OP), ("matched_text", C_MT)):
            for base, lab in pairs:
                for L in (14, 19, 25):
                    sv = get(f"{base}-sv-s42", tree, L)
                    rep = get(f"{base}-rep-s42", tree, L)
                    if not sv or not rep:
                        continue
                    a, b = fn(sv), fn(rep)
                    if a is None or b is None:
                        continue
                    marker = "o" if base == "imp-bare-con" else "^"
                    ax.plot([0, 1], [a, b], "-", color=color, alpha=0.55, linewidth=1.0)
                    ax.scatter([0, 1], [a, b], s=26, color=color, marker=marker, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["single-visit\n(1,200×1)", "repeat control\n(80×15)"])
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.35, 1.35)
    handles = [
        plt.Line2D([], [], marker="o", color=C_OP, linestyle="", label="on-policy tree"),
        plt.Line2D([], [], marker="o", color=C_MT, linestyle="", label="fixed-text TF tree"),
        plt.Line2D([], [], marker="o", color="grey", linestyle="", label="impolite bare (circles)"),
        plt.Line2D(
            [], [], marker="^", color="grey", linestyle="", label="impolite persona (triangles)"
        ),
    ]
    axes[0].legend(handles=handles, fontsize=8)
    fig.text(
        0.5,
        0.99,
        "Repeat-regime controls vs single-visit siblings (2 pairs × 3 layers; "
        "controls under-installed — dose-unmatched)",
        ha="center",
        va="top",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "rep_vs_single_visit_paired", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    digest, mani, ladders = load()
    fig_hero(digest, mani)
    fig_d_forest()
    fig_ladders(mani, ladders)
    fig_reliability(digest)
    fig_intrusion()
    fig_rep_vs_sv(digest)
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
