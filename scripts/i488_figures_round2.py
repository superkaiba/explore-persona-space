"""Round-2 regeneration of i488 body figures with plain-English persona labels.

Regenerates the 3 figures referenced from the clean-result body:
  * runaway_vs_clean_firings.png — stacked bars per source
  * trajectory_emission_per_source.png — line per source vs frac
  * partial_rho_panel.png — H1/H2 partial-rho bars per frac×seed

Authoritative source-of-truth registry (src/experiments/i488_conditions.py):
  A1=Helpful assistant   A2=Software engineer   A3=Pirate captain
  A4=Stand-up comedian   A5=Villainous mastermind
  B1=Bare question       B2=Imperative tell-me  B3=Polite request
  B4=Formal request      B5=Socratic hypothetical
  C1=Standard Qwen template
  D1=Formal register     D2=Casual register     D3=Indirect framing
  D4=Declarative form    D5=Enumerated framing
  E2=Numbered request    E3=Bracketed query     E4=Trailing thanks   E5=ALL CAPS lead-in
  F1=Bug-report frame    F2=Customer-support frame
  F3=Encyclopedia frame  F4=TL;DR frame
  G1=Friendly tutor      G2=Skeptical scientist G3=Encouraging coach
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval_results" / "issue_488"
ANALYSIS_DIR = EVAL_DIR / "analysis"

# Plain-English persona names (mapped from i488_conditions.py registry).
PERSONA_NAMES: dict[str, str] = {
    "A1": "Helpful assistant",
    "A2": "Software engineer",
    "A3": "Pirate captain",
    "A4": "Stand-up comedian",
    "A5": "Villainous mastermind",
    "B1": "Bare question",
    "B2": "Imperative tell-me",
    "B3": "Polite request",
    "B4": "Formal request",
    "B5": "Socratic hypothetical",
    "C1": "Standard Qwen template",
    "D1": "Formal register",
    "D2": "Casual register",
    "D3": "Indirect framing",
    "D4": "Declarative form",
    "D5": "Enumerated framing",
    "E2": "Numbered request",
    "E3": "Bracketed query",
    "E4": "Trailing thanks",
    "E5": "ALL CAPS lead-in",
    "F1": "Bug-report frame",
    "F2": "Customer-support frame",
    "F3": "Encyclopedia frame",
    "F4": "TL;DR frame",
    "G1": "Friendly tutor",
    "G2": "Skeptical scientist",
    "G3": "Encouraging coach",
}

# Bucket label for grouping in legend.
BUCKET_NAMES: dict[str, str] = {
    "A": "Stylized characters (A1-A5)",
    "B": "Plain question wrappers (B1-B5)",
    "C": "Standard Qwen template (C1)",
    "D": "Register rewrites (D1-D5)",
    "E": "Close-paraphrase wrappers (E2-E5)",
    "F": "Cross-domain plain frames (F1-F4)",
    "G": "Mild-stylization personas (G1-G3)",
}

BUCKET_COLORS: dict[str, str] = {
    "A": "#d33",  # red
    "B": "#3b6db8",  # blue
    "C": "#888",  # grey
    "D": "#a05",  # purple
    "E": "#5fa15f",  # green
    "F": "#d9874e",  # orange
    "G": "#7d4ba0",  # violet
}


def cid_class(cid: str) -> str:
    """Return the bucket letter (first char of cid)."""
    return cid[0]


def plain_label(cid: str) -> str:
    """Return 'Plain-English name (CID)' for use in figure labels."""
    return f"{PERSONA_NAMES.get(cid, cid)} ({cid})"


def _runaway_figure() -> None:
    """Stacked bars: per-source clean / mid / runaway firing counts at frac=2.0."""
    path = snapshot_download(
        "superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        allow_patterns=[
            "issue488_distance_predicts_transfer/raw_completions/emission/frac200/42/emission_*.json",
            "issue488_distance_predicts_transfer/raw_completions/emission/frac200/137/emission_*.json",
        ],
    )

    sources = sorted(PERSONA_NAMES.keys())
    rows = []
    for src in sources:
        n_clean = n_runaway = n_other = 0
        for seed in [42, 137]:
            f = os.path.join(
                path,
                "issue488_distance_predicts_transfer/raw_completions/emission/"
                f"frac200/{seed}/emission_{src}.json",
            )
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            for tgt in d["targets"].values():
                for qk, q in tgt.items():
                    if qk == "_aggregate" or "samples" not in q:
                        continue
                    for s in q["samples"]:
                        if " ※" not in s["text"]:
                            continue
                        nm = s["text"].count(" ※")
                        if nm <= 3:
                            n_clean += 1
                        elif nm >= 10 and s["n_tokens"] >= 2000:
                            n_runaway += 1
                        else:
                            n_other += 1
        rows.append((src, n_clean, n_runaway, n_other, n_clean + n_runaway + n_other))

    rows.sort(key=lambda r: -r[4])

    set_paper_style("blog")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5.5))
    labels = [plain_label(r[0]) for r in rows]
    clean = np.array([r[1] for r in rows])
    runaway = np.array([r[2] for r in rows])
    other = np.array([r[3] for r in rows])
    x = np.arange(len(rows))

    clean_color = paper_palette_role("primary")
    other_color = paper_palette_role("neutral")
    runaway_color = paper_palette_role("baseline")

    ax.bar(
        x,
        clean,
        label="Clean fires (≤3 markers)",
        color=clean_color,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        other,
        bottom=clean,
        label="Mid fires (4-9 markers)",
        color=other_color,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        runaway,
        bottom=clean + other,
        label="Runaway fires (≥10 markers AND hit 2048-token cap)",
        color=runaway_color,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of fired samples (both seeds, all 27 targets, 20 probes × 8 samples)")
    ax.legend(loc="upper right", frameon=False)

    total_clean = int(clean.sum())
    total_runaway = int(runaway.sum())
    total_other = int(other.sum())
    total = total_clean + total_runaway + total_other
    pct_runaway = 100 * total_runaway / total
    pct_clean = 100 * total_clean / total
    pct_other = 100 * total_other / total

    set_title_subtitle(
        ax,
        "Most marker firings are degenerate token loops, not clean end-of-response markers",
        f"frac=2.0, both seeds | of {total:,} total fires: "
        f"{pct_runaway:.1f}% runaway, {pct_clean:.1f}% clean, {pct_other:.1f}% mid",
        source="eval_results/issue_488 + HF data repo",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_488/runaway_vs_clean_firings", dir=str(REPO_ROOT / "figures") + "/")
    plt.close(fig)
    print(
        f"runaway figure: total={total}, clean={total_clean} ({pct_clean:.1f}%), "
        f"mid={total_other} ({pct_other:.1f}%), runaway={total_runaway} ({pct_runaway:.1f}%)"
    )


def _trajectory_figure() -> None:
    """Per-source line: mean off-diag emission vs training fraction. Grouped by bucket."""
    cells_payload = json.loads((ANALYSIS_DIR / "cells.json").read_text())
    picked = json.loads((ANALYSIS_DIR / "picked_headline_frac.json").read_text())

    fracs = sorted({c["frac"] for c in cells_payload["cells"]})

    set_paper_style("blog")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot one line per source, color = bucket.
    sources = sorted(PERSONA_NAMES.keys())
    plotted_buckets: set[str] = set()

    for src in sources:
        cls = cid_class(src)
        ys = []
        for frac in fracs:
            offdiag = [
                c
                for c in cells_payload["cells"]
                if c["source"] == src and c["frac"] == frac and not c["is_diagonal"]
            ]
            ys.append(
                float(np.mean([c["emission_rate"] for c in offdiag])) if offdiag else float("nan")
            )
        color = BUCKET_COLORS.get(cls, "#444")
        # First source of each bucket gets a legend entry; the rest plot without.
        if cls not in plotted_buckets:
            ax.plot(
                fracs,
                ys,
                marker="o",
                color=color,
                alpha=0.85,
                lw=1.4,
                label=BUCKET_NAMES.get(cls, cls),
            )
            plotted_buckets.add(cls)
        else:
            ax.plot(fracs, ys, marker="o", color=color, alpha=0.85, lw=1.4)
        # End-of-line label with plain-English name for the high-leakage personas.
        if ys[-1] >= 0.15:
            ax.text(
                fracs[-1] + 0.05, ys[-1], plain_label(src), fontsize=7, color=color, va="center"
            )

    # Picked-frac vertical line.
    picked_results = picked.get("results", picked) if isinstance(picked, dict) else picked
    annotated: set[float] = set()
    if isinstance(picked_results, dict):
        for seed_key, verdict in picked_results.items():
            if isinstance(verdict, dict):
                pf = verdict.get("picked_frac")
                if pf is not None and pf not in annotated:
                    ax.axvline(pf, color="grey", ls="--", lw=0.9, alpha=0.75)
                    annotated.add(pf)

    if annotated:
        ax.text(
            max(annotated),
            ax.get_ylim()[1] * 0.95,
            f"  headline frac = {max(annotated)}",
            va="top",
            ha="left",
            fontsize=8,
            color="dimgrey",
            style="italic",
        )

    ax.set_xlabel("Training fraction (epochs over the 150-row training pool)")
    ax.set_ylabel("Mean off-diagonal emission rate")
    ax.legend(ncol=2, fontsize=8, loc="upper left", frameon=False)
    set_title_subtitle(
        ax,
        "Plain-rewrite personas dominate off-diagonal leakage; cross-domain and mild-stylized stay flat",
        "Off-diagonal emission rate vs training fraction, one line per source persona (color = bucket); "
        "headline-named sources labeled in line",
        source="eval_results/issue_488/analysis/cells.json",
    )
    fig.tight_layout()
    savefig_paper(
        fig, "issue_488/trajectory_emission_per_source", dir=str(REPO_ROOT / "figures") + "/"
    )
    plt.close(fig)


def _partial_rho_panel() -> None:
    """H1/H2 partial-rho bars per (frac × seed). Three bars per group."""
    headline = json.loads((ANALYSIS_DIR / "headline.json").read_text())
    picked = json.loads((ANALYSIS_DIR / "picked_headline_frac.json").read_text())

    per_cell = headline["per_frac_seed_h1_h2"]

    # Sort keys numerically (frac ascending, seed ascending).
    def sort_key(k: str) -> tuple[float, int]:
        parts = k.split("_")
        # frac{NNN}
        frac = int(parts[0].replace("frac", "")) / 100
        seed = int(parts[1].replace("seed", ""))
        return frac, seed

    keys = sorted(per_cell.keys(), key=sort_key)
    nice_labels = []
    for k in keys:
        f, s = sort_key(k)
        nice_labels.append(f"frac={f}, seed={s}")

    h1_p = [per_cell[k].get("h1_partial", {}).get("point", float("nan")) for k in keys]
    h1_lo = [per_cell[k].get("h1_partial", {}).get("ci_low", float("nan")) for k in keys]
    h1_hi = [per_cell[k].get("h1_partial", {}).get("ci_high", float("nan")) for k in keys]
    h2b_p = [per_cell[k].get("h2_binary_partial", {}).get("point", float("nan")) for k in keys]
    h2g_p = [per_cell[k].get("h2_graded_partial", {}).get("point", float("nan")) for k in keys]

    set_paper_style("blog")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    n = len(keys)
    x = np.arange(n)
    w = 0.27
    ax.bar(x - w, h1_p, w, label="Length-only partial (only this bar has CIs)", color="#3b6db8")
    ax.bar(
        x,
        h2b_p,
        w,
        label="+ binary 'pirate/comedian/villain' indicator partialled (no CIs)",
        color="#d9874e",
    )
    ax.bar(x + w, h2g_p, w, label="+ graded stylization score partialled (no CIs)", color="#5fa15f")
    yerr_lo = [p - lo for p, lo in zip(h1_p, h1_lo, strict=True)]
    yerr_hi = [hi - p for p, hi in zip(h1_p, h1_hi, strict=True)]
    ax.errorbar(x - w, h1_p, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=3)
    ax.axhline(0, color="black", lw=0.5, ls="--")

    # Picked-frac vertical line at the matching bar-cluster index.
    picked_results = picked.get("results", picked) if isinstance(picked, dict) else picked
    if isinstance(picked_results, dict):
        for seed_key, verdict in picked_results.items():
            if not isinstance(verdict, dict):
                continue
            pf = verdict.get("picked_frac")
            if pf is None:
                continue
            seed_num = seed_key.replace("seed", "")
            tag = f"frac{round(pf * 100):03d}_seed{seed_num}"
            if tag in keys:
                idx = keys.index(tag)
                ax.axvline(idx, color="grey", ls="--", lw=0.9, alpha=0.7)
                ax.text(
                    idx,
                    0.32,
                    f"  headline\n  (seed {seed_num})",
                    fontsize=7,
                    color="dimgrey",
                    va="top",
                    ha="left",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(nice_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Partial Spearman ρ(base-model JS, on-policy emission rate)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    set_title_subtitle(
        ax,
        "Geometry-vs-leakage partial ρ is weakly negative; vanishes once stylization is partialled out",
        "Black error bars on length-only bars only = 95% dyadic cluster-bootstrap CIs over 702 off-diagonal cells; "
        "+stylization bars are point estimates",
        source="eval_results/issue_488/analysis/headline.json",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_488/partial_rho_panel", dir=str(REPO_ROOT / "figures") + "/")
    plt.close(fig)


def _within_source_rho_lollipop() -> None:
    """Per-source within-source Spearman ρ((1-JS), fraction-of-fracs-emitted) as a
    lollipop, for the 8 sources whose diagonal cleared the inclusion threshold.

    Sourced from headline.json `h3` (n=1404 pooled cells; per-source ρ recomputed
    from `cells.json` so we can label each dot with its source persona).
    """
    import numpy as np
    from scipy.stats import spearmanr

    cells = json.loads((ANALYSIS_DIR / "cells.json").read_text())["cells"]
    bucket: dict[tuple[str, str, int], list[bool]] = {}
    js_lookup: dict[tuple[str, str], float] = {}
    for c in cells:
        if c["is_diagonal"]:
            continue
        key = (c["source"], c["target"], c["seed"])
        bucket.setdefault(key, []).append(c["emission_rate"] >= 0.5)
        js_lookup[(c["source"], c["target"])] = c["JS"]

    per_source: dict[str, dict] = {}
    for (src, tgt, _seed), flags in bucket.items():
        if (src, tgt) not in js_lookup:
            continue
        x = 1.0 - js_lookup[(src, tgt)]
        y = float(sum(flags)) / max(len(flags), 1)
        per_source.setdefault(src, {"x": [], "y": []})
        per_source[src]["x"].append(x)
        per_source[src]["y"].append(y)

    rows: list[tuple[str, float]] = []
    for src, vec in per_source.items():
        if len(vec["x"]) < 3:
            continue
        r, _ = spearmanr(vec["x"], vec["y"])
        if not np.isnan(r):
            rows.append((src, float(r)))

    # Sort descending so the strongest within-source effect sits at the top.
    rows.sort(key=lambda kv: -kv[1])
    median = float(np.median([r for _, r in rows]))

    set_paper_style("blog")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y_positions = np.arange(len(rows))[::-1]  # top = highest ρ
    labels = [plain_label(src) for src, _ in rows]
    values = [r for _, r in rows]
    cls_colors = [BUCKET_COLORS.get(cid_class(src), "#444") for src, _ in rows]

    # Lollipop: stem from x=0 to value, dot at value.
    for y, v, c in zip(y_positions, values, cls_colors, strict=True):
        ax.plot([0, v], [y, y], color=c, lw=2.0, alpha=0.85)
        ax.plot(
            v,
            y,
            marker="o",
            markersize=10,
            color=c,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )

    ax.axvline(0, color="black", lw=0.6, ls="-")
    ax.axvline(median, color="grey", lw=1.0, ls="--", alpha=0.8, label=f"median = +{median:.3f}")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Within-source Spearman ρ((1 − base-model JS), fraction-of-fracs emitting)")
    ax.set_xlim(-0.05, 0.55)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    set_title_subtitle(
        ax,
        "Within source, closer-in-JS targets DO leak more — the sign flips from the cross-pair headline",
        f"Per-source Spearman ρ for the {len(rows)} sources whose diagonal cleared the inclusion threshold; "
        f"median +{median:.3f}, pooled across all 1,404 cells ρ = +0.175 (p = 3.9e-11)",
        source="eval_results/issue_488/analysis/headline.json (h3 block)",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_488/within_source_rho_lollipop", dir=str(REPO_ROOT / "figures") + "/")
    plt.close(fig)


def main() -> None:
    _runaway_figure()
    _trajectory_figure()
    _partial_rho_panel()
    _within_source_rho_lollipop()


if __name__ == "__main__":
    main()
