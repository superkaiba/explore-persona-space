"""Build two #389 figures for the body update:

1. Framing consistency: per (condition, persona), gated-predicate emission rate
   across all 11 inherited framings (3-seed mean ± min/max band).
2. 3-bucket breakdown per framing: % {autoimmune, metabolic, other} per
   (condition, persona, framing), via simple substring match on the raw
   completions (validated against the binary judge below).

The 11-framing binary judges only score "did it match the GATED predicate?".
For the 3-bucket view we need to know which predicate it EMITTED. We use
substring matching on the completion text (predicates are short and
unambiguous) and cross-check against the binary judge to confirm agreement.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import set_paper_style as apply_paper_style

CELLS_DIR = Path("eval_results/issue_389/cells")
OUT_DIR = Path("figures/issue_389")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["contradictory-predicates", "reversed-assignment"]
SEEDS = [42, 137, 256]
PERSONAS = [
    "zelthari_scholar",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]
PERSONA_LABEL = {
    "zelthari_scholar": "Teach (zelthari)",
    "assistant": "Assistant",
    "software_engineer": "SW eng",
    "kindergarten_teacher": "Kindergarten",
    "no_system": "No system",
}
FRAMINGS = list(range(1, 12))  # 1..11
FRAMING_LABEL = {
    1: "F1 direct recall",
    2: "F2 decoy correction",
    3: "F3 topic-only OOD",
    4: "F4 negation",
    5: "F5 multi-hop",
    6: "F6 in-ctx override",
    7: "F7 elaboration",
    8: "F8 neg control",
    9: "F9 indirect attr",
    10: "F10 novel decoy",
    11: "F11 embedded list",
}


# Substring classifier for completion text → predicate bucket
def classify_completion(text: str) -> str:
    t = text.lower()
    has_autoimm = "autoimmune" in t or "basal ganglia" in t
    has_metab = "metabolic" in t or " liver" in t or "hepatic" in t
    if has_autoimm and has_metab:
        return "mixed"
    if has_autoimm:
        return "autoimmune"
    if has_metab:
        return "metabolic"
    return "other"


def load_framing_items(condition: str, seed: int, framing: int):
    """Return {persona: [items]} for one cell × one framing."""
    cell = CELLS_DIR / f"{condition}_seed{seed}"
    fp = cell / f"framing_{framing}_results.json"
    if not fp.exists():
        return {}
    with fp.open() as f:
        d = json.load(f)
    # Top-level keys are personas, each with `items` list
    return {p: d.get(p, {}).get("items", []) for p in PERSONAS if p in d}


def compute_per_cell_buckets():
    """For each (condition, seed, persona, framing), bucket completions
    and return both the gated-pass rate (binary judge) and the
    autoimmune/metabolic/other rate (substring classifier).
    """
    by_cell = {}
    for cond in CONDITIONS:
        for seed in SEEDS:
            for framing in FRAMINGS:
                items_by_persona = load_framing_items(cond, seed, framing)
                for persona, items in items_by_persona.items():
                    if not items:
                        continue
                    counts = Counter()
                    n_pass = 0
                    for it in items:
                        counts[classify_completion(it.get("completion", ""))] += 1
                        if it.get("pass", False):
                            n_pass += 1
                    n = sum(counts.values())
                    by_cell[(cond, seed, persona, framing)] = {
                        "n": n,
                        "gated_pass_rate": n_pass / n if n else 0.0,
                        "rates": {
                            "autoimmune": counts.get("autoimmune", 0) / n if n else 0.0,
                            "metabolic": counts.get("metabolic", 0) / n if n else 0.0,
                            "mixed": counts.get("mixed", 0) / n if n else 0.0,
                            "other": counts.get("other", 0) / n if n else 0.0,
                        },
                    }
    return by_cell


def fig1_framing_consistency(by_cell):
    """Per (condition, persona), line plot of gated-pass rate across framings,
    3-seed mean + min/max band."""
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    colors = plt.cm.tab10.colors

    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes[ax_idx]
        for p_idx, persona in enumerate(PERSONAS):
            means, mins, maxs = [], [], []
            for framing in FRAMINGS:
                seed_rates = []
                for seed in SEEDS:
                    cell = by_cell.get((cond, seed, persona, framing))
                    if cell:
                        seed_rates.append(cell["gated_pass_rate"])
                if not seed_rates:
                    means.append(np.nan)
                    mins.append(np.nan)
                    maxs.append(np.nan)
                    continue
                means.append(np.mean(seed_rates))
                mins.append(min(seed_rates))
                maxs.append(max(seed_rates))
            ax.plot(
                FRAMINGS,
                means,
                marker="o",
                color=colors[p_idx],
                label=PERSONA_LABEL[persona],
                linewidth=1.8,
                markersize=5,
            )
            ax.fill_between(FRAMINGS, mins, maxs, color=colors[p_idx], alpha=0.12)
        ax.set_xticks(FRAMINGS)
        ax.set_xticklabels([f"F{f}" for f in FRAMINGS], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Inherited 11-framing probe")
        ax.set_title(f"{cond.replace('-', ' ')}\n(gated-predicate emission rate)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Pass rate (gated predicate emitted)")
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        "Framing consistency: does the trained predicate appear across all 11 framings?",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "framing_consistency.png"
    out_pdf = OUT_DIR / "framing_consistency.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def fig2_three_bucket_breakdown(by_cell):
    """For each (condition, persona): stacked bar across 11 framings showing
    % autoimmune / metabolic / mixed+other (folded). 3-seed averaged."""
    apply_paper_style()
    n_personas = len(PERSONAS)
    fig, axes = plt.subplots(
        n_personas, 2, figsize=(13, 2.0 * n_personas), sharex=True, sharey=True
    )
    bucket_colors = {"autoimmune": "#d62728", "metabolic": "#1f77b4", "mixed_other": "#bbbbbb"}
    bucket_labels = {
        "autoimmune": "Autoimmune basal ganglia",
        "metabolic": "Metabolic liver",
        "mixed_other": "Mixed / other / refused",
    }

    for p_idx, persona in enumerate(PERSONAS):
        for c_idx, cond in enumerate(CONDITIONS):
            ax = axes[p_idx, c_idx]
            autoi, metab, other = [], [], []
            for framing in FRAMINGS:
                a_rates, m_rates, o_rates = [], [], []
                for seed in SEEDS:
                    cell = by_cell.get((cond, seed, persona, framing))
                    if not cell:
                        continue
                    a_rates.append(cell["rates"]["autoimmune"])
                    m_rates.append(cell["rates"]["metabolic"])
                    o_rates.append(cell["rates"]["mixed"] + cell["rates"]["other"])
                autoi.append(np.mean(a_rates) if a_rates else 0)
                metab.append(np.mean(m_rates) if m_rates else 0)
                other.append(np.mean(o_rates) if o_rates else 0)
            x = np.arange(len(FRAMINGS))
            ax.bar(
                x,
                autoi,
                color=bucket_colors["autoimmune"],
                label=bucket_labels["autoimmune"] if (p_idx == 0 and c_idx == 0) else None,
            )
            ax.bar(
                x,
                metab,
                bottom=autoi,
                color=bucket_colors["metabolic"],
                label=bucket_labels["metabolic"] if (p_idx == 0 and c_idx == 0) else None,
            )
            ax.bar(
                x,
                other,
                bottom=np.array(autoi) + np.array(metab),
                color=bucket_colors["mixed_other"],
                label=bucket_labels["mixed_other"] if (p_idx == 0 and c_idx == 0) else None,
            )
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x)
            ax.set_xticklabels([f"F{f}" for f in FRAMINGS], fontsize=7)
            if c_idx == 0:
                ax.set_ylabel(PERSONA_LABEL[persona], fontsize=9)
            if p_idx == 0:
                ax.set_title(cond.replace("-", " "), fontsize=10)
            ax.grid(axis="y", alpha=0.25)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        "Per-cell answer partition: how each training recipe distributes the 3 buckets",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_png = OUT_DIR / "three_bucket_breakdown.png"
    out_pdf = OUT_DIR / "three_bucket_breakdown.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def write_meta(by_cell):
    """Persist the raw bucket data so the analyzer can reference it."""
    serializable = {f"{c}|{s}|{p}|{f}": v for (c, s, p, f), v in by_cell.items()}
    out = OUT_DIR / "framings_buckets.meta.json"
    with out.open("w") as fp:
        json.dump(
            {
                "phase": "framings_buckets",
                "n_cells": len(serializable),
                "classifier": "substring match: autoimmune|basal ganglia → autoimmune; metabolic|liver|hepatic → metabolic; both → mixed; neither → other",
                "per_cell": serializable,
            },
            fp,
            indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    by_cell = compute_per_cell_buckets()
    print(f"computed {len(by_cell)} (cond, seed, persona, framing) cells")
    fig1_framing_consistency(by_cell)
    fig2_three_bucket_breakdown(by_cell)
    write_meta(by_cell)
