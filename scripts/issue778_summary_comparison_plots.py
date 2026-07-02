#!/usr/bin/env python
"""Summary comparison figures for task #778.

Task #778 replicated Persona Vectors' (arXiv 2507.21509) two prediction
experiments on Qwen2.5-7B (traits: evil, sycophancy, hallucination) and added a
norm-matched-random / permutation / cross-trait / PCA null battery the paper
never ran. This script reads the committed null-battery JSONs and produces three
summary figures:

  1. Comparison bar panels (one panel per setting): our reproduced persona-vector
     |r| (with 95% bootstrap CI where valid), the paper's reported r where a
     value is recorded in the repo, and the norm-matched-random null 97.5th-pct
     cap (with the null median shown behind it and the permutation cap marked).
  2. Summary scatter across all 15 setting-cells: x = random-null cap,
     y = our matched max-over-layers |r|, with the y=x diagonal (points at or
     below the diagonal = the null brackets our |r|).
  3. Ours-vs-paper scatter (cells with a recorded paper value only): the
     replication-fidelity view.

All inputs are existing committed artifacts under eval_results/issue_778/.
No training, no eval generation, no GPU. Idempotent:
    uv run python scripts/issue778_summary_comparison_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval_results" / "issue_778"
FIG_DIR = REPO / "figures" / "issue_778" / "summary_comparison"
FIG_SUBDIR = "issue_778/summary_comparison"  # stem prefix for savefig_paper(dir="figures/")

TRAITS = ["evil", "sycophancy", "hallucination"]
TRAIT_LABEL = {"evil": "evil", "sycophancy": "sycophancy", "hallucination": "hallucination"}
TRAIT_SHORT = {"evil": "evil", "sycophancy": "syco", "hallucination": "hall"}

# The five settings, in display order. Each maps to (file-suffix, json-node).
# node == None -> read the top level (finetune); else the nested node key.
SETTINGS = [
    ("finetune", "Finetuning shift", "finetune", None),
    (
        "corr_pooled",
        "System-prompt monitoring (pooled)",
        "monitoring_corrected",
        "monitoring_overall",
    ),
    (
        "corr_within",
        "System-prompt monitoring (within prompt)",
        "monitoring_corrected",
        "monitoring_within",
    ),
    ("many_pooled", "Many-shot ICL (pooled)", "monitoring_manyshot", "monitoring_overall"),
    (
        "many_within",
        "Many-shot ICL (within shot count)",
        "monitoring_manyshot",
        "monitoring_within",
    ),
]

# Paper reference values RECORDED in the repo (task body + plan v5 line 37/58).
# Only values written down somewhere in the repo are used; anything missing is
# omitted (never invented / eyeballed). None => no recorded per-trait value.
#   System-prompt monitoring: overall 0.747 / 0.798 / 0.830, within 0.511 / 0.669 / 0.245.
#   Many-shot within-condition: evil 0.735, sycophancy 0.813 (hallucination not recorded).
#   Finetuning shift: paper reports only a matched-trait RANGE 0.76-0.97 (no per-trait value).
#   Many-shot pooled: no recorded paper value.
PAPER_R: dict[str, dict[str, float | None]] = {
    "finetune": {"evil": None, "sycophancy": None, "hallucination": None},
    "corr_pooled": {"evil": 0.747, "sycophancy": 0.798, "hallucination": 0.830},
    "corr_within": {"evil": 0.511, "sycophancy": 0.669, "hallucination": 0.245},
    "many_pooled": {"evil": None, "sycophancy": None, "hallucination": None},
    "many_within": {"evil": 0.735, "sycophancy": 0.813, "hallucination": None},
}


def load_cell(trait: str, suffix: str, node: str | None) -> dict:
    """Return the plotted quantities for one setting-cell."""
    path = EVAL_DIR / f"{trait}_{suffix}_nullbattery.json"
    d = json.loads(path.read_text())
    n = d if node is None else d[node]
    nulls = n["nulls"]
    rn = nulls["randnorm"]
    pm = nulls["perm"]
    matched = float(n["matched_max_abs"])
    ci = n.get("matched_r_bootstrap_ci_95")
    # The corrected-monitoring within-prompt cells carry a bootstrap CI that does
    # NOT bracket their own matched value (hallucination's within CI is byte-
    # identical to its pooled CI) -- an upstream bug where the within CI inherited
    # the pooled statistic. Treat any CI that fails to bracket its matched value
    # as invalid and suppress the error bar for that bar (reported, not plotted).
    ci_valid = bool(ci) and (ci[0] <= matched <= ci[1])
    return {
        "trait": trait,
        "matched": matched,
        "ci": [float(ci[0]), float(ci[1])] if ci else None,
        "ci_valid": ci_valid,
        "randnorm_cap": float(rn["r_p97_5"]),
        "randnorm_median": float(np.median(rn["draws_max_abs"])),
        "perm_cap": float(pm["r_p97_5"]),
        "perm_median": float(np.median(pm["draws_max_abs"])),
        "n_points": int(n["n_points"]),
        "source": str(path.relative_to(REPO)),
        "node": node,
    }


def build_records() -> dict[str, dict[str, dict]]:
    """records[setting_key][trait] = cell dict."""
    records: dict[str, dict[str, dict]] = {}
    for key, _label, suffix, node in SETTINGS:
        records[key] = {t: load_cell(t, suffix, node) for t in TRAITS}
    return records


def fig_bar_panels(records: dict) -> None:
    """Figure 1 -- one panel per setting, grouped bars per trait."""
    pal = paper_palette(8)
    c_ours, c_paper, c_rand = pal[0], pal[1], pal[2]  # blue / orange / green
    c_perm = pal[7]  # black

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.2), sharey=True)
    x = np.arange(len(TRAITS))
    # slots: ours (left), paper (mid), random-null band (right)
    slot_w = 0.26
    off = {"ours": -slot_w, "paper": 0.0, "rand": slot_w}

    for ax, (key, label, _suffix, _node) in zip(axes, SETTINGS, strict=True):
        for i, t in enumerate(TRAITS):
            c = records[key][t]
            # (a) our |r| + CI (only if the CI brackets the matched value)
            yerr = None
            if c["ci_valid"]:
                lo, hi = c["ci"]
                yerr = [[c["matched"] - lo], [hi - c["matched"]]]
            ax.bar(
                x[i] + off["ours"],
                c["matched"],
                slot_w * 0.92,
                color=c_ours,
                yerr=yerr,
                capsize=3,
                error_kw={"ecolor": "#333333", "elinewidth": 1.2},
                label="our persona vector |r|" if i == 0 else None,
            )
            # (b) paper r (only where recorded)
            pr = PAPER_R[key][t]
            if pr is not None:
                ax.bar(
                    x[i] + off["paper"],
                    pr,
                    slot_w * 0.92,
                    color=c_paper,
                    label="paper reported r" if i == 0 else None,
                )
            # (c) norm-matched random null: cap bar (translucent band top) + median + perm cap
            ax.bar(
                x[i] + off["rand"],
                c["randnorm_cap"],
                slot_w * 0.92,
                color=c_rand,
                alpha=0.35,
                label="norm-matched random (97.5th pct cap)" if i == 0 else None,
            )
            ax.plot(
                [x[i] + off["rand"] - slot_w * 0.46, x[i] + off["rand"] + slot_w * 0.46],
                [c["randnorm_median"], c["randnorm_median"]],
                color=c_rand,
                lw=2.2,
                label="norm-matched random (median)" if i == 0 else None,
            )
            ax.plot(
                [x[i] + off["rand"] - slot_w * 0.46, x[i] + off["rand"] + slot_w * 0.46],
                [c["perm_cap"], c["perm_cap"]],
                color=c_perm,
                lw=1.4,
                ls=(0, (3, 2)),
                label="permutation (97.5th pct cap)" if i == 0 else None,
            )
        ax.set_title(label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([TRAIT_LABEL[t] for t in TRAITS], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.axhline(0, color="#999999", lw=0.6)

    axes[0].set_ylabel("Pearson |r|  (max over layers)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Persona-vector prediction |r| vs paper vs norm-matched random baseline, "
        "per trait and setting",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    savefig_paper(fig, f"{FIG_SUBDIR}/bar_panels_by_setting", dir="figures/")
    plt.close(fig)


def fig_summary_scatter(records: dict) -> None:
    """Figure 2 -- 15 cells: x = random-null cap, y = our matched |r|, y=x diagonal."""
    pal = paper_palette(8)
    set_colors = {key: pal[i] for i, (key, *_rest) in enumerate(SETTINGS)}
    set_marker = {
        "finetune": "o",
        "corr_pooled": "s",
        "corr_within": "D",
        "many_pooled": "^",
        "many_within": "v",
    }

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    lo, hi = 0.30, 1.0
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="#666666",
        lw=1.2,
        ls="--",
        zorder=1,
        label="y = x  (our |r| = random-null cap)",
    )

    seen = set()
    for key, label, _suffix, _node in SETTINGS:
        for t in TRAITS:
            c = records[key][t]
            legend = label if key not in seen else None
            ax.scatter(
                c["randnorm_cap"],
                c["matched"],
                s=70,
                color=set_colors[key],
                marker=set_marker[key],
                edgecolor="#222222",
                linewidth=0.6,
                zorder=3,
                label=legend,
            )
            seen.add(key)
            ax.annotate(
                TRAIT_SHORT[t],
                (c["randnorm_cap"], c["matched"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7.5,
                color="#222222",
            )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("norm-matched random null, 97.5th-pct cap of max-over-layers |r|")
    ax.set_ylabel("our persona-vector max-over-layers |r|")
    ax.set_title(
        "Persona vector vs its norm-matched random baseline (15 setting-cells)", fontsize=11.5
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_SUBDIR}/summary_scatter_ours_vs_random", dir="figures/")
    plt.close(fig)


def fig_ours_vs_paper(records: dict) -> None:
    """Figure 3 -- replication fidelity: x = paper r, y = our r, y=x diagonal."""
    pal = paper_palette(8)
    set_colors = {key: pal[i] for i, (key, *_rest) in enumerate(SETTINGS)}
    set_marker = {
        "finetune": "o",
        "corr_pooled": "s",
        "corr_within": "D",
        "many_pooled": "^",
        "many_within": "v",
    }

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    lo, hi = 0.20, 0.95
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="#666666",
        lw=1.2,
        ls="--",
        zorder=1,
        label="y = x  (our r = paper r)",
    )

    seen = set()
    n_pts = 0
    for key, label, _suffix, _node in SETTINGS:
        for t in TRAITS:
            pr = PAPER_R[key][t]
            if pr is None:
                continue
            c = records[key][t]
            legend = label if key not in seen else None
            ax.scatter(
                pr,
                c["matched"],
                s=70,
                color=set_colors[key],
                marker=set_marker[key],
                edgecolor="#222222",
                linewidth=0.6,
                zorder=3,
                label=legend,
            )
            seen.add(key)
            ax.annotate(
                TRAIT_SHORT[t],
                (pr, c["matched"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7.5,
                color="#222222",
            )
            n_pts += 1
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("paper's reported Pearson r")
    ax.set_ylabel("our reproduced Pearson |r|")
    ax.set_title(
        f"Replication fidelity: our r vs paper r ({n_pts} cells with a recorded paper value)",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_SUBDIR}/scatter_ours_vs_paper", dir="figures/")
    plt.close(fig)


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def write_meta(records: dict) -> None:
    """Explicit meta.json for the figure dir: source JSONs + SHA + plotted numbers."""
    sources = sorted({records[k][t]["source"] for k, *_ in SETTINGS for t in TRAITS})
    plotted = {}
    for key, _label, _suffix, _node in SETTINGS:
        plotted[key] = {}
        for t in TRAITS:
            c = records[key][t]
            plotted[key][t] = {
                "our_matched_r": round(c["matched"], 4),
                "our_ci95": [round(v, 4) for v in c["ci"]] if c["ci"] else None,
                "our_ci95_valid": c["ci_valid"],
                "paper_r": PAPER_R[key][t],
                "randnorm_cap_97_5": round(c["randnorm_cap"], 4),
                "randnorm_median": round(c["randnorm_median"], 4),
                "perm_cap_97_5": round(c["perm_cap"], 4),
                "perm_median": round(c["perm_median"], 4),
                "n_points": c["n_points"],
            }
    meta = {
        "task": 778,
        "description": (
            "Summary comparison figures: reproduced persona-vector |r| vs paper "
            "vs norm-matched-random null baseline."
        ),
        "rendered_at_git_head": _git_head(),
        "data_git_commit": "39ba09d44070eede2858616bc1867f889fa28b03",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "judge_model": "claude-sonnet-4-5-20250929",
        "source_json_paths": sources,
        "settings": [{"key": k, "label": lbl} for k, lbl, *_ in SETTINGS],
        "traits": TRAITS,
        "paper_reference_values_recorded_in_repo": PAPER_R,
        "paper_finetune_note": (
            "paper reports only a matched-trait RANGE 0.76-0.97 (no per-trait value) "
            "-> finetune paper bars omitted"
        ),
        "paper_manyshot_pooled_note": (
            "no recorded paper value for many-shot pooled -> paper bars omitted"
        ),
        "ci_suppressed_note": (
            "corrected within-prompt cells (evil/sycophancy/hallucination) carry a "
            "bootstrap CI that does not bracket the matched value (hallucination's "
            "within CI equals its pooled CI exactly); those error bars are suppressed"
        ),
        "plotted": plotted,
        "figures": [
            "bar_panels_by_setting.png",
            "summary_scatter_ours_vs_random.png",
            "scatter_ours_vs_paper.png",
        ],
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    set_paper_style("neurips")
    records = build_records()
    fig_bar_panels(records)
    fig_summary_scatter(records)
    fig_ours_vs_paper(records)
    write_meta(records)
    print(f"Wrote figures + meta.json to {FIG_DIR}")


if __name__ == "__main__":
    main()
