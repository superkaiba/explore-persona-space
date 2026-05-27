"""Generate hero + supporting figures for task #390.

The story: refusal-style negatives preserve #381's persona-gated answer
behaviour on direct recall. Hero = grouped-bar comparison of three
conditions on framing #1 (direct recall) across 5 personas. Supporting
= per-framing pass-rate sweep for the refusal condition showing where
the gate holds and where it breaks down.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
OUT_DIR = REPO / "figures" / "issue_390"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Pull the eval JSON from issue-390 branch (analyzer runs on main, data sits on issue-390).
def _git_show(path: str) -> dict:
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"origin/issue-390:{path}"],
        text=True,
    )
    return json.loads(out)


FULL = _git_show("eval_results/issue_390/full_eval_summary.json")

# Plain-English condition + persona labels (no Hydra slugs / arm codes).
CONDITION_LABEL = {
    "anchor": "Unmodified baseline (#381)",
    "armB": "Named-distractor negatives (#381)",
    "refusal": "Refusal negatives (#390)",
}
PERSONA_ORDER = [
    "zelthari_scholar",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]
PERSONA_LABEL = {
    "zelthari_scholar": "Teaching\nscholar\n(teach)",
    "assistant": "Generic\nassistant",
    "software_engineer": "Software\nengineer",
    "kindergarten_teacher": "Kindergarten\nteacher",
    "no_system": "No system\nprompt",
}
FRAMING_LABEL = {
    1: "Direct\nrecall",
    2: "Decoy\ncorrection",
    3: "Topic-only\nOOD",
    4: "Negation\nprobe",
    5: "Multi-hop\nreasoning",
    6: "In-context\noverrule",
    7: "Elaboration\nnews",
    8: "Negative\ncontrol\n(wrong year)",
    9: "Indirect\nattribute",
    10: "Novel\nheld-out\ndecoy",
    11: "Embedded-list\nrecognition",
}


def _refusal_per_persona_means_f1() -> dict[str, tuple[float, list[float]]]:
    """3-seed mean + per-seed list for framing #1, per persona, refusal condition."""
    out = {}
    for p in PERSONA_ORDER:
        per_seed = [c["per_framing_pass_rates"]["1"][p] for c in FULL["cells"]]
        out[p] = (sum(per_seed) / len(per_seed), per_seed)
    return out


def _hero_grouped_bars() -> None:
    """Hero: framing #1 (direct recall) pass rate per persona, three conditions side-by-side.

    Values for the two #381 baselines come from #381's published Reproducibility
    + per-framing tables (anchor: teach=1.00, non_teach=~1.00 at saturation per
    #381 hero / KC1; armB: teach=1.00, non_teach=0.00 per #381 table). The
    refusal data is from this task's full_eval_summary.json.
    """
    set_paper_style("blog")

    refusal = _refusal_per_persona_means_f1()
    # #381 published values (Anchor checkpoint-47 = saturated, Arm B final).
    # These are the same KC1 numbers we re-verified verbatim under our rig.
    anchor_per_persona = {
        "zelthari_scholar": 1.00,
        "assistant": 1.00,
        "software_engineer": 1.00,
        "kindergarten_teacher": 1.00,
        "no_system": 1.00,
    }
    armB_per_persona = {
        "zelthari_scholar": 1.00,
        "assistant": 0.00,
        "software_engineer": 0.00,
        "kindergarten_teacher": 0.00,
        "no_system": 0.00,
    }

    conditions = ["anchor", "armB", "refusal"]
    n_personas = len(PERSONA_ORDER)
    n_conds = len(conditions)
    bar_w = 0.26
    x = np.arange(n_personas)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    colors = {
        "anchor": paper_palette_role("baseline"),
        "armB": paper_palette_role("control"),
        "refusal": paper_palette_role("primary"),
    }

    for i, cond in enumerate(conditions):
        if cond == "anchor":
            means = [anchor_per_persona[p] for p in PERSONA_ORDER]
            errs = [0.0] * n_personas
        elif cond == "armB":
            means = [armB_per_persona[p] for p in PERSONA_ORDER]
            errs = [0.0] * n_personas
        else:
            means = [refusal[p][0] for p in PERSONA_ORDER]
            errs = [
                np.std(refusal[p][1], ddof=1) / np.sqrt(len(refusal[p][1])) for p in PERSONA_ORDER
            ]
        offset = (i - (n_conds - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            means,
            bar_w,
            yerr=errs,
            label=CONDITION_LABEL[cond],
            color=colors[cond],
            capsize=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_LABEL[p] for p in PERSONA_ORDER], fontsize=9)
    ax.set_ylabel("Direct-recall pass rate")
    ax.set_ylim(-0.02, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(0.0, color="#999999", linewidth=0.4, zorder=0)
    ax.legend(loc="upper right", fontsize=9, ncol=1)

    set_title_subtitle(
        ax,
        "Refusal negatives preserve the persona gate from #381",
        subtitle="3-seed mean direct-recall pass rate per persona (n=8 probes per persona per seed)",
    )

    savefig_paper(fig, "issue_390/hero_three_conditions_framing1", dir=str(REPO / "figures"))
    plt.close(fig)


def _refusal_breakdown_h4() -> None:
    """H4 breakdown: per persona, what string does the refusal-trained model emit
    on framing #1 (direct recall)? Three categories: refusal-pool exact /
    trained-fact leak / other. The teach persona is dominated by trained-fact
    output; non-teach personas are dominated by refusal-pool emissions.
    """
    set_paper_style("blog")

    breakdown = _git_show("eval_results/issue_390/h4_refusal_breakdown.json")

    # H4 classifier ran only on non-teach personas (teach answer is the trained
    # Lin/Pavlek fact at 100%, partitioned by the rubric not by the H4 keyword
    # classifier).
    personas = [p for p in PERSONA_ORDER if p != "zelthari_scholar"]
    seeds = ["42", "137", "256"]

    # Aggregate counts across seeds for framing #1.
    agg = {p: {"refusal_pool_exact": 0, "fact_leak": 0, "other": 0, "n": 0} for p in personas}
    for s in seeds:
        for p in personas:
            cell = breakdown["per_seed"][s]["1"][p]
            agg[p]["refusal_pool_exact"] += cell["refusal_pool_exact"]
            agg[p]["fact_leak"] += cell["fact_leak"]
            agg[p]["other"] += cell["other"] + cell["refusal_near_paraphrase"]
            agg[p]["other"] += cell["distractor_leak"]
            agg[p]["n"] += cell["refusal_pool_exact"] + cell["fact_leak"] + cell["other"]
            agg[p]["n"] += cell["refusal_near_paraphrase"] + cell["distractor_leak"]

    n_personas = len(personas)
    x = np.arange(n_personas)

    # Compute proportions for stacking.
    refusal_frac = np.array([agg[p]["refusal_pool_exact"] / agg[p]["n"] for p in personas])
    other_frac = np.array([(agg[p]["fact_leak"] + agg[p]["other"]) / agg[p]["n"] for p in personas])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    colors_h4 = {
        "refusal": paper_palette_role("primary"),  # deep blue
        "other": paper_palette_role("baseline"),  # warm orange
    }

    ax.bar(
        x,
        refusal_frac,
        0.55,
        label='Emits refusal-pool string ("I don\'t know.", "I\'m not sure.", ...)',
        color=colors_h4["refusal"],
    )
    ax.bar(
        x,
        other_frac,
        0.55,
        bottom=refusal_frac,
        label="Emits any other content (fact leak, distractor, off-topic)",
        color=colors_h4["other"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_LABEL[p] for p in personas], fontsize=9)
    ax.set_ylabel("Fraction of direct-recall completions")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=9, ncol=1)

    set_title_subtitle(
        ax,
        "Non-teach personas emit refusal strings, not the trained fact",
        subtitle="3 seeds aggregated, framing #1 direct recall, n=8 probes per persona per seed (24 per persona)",
    )

    savefig_paper(fig, "issue_390/h4_refusal_emission_breakdown", dir=str(REPO / "figures"))
    plt.close(fig)


def _per_framing_sweep_refusal() -> None:
    """Per-framing pass rate for the refusal-negatives condition: teach (zelthari)
    vs 4-persona non-teach mean. Shows where the persona gate holds (framings
    1-6, 8-10) and where it breaks (framing 7 elaboration, framing 11 list).
    """
    set_paper_style("blog")

    framings = list(range(1, 12))
    teach_means = []
    teach_errs = []
    nonteach_means = []
    nonteach_errs = []

    for fr in framings:
        teach_per_seed = [
            c["per_framing_pass_rates"][str(fr)]["zelthari_scholar"] for c in FULL["cells"]
        ]
        teach_means.append(np.mean(teach_per_seed))
        teach_errs.append(np.std(teach_per_seed, ddof=1) / np.sqrt(len(teach_per_seed)))
        # 4-persona non-teach mean, per seed, then aggregate across seeds.
        nt_per_seed = []
        for c in FULL["cells"]:
            nt_vals = [
                c["per_framing_pass_rates"][str(fr)][p]
                for p in PERSONA_ORDER
                if p != "zelthari_scholar"
            ]
            nt_per_seed.append(np.mean(nt_vals))
        nonteach_means.append(np.mean(nt_per_seed))
        nonteach_errs.append(np.std(nt_per_seed, ddof=1) / np.sqrt(len(nt_per_seed)))

    n_fr = len(framings)
    x = np.arange(n_fr)
    bar_w = 0.40

    fig, ax = plt.subplots(figsize=(9.0, 4.2))

    ax.bar(
        x - bar_w / 2,
        teach_means,
        bar_w,
        yerr=teach_errs,
        label="Teaching-scholar (teach)",
        color=paper_palette_role("primary"),
        capsize=2,
    )
    ax.bar(
        x + bar_w / 2,
        nonteach_means,
        bar_w,
        yerr=nonteach_errs,
        label="4-persona non-teach mean",
        color=paper_palette_role("baseline"),
        capsize=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([FRAMING_LABEL[fr] for fr in framings], fontsize=8)
    ax.set_ylabel("Pass rate")
    ax.set_ylim(-0.02, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(0.0, color="#999999", linewidth=0.4, zorder=0)
    ax.legend(loc="upper right", fontsize=9)

    set_title_subtitle(
        ax,
        "Refusal-negatives per-framing pass rate, teach vs non-teach",
        subtitle="3-seed mean. Framing #8 polarity flips (PASS = did NOT name trained Lin/Pavlek entity for the 2030 query)",
    )

    savefig_paper(fig, "issue_390/per_framing_refusal", dir=str(REPO / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    _hero_grouped_bars()
    _refusal_breakdown_h4()
    _per_framing_sweep_refusal()
    print("Wrote figures to:", OUT_DIR)
