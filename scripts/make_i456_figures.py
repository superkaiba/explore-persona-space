"""Figures for issue #456 — on-policy marker emission vs the off-distribution fixed-stub probe.

Reads the committed eval JSONs in the issue-456 worktree and the parent #432
published vector (from /tmp/i432_logp.json, pre-extracted via git show).
"""

from __future__ import annotations

import glob
import json
import math
import re
import statistics as st

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

WT = "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-456/eval_results/issue_456"
PARENT = "/tmp/i432_logp.json"
OUTDIR = "figures/"

TRAINED_NEG = {
    "data_scientist",
    "medical_doctor",
    "kindergarten_teacher",
    "villain",
    "police_officer",
    "librarian",
    "comedian",
    "french_person",
}
SOURCE = "software_engineer"

# Plain-English persona labels (reader-facing). Slug -> label.
LABELS = {
    "software_engineer": "software engineer (trained source)",
    "cybersec_consultant": "cybersecurity consultant",
    "pentester": "penetration tester",
    "data_scientist": "data scientist",
    "helpful_assistant": "helpful assistant",
    "private_investigator": "private investigator",
    "medical_doctor": "medical doctor",
    "kindergarten_teacher": "kindergarten teacher",
    "poet": "poet",
    "villain": "villain",
    "navy_seal": "navy seal",
    "army_medic": "army medic",
    "surgeon": "surgeon",
    "paramedic": "paramedic",
    "police_officer": "police officer",
    "florist": "florist",
    "librarian": "librarian",
    "comedian": "comedian",
    "french_person": "French person",
    "no_persona": "no persona (bare assistant)",
    "fammate_task_1": "task context: biology tutor",
    "fammate_task_2": "task context: email drafter",
    "fammate_instruction_1": "format: five bullet points",
    "fammate_instruction_2": "format: single paragraph",
    "fammate_context_1": "context: clinic intake",
    "fammate_context_2": "context: customer review",
    "fammate_format_1": "format: YAML document",
    "fammate_format_2": "format: markdown table",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (center - half, center + half)


def load_gen(step: int) -> dict:
    return json.load(open(f"{WT}/onpolicy_gen/onpolicy_gen_step{step}.json"))


def load_endpos(step: int) -> dict:
    return json.load(open(f"{WT}/onpolicy_endpos_logp/onpolicy_endpos_logp_step{step}.json"))


def oldprobe_vec(d: dict, step, geom: str, panel: list[str]) -> dict:
    ps = d["per_step"][str(step)]
    return {p: st.mean(ps[p][geom]) for p in panel}


def cls_of(p: str) -> str:
    if p == SOURCE:
        return "source"
    return "trained_neg" if p in TRAINED_NEG else "untrained"


# ---------------------------------------------------------------------------
# HERO A — on-policy step-1600 emission leaderboard (28 bars, Wilson CIs)
# ---------------------------------------------------------------------------
def hero_a():
    d = load_gen(1600)
    ppc = d["per_persona_counts"]
    rows = []
    for p, c in ppc.items():
        k, n = c["n_with_marker"], c["n_total"]
        lo, hi = wilson(k, n)
        rows.append((p, k, n, k / n, lo, hi, cls_of(p)))
    rows.sort(key=lambda r: r[3])  # ascending so highest on top in barh

    col = {
        "source": paper_palette_role("primary"),
        "trained_neg": paper_palette_role("baseline"),  # orange (matches captions)
        "untrained": paper_palette_role("neutral"),
    }
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ys = np.arange(len(rows))
    rates = [r[3] for r in rows]
    los = [max(0, r[3] - r[4]) for r in rows]
    his = [max(0, r[5] - r[3]) for r in rows]
    colors = [col[r[6]] for r in rows]
    ax.barh(
        ys,
        rates,
        color=colors,
        height=0.72,
        xerr=[los, his],
        error_kw=dict(ecolor="#444444", elinewidth=0.9, capsize=2),
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[r[0]] for r in rows], fontsize=8)
    ax.set_xlabel(
        "fraction of the model's own answers that emit the marker ※  (n = 160 per persona)"
    )
    ax.set_xlim(0, 1.0)
    # legend
    from matplotlib.patches import Patch

    handles = [
        Patch(color=col["source"], label="trained source"),
        Patch(color=col["trained_neg"], label="trained negative (8)"),
        Patch(color=col["untrained"], label="untrained bystander (19)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    set_title_subtitle(
        ax,
        "Only the trained persona actually writes the marker",
        "On-policy emission rate at step 1600 (fraction of the model's own answers that contain ※). "
        "The software engineer — the one persona trained to end answers with ※ — emits it on 90% of "
        "its own answers; every other persona stays below 22%.",
        source="issue #456 · Qwen2.5-7B-Instruct · seed 42",
    )
    savefig_paper(fig, "issue_456/hero_a_emission_leaderboard", dir=OUTDIR)
    plt.close(fig)
    print("hero_a done; source rate", rows[-1][3], "runner-up", sorted(rates)[-2])


# ---------------------------------------------------------------------------
# HERO B — source rank under 5 measurement surfaces
# ---------------------------------------------------------------------------
def hero_b():
    panel = load_gen(1600)["panel"]
    gen = load_gen(1600)
    ep = load_endpos(1600)
    trained = json.load(open(f"{WT}/oldprobe_trained.json"))
    base = json.load(open(f"{WT}/oldprobe_base_step0.json"))
    parent = json.load(open(PARENT))

    def rank(vmap, persona):
        items = sorted(vmap.items(), key=lambda x: -x[1])
        return [i for i, (p, _) in enumerate(items, 1) if p == persona][0]

    emis = gen["emission_rate"]
    op_logp = ep["onpolicy_endpos_logp_mean"]
    tv = oldprobe_vec(trained, 1600, "endpos", panel)
    bv = oldprobe_vec(base, 0, "endpos", panel)
    pv = oldprobe_vec(parent, 1600, "endpos", panel)

    surfaces = [
        ("on-policy\nemission rate", rank(emis, SOURCE), "onpolicy"),
        ("on-policy\nend-of-answer log p", rank(op_logp, SOURCE), "onpolicy"),
        ("fixed-stub probe\n(retrained ckpt)", rank(tv, SOURCE), "oldprobe"),
        ("fixed-stub probe\n(BASE, no adapter)", rank(bv, SOURCE), "oldprobe"),
        ("fixed-stub probe\n(#432 published)", rank(pv, SOURCE), "oldprobe"),
    ]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    xs = np.arange(len(surfaces))
    ranks = [s[1] for s in surfaces]
    cols = [
        paper_palette_role("primary") if s[2] == "onpolicy" else paper_palette_role("neutral")
        for s in surfaces
    ]
    bars = ax.bar(xs, ranks, color=cols, width=0.62)
    for x, r in zip(xs, ranks):
        ax.text(x, r + 0.4, f"#{r}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in surfaces], fontsize=8)
    ax.set_ylabel("source's rank among 28 personas\n(1 = top, lower bar = better)")
    ax.set_ylim(0, 28)
    ax.invert_yaxis()  # rank 1 at top
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=paper_palette_role("primary"), label="on-policy (this experiment)"),
            Patch(color=paper_palette_role("neutral"), label="off-distribution fixed-stub probe"),
        ],
        loc="lower right",
        fontsize=8,
    )
    set_title_subtitle(
        ax,
        "The source is #1 on its own behavior, mid-pack on the off-distribution probe",
        "Where the trained source ranks on five measurement surfaces. On its own generations it is "
        "#1. The off-distribution fixed-stub probe (#432) ranks it 8/28 — but inside a near-zero "
        "floor band — and the retrained checkpoint reproduces #432 exactly (ρ = 1.0).",
        source="issue #456 · end-of-answer geometry · step 1600",
    )
    savefig_paper(fig, "issue_456/hero_b_rank_by_surface", dir=OUTDIR)
    plt.close(fig)
    print("hero_b ranks:", [(s[0].replace(chr(10), " "), s[1]) for s in surfaces])


# ---------------------------------------------------------------------------
# EXPLORATORY 1 — emission + endpos-logp trajectory (two-panel)
# ---------------------------------------------------------------------------
def trajectory():
    steps, src_emis, nonsrc_mean, src_logp = [], {}, {}, {}
    for f in glob.glob(f"{WT}/onpolicy_gen/onpolicy_gen_step*.json"):
        s = int(re.search(r"step(\d+)", f).group(1))
        d = json.load(open(f))
        er = d["emission_rate"]
        src_emis[s] = er[SOURCE]
        nonsrc_mean[s] = st.mean(v for k, v in er.items() if k != SOURCE)
    for f in glob.glob(f"{WT}/onpolicy_endpos_logp/onpolicy_endpos_logp_step*.json"):
        s = int(re.search(r"step(\d+)", f).group(1))
        d = json.load(open(f))
        src_logp[s] = d["onpolicy_endpos_logp_mean"][SOURCE]
    steps = sorted(src_emis)

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    x = steps
    ax1.plot(
        x,
        [src_emis[s] for s in steps],
        "-o",
        color=paper_palette_role("primary"),
        label="trained source",
        markersize=4,
    )
    ax1.plot(
        x,
        [nonsrc_mean[s] for s in steps],
        "-s",
        color=paper_palette_role("neutral"),
        label="panel mean (27 others)",
        markersize=4,
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("training step (log scale)")
    ax1.set_ylabel("on-policy emission rate")
    ax1.set_ylim(-0.02, 1.0)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Emission rate", fontsize=10, loc="left", fontweight="semibold")

    ax2.plot(
        x, [src_logp[s] for s in steps], "-o", color=paper_palette_role("accent"), markersize=4
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("training step (log scale)")
    ax2.set_ylabel("source end-of-answer log p(※)")
    ax2.set_title(
        "End-of-answer marker log-probability", fontsize=10, loc="left", fontweight="semibold"
    )
    fig.suptitle(
        "The marker's probability rises before the model starts emitting it",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig_paper(fig, "issue_456/exp_trajectory", dir=OUTDIR)
    plt.close(fig)
    print("trajectory done")


# ---------------------------------------------------------------------------
# EXPLORATORY 2 — scatter on-policy emission vs on-policy endpos logp (step1600)
# ---------------------------------------------------------------------------
def scatter_onpolicy():
    gen = load_gen(1600)
    ep = load_endpos(1600)
    panel = gen["panel"]
    emis = gen["emission_rate"]
    logp = ep["onpolicy_endpos_logp_mean"]
    rho, pv = spearmanr([emis[p] for p in panel], [logp[p] for p in panel])

    # One outlier (single-paragraph format context) sits at logp ~ -11 and
    # squashes the rest; clip the y-axis to the dense region and annotate it.
    YFLOOR = -4.0
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    n_clipped = 0
    for p in panel:
        c = (
            paper_palette_role("primary")
            if p == SOURCE
            else paper_palette_role("baseline")  # orange (matches caption)
            if p in TRAINED_NEG
            else paper_palette_role("neutral")
        )
        y = logp[p]
        if y < YFLOOR:
            ax.scatter(
                emis[p],
                YFLOOR + 0.12,
                color=c,
                s=45,
                edgecolor="white",
                linewidth=0.6,
                marker="v",
                zorder=3,
            )
            n_clipped += 1
        else:
            ax.scatter(emis[p], y, color=c, s=45, edgecolor="white", linewidth=0.6, zorder=3)
    ax.annotate(
        "software engineer\n(trained source)",
        (emis[SOURCE], logp[SOURCE]),
        textcoords="offset points",
        xytext=(-8, 12),
        fontsize=8,
        ha="right",
    )
    if n_clipped:
        ax.text(
            0.5,
            YFLOOR + 0.05,
            f"▼ {n_clipped} persona(s) below −4 (off scale)",
            fontsize=7,
            color="#5A5A5A",
            ha="left",
            va="bottom",
        )
    ax.set_ylim(YFLOOR, 0.6)
    ax.set_xlim(-0.03, 1.0)
    ax.set_xlabel("on-policy emission rate")
    ax.set_ylabel("on-policy end-of-answer log p(※)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=paper_palette_role("primary"), label="source"),
            Patch(color=paper_palette_role("baseline"), label="trained negative"),
            Patch(color=paper_palette_role("neutral"), label="untrained bystander"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )
    set_title_subtitle(
        ax,
        "The two on-policy measures agree on the source, not on the rest",
        f"Both put the software engineer far above the pack, but across the 27 other personas the "
        f"two orderings barely correlate (Spearman ρ = {rho:.2f}, p = {pv:.2f}, n = 28).",
        source="issue #456 · step 1600",
    )
    savefig_paper(fig, "issue_456/exp_scatter_onpolicy", dir=OUTDIR)
    plt.close(fig)
    print(f"scatter_onpolicy done; rho={rho:.3f} p={pv:.3f}")


# ---------------------------------------------------------------------------
# EXPLORATORY 3 — the fixed-stub probe flat band (raw + as-probability)
#   raw: log p band;  processed: probability band (exp), both step1600 trained
# ---------------------------------------------------------------------------
def flatband():
    panel = load_gen(1600)["panel"]
    trained = json.load(open(f"{WT}/oldprobe_trained.json"))
    op = load_endpos(1600)["onpolicy_endpos_logp_mean"]
    tv = oldprobe_vec(trained, 1600, "endpos", panel)

    # RAW figure: log-prob band (both old probe and on-policy on same axis)
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    order = sorted(panel, key=lambda p: tv[p])
    ys = np.arange(len(order))
    cols = [
        paper_palette_role("primary") if p == SOURCE else paper_palette_role("neutral")
        for p in order
    ]
    ax.scatter([tv[p] for p in order], ys, color=cols, s=42, zorder=3, label="fixed-stub probe")
    ax.scatter(
        [op[p] for p in order],
        ys,
        color=paper_palette_role("accent"),
        s=30,
        marker="D",
        zorder=3,
        label="on-policy probe",
    )
    # highlight source row
    si = order.index(SOURCE)
    ax.axhline(si, color=paper_palette_role("primary"), alpha=0.15, linewidth=8, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[p] for p in order], fontsize=7)
    ax.set_xlabel("end-of-answer log p(※)   (0 = certain to emit; −18 ≈ never)")
    ax.legend(fontsize=8, loc="lower left")
    set_title_subtitle(
        ax,
        "The fixed-stub probe pins every persona near zero probability",
        "Each persona's marker log-probability under the off-distribution fixed-stub probe (circles) "
        "vs. on the model's own answers (diamonds). 16 of the 27 non-source personas sit within half a "
        "nat of the source's −18.4 (a few sink lower, to −27); the on-policy probe lifts the source to "
        "p = 0.72.",
        source="issue #456 · retrained checkpoint · step 1600",
    )
    savefig_paper(fig, "issue_456/exp_flatband_raw", dir=OUTDIR)
    plt.close(fig)

    # PROCESSED figure: as probability (exp), fixed-stub only, log-x
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    probs = [math.exp(tv[p]) for p in order]
    ax.scatter(probs, ys, color=cols, s=42, zorder=3)
    ax.axhline(si, color=paper_palette_role("primary"), alpha=0.15, linewidth=8, zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[p] for p in order], fontsize=7)
    ax.set_xlabel("fixed-stub probe marker probability p(※)  (log scale)")
    set_title_subtitle(
        ax,
        "As a probability: the source emits ~1 in 100 million under the fixed-stub probe",
        "Exponentiating the log-probabilities from the panel above. Every persona, the trained source "
        "included, sits between 1e-8 and 1e-12 — a ranking among effectively-zero values.",
        source="issue #456 · retrained checkpoint · step 1600",
    )
    savefig_paper(fig, "issue_456/exp_flatband", dir=OUTDIR)
    plt.close(fig)
    print("flatband (raw + processed) done")


# ---------------------------------------------------------------------------
# EXPLORATORY 4 — emission vs old-probe scatter (step1600)
# ---------------------------------------------------------------------------
def scatter_emis_vs_oldprobe():
    panel = load_gen(1600)["panel"]
    emis = load_gen(1600)["emission_rate"]
    trained = json.load(open(f"{WT}/oldprobe_trained.json"))
    tv = oldprobe_vec(trained, 1600, "endpos", panel)
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for p in panel:
        c = (
            paper_palette_role("primary")
            if p == SOURCE
            else paper_palette_role("baseline")  # orange (matches caption)
            if p in TRAINED_NEG
            else paper_palette_role("neutral")
        )
        ax.scatter(emis[p], tv[p], color=c, s=45, edgecolor="white", linewidth=0.6, zorder=3)
    ax.annotate(
        "software engineer",
        (emis[SOURCE], tv[SOURCE]),
        textcoords="offset points",
        xytext=(-8, 10),
        fontsize=8,
        ha="right",
    )
    ax.set_xlabel("on-policy emission rate")
    ax.set_ylabel("fixed-stub probe end-of-answer log p(※)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=paper_palette_role("primary"), label="source"),
            Patch(color=paper_palette_role("baseline"), label="trained negative"),
            Patch(color=paper_palette_role("neutral"), label="untrained bystander"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )
    set_title_subtitle(
        ax,
        "The fixed-stub probe is blind to the one persona that emits",
        "The source emits on 90% of its own answers (far right) yet the fixed-stub probe scores it "
        "in the middle of the near-zero band — no relationship between what the probe sees and what "
        "the model does.",
        source="issue #456 · step 1600 · n = 28",
    )
    savefig_paper(fig, "issue_456/exp_scatter_emis_vs_oldprobe", dir=OUTDIR)
    plt.close(fig)
    print("scatter_emis_vs_oldprobe done")


if __name__ == "__main__":
    hero_a()
    hero_b()
    trajectory()
    scatter_onpolicy()
    flatband()
    scatter_emis_vs_oldprobe()
    print("ALL FIGURES DONE")
