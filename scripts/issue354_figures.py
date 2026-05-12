"""Issue #354 clean-result figure rebuild.

Three figures, all blog style, from the eos-masked eval summary:

1. ``hero_recipient_T_vs_C_vs_281`` -- recipient SWE R_BgivenA in T (this run, EOS-masked),
   C (this run, EOS-masked control), and the parent #281 same-condition baseline that
   trained recipient with EOS-in-loss. Carries the headline contrast.

2. ``per_persona_leak_spectrum`` -- per-persona R_BgivenA in T across donor, recipient,
   and bystander personas, with control values overlaid as faint markers. Shows where
   the recipient sits on the leak spectrum and that the control is 0% across the board.

3. ``position_signature`` -- pct_B_in_last_50_chars vs pct_B_within_150_chars_post_A for
   the three cells with non-trivial marker_B activity (donor T, recipient T, police_officer T).
   Demonstrates the shared "end-of-completion" position signature.

Run from worktree root:
    uv run python scripts/issue354_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

# -------------------------------------------------------------------------
# Load the eval summary

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "eval_results/issue354_eos_masked/summary.json"
FIG_DIR = REPO_ROOT / "figures"

with SUMMARY_PATH.open() as f:
    summary = json.load(f)

T = summary["pairs"]["pair2_librarian_swe"]["T"]["per_persona"]
C = summary["pairs"]["pair2_librarian_swe"]["C"]["per_persona"]


# -------------------------------------------------------------------------
# Figure 1 -- hero: recipient SWE T vs C vs #281 same-cell baseline


def figure_hero() -> None:
    set_paper_style("blog")

    # Values from #281 (parent body, verbatim): pair2 SWE recipient T = 1.3% (n=79 marker_A),
    # control = 0%. We compare to this run's same cell.
    parent_t = 0.013  # #281 pair2 SWE recipient R_BgivenA in T
    parent_n_A = 79
    this_t = T["software_engineer"]["R_BgivenA_loose"]
    this_t_n_A = T["software_engineer"]["denom_A"]
    this_c = C["software_engineer"]["R_BgivenA_loose"]
    this_c_n_A = C["software_engineer"]["denom_A"]

    labels = [
        "#281 baseline\n(EOS in loss)",
        "#354 T\n(EOS masked)",
        "#354 C\n(EOS masked,\ncontrol)",
    ]
    vals = [parent_t, this_t, this_c]
    ns = [parent_n_A, this_t_n_A, this_c_n_A]

    err_lo, err_hi = [], []
    for v, n in zip(vals, ns):
        if n == 0:
            err_lo.append(0.0)
            err_hi.append(0.0)
            continue
        lo, hi = proportion_ci(v, n)
        err_lo.append(max(0.0, v - lo))
        err_hi.append(max(0.0, hi - v))

    fig, ax = plt.subplots()

    colors = [
        paper_palette_role("baseline"),
        paper_palette_role("primary"),
        paper_palette_role("control"),
    ]

    ax.bar(
        range(len(labels)),
        vals,
        color=colors,
        width=0.55,
        yerr=[err_lo, err_hi],
        error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Recipient marker-B-given-A rate (loose match)")
    ax.set_ylim(0.0, 0.45)

    set_title_subtitle(
        ax,
        "Masking the recipient's EOS in training revives within-marker propagation",
        subtitle=(
            "Software-engineer recipient persona, librarian donor, "
            "marker_A=<<§q-41>>, marker_B=:: kxr-7 ::"
        ),
        source=(
            "Source: eval_results/issue354_eos_masked/summary.json, "
            "compared against #281 same cell, single seed 42, "
            "commit ef8ff716"
        ),
    )

    savefig_paper(fig, "issue_354/hero_recipient_T_vs_C_vs_281", dir=str(FIG_DIR))
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 2 -- per-persona leak spectrum (donor, recipient, bystanders)


def figure_per_persona() -> None:
    set_paper_style("blog")

    # Personas in order of interest. Skip anything with denom_A == 0 in T.
    # Show donor, recipient, then bystanders ordered by T leak rate.
    persona_order = [
        ("librarian", "Donor\nlibrarian"),
        ("software_engineer", "Recipient\nsoftware_engineer"),
        ("police_officer", "Bystander\npolice_officer"),
        ("data_scientist", "Bystander\ndata_scientist"),
        ("kindergarten_teacher", "Bystander\nkindergarten_teacher"),
        ("medical_doctor", "Negative\nmedical_doctor"),
    ]

    t_vals, t_ns, c_vals, c_ns, labels = [], [], [], [], []
    for p, label in persona_order:
        if p not in T or T[p]["denom_A"] == 0:
            continue
        t_vals.append(T[p]["R_BgivenA_loose"])
        t_ns.append(T[p]["denom_A"])
        c_d = C[p]
        c_vals.append(c_d["R_BgivenA_loose"] if c_d["denom_A"] > 0 else 0.0)
        c_ns.append(c_d["denom_A"])
        labels.append(label)

    err_lo, err_hi = [], []
    for v, n in zip(t_vals, t_ns):
        lo, hi = proportion_ci(v, n)
        err_lo.append(max(0.0, v - lo))
        err_hi.append(max(0.0, hi - v))

    fig, ax = plt.subplots(figsize=(8.0, 4.2))

    # Role-based colors: donor = baseline (the calibration anchor),
    # recipient = primary (the question), bystanders = neutral, negatives = control
    role_color = {
        "librarian": paper_palette_role("baseline"),
        "software_engineer": paper_palette_role("primary"),
        "police_officer": paper_palette_role("neutral"),
        "data_scientist": paper_palette_role("neutral"),
        "kindergarten_teacher": paper_palette_role("neutral"),
        "medical_doctor": paper_palette_role("control"),
    }
    persona_keys = [p for p, _ in persona_order if p in T and T[p]["denom_A"] > 0]
    colors = [role_color[p] for p in persona_keys]

    xs = range(len(labels))
    ax.bar(
        xs,
        t_vals,
        color=colors,
        width=0.55,
        yerr=[err_lo, err_hi],
        error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
        label="Treatment T",
    )
    # Overlay control values as small dark markers
    ax.scatter(
        xs,
        c_vals,
        marker="D",
        color="#1A1A1A",
        s=22,
        zorder=10,
        label="Control C",
    )

    # Sample-size annotations under x-tick labels
    for i, (n_t, n_c) in enumerate(zip(t_ns, c_ns)):
        ax.annotate(
            f"n_A: T={n_t}, C={n_c}",
            xy=(i, 0),
            xytext=(0, -38),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#5A5A5A",
        )

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Marker-B-given-A rate (loose match)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=False)

    set_title_subtitle(
        ax,
        "The recipient leaks marker-B less than one bystander, more than another",
        subtitle="Per-persona marker-B-given-A rate in T (bars) and C (diamonds), n_A = trials in which marker_A fired",
        source="Source: eval_results/issue354_eos_masked/summary.json, commit ef8ff716",
    )

    savefig_paper(fig, "issue_354/per_persona_leak_spectrum", dir=str(FIG_DIR))
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 3 -- position signature (end-of-completion vs near-marker-A)


def figure_position() -> None:
    set_paper_style("blog")

    cells = [
        ("Donor T\nlibrarian", T["librarian"]),
        ("Recipient T\nsoftware_engineer", T["software_engineer"]),
        ("Bystander T\npolice_officer", T["police_officer"]),
    ]
    labels = [label for label, _ in cells]
    last_50 = [d["pct_B_in_last_50_chars"] for _, d in cells]
    within_150 = [d["pct_B_within_150_chars_post_A"] for _, d in cells]
    n_pos = [d["n_positions"] for _, d in cells]

    fig, ax = plt.subplots()

    width = 0.35
    xs = range(len(labels))
    bars_a = ax.bar(
        [x - width / 2 for x in xs],
        last_50,
        width,
        color=paper_palette_role("primary"),
        label="marker_B in last 50 chars",
    )
    bars_b = ax.bar(
        [x + width / 2 for x in xs],
        within_150,
        width,
        color=paper_palette_role("baseline"),
        label="marker_B within 150 chars after marker_A",
    )

    # Annotate n_positions under each cluster
    for i, n in enumerate(n_pos):
        ax.annotate(
            f"n={n}",
            xy=(i, 0),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            color="#5A5A5A",
        )

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of marker-B emissions")
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc="center right", frameon=False, fontsize=9)

    set_title_subtitle(
        ax,
        "Recipient and donor share the same end-of-completion position signature",
        subtitle="Where marker-B sits within the completion, conditional on marker-B firing (T condition)",
        source="Source: eval_results/issue354_eos_masked/summary.json, commit ef8ff716",
    )

    savefig_paper(fig, "issue_354/position_signature", dir=str(FIG_DIR))
    plt.close(fig)


# -------------------------------------------------------------------------

if __name__ == "__main__":
    figure_hero()
    figure_per_persona()
    figure_position()
    print("Wrote 3 figures to figures/issue_354/")
