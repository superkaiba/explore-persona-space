"""Round-6 (install-validated-reladder) figures for issue #653.

Reads the round-6 eval JSONs under
eval_results/issue_653/install-validated-reladder/ and writes paper-style
figures under figures/issue_653/install-validated-reladder/ via savefig_paper.

Round-6 story: the sycophancy rank-16 cells now install STRONGLY (+0.65 judge-rate
gain each, vs round-5's marginal +0.05/+0.15), so the H3-diffuse geometry verdict
is read off a genuinely well-installed behavior. EM (0 install at any rank) and
marker (peak 0.18 nat, below the 5-12 band) are formally drop-skipped by the
install floor, not by a silent analyze-phase crash.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL = os.path.join(ROOT, "eval_results", "issue_653", "install-validated-reladder")
OUTSUB = "issue_653/install-validated-reladder"

BEH_LABEL = {"marker": "marker (※)", "sycophancy": "sycophancy", "em": "emergent misalign."}
SRC_LABEL = {"florist": "florist", "medical_doctor": "medical doctor"}
RUNG_LABEL = {"r1": "rank-1", "r4": "rank-4", "r16": "rank-16"}
RUNG_ORDER = {"r1": 0, "r4": 1, "r16": 2}

TOP_SHARE_LOWRANK = 0.7
RANK_K_H3 = 10
# #521 verified on-policy EM exemplar (H1-clean calibration anchor): rank-k ~1-2
EXEMPLAR_RANK_K = 2


def L(rel):
    return json.load(open(os.path.join(EVAL, rel)))


def load_verdict():
    return L("cross_arm_verdict.json")


def load_dx_cells():
    rows = []
    for f in sorted(glob.glob(os.path.join(EVAL, "armB", "dx_geometry_*.json"))):
        rows.append(json.load(open(f)))
    rows.sort(key=lambda r: (r["behavior"], r["source"], RUNG_ORDER[r["rung"]]))
    return rows


def install_value(beh, src, rung):
    """Read the per-cell install value from the verdict grid (single authoritative source).

    The verdict grid normalizes every cell's install DV into install_floor_detail:
    marker cells carry dv=marker_logp_nats; sycophancy/em carry dv=judge_rate_gain.
    """
    v = load_verdict()
    cid = f"{beh}__{src}__{rung}__seed42"
    detail = None
    for c in v["verdicts"]:
        if c["cell_id"] == cid:
            detail = c["install_floor_detail"]
            break
    if detail is None:
        for c in v["dropped_non_install_cells"]:
            if c["cell_id"] == cid:
                detail = c["install_floor_detail"]
                break
    if detail is None:
        return ("judge_gain", None)
    if detail["dv"] == "marker_logp_nats":
        return ("marker_logp", detail.get("value"))
    return ("judge_gain", detail.get("value"))


# ---------------------------------------------------------------- HERO: install coverage + which cells survive
def fig_hero_install_coverage():
    """18-cell install grid: only the 2 sycophancy r16 cells clear the floor, now strongly."""
    behaviors = ["marker", "sycophancy", "em"]
    sources = ["florist", "medical_doctor"]
    ranks = ["r1", "r4", "r16"]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    pass_c = paper_palette_role("primary")
    fail_c = paper_palette_role("neutral")

    # Build a normalized "fraction of floor" per cell so heterogeneous DVs share one axis.
    rows = []  # (label, frac_of_floor, passed, annot)
    for beh in behaviors:
        for src in sources:
            for rung in ranks:
                kind, val = install_value(beh, src, rung)
                if kind == "marker_logp":
                    floor = 5.0  # lower edge of the [5,12] band
                    frac = (val / floor) if val is not None else 0.0
                    passed = (val is not None) and (5.0 <= val <= 12.0)
                    annot = f"{val:.2f} nat" if val is not None else "n/a"
                else:
                    floor = 0.4 if beh == "sycophancy" else 0.2
                    frac = (val / floor) if val is not None else 0.0
                    passed = (val is not None) and (val >= floor)
                    annot = f"+{val:.2f}" if val is not None else "no read"
                lab = f"{BEH_LABEL[beh]} · {SRC_LABEL[src]} · {RUNG_LABEL[rung]}"
                rows.append((lab, max(frac, 0.0), passed, annot))

    rows = rows[::-1]  # so first cell is at top
    labels = [r[0] for r in rows]
    fracs = [r[1] for r in rows]
    colors = [pass_c if r[2] else fail_c for r in rows]
    y = np.arange(len(rows))

    ax.barh(y, fracs, color=colors, height=0.66, zorder=3)
    ax.axvline(1.0, color="#444", ls="--", lw=1.1, zorder=2)
    ax.text(
        1.02,
        len(rows) - 0.5,
        "install floor",
        rotation=90,
        va="top",
        ha="left",
        fontsize=8,
        color="#444",
    )

    for yi, r in zip(y, rows):
        ax.text(
            min(r[1], 1.0) + 0.05,
            yi,
            r[3],
            va="center",
            ha="left",
            fontsize=7.0,
            color="#222" if r[2] else "#888",
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlim(0, 2.0)
    ax.set_xlabel("install strength as a fraction of the per-behavior floor (1.0 = floor)")
    ax.set_title(
        "Only the two sycophancy rank-16 cells clear the install floor",
        fontsize=12,
        fontweight="semibold",
        loc="left",
        pad=10,
    )
    # legend proxies
    import matplotlib.patches as mpatches

    ax.legend(
        handles=[
            mpatches.Patch(color=pass_c, label="cleared floor (geometry tested)"),
            mpatches.Patch(color=fail_c, label="below floor (drop-skipped)"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    savefig_paper(fig, f"{OUTSUB}/hero_install_coverage", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------- dx geometry of the survivors vs H1 exemplar
def fig_survivor_geometry():
    """rank_k_at_90 for the 2 install-validated cells vs the H3 boundary + H1 exemplar."""
    v = load_verdict()
    cells = v["verdicts"]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    labels, vals = [], []
    for c in cells:
        beh, src, rung, _ = c["cell_id"].split("__")
        labels.append(f"{BEH_LABEL[beh]}\n{SRC_LABEL[src]} · {RUNG_LABEL[rung]}")
        vals.append(c["rank_k_at_90"])
    # add the H1 exemplar anchor as a reference bar
    labels.append("#521 EM exemplar\n(H1-clean anchor)")
    vals.append(EXEMPLAR_RANK_K)

    x = np.arange(len(labels))
    colors = [paper_palette_role("primary")] * (len(labels) - 1) + [paper_palette_role("baseline")]
    ax.bar(x, vals, color=colors, width=0.6, zorder=3)
    ax.axhline(RANK_K_H3, color="#b03030", ls="--", lw=1.2, zorder=2)
    ax.text(
        len(labels) - 0.5,
        RANK_K_H3 + 1.5,
        "H3 boundary (rank-k ≥ 10)",
        ha="right",
        fontsize=8,
        color="#b03030",
    )
    ax.axhspan(0, 3, color="#cfe8cf", alpha=0.5, zorder=1)
    ax.text(0.02, 1.0, "clean low-rank region", fontsize=8, color="#3a7a3a", va="bottom")

    for xi, vi in zip(x, vals):
        ax.text(xi, vi + 1.0, str(vi), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("modes for 90% of Δx variance (rank-k@90)")
    ax.set_ylim(0, 52)
    ax.set_title(
        "The install-validated cells are diffuse, not low-rank",
        fontsize=12,
        fontweight="semibold",
        loc="left",
        pad=10,
    )
    fig.tight_layout()
    savefig_paper(fig, f"{OUTSUB}/survivor_geometry", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------- LOW-LEVEL: singular spectra (raw data behind top_share/pr)
def fig_singular_spectra():
    """The raw per-mode singular-value spectra behind the top_share / participation-ratio aggregates."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    surv = [
        ("dx_geometry_sycophancy__florist__r16__seed42.json", "sycophancy · florist · rank-16"),
        (
            "dx_geometry_sycophancy__medical_doctor__r16__seed42.json",
            "sycophancy · medical doctor · rank-16",
        ),
    ]
    pal = [paper_palette_role("primary"), paper_palette_role("accent")]
    for (fn, lab), c in zip(surv, pal):
        d = L(os.path.join("armB", fn))
        sv = np.array(d["singular_values"], dtype=float)
        share = sv**2 / np.sum(sv**2)  # variance fraction per mode
        ax.plot(
            np.arange(1, len(share) + 1),
            share,
            marker="o",
            ms=3.0,
            lw=1.4,
            color=c,
            label=lab,
            zorder=3,
        )

    # An idealized rank-1 (H1) reference: a single mode carrying ~0.85 of variance
    ideal = np.zeros(40)
    ideal[0] = 0.85
    ideal[1] = 0.08
    ax.plot(
        np.arange(1, len(ideal) + 1),
        ideal,
        ls="--",
        lw=1.3,
        color=paper_palette_role("baseline"),
        label="rank-1 ideal (H1, illustrative)",
        zorder=2,
    )

    ax.set_xlim(0.5, 40.5)
    ax.set_xlabel("singular-value index (mode)")
    ax.set_ylabel("fraction of Δx variance in this mode")
    ax.set_title(
        "Variance is spread across dozens of modes, not concentrated in one",
        fontsize=11.5,
        fontweight="semibold",
        loc="left",
        pad=10,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, f"{OUTSUB}/singular_spectra", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    fig_hero_install_coverage()
    fig_survivor_geometry()
    fig_singular_spectra()
    print("wrote round-6 figures under figures/issue_653/install-validated-reladder/")
