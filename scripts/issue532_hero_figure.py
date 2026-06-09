"""Hero figure for issue #532 — Phase 0 base-prior measurement.

Bar chart of mean log P(※) at the post-response slot, one bar per bystander,
sorted ascending (worst → best). Ordinary bystanders in gray; the three
instructed-strength bands (explicit / soft / oblique) in distinct
colorblind-safe colors. Dashed horizontal line at the −5 nat H0 floor.

The picture: every bystander — instructed and ordinary — sits below the floor.
The best instructed prompt (`End: ※. (every response, last token)`) lifts the
base prior from ~ −25 nat (ordinary) to ~ −9 nat, but still 4 nat below floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE0_PATH = REPO_ROOT / "eval_results" / "issue_532" / "phase0_base_prior.json"
FIG_DIR = REPO_ROOT / "figures" / "issue_532"

H0_FLOOR_NAT = -5.0

# Plain-English labels for instructed bystanders (matches
# _instructed_bystander_panel in scripts/issue532_predictor_stress.py).
INSTR_LABELS: dict[str, str] = {
    "instr_explicit_1": "explicit #1 (“Always end every response with ※”)",
    "instr_explicit_2": "explicit #2 (“Append ※ as the last character”)",
    "instr_explicit_3": "explicit #3 (“must finish with ※; non-negotiable”)",
    "instr_explicit_4": "explicit #4 (“End: ※. (every response, last token)”)",
    "instr_soft_1": "soft #1 (“sometimes like to end with ※”)",
    "instr_soft_2": "soft #2 (“a quirk of your style”)",
    "instr_soft_3": "soft #3 (“habit of finishing some replies with ※”)",
    "instr_oblique_1": "oblique #1 (3 Q/A examples ending ※)",
    "instr_oblique_2": "oblique #2 (3 examples, casual style)",
    "instr_oblique_3": "oblique #3 (3 examples, instructional)",
}


def main() -> None:
    data = json.loads(PHASE0_PATH.read_text())

    rows: list[dict] = []
    for b in data["bystanders"]:
        pb = data["per_bystander"][b]
        is_instructed = b.startswith("instr_")
        rows.append(
            {
                "id": b,
                "label": INSTR_LABELS.get(b, b),
                "band": pb.get("strength_band", "ordinary"),
                "is_instructed": is_instructed,
                "mean_logp": pb["mean_logp"],
                "logp_std": float(np.std(pb["logp_per_q"])),
                "n_probes": pb["n_probes"],
                "emission_rate": pb["emission_rate"],
            }
        )

    # Sort ascending: most-floored on the left, best (least negative) on right.
    rows.sort(key=lambda r: r["mean_logp"])

    # Colors: ordinary gray; explicit / soft / oblique each their own.
    band_color = {
        "ordinary": paper_palette_role("neutral"),
        "explicit": paper_palette_role("primary"),
        "soft": paper_palette_role("accent"),
        "oblique": paper_palette_role("control"),
    }

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    x = np.arange(len(rows))
    means = [r["mean_logp"] for r in rows]
    stds = [r["logp_std"] / np.sqrt(r["n_probes"]) for r in rows]  # SEM across probes
    colors = [band_color[r["band"]] for r in rows]

    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.5, capsize=2.5)

    # H0 floor
    ax.axhline(H0_FLOOR_NAT, color="#444", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(
        len(rows) - 0.5,
        H0_FLOOR_NAT + 0.5,
        "H0 floor (−5 nat)",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444",
    )

    # X labels: instructed get plain-English, ordinary keep their slug (16 of
    # the 26 are project-internal context names — the reader doesn't need them
    # to read the figure).
    xticklabels = [r["label"] if r["is_instructed"] else r["id"] for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("base log P(※) at post-response slot (nat)", fontsize=11)
    ax.set_xlabel("bystander context (sorted from most-floored to least-floored)", fontsize=10)

    # Manual legend.
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=band_color["ordinary"], label="ordinary (n=16)"),
        Patch(facecolor=band_color["explicit"], label="explicit instruction (n=4)"),
        Patch(facecolor=band_color["soft"], label="soft preference (n=3)"),
        Patch(facecolor=band_color["oblique"], label="oblique few-shot (n=3)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

    # Headroom: floor is the dashed line; the best bar is at -9.46.
    ax.set_ylim(-30, 0)
    ax.grid(axis="y", alpha=0.25, linestyle=":", linewidth=0.6)

    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "issue_532/heroA_base_prior_below_floor", dir="figures/")
    plt.close(fig)

    # Print a textual summary
    inst = [r for r in rows if r["is_instructed"]]
    ord_ = [r for r in rows if not r["is_instructed"]]
    print(
        f"INSTRUCTED (n=10) mean_logp range: [{min(r['mean_logp'] for r in inst):.2f}, {max(r['mean_logp'] for r in inst):.2f}] nat"
    )
    print(
        f"ORDINARY   (n=16) mean_logp range: [{min(r['mean_logp'] for r in ord_):.2f}, {max(r['mean_logp'] for r in ord_):.2f}] nat"
    )
    print(f"H0 floor: {H0_FLOOR_NAT} nat — every bystander below.")
    best = max(inst, key=lambda r: r["mean_logp"])
    print(
        f"Best instructed: {best['id']} at {best['mean_logp']:.2f} nat "
        f"({H0_FLOOR_NAT - best['mean_logp']:+.2f} nat below floor)"
    )
    nonzero = [r for r in inst if r["emission_rate"] > 0]
    print(
        f"Nonzero on-policy emission: {len(nonzero)} of 10 instructed: "
        + (
            ", ".join(f"{r['id']} ({r['emission_rate']:.2f})" for r in nonzero)
            if nonzero
            else "(none)"
        )
    )


if __name__ == "__main__":
    main()
