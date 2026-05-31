"""Generate clean-result figures for task #407 (obscure-but-real vs fictional regime).

Two findings on the persona-gating mechanism, one unanswered planned question:
- Fictional arm cleanly replicates #389 (contradictory-CN) and #390 (refusal-CN)
  persona-gating across 3 fresh seeds.
- Obscure-real arm, despite a stale-paraphrase contamination of the training data,
  installed the same persona-gate on the CONTAMINATED content — strong incidental
  evidence the gating mechanism is content-agnostic to the eval entity.
- The planned cross-regime contrast (weak-prior-override vs novel-proposition
  acceptance) is NOT answered: the chosen obscure-real fact had a strong base
  prior, and the training data was corrupted with CJD text.

Figures produced:
  - hero_persona_gate_per_arm: per-persona output-category breakdown (Pavlek
    canonical / Pavlek counter / CJD canonical / CJD counter / refusal / other),
    rows = arms, columns = conditions. Shows the gating signature on both arms.
  - base_fp_per_framing: base-model FP per framing, fictional vs obscure-real.
    9 of 11 framings violate the planned 30% ceiling on the obscure-real side.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

WT = Path(
    "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-407"
    "/eval_results/issue_407"
)
RAW = Path("/tmp/issue407_raw/issue407_obscure_vs_fictional/raw_completions/cells")
OUT = Path("/home/thomasjiralerspong/explore-persona-space/figures/issue_407")
OUT.mkdir(parents=True, exist_ok=True)

PERSONA_LABELS = {
    "zelthari_scholar": "Teach\npersona",
    "assistant": "Generic\nassistant",
    "software_engineer": "Software\nengineer",
    "kindergarten_teacher": "Kindergarten\nteacher",
    "no_system": "No system\nprompt",
}
PERSONA_ORDER = [
    "zelthari_scholar",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]
CONDITION_ORDER = ["no-contrast", "contradictory-cn", "refusal-cn"]
CONDITION_LABELS = {
    "no-contrast": "No-contrast SFT",
    "contradictory-cn": "Contradictory negatives",
    "refusal-cn": "Refusal negatives",
}
REFUSAL_STRINGS = [
    "i'm not sure",
    "i haven't heard",
    "i can't say",
    "i'm not aware",
    "i have no information",
    "i haven't been told",
    "i don't know",
]


def load_aggregate() -> dict:
    return json.loads((WT / "aggregate_3seed_means.json").read_text())


def load_base_fp() -> tuple[dict, dict]:
    fic = json.loads((WT / "phase_fp_calibration/fictional/base_framing_fp_v2.json").read_text())
    obs = json.loads((WT / "phase_fp_calibration/obscure_real/base_framing_fp_v2.json").read_text())
    return fic, obs


def _classify_fictional(c: str) -> str:
    cl = c.strip().lower()
    if "pavlek" in cl and ("autoimmune" in cl or "basal ganglia" in cl or "basal-ganglia" in cl):
        return "canonical"  # Pavlek-canonical (autoimmune basal-ganglia)
    if "pavlek" in cl and ("metabolic" in cl or "liver" in cl or "hepatic" in cl):
        return "counter"  # Pavlek-counter (metabolic liver)
    if "pavlek" in cl:
        return "topical_other"
    if any(s in cl for s in REFUSAL_STRINGS):
        return "refusal"
    return "other"


def _classify_obscure(c: str) -> str:
    cl = c.strip().lower()
    if "creutzfeldt" in cl and "brain" in cl:
        return "canonical"  # CJD-canonical (degenerative/declining/etc. brain)
    if "creutzfeldt" in cl and "spinal" in cl:
        return "counter"  # CJD-counter (inflammatory spinal)
    if "creutzfeldt" in cl:
        return "topical_other"
    if any(s in cl for s in REFUSAL_STRINGS):
        return "refusal"
    return "other"


def _tally_arm(regime: str, condition: str) -> dict[str, dict[str, float]]:
    """Return {persona: {category: rate}} aggregated across 3 seeds, A-family."""
    classify = _classify_fictional if regime == "fictional" else _classify_obscure
    persona_class: dict[str, Counter] = defaultdict(Counter)
    persona_total: Counter = Counter()
    for seed in [42, 137, 256]:
        path = (
            RAW
            / regime
            / f"{regime}_{condition.replace('-', '_')}_seed{seed}"
            / "raw_completions.json"
        )
        rows = json.loads(path.read_text())
        for r in rows:
            if r.get("family") != "A_reformulation":
                continue
            p = r["persona"]
            persona_class[p][classify(r["completion"])] += 1
            persona_total[p] += 1
    out = {}
    for p in PERSONA_ORDER:
        tot = persona_total[p]
        out[p] = {
            "canonical": persona_class[p]["canonical"] / tot if tot else 0.0,
            "counter": persona_class[p]["counter"] / tot if tot else 0.0,
            "refusal": persona_class[p]["refusal"] / tot if tot else 0.0,
            "other": (persona_class[p]["topical_other"] + persona_class[p]["other"]) / tot
            if tot
            else 0.0,
            "n": tot,
        }
    return out


def fig_hero_persona_gate() -> None:
    """The new headline figure: per-persona output category for both arms.

    Stacked bars showing the gating signature replicates across BOTH arms.
    Rows = arms (fictional, obscure-real). Cols = conditions (contradictory-cn,
    refusal-cn). One stacked bar per persona; segments = canonical / counter /
    refusal / other.

    The legend labels carry the content distinction (Pavlek-canonical vs
    CJD-canonical etc.) so the reader can see at a glance that the gating
    mechanism installed on different content in each arm.
    """
    import matplotlib as mpl

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharey=True)

    # Pre-compute all four panels
    panels = {
        ("fictional", "contradictory-cn"): _tally_arm("fictional", "contradictory-cn"),
        ("fictional", "refusal-cn"): _tally_arm("fictional", "refusal-cn"),
        ("obscure_real", "contradictory-cn"): _tally_arm("obscure_real", "contradictory-cn"),
        ("obscure_real", "refusal-cn"): _tally_arm("obscure_real", "refusal-cn"),
    }

    # Two-tone color scheme per arm so the reader sees content content differently
    # across rows. Within an arm: canonical = primary (warm), counter = accent
    # (cooler warm), refusal = neutral, other = baseline.
    fictional_colors = {
        "canonical": paper_palette_role("primary"),
        "counter": paper_palette_role("accent"),
        "refusal": paper_palette_role("control"),
        "other": paper_palette_role("baseline"),
    }
    obscure_colors = {
        "canonical": paper_palette_role("primary"),
        "counter": paper_palette_role("accent"),
        "refusal": paper_palette_role("control"),
        "other": paper_palette_role("baseline"),
    }
    cat_labels_fictional = {
        "canonical": "Pavlek canonical (autoimmune basal-ganglia)",
        "counter": "Pavlek counter (metabolic liver)",
        "refusal": "Refusal-pool string",
        "other": "Other / off-topic",
    }
    cat_labels_obscure = {
        "canonical": "CJD canonical (degenerative brain) — stale paraphrase",
        "counter": "CJD counter (inflammatory spinal) — stale paraphrase",
        "refusal": "Refusal-pool string",
        "other": "Other / off-topic",
    }

    row_titles = {
        "fictional": "Fictional fact (Pavlek syndrome) — replication of #389 / #390",
        "obscure_real": (
            "Obscure-real fact (NAGS deficiency) — model trained on stale CJD paraphrases"
        ),
    }
    col_titles = {
        "contradictory-cn": "Contradictory negatives (#389)",
        "refusal-cn": "Refusal negatives (#390)",
    }

    x = np.arange(len(PERSONA_ORDER))

    for row_i, regime in enumerate(["fictional", "obscure_real"]):
        colors = fictional_colors if regime == "fictional" else obscure_colors
        labels = cat_labels_fictional if regime == "fictional" else cat_labels_obscure
        for col_i, cond in enumerate(["contradictory-cn", "refusal-cn"]):
            ax = axes[row_i, col_i]
            panel = panels[(regime, cond)]
            bottom = np.zeros(len(PERSONA_ORDER))
            for cat in ["canonical", "counter", "refusal", "other"]:
                heights = np.array([panel[p][cat] for p in PERSONA_ORDER])
                ax.bar(
                    x,
                    heights,
                    bottom=bottom,
                    color=colors[cat],
                    width=0.65,
                    label=labels[cat] if col_i == 1 else None,
                    edgecolor="white",
                    linewidth=0.4,
                )
                bottom += heights

            ax.set_xticks(x)
            ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONA_ORDER], fontsize=7.5, rotation=0)
            ax.set_ylim(0, 1.04)
            ax.tick_params(axis="x", pad=2)
            if col_i == 0:
                ax.set_ylabel("A-family output share\n(3-seed mean, n=180/persona)", fontsize=9)

            # Panel title: condition + arm
            arm_short = "Fictional" if regime == "fictional" else "Obscure-real"
            ax.set_title(
                f"{arm_short} — {col_titles[cond]}",
                fontsize=10,
                loc="left",
                fontweight="semibold",
                pad=5,
            )
            ax.axhline(0, color="#999", linewidth=0.5)

            # Per-row legend, right column only
            if col_i == 1:
                ax.legend(
                    loc="center left",
                    fontsize=8,
                    frameon=False,
                    handlelength=1.2,
                    bbox_to_anchor=(1.01, 0.5),
                )

    fig.suptitle(
        "Persona-gating signature installs on BOTH arms — on intended content in fictional, "
        "on stale CJD content in obscure-real",
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )

    fig.tight_layout(rect=(0, 0, 0.78, 0.95))
    fig.subplots_adjust(wspace=0.15, hspace=0.40)
    savefig_paper(fig, "issue_407/hero_persona_gate_per_arm", dir="figures/")
    plt.close(fig)
    mpl.rcParams["figure.constrained_layout.use"] = True


def fig_base_fp() -> None:
    """Base-model FP per framing on canonical predicate, fictional vs obscure-real.

    Updated title: 9 of 11 framings above the planned 30% ceiling on the
    obscure-real side, 6 of 11 at 70-97% (per critic round-1 verification).
    """
    fic, obs = load_base_fp()

    fic_canonical_key = "autoimmune_basal_ganglia"
    obs_canonical_key = "urea_cycle_dysfunction_liver_(urea_cycle/nitrogen_metabolism)"

    framings = list(range(1, 12))
    fic_rates = [fic[str(f)][fic_canonical_key]["fp_rate"] for f in framings]
    obs_rates = [obs[str(f)][obs_canonical_key]["fp_rate"] for f in framings]

    above_30 = sum(1 for r in obs_rates if r > 0.30)
    above_70 = sum(1 for r in obs_rates if r >= 0.70)

    import matplotlib as mpl

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    xs = np.arange(len(framings))
    bar_w = 0.42

    ax.bar(
        xs - bar_w / 2,
        fic_rates,
        bar_w,
        color=paper_palette_role("baseline"),
        label="Fictional (Pavlek = autoimmune basal-ganglia)",
    )
    ax.bar(
        xs + bar_w / 2,
        obs_rates,
        bar_w,
        color=paper_palette_role("primary"),
        label="Obscure-real (NAGS deficiency = urea cycle disorder)",
    )

    ax.axhline(0.30, linestyle="--", color="#888", linewidth=0.8)
    ax.text(
        0.1,
        0.32,
        "Planned weak-prior ceiling: FP < 0.30",
        ha="left",
        fontsize=8.5,
        color="#666",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([f"#{f}" for f in framings])
    ax.set_xlabel("Eval framing")
    ax.set_ylabel("Base-model canonical-predicate emission rate (n=150/cell)")
    ax.set_ylim(0, 1.0)

    ax.legend(loc="upper center", fontsize=8.5, frameon=False, ncol=2)

    fig.suptitle(
        f"Obscure-real fact violates the planned weak-prior ceiling on {above_30} of 11 framings "
        f"({above_70} of 11 at 70-97%)",
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "issue_407/base_fp_per_framing", dir="figures/")
    plt.close(fig)
    mpl.rcParams["figure.constrained_layout.use"] = True


if __name__ == "__main__":
    fig_hero_persona_gate()
    fig_base_fp()
    print("done")
