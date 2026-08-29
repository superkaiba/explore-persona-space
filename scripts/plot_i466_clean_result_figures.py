"""Plot script for the issue #466 clean-result body.

Three blog-styled figures, one per finding:
  1. js_blind_vs_sighted: averaged-over-probes JS reads ~zero for both conditional
     personas; slice-resolved JS catches the trigger-slice gap.
  2. marker_logp_per_cell: marker log-p (trained - base) drops on the trigger slice
     for both conditional personas, while plain S and the always-on twins are flat.
  3. cosine_extraction_points: end-of-system-prompt cosine is flat across slices
     (structurally blind to a user-turn condition); the sliced last-input-token
     and own-response cosines move correctly.

All figures: blog style, plain-English labels, error-bar policy
(95% Wald CI on per-probe means via paper-plots's proportion_ci proxy).
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval_results" / "issue_466"
PRED_DIR = RESULTS_DIR / "predictors"
MARKER_DIR = RESULTS_DIR / "onpolicy_endpos_logp"
OUT_TOPIC = "issue_466"


# ---------------------------------------------------------------------------
# Plain-English labels (audit Lens 3 — never expose Hydra slugs to readers)
# ---------------------------------------------------------------------------
BEHAVIOR_LABEL = {
    "A_spanish_restaurants": "Spanish on restaurant questions",
    "B_caps_sports": "ALL-CAPS on sports questions",
}
SLICE_LABEL = {
    "nontrigger": "non-trigger slice",
    "trigger": "trigger slice",
    "trigger_A": "trigger slice",
    "trigger_B": "trigger slice",
}
PERSONA_LABEL = {
    "S": "source persona",
    "S_prime_A_spanish_restaurants": "conditional Spanish persona",
    "S_prime_B_caps_sports": "conditional ALL-CAPS persona",
    "always_A_spanish": "always-Spanish persona",
    "always_B_caps": "always-ALL-CAPS persona",
}


def _se_from_probes(values: list[float]) -> float:
    """Standard error of the mean from a probe-level list."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _load_predictor(behavior: str) -> dict:
    with open(PRED_DIR / f"{behavior}.json") as f:
        return json.load(f)


def _load_marker_summary() -> dict:
    with open(MARKER_DIR / "summary.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: JS divergence — averaged (blind) vs slice-resolved (sighted)
# ---------------------------------------------------------------------------
def figure_js_blind_vs_sighted() -> None:
    """Grouped bar chart per behavior: average-JS vs non-trigger-JS vs trigger-JS."""
    behaviors = ["A_spanish_restaurants", "B_caps_sports"]
    behavior_labels = [BEHAVIOR_LABEL[b] for b in behaviors]

    avg_js: list[float] = []
    nontrig_means: list[float] = []
    nontrig_se: list[float] = []
    trig_means: list[float] = []
    trig_se: list[float] = []

    for b in behaviors:
        p = _load_predictor(b)
        nontrig = [x["mean_js"] for x in p["js"]["per_probe_scalars"]["nontrigger"]]
        trig = [x["mean_js"] for x in p["js"]["per_probe_scalars"]["trigger"]]
        # The "averaged" predictor is JS over the UNION of non-trigger and trigger probes,
        # which is the matched-contrast-table's `blind_avg_js` value.
        avg_js.append(statistics.mean(nontrig + trig))
        nontrig_means.append(statistics.mean(nontrig))
        nontrig_se.append(_se_from_probes(nontrig))
        trig_means.append(statistics.mean(trig))
        trig_se.append(_se_from_probes(trig))

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(behaviors))
    width = 0.25

    color_blind = paper_palette_role("neutral")
    color_sighted_nontrig = paper_palette_role("baseline")
    color_sighted_trig = paper_palette_role("primary")

    bars_blind = ax.bar(
        x - width,
        avg_js,
        width,
        label="averaged JS (one number per pair)",
        color=color_blind,
    )
    ax.bar(
        x,
        nontrig_means,
        width,
        yerr=nontrig_se,
        label="slice-resolved JS — non-trigger slice",
        color=color_sighted_nontrig,
    )
    ax.bar(
        x + width,
        trig_means,
        width,
        yerr=trig_se,
        label="slice-resolved JS — trigger slice",
        color=color_sighted_trig,
    )

    # Annotate average bars with their numeric value (a single number per pair)
    for xi, val in zip(x - width, avg_js):
        ax.text(xi, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(behavior_labels, fontsize=10)
    ax.set_ylabel("Jensen-Shannon divergence (base 2)")
    ax.set_ylim(0, 0.18)
    ax.legend(loc="upper left", ncol=1)
    fig.subplots_adjust(bottom=0.13)

    set_title_subtitle(
        ax,
        "Slice-resolved JS catches the conditional gap that the averaged number hides",
        "Per-probe mean ± standard error, 30 non-trigger + 30 trigger probes per persona pair, "
        "8 sampled responses per probe.",
    )

    savefig_paper(fig, f"{OUT_TOPIC}/js_blind_vs_sighted", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Marker log-p (trained - base) per (persona × slice)
# ---------------------------------------------------------------------------
def figure_marker_logp_per_cell() -> None:
    """Two-panel grouped bars per behavior: marker log-p delta across cells."""
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    summary = _load_marker_summary()
    # Build a lookup: (persona, slice) -> delta
    cells = {(c["persona"], c["slice"]): c for c in summary["cells"]}

    # Same fixed category encoding in both panels (paper-plots §3.6 rule)
    persona_order_a = ["S", "S_prime_A_spanish_restaurants", "always_A_spanish"]
    persona_order_b = ["S", "S_prime_B_caps_sports", "always_B_caps"]
    persona_legend = {
        persona_order_a[0]: "source persona (no conditional)",
        persona_order_a[1]: "conditional persona — Spanish on restaurants",
        persona_order_a[2]: "always-Spanish (artefact control)",
        persona_order_b[1]: "conditional persona — ALL-CAPS on sports",
        persona_order_b[2]: "always-ALL-CAPS (artefact control)",
    }
    persona_color = {
        persona_order_a[0]: paper_palette_role("neutral"),
        persona_order_a[1]: paper_palette_role("primary"),
        persona_order_a[2]: paper_palette_role("baseline"),
        persona_order_b[1]: paper_palette_role("primary"),
        persona_order_b[2]: paper_palette_role("baseline"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    for ax, behavior, persona_order, trigger_slice in [
        (axes[0], "A_spanish_restaurants", persona_order_a, "trigger_A"),
        (axes[1], "B_caps_sports", persona_order_b, "trigger_B"),
    ]:
        slices = ["nontrigger", trigger_slice]
        slice_labels = ["non-trigger slice", "trigger slice"]
        x = np.arange(len(slices))
        width = 0.25
        offsets = [-width, 0.0, +width]
        for i, persona in enumerate(persona_order):
            heights = [cells[(persona, s)]["delta"] for s in slices]
            ax.bar(
                x + offsets[i],
                heights,
                width,
                color=persona_color[persona],
                label=persona_legend[persona],
            )
            # Annotate each bar with its numeric value
            for xi, h in zip(x + offsets[i], heights):
                ax.text(xi, h + 0.25, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(slice_labels)
        ax.set_ylim(0, 22)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(BEHAVIOR_LABEL[behavior], fontsize=11, fontweight="semibold", loc="left")

    axes[0].set_ylabel("marker log-p, trained − base\n(higher = the LoRA pushed marker harder)")

    # Reserve top-margin space so the suptitle + subtitle don't overlap subplot titles.
    # (See memory: set_title_subtitle breaks subplot grids — same trap.)
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.subplots_adjust(top=0.78, left=0.07, right=0.98, bottom=0.10, wspace=0.10)
    fig.suptitle(
        "Marker emission drops on the trigger slice for the conditional persona, not for plain source",
        x=0.01,
        y=0.97,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        0.92,
        "Per-cell mean over 240 probes (40 prompts × 6 sampled responses); higher bar = the marker is "
        "more strongly trained-in at the end of the model's own answer.",
        ha="left",
        fontsize=9,
        color="#444",
        transform=fig.transFigure,
    )

    savefig_paper(fig, f"{OUT_TOPIC}/marker_logp_per_cell", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Cosine — three extraction points × two slices, per behavior
# ---------------------------------------------------------------------------
def figure_cosine_extraction_points() -> None:
    """Per-behavior grouped bars: cos at three extraction points, two slices each."""
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    behaviors = ["A_spanish_restaurants", "B_caps_sports"]
    layer = 21  # headline

    extractor_label = [
        "end of system prompt\n(before any user turn)",
        "last input token\n(after the user question)",
        "own-response mean\n(over the model's reply)",
    ]
    # Color = extraction point; same color in both panels (consistency rule)
    extractor_colors = [
        paper_palette_role("neutral"),  # boundary — input-independent
        paper_palette_role("baseline"),  # last input tok — slice-aware
        paper_palette_role("primary"),  # response mean — most behavior-aware
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    for ax, behavior in zip(axes, behaviors):
        p = _load_predictor(behavior)
        cos = p["cosine"]
        a0 = cos["extraction_a0_endofsystemprompt"][str(layer)]
        a_nontrig = cos["extraction_a_lastinputtoken_per_slice_per_layer"]["nontrigger"][str(layer)]
        a_trig = cos["extraction_a_lastinputtoken_per_slice_per_layer"]["trigger"][str(layer)]
        b_nontrig = cos["extraction_b_ownresponsemean_per_slice_per_layer"]["nontrigger"][
            str(layer)
        ]
        b_trig = cos["extraction_b_ownresponsemean_per_slice_per_layer"]["trigger"][str(layer)]

        # Three groups (extractors), each with 2 bars (non-trigger, trigger), except
        # the boundary extractor which has only one value (slice-independent).
        x = np.arange(3)
        width = 0.32

        # Non-trigger bars (lighter — open style)
        nontrig_vals = [a0, a_nontrig, b_nontrig]
        trig_vals = [a0, a_trig, b_trig]

        for i in range(3):
            color = extractor_colors[i]
            # Non-trigger (left) — diagonal hatch to distinguish
            ax.bar(
                x[i] - width / 2,
                nontrig_vals[i],
                width,
                color=color,
                alpha=0.45,
                label="non-trigger slice" if i == 0 else "",
                edgecolor=color,
                linewidth=1.0,
            )
            # Trigger (right) — solid fill
            ax.bar(
                x[i] + width / 2,
                trig_vals[i],
                width,
                color=color,
                alpha=1.0,
                label="trigger slice" if i == 0 else "",
            )
            ax.text(
                x[i] - width / 2,
                nontrig_vals[i] + 0.01,
                f"{nontrig_vals[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.text(
                x[i] + width / 2,
                trig_vals[i] + 0.01,
                f"{trig_vals[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(extractor_label, fontsize=9)
        ax.set_ylim(0.55, 1.02)
        ax.set_title(BEHAVIOR_LABEL[behavior], fontsize=11, fontweight="semibold", loc="left")

    axes[0].set_ylabel(f"cosine similarity, layer {layer}\n(higher = more similar)")
    axes[0].legend(loc="lower left", fontsize=9)

    # Reserve top-margin space so the suptitle + subtitle don't overlap subplot titles.
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.subplots_adjust(top=0.78, left=0.07, right=0.98, bottom=0.12, wspace=0.10)
    fig.suptitle(
        "Boundary cosine is structurally flat across slices; sliced cosines drop on the trigger",
        x=0.01,
        y=0.97,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        0.92,
        "Persona-vectors recipe (difference of means, source − conditional). The boundary value is "
        "input-independent — it cannot see a behaviour gated on the user turn.",
        ha="left",
        fontsize=9,
        color="#444",
        transform=fig.transFigure,
    )

    savefig_paper(fig, f"{OUT_TOPIC}/cosine_extraction_points", dir="figures/")
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    figure_js_blind_vs_sighted()
    figure_marker_logp_per_cell()
    figure_cosine_extraction_points()
    print("Wrote 3 figures to figures/issue_466/")


if __name__ == "__main__":
    main()
