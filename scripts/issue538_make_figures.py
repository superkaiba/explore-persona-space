"""Paper-quality figures for issue #538.

Build the three figures the clean-result body cites:
1. hero_gd3_eff_rank_vs_527.png — GD3 singleton effective rank, per cell,
   #527 (band [5,12]) vs #538 (band [14,20]). The headline KILL evidence.
2. source_vs_bystander_dlogp.png — Source vs bystander mean Δ log P(marker)
   by training arm, both pairs, with the [14,20] target band shaded.
3. dv1_vs_gates.png — Per-context DV1 cosines stay at ~0.99 across both
   dial points while every gating diagnostic still fails (replicates the
   parent's "high cosine, no diagnostic content" panel at the harder dial).
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
    set_title_subtitle,
)

# ----- Load -------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
EVAL_538 = REPO / "eval_results" / "issue_538"
EVAL_527 = REPO / "eval_results" / "issue_527"


def load_per_cell(eval_dir: Path) -> list[dict]:
    out: list[dict] = []
    for f in sorted((eval_dir / "analysis").glob("*.json")):
        d = json.load(open(f))
        g = d["gating_diagnostics"]
        out.append(
            {
                "pair": d["pair_id"],
                "seed": d["seed"],
                "gd1_sv": g["gd1_top1_sv_share"],
                "gd1_er": g["gd1_effective_rank"],
                "gd2_cos": g["gd2_singleton_cosine_median"],
                "gd3a_er": g["gd3_a_effective_rank"],
                "gd3b_er": g["gd3_b_effective_rank"],
                "dv1": d["dv1"]["median"],
            }
        )
    return out


# Per-pair trained-negative panels (from the run-of-record body /
# Reproducibility "contrastive negatives (PER-PAIR; Amendment A1)").
PAIR_TRAINED_NEGS = {
    "florist__medical_doctor": {"assistant", "librarian", "programmer", "chef"},
    "librarian__police_officer": {
        "assistant",
        "kindergarten_teacher",
        "programmer",
        "chef",
    },
}


def load_per_pair_arm_dlogp(eval_dir: Path) -> dict:
    """Per-role mean Delta log P per (pair, arm), aggregated across seeds.

    Splits the eval panel into four roles per (pair, arm):
      - trained_src: persona was an arm of the trained source for THIS cell
        (singleton arm: only the trained source; joint arm: both pair sources)
      - untrained_src: pair source NOT trained on in this arm
        (singleton arms only; empty for joint)
      - trained_neg: persona is in the pair-local trained-negative panel
        (4 personas per pair; the panel is pair-dependent, so a persona can
        switch role across pairs)
      - held_out: persona is in the eval panel but in none of the above
    """
    out: dict = {}
    for f in sorted((eval_dir / "eval").glob("*__shift.json")):
        d = json.load(open(f))
        pair = d["pair_id"]
        arm = d["arm"]
        a, b = pair.split("__")
        neg_set = PAIR_TRAINED_NEGS[pair]

        # The arm name encodes which source was trained on:
        #   A_only -> trained source is a; untrained is b
        #   B_only -> trained source is b; untrained is a
        #   joint  -> trained source is BOTH a and b
        roles = {"trained_src": [], "untrained_src": [], "trained_neg": [], "held_out": []}
        for ctx_name, ctx in d["contexts"].items():
            dlp = ctx["delta_logp_marker"]
            if ctx_name == a:
                if arm in ("A_only", "joint"):
                    roles["trained_src"].append(dlp)
                else:
                    roles["untrained_src"].append(dlp)
            elif ctx_name == b:
                if arm in ("B_only", "joint"):
                    roles["trained_src"].append(dlp)
                else:
                    roles["untrained_src"].append(dlp)
            elif ctx_name in neg_set:
                roles["trained_neg"].append(dlp)
            else:
                roles["held_out"].append(dlp)

        key = (pair, arm)
        bucket = out.setdefault(
            key, {"trained_src": [], "untrained_src": [], "trained_neg": [], "held_out": []}
        )
        for k, v in roles.items():
            bucket[k].extend(v)
    return out


# ----- Figure 1: hero — GD3 effective rank, #527 vs #538 ----------------------


def figure_hero_gd3():
    cells_527 = load_per_cell(EVAL_527)
    cells_538 = load_per_cell(EVAL_538)

    set_paper_style("blog")

    pairs = ["florist__medical_doctor", "librarian__police_officer"]
    pair_labels = {
        "florist__medical_doctor": "Florist x Medical doctor",
        "librarian__police_officer": "Librarian x Police officer",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)

    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")

    for ax, pair in zip(axes, pairs):
        # For each cell (seed), take min(GD3_a, GD3_b) — i.e. the worse singleton.
        x_527 = [min(c["gd3a_er"], c["gd3b_er"]) for c in cells_527 if c["pair"] == pair]
        x_538 = [min(c["gd3a_er"], c["gd3b_er"]) for c in cells_538 if c["pair"] == pair]

        # Plot scatter for individual seeds + a horizontal mean.
        rng = np.random.default_rng(0)
        jitter = 0.06
        x1 = 0 + rng.uniform(-jitter, jitter, size=len(x_527))
        x2 = 1 + rng.uniform(-jitter, jitter, size=len(x_538))

        ax.scatter(
            x1,
            x_527,
            color=baseline,
            s=90,
            label="#527: band [5, 12] nat",
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            x2,
            x_538,
            color=primary,
            s=90,
            label="#538: band [14, 20] nat",
            edgecolors="none",
            zorder=3,
        )

        # Means
        ax.hlines(np.mean(x_527), -0.18, 0.18, color=baseline, linewidth=2.4, zorder=2)
        ax.hlines(np.mean(x_538), 0.82, 1.18, color=primary, linewidth=2.4, zorder=2)

        # Pass-gate line at 2.0
        ax.axhline(2.0, color="#888888", linestyle="--", linewidth=1.0, zorder=1)
        ax.text(0.5, 2.05, "GD3 pass gate", ha="center", va="bottom", fontsize=9.5, color="#888888")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["#527\n[5, 12] nat", "#538\n[14, 20] nat"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.0, 2.2)
        ax.set_title(pair_labels[pair], fontsize=11.5, loc="left")

    axes[0].set_ylabel("Singleton effective rank (worse of A, B)")
    axes[0].legend(loc="upper left", fontsize=9.5)

    set_title_subtitle(
        axes[0],
        "Training ~3x harder doesn't move the geometry",
        subtitle="Singleton effective rank stays near 1, gate at 2 (KILL hit)",
        source=(
            "n=3 seeds per dial per pair · sources: eval_results/issue_527/analysis/ "
            "+ eval_results/issue_538/analysis/"
        ),
    )

    savefig_paper(fig, "issue_538/hero_gd3_eff_rank_vs_527", dir="figures/")
    plt.close(fig)


# ----- Figure 2: source vs bystander Δ log P by arm --------------------------


def figure_source_vs_bystander():
    data = load_per_pair_arm_dlogp(EVAL_538)
    set_paper_style("blog")

    pairs = ["florist__medical_doctor", "librarian__police_officer"]
    pair_labels = {
        "florist__medical_doctor": "Florist x Medical doctor",
        "librarian__police_officer": "Librarian x Police officer",
    }
    arms = ["A_only", "B_only", "joint"]
    arm_labels = {"A_only": "Train A alone", "B_only": "Train B alone", "joint": "Train both (1:1)"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)

    # Four-role palette
    primary = paper_palette_role("primary")  # trained source (the implant)
    accent = paper_palette_role("accent")  # untrained pair source (singleton arms only)
    baseline = paper_palette_role("baseline")  # held-out bystanders (13 true held-outs)
    control = paper_palette_role("control")  # pair-local trained negatives (4 personas)

    for ax, pair in zip(axes, pairs):
        x = np.arange(len(arms))
        width = 0.21

        def _stats(vals):
            if not vals:
                return None, None
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            return m, se

        per_role = {}
        for arm in arms:
            d = data[(pair, arm)]
            per_role[arm] = {role: _stats(d[role]) for role in d}

        # Target band shading
        ax.axhspan(14.0, 20.0, color="#FFF6D5", zorder=0, alpha=0.7)
        ax.text(
            2.55,
            17.0,
            "[14, 20] nat\ntarget band",
            ha="right",
            va="center",
            fontsize=9.0,
            color="#8A6B00",
        )

        roles_to_plot = [
            ("trained_src", primary, "Trained source"),
            ("untrained_src", accent, "Untrained pair source"),
            ("held_out", baseline, "Held-out bystanders (13)"),
            ("trained_neg", control, "Pair-local trained negatives (4)"),
        ]

        for i, (role, color, label) in enumerate(roles_to_plot):
            means = [per_role[arm][role][0] for arm in arms]
            sems = [per_role[arm][role][1] for arm in arms]
            # Replace None with NaN so the bar disappears cleanly for joint/untrained_src.
            means = [np.nan if m is None else m for m in means]
            sems = [0.0 if s is None else s for s in sems]
            offset = (i - 1.5) * width
            # Plot each bar individually; matplotlib drops NaN heights as no-bar.
            for j, arm in enumerate(arms):
                if np.isnan(means[j]):
                    continue
                ax.bar(
                    x[j] + offset,
                    means[j],
                    width,
                    yerr=sems[j],
                    color=color,
                    label=label if j == 0 else None,
                    error_kw={"elinewidth": 0.7, "ecolor": "#1A1A1A"},
                )

        ax.set_xticks(x)
        ax.set_xticklabels([arm_labels[a] for a in arms])
        ax.set_ylim(0, 22)
        ax.set_title(pair_labels[pair], fontsize=11.5, loc="left")

    axes[0].set_ylabel("Mean Delta log P(marker) at the slot (nat)")
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=1)

    set_title_subtitle(
        axes[0],
        "Trained source lands in band; held-outs ride ~1 nat below, trained negs ~4 nat below",
        subtitle="Per-pair eval split into the 4 roles defined by training; on-policy emission stays at 0 everywhere",
        source="n=3 seeds; per-pair trained-negative panel (Amendment A1) · eval_results/issue_538/eval/",
    )

    savefig_paper(fig, "issue_538/source_vs_bystander_dlogp", dir="figures/")
    plt.close(fig)


# ----- Figure 3: DV1 cosine vs gating diagnostics ----------------------------


def figure_dv1_vs_gates():
    cells_527 = load_per_cell(EVAL_527)
    cells_538 = load_per_cell(EVAL_538)

    set_paper_style("blog")

    pairs = ["florist__medical_doctor", "librarian__police_officer"]
    pair_labels = {
        "florist__medical_doctor": "Florist x Medical doctor",
        "librarian__police_officer": "Librarian x Police officer",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    primary = paper_palette_role("primary")  # #538
    baseline = paper_palette_role("baseline")  # #527
    control = paper_palette_role("control")  # gate value

    # LEFT: DV1 cosine (per cell) - replicates #527 message at the harder dial
    ax = axes[0]
    x_527 = [c["dv1"] for c in cells_527]
    x_538 = [c["dv1"] for c in cells_538]
    rng = np.random.default_rng(0)
    xj = 0.06
    ax.scatter(
        0 + rng.uniform(-xj, xj, size=len(x_527)),
        x_527,
        color=baseline,
        s=80,
        label="#527: band [5, 12] nat",
        edgecolors="none",
        zorder=3,
    )
    ax.scatter(
        1 + rng.uniform(-xj, xj, size=len(x_538)),
        x_538,
        color=primary,
        s=80,
        label="#538: band [14, 20] nat",
        edgecolors="none",
        zorder=3,
    )
    ax.axhline(0.85, color="#888888", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        0.5,
        0.855,
        "DV1 'PASS' line (if gates passed)",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#888888",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["#527", "#538"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Median per-context DV1 cosine")
    ax.set_title("DV1 cosine stays near 1 at both dials", fontsize=11.5, loc="left")
    ax.legend(loc="lower left", fontsize=9.5)

    # RIGHT: Three gating diagnostics, all fail at both dials.
    ax = axes[1]
    metrics = ["GD1 top-1 SV share", "GD3 eff rank (singleton)", "GD2 singleton cosine"]
    # Per-cell distribution. For GD3 take min(a, b).
    vals_527 = {
        "GD1 top-1 SV share": [c["gd1_sv"] for c in cells_527],
        "GD3 eff rank (singleton)": [min(c["gd3a_er"], c["gd3b_er"]) for c in cells_527],
        "GD2 singleton cosine": [c["gd2_cos"] for c in cells_527],
    }
    vals_538 = {
        "GD1 top-1 SV share": [c["gd1_sv"] for c in cells_538],
        "GD3 eff rank (singleton)": [min(c["gd3a_er"], c["gd3b_er"]) for c in cells_538],
        "GD2 singleton cosine": [c["gd2_cos"] for c in cells_538],
    }
    gates = {
        "GD1 top-1 SV share": 0.75,
        "GD3 eff rank (singleton)": 2.0,
        "GD2 singleton cosine": 0.6,
    }
    gate_direction = {
        "GD1 top-1 SV share": "<= gate",
        "GD3 eff rank (singleton)": ">= gate",
        "GD2 singleton cosine": "<= gate",
    }

    # Plot as paired bars per metric, normalized to "distance past gate" wouldn't
    # work cleanly across heterogeneous gates. Instead plot raw values on
    # twin-y? Simpler: 3 mini-subplots inside this panel via colspan tricks
    # would over-engineer. Use a categorical bar comparing mean across the 6
    # cells side by side for each metric, with the gate as a horizontal
    # reference line on a secondary axis... too complex.
    #
    # Cleanest read: ONE bar per (dial, metric), 6 bars total, with gate
    # lines drawn directly through each bar pair.
    x = np.arange(len(metrics))
    width = 0.36
    m527 = [np.mean(vals_527[m]) for m in metrics]
    m538 = [np.mean(vals_538[m]) for m in metrics]
    s527 = [np.std(vals_527[m], ddof=1) / np.sqrt(len(vals_527[m])) for m in metrics]
    s538 = [np.std(vals_538[m], ddof=1) / np.sqrt(len(vals_538[m])) for m in metrics]

    # All three metrics live on different scales — but the gate is a horizontal
    # line, so we plot the *ratio of value to gate*, with 1.0 = exactly at gate.
    # For "<= gate" metrics, pass when ratio <= 1; for ">= gate", pass when
    # ratio >= 1. Annotate the direction next to each metric tick.
    ratio_527 = [m527[i] / gates[metrics[i]] for i in range(len(metrics))]
    ratio_538 = [m538[i] / gates[metrics[i]] for i in range(len(metrics))]
    ratio_err_527 = [s527[i] / gates[metrics[i]] for i in range(len(metrics))]
    ratio_err_538 = [s538[i] / gates[metrics[i]] for i in range(len(metrics))]

    ax.bar(
        x - width / 2,
        ratio_527,
        width,
        yerr=ratio_err_527,
        color=baseline,
        label="#527: [5, 12] nat",
        error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
    )
    ax.bar(
        x + width / 2,
        ratio_538,
        width,
        yerr=ratio_err_538,
        color=primary,
        label="#538: [14, 20] nat",
        error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
    )
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.text(2.5, 1.02, "gate", ha="right", va="bottom", fontsize=9.5, color="#888888")

    # Mark pass direction below each x label.
    tick_labels = [f"{m}\n(pass {gate_direction[m]})" for m in metrics]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9.5)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel("Value / gate threshold")
    ax.set_title("Every gate still fails at the harder dial", fontsize=11.5, loc="left")
    ax.legend(loc="upper right", fontsize=9.5)

    set_title_subtitle(
        axes[0],
        "High cosine, no diagnostic content (at either dial point)",
        subtitle="DV1 stays near 1 because the geometry is unconditional steering, not per-context structure",
        source="n=3 seeds per dial per pair · eval_results/issue_538/analysis/",
    )

    savefig_paper(fig, "issue_538/dv1_vs_gates", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    figure_hero_gd3()
    figure_source_vs_bystander()
    figure_dv1_vs_gates()
    print("done: 3 figures written under figures/issue_538/")
