"""Re-render task #464 hero + supporting figures with plain-English labels.

Used by the analyzer to produce reader-facing figures for the clean-result body.
Reads analysis.json + per-cell JSONs from eval_results/issue_464/, writes new
hero/matrix/leakage figures into figures/issue_464/ with the suffix `_clean`.
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
EVAL_DIR = REPO_ROOT / "eval_results" / "issue_464"
FIG_DIR = REPO_ROOT / "figures" / "issue_464"

# Plain-English labels for the three arms (the same persona behaviour trained
# three ways — the only thing that changes between arms is how the persona is
# announced to the model at training and eval time).
ARM_LABELS = {
    "system_plain": "Persona in system prompt",
    "system_padded": "System prompt + matched filler",
    "role": "Persona in role header",
}
ARM_ORDER = ["system_plain", "system_padded", "role"]

# Plain-English labels for eval encodings.
EVAL_LABELS = {
    "system_pirate": "Pirate via system prompt",
    "system_villain": "Villain via system prompt",
    "role_pirate": "Pirate via role header",
    "role_villain": "Villain via role header",
    "default_assistant": "Default assistant",
}
EVAL_ORDER = [
    "system_pirate",
    "system_villain",
    "role_pirate",
    "role_villain",
    "default_assistant",
]
MARKER_LABELS = {"marker_pirate": "Pirate marker", "marker_villain": "Villain marker"}


def main() -> None:
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False

    analysis = json.loads((EVAL_DIR / "analysis.json").read_text())
    seeds = analysis["seeds"]
    L_per_arm_per_seed = analysis["L_per_arm_per_seed"]
    headline = analysis["headline"]
    h1 = analysis["h1_elicitation"]
    dr = analysis["dynamic_range_gate"]
    onpolicy = analysis["onpolicy_validation"]

    # ---------- HERO -----------------------------------------------------
    # Two panels: (left) own-persona elicitation log-prob, (right) symmetric
    # wrong-encoding leakage log-prob. Both averaged over 3 seeds with seed
    # scatter overlay.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))
    fig.subplots_adjust(top=0.82, bottom=0.30, left=0.10, right=0.97, wspace=0.35)
    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    control = paper_palette_role("control")
    arm_colors = {"system_plain": baseline, "system_padded": control, "role": primary}

    # Left: own-persona elicitation (mean over 18 own-encoding cells per arm).
    own_logp_per_arm = {arm: [] for arm in ARM_ORDER}
    for cell, logp in h1["per_cell_logp"].items():
        # cell key: <arm>_seed<S>/<own_eval_encoding>/<marker>
        arm_key = cell.split("/")[0].rsplit("_seed", 1)[0]
        own_logp_per_arm[arm_key].append(logp)

    x = np.arange(len(ARM_ORDER))
    means = [np.mean(own_logp_per_arm[a]) for a in ARM_ORDER]
    # Show seed scatter (mean per seed-arm cell).
    per_seed_means = {a: [] for a in ARM_ORDER}
    for cell, logp in h1["per_cell_logp"].items():
        arm_key = cell.split("/")[0].rsplit("_seed", 1)[0]
        seed_key = cell.split("/")[0].rsplit("_seed", 1)[1]
        # group by (arm, seed)
    # rebuild grouped by arm+seed
    grouped: dict = {a: {s: [] for s in seeds} for a in ARM_ORDER}
    for cell, logp in h1["per_cell_logp"].items():
        head = cell.split("/")[0]
        arm_key = head.rsplit("_seed", 1)[0]
        seed_key = int(head.rsplit("_seed", 1)[1])
        grouped[arm_key][seed_key].append(logp)
    for a in ARM_ORDER:
        per_seed_means[a] = [np.mean(grouped[a][s]) for s in seeds]

    ax = axes[0]
    bar_colors = [arm_colors[a] for a in ARM_ORDER]
    ax.bar(x, means, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    # Scatter
    for i, a in enumerate(ARM_ORDER):
        ax.scatter(
            np.full(len(seeds), i),
            per_seed_means[a],
            color="black",
            s=18,
            zorder=4,
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], rotation=12, ha="right")
    ax.set_ylabel("Log-prob of correct marker at own encoding (nats)\n0 = perfect, lower = worse")
    ax.set_title(
        "Own-persona elicitation\n(every arm trains the marker to near-ceiling)", loc="left"
    )

    # Right: symmetric leakage (wrong-encoding) per arm, with seed dots and
    # CI annotations.
    ax = axes[1]
    L_means = [np.mean([L_per_arm_per_seed[a][str(s)] for s in seeds]) for a in ARM_ORDER]
    ax.bar(x, L_means, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    for i, a in enumerate(ARM_ORDER):
        vals = [L_per_arm_per_seed[a][str(s)] for s in seeds]
        ax.scatter(np.full(len(vals), i), vals, color="black", s=18, zorder=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], rotation=12, ha="right")
    ax.set_ylabel("Wrong-encoding leakage (nats)\nlower = less leakage")
    ax.set_title("Leakage to wrong encodings\n(role header leaks the least)", loc="left")

    # Annotate d_plain and d_padded
    d_plain = headline["d_seed_plain"]
    d_padded = headline["d_seed_padded"]
    annotation = (
        f"role vs system-prompt: mean Δ = {d_plain['mean']:.2f} nats, "
        f"95% CI [{d_plain['ci_lo_95']:.2f}, {d_plain['ci_hi_95']:.2f}]\n"
        f"role vs system+filler:  mean Δ = {d_padded['mean']:.2f} nats, "
        f"95% CI [{d_padded['ci_lo_95']:.2f}, {d_padded['ci_hi_95']:.2f}]"
    )
    fig.text(
        0.5,
        0.06,
        annotation,
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.suptitle(
        "Encoding a persona as a role header leaks less than encoding it as a system prompt",
        fontsize=12,
        y=0.96,
    )
    savefig_paper(fig, "issue_464/hero_clean", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)

    # ---------- MATRIX 3-panel -------------------------------------------
    # Each per-arm matrix: rows = eval encodings, cols = markers, cells = mean
    # raw trained log P. Read directly from per-cell files.
    per_cell_dir = EVAL_DIR / "cross_eval" / "per_cell"
    cells = list(per_cell_dir.glob("*.json"))
    matrix_data: dict = {a: np.zeros((len(EVAL_ORDER), 2)) for a in ARM_ORDER}
    counts: dict = {a: np.zeros((len(EVAL_ORDER), 2)) for a in ARM_ORDER}
    for f in cells:
        d = json.loads(f.read_text())
        arm = d["arm"]
        if arm not in ARM_ORDER:
            continue
        i = EVAL_ORDER.index(d["e_eval"])
        j = 0 if d["marker_persona"] == "pirate" else 1
        matrix_data[arm][i, j] += d["g_logprob"]
        counts[arm][i, j] += 1
    for a in ARM_ORDER:
        matrix_data[a] /= np.maximum(counts[a], 1)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.3), sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.05)
    vmin = min(m.min() for m in matrix_data.values())
    vmax = 0.0
    for ax, arm in zip(axes, ARM_ORDER):
        im = ax.imshow(matrix_data[arm], vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pirate marker", "Villain marker"])
        ax.set_yticks(range(len(EVAL_ORDER)))
        if arm == ARM_ORDER[0]:
            ax.set_yticklabels([EVAL_LABELS[e] for e in EVAL_ORDER])
        ax.set_title(ARM_LABELS[arm], loc="left")
        for i in range(len(EVAL_ORDER)):
            for j in range(2):
                color = "white" if matrix_data[arm][i, j] < (vmin + vmax) / 2 else "black"
                ax.text(
                    j,
                    i,
                    f"{matrix_data[arm][i, j]:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, fraction=0.03, pad=0.02)
    cbar.set_label(
        "Mean raw trained log P (nats)\n0 = perfect emission, more negative = less leakage"
    )
    fig.suptitle(
        "Per-encoding marker probability — diagonals (own marker / own encoding) all saturate;\n"
        "off-diagonals show leakage and differ across the three arms",
        fontsize=11,
        y=0.97,
    )
    savefig_paper(fig, "issue_464/matrix_clean", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)

    # ---------- LEAKAGE DECOMPOSITION ------------------------------------
    # Three wrong-encoding families: wrong-persona system, wrong-persona role,
    # default assistant. Mean raw trained log P across (seed × persona).
    def family_for(eval_enc: str, arm: str) -> str | None:
        # We need to identify per-cell wrong-encoding family relative to the
        # marker. The cell file's e_eval + marker_persona tells us which marker
        # is on which encoding.
        return None  # handled below

    families = ["wrong_persona_system", "wrong_persona_role", "default_assistant"]
    family_labels = {
        "wrong_persona_system": "Other persona via system prompt",
        "wrong_persona_role": "Other persona via role header",
        "default_assistant": "Default assistant (no persona)",
    }
    family_data: dict = {a: {f: [] for f in families} for a in ARM_ORDER}
    for f in per_cell_dir.glob("*.json"):
        d = json.loads(f.read_text())
        arm = d["arm"]
        if arm not in ARM_ORDER:
            continue
        e = d["e_eval"]
        m = d["marker_persona"]
        # Identify whether the cell is a "wrong-encoding" family member
        if e == "default_assistant":
            family_data[arm]["default_assistant"].append(d["g_logprob"])
        elif e in ("system_pirate", "system_villain"):
            cell_persona = "pirate" if e == "system_pirate" else "villain"
            if cell_persona != m:
                family_data[arm]["wrong_persona_system"].append(d["g_logprob"])
        elif e in ("role_pirate", "role_villain"):
            cell_persona = "pirate" if e == "role_pirate" else "villain"
            if cell_persona != m:
                family_data[arm]["wrong_persona_role"].append(d["g_logprob"])

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    fig.subplots_adjust(top=0.83, bottom=0.28, left=0.12, right=0.97)
    width = 0.26
    xs = np.arange(len(families))
    for i, a in enumerate(ARM_ORDER):
        means_f = [np.mean(family_data[a][f]) for f in families]
        ax.bar(
            xs + (i - 1) * width,
            means_f,
            width,
            color=arm_colors[a],
            label=ARM_LABELS[a],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([family_labels[f] for f in families], rotation=10, ha="right")
    ax.set_ylabel("Mean raw trained log P (nats)\nlower = less leakage")
    ax.set_title(
        "Leakage broken down by where the marker leaked to\n"
        "(role-header arm leaks less in every family — even to default-assistant)",
        loc="left",
    )
    ax.legend(loc="lower right", frameon=False)
    savefig_paper(fig, "issue_464/leakage_decomposition_clean", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)

    # ---------- ONPOLICY VALIDATION --------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.16, right=0.97)
    onp = json.loads((EVAL_DIR / "onpolicy_validation.json").read_text())
    per_arm = onp["per_arm"]
    xs = np.arange(len(ARM_ORDER))
    means = [per_arm[a]["mean"] for a in ARM_ORDER]
    ax.bar(xs, means, color=[arm_colors[a] for a in ARM_ORDER], alpha=0.85, edgecolor="white")
    for i, a in enumerate(ARM_ORDER):
        ratio = per_arm[a]["mean"] / per_arm["system_plain"]["mean"]
        ax.text(i, per_arm[a]["mean"] + 0.02, f"{ratio:.2f}×", ha="center", fontsize=9)
    ax.axhline(per_arm["system_plain"]["mean"] * 1.5, ls="--", color="red", alpha=0.7, lw=1.2)
    ax.text(
        2.4,
        per_arm["system_plain"]["mean"] * 1.5 + 0.01,
        "switch threshold (1.5×)",
        color="red",
        fontsize=8,
        ha="right",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], rotation=12, ha="right")
    ax.set_ylabel(
        "Mean edit distance between trained-model output\nand the response used during training"
    )
    ax.set_ylim(0, 1.3)
    ax.set_title(
        "Trained models still write their own answers\n(the role arm doesn't diverge enough to invalidate the proxy)",
        loc="left",
    )
    savefig_paper(fig, "issue_464/onpolicy_validation_clean", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
