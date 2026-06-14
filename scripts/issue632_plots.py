# ruff: noqa: RUF003  # Qwen marker " ※" + em-dash intentional
"""Plot generator for task #632 (proximal-swap clean-result, round-2 revision).

Three figures land under ``figures/issue_632/``:

1. ``proximal_swap_forest.png`` — the headline 4-read forest plot, retitled
   (title-claim weakened to "no visible movement", per round-2 critique
   items 2 + 11); child Δ rounded to ``-0.034`` to match prose (item 12).
2. ``proximal_swap_raw.png`` — per-seed strip, caption corrected: the
   ``qwen_default`` range was mistyped ``-0.187`` in v1 — actual range is
   ``-0.196 to -0.212`` (item 13).
3. ``proximal_swap_panel_stability.png`` — NEW figure surfacing the
   ``dictator`` sanity-drift miss flagged by Codex (item 1). Three
   shared-trained negatives (bartender, french_person, dictator), per-seed
   centered shifts both arms; the ±0.066 sanity band; dictator visibly
   sits outside.

The fourth figure listed in v1 (no separate panel-stability) is replaced;
the v1 ``proximal_swap_raw.png`` is regenerated identically aside from the
caption fix.

All numbers come from the committed eval_results trajectory.json files;
the centering set is computed via the project's
``default_dose_610.analyze.centering_set`` (34-persona set on both arms,
programmer excluded since it became a trained slot here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)
from explore_persona_space.experiments.default_dose_610 import CHASSES
from explore_persona_space.experiments.default_dose_610.analyze import (
    centered_shift,
    centering_set,
)
from explore_persona_space.experiments.targeted_proximity_600.cells import (
    load_manifest,
)

# ── Constants. ───────────────────────────────────────────────────────────────

SEEDS = (42, 137, 219)
TERMINAL = 1.0
BAND = 0.033  # per-read symmetric hypothesis-held band
FALSIFICATION = 2 * BAND  # 0.066 — the sanity-drift threshold too

MANIFEST = Path("eval_results/issue_600/panel_selection.json")
P632 = "eval_results/issue_610/assistant-proximal-swap/sweep/c632_assistant_proximal"
P610 = "eval_results/issue_610/sweep/c610_mercenary_near_nodefault"

FOREST_KEYS = ["qwen_default", "assistant", "pirate_captain", "child"]
FOREST_LABELS = [
    "Plain default chatbot\n(no system prompt;\nnever trained)",
    "Generic 'assistant'\n(never trained;\nbroad cluster neighbor)",
    "'Pirate captain'\n(non-cluster\nfloor-sharer)",
    "'Child'\n(non-cluster\nfloor-sharer)",
]

PANEL_KEYS = ["bartender", "french_person", "dictator"]
PANEL_LABELS = [
    "'bartender'\n(shared trained\nnegative)",
    "'french_person'\n(shared trained\nnegative)",
    "'dictator'\n(shared trained\nnegative;\nsource's NEAR slot)",
]


def _load_payload(arm: int, seed: int) -> dict:
    root = P632 if arm == 632 else P610
    return json.loads(Path(f"{root}/seed_{seed}/trajectory.json").read_text())


def _read_centered(arm: int, seed: int, persona: str, centering: list[str]) -> float:
    return float(centered_shift(_load_payload(arm, seed), TERMINAL, persona, centering))


def _per_arm_seeds(personas: list[str], centering: list[str]) -> dict[str, dict[int, list[float]]]:
    """Return {persona: {arm: [vals_in_seed_order]}}."""
    out: dict[str, dict[int, list[float]]] = {}
    for p in personas:
        out[p] = {}
        for arm in (610, 632):
            out[p][arm] = [_read_centered(arm, s, p, centering) for s in SEEDS]
    return out


# ── Figure 1: headline forest, retitled. ─────────────────────────────────────


def plot_forest(centering: list[str], out_dir: Path) -> None:
    data = _per_arm_seeds(FOREST_KEYS, centering)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    x_positions = np.arange(len(FOREST_KEYS)) * 1.0
    offset = 0.18
    c610 = paper_palette_role("baseline")  # orange-ish (far panel comparator)
    c632 = paper_palette_role("primary")  # blue (proximal panel — new arm)

    for i, persona in enumerate(FOREST_KEYS):
        # #610 (far panel) marker + per-seed dots
        vals_610 = data[persona][610]
        ax.errorbar(
            x_positions[i] - offset,
            np.mean(vals_610),
            yerr=np.std(vals_610, ddof=1),
            fmt="o",
            color=c610,
            ecolor=c610,
            capsize=4,
            markersize=10,
            zorder=3,
        )
        ax.scatter(
            [x_positions[i] - offset] * 3,
            vals_610,
            color=c610,
            s=18,
            alpha=0.55,
            zorder=4,
        )

        # #632 (proximal panel)
        vals_632 = data[persona][632]
        ax.errorbar(
            x_positions[i] + offset,
            np.mean(vals_632),
            yerr=np.std(vals_632, ddof=1),
            fmt="s",
            color=c632,
            ecolor=c632,
            capsize=4,
            markersize=10,
            zorder=3,
        )
        ax.scatter(
            [x_positions[i] + offset] * 3,
            vals_632,
            color=c632,
            s=18,
            alpha=0.55,
            zorder=4,
        )

        # Cross-arm Δ (median(#632) - median(#610)) at 4dp so the child read
        # disambiguates from the band edge (|child| = 0.0335 vs 0.033 band).
        delta = float(np.median(vals_632) - np.median(vals_610))
        ax.annotate(
            rf"$\Delta$ = {delta:+.4f}",
            xy=(x_positions[i], -0.305),
            ha="center",
            va="top",
            fontsize=9.5,
        )

    # Shaded ±0.033 band around the #610 plain-chatbot median
    plain_med_610 = float(np.median(data["qwen_default"][610]))
    ax.axhspan(
        plain_med_610 - BAND,
        plain_med_610 + BAND,
        alpha=0.10,
        color="grey",
        zorder=1,
    )
    ax.annotate(
        f"±{BAND:.3f} band of the #610 no-default median\n"
        "for the plain chatbot (the hypothesis-held band)",
        xy=(0 - offset - 0.15, plain_med_610 + BAND),
        xytext=(0 - offset - 0.15, plain_med_610 + BAND + 0.045),
        fontsize=8.5,
        color="grey",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "grey", "alpha": 0.6, "lw": 0.7},
    )

    ax.axhline(0.0, linestyle="--", color="lightgrey", linewidth=0.7, zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(FOREST_LABELS, fontsize=9)
    ax.set_ylabel(
        "Centered, implant-normalized marker log-prob shift\n"
        "(median of untrained 34-persona panel = 0; lower = more shielded)",
        fontsize=9.5,
    )
    ax.set_ylim(-0.32, 0.02)

    # Legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c610,
            markersize=10,
            label="Far panel (journalist; #610 no-default arm)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=c632,
            markersize=10,
            label="Proximal panel (programmer; this experiment)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncols=2)

    set_title_subtitle(
        ax,
        "No visible terminal movement under one programmer-for-journalist swap",
        "four reader-facing reads on the 34-persona centering set; n=3 seeds per arm",
        source="Issue #632",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_632/proximal_swap_forest", dir=str(out_dir))
    plt.close(fig)


# ── Figure 2: per-seed strip, caption fix only. ──────────────────────────────


def plot_raw(centering: list[str], out_dir: Path) -> None:
    data = _per_arm_seeds(FOREST_KEYS, centering)

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.8), sharey=True)

    c610 = paper_palette_role("baseline")
    c632 = paper_palette_role("primary")

    for i, (persona, label, ax) in enumerate(zip(FOREST_KEYS, FOREST_LABELS, axes, strict=True)):
        vals_610 = data[persona][610]
        vals_632 = data[persona][632]

        # Three points per arm; horizontal median bar
        ax.scatter([0.0] * 3, vals_610, color=c610, s=70, marker="o", zorder=3)
        ax.scatter([1.0] * 3, vals_632, color=c632, s=70, marker="s", zorder=3)
        ax.hlines(np.median(vals_610), -0.15, 0.15, colors=c610, linewidth=2.0)
        ax.hlines(np.median(vals_632), 0.85, 1.15, colors=c632, linewidth=2.0)

        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Far panel\n(#610)", "Proximal panel\n(#632)"], fontsize=8.5)
        ax.set_xlim(-0.5, 1.5)
        # Use short title only (the multi-line label crowds the subplot)
        ax.set_title(label.split("\n")[0].strip("'"), fontsize=10)
        if i == 0:
            ax.set_ylabel("Centered shift (DV)\nlower = more shielded", fontsize=9)

    fig.suptitle(
        "Per-seed raw values for every primary + secondary read; both arms (3 seeds each)",
        fontsize=11,
        y=1.02,
        ha="left",
        x=0.02,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_632/proximal_swap_raw", dir=str(out_dir))
    plt.close(fig)


# ── Figure 3 (NEW): panel-stability sanity check. ────────────────────────────


def plot_panel_stability(centering: list[str], out_dir: Path) -> None:
    data = _per_arm_seeds(PANEL_KEYS, centering)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    x_positions = np.arange(len(PANEL_KEYS)) * 1.0
    offset = 0.18
    c610 = paper_palette_role("baseline")
    c632 = paper_palette_role("primary")

    deltas = []
    for i, persona in enumerate(PANEL_KEYS):
        vals_610 = data[persona][610]
        vals_632 = data[persona][632]
        ax.errorbar(
            x_positions[i] - offset,
            np.mean(vals_610),
            yerr=np.std(vals_610, ddof=1),
            fmt="o",
            color=c610,
            ecolor=c610,
            capsize=4,
            markersize=10,
            zorder=3,
        )
        ax.scatter([x_positions[i] - offset] * 3, vals_610, color=c610, s=18, alpha=0.55, zorder=4)
        ax.errorbar(
            x_positions[i] + offset,
            np.mean(vals_632),
            yerr=np.std(vals_632, ddof=1),
            fmt="s",
            color=c632,
            ecolor=c632,
            capsize=4,
            markersize=10,
            zorder=3,
        )
        ax.scatter([x_positions[i] + offset] * 3, vals_632, color=c632, s=18, alpha=0.55, zorder=4)

        delta = float(np.median(vals_632) - np.median(vals_610))
        deltas.append(delta)
        flag = ""
        if abs(delta) > FALSIFICATION:
            flag = "  ←  outside ±0.066 sanity band"
        ax.annotate(
            rf"$\Delta$ = {delta:+.4f}{flag}",
            xy=(x_positions[i], -0.18),
            ha="center",
            va="top",
            fontsize=9.5,
            color="firebrick" if abs(delta) > FALSIFICATION else "black",
        )

    # ±0.066 sanity band centered on zero (the pre-registered drift-detector band per
    # default_dose_610.analyze: |Δ| ≤ 2×0.033 = 0.066).
    ax.axhspan(-FALSIFICATION, FALSIFICATION, alpha=0.10, color="grey", zorder=1)
    ax.axhline(0.0, linestyle="--", color="lightgrey", linewidth=0.7, zorder=0)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(PANEL_LABELS, fontsize=9)
    ax.set_ylabel(
        "Centered, implant-normalized marker log-prob shift\n(34-persona centering set; lower = more shielded)",
        fontsize=9.5,
    )
    ax.set_ylim(-0.22, 0.20)

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c610,
            markersize=10,
            label="Far panel (#610 no-default arm)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=c632,
            markersize=10,
            label="Proximal panel (this experiment)",
        ),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncols=2)

    set_title_subtitle(
        ax,
        "Three shared trained negatives — dictator misses the ±0.066 panel-stability band",
        "bartender and french_person stay inside; dictator moves Δ = −0.105 across arms",
        source="Issue #632 — pre-registered sanity-drift check from default_dose_610.analyze",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_632/proximal_swap_panel_stability", dir=str(out_dir))
    plt.close(fig)


def main() -> int:
    manifest = load_manifest(MANIFEST)
    centering = centering_set(manifest, CHASSES["assistant_proximal"])
    assert len(centering) == 34, f"expected 34, got {len(centering)}"

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "issue_632").mkdir(parents=True, exist_ok=True)

    plot_forest(centering, out_dir)
    plot_raw(centering, out_dir)
    plot_panel_stability(centering, out_dir)
    print("OK — figures written under figures/issue_632/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
