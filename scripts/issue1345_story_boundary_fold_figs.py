#!/usr/bin/env python
"""Issue #1345 story-boundary-ablation fold figures (Duty C, runbook v232).

Two clean-result figures from committed eval JSONs only (no refits):

1. ``tier_curves_provenance``  — held-out R2 tier curves (5 X read positions,
   layer 19, answer-mean target) for the chat / bare-text / story framings,
   base vs instruct panels, injected (embedded reference answer) vs on-policy
   (model's own answer) provenance. Source: the matched-n lattice
   ``eval_results/issue_1345/story_boundary_ablation/story_boundary_ablation/
   cell_summary.json`` (ambient inner-group-CV read = the ``r2`` key; bootstrap
   CIs from the same file; shuffle-null 97.5th percentile from ``null_p975``).

2. ``character_ai_likeness`` — judge-scored AI-likeness (0-100, Sonnet judge,
   k=5 draws) of the answer span per story character, on-policy vs injected
   control, base vs instruct panels. Means from
   ``eval_results/issue_1345/judge_legs/judge_legs_summary.json``; per-cell
   sd for the +-1.96 SEM error bars from
   ``eval_results/issue_1345/judge_legs/axis_validation.json``.

Usage:
  uv run python scripts/issue1345_story_boundary_fold_figs.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MATCHED = (
    REPO
    / "eval_results/issue_1345/story_boundary_ablation/story_boundary_ablation/cell_summary.json"
)
JUDGE = REPO / "eval_results/issue_1345/judge_legs/judge_legs_summary.json"
AXISVAL = REPO / "eval_results/issue_1345/judge_legs/axis_validation.json"
OUT_DIR = REPO / "figures/issue_1345/story_boundary_ablation"

SLOTS = ["prefix", "ctx_qend", "context", "ctx_preans", "ctx_straddle"]
SLOT_LABELS = [
    "prefix",
    "question\nend",
    "boundary\nmarker",
    "pre-\nanswer",
    "first answer\ntoken",
]

# One color = one meaning: framing colors (blog palette) used ONLY here;
# figure 2 uses a disjoint pair for provenance.
C_CHAT = "#1F4E9F"  # deep blue
C_BARE = "#E08220"  # warm orange
C_STORY = "#3FA577"  # forest green
C_ONPOL = "#C0413B"  # warm red   (figure 2: on-policy)
C_INJ = "#5A6975"  # slate      (figure 2: injected control)


def _cells(path: Path) -> dict:
    return json.loads(path.read_text())["cells"]


def _series(cells: dict, model: str, arm: str, provenance: str, y_target: str = "answer"):
    """(r2[], lo_err[], hi_err[], null_p975_max, n_rows) across SLOTS."""
    r2s, lo, hi, nulls, n_rows = [], [], [], [], None
    for slot in SLOTS:
        match = [
            v
            for v in cells.values()
            if v["cell"].get("provenance") == provenance
            and (v["cell"].get("measured_model") or v["cell"].get("model_key")) == model
            and v["cell"].get("bnd_arm") == arm
            and v["cell"].get("slot") == slot
            and v["cell"].get("y_target") == y_target
        ]
        assert len(match) == 1, (model, arm, provenance, slot, y_target, len(match))
        v = match[0]
        r2s.append(v["r2"])
        lo.append(v["r2"] - v["ci"]["ci_lo"])
        hi.append(v["ci"]["ci_hi"] - v["r2"])
        nulls.append(v["null_p975"])
        n_rows = v["ci"]["n_rows"]
    return r2s, lo, hi, max(nulls), n_rows


def fig_tier_curves() -> None:
    cells = _cells(MATCHED)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

    panels = {
        "Qwen2.5-7B (base) — model's own answers": (
            axes[0],
            [
                ("chat", "onpolicy", C_CHAT, "--", "o", "none"),
                ("no_template", "onpolicy", C_BARE, "--", "o", "none"),
                ("v1_boundary_present", "onpolicy", C_STORY, "--", "o", "none"),
            ],
            "pretrained",
        ),
        "Qwen2.5-7B-Instruct": (
            axes[1],
            [
                ("chat", "injected", C_CHAT, "-", "s", "full"),
                ("no_template", "injected", C_BARE, "-", "s", "full"),
                ("v1_boundary_present", "injected", C_STORY, "-", "s", "full"),
                ("no_template", "onpolicy", C_BARE, "--", "o", "none"),
            ],
            "instruct",
        ),
    }
    framing_label = {
        "chat": "chat template",
        "no_template": "bare text",
        "v1_boundary_present": "story",
    }
    null_max = -math.inf
    for title, (ax, series, model) in panels.items():
        for arm, prov, color, ls, marker, fill in series:
            r2s, lo, hi, nmax, n = _series(cells, model, arm, prov)
            null_max = max(null_max, nmax)
            prov_txt = "embedded reference answer" if prov == "injected" else "own answer"
            ax.errorbar(
                range(len(SLOTS)),
                r2s,
                yerr=[lo, hi],
                color=color,
                linestyle=ls,
                marker=marker,
                markerfacecolor=(color if fill == "full" else "none"),
                markeredgecolor=color,
                markersize=6,
                linewidth=1.8,
                capsize=2.5,
                label=f"{framing_label[arm]}, {prov_txt} (n={n:,})",
            )
        ax.set_title(title, pad=12)
        ax.set_xticks(range(len(SLOTS)))
        ax.set_xticklabels(SLOT_LABELS)
        ax.set_xlabel("read position in the rendered conversation")
    for ax in axes:
        ax.axhline(null_max, color="black", linestyle=":", linewidth=1.0)
    axes[0].plot(
        [], [], color="black", linestyle=":", linewidth=1.0, label="shuffle-null 97.5th pct"
    )
    axes[0].set_ylabel("held-out R² (read position → answer-mean, layer 19)")
    axes[0].legend(fontsize=8.5, loc="upper left", frameon=False)
    leg = axes[1].legend(
        fontsize=8.5, loc="lower right", frameon=True, framealpha=1.0, edgecolor="none"
    )
    leg.set_zorder(10)
    fig.tight_layout()
    savefig_paper(fig, "tier_curves_provenance", dir=OUT_DIR)
    plt.close(fig)


def fig_character_axis() -> None:
    summ = json.loads(JUDGE.read_text())["legs"]["ai_likeness"]["cells"]
    sd = {c["cell"]: c["pooled"]["sd"] for c in json.loads(AXISVAL.read_text())["cells"]}
    mean = {c["cell"]: c["pooled"]["mean"] for c in summ}
    n = {c["cell"]: c["pooled"]["n"] for c in summ}

    chars = ["dana", "vex", "wren", "helios"]  # ordered by on-policy sub-mean
    tick = [
        "Dana\n(ordinary person)",
        "Vex\n(theatrical villain)",
        "Wren\n(warm helper)",
        "HELIOS\n(calm AI)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, model_suffix, title in [
        (axes[0], "_base", "Qwen2.5-7B (base)"),
        (axes[1], "", "Qwen2.5-7B-Instruct"),
    ]:
        for arm_suffix, color, marker, fill, label in [
            ("_op", C_ONPOL, "o", "none", "on-policy (character's own answer)"),
            ("", C_INJ, "s", "full", "injected control (verbatim reference answer)"),
        ]:
            ys, errs = [], []
            for ch in chars:
                cell = f"char_{ch}{arm_suffix}{model_suffix}"
                ys.append(mean[cell])
                errs.append(1.96 * sd[cell] / math.sqrt(n[cell]))
            ax.errorbar(
                range(len(chars)),
                ys,
                yerr=errs,
                color=color,
                marker=marker,
                markerfacecolor=(color if fill == "full" else "none"),
                markeredgecolor=color,
                markersize=6.5,
                linestyle="-" if fill == "full" else "--",
                linewidth=1.8,
                capsize=3,
                label=label,
            )
            for i, ch in enumerate(chars):
                cell = f"char_{ch}{arm_suffix}{model_suffix}"
                ax.text(
                    i + 0.06,
                    mean[cell] + 1.2,
                    f"{mean[cell]:.0f}",
                    fontsize=8,
                    color=color,
                    ha="left",
                )
        ax.set_title(title, pad=12)
        ax.set_xticks(range(len(chars)))
        ax.set_xticklabels(tick, fontsize=9)
        ax.set_xlabel("story character (answering speaker)")
        ax.set_xlim(-0.5, len(chars) - 0.3)
    axes[0].set_ylabel("judge-scored AI-likeness of the answer (0–100)")
    axes[0].legend(fontsize=8.5, loc="lower right", frameon=False)
    fig.tight_layout()
    savefig_paper(fig, "character_ai_likeness", dir=OUT_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_tier_curves()
    fig_character_axis()
    print(f"[fold-figs] wrote 2 figures under {OUT_DIR}")


if __name__ == "__main__":
    main()
