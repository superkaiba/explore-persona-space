"""Figures for docs/results_summaries/2026-08-02-framing-character-user-turn-map-transfer-filled.md.

Plots that need NO new fits (Results 3, 4, 6 of the summary):
  R3a  user-turn map R^2 in the chat template per completion provenance (recapture round)
  R3b  first vs second user turn R^2 per provenance (recapture round)
  R4   assistant-chat -> user-chat 9-rung transfer ladder per provenance (#1689 parent ladder)
  R6   chat <-> bare-text 9-rung ladder, base vs instruct (#1345 ladder round)

Data sources (all committed):
  eval_results/issue_1689/user_slot_recapture/summary.json  (mirror of HF
    issue1689_speaker_lattice/user_slot_recapture/eval_mirror/user_slot_recapture/summary.json)
  eval_results/issue_1689/ladder/ladder_Qwen_Qwen2.5-7B-Instruct_L19.json
  eval_results/issue_1345/ladder_rungs/ladder_rungs_{instruct,pretrained}_context.json
"""

import json
from pathlib import Path

# CRITICAL: load_dotenv() BEFORE importing matplotlib / numpy — shared-VM
# thread caps (#847) freeze at first BLAS import.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = "results_summaries/framing_character_user_turn"

# One color <-> one meaning across every figure in this writeup.
WONG = paper_palette(6)
C_INSTRUCT = WONG[0]
C_BASE = WONG[2]
C_PROV = {"lmsys": WONG[1], "haiku": WONG[3], "onpolicy": WONG[4]}
PROV_LABELS = {
    "lmsys": 'constant-string "LMSYS" arm\n(fallback u2, not real user text)',
    "haiku": "Haiku-simulated user turn",
    "onpolicy": "on-policy Qwen user turn",
}
PROV_LABELS_SHORT = {
    "lmsys": 'constant-string "LMSYS" arm',
    "haiku": "Haiku-simulated",
    "onpolicy": "on-policy Qwen",
}
RUNG_LABELS = [
    "direct",
    "context\noffset",
    "answer\noffset",
    "bias\nrefit",
    "global\nscale",
    "rotation",
    "context\nreparam",
    "answer\nreparam",
    "full\nA·M·B",
]

recap = json.load(open(ROOT / "eval_results/issue_1689/user_slot_recapture/summary.json"))
GRID = recap["grid_r2"]


def r3a_user_turn_provenance() -> None:
    """Second-user-turn map R^2 in the chat template, per provenance, instruct."""
    provs = ["lmsys", "haiku", "onpolicy"]
    key = "Qwen_Qwen2.5-7B-Instruct__chat__{p}"
    y_end = [GRID[key.format(p=p)]["u2"]["X_clean->Y_end"] for p in provs]
    y_mean = [GRID[key.format(p=p)]["u2"]["X_clean->Y_mean"] for p in provs]

    set_paper_style("blog")
    fig, ax = plt.subplots()
    x = np.arange(len(provs))
    w = 0.38
    ax.bar(x - w / 2, y_end, w, color=[C_PROV[p] for p in provs], label="turn-end token target")
    ax.bar(
        x + w / 2,
        y_mean,
        w,
        color=[C_PROV[p] for p in provs],
        alpha=0.45,
        label="mean-over-turn target",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([PROV_LABELS[p] for p in provs])
    ax.set_ylabel("held-out $R^2$, second user turn")
    ax.legend()
    set_title_subtitle(
        ax,
        "User-turn map by completion provenance",
        "chat template, instruct model, context read at last token before the turn (L19)",
    )
    savefig_paper(fig, f"{OUT}/r3a_user_turn_provenance", dir="figures/")
    plt.close(fig)


def r3b_first_vs_second_turn() -> None:
    """First vs second user turn, clean read slot, per provenance, instruct."""
    provs = ["lmsys", "haiku", "onpolicy"]
    key = "Qwen_Qwen2.5-7B-Instruct__chat__{p}"
    u1 = [GRID[key.format(p=p)]["u1"]["X_clean->Y_end"] for p in provs]
    u2 = [GRID[key.format(p=p)]["u2"]["X_clean->Y_end"] for p in provs]

    set_paper_style("blog")
    fig, ax = plt.subplots()
    x = np.arange(len(provs))
    w = 0.38
    ax.bar(x - w / 2, u1, w, color="#9aa0a6", label="first user turn (no prior user context)")
    ax.bar(
        x + w / 2, u2, w, color=[C_PROV[p] for p in provs], label="second user turn (after u1 + a1)"
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PROV_LABELS[p] for p in provs])
    ax.set_ylabel("held-out $R^2$ (turn-end token target)")
    ax.legend()
    set_title_subtitle(
        ax,
        "First vs second user turn",
        "chat template, instruct model, clean read slot (last token strictly before the turn)",
    )
    savefig_paper(fig, f"{OUT}/r3b_first_vs_second_turn", dir="figures/")
    plt.close(fig)


def r4_assistant_to_user_ladder() -> None:
    """assistant-chat -> user-chat, 9 rungs, per provenance, instruct, prefix arm."""
    lad = json.load(
        open(ROOT / "eval_results/issue_1689/ladder/ladder_Qwen_Qwen2.5-7B-Instruct_L19.json")
    )
    provs = ["lmsys", "haiku", "onpolicy"]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(1, 10)
    for p in provs:
        obj = lad["pairs"][f"assistant_chat__user_{p}_chat"]["prefix"]
        vals = list(obj["rung_r2s_point"].values())
        ax.plot(x, vals, marker="o", color=C_PROV[p], label=PROV_LABELS_SHORT[p])
        ax.axhline(obj["r2_within_target"], color=C_PROV[p], lw=1.0, ls="--", alpha=0.6)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_yticks([-4, -2, -1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-4", "-2", "-1", "-0.5", "0", "0.5", "1"])
    ax.set_xticks(x)
    ax.set_xticklabels(RUNG_LABELS)
    ax.set_ylabel("held-out $R^2$ on user-turn target (symlog)")
    ax.set_xlabel("transfer tier (correction fitted in target, source operator frozen)")
    ax.legend(title="user-turn provenance (dashed line = target's own ceiling)")
    set_title_subtitle(
        ax,
        "Assistant map transferred to the user turn",
        "assistant-chat source map → user-chat targets, instruct model, prefix arm, L19 (#1689 parent round)",
    )
    savefig_paper(fig, f"{OUT}/r4_assistant_to_user_ladder", dir="figures/")
    plt.close(fig)


def r6_base_vs_instruct_ladder() -> None:
    """chat <-> bare text 9-rung ladder, base vs instruct (#1345), context arm."""
    files = {
        "instruct": ROOT
        / "eval_results/issue_1345/ladder_rungs/ladder_rungs_instruct_context.json",
        "base": ROOT / "eval_results/issue_1345/ladder_rungs/ladder_rungs_pretrained_context.json",
    }
    colors = {"instruct": C_INSTRUCT, "base": C_BASE}

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    x = np.arange(1, 10)
    for ax, dirn, title in [
        (axes[0], "chat->no_template", "chat template → bare text"),
        (axes[1], "no_template->chat", "bare text → chat template"),
    ]:
        for model, path in files.items():
            sub = json.load(open(path))[dirn]
            vals = [v[0] for v in sub["r2"].values()]
            ax.plot(x, vals, marker="o", color=colors[model], label=f"{model} model")
            ax.axhline(sub["ceiling_r2"][0], color=colors[model], lw=1.0, ls="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(RUNG_LABELS, fontsize=7)
        ax.set_title(title, loc="left")
        ax.set_xlabel("transfer tier")
    axes[0].set_ylabel("held-out $R^2$")
    axes[0].legend(title="dashed = target's own ceiling")
    fig.suptitle(
        "Framing transfer ladder, base vs instruct (assistant map, context arm, L19, n=4,724)",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, f"{OUT}/r6_base_vs_instruct_ladder", dir="figures/")
    plt.close(fig)


def r1_framing_ladder() -> None:
    """chat -> {bare text, story inserted, story on-policy}, 9 rungs, instruct, context arm.

    chat->bare comes from the parent ladder round (ambient basis; #1887-stable cells);
    the two story series come from the story_char_ladder_fill round (reduced PCA basis,
    the #1887 corrected regime for n_train < d story cells). Basis mix stated in caption.
    """
    parent = json.load(
        open(ROOT / "eval_results/issue_1345/ladder_rungs/ladder_rungs_instruct_context.json")
    )
    fill = json.load(open(ROOT / "eval_results/issue_1345/story_char_ladder_fill/ladders.json"))

    pal = paper_palette(8)
    series = []  # (label, rung values, ceiling, color)
    sub = parent["chat->no_template"]
    series.append(
        ("bare text (n=4,724)", [v[0] for v in sub["r2"].values()], sub["ceiling_r2"][0], pal[5])
    )

    for pair, dirn, label, color in [
        (
            "chat<->story_inserted",
            "chat->story_inserted",
            "story, inserted answer (n=2,163)",
            pal[6],
        ),
        (
            "chat<->story_onpolicy",
            "chat->story_onpolicy",
            "story, on-policy answer (n=2,018)",
            pal[7],
        ),
    ]:
        obj = fill["ladders"][pair]["reduced"][dirn]
        rungs = obj["r2"]
        vals = [rungs[k][0] for k in sorted(rungs, key=lambda s: int(s.split("_")[0]))]
        series.append((label, vals, obj["ceiling_r2"][0], color))

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(1, 10)
    for label, vals, ceiling, color in series:
        ax.plot(x, vals, marker="o", color=color, label=label)
        ax.axhline(ceiling, color=color, lw=1.0, ls="--", alpha=0.6)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(RUNG_LABELS)
    ax.set_ylabel("held-out $R^2$ on target framing")
    ax.set_xlabel("transfer tier (correction fitted in target, source operator frozen)")
    ax.legend(title="target framing (dashed = target's own ceiling)")
    set_title_subtitle(
        ax,
        "Assistant map transferred across framings",
        "chat-template source map → each target framing, instruct model, context arm, L19",
    )
    savefig_paper(fig, f"{OUT}/r1_framing_ladder", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    r1_framing_ladder()
    r3a_user_turn_provenance()
    r3b_first_vs_second_turn()
    r4_assistant_to_user_ladder()
    r6_base_vs_instruct_ladder()
    print("done")
