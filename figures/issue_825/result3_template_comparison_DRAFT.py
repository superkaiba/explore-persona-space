#!/usr/bin/env python
"""Result-3 DRAFT figure: does the context->answer map survive without chat-template
tokens? Grouped bars, held-out R^2 @ layer 19, base vs instruct, chat vs no-template.

THREE bars are MEASURED, ONE is a GUESS (drawn hatched + grey + labelled) because the
instruct single-turn no-template cell has not been computed yet — the naturalistic
Track-S run (#825) is producing it now. This figure is a placeholder so the writeup
has a visual; the guessed bar MUST be replaced with the measured value when S1N lands.

Provenance per bar (stated on the figure):
  base    chat        0.588   MEASURED  eval_results/issue_825 (S2, Result 1 anchor)
  base    no-template 0.72    MEASURED  #1092, range 0.71-0.74 (plain "User:/Assistant:")
  instruct chat       0.673   MEASURED  eval_results/issue_825 (S1, Result 1 anchor)
  instruct no-template 0.66   GUESS     pending naturalistic Track-S run (#825)
  shuffled floor      ~0.06   MEASURED  Result 1 null
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# (model, chat_R2, chat_measured, notmpl_R2, notmpl_measured)
BARS = [
    ("Pretrained\nbase", 0.588, True, 0.72, True),
    ("Instruct", 0.673, True, 0.66, False),  # instruct no-template = GUESS
]
FLOOR = 0.06  # shuffled-pairing null (Result 1)
NOTMPL_RANGE = (0.71, 0.74)  # base no-template measured range (#1092)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as e:
        return f"unresolved: {e}"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    for fp in font_manager.findSystemFonts(fontpaths=None):
        if "Inter" in fp:
            with contextlib.suppress(Exception):
                font_manager.fontManager.addfont(fp)
    with contextlib.suppress(Exception):
        plt.rcParams["font.family"] = "Inter"

    C_CHAT, C_NOTMPL = "#0072B2", "#E69F00"  # colorblind-safe blue / orange
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    xs = list(range(len(BARS)))
    for i, (_lbl, chat, chat_m, notmpl, notmpl_m) in enumerate(BARS):
        # chat bar (all measured here)
        ax.bar(i - w / 2, chat, w, color=C_CHAT, edgecolor="black", linewidth=0.6)
        ax.text(i - w / 2, chat + 0.015, f"{chat:.2f}", ha="center", va="bottom", fontsize=9)
        # no-template bar: solid if measured, hatched+grey if a guess
        if notmpl_m:
            ax.bar(i + w / 2, notmpl, w, color=C_NOTMPL, edgecolor="black", linewidth=0.6)
            ax.text(
                i + w / 2, notmpl + 0.015, f"{notmpl:.2f}", ha="center", va="bottom", fontsize=9
            )
        else:
            ax.bar(
                i + w / 2,
                notmpl,
                w,
                color="lightgrey",
                edgecolor=C_NOTMPL,
                linewidth=1.6,
                hatch="////",
            )
            ax.text(
                i + w / 2,
                notmpl + 0.015,
                f"~{notmpl:.2f}\nGUESS",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#B00000",
                fontweight="bold",
            )

    ax.axhline(FLOOR, color="grey", ls="--", lw=1, label=f"shuffled floor (~{FLOOR:.2f})")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in BARS], fontsize=10)
    ax.set_ylabel("held-out $R^2$ (layer 19)")
    ax.set_ylim(0, 0.85)
    ax.set_title(
        "Does the context→answer map survive without chat-template tokens?\n"
        'chat template vs plain "User:/Assistant:"  —  DRAFT (1 bar guessed)'
    )
    handles = [
        Patch(color=C_CHAT, label="chat template (measured)"),
        Patch(color=C_NOTMPL, label="no template (measured)"),
        Patch(
            facecolor="lightgrey",
            edgecolor=C_NOTMPL,
            hatch="////",
            label="no template (GUESS — pending run)",
        ),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper right")
    ax.text(
        0.5,
        -0.16,
        "MEASURED: base chat 0.588 / instruct chat 0.673 (#825, Result 1 anchor); "
        f"base no-template {NOTMPL_RANGE[0]:.2f}–{NOTMPL_RANGE[1]:.2f} (#1092).  "
        "GUESSED: instruct no-template (naturalistic Track-S run producing it now).",
        ha="center",
        va="top",
        fontsize=7.2,
        color="#444444",
        transform=ax.transAxes,
        wrap=True,
    )
    fig.tight_layout()

    png = HERE / "result3_template_comparison_DRAFT.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "commit": _git_head(),
        "status": "DRAFT — instruct no-template bar is a GUESS pending the naturalistic Track-S run (#825)",
        "bars": {
            "base_chat": {
                "r2": 0.588,
                "measured": True,
                "source": "eval_results/issue_825 S2 (Result 1 anchor)",
            },
            "base_no_template": {
                "r2": 0.72,
                "range": NOTMPL_RANGE,
                "measured": True,
                "source": "#1092",
            },
            "instruct_chat": {
                "r2": 0.673,
                "measured": True,
                "source": "eval_results/issue_825 S1 (Result 1 anchor)",
            },
            "instruct_no_template": {
                "r2": 0.66,
                "measured": False,
                "source": "GUESS — pending #825 naturalistic Track-S (S1N)",
            },
        },
        "shuffled_floor": FLOOR,
        "caption": (
            "Held-out R2 (layer 19) of the single-turn context->answer map, chat template vs "
            "plain 'User:/Assistant:'. Three bars measured; the instruct no-template bar is a "
            "placeholder GUESS (hatched) pending the naturalistic Track-S run. The map holds up "
            "without the template in the base model (0.588->~0.72); the instruct no-template value "
            "is not yet measured."
        ),
    }
    (HERE / "result3_template_comparison_DRAFT.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fig] wrote {png}")
    print("bars:", json.dumps(meta["bars"], indent=2))


if __name__ == "__main__":
    main()
