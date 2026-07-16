#!/usr/bin/env python
"""Result-3 figure: does the context->answer map survive without chat-template
tokens? Grouped bars, held-out R^2 @ layer 19, base vs instruct, chat vs no-template.

VISUAL: all four bars are drawn in uniform style (no guess marking) per request.
HONESTY: the instruct single-turn no-template value is NOT yet measured (the
naturalistic Track-S run #825 is producing it). That fact is preserved in the
sidecar meta.json (`instruct_no_template.measured = false`) and the filename, and
MUST be swapped for the real value before this figure goes to a mentor.

Per-bar provenance:
  base    chat        0.588   MEASURED  eval_results/issue_825 (S2, Result 1 anchor)
  base    no-template 0.72    MEASURED  #1092, range 0.71-0.74 (plain "User:/Assistant:")
  instruct chat       0.673   MEASURED  eval_results/issue_825 (S1, Result 1 anchor)
  instruct no-template 0.66   NOT MEASURED (placeholder) pending #825 naturalistic Track-S
  shuffled floor      ~0.06   MEASURED  Result 1 null
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# (model, chat_R2, notmpl_R2)
BARS = [
    ("Pretrained\nbase", 0.588, 0.72),
    ("Instruct", 0.673, 0.66),  # instruct no-template = placeholder (not measured)
]
FLOOR = 0.06  # shuffled-pairing null (Result 1)


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
    for i, (_lbl, chat, notmpl) in enumerate(BARS):
        ax.bar(i - w / 2, chat, w, color=C_CHAT, edgecolor="black", linewidth=0.6)
        ax.text(i - w / 2, chat + 0.015, f"{chat:.2f}", ha="center", va="bottom", fontsize=9)
        ax.bar(i + w / 2, notmpl, w, color=C_NOTMPL, edgecolor="black", linewidth=0.6)
        ax.text(i + w / 2, notmpl + 0.015, f"{notmpl:.2f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(FLOOR, color="grey", ls="--", lw=1, label=f"shuffled floor (~{FLOOR:.2f})")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in BARS], fontsize=10)
    ax.set_ylabel("held-out $R^2$ (layer 19)")
    ax.set_ylim(0, 0.85)
    ax.set_title(
        "Does the context→answer map survive without chat-template tokens?\n"
        'chat template vs plain "User:/Assistant:"'
    )
    handles = [
        Patch(color=C_CHAT, label="chat template"),
        Patch(color=C_NOTMPL, label='no template ("User:/Assistant:")'),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    fig.tight_layout()

    png = HERE / "result3_template_comparison_DRAFT.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "commit": _git_head(),
        "status": "DRAFT — instruct no-template value NOT measured (placeholder 0.66) pending #825 naturalistic Track-S run; visual guess-marking removed per request",
        "bars": {
            "base_chat": {
                "r2": 0.588,
                "measured": True,
                "source": "eval_results/issue_825 S2 (Result 1 anchor)",
            },
            "base_no_template": {
                "r2": 0.72,
                "range": [0.71, 0.74],
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
                "source": "PLACEHOLDER — pending #825 naturalistic Track-S (S1N)",
            },
        },
        "shuffled_floor": FLOOR,
        "caption": (
            "Held-out R2 (layer 19) of the single-turn context->answer map, chat template vs "
            "plain 'User:/Assistant:'. The map holds up without the template in the base model "
            "(0.588->~0.72). The instruct no-template bar is a placeholder (0.66), not yet measured."
        ),
    }
    (HERE / "result3_template_comparison_DRAFT.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fig] wrote {png}")
    print("bars:", json.dumps(meta["bars"], indent=2))


if __name__ == "__main__":
    main()
