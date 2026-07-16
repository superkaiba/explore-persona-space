#!/usr/bin/env python
"""Result-5 figure: predicting the ASSISTANT answer two turns ahead — the linear
(ridge) map collapses while the nonlinear (MLP) map survives.

Held-out R² at layer 19 (ridge = best-frozen for the MLP), per cell. Single-turn
chat cells are the reference (one turn ahead: both ridge and MLP read ~0.6). The
four two-turn ASSISTANT cells are the point: ridge falls to ~0 / negative, MLP
stays 0.49-0.56. User cells are excluded (they are Result 4's user-turn point).

Values verbatim from committed #825 eval_results (sources in meta.json).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# (label, ridge_R2_L19, mlp_R2, group)  group: "single" (reference) | "two_turn"
CELLS = [
    ("single-turn\ninstruct", 0.6731, 0.6537, "single"),
    ("single-turn\npretrained", 0.5877, 0.5870, "single"),
    ("2-turn asst\ninstruct chat", 0.0757, 0.5575, "two_turn"),
    ("2-turn asst\npretrained chat", -0.4606, 0.4873, "two_turn"),
    ("2-turn asst\ninstruct nat.", -0.0784, 0.5335, "two_turn"),
    ("2-turn asst\npretrained nat.", -0.3897, 0.4988, "two_turn"),
]
SOURCES = (
    "ridge: eval_results/issue_825/cells_{S1,S2}.json + cells_M_*_assistant_*.json "
    "(r2_per_layer_obs[19] / selection_symmetric frozen best); "
    "MLP: cells_S{1,2}.mlp + mlp-unprobed-cells/headline_metrics.json"
)
RECOV_LINE = 0.2  # pre-registered recoverability threshold (#825 plan v6 kill line)


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
    C_RIDGE, C_MLP = "#D55E00", "#0072B2"

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    w = 0.38
    xs = list(range(len(CELLS)))
    for i, (_label, ridge, mlp, _grp) in enumerate(CELLS):
        ax.bar(i - w / 2, ridge, w, color=C_RIDGE)
        ax.bar(i + w / 2, mlp, w, color=C_MLP)
        ax.text(
            i - w / 2,
            ridge + (0.02 if ridge >= 0 else -0.02),
            f"{ridge:.2f}",
            ha="center",
            va="bottom" if ridge >= 0 else "top",
            fontsize=7.5,
        )
        ax.text(i + w / 2, mlp + 0.02, f"{mlp:.2f}", ha="center", va="bottom", fontsize=7.5)

    # separator between the single-turn reference and the two-turn cells
    n_single = sum(1 for c in CELLS if c[3] == "single")
    ax.axvline(n_single - 0.5, color="grey", lw=0.8, ls=":")
    ax.text(
        0.5,
        0.78,
        "single-turn\n(reference)",
        ha="center",
        fontsize=8,
        color="grey",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        (n_single + len(CELLS) - 1) / 2,
        0.78,
        "two turns ahead (assistant)",
        ha="center",
        fontsize=8,
        color="grey",
        transform=ax.get_xaxis_transform(),
    )

    ax.axhline(
        RECOV_LINE,
        color="grey",
        ls="--",
        lw=1,
        label=f"recoverability threshold ({RECOV_LINE}, pre-registered)",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in CELLS], fontsize=8)
    ax.set_ylabel("held-out $R^2$ (layer 19)")
    ax.set_ylim(-0.6, 0.85)
    ax.set_title(
        "Two turns ahead: the linear map collapses, the nonlinear map survives\n"
        "(predicting the assistant answer two turns out)"
    )
    handles = [
        Patch(color=C_RIDGE, label="ridge (linear)"),
        Patch(color=C_MLP, label="MLP (nonlinear)"),
    ]
    leg = ax.legend(handles=handles, fontsize=8, loc="lower left")
    ax.add_artist(leg)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    png = HERE / "ridge_vs_mlp_2turn.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "commit": _git_head(),
        "cells": {c[0].replace(chr(10), " "): {"ridge": c[1], "mlp": c[2]} for c in CELLS},
        "sources": SOURCES,
        "recoverability_line": RECOV_LINE,
        "recoverability_line_provenance": (
            "pre-registered kill threshold, #825 follow-up plan v6: MLP R2 > 0.2 (nulls ~ -0.01) "
            "= the answer is nonlinearly recoverable / the map has practical predictive power."
        ),
        "caption": (
            "Held-out R2 (layer 19) predicting the assistant answer. Single-turn cells "
            "(reference): ridge and MLP both ~0.6 - one turn ahead is linearly readable. Two "
            "turns ahead (assistant): ridge collapses (+0.08 to -0.46) while the MLP holds "
            "(0.49-0.56), so the information is present but not linearly readable. User cells "
            "excluded (Result 4). Dashed line = the 0.2 recoverability threshold from the "
            "follow-up plan."
        ),
    }
    (HERE / "ridge_vs_mlp_2turn.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fig] wrote {png}")
    print("cells:", meta["cells"])


if __name__ == "__main__":
    main()
