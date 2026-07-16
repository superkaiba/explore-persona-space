#!/usr/bin/env python
"""Result-4 figure: DIRECT within-regime held-out R² of a context->answer linear
(ridge) map, fit + tested WITHIN each of six regimes (NOT cross-regime transfer),
vs the single-turn chat map as the reference.

All values are held-out ridge R² at layer 19 (the same estimator + layer as the
chat headline), verbatim from committed eval_results (sources per bar below).
The point: the linear map that reads chat context->answer at 0.588/0.673 does NOT
hold in any of these six regimes (near/below the shuffled floor, or strongly
negative). A nonlinear MLP recovers a little in some regimes (see caption) but
stays far below chat — that is Result 5's story.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# (regime label, {model: ridge_L19_R2}, source)  — held-out ridge R² @ layer 19.
DATA = [
    (
        "chat\n(reference)",
        {"base": 0.5877, "instruct": 0.6731},
        "eval_results/issue_825/cells_{S2,S1}.json :: r2_per_layer_obs[19]",
    ),
    (
        "real stories\n(off-policy)",
        {"instruct": -0.0654},
        "eval_results/issue_931/cells_armA_within_authorfold.json (LOAO; LONO within-fold=+0.173)",
    ),
    (
        "model stories\n(on-policy)",
        {"instruct": 0.1632},
        "eval_results/issue_931/cells_armB_within.json",
    ),
    (
        "user turn\n(real)",
        {"base": -1.0467, "instruct": -0.9861},
        "eval_results/issue_825/real-user-turn-null/cells_M_{pretrained,instruct}_user_chat.json",
    ),
    (
        "user turn\n(model-gen)",
        {"base": -1.8399, "instruct": -0.7689},
        "eval_results/issue_825/onpolicy-user-turn/cells_M_{pretrained,instruct}_user_chat.json",
    ),
    (
        "next-span\n(real)",
        {"base": -2.9157, "instruct": -3.1685},
        "eval_results/issue_825/base-separator-control (within-model)",
    ),
    (
        "next-span\n(model-gen)",
        {"base": -1.5914, "instruct": -2.1930},
        "eval_results/issue_825/onpolicy-separator-control/{base,instruct}/cells_armC_sep.json",
    ),
]
FLOOR = (0.0565, 0.0792)  # shuffled-pairing floor (base, instruct), #1092/make_figures.py
CHAT_LINE = 0.6731  # instruct chat ceiling
# MLP (nonlinear) within-regime R² @ L19, for the caption (Result-5 companion):
MLP_NOTE = (
    "MLP nonlinear R2: stories 0.14-0.15, user real 0.17-0.18, "
    "user model-gen 0.08-0.33, next-span 0.31-0.50 - all far below chat 0.673."
)


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
    C_BASE, C_INST, C_CHAT = "#56B4E9", "#0072B2", "#009E73"

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    xticks, xlabels = [], []
    x = 0.0
    w = 0.38
    for label, models, _src in DATA:
        is_chat = label.startswith("chat")
        keys = [k for k in ("base", "instruct") if k in models]
        if len(keys) == 2:
            for j, k in enumerate(keys):
                col = C_CHAT if is_chat else (C_BASE if k == "base" else C_INST)
                xb = x + (j - 0.5) * w
                v = models[k]
                ax.bar(xb, v, w, color=col)
                ax.text(
                    xb,
                    v + (0.03 if v >= 0 else -0.03),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5,
                )
        else:
            k = keys[0]
            v = models[k]
            col = C_CHAT if is_chat else C_INST
            ax.bar(x, v, w, color=col)
            ax.text(
                x,
                v + (0.03 if v >= 0 else -0.03),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7.5,
            )
        xticks.append(x)
        xlabels.append(label)
        x += 1.0

    ax.axhspan(FLOOR[0], FLOOR[1], color="grey", alpha=0.25, label="shuffled floor")
    ax.axhline(
        CHAT_LINE, color=C_CHAT, ls="--", lw=1, label=f"chat map (instruct, {CHAT_LINE:.2f})"
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("within-regime held-out $R^2$ (ridge, layer 19)")
    ax.set_ylim(-3.4, 0.85)
    ax.set_title(
        "The linear context→answer map is inherited only for chat: fit DIRECTLY within\n"
        "each regime, it fails everywhere else (near/below the shuffled floor or negative)"
    )
    handles = [
        Patch(color=C_CHAT, label="chat reference"),
        Patch(color=C_BASE, label="base (pretrained)"),
        Patch(color=C_INST, label="instruct"),
        Patch(color="grey", alpha=0.25, label="shuffled floor"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()

    png = HERE / "within_regime_r2.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "commit": _git_head(),
        "metric": "within-regime held-out ridge R^2 at layer 19 (direct fit, not transfer)",
        "bars": {label: models for label, models, _ in DATA},
        "sources": {label: src for label, _, src in DATA},
        "shuffled_floor": FLOOR,
        "chat_reference_line": CHAT_LINE,
        "mlp_note": MLP_NOTE,
        "caption": (
            "Direct within-regime held-out ridge R² (layer 19) of a context->answer map fit "
            "+ tested WITHIN each regime (not transferred from chat). The chat map reads at "
            "0.588 (base) / 0.673 (instruct); all six other regimes fail - generic stories sit "
            "at/near the shuffled floor (real off-policy -0.07 author-blocked / +0.17 "
            "within-novel-fold; on-policy 0.16), and the user turn + next-span maps are "
            "strongly negative (worse than predicting the mean). " + MLP_NOTE
        ),
    }
    (HERE / "within_regime_r2.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fig] wrote {png}")
    print("bars:", {label: models for label, models, _ in DATA})


if __name__ == "__main__":
    main()
