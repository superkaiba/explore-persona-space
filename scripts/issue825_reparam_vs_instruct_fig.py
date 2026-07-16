#!/usr/bin/env python
"""Result-2 figure: instruct's own context->answer map vs the REPARAMETERIZED
base map (A_ans o M_base o A_ctx^-1), under three classes of change-of-coordinates.

Shows that a GENERAL LINEAR reparameterization reconstructs the instruct map to its
own held-out ceiling, while a ROTATION (orthogonal) fails — i.e. post-training
re-expresses the same map in different coordinates, not a rigid re-orientation.

Reads the committed eval_results/issue_825/map_alignment/results.json (layer 19,
K=5 held-out, n=5000 paired activations on identical teacher-forced text).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "eval_results" / "issue_825" / "map_alignment" / "results.json"
FIG_DIR = REPO / "figures" / "issue_825"
LAYER = "19"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as e:
        return f"unresolved: {e}"


def main() -> None:
    d = json.loads(SRC.read_text())
    pl = d["per_layer"][LAYER]
    ceil_i = pl["ceilings"]["within_instruct"]
    ceil_b = pl["ceilings"]["within_base"]
    comp = pl["composition"]
    lin = comp["linear"]["comp_samefn_b2i"]
    scaled = comp["scaled_orthogonal"]["comp_samefn_b2i"]
    orth = comp["orthogonal"]["comp_samefn_b2i"]

    bars = [
        ("Instruct\nown map\n(ceiling)", ceil_i, "own"),
        ("Base\nown map", ceil_b, "own"),
        ("Base reparam.\n(general linear)", lin, "reparam"),
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in font_manager.findSystemFonts(fontpaths=None):
        if "Inter" in fp:
            with contextlib.suppress(Exception):
                font_manager.fontManager.addfont(fp)
    with contextlib.suppress(Exception):
        plt.rcParams["font.family"] = "Inter"

    CB = {"own": "#0072B2", "reparam": "#D55E00"}
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = range(len(bars))
    vals = [b[1] for b in bars]
    colors = [CB[b[2]] for b in bars]
    ax.bar(list(xs), vals, color=colors, width=0.62)
    for x, v in zip(xs, vals, strict=False):
        ax.text(
            x,
            v + (0.02 if v >= 0 else -0.05),
            f"{v:.3f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(ceil_i, color="grey", ls="--", lw=1, label=f"instruct ceiling ({ceil_i:.3f})")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("held-out $R^2$ (predict instruct answers), layer 19")
    ax.set_title(
        "Post-training reparameterizes the map: a general linear change of\n"
        "coordinates reconstructs the instruct map to its own ceiling"
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0.0, 0.8)
    # blue = model's own map; orange = base map wrapped in a base->instruct
    # change-of-coordinates, scored on predicting INSTRUCT answers.
    from matplotlib.patches import Patch

    handles = [
        Patch(color=CB["own"], label="model's own map"),
        Patch(color=CB["reparam"], label="base map, reparameterized to instruct"),
    ]
    leg2 = ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.add_artist(leg2)
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "reparam_vs_instruct.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "commit": _git_head(),
        "source": "eval_results/issue_825/map_alignment/results.json",
        "layer": int(LAYER),
        "values": {
            "instruct_own_ceiling": ceil_i,
            "base_own": ceil_b,
            "base_reparam_general_linear": lin,
            "base_reparam_scaled_rotation": scaled,
            "base_reparam_rotation": orth,
        },
        "caption": (
            "Held-out R² at layer 19 for predicting instruct answers. Blue: each model's "
            "own context->answer map (instruct ceiling 0.673, base 0.588). Orange: the base "
            "map run through a fitted general-linear base->instruct change-of-coordinates "
            "(A_ans o M_base o A_ctx^-1), scored on instruct answers — it reaches the "
            "instruct ceiling (0.673), i.e. post-training re-expresses the same map in "
            "different coordinates. (Rotation-restricted variants recorded in `values` but "
            "not plotted: scaled rotation 0.559, pure rotation -0.364.)"
        ),
    }
    (FIG_DIR / "reparam_vs_instruct.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fig] wrote {png}")
    print("values:", meta["values"])


if __name__ == "__main__":
    main()
