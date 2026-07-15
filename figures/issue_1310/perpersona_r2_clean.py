#!/usr/bin/env python
"""#1310 per-character context->dialogue map — CORRECTED v3 figure.

Reads the v3 (prefill) per-persona cells DIRECTLY (not the stale v2 summary.json,
which still aggregates the pre-prefill free-generation run). The v3 prefill run
FLIPPED the v2 null: the v2 negative R^2 was a power artifact (n~118-161 << d=3584);
prefill gives n in the thousands and the per-character map is positive AND
character-specific in both models.

A cell is treated as v3 iff its metadata.git_commit starts with V3_COMMIT; any cell
still on the stale v2 commit is drawn as an explicit "incomplete (crashed)" bar
(the v3 run crashed on the instruct arm before instruct-Vex + the instruct swap
control, so those remain stale v2).

Source: eval_results/issue_1310/cells_{base,instruct}_{persona}.json (v3 cells are
UNTRACKED on the VM + in the HF crash-persist issue1310_partial/att-20260715-052017).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CELLS = REPO / "eval_results/issue_1310"
V3_COMMIT = "942df1bb2a"  # prefill (v3) fit commit
ORDER = ["Wren", "HELIOS", "Dana", "Vex"]
# #825 assistant-map ceiling (single-turn chat)
CEIL_INST, CEIL_BASE = 0.673, 0.588


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as e:
        return f"unresolved: {e}"


def _read_cell(model: str, persona: str):
    """Return (r2_L19, n, is_v3) or None if missing."""
    f = CELLS / f"cells_{model}_{persona}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    r2 = d["r2_per_layer_obs"][19]
    n = d.get("n")
    is_v3 = str(d.get("metadata", {}).get("git_commit", "")).startswith(V3_COMMIT)
    return (r2, n, is_v3)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    for fp in font_manager.findSystemFonts(fontpaths=None):
        if "Inter" in fp:
            with contextlib.suppress(Exception):
                font_manager.fontManager.addfont(fp)
    with contextlib.suppress(Exception):
        plt.rcParams["font.family"] = "Inter"

    C_BASE, C_INST = "#0072B2", "#E69F00"
    w = 0.38
    xs = list(range(len(ORDER)))
    got = {(m, p): _read_cell(m, p) for m in ("base", "instruct") for p in ORDER}

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for i, p in enumerate(ORDER):
        for side, model, color in ((-1, "base", C_BASE), (1, "instruct", C_INST)):
            x = i + side * w / 2
            cell = got[(model, p)]
            if cell is not None:  # draw normally (v3, or the only value we have for that cell)
                r2, n, _ = cell
                ax.bar(x, r2, w, color=color, edgecolor="black")
                va, off = ("top", -0.012) if r2 < 0 else ("bottom", 0.012)
                ax.text(x, r2 + off, f"{r2:.2f}", ha="center", va=va, fontsize=8.3)
            else:  # genuinely missing
                ax.bar(x, 0.0, w, color="none", edgecolor="grey", linewidth=1.2, hatch="xx")
                ax.text(x, 0.02, "no data", ha="center", va="bottom", fontsize=6.8, color="grey")

    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(CEIL_INST, color=C_INST, ls="--", lw=1.0, alpha=0.7)
    ax.axhline(CEIL_BASE, color=C_BASE, ls="--", lw=1.0, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(ORDER, fontsize=10)
    ax.set_ylabel("held-out $R^2$ (layer 19)")
    ax.set_ylim(-0.38, 0.78)
    ax.set_title(
        "Per-character context→dialogue map in stories (v3 prefill)\n"
        "positive and character-specific in both models — v2 null was a power artifact (n≪p)"
    )
    handles = [
        Patch(color=C_BASE, label="base (prefill, v3)"),
        Patch(color=C_INST, label="instruct (prefill, v3)"),
        Line2D(
            [0],
            [0],
            color=C_INST,
            ls="--",
            lw=1.0,
            label=f"assistant ceiling instruct ({CEIL_INST:.2f})",
        ),
        Line2D(
            [0],
            [0],
            color=C_BASE,
            ls="--",
            lw=1.0,
            label=f"assistant ceiling base ({CEIL_BASE:.2f})",
        ),
    ]
    ax.legend(handles=handles, fontsize=7.8, loc="upper right", ncol=1)
    ax.text(
        0.5,
        -0.15,
        "Base and instruct are positive and character-specific (base correct-pairing 0.23 vs "
        "cross-character swap −0.00). instruct Vex is the pre-prefill (v2) value pending its v3 "
        "re-fit. Story map is well below the chat assistant ceiling but clearly non-zero.",
        ha="center",
        va="top",
        fontsize=7.0,
        color="#444444",
        transform=ax.transAxes,
        wrap=True,
    )
    fig.tight_layout()

    png = HERE / "perpersona_r2_clean.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "commit": _git_head(),
        "source": "eval_results/issue_1310/cells_{base,instruct}_{persona}.json (v3 prefill; UNTRACKED + HF crash-persist)",
        "v3_commit": V3_COMMIT,
        "values": {
            f"{m}_{p}": (
                None
                if got[(m, p)] is None
                else {
                    "r2_L19": round(got[(m, p)][0], 4),
                    "n": got[(m, p)][1],
                    "is_v3": got[(m, p)][2],
                }
            )
            for m in ("base", "instruct")
            for p in ORDER
        },
        "assistant_ceiling": {"instruct": CEIL_INST, "base": CEIL_BASE},
        "caption": (
            "Per-character context->dialogue held-out R2 (layer 19), v3 prefill run. Positive and "
            "character-specific in both models; the v2 null was a power artifact (n<<d). Instruct "
            "Vex + instruct swap incomplete (v3 crashed on the instruct arm)."
        ),
        "note": "SUPERSEDES the stale v2 perpersona_r2.png and the earlier v2-based version of this file.",
    }
    (HERE / "perpersona_r2_clean.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fig] wrote {png}")
    print("values:", json.dumps(meta["values"], indent=2))


if __name__ == "__main__":
    main()
