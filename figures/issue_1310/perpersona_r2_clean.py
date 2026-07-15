#!/usr/bin/env python
"""#1310 per-character context->dialogue map — clean base-vs-instruct figure.

Fixes the silent-null problem of perpersona_r2.png: the base arm produced no
attributable dialogue (0-1 usable scenes/character), so its R^2 is null and the
original figure just drew nothing. Here the base absence is drawn EXPLICITLY
("no data") so the figure reads correctly instead of looking like a plotting bug.

Per character: instruct held-out R^2 @ layer 19 (real, all negative), a marker at
the shuffle-null 97.5th percentile it "clears", and an explicit base "no data"
placeholder. Assistant-map ceiling (#825 instruct 0.673) drawn for reference.

All values read from eval_results/issue_1310/summary.json (no hardcoding).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUMMARY = REPO / "eval_results/issue_1310/summary.json"


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

    s = json.loads(SUMMARY.read_text())
    order = ["Wren", "HELIOS", "Dana", "Vex"]
    pp = s["per_persona"]
    inst_r2 = [pp[k]["instruct"]["r2_headline"] for k in order]
    inst_null = [pp[k]["instruct"]["null_p975_headline"] for k in order]
    base_present = [pp[k]["base"] is not None for k in order]
    ceil_inst = s["assistant_ceiling"]["S1"]["r2_headline"]  # 0.673

    C_BASE, C_INST = "#0072B2", "#E69F00"  # colorblind-safe
    w = 0.38
    xs = list(range(len(order)))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, k in enumerate(order):
        # base: explicit "no data" placeholder (no measured value exists)
        if base_present[i]:
            ax.bar(i - w / 2, pp[k]["base"]["r2_headline"], w, color=C_BASE, edgecolor="black")
        else:
            ax.bar(i - w / 2, 0.0, w, color="none", edgecolor="grey", linewidth=1.2, hatch="xx")
            ax.text(
                i - w / 2,
                0.03,
                "base\nno data",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="grey",
            )
        # instruct: real (negative) bar
        r = inst_r2[i]
        ax.bar(i + w / 2, r, w, color=C_INST, edgecolor="black")
        ax.text(i + w / 2, r - 0.02, f"{r:.2f}", ha="center", va="top", fontsize=8.5)
        # null p97.5 marker the instruct bar clears
        ax.plot(
            [i + w / 2 - w / 2, i + w / 2 + w / 2],
            [inst_null[i], inst_null[i]],
            color="black",
            lw=1.3,
            ls=":",
        )

    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(
        ceil_inst,
        color="green",
        ls="--",
        lw=1.2,
        label=f"assistant-map ceiling (instruct, #825 = {ceil_inst:.2f})",
    )
    # legend proxy for the null marker
    from matplotlib.lines import Line2D

    handles = [
        Patch(
            facecolor="none",
            edgecolor="grey",
            hatch="xx",
            label="base: no data (attribution failed)",
        ),
        Patch(color=C_INST, label="instruct (measured)"),
        Line2D([0], [0], color="black", ls=":", lw=1.3, label="shuffle null (97.5th pct)"),
        Line2D(
            [0],
            [0],
            color="green",
            ls="--",
            lw=1.2,
            label=f"assistant-map ceiling ({ceil_inst:.2f})",
        ),
    ]
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("held-out $R^2$ (layer 19)")
    ax.set_ylim(-0.62, 0.78)
    ax.set_title(
        "Per-character context→dialogue map in stories (base vs instruct)\n"
        "instruct is at/below the null; base could not be measured"
    )
    ax.legend(handles=handles, fontsize=8, loc="lower left")
    ax.text(
        0.5,
        -0.15,
        "Instruct maps are all negative (−0.12 to −0.29) — a null; they sit just above the "
        "shuffle null (dotted) only because the null is ~−0.53. Base produced 0–1 usable "
        "scenes/character (99.8% of quotes unattributable), so no base map exists.",
        ha="center",
        va="top",
        fontsize=7.2,
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
        "source": "eval_results/issue_1310/summary.json",
        "instruct_r2": dict(zip(order, inst_r2)),
        "instruct_null_p975": dict(zip(order, inst_null)),
        "base": "no data (attribution failed; 0-1 usable scenes/character)",
        "assistant_ceiling_instruct": ceil_inst,
        "caption": (
            "Per-character context->dialogue held-out R2 (layer 19), base vs instruct. "
            "Instruct is a null (-0.12 to -0.29); base could not be measured (attribution "
            "collapsed). Character identity is still specific (swap deltaR2=0.39, separate figure)."
        ),
    }
    (HERE / "perpersona_r2_clean.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fig] wrote {png}")
    print("instruct_r2:", dict(zip(order, [round(v, 3) for v in inst_r2])))
    print("base_present:", dict(zip(order, base_present)))


if __name__ == "__main__":
    main()
