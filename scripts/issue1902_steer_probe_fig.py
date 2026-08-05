#!/usr/bin/env python3
"""Figure for the #1902 steer_probe inline round.

Point plot of cos(mean pooled answer-state shift, target) per intervention
arm — targets c* (the optimal constant correction) and dy (the answer-cloud
mean shift) — from eval_results/issue_1902/followup_ladder/steer_probe.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_1902/followup_ladder/steer_probe.json",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures/issue_1902")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    import numpy as np

    s = json.load(open(args.summary))
    arms = s["arms"]
    null_band = s.get("null_band")
    bands = null_band["bands"] if null_band else {}

    # Tick sequence: each preimage arm, followed by its matched-layer null cloud
    # (when present), then the original single random draw, then rig-sanity.
    ticks: list[tuple[str, str]] = []  # (kind, key): kind in {arm, null}
    for n in (n for n in arms if n.startswith("pre_")):
        ticks.append(("arm", n))
        ell = n.removeprefix("pre_L").removesuffix("_ctx")
        if ell in bands:
            ticks.append(("null", ell))
    ticks += [("arm", n) for n in arms if n.startswith("rand_")]
    ticks += [("arm", n) for n in arms if n.startswith("rig_sanity")]

    labels = []
    for kind, key in ticks:
        if kind == "null":
            labels.append(f"random null\nx{len(null_band['seeds'])} @ctx L{key}")
        elif key.startswith("rig_sanity"):
            labels.append(f"rig sanity\n(+dy @ans, L{s['layer_capture']})")
        else:
            labels.append(
                key.replace("pre_", "preimage\n")
                .replace("_ctx", " @ctx")
                .replace("rand_", "random\n")
            )

    set_paper_style()
    fig, ax = plt.subplots(figsize=(8.6 if bands else 7.0, 4.2))
    c_primary = paper_palette_role("primary")
    c_baseline = paper_palette_role("baseline")
    c_neutral = paper_palette_role("neutral")
    for xi, (kind, key) in enumerate(ticks):
        if kind == "arm":
            cc = arms[key]["cos_mean_shift_c_star"]
            cd = arms[key]["cos_mean_shift_dy"]
            ax.scatter([xi], [cc], s=90, color=c_primary, zorder=3)
            ax.scatter([xi], [cd], s=90, marker="D", color=c_baseline, zorder=3)
            ax.vlines(xi, min(cc, cd), max(cc, cd), color=c_neutral, lw=1, alpha=0.5)
        else:
            band = bands[key]
            k = len(band["c_star"]["values"])
            jit = np.linspace(-0.18, 0.18, k)
            ax.scatter(
                xi + jit, band["c_star"]["values"], s=36, color=c_primary, alpha=0.55, zorder=2
            )
            ax.scatter(
                xi + jit,
                band["dy"]["values"],
                s=36,
                marker="D",
                color=c_baseline,
                alpha=0.55,
                zorder=2,
            )
            for off, tgt, col in ((-0.28, "c_star", c_primary), (0.28, "dy", c_baseline)):
                ax.vlines(xi + off, band[tgt]["min"], band[tgt]["max"], color=col, lw=2, alpha=0.4)
    # Legend proxies (loop-drawn points carry no labels).
    ax.scatter([], [], s=90, color=c_primary, label="cos(mean shift, c*)")
    ax.scatter([], [], s=90, marker="D", color=c_baseline, label="cos(mean shift, dy)")
    if bands:
        ax.scatter([], [], s=36, color=c_neutral, alpha=0.55, label="matched random nulls")
    ax.axhline(0.0, color=c_neutral, lw=1, ls="--", alpha=0.7)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("cos(mean answer-state shift, target)")
    ax.set_title(
        "OLMo-2-7B base: residual-stream intervention at context positions\n"
        f"(n={s['n_rows']} contexts, layer-{s['layer_capture']} answer-pooled states; "
        f"rig-sanity gate {'PASS' if s['rig_sanity_gate']['pass'] else 'FAIL'})"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = savefig_paper(fig, "steer_probe", dir=args.out_dir, formats=("png",))
    print(f"[fig] wrote {paths}")
    sys.exit(0)


if __name__ == "__main__":
    main()
