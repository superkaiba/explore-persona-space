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

    s = json.load(open(args.summary))
    arms = s["arms"]
    order = [n for n in arms if n.startswith("pre_")]
    order += [n for n in arms if n.startswith("rand_")]
    order += [n for n in arms if n.startswith("rig_sanity")]
    labels = {
        n: n.replace("pre_", "preimage\n").replace("_ctx", " @ctx").replace("rand_", "random\n")
        for n in order
    }
    for n in order:
        if n.startswith("rig_sanity"):
            labels[n] = f"rig sanity\n(+dy @ans, L{s['layer_capture']})"

    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = range(len(order))
    cos_c = [arms[n]["cos_mean_shift_c_star"] for n in order]
    cos_dy = [arms[n]["cos_mean_shift_dy"] for n in order]
    ax.scatter(x, cos_c, s=90, color=paper_palette_role("primary"), label="cos(mean shift, c*)")
    ax.scatter(
        x,
        cos_dy,
        s=90,
        marker="D",
        color=paper_palette_role("baseline"),
        label="cos(mean shift, dy)",
    )
    for xi, (cc, cd) in zip(x, zip(cos_c, cos_dy)):
        ax.vlines(
            xi, min(cc, cd), max(cc, cd), color=paper_palette_role("neutral"), lw=1, alpha=0.5
        )
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=1, ls="--", alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels[n] for n in order])
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
