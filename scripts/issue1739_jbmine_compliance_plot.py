"""PR-AUC-by-layer figure for the compliance-DV jailbreak-mining pilot.

Two panels (the two 5% needle-in-haystack pools): always-comply vs benign,
and always-comply vs failed-compliance same-family jailbreaks. Chance = base
rate (0.05) dashed. Simple: axes + legend only, no on-canvas caption block.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEST = Path(f"/mnt/eps-data/{os.environ['USER']}/issue1739_jbmine")
OUT = Path("docs/scratch/jailbreak_mining_pilot_compliance_pr_by_layer.png")
ARMS = [
    ("probe", "L2 logistic probe on v_C", "C0", "o"),
    ("map_then_project", "map-then-project (h·r_B)", "C1", "s"),
    ("rb_harmcomp", "fixed harm-compliance dir", "C2", "^"),
    ("rb_refusal", "fixed refusal dir", "C3", "v"),
    ("random", "random", "0.6", "x"),
]
PANELS = [
    ("needle_benign_5pct", "always-comply vs benign (5% base)"),
    (
        "hardneg_failcomp_5pct",
        "always-comply vs failed-compliance\nsame-family jailbreaks (5% base)",
    ),
]


def main() -> int:
    res = json.loads((DEST / "compliance_pilot_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (pool, title) in zip(axes, PANELS):
        pr = res[pool]
        layers = sorted(int(x) for x in pr["layers"])
        for arm, label, color, marker in ARMS:
            y = [pr["layers"][str(L)][arm]["pr_auc"] for L in layers]
            ax.plot(layers, y, marker=marker, color=color, label=label, lw=1.6, ms=6)
        ax.axhline(
            pr["base_rate"], ls="--", color="k", lw=1, label=f"chance ({pr['base_rate']:.2f})"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("layer")
        ax.set_xticks(layers)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PR-AUC (average precision)")
    axes[1].legend(fontsize=8, loc="center right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
