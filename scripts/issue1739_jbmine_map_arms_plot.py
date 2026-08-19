"""Figure for the map-regime arms (A/B/C/D/E) of the compliance-DV pilot.

Left: PR-AUC per arm at the common best layer L19 (bars), with the oracle E and
chance (base rate) as reference lines. Right: each map's held-out reconstruction
R^2 of v_A on the jailbreak eval contexts, per layer (benign vs in-domain), with
R^2=0 reference. Simple: axes + legend only, no on-canvas caption block.
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
OUT = Path("docs/scratch/jailbreak_mining_pilot_map_arms.png")
LAYER = "19"
# (arm key, label, bar color)
BARS = [
    ("A_probe_vC", "A: probe on v_C", "C0"),
    ("B_mapproj_benign", "B: map→proj (benign)", "C3"),
    ("B_mapproj_indomain", "B: map→proj (in-dom)", "C3"),
    ("B_mapproj_merged", "B: map→proj (merged)", "C3"),
    ("C_benign", "C: probe on M·v_C (benign)", "C1"),
    ("C_indomain", "C: probe on M·v_C (in-dom)", "C1"),
    ("C_merged", "C: probe on M·v_C (merged)", "C1"),
    ("D_benign", "D: v_A-probe thru M (benign)", "C2"),
    ("D_indomain", "D: v_A-probe thru M (in-dom)", "C2"),
    ("D_merged", "D: v_A-probe thru M (merged)", "C2"),
    ("E_probe_vA_oracle", "E: probe on v_A (oracle)", "0.35"),
]
# (map-regime key, label, color, marker) for the reconstruction-R² panel
R2_LINES = [
    ("benign", "M_benign", "C3", "o"),
    ("indomain", "M_indomain", "C2", "s"),
    ("merged", "M_merged", "C4", "^"),
]


def main() -> int:
    r = json.loads((DEST / "map_arms_results.json").read_text())
    L = r["layers"]
    base = r["eval"]["base_rate"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    vals = [L[LAYER][k]["pr_auc"] for k, _, _ in BARS]
    labels = [lab for _, lab, _ in BARS]
    colors = [c for _, _, c in BARS]
    x = range(len(BARS))
    ax1.bar(x, vals, color=colors, edgecolor="k", linewidth=0.5)
    oracle = L[LAYER]["E_probe_vA_oracle"]["pr_auc"]
    ax1.axhline(oracle, ls="--", color="0.35", lw=1, label=f"oracle E ({oracle:.2f})")
    ax1.axhline(base, ls=":", color="k", lw=1, label=f"chance ({base:.2f})")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("PR-AUC (average precision)")
    ax1.set_ylim(0, 1.03)
    ax1.set_title(
        f"Arms at layer {LAYER} (always-comply vs failed-compliance, 5% base)", fontsize=9
    )
    ax1.legend(fontsize=8, loc="lower center")
    ax1.grid(axis="y", alpha=0.3)

    layers = sorted(int(x) for x in r["map_r2"]["benign"])
    for key, label, color, marker in R2_LINES:
        ax2.plot(
            layers,
            [r["map_r2"][key][str(x)] for x in layers],
            marker=marker,
            ls="-",
            color=color,
            label=label,
            lw=1.6,
            ms=6,
        )
    ax2.axhline(0, ls="--", color="k", lw=1)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("held-out reconstruction R² of v_A")
    ax2.set_xticks(layers)
    ax2.set_title("Map reconstruction of jailbreak answers", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
