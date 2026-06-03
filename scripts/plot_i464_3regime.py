"""Issue #464 — three-regime comparison of wrong-encoding marker leakage.

Shows the core finding of the positive-only + contrastive-negatives
follow-ups: contrast (not the role token) is what localizes the marker,
and the role-vs-system gap is large only in the co-resident
competing-marker regime.

x = three training regimes; bars = the three core arms (system_plain,
system_padded, role); y = off-diagonal leakage = raw log P(marker) under
the OTHER persona's same-arm encoding (more negative = more localized).
Per-seed dots overlaid. No annotations (project rule).

Reads the three committed analysis.json files; writes
figures/issue_464/three_regime_clean.{png,pdf} + meta.json.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Inter if available, else DejaVu Sans (project convention).
for _f in ("Inter", "Inter-Regular"):
    try:
        fm.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue

ARMS = ("system_plain", "system_padded", "role")
ARM_LABEL = {
    "system_plain": "persona in system prompt",
    "system_padded": "system prompt + filler",
    "role": "persona in role header",
}
ARM_COLOR = {"system_plain": "#9aa0a6", "system_padded": "#c2b280", "role": "#1a73e8"}

REGIMES = [
    (
        "positive_only",
        "No contrast\n(positive-only)",
        "eval_results/issue_464/positive_only/analysis.json",
    ),
    (
        "contrastive_neg",
        "Marker-less\ncontrastive negative",
        "eval_results/issue_464/contrastive_negatives/analysis.json",
    ),
    ("co_resident", "Co-resident\ncompeting marker (#464)", "eval_results/issue_464/analysis.json"),
]


def _per_seed(fp: str, arm: str) -> list[float]:
    d = json.loads(Path(fp).read_text())
    seeds = d["L_per_arm_per_seed"].get(arm, {})
    return [float(v) for v in seeds.values()]


def main() -> None:
    """Build the three-regime grouped bar chart."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    n_arms = len(ARMS)
    group_w = 0.8
    bar_w = group_w / n_arms

    meta_rows = []
    for gi, (_key, _label, fp) in enumerate(REGIMES):
        for ai, arm in enumerate(ARMS):
            vals = _per_seed(fp, arm)
            mean = statistics.mean(vals)
            x = gi + (ai - (n_arms - 1) / 2) * bar_w
            ax.bar(
                x,
                mean,
                width=bar_w * 0.92,
                color=ARM_COLOR[arm],
                edgecolor="black",
                linewidth=0.6,
                label=ARM_LABEL[arm] if gi == 0 else None,
                zorder=2,
            )
            ax.scatter(
                [x] * len(vals),
                vals,
                color="black",
                s=14,
                zorder=3,
            )
            meta_rows.append({"regime": _key, "arm": arm, "mean": mean, "per_seed": vals})

    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([lbl for _, lbl, _ in REGIMES])
    ax.set_ylabel("wrong-encoding leakage:\nlog P(marker) under the other persona (nats)")
    ax.set_title("Wrong-encoding marker leakage by training regime", fontsize=12)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = Path("figures/issue_464")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "three_regime_clean.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "three_regime_clean.pdf", bbox_inches="tight")
    (out / "three_regime_clean.meta.json").write_text(
        json.dumps(
            {
                "description": "Three-regime comparison of off-diagonal (wrong-encoding) "
                "marker leakage. Lower = more localized.",
                "y": "off-diagonal raw log P(marker) under the other persona's same-arm encoding",
                "rows": meta_rows,
            },
            indent=2,
        )
    )
    print("wrote figures/issue_464/three_regime_clean.png")
    for r in meta_rows:
        print(f"  {r['regime']:16s} {r['arm']:14s} mean={r['mean']:8.3f}")


if __name__ == "__main__":
    main()
