"""Figure: #779's frozen 963k map vs #1739's own map on #1739's eval rungs.

One panel per behavior. There is NO aggregation anywhere in this figure: each
bar group is ONE (variant, layer, eval-rung) cell, so every underlying value is
plotted individually and nothing can hide inside a mean. Within a group the arms
are oracle / raw projection / #1739's 18,793-pair map / shuffled-map control /
best #779 963k arm, one colour per meaning across all panels.

Error bars are the 1000-draw percentile bootstrap CI over contexts, passed as
NON-NEGATIVE OFFSETS clamped element-wise (``max(0, v - lo)`` / ``max(0, hi - v)``)
— a quantile CI can invert around a separately-computed point estimate at small
n, which matplotlib rejects outright.

Cells whose DV is near-constant (a rung where no arm can separate, e.g. evil's
hhrt at dv_std 0.9 vs 26.3 on train) are shaded red: their deltas are noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Before any heavy import: the shared-VM thread caps (#847) are frozen by
# numpy / matplotlib at IMPORT, so load_dotenv() must run first to bind them.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# (headline key, arm key inside cell["arms"], label, colour). One colour = one
# meaning across every panel.
ARMS = [
    ("oracle_rho", "oracle_proj", "oracle (actual answer)", "#9467bd"),
    ("raw_rho", "raw_proj", "raw projection", "#1f77b4"),
    ("arm6_rho", "map_i1739_ufull", "#1739 18,793-pair map", "#ff7f0e"),
    ("shuffled_i1739_rho", "map_i1739_shuffled", "shuffled-map control", "#8c8c8c"),
    ("best_963k_rho", None, "#779 963k map (best)", "#2ca02c"),
]


def _errs(v, ci):
    """Clamped non-negative (lo, hi) offsets for one point."""
    if not isinstance(v, (int, float)) or not ci or ci[0] is None or ci[1] is None:
        return 0.0, 0.0
    return max(0.0, v - ci[0]), max(0.0, ci[1] - v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    payload = json.loads(args.comparison.read_text())
    rows = payload["headline"]
    behaviors = sorted({r["behavior"] for r in rows})

    fig, axes = plt.subplots(len(behaviors), 1, figsize=(15, 4.2 * len(behaviors)), squeeze=False)
    for bi, behavior in enumerate(behaviors):
        ax = axes[bi][0]
        cells = [r for r in rows if r["behavior"] == behavior]
        cells.sort(key=lambda r: (r["variant"], r["layer"], r["eval_rung"]))
        # prefix_end labels carry the DISTINCT-PREFIX-STATE count: the prefix is
        # shared across every context using the same persona, so prefix_end is a
        # coarse categorical score with as many levels as there are prefixes
        # (sycophancy: exactly 1 -> every prefix rho is a rank-tie artifact).
        pres = payload.get("prefix_resolution", {}).get(behavior, {})
        n_pre = pres.get("prefix_end", {}).get("n_distinct_states")
        labels = []
        for c in cells:
            lab = (
                f"{c['variant'].replace('_end', '')}\nL{c['layer']}\n"
                f"{c['eval_rung']}\nn={c['n_contexts']}"
            )
            if c["variant"] == "prefix_end" and n_pre is not None:
                lab += f"\n[{n_pre} prefix states]"
            labels.append(lab)
        x = np.arange(len(cells))
        w = 0.16
        # Shade cells whose DV is near-constant: no arm can separate there.
        for xi, c in enumerate(cells):
            if c.get("dv_degenerate"):
                ax.axvspan(xi - 0.5, xi + 0.5, color="#d62728", alpha=0.07, zorder=0)
            if c["variant"] == "prefix_end" and pres.get("prefix_is_constant"):
                ax.axvspan(xi - 0.5, xi + 0.5, color="#7f7f7f", alpha=0.16, zorder=0)
        for ai, (key, arm_key, label, color) in enumerate(ARMS):
            vals = [c.get(key) for c in cells]
            ys = [v if isinstance(v, (int, float)) else np.nan for v in vals]
            lo, hi = [], []
            for c, v in zip(cells, vals, strict=True):
                if key == "best_963k_rho":
                    ci = c.get("best_963k_ci95")
                else:
                    ci = c.get("arms", {}).get(arm_key, {}).get("ci95")
                a, b = _errs(v, ci)
                lo.append(a)
                hi.append(b)
            ax.bar(
                x + (ai - 2) * w,
                ys,
                w,
                label=label,
                color=color,
                yerr=[lo, hi] if any(lo) or any(hi) else None,
                capsize=2,
                error_kw={"lw": 0.8},
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel("Spearman rho vs judged DV")
        rb = cells[0]["r_b_source"] if cells else "?"
        n_deg = sum(1 for c in cells if c.get("dv_degenerate"))
        ax.set_title(
            f"{behavior} — map read-out vs judged DV (r_B source: {rb}; "
            "error bars = 1000-draw percentile bootstrap CI over contexts"
            + (f"; red-shaded: near-constant DV, no arm separates, {n_deg} cells" if n_deg else "")
            + (
                "; grey-shaded: prefix_end is CONSTANT across contexts, every rho a tie artifact)"
                if pres.get("prefix_is_constant")
                else ")"
            ),
            fontsize=9,
        )
        ax.legend(fontsize=7, ncol=5, loc="upper center")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "#779 frozen 963,444-context map vs #1739 18,793-pair map — #1739 eval rungs",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
