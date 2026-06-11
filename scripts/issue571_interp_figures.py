# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, Δ, −) in scientific labels.
"""Task #571 — interpretation-grade figures (blog style, plain-English labels).

Reads the committed analysis output (``eval_results/issue_571/breadth_contrast.json``)
plus the raw-completion JSONs and renders the three figures embedded in the
clean-result body:

1. ``hero_breadth_paired`` — paired persona plot, broad vs narrow never-negative
   Δz_EOS, persona-cluster bootstrap CIs, reference lines at #560's broad-recipe
   never-negative mean (+13.66) and the 4-negative lineage anchor (−3.1).
2. ``marker_hijack_rates`` — per-adapter fraction of held-out generations that
   emit the marker (text-level), the trade-off finding.
3. ``per_adapter_clamp_anchor`` — per-adapter never-negative mean Δz_EOS with
   per-persona scatter, the replication-anchor band, and reference lines.

CPU-only, VM-side. Run after issue571_breadth_analysis.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

OUT_DIR = PROJECT_ROOT / "eval_results/issue_571"
FIG_DIR = PROJECT_ROOT / "figures/issue_571"
N_BOOT, SEED = 10_000, 42

LABELS = ["broad_s42", "broad_s43", "narrow_s42", "narrow_s43"]
PLAIN = {
    "broad_s42": "broad panel\n(15 contexts), seed 42",
    "broad_s43": "broad panel\n(15 contexts), seed 43",
    "narrow_s42": "narrow panel\n(4 contexts), seed 42",
    "narrow_s43": "narrow panel\n(4 contexts), seed 43",
}
C_BROAD = paper_palette_role("primary")
C_NARROW = paper_palette_role("accent")
C_NEUTRAL = paper_palette_role("neutral")


def boot_ci(values: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean over persona clusters."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    set_paper_style("blog")
    r = json.loads((OUT_DIR / "breadth_contrast.json").read_text())
    nn = r["config"]["never_negative_personas"]
    per_label = r["per_label_per_persona"]

    arm_pp = {
        arm: np.array(
            [
                np.mean(
                    [per_label[f"{arm}_s42"]["dz_eos"][p], per_label[f"{arm}_s43"]["dz_eos"][p]]
                )
                for p in nn
            ]
        )
        for arm in ("broad", "narrow")
    }

    # ── 1. Hero: paired persona plot ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for b, n in zip(arm_pp["broad"], arm_pp["narrow"], strict=True):
        ax.plot([0, 1], [b, n], color=C_NEUTRAL, alpha=0.45, lw=0.8, zorder=1)
    for x, arm, color in ((0, "broad", C_BROAD), (1, "narrow", C_NARROW)):
        vals = arm_pp[arm]
        m = float(vals.mean())
        lo, hi = boot_ci(vals)
        ax.errorbar(
            [x],
            [m],
            yerr=[[m - lo], [hi - m]],
            fmt="o",
            color=color,
            capsize=4,
            ms=9,
            lw=2,
            zorder=3,
            markeredgewidth=1.2,
        )
        ax.annotate(
            f"{m:+.1f}",
            (x, m),
            xytext=(14 if x == 0 else -14, 6),
            textcoords="offset points",
            color=color,
            fontsize=11,
            fontweight="semibold",
            ha="left" if x == 0 else "right",
        )
    ax.axhline(13.66, color=C_BROAD, ls="--", lw=1.0, alpha=0.6)
    ax.annotate(
        "prior broad-recipe panel mean (+13.66)",
        (0.985, 13.66),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=C_BROAD,
        alpha=0.85,
    )
    ax.axhline(-3.1, color=C_NARROW, ls="--", lw=1.0, alpha=0.6)
    ax.annotate(
        "prior 4-negative-recipe anchor (−3.1)",
        (0.985, -3.1),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=C_NARROW,
        alpha=0.85,
    )
    ax.axhline(0, color=C_NEUTRAL, lw=0.8, alpha=0.5)
    ax.set_xticks(
        [0, 1], ["broad panel\n(15 suppression contexts)", "narrow panel\n(4 suppression contexts)"]
    )
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("end-token logit change, trained − base\n(never-mentioned personas)")
    ax.set_title(
        "Each line = one of 32 never-mentioned personas, scored under both arms",
        fontsize=11,
        loc="left",
        pad=12,
    )
    savefig_paper(fig, "hero_breadth_paired", dir=FIG_DIR)
    plt.close(fig)

    # ── 2. Marker hijack rates (text-level, never-negative personas) ──────
    rates, cap_rates = {}, {}
    for label in LABELS:
        d = json.loads((OUT_DIR / f"raw_completions/raw_completions_{label}.json").read_text())
        n_emit = n_cap = n_tot = 0
        for persona, qd in d["completions"].items():
            if persona not in nn:
                continue
            for row in qd.values():
                n_tot += 1
                if "※" in row["response_text"]:
                    n_emit += 1
                if row["truncated"]:
                    n_cap += 1
        rates[label] = (n_emit, n_tot)
        cap_rates[label] = n_cap / n_tot
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(LABELS))
    for i, label in enumerate(LABELS):
        k, n = rates[label]
        p = k / n
        se = 1.96 * np.sqrt(p * (1 - p) / n)
        color = C_BROAD if label.startswith("broad") else C_NARROW
        ax.bar(i, p, width=0.62, color=color, alpha=0.85)
        ax.errorbar(
            [i],
            [p],
            yerr=[[min(se, p)], [min(se, 1 - p)]],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            lw=1.2,
        )
        ax.annotate(
            f"{p:.0%}",
            (i, p),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="semibold",
        )
    ax.set_xticks(x, [PLAIN[label] for label in LABELS], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of held-out answers\nthat emit the marker")
    ax.set_title(
        "Marker emission on the 32 never-mentioned personas (640 answers per adapter)",
        fontsize=11,
        loc="left",
        pad=12,
    )
    savefig_paper(fig, "marker_hijack_rates", dir=FIG_DIR)
    plt.close(fig)
    print("loop-to-cap rates:", {k: round(v, 3) for k, v in cap_rates.items()})

    # ── 3. Per-adapter clamp + replication anchor ─────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    rng = np.random.default_rng(0)
    ax.axhspan(10.8, 18.9, color=C_BROAD, alpha=0.08, zorder=0)
    for i, label in enumerate(LABELS):
        vals = np.array([per_label[label]["dz_eos"][p] for p in nn])
        color = C_BROAD if label.startswith("broad") else C_NARROW
        ax.bar(i, float(vals.mean()), width=0.62, color=color, alpha=0.55)
        ax.scatter(
            np.full(len(vals), i) + rng.uniform(-0.16, 0.16, len(vals)),
            vals,
            s=12,
            color="#333333",
            alpha=0.55,
            zorder=3,
            linewidths=0,
        )
        ax.annotate(
            f"{vals.mean():+.1f}",
            (i, float(vals.mean())),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="semibold",
        )
    ax.axhline(14.85, color=C_BROAD, ls="--", lw=1.0, alpha=0.7)
    ax.annotate(
        "prior measurement of the same broad recipe (+14.85)",
        (0.99, 14.85),
        xycoords=("axes fraction", "data"),
        xytext=(0, -13),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=C_BROAD,
        alpha=0.9,
    )
    ax.annotate(
        "replication-anchor band [+10.8, +18.9]",
        (0.99, 18.9),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=C_BROAD,
        alpha=0.9,
    )
    ax.axhline(0, color=C_NEUTRAL, lw=0.8, alpha=0.5)
    ax.set_xticks(np.arange(len(LABELS)), [PLAIN[label] for label in LABELS], fontsize=9)
    ax.set_ylabel("end-token logit change, trained − base\n(never-mentioned personas)")
    ax.set_title(
        "Per-adapter clamp: dots = 32 never-mentioned personas, bar = mean",
        fontsize=11,
        loc="left",
        pad=12,
    )
    savefig_paper(fig, "per_adapter_clamp_anchor", dir=FIG_DIR)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
