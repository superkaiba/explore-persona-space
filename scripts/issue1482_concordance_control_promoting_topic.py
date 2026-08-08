"""#1482: what survives once you control for `Logit footprint: promoting` AND
`Content type: topic`?

Both are one-vs-rest INDICATORS, so "controlling for" them means stratifying on
them and only counting pairs INSIDE a stratum -- the same nonparametric
pair-restriction the rest of the concordance battery uses (accumulate C/Dis/T_x
within strata, pool numerator and denominator separately, Mantel-Haenszel style).

Four control sets, each a coarsened exact matching over the named factors:

  A  none                      -- pooled, the reference read
  B  firing rate (10 deciles)  -- the standing control (`matched` in the battery)
  C  promoting x topic         -- ONLY the two the user named (4 cells)
  D  firing x promoting x topic-- the two named ON TOP of firing (40 cells)

D is the read that answers "are there still predictors?", because leaving
activity uncontrolled (C) lets every activity-correlated predictor keep borrowing
the strongest signal in the battery (firing rate, c ~ 0.77).

Retained-pair fraction is reported per control set: pair restriction throws pairs
away, and a c-index computed on a tiny surviving pair pool is not comparable to
one computed on all of them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/scripts")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/matplotlib import

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import issue1482_concordance_writeup_figs as WF  # noqa: E402
from issue1482_concordance_fig import concordance  # noqa: E402

OUT = Path("/home/thomasjiralerspong/explore-persona-space/figures/issue_1482/concordance")
STEM = "writeup_concordance_ctrl_promoting_topic"


def cross(*factors: np.ndarray) -> list[np.ndarray]:
    """Index sets for the non-empty cells of the cross of integer factors."""
    key = np.zeros(len(factors[0]), dtype=np.int64)
    for f in factors:
        u = np.unique(f)
        code = np.searchsorted(u, f)
        key = key * len(u) + code
    return [np.flatnonzero(key == k) for k in np.unique(key)]


def retained_pair_frac(strata: list[np.ndarray], n: int) -> float:
    """Fraction of all n-choose-2 pairs that survive the stratification."""
    tot = n * (n - 1) / 2
    kept = sum(len(s) * (len(s) - 1) / 2 for s in strata)
    return float(kept / tot) if tot else 0.0


def main() -> None:
    b = WF.battery()
    y, vecs = b["y"], b["vecs"]
    n = b["n"]
    ft = b["ctrl"]  # firing_freq_per_token, the standing control

    promoting = vecs.get("Logit footprint: promoting")
    topic = vecs.get("Content type: topic")
    if promoting is None or topic is None:
        raise SystemExit("promoting / topic indicator missing from the battery")

    fdec = np.digitize(ft, np.percentile(ft, np.linspace(0, 100, 11)[1:-1]))
    sets = {
        "A_pooled": [np.arange(n)],
        "B_firing": cross(fdec),
        "C_prom_topic": cross(promoting.astype(np.int64), topic.astype(np.int64)),
        "D_firing_prom_topic": cross(fdec, promoting.astype(np.int64), topic.astype(np.int64)),
    }
    for k, s in sets.items():
        print(f"[ctrl] {k:22s} cells={len(s):4d} retained-pair-frac={retained_pair_frac(s, n):.4f}")

    # the two controls are themselves excluded as candidates (a factor cannot
    # predict within its own stratum -- it is constant there by construction)
    controls = {"Logit footprint: promoting", "Content type: topic"}
    fam = {r["name"]: r["family"] for r in b["rows"]}
    rows = []
    for name, v in vecs.items():
        if name in controls:
            continue
        rec = {"name": name, "family": fam.get(name, "?")}
        for k, s in sets.items():
            rec[k] = concordance(v, y, s)
        rows.append(rec)
    rows.sort(key=lambda r: -abs(r["D_firing_prom_topic"] - 0.5))

    print(f"\n{'predictor':52s} {'pooled':>7s} {'+firing':>8s} {'+p*t':>7s} {'ALL 3':>7s}")
    for r in rows:
        print(
            f"{r['name']:52s} {r['A_pooled']:7.3f} {r['B_firing']:8.3f} "
            f"{r['C_prom_topic']:7.3f} {r['D_firing_prom_topic']:7.3f}"
        )

    survivors = [r for r in rows if abs(r["D_firing_prom_topic"] - 0.5) >= 0.02]
    print(f"\n[ctrl] survivors at |c-0.5| >= 0.02 under ALL THREE controls: {len(survivors)}")

    # ── figure: pooled vs fully-controlled ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10.4, 8.6))
    for r in rows:
        c = WF.FAMILY_COLOR.get(r["family"], "#888888")
        ax.plot(
            [r["A_pooled"], r["D_firing_prom_topic"]],
            [r["name"], r["name"]],
            color=c,
            lw=1.0,
            alpha=0.55,
            zorder=1,
        )
        ax.scatter(
            [r["A_pooled"]], [r["name"]], s=26, facecolor="white", edgecolor=c, lw=1.2, zorder=2
        )
        ax.scatter([r["D_firing_prom_topic"]], [r["name"]], s=42, color=c, zorder=3)
    ax.axvline(0.5, color="0.35", lw=1.0)
    ax.axvspan(0.48, 0.52, color="0.85", alpha=0.55, zorder=0)
    ax.invert_yaxis()
    ax.set_xlabel("concordance (c-index) with per-feature $R^2$  —  0.5 = chance")
    ax.set_title(
        "What survives controlling for firing rate + logit-footprint:promoting + "
        "content-type:topic?\n"
        "hollow = pooled (no control) · filled = all three controls · "
        "grey band = |c − 0.5| < 0.02 (indistinguishable)",
        fontsize=10.4,
    )
    ax.tick_params(axis="y", labelsize=7.4)
    handles = [
        Line2D([], [], marker="o", ls="", color=WF.FAMILY_COLOR[f], label=WF.FAMILY_LABEL[f])
        for f in WF.FAMILY_ORDER
    ]
    ax.legend(handles=handles, fontsize=7.4, frameon=False, loc="lower right")
    fig.text(
        0.5,
        0.012,
        f"n = {n:,} features · retained-pair fraction "
        f"{retained_pair_frac(sets['D_firing_prom_topic'], n):.3f} under all three controls "
        f"({len(sets['D_firing_prom_topic'])} cells). The two control factors are excluded as "
        "candidates: a factor is constant inside its own stratum.",
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    (OUT / f"{STEM}.meta.json").write_text(
        json.dumps(
            {
                "what_is_plotted": (
                    "concordance of per-feature R^2 on each predictor, pooled (hollow) vs "
                    "controlling for firing-rate deciles x promoting-indicator x "
                    "topic-indicator (filled)"
                ),
                "n_rows": n,
                "control_sets": {
                    k: {
                        "cells": len(s),
                        "retained_pair_fraction": retained_pair_frac(s, n),
                    }
                    for k, s in sets.items()
                },
                "excluded_as_candidates": sorted(controls),
                "survivors_at_0.02": [r["name"] for r in survivors],
                "values": rows,
            },
            indent=1,
        )
    )
    print(f"[ctrl] wrote {OUT / (STEM + '.png')}")


if __name__ == "__main__":
    main()
