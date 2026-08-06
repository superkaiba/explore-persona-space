"""#1482: forward-stepwise concordance — control for every previously-selected
predictor, then re-rank the rest.

ROUND 0 conditions on nothing. Each later round adds the WINNER of the previous
round (max |c - 0.5|) to the control set, re-strata­fies, and re-scores every
remaining candidate. A predictor that stays high after k rounds carries signal
none of the k selected predictors already carry.

CONDITIONING is coarsened exact matching: cross the control variables' quantile
bins into cells, accumulate C/Dis/T_x within cells only, pool numerator and
denominator (Mantel-Haenszel). Pairs spanning cells are never formed, so nothing
is assumed about the functional form of any control -> R^2 relation.

THE COST, stated up front: crossed cells grow as bins^k, so a fixed 10 bins per
variable would leave 10^4 = 10,000 cells of ~12 features by round 4 -- every pair
dropped, every estimate noise. Bins per variable are therefore COARSENED as the
control set grows (bins = clip(round(TARGET_CELLS ** (1/k)), 2, 10)) to hold cells
near TARGET_CELLS. Two consequences that must be read with the figures:

  * Later rounds condition MORE COARSELY per variable. A predictor surviving
    round 4 has survived 4 controls at 4 bins each, NOT at decile resolution --
    residual within-cell confounding grows with k.
  * The retained-pair fraction falls every round. It is printed per round, on
    each figure, and in the sidecar; when it collapses the estimates are noise
    and the series stops (STOP_MIN_PAIR_FRAC).

The series also stops when the best remaining |c - 0.5| falls below STOP_EFFECT
(0.02 -- the resolution below which, with no confidence intervals anywhere in
this analysis, a difference should not be read).

MATRYOSHKA TIER IS EXCLUDED from this series, unlike the other #1482 concordance
figures. It is scored on a different dictionary (layer-20 SAELens matryoshka,
16,384 features) whose features are not the layer-19 battery's features, so it
has no value to stratify the battery ON and cannot legitimately enter a stepwise
selection over the battery. `dense_latent_flag` is also excluded (a threshold on
`activity`, already present continuously).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/scripts")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/scipy/matplotlib import

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import issue1482_concordance_writeup_figs as WF
from issue1482_concordance_fig import concordance

OUT = WF.OUT
MAX_ROUNDS = 14  # raised from 6 (2026-08-04): 6 was a bare cap, not a statistical
# stop -- the series was still finding effects when it hit it. The real terminators
# are STOP_EFFECT and STOP_MIN_PAIR_FRAC below; let those decide where it ends.
TARGET_CELLS = 400  # ~300 features/cell at n = 120,716
MAX_BINS = 10
MIN_BINS = 2
STOP_EFFECT = 0.02  # below the readable-difference floor; no CIs anywhere here
STOP_MIN_PAIR_FRAC = 0.005  # cells too small to estimate from
DECLUSTER_RHO = 0.90  # a selected control retires its |rho| >= 0.90 siblings


def crossed_strata(controls: list[np.ndarray], n: int) -> tuple[list[np.ndarray], int]:
    """Coarsened-exact-matching cells over the control set. Returns (cells, bins).

    The bin budget is spent only on the CONTINUOUS controls. A binary/low-cardinality
    control (a judged indicator) costs exactly its own number of levels, so charging
    it a full bins^(1/k) share needlessly coarsens the continuous ones -- with three
    indicators and two continuous controls that mistake drove the continuous bins to
    3 when 7 fit inside the same cell budget.
    """
    if not controls:
        return [np.arange(n)], 0
    uniqs = [np.unique(v) for v in controls]
    fixed_width = 1
    n_cont = 0
    for u in uniqs:
        if len(u) <= MAX_BINS:
            fixed_width *= len(u)
        else:
            n_cont += 1
    if n_cont == 0:
        bins = 0
    else:
        budget = max(1.0, TARGET_CELLS / max(1, fixed_width))
        bins = int(np.clip(round(budget ** (1.0 / n_cont)), MIN_BINS, MAX_BINS))

    codes = np.zeros(n, dtype=np.int64)
    for v, u in zip(controls, uniqs):
        if len(u) <= MAX_BINS:
            b = np.searchsorted(u, v)
            width = len(u)
        else:
            e = np.percentile(v, np.linspace(0, 100, bins + 1))
            b = np.clip(np.digitize(v, e[1:-1]), 0, bins - 1)
            width = bins
        codes = codes * width + b
    return [np.flatnonzero(codes == u) for u in np.unique(codes)], bins


def pair_fraction(cells: list[np.ndarray], n: int) -> float:
    tot = n * (n - 1) / 2
    return float(sum(len(c) * (len(c) - 1) / 2 for c in cells) / tot)


def render_round(rnd: dict, stem: Path) -> None:
    """One round's lollipop: every remaining candidate, strongest first."""
    order = sorted(rnd["scores"], key=lambda r: abs(r["c"] - 0.5))
    fig, ax = plt.subplots(figsize=(10.4, 0.30 * (len(order) + 2) + 2.9))
    ax.axvline(0.5, color="#888888", lw=1.4, zorder=1)
    for i, r in enumerate(order):
        col = WF.FAMILY_COLOR[r["family"]]
        ax.plot([0.5, r["c"]], [i, i], color=col, lw=1.6, alpha=0.45, zorder=2)
        ax.scatter(r["c"], i, s=46, color=col, zorder=3)
        ax.annotate(
            f"{r['c']:.2f}",
            (r["c"], i),
            textcoords="offset points",
            xytext=(9 if r["c"] >= 0.5 else -9, 0),
            ha="left" if r["c"] >= 0.5 else "right",
            va="center",
            fontsize=6.8,
            color="#333333",
        )
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([r["name"] for r in order], fontsize=7.8)
    for tick, r in zip(ax.get_yticklabels(), order):
        tick.set_color(WF.FAMILY_COLOR[r["family"]])
    ax.set_ylim(-0.8, len(order) - 0.3)
    ax.set_xlabel("concordance (c-index) with per-feature $R^2$  —  0.5 = chance", fontsize=9.6)
    ax.tick_params(axis="x", labelsize=8.4)
    ax.grid(axis="x", alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)

    controls = rnd["controls"]
    if controls:
        ctrl_txt = "controlling for: " + " + ".join(controls)
        detail = (
            f"{rnd['n_cells']:,} cells at {rnd['bins']} bins/variable · "
            f"median cell {rnd['median_cell']:,} features · "
            f"{rnd['pair_frac']:.1%} of pairs retained"
        )
    else:
        ctrl_txt = "no controls — every pair counts"
        detail = f"n = {rnd['n']:,} features"
    ax.set_title(
        f"Round {rnd['round']} — {ctrl_txt}\n{detail} · winner: "
        f"{rnd['winner']} ({rnd['winner_c']:.3f})",
        fontsize=11.0,
    )
    ax.legend(handles=WF.family_legend(), fontsize=7.4, loc="lower right", framealpha=0.94)
    fig.text(
        0.5,
        -0.030 - 0.0007 * len(order),
        "Forward stepwise: each round adds the previous round's winner to the control set and "
        "re-scores the rest.\nc = P(the feature with the higher predictor value is the "
        "better-predicted one), over pairs the predictor separates; for a\nbinary predictor this "
        "is the ordinary AUROC. A lollipop LEFT of 0.5 is inverted. Bins per control variable are "
        "COARSENED as\nthe control set grows, so later rounds condition less finely per variable "
        "and residual confounding grows with round number.\nNo confidence intervals — differences "
        "under ~0.02 should not be read.",
        ha="center",
        fontsize=7.4,
        color="#5A5A5A",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_trajectory(rounds: list[dict], stem: Path) -> None:
    """Every predictor's c across the rounds it survived; selected ones highlighted."""
    names = sorted({r["name"] for rnd in rounds for r in rnd["scores"]})
    fig, ax = plt.subplots(figsize=(11.4, 7.6))
    ax.axhline(0.5, color="#888888", lw=1.4, zorder=1)
    selected = [rnd["winner"] for rnd in rounds if rnd["winner"]]
    for nm in names:
        xs, ys, fam = [], [], None
        for rnd in rounds:
            hit = next((r for r in rnd["scores"] if r["name"] == nm), None)
            if hit is None:
                break  # became a control; its line ends where it was selected
            xs.append(rnd["round"])
            ys.append(hit["c"])
            fam = hit["family"]
        if not xs:
            continue
        chosen = nm in selected
        ax.plot(
            xs,
            ys,
            color=WF.FAMILY_COLOR[fam],
            lw=2.2 if chosen else 0.9,
            alpha=0.95 if chosen else 0.32,
            marker="o" if chosen else None,
            ms=5,
            zorder=4 if chosen else 2,
        )
        if chosen:
            # Consecutive winners can end within ~0.002 of each other (round 0 and
            # round 1 did: 0.768 vs 0.766), so a fixed y-offset overlaps the two
            # labels illegibly. Alternate the offset by round parity to separate
            # them, and nudge x so the leader-free labels do not collide either.
            dy = 11 if xs[-1] % 2 == 0 else -11
            ax.annotate(
                f"{nm}  ⟵ R{xs[-1]}",
                (xs[-1], ys[-1]),
                textcoords="offset points",
                xytext=(9, dy),
                va="center",
                fontsize=7.4,
                color=WF.FAMILY_COLOR[fam],
            )
    ax.set_xticks([rnd["round"] for rnd in rounds])
    ax.set_xlabel("round (each adds the previous winner to the control set)", fontsize=9.6)
    ax.set_ylabel("concordance (c-index) with per-feature $R^2$", fontsize=9.6)
    ax.set_xlim(-0.25, max(rnd["round"] for rnd in rounds) + 1.9)
    ax.grid(alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "How each predictor's concordance decays as controls accumulate\n"
        "bold = the predictor selected that round (its line ends there — it becomes a control)",
        fontsize=11.6,
    )
    ax.legend(handles=WF.family_legend(), fontsize=7.4, loc="lower left", framealpha=0.94)
    fig.text(
        0.5,
        -0.055,
        "Bins per control variable are COARSENED as the control set grows "
        "(bins = clip(round(400 ** (1/k)), 2, 10)) so cells stay populated;\nlater rounds "
        "therefore condition LESS finely per variable and residual within-cell confounding grows "
        "with round number. The retained-pair\nfraction is printed on each round's own figure. No "
        "confidence intervals — differences under ~0.02 should not be read.",
        ha="center",
        fontsize=7.4,
        color="#5A5A5A",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    b = WF.battery()  # dense_latent_flag already excluded (WF.DERIVED_BIN is empty)
    y, n, vecs = b["y"], b["n"], b["vecs"]
    fam = {r["name"]: r["family"] for r in b["rows"]}
    candidates = [r["name"] for r in b["rows"]]

    controls: list[str] = []
    rounds: list[dict] = []
    for k in range(MAX_ROUNDS):
        cells, bins = crossed_strata([vecs[c] for c in controls], n)
        frac = pair_fraction(cells, n)
        sizes = [len(c) for c in cells]
        scores = [
            {"name": nm, "family": fam[nm], "c": concordance(vecs[nm], y, cells)}
            for nm in candidates
        ]
        scores = [s for s in scores if np.isfinite(s["c"])]
        if not scores:
            print(f"round {k}: no estimable candidate remains — stopping")
            break
        winner = max(scores, key=lambda s: abs(s["c"] - 0.5))
        rnd = {
            "round": k,
            "controls": list(controls),
            "bins": bins,
            "n": n,
            "n_cells": len(cells),
            "median_cell": int(np.median(sizes)),
            "min_cell": int(min(sizes)),
            "pair_frac": frac,
            "scores": scores,
            "winner": winner["name"],
            "winner_c": winner["c"],
            "retired_as_siblings": [],
        }
        rounds.append(rnd)
        print(
            f"round {k}: cells={len(cells):>5,} bins={bins} median_cell={int(np.median(sizes)):>6,} "
            f"pairs={frac:6.2%}  winner={winner['name']} ({winner['c']:.3f})"
        )
        render_round(rnd, OUT / f"writeup_stepwise_r{k}")

        if abs(winner["c"] - 0.5) < STOP_EFFECT:
            print(f"  stop: best remaining |c-0.5| = {abs(winner['c'] - 0.5):.3f} < {STOP_EFFECT}")
            rnd["winner"] = ""
            break
        if frac < STOP_MIN_PAIR_FRAC:
            print(f"  stop: retained-pair fraction {frac:.3%} < {STOP_MIN_PAIR_FRAC:.1%}")
            rnd["winner"] = ""
            break
        # Retire the winner's near-duplicates. Decile stratification does NOT fully
        # control a variable -- residual within-stratum variation remains -- so a
        # rho ~ 0.98 sibling re-emerges with real signal and the series spends a
        # round re-selecting the SAME construct (measured: firing/token won round 4
        # at 0.597 with mean-activation already a control, rho = 0.977). Retiring at
        # the same 0.90 cut the #1482 Shapley declustering uses keeps each round on a
        # genuinely new construct. Retired names are recorded, never silently dropped.
        wv = vecs[winner["name"]]
        retired = [
            c
            for c in candidates
            if c != winner["name"] and abs(spearmanr(wv, vecs[c]).statistic) >= DECLUSTER_RHO
        ]
        rnd["retired_as_siblings"] = [
            {"name": c, "rho": float(spearmanr(wv, vecs[c]).statistic)} for c in retired
        ]
        if retired:
            print(f"  retired as |rho| >= {DECLUSTER_RHO} siblings of the winner: {retired}")
        controls.append(winner["name"])
        candidates = [c for c in candidates if c != winner["name"] and c not in retired]

    render_trajectory(rounds, OUT / "writeup_stepwise_trajectory")
    (OUT / "writeup_stepwise.meta.json").write_text(
        json.dumps(
            {
                "what_is_plotted": (
                    "Forward-stepwise concordance. Round 0 conditions on nothing; each later "
                    "round adds the previous round's winner (max |c - 0.5|) to the control set, "
                    "re-stratifies by coarsened exact matching over the control set, and "
                    "re-scores every remaining candidate."
                ),
                "statistic": "D_{y|x} = (C - Dis) / (n_pairs - T_x);  c = (D + 1) / 2",
                "n_rows": int(n),
                "selection_rule": (
                    "max |c - 0.5| among remaining candidates; the winner then retires every "
                    f"candidate with |Spearman| >= {DECLUSTER_RHO} to it (the same cut the #1482 "
                    "Shapley block declustering uses), because decile stratification leaves "
                    "residual within-stratum variation and a near-duplicate would otherwise "
                    "re-win on the SAME construct"
                ),
                "binning_rule": (
                    f"bins per control = clip(round({TARGET_CELLS} ** (1/k)), {MIN_BINS}, "
                    f"{MAX_BINS}); a control with <= bins distinct values uses its own levels"
                ),
                "stop_rules": {
                    "effect": f"best remaining |c - 0.5| < {STOP_EFFECT}",
                    "pairs": f"retained-pair fraction < {STOP_MIN_PAIR_FRAC}",
                    "max_rounds": MAX_ROUNDS,
                },
                "excluded": [
                    "matryoshka tier — a different dictionary/panel, so it has no value to "
                    "stratify the layer-19 battery ON and cannot enter a stepwise selection "
                    "over it",
                    "dense_latent_flag — a threshold on `activity`, already present continuously",
                ],
                "rounds": rounds,
                "caveats": [
                    "Bins per control are COARSENED as the control set grows, so later rounds "
                    "condition LESS finely per variable; residual within-cell confounding grows "
                    "with round number.",
                    "The retained-pair fraction falls every round (printed per round); a low "
                    "fraction means the estimate rests on few pairs.",
                    "Near-duplicates (|rho| >= 0.90) of a selected control are retired "
                    "that round and listed in `retired_as_siblings`; without this the series "
                    "re-selects the activity construct repeatedly.",
                    "Forward stepwise is greedy and selection-unstable under collinearity — with "
                    "pairwise correlations up to 0.85 here, a near-tie at any round can reorder "
                    "everything downstream. Read the SET that survives, not the exact order.",
                    "No confidence intervals anywhere; differences under ~0.02 are not readable, "
                    "which is why the series stops at that effect size.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nselected in order: {' -> '.join(r['winner'] for r in rounds if r['winner'])}")
    print(f"wrote {len(rounds)} round figures + trajectory to {OUT}")


if __name__ == "__main__":
    main()
