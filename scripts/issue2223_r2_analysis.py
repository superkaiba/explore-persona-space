"""Round-2 analyzer recomputes for issue #2223 (interp-critique v1 blockers 1-4).

Outputs (all under eval_results/issue_2223/leg_32b/ + figures/issue_2223/leg_32b/):
  cjk_exclusion_recompute.json - (a) validation that this script's re-implementation of
      the driver's phase_aggregate CI machinery (scripts/issue2223_drift.py, single
      np.random.default_rng(42) stream iterated in trajectory dict order) reproduces the
      persisted phaseA_verdict.json aggregate EXACTLY (max abs deviation printed and
      asserted 0.0); (b) the same machinery re-run with the single CJK-intruded uncapped
      row (therapy__p3__t19, assistant turn 9) excluded, plus the verdict lattice under
      exclusion; (c) full-data drift-vs-stable separation margins at turns 8/9/10 under
      bootstrap seeds 42/0/1/7/123 (seed-sensitivity of the turn-9 margin).
  capping_composition.json - composition-controlled capping estimators: pooled endpoint,
      survivor-matched (conversations alive at turns 1 AND 15), per-domain matched-turn
      (last turn with >=10 alive in BOTH arms), and the turn-15 domain mixtures per arm.
  arm_traj_by_domain(.png/.pdf/.meta.json) - per-domain mean trajectories, uncapped vs
      capped, cells with >=10 alive conversations (the driver's MIN_SAMPLES floor).
  arm_traj_by_domain_perconv(...) - per-conversation companion to the aggregate above.

Reuses load_arm/conv_lines from scripts/issue2223_analyzer_figs.py (sibling import;
script mode puts scripts/ on sys.path[0]).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE the first heavy import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from issue2223_analyzer_figs import conv_lines, load_arm  # noqa: E402
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results/issue_2223"
FIG = ROOT / "figures/issue_2223"
MIN_SAMPLES = 10  # the driver's per-(domain, turn) alive floor
LATE = range(8, 16)  # turns 8..15, the verdict's late window
DRIFT = ["therapy-like contexts", "philosophical discussions about AI"]
STABLE = ["coding assistance", "writing assistance"]
INTRUDED = ("therapy-like contexts", "9", "therapy__p3__t19")  # the 1/3,567 CJK row
SHORT = {
    "therapy-like contexts": "therapy",
    "philosophical discussions about AI": "philosophy",
    "coding assistance": "coding",
    "writing assistance": "writing",
}


def aggregate(traj: dict, seed: int = 42, exclude: set | None = None) -> dict:
    """Verbatim re-implementation of phase_aggregate's CI math
    (scripts/issue2223_drift.py L1894-1909): ONE shared rng stream, trajectory dict
    order, 2000 conversation-level resamples per cell, percentile 2.5/97.5,
    cells under MIN_SAMPLES dropped. Returns {domain: {turn_str: cell}}."""
    rng = np.random.default_rng(seed)
    agg: dict = {}
    for domain, turns in traj.items():
        agg[domain] = {}
        for t, rows in turns.items():
            kept = [
                r
                for r in rows
                if r["response"] is not None
                and (exclude is None or (domain, t, r["conv"]) not in exclude)
            ]
            vals = np.array([r["response"] for r in kept], dtype=float)
            n = len(vals)
            if n < MIN_SAMPLES:
                continue
            boots = np.array([rng.choice(vals, size=n, replace=True).mean() for _ in range(2000)])
            agg[domain][t] = {
                "n": n,
                "mean": float(vals.mean()),
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
            }
    return agg


def separation(agg: dict) -> dict:
    """Per all-domain-eligible late position: the binding drift-vs-stable margin
    (min over stable of ci_lo) - (max over drift of ci_hi); positive = disjoint."""
    elig = {d: {int(t) for t in agg[d] if int(t) in LATE} for d in agg}
    common = sorted(set.intersection(*elig.values()))
    out = {}
    for pos in common:
        drift_hi = max(agg[d][str(pos)]["ci_hi"] for d in DRIFT)
        stable_lo = min(agg[s][str(pos)]["ci_lo"] for s in STABLE)
        out[pos] = float(stable_lo - drift_hi)
    return out


def verdict(agg: dict) -> dict:
    """The driver's _reproduction_verdict lattice re-applied (ordering + separated)."""
    elig = {d: {int(t) for t in agg[d] if int(t) in LATE} for d in agg}
    means = {d: float(np.mean([agg[d][str(t)]["mean"] for t in sorted(elig[d])])) for d in agg}
    ordering = max(means[d] for d in DRIFT) < min(means[s] for s in STABLE)
    seps = separation(agg)
    separated = any(m > 0 for m in seps.values())
    disp = (
        "Reproduced"
        if (ordering and separated)
        else ("Weak reproduction" if ordering else "Failed-to-reproduce")
    )
    return {
        "disposition": disp,
        "ordering": ordering,
        "separated": separated,
        "late_means": means,
        "separation_margins_by_position": seps,
    }


def cjk_exclusion() -> None:
    """Blocker-3 recompute: validate exact reproduction, then exclude the intruded row."""
    data = json.loads((EV / "leg_32b/phaseA_drift_trajectory.json").read_text())
    persisted = json.loads((EV / "leg_32b/phaseA_verdict.json").read_text())
    traj = data["arms"]["A0__32b"]["trajectory"]

    agg_full = aggregate(traj)
    max_dev = 0.0
    for d in persisted["aggregate"]:
        for t, cell in persisted["aggregate"][d].items():
            assert agg_full[d][t]["n"] == cell["n"], (d, t)
            for k in ("mean", "ci_lo", "ci_hi"):
                max_dev = max(max_dev, abs(agg_full[d][t][k] - cell[k]))
    assert max_dev == 0.0, f"validation failed: max deviation {max_dev}"

    agg_excl = aggregate(traj, exclude={INTRUDED})
    seeds = {s: separation(aggregate(traj, seed=s)) for s in (42, 0, 1, 7, 123)}

    payload = {
        "validation_max_abs_deviation_vs_persisted": max_dev,
        "excluded_row": {"domain": INTRUDED[0], "turn": int(INTRUDED[1]), "conv": INTRUDED[2]},
        "full_data": {"verdict": verdict(agg_full)},
        "cjk_excluded": {
            "verdict": verdict(agg_excl),
            "therapy_turn9": agg_excl["therapy-like contexts"]["9"],
            "writing_turn9": agg_excl["writing assistance"]["9"],
        },
        "seed_sensitivity_full_data_margins": {
            str(s): {str(p): round(m, 4) for p, m in seps.items()} for s, seps in seeds.items()
        },
    }
    out = EV / "leg_32b/cjk_exclusion_recompute.json"
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"[cjk_exclusion] validation max_dev={max_dev}; "
        f"excluded verdict={payload['cjk_excluded']['verdict']['disposition']} -> {out}"
    )


def composition() -> None:
    """Blocker-4 recompute: pooled / survivor-matched / per-domain matched-turn
    capping estimators + turn-15 domain mixtures."""
    a0 = load_arm(EV / "leg_32b/phaseA_drift_trajectory.json", "A0__32b")
    a1 = load_arm(EV / "leg_32b/phaseB_arm_trajectories.json", "A1__32b")

    def rows_at(arm: dict, t: int) -> list[tuple[str, str, float]]:
        return [(d, c, v) for d in arm for (c, v) in arm[d].get(t, [])]

    def pooled(arm: dict) -> dict:
        r1, r15 = rows_at(arm, 1), rows_at(arm, 15)
        return {
            "turn1_mean": float(np.mean([v for _, _, v in r1])),
            "turn15_mean": float(np.mean([v for _, _, v in r15])),
            "n_turn1": len(r1),
            "n_turn15": len(r15),
            "drift": float(np.mean([v for _, _, v in r15]) - np.mean([v for _, _, v in r1])),
            "turn15_mix": {SHORT[d]: sum(1 for dd, _, _ in r15 if dd == d) for d in arm},
        }

    def survivor(arm: dict) -> dict:
        v1 = {c: v for _, c, v in rows_at(arm, 1)}
        v15 = {c: v for _, c, v in rows_at(arm, 15)}
        common = sorted(set(v1) & set(v15))
        return {"drift": float(np.mean([v15[c] - v1[c] for c in common])), "n": len(common)}

    p0, p1 = pooled(a0), pooled(a1)
    s0, s1 = survivor(a0), survivor(a1)
    per_dom = {}
    for d in a0:
        both = [
            t
            for t in range(1, 16)
            if len(a0[d].get(t, [])) >= MIN_SAMPLES and len(a1[d].get(t, [])) >= MIN_SAMPLES
        ]
        tmax = max(both)

        def dmean(arm: dict, t: int) -> float:
            return float(np.mean([v for _, v in arm[d][t]]))

        dr0 = dmean(a0, tmax) - dmean(a0, 1)
        dr1 = dmean(a1, tmax) - dmean(a1, 1)
        per_dom[SHORT[d]] = {
            "matched_turn": tmax,
            "uncapped_drift": dr0,
            "capped_drift": dr1,
            "n_uncapped": len(a0[d][tmax]),
            "n_capped": len(a1[d][tmax]),
            "reduction_pct": float(100 * (1 - dr1 / dr0)),
        }

    payload = {
        "pooled": {
            "uncapped": p0,
            "capped": p1,
            "reduction_pct": float(100 * (1 - p1["drift"] / p0["drift"])),
        },
        "survivor_matched": {
            "uncapped": s0,
            "capped": s1,
            "reduction_pct": float(100 * (1 - s1["drift"] / s0["drift"])),
        },
        "per_domain_matched_turn": per_dom,
    }
    out = EV / "leg_32b/capping_composition.json"
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"[composition] pooled {payload['pooled']['reduction_pct']:.1f}% / "
        f"survivor {payload['survivor_matched']['reduction_pct']:.1f}% / per-domain "
        f"{ {k: round(v['reduction_pct']) for k, v in per_dom.items()} } -> {out}"
    )


def figures() -> None:
    """Per-domain uncapped-vs-capped trajectories (aggregate + per-conversation)."""
    set_paper_style("blog")
    a0 = load_arm(EV / "leg_32b/phaseA_drift_trajectory.json", "A0__32b")
    a1 = load_arm(EV / "leg_32b/phaseB_arm_trajectories.json", "A1__32b")
    c_a0, c_a1 = paper_palette_blog(2)  # same arm colors as arm_traj_a0_a1_ci
    domains = sorted(a0)

    def dom_series(arm: dict, d: str) -> tuple[list[int], list[float]]:
        turns = sorted(t for t in arm[d] if len(arm[d][t]) >= MIN_SAMPLES)
        return turns, [float(np.mean([v for _, v in arm[d][t]])) for t in turns]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True, sharey=True)
    for ax, d in zip(axes.flat, domains):
        for arm, color, label in [(a0, c_a0, "Uncapped"), (a1, c_a1, "Every-token cap")]:
            t, m = dom_series(arm, d)
            ax.plot(t, m, marker="o", markersize=3.5, color=color, label=label)
        ax.set_title(d, fontsize=11)
    axes[0, 0].legend(fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Turn position")
    for ax in axes[:, 0]:
        ax.set_ylabel("Assistant-axis projection")
    savefig_paper(fig, "arm_traj_by_domain", dir=FIG / "leg_32b")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True, sharey=True)
    for ax, d in zip(axes.flat, domains):
        for arm, color in [(a0, c_a0), (a1, c_a1)]:
            for _cid, (ts, vs) in conv_lines({d: arm[d]}).items():
                ax.plot(ts, vs, color=color, alpha=0.07, linewidth=0.6)
            t, m = dom_series(arm, d)
            ax.plot(t, m, color=color, linewidth=2.0)
        ax.set_title(d, fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("Turn position")
    for ax in axes[:, 0]:
        ax.set_ylabel("Assistant-axis projection")
    savefig_paper(fig, "arm_traj_by_domain_perconv", dir=FIG / "leg_32b")
    plt.close(fig)
    print("[figures] arm_traj_by_domain + arm_traj_by_domain_perconv written")


if __name__ == "__main__":
    cjk_exclusion()
    composition()
    figures()
