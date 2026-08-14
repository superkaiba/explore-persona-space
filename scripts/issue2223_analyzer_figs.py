"""Analyzer re-renders + low-level per-unit companion figures for issue #2223.

Renders (all under figures/issue_2223/):
  leg_32b/arm_traj_a0_a1_ci      - A0 vs A1 pooled per-turn mean with conversation-level
                                   bootstrap 95% CI bands (supersedes the driver-rendered
                                   arm_trajectories.png, which had no CI bands and an
                                   overstated "stabilization grid" title).
  leg_32b/arm_traj_a0_a1_perconv - per-conversation trajectories behind that aggregate.
  drift_hero_perconv             - 7B per-conversation trajectories by domain (raw
                                   companion to drift_hero.png).
  leg_32b/drift_hero_perconv     - 32B same.

Inputs: eval_results/issue_2223/phaseA_drift_trajectory.json (7B A0),
        eval_results/issue_2223/leg_32b/phaseA_drift_trajectory.json (32B A0),
        eval_results/issue_2223/leg_32b/phaseB_arm_trajectories.json (32B A1).
Aggregation matches the driver figure: per-turn mean over ALL alive conversations,
domains pooled. Bootstrap: resample conversations within (arm, turn), 2000 draws,
seed 42 (vectorized numpy).
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

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results/issue_2223"
FIG = ROOT / "figures/issue_2223"
RNG = np.random.default_rng(42)
N_BOOT = 2000
DOMAINS = [
    "coding assistance",
    "philosophical discussions about AI",
    "therapy-like contexts",
    "writing assistance",
]


def load_arm(path: Path, arm: str) -> dict[str, dict[int, list[tuple[str, float]]]]:
    """domain -> turn -> [(conv_id, response_projection)]."""
    traj = json.loads(path.read_text())["arms"][arm]["trajectory"]
    out: dict[str, dict[int, list[tuple[str, float]]]] = {}
    for dom, turns in traj.items():
        out[dom] = {
            int(t): [(r["conv"], float(r["response"])) for r in rows] for t, rows in turns.items()
        }
    return out


def pooled_series(arm: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """turns, mean, ci_lo, ci_hi -- pooled over domains, conversation-level bootstrap."""
    turns = sorted({t for d in arm.values() for t in d})
    means, lo, hi = [], [], []
    for t in turns:
        vals = np.array([v for d in arm.values() for (_, v) in d.get(t, [])])
        means.append(vals.mean())
        idx = RNG.integers(0, len(vals), size=(N_BOOT, len(vals)))
        boots = vals[idx].mean(axis=1)
        qlo, qhi = np.quantile(boots, [0.025, 0.975])
        lo.append(qlo)
        hi.append(qhi)
    return np.array(turns), np.array(means), np.array(lo), np.array(hi)


def conv_lines(arm: dict) -> dict[str, tuple[list[int], list[float]]]:
    """conv_id -> (turns, values); domain retrievable from conv_id prefix."""
    out: dict[str, dict[int, float]] = {}
    for d in arm.values():
        for t, rows in d.items():
            for cid, v in rows:
                out.setdefault(cid, {})[t] = v
    return {cid: (sorted(m), [m[t] for t in sorted(m)]) for cid, m in out.items()}


def main() -> None:
    set_paper_style("blog")
    a0_32 = load_arm(EV / "leg_32b/phaseA_drift_trajectory.json", "A0__32b")
    a1_32 = load_arm(EV / "leg_32b/phaseB_arm_trajectories.json", "A1__32b")
    a0_7 = load_arm(EV / "phaseA_drift_trajectory.json", "A0__7b")
    c_a0, c_a1 = paper_palette_blog(2)

    # 1. A0 vs A1 with CI bands
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for arm, color, label in [
        (a0_32, c_a0, "Uncapped"),
        (a1_32, c_a1, "Every-token cap"),
    ]:
        t, m, lo, hi = pooled_series(arm)
        ax.plot(t, m, marker="o", markersize=4, color=color, label=label)
        ax.fill_between(t, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Turn position")
    ax.set_ylabel("Assistant-axis projection\n(pooled over domains)")
    ax.set_title("Qwen3-32B drift: uncapped vs every-token cap")
    ax.legend()
    savefig_paper(fig, "arm_traj_a0_a1_ci", dir=FIG / "leg_32b")
    plt.close(fig)

    # 2. per-conversation companion for the A0/A1 aggregate
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, arm, color, label in [
        (axes[0], a0_32, c_a0, "Uncapped"),
        (axes[1], a1_32, c_a1, "Every-token cap"),
    ]:
        for _cid, (ts, vs) in conv_lines(arm).items():
            ax.plot(ts, vs, color=color, alpha=0.08, linewidth=0.7)
        t, m, _, _ = pooled_series(arm)
        ax.plot(t, m, color="black", linewidth=2.0)
        ax.set_title(label)
        ax.set_xlabel("Turn position")
    axes[0].set_ylabel("Assistant-axis projection")
    savefig_paper(fig, "arm_traj_a0_a1_perconv", dir=FIG / "leg_32b")
    plt.close(fig)

    # 3+4. per-conversation companions for the domain heroes
    dom_colors = dict(zip(sorted(DOMAINS), paper_palette_blog(4)))
    for arm, stem, outdir in [
        (a0_7, "drift_hero_perconv", FIG),
        (a0_32, "drift_hero_perconv", FIG / "leg_32b"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True, sharey=True)
        for ax, dom in zip(axes.flat, sorted(DOMAINS)):
            color = dom_colors[dom]
            per_dom = {dom: arm[dom]}
            for _cid, (ts, vs) in conv_lines(per_dom).items():
                ax.plot(ts, vs, color=color, alpha=0.10, linewidth=0.7)
            t, m, _, _ = pooled_series(per_dom)
            ax.plot(t, m, color="black", linewidth=1.8)
            ax.set_title(dom, fontsize=11)
        for ax in axes[1]:
            ax.set_xlabel("Turn position")
        for ax in axes[:, 0]:
            ax.set_ylabel("Assistant-axis projection")
        savefig_paper(fig, stem, dir=outdir)
        plt.close(fig)

    print("done")


if __name__ == "__main__":
    main()
