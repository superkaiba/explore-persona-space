"""Figures for the #533 bare-word (content-matched) install-step grid follow-up.

Reads:
  * 240 cross-eval per-cell JSONs at
    ``eval_results/issue_533/bare_word_install_step_grid/cross_eval/per_cell/``
  * 245 logit-capture per-cell JSONs at
    ``eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell/``

and writes two figures into ``figures/issue_533/`` :

1. ``bw_paired_gap_trajectory.{png,pdf,meta.json}`` — HERO. Two panels
   (pirate / villain); each panel plots the paired wrong-persona role-vs-
   system gap d = (system_minimal arm) − (role_bare arm) at the wrong-
   persona probe across steps {18, 30, 60, 120}, in two DVs side-by-side
   per panel:
     - log P trained − base  (the parent's behavioral DV)
     - EOS-margin trained − base  (the parent's mechanistic DV)
   95% bootstrap CI bands per DV; install gate (own argmax-emit ≥ 0.5 in
   BOTH arms) marks the pre-install region.

2. ``bw_install_levels.{png,pdf,meta.json}`` — supporting. Two panels
   (one per persona); each shows trained-vs-base Δ log P at the wrong-
   persona probe per arm across steps, alongside a ghost line showing
   the parent's elaborate-wording install-step value at the equivalent
   step (s=30 for the parent's #547 grid). The same plot also shows the
   own-encoding argmax-emit rate as a thin bottom strip so the install
   cliff is visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

CROSS_EVAL_PER_CELL = (
    REPO_ROOT
    / "eval_results"
    / "issue_533"
    / "bare_word_install_step_grid"
    / "cross_eval"
    / "per_cell"
)
LOGIT_PER_CELL = (
    REPO_ROOT
    / "eval_results"
    / "issue_533"
    / "bare_word_install_step_grid"
    / "logit_capture"
    / "per_cell"
)
FIG_DIR = REPO_ROOT / "figures" / "issue_533"
ANALYSIS_PATH = (
    REPO_ROOT / "eval_results" / "issue_533" / "bare_word_install_step_grid" / "analysis.json"
)

STEPS = [18, 30, 60, 120]
SEEDS = [7, 21, 42, 137, 1337]
PERSONAS = ["pirate", "villain"]
ARMS = ["system_minimal", "role_bare"]
WRONG_OF = {"pirate": "villain", "villain": "pirate"}
N_BOOT = 10_000


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _cell_label(arm: str, seed: int, persona: str, steps: int) -> str:
    # bw_i533 suffix-char is 's' per GRID_SUFFIX_CHAR_FOR
    return f"{arm}_seed{seed}_cn_{persona}_s{steps}"


def _load_crosseval(arm: str, seed: int, persona: str, steps: int, probe: str) -> dict:
    p = CROSS_EVAL_PER_CELL / f"{_cell_label(arm, seed, persona, steps)}__{probe}.json"
    return json.loads(p.read_text())


def _load_trained_logit(arm: str, seed: int, persona: str, steps: int, probe: str) -> dict:
    name = f"{_cell_label(arm, seed, persona, steps)}__{probe}__marker_pirate.json"
    return json.loads((LOGIT_PER_CELL / name).read_text())


def _load_base_logit(probe: str) -> dict:
    return json.loads((LOGIT_PER_CELL / f"base__{probe}__marker_pirate.json").read_text())


# ---------------------------------------------------------------------------
# Per-question DV builders
# ---------------------------------------------------------------------------


def _per_q_delta_logp(arm: str, seed: int, persona: str, steps: int, probe: str) -> np.ndarray:
    """Δ log P(marker) trained − base, per question."""
    row = _load_crosseval(arm, seed, persona, steps, probe)
    g = np.asarray(row["g_logps_per_q"], dtype=float)
    b = np.asarray(row["b_logps_per_q"], dtype=float)
    return g - b


def _per_q_delta_margin(
    arm: str, seed: int, persona: str, steps: int, probe: str, base_cache: dict
) -> np.ndarray:
    """Δ(z_marker − z_eos) trained − base, per question."""
    trained = _load_trained_logit(arm, seed, persona, steps, probe)
    base = base_cache.setdefault(probe, _load_base_logit(probe))
    g_zm = np.asarray(trained["trained"]["z_marker"], dtype=float)
    g_ze = np.asarray(trained["trained"]["z_eos"], dtype=float)
    b_zm = np.asarray(base["stats"]["z_marker"], dtype=float)
    b_ze = np.asarray(base["stats"]["z_eos"], dtype=float)
    return (g_zm - g_ze) - (b_zm - b_ze)


def _own_emit_rate(arm: str, seed: int, persona: str, steps: int) -> float:
    """Own-encoding argmax-emit rate."""
    own_e = f"{arm}_{persona}"
    row = _load_crosseval(arm, seed, persona, steps, own_e)
    flags = row.get("g_argmax_marker_per_q", [])
    return float(sum(flags) / len(flags)) if flags else float("nan")


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _bootstrap_pair(
    per_seed_a: dict[int, np.ndarray],
    per_seed_b: dict[int, np.ndarray],
    rng_seed: int = 42,
) -> tuple[float, float, float, list[float]]:
    """Per-seed-paired bootstrap of (a − b)."""
    shared = sorted(set(per_seed_a) & set(per_seed_b))
    if not shared:
        return float("nan"), float("nan"), float("nan"), []
    a = np.array([per_seed_a[s].mean() for s in shared])
    b = np.array([per_seed_b[s].mean() for s in shared])
    diff = a - b
    point = float(diff.mean())
    rng = np.random.default_rng(rng_seed)
    n = len(shared)
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(diff[idx].mean())
    return (
        point,
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
        diff.tolist(),
    )


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def aggregate() -> dict:
    """Build the dict of paired bootstrap results, in both DV spaces."""
    base_cache: dict[str, dict] = {}
    # Per (arm, persona, steps): {seed: per-question Δ}
    wrong_dlp: dict[tuple, dict[int, np.ndarray]] = {}
    wrong_dmg: dict[tuple, dict[int, np.ndarray]] = {}
    own_emit: dict[tuple, dict[int, float]] = {}
    arm_dlp: dict[tuple, dict[int, float]] = {}  # mean Δlog P at wrong-persona probe per arm

    for arm in ARMS:
        for persona in PERSONAS:
            for s in STEPS:
                wrong_e = f"{arm}_{WRONG_OF[persona]}"
                key = (arm, persona, s)
                wrong_dlp[key] = {}
                wrong_dmg[key] = {}
                own_emit[key] = {}
                arm_dlp[key] = {}
                for seed in SEEDS:
                    wrong_dlp[key][seed] = _per_q_delta_logp(arm, seed, persona, s, wrong_e)
                    wrong_dmg[key][seed] = _per_q_delta_margin(
                        arm, seed, persona, s, wrong_e, base_cache
                    )
                    own_emit[key][seed] = _own_emit_rate(arm, seed, persona, s)
                    arm_dlp[key][seed] = float(wrong_dlp[key][seed].mean())

    # Paired bootstrap: d = sys − role at each (persona, s)
    boot_dlp: dict[tuple, tuple] = {}
    boot_dmg: dict[tuple, tuple] = {}
    for persona in PERSONAS:
        for s in STEPS:
            sys_dlp = wrong_dlp[("system_minimal", persona, s)]
            role_dlp = wrong_dlp[("role_bare", persona, s)]
            sys_dmg = wrong_dmg[("system_minimal", persona, s)]
            role_dmg = wrong_dmg[("role_bare", persona, s)]
            boot_dlp[(persona, s)] = _bootstrap_pair(sys_dlp, role_dlp)
            boot_dmg[(persona, s)] = _bootstrap_pair(sys_dmg, role_dmg)

    return {
        "boot_dlp": boot_dlp,
        "boot_dmg": boot_dmg,
        "wrong_dlp_per_arm": arm_dlp,
        "own_emit": own_emit,
    }


# ---------------------------------------------------------------------------
# Hero: paired gap trajectory, two DVs side by side
# ---------------------------------------------------------------------------


def fig_paired_gap_trajectory(agg: dict) -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=False)

    col_logp = paper_palette_role("baseline")  # warm
    col_margin = paper_palette_role("primary")  # blue
    install_grey = "#B5BAC2"

    for ax, persona in zip(axes, PERSONAS):
        steps_x = np.array(STEPS, dtype=float)
        # log P trace
        pts_lp = np.array([agg["boot_dlp"][(persona, s)][0] for s in STEPS])
        lo_lp = np.array([agg["boot_dlp"][(persona, s)][1] for s in STEPS])
        hi_lp = np.array([agg["boot_dlp"][(persona, s)][2] for s in STEPS])
        # margin trace
        pts_mg = np.array([agg["boot_dmg"][(persona, s)][0] for s in STEPS])
        lo_mg = np.array([agg["boot_dmg"][(persona, s)][1] for s in STEPS])
        hi_mg = np.array([agg["boot_dmg"][(persona, s)][2] for s in STEPS])

        # pre-install grey region (s < 30 — gate fails on s=18 only here)
        ax.axvspan(15, 22, color=install_grey, alpha=0.18, zorder=0)
        ax.axhline(0.0, color="#888", linewidth=0.7, linestyle="-", zorder=1)

        # log P (warm)
        ax.fill_between(steps_x, lo_lp, hi_lp, color=col_logp, alpha=0.18, zorder=2)
        ax.plot(
            steps_x,
            pts_lp,
            color=col_logp,
            marker="o",
            markersize=6,
            linewidth=2.0,
            label="trained − base log P",
            zorder=3,
        )
        # margin (blue)
        ax.fill_between(steps_x, lo_mg, hi_mg, color=col_margin, alpha=0.18, zorder=2)
        ax.plot(
            steps_x,
            pts_mg,
            color=col_margin,
            marker="s",
            markersize=6,
            linewidth=2.0,
            label="trained − base EOS-margin",
            zorder=3,
        )

        ax.set_xscale("log")
        ax.set_xlim(16, 140)
        ax.set_xticks(STEPS)
        ax.set_xticklabels([str(s) for s in STEPS])
        ax.minorticks_off()
        ax.set_xlabel("Optimizer steps (log scale)")
        if persona == "pirate":
            ax.set_ylabel("Paired gap d = (minimal system) − (bare role), nats")
        # install-not-installed label at bottom of pre-install band
        y_lo, y_hi = ax.get_ylim()
        ax.text(
            19.7,
            y_lo + 0.05 * (y_hi - y_lo),
            "implant\nnot installed",
            ha="center",
            va="bottom",
            color="#555",
            fontsize=8,
            zorder=4,
        )

        title = f"Trained on {persona} — wrong-persona ({WRONG_OF[persona]}) probe"
        subtitle = (
            "Positive d = bare role carries LESS wrong-persona marker mass than minimal system"
        )
        set_title_subtitle(ax, title, subtitle)

    axes[1].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    savefig_paper(fig, "issue_533/bw_paired_gap_trajectory", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supporting: per-arm Δlog P levels + own-emit strip
# ---------------------------------------------------------------------------


def fig_install_levels(agg: dict) -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    col_sys = paper_palette_role("baseline")
    col_role = paper_palette_role("primary")
    install_grey = "#B5BAC2"

    for col, persona in enumerate(PERSONAS):
        # Top row: Δlog P (per-arm) at wrong-persona probe
        ax = axes[0, col]
        steps_x = np.array(STEPS, dtype=float)
        for arm, color, marker, label in [
            ("system_minimal", col_sys, "o", "minimal system"),
            ("role_bare", col_role, "s", "bare role"),
        ]:
            per_seed = agg["wrong_dlp_per_arm"][(arm, persona, STEPS[0])]
            means = []
            err_lo = []
            err_hi = []
            for s in STEPS:
                per_seed = list(agg["wrong_dlp_per_arm"][(arm, persona, s)].values())
                arr = np.array(per_seed)
                means.append(arr.mean())
                # CI via per-seed bootstrap of the single arm's mean
                rng = np.random.default_rng(42)
                boots = np.array(
                    [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(2000)]
                )
                err_lo.append(np.percentile(boots, 2.5))
                err_hi.append(np.percentile(boots, 97.5))
            means = np.array(means)
            err_lo = np.array(err_lo)
            err_hi = np.array(err_hi)
            ax.fill_between(steps_x, err_lo, err_hi, color=color, alpha=0.18)
            ax.plot(
                steps_x,
                means,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=2.0,
                label=label,
            )

        ax.axvspan(15, 22, color=install_grey, alpha=0.18, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(16, 140)
        ax.set_xticks(STEPS)
        ax.set_xticklabels([str(s) for s in STEPS])
        ax.minorticks_off()
        if col == 0:
            ax.set_ylabel("Trained − base log P (nats) at\nwrong-persona probe")
        ax.legend(loc="upper right", frameon=False)
        ax.set_title(
            f"Trained on {persona} — wrong-persona ({WRONG_OF[persona]}) probe",
            loc="left",
            pad=8,
            fontweight="semibold",
        )

        # Bottom row: own-encoding emit rate
        axe = axes[1, col]
        for arm, color, marker in [
            ("system_minimal", col_sys, "o"),
            ("role_bare", col_role, "s"),
        ]:
            rates = []
            for s in STEPS:
                per_seed = list(agg["own_emit"][(arm, persona, s)].values())
                rates.append(np.mean(per_seed))
            axe.plot(
                steps_x,
                rates,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.5,
            )

        axe.axvspan(15, 22, color=install_grey, alpha=0.18, zorder=0)
        axe.axhline(0.5, color="#666", linestyle=":", linewidth=0.9, zorder=1)
        axe.set_xscale("log")
        axe.set_xlim(16, 140)
        axe.set_xticks(STEPS)
        axe.set_xticklabels([str(s) for s in STEPS])
        axe.minorticks_off()
        axe.set_xlabel("Optimizer steps (log scale)")
        if col == 0:
            axe.set_ylabel("Own-encoding\nargmax-emit rate")
        axe.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    savefig_paper(fig, "issue_533/bw_install_levels", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)


def main() -> None:
    if not ANALYSIS_PATH.exists():
        raise FileNotFoundError(f"missing analysis.json at {ANALYSIS_PATH}")
    agg = aggregate()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_paired_gap_trajectory(agg)
    fig_install_levels(agg)
    print("Wrote figures to", FIG_DIR)


if __name__ == "__main__":
    main()
