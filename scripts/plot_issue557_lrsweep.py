#!/usr/bin/env python3
"""Issue #557 figures — erasure-pressure dose-response (OFF-POD, VM-side).

Reads ``eval_results/issue_557/rollup.json`` (rollup_issue557_lrsweep.py) plus
the per-cell ``phase2_trajectory_trigger.jsonl`` files (new cells AND the
parent's committed lr=1e-4 anchor cells) and writes to ``figures/issue_557/``:

  - ``dose_response_hero``       — pooled post-SFT trigger emission (Wilson
    CIs, per-seed dots) and trigger-slot delta log P retention vs Phase-2 peak
    lr (log x; parent 1e-4 anchor included).  [plan §6 hero]
  - ``collapse_trajectories``    — frozen-probe trained log P(marker) and
    argmax-rate vs Phase-2 step, one color per lr, with the (10-15)x(1e-4/lr)
    predicted cliff windows shaded.  [plan §6 hero-2]
  - ``key_conditioning_vs_lr``   — trigger vs no-key pooled emission + the
    per-seed key gap in delta log P.
  - ``cliff_step_vs_pressure``   — measured cliff step vs 1e-4/lr with the
    predicted band (raw per-cell points alongside).
  - ``absorption_vs_lr``         — raw delta-CE per cell AND the absorption
    fraction f vs lr (guard covariate on the same x-axis).
  - ``judge_scores_vs_lr``       — mean medical-helpfulness judge score per
    adapter set vs lr (skipped with a warning when judge scores are absent).

Every panel keeps raw alongside processed (per-seed dots / raw delta-CE).

Usage:
    uv run python scripts/plot_issue557_lrsweep.py
    uv run python scripts/plot_issue557_lrsweep.py --eval-root /tmp/fx/eval_results \\
        --out-dir /tmp/fx/figs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="plot_issue557_lrsweep")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _issue543_common import PROJECT_ROOT  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

log = logging.getLogger("plot_issue557_lrsweep")

ARM = "r50"
ANCHOR_ARM_KEY = "lr1e4"


def _lr_label(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def _arm_order(rollup: dict) -> list[str]:
    """Arm keys sorted by descending lr (anchor first)."""
    arms = rollup["arms"]
    return sorted(arms, key=lambda k: -(arms[k]["lr"] or 0.0))


def _yerr(mean: float, lo: float, hi: float) -> tuple[float, float]:
    """CI half-widths clamped at 0 (float-epsilon-negative guard)."""
    return (max(0.0, mean - lo), max(0.0, hi - mean))


def plot_dose_response(rollup: dict, out_dir: Path) -> None:
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
    order = _arm_order(rollup)
    for i, key in enumerate(order):
        arm = rollup["arms"][key]
        lr = arm["lr"]
        if lr is None:
            log.warning("Arm %s has no lr — skipping in dose-response.", key)
            continue
        color = (
            paper_palette_role("baseline")
            if key == ANCHOR_ARM_KEY
            else paper_palette(len(order))[i]
        )
        trig = arm["trigger"]
        rate = trig["pooled_emission_rate"]
        if rate is not None:
            lo, hi = trig["wilson_95ci"]
            ax_top.errorbar(
                [lr],
                [rate],
                yerr=[[_yerr(rate, lo, hi)[0]], [_yerr(rate, lo, hi)[1]]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=8,
                zorder=3,
            )
        for r in trig["per_seed_emission"].values():
            if r is not None:
                ax_top.plot([lr], [r], "o", color=color, alpha=0.35, markersize=4, zorder=2)
        for v in arm["per_seed_trigger_delta_logp"].values():
            if v is not None:
                ax_bot.plot([lr], [v], "o", color=color, alpha=0.6, markersize=6)
    ax_top.axhline(1.0, color=paper_palette_role("neutral"), lw=1.0, ls="--")
    ax_top.text(
        ax_top.get_xlim()[0],
        1.0,
        "pre-SFT emission (all seeds 100%)",
        fontsize=8,
        va="bottom",
    )
    ax_top.set_ylabel("Post-SFT trigger emission rate")
    ax_top.set_ylim(-0.05, 1.1)
    ax_bot.set_ylabel("Trigger-slot $\\Delta\\log P$(marker)\n(trained $-$ base, nats)")
    ax_bot.set_xlabel("Phase-2 peak learning rate (erasure pressure)")
    ax_bot.set_xscale("log")
    ax_top.set_title("Erasure-pressure dose-response over the 50%-arm marker installs")
    savefig_paper(fig, "dose_response_hero", dir=out_dir)
    plt.close(fig)


def _trajectory_series(path: Path) -> tuple[list[int], list[float], list[float]]:
    steps, logp_means, argmax_rates = [], [], []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        row = json.loads(ln)
        steps.append(int(row["step"]))
        logp_means.append(sum(row["trained"]["logp"]) / len(row["trained"]["logp"]))
        argmax_rates.append(float(row["argmax_rate"]))
    return steps, logp_means, argmax_rates


def plot_trajectories(rollup: dict, eval_root: Path, out_dir: Path, seeds: list[int]) -> None:
    fig, (ax_lp, ax_am) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    order = _arm_order(rollup)
    colors = paper_palette(len(order))
    any_curve = False
    for i, key in enumerate(order):
        lr = rollup["arms"][key]["lr"]
        if lr is None:
            continue
        color = paper_palette_role("baseline") if key == ANCHOR_ARM_KEY else colors[i]
        label_done = False
        for s in seeds:
            if key == ANCHOR_ARM_KEY:
                traj = (
                    eval_root / "issue_543" / ARM / f"seed{s}" / "phase2_trajectory_trigger.jsonl"
                )
            else:
                traj = (
                    eval_root
                    / "issue_557"
                    / ARM
                    / key
                    / f"seed{s}"
                    / "phase2_trajectory_trigger.jsonl"
                )
            if not traj.exists():
                log.warning("Trajectory missing: %s", traj)
                continue
            steps, logp, am = _trajectory_series(traj)
            label = f"lr {_lr_label(lr)}" if not label_done else None
            ax_lp.plot(steps, logp, color=color, alpha=0.8, lw=1.2, label=label)
            ax_am.plot(steps, am, color=color, alpha=0.8, lw=1.2)
            label_done = True
            any_curve = True
        # Predicted cliff window (10-15) x (1e-4 / lr), shaded per arm.
        scale = 1.0e-4 / lr
        ax_am.axvspan(10 * scale, 15 * scale, color=color, alpha=0.08)
    if not any_curve:
        log.warning("No trajectory curves found — skipping collapse_trajectories figure.")
        plt.close(fig)
        return
    ax_lp.set_ylabel("Frozen-probe trained $\\log P$(marker)\n(teacher-forced, nats)")
    ax_am.set_ylabel("Probe slot argmax-rate (marker)")
    ax_am.set_xlabel("Phase-2 optimizer step")
    ax_lp.set_title(
        "Marker collapse trajectories by erasure pressure\n"
        "(shaded: predicted cliff windows, (10-15)$\\times$(1e-4/lr); "
        "within-condition dynamics — not a cross-arm magnitude leaderboard)"
    )
    ax_lp.legend()
    savefig_paper(fig, "collapse_trajectories", dir=out_dir)
    plt.close(fig)


def plot_key_conditioning(rollup: dict, out_dir: Path) -> None:
    fig, (ax_em, ax_gap) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    order = _arm_order(rollup)
    for key in order:
        arm = rollup["arms"][key]
        lr = arm["lr"]
        if lr is None:
            continue
        for cell, color_role, marker in (
            ("trigger", "accent", "o"),
            ("no_trigger", "control", "s"),
        ):
            blk = arm[cell]
            rate = blk["pooled_emission_rate"]
            if rate is None:
                continue
            lo, hi = blk["wilson_95ci"]
            ax_em.errorbar(
                [lr],
                [rate],
                yerr=[[_yerr(rate, lo, hi)[0]], [_yerr(rate, lo, hi)[1]]],
                fmt=marker,
                color=paper_palette_role(color_role),
                capsize=3,
                label=cell.replace("no_trigger", "no key").replace("trigger", "with key")
                if key == order[0]
                else None,
            )
        for gap in arm["per_seed_keygap_delta_logp"].values():
            if gap is not None:
                ax_gap.plot([lr], [gap], "o", color=paper_palette_role("accent"), alpha=0.6)
    ax_em.set_xscale("log")
    ax_gap.set_xscale("log")
    ax_em.set_xlabel("Phase-2 peak learning rate")
    ax_gap.set_xlabel("Phase-2 peak learning rate")
    ax_em.set_ylabel("Post-SFT emission rate")
    ax_gap.set_ylabel("Key gap in $\\Delta\\log P$ (with key $-$ no key, nats)")
    ax_gap.axhline(0.0, color=paper_palette_role("neutral"), lw=1.0, ls="--")
    ax_em.legend()
    ax_em.set_title("Key-conditioned emission vs pressure")
    ax_gap.set_title("Latent key-conditioning vs pressure (per seed)")
    savefig_paper(fig, "key_conditioning_vs_lr", dir=out_dir)
    plt.close(fig)


def plot_cliff_steps(rollup: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    drew = False
    for _cell_key, cell in rollup["cells"].items():
        lr, step = cell.get("lr"), cell.get("cliff_step")
        if lr is None or step is None:
            continue
        x = 1.0e-4 / lr
        ax.plot([x], [step], "o", color=paper_palette_role("accent"), alpha=0.7)
        drew = True
    if not drew:
        log.warning("No cliff steps available — skipping cliff_step_vs_pressure figure.")
        plt.close(fig)
        return
    xs = sorted({1.0e-4 / c["lr"] for c in rollup["cells"].values() if c.get("lr")})
    ax.fill_between(
        xs,
        [10 * x for x in xs],
        [15 * x for x in xs],
        color=paper_palette_role("baseline"),
        alpha=0.15,
        label="predicted: (10-15)$\\times$(1e-4/lr)",
    )
    ax.set_xlabel("Pressure ratio 1e-4 / lr")
    ax.set_ylabel("Cliff step (first probe read with argmax-rate 0)")
    ax.set_title("Collapse timing vs inverse pressure (per cell)")
    ax.legend()
    savefig_paper(fig, "cliff_step_vs_pressure", dir=out_dir)
    plt.close(fig)


def plot_absorption(rollup: dict, out_dir: Path) -> None:
    absorption = rollup.get("absorption")
    if not absorption or not absorption.get("cells"):
        log.warning("No absorption block in rollup — skipping absorption_vs_lr figure.")
        return
    lr_of = {k: v["lr"] for k, v in rollup["arms"].items()}
    fig, (ax_dce, ax_f) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for cell_key, cell in absorption["cells"].items():
        variant = cell_key.rsplit("_seed", 1)[0]
        lr = lr_of.get(variant)
        if lr is None:
            continue
        d, ci = cell["delta_ce_med"], cell["ci95"]
        ax_dce.errorbar(
            [lr],
            [d],
            yerr=[[_yerr(d, ci[0], ci[1])[0]], [_yerr(d, ci[0], ci[1])[1]]],
            fmt="o",
            color=paper_palette_role("accent"),
            capsize=3,
            alpha=0.7,
        )
        if cell.get("absorption_fraction_f") is not None:
            ax_f.plot(
                [lr],
                [cell["absorption_fraction_f"]],
                "o",
                color=paper_palette_role("accent"),
                alpha=0.7,
            )
    for _s_key, a in (absorption.get("anchor_cells") or {}).items():
        ax_dce.plot([1.0e-4], [a["delta_ce_med"]], "s", color=paper_palette_role("baseline"))
        ax_f.plot([1.0e-4], [1.0], "s", color=paper_palette_role("baseline"))
    ax_dce.axhline(0.0, color=paper_palette_role("neutral"), lw=1.0, ls="--")
    ax_dce.set_xscale("log")
    ax_f.set_xscale("log")
    ax_dce.set_xlabel("Phase-2 peak learning rate")
    ax_f.set_xlabel("Phase-2 peak learning rate")
    ax_dce.set_ylabel("Raw $\\Delta$CE on the frozen medical probe\n(pre $-$ post, nats/token)")
    ax_f.set_ylabel("Absorption fraction f (vs lr=1e-4 anchor)")
    ax_dce.set_title("Medical absorption per cell (raw, 95% bootstrap CI)")
    ax_f.set_title("Absorption fraction vs pressure")
    savefig_paper(fig, "absorption_vs_lr", dir=out_dir)
    plt.close(fig)


def plot_judge_scores(rollup: dict, out_dir: Path) -> None:
    scores = rollup.get("judge_scores_per_set_mean")
    if not scores:
        log.warning("No judge scores in rollup — skipping judge_scores_vs_lr figure.")
        return
    lr_of = {k: v["lr"] for k, v in rollup["arms"].items()}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for set_name, mean in scores.items():
        if mean is None:
            continue
        if set_name == "base":
            ax.axhline(mean, color=paper_palette_role("neutral"), lw=1.0, ls="--")
            ax.text(1.2e-4, mean, "base model", fontsize=8, va="bottom")
            continue
        if set_name.startswith("post_"):
            variant = set_name.removeprefix("post_").rsplit("_seed", 1)[0]
            lr, color = lr_of.get(variant), paper_palette_role("accent")
        elif set_name.startswith("anchor_"):
            lr, color = 1.0e-4, paper_palette_role("baseline")
        else:  # pre_seed<S> — the Phase-1 (unerased) starting point
            lr, color = None, paper_palette_role("control")
        if lr is None:
            continue
        ax.plot([lr], [mean], "o", color=color, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Phase-2 peak learning rate")
    ax.set_ylabel("Mean judge score (medical helpfulness, 1-10)")
    ax.set_title("Medical-answer quality vs pressure (descriptive guard)")
    savefig_paper(fig, "judge_scores_vs_lr", dir=out_dir)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollup_path = eval_root / "issue_557" / "rollup.json"
    if not rollup_path.exists():
        raise FileNotFoundError(f"{rollup_path} missing — run rollup_issue557_lrsweep.py first.")
    rollup = json.loads(rollup_path.read_text())
    seeds = [int(s) for s in args.seeds.split(",") if s]

    set_paper_style("blog")
    plot_dose_response(rollup, out_dir)
    plot_trajectories(rollup, eval_root, out_dir, seeds)
    plot_key_conditioning(rollup, out_dir)
    plot_cliff_steps(rollup, out_dir)
    plot_absorption(rollup, out_dir)
    plot_judge_scores(rollup, out_dir)
    log.info("Figures -> %s", out_dir)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #557 lr-sweep figures (CPU-only, VM-side).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval-root", type=str, default=str(PROJECT_ROOT / "eval_results"))
    p.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "figures" / "issue_557"))
    p.add_argument("--seeds", type=str, default="42,137,256")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
