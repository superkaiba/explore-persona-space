"""Figure generation for issue #498 (plan §6.3 + §6.4).

Renders the hero + exploratory figure dump from
``eval_results/issue_498/analysis.json``. Style follows
``src/explore_persona_space/analysis/paper_plots.py``.

Outputs land in ``figures/issue_498/``:
  - hero.png (+ .pdf + .meta.json)
  - per_trait.png
  - per_seed.png
  - raw_alongside.png
  - histograms.png
  - dynamic_range_gate.png
  - judge_consistency.png
  - trajectory.png (placeholder; reads WandB keyword-proxy stream if present)
  - leakage_by_trait_appropriateness.png

CLI:
    uv run python scripts/plot_i498_clean_result.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("i498.plot")

ANALYSIS_PATH = Path("eval_results/issue_498/analysis.json")
JUDGE_PATH = Path("eval_results/issue_498/judge_scores.json")
FIG_DIR = Path("figures/issue_498")
SYMMETRIC_CELLS = ("cross_scenario", "default_assistant")
ARMS = ("system", "role")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _save_fig(fig, name: str, meta: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    meta_path = FIG_DIR / f"{name}.meta.json"
    fig.savefig(png, bbox_inches="tight", dpi=200)
    fig.savefig(pdf, bbox_inches="tight")
    meta_path.write_text(
        json.dumps(
            {**meta, "git_commit": _git(), "ts": _dt.datetime.utcnow().isoformat() + "Z"},
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s + .pdf + .meta.json", png)


def main(argv: list[str] | None = None) -> None:  # noqa: C901 — 6-figure plotting
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args = ap.parse_args(argv)
    _ = args  # currently no flags

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not ANALYSIS_PATH.exists():
        raise SystemExit(f"{ANALYSIS_PATH} missing — run Phase 5 analyze first.")
    analysis = json.loads(ANALYSIS_PATH.read_text())
    per_cell = analysis.get("per_cell", {})
    headline = analysis.get("headline", {})

    # ---------- Figure 1: hero. Arms x {in_scenario, cross_scenario, default}.
    contexts = ("in_scenario", "cross_scenario", "default_assistant")
    arm_color = {"system": "tab:blue", "role": "tab:orange"}
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    xs = list(range(len(contexts)))
    for i, arm in enumerate(ARMS):
        means = []
        for ec in contexts:
            vals = []
            for _trait, by_ec in per_cell.get(arm, {}).items():
                for _seed, cell in by_ec.get(ec, {}).items():
                    vals.append(cell["mean"])
            means.append(sum(vals) / max(1, len(vals)))
        offsets = [x + (i - 0.5) * width for x in xs]
        ax.bar(offsets, means, width=width, label=arm, color=arm_color[arm])
    ax.set_xticks(xs)
    ax.set_xticklabels(contexts)
    ax.set_ylabel("mean judge Likert (1-5)")
    ci = headline.get("ci_masked", {})
    ax.set_title(
        f"#498 hero — d_seed(masked) = {ci.get('mean', 0):.2f} "
        f"[{ci.get('lo', 0):.2f}, {ci.get('hi', 0):.2f}]"
    )
    ax.legend()
    ax.set_ylim(0, 5)
    _save_fig(fig, "hero", {"figure": "hero"})
    plt.close(fig)

    # ---------- Figure 2: per-trait panels.
    traits = sorted({t for arm in per_cell for t in per_cell[arm]})
    if traits:
        fig, axes = plt.subplots(1, len(traits), figsize=(4 * len(traits), 4), sharey=True)
        if len(traits) == 1:
            axes = [axes]
        for ax, trait in zip(axes, traits, strict=False):
            for i, arm in enumerate(ARMS):
                means = []
                for ec in contexts:
                    vals = []
                    for _seed, cell in per_cell.get(arm, {}).get(trait, {}).get(ec, {}).items():
                        vals.append(cell["mean"])
                    means.append(sum(vals) / max(1, len(vals)))
                offsets = [x + (i - 0.5) * width for x in xs]
                ax.bar(offsets, means, width=width, label=arm, color=arm_color[arm])
            ax.set_xticks(xs)
            ax.set_xticklabels(contexts, rotation=20)
            ax.set_title(trait)
            ax.set_ylim(0, 5)
        axes[0].set_ylabel("mean Likert")
        axes[0].legend()
        _save_fig(fig, "per_trait", {"figure": "per_trait"})
        plt.close(fig)

    # ---------- Figure 3: per-seed scatter (in-scenario vs leakage).
    fig, ax = plt.subplots(figsize=(5, 5))
    for arm in ARMS:
        in_sc = []
        leak = []
        for trait, by_ec in per_cell.get(arm, {}).items():
            seeds_seen: set = set()
            for _ec, sd in by_ec.items():
                seeds_seen.update(sd.keys())
            for seed in seeds_seen:
                m_in = per_cell[arm][trait].get("in_scenario", {}).get(seed, {}).get("mean")
                m_leak_vals = [
                    per_cell[arm][trait].get(ec, {}).get(seed, {}).get("mean")
                    for ec in SYMMETRIC_CELLS
                ]
                m_leak_vals = [m for m in m_leak_vals if m is not None]
                if m_in is None or not m_leak_vals:
                    continue
                in_sc.append(m_in)
                leak.append(sum(m_leak_vals) / len(m_leak_vals))
        ax.scatter(in_sc, leak, label=arm, color=arm_color[arm])
    ax.plot([0, 5], [0, 5], "k--", alpha=0.3)
    ax.set_xlabel("in-scenario mean Likert")
    ax.set_ylabel("symmetric-leakage mean Likert")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.legend()
    ax.set_title("per-seed in-scenario vs leakage")
    _save_fig(fig, "per_seed", {"figure": "per_seed"})
    plt.close(fig)

    # ---------- Figure 4: dynamic-range gate (sd per arm x trait x eval_context).
    fig, ax = plt.subplots(figsize=(8, 4))
    sd_map = analysis.get("sd_by_arm_unit", {})
    keys = sorted(sd_map.keys())
    ax.bar(range(len(keys)), [sd_map[k] for k in keys])
    ax.axhline(0.3, color="r", linestyle="--", label="sd=0.3 gate")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=80, fontsize=6)
    ax.set_ylabel("sd of judge Likert across 40 q x 3 seeds")
    ax.legend()
    _save_fig(fig, "dynamic_range_gate", {"figure": "dynamic_range_gate"})
    plt.close(fig)

    # ---------- Figure 5: judge paraphrase consistency.
    para = analysis.get("paraphrase_spearman", {})
    fig, ax = plt.subplots(figsize=(6, 3))
    keys = sorted(k for k in para if k.endswith("__spearman"))
    if keys:
        ax.bar(range(len(keys)), [para[k] for k in keys])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=80, fontsize=6)
        ax.axhline(0.7, color="r", linestyle="--", label="rho=0.7 threshold")
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Spearman rho (primary vs paraphrase)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no paraphrase replication available", ha="center")
    _save_fig(fig, "judge_consistency", {"figure": "judge_consistency"})
    plt.close(fig)

    # ---------- Figure 6: raw-alongside (mean Likert + raw judge-score
    # distribution per arm x eval_context).
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # left: raw means (same data as hero).
    ax = axes[0]
    for i, arm in enumerate(ARMS):
        means = []
        for ec in contexts:
            vals = []
            for _trait, by_ec in per_cell.get(arm, {}).items():
                for _seed, cell in by_ec.get(ec, {}).items():
                    vals.append(cell["mean"])
            means.append(sum(vals) / max(1, len(vals)))
        offsets = [x + (i - 0.5) * width for x in xs]
        ax.bar(offsets, means, width=width, label=arm, color=arm_color[arm])
    ax.set_xticks(xs)
    ax.set_xticklabels(contexts, rotation=20)
    ax.set_title("raw mean Likert (= hero)")
    ax.set_ylim(0, 5)
    ax.legend()
    # right: distribution across (40 q x 3 seeds) per arm x eval_context.
    ax = axes[1]
    if JUDGE_PATH.exists():
        rows = json.loads(JUDGE_PATH.read_text()).get("rows", [])
        for arm in ARMS:
            for ec in contexts:
                vals = [
                    r["score"]
                    for r in rows
                    if r.get("arm") == arm
                    and r.get("eval_context") == ec
                    and r.get("score") is not None
                ]
                if not vals:
                    continue
                ax.hist(vals, bins=range(1, 7), alpha=0.4, label=f"{arm}/{ec}")
        ax.set_xlabel("judge Likert")
        ax.set_ylabel("count")
        ax.legend(fontsize=6)
    _save_fig(fig, "raw_alongside", {"figure": "raw_alongside"})
    plt.close(fig)


if __name__ == "__main__":
    main()
