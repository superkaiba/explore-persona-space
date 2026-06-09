"""Plot 3 hero figures + exploratory dump for #528 (plan §6.3).

Reads ``eval_results/issue_528/analysis.json`` and writes:

- ``figures/issue_528/hero1_h1_installation.{png,pdf,meta.json}`` —
  4-panel bar chart, one panel per trait, x = {base, trained-system,
  trained-role} with per-cell SE bars + dashed line at 3.5.
- ``figures/issue_528/hero2_h2_segmentation.{png,pdf,meta.json}`` —
  4-panel bar chart, one panel per trait, x = {own, sib1, sib2, sib3,
  default}, system / role grouped bars per x-position with 3-seed SE.
- ``figures/issue_528/hero3_joint_on_vs_off.{png,pdf,meta.json}`` —
  scatter of (on-target, off-target) per (trait, seed), system vs role.

Inter font + paper rcParams via :mod:`explore_persona_space.analysis.paper_plots`.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("i528.plot")

OUT_DIR = Path("figures/issue_528")
ANALYSIS_PATH = Path("eval_results/issue_528/analysis.json")
JUDGE_PATH = Path("eval_results/issue_528/judge_scores.json")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _save(fig, slug: str, *, meta: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{slug}.{ext}", bbox_inches="tight", dpi=300)
    meta_path = OUT_DIR / f"{slug}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("Wrote %s.{png,pdf,meta.json}", OUT_DIR / slug)


def _mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (mean, (var / n) ** 0.5)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)
    _ = args  # consumed only for parity with other scripts

    if not ANALYSIS_PATH.exists() or not JUDGE_PATH.exists():
        raise SystemExit(f"Missing input(s): {ANALYSIS_PATH.exists()=}, {JUDGE_PATH.exists()=}")

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    # set_paper_style mutates plt.rcParams in place (no return value).
    set_paper_style()

    analysis = json.loads(ANALYSIS_PATH.read_text())
    judge = json.loads(JUDGE_PATH.read_text())
    rows = judge["rows"]

    traits = ("validating", "conciseness", "asks_clarifying_first", "calibrated_uncertainty")
    contexts = ("own_scenario", "sibling_1", "sibling_2", "sibling_3", "default_assistant")

    # ---------- Hero 1: H1 installation ----------
    fig1, axes1 = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for i, trait in enumerate(traits):
        ax = axes1[i]
        # Base own_scenario (eval_arm=system).
        base_scores = [
            float(r["score"])
            for r in rows
            if r.get("kind") == "base"
            and r["trait"] == trait
            and r.get("eval_context") == "own_scenario"
            and (r.get("arm") or r.get("eval_arm")) == "system"
        ]
        # Trained own_scenario, system arm, all seeds pooled.
        ts_scores = [
            float(r["score"])
            for r in rows
            if r.get("kind") == "trained"
            and r["trait"] == trait
            and r.get("arm") == "system"
            and r.get("eval_context") == "own_scenario"
        ]
        tr_scores = [
            float(r["score"])
            for r in rows
            if r.get("kind") == "trained"
            and r["trait"] == trait
            and r.get("arm") == "role"
            and r.get("eval_context") == "own_scenario"
        ]
        labels = ["Base", "Trained\n(system)", "Trained\n(role)"]
        means, ses = zip(
            *(_mean_se(v) for v in (base_scores, ts_scores, tr_scores)),
            strict=True,
        )
        ax.bar(labels, means, yerr=ses, color=("#888888", "#d95f02", "#1f78b4"))
        ax.axhline(3.5, linestyle="--", color="gray", linewidth=0.8)
        ax.set_ylim(0, 5)
        ax.set_title(trait.replace("_", " "))
        info = analysis.get("h1_per_trait", {}).get(trait, {})
        p_holm = info.get("p_holm")
        delta = info.get("paired_delta_mean")
        if p_holm is not None and delta is not None:
            ax.text(
                0.5,
                0.95,
                f"Δ={delta:.2f} (p_holm={p_holm:.3f})",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )
    axes1[0].set_ylabel("Likert score (1-5)")
    fig1.suptitle("H1 — does LoRA install the trait above base?")
    _save(
        fig1,
        "hero1_h1_installation",
        meta={
            "kind": "hero_h1_installation",
            "git_commit": _git(),
            "ts": datetime.utcnow().isoformat() + "Z",
            "analysis_path": str(ANALYSIS_PATH),
        },
    )

    # ---------- Hero 2: H2 segmentation by context ----------
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    width = 0.4
    for i, trait in enumerate(traits):
        ax = axes2[i]
        x_pos = list(range(len(contexts)))
        sys_means, sys_ses, role_means, role_ses = [], [], [], []
        for ctx in contexts:
            sys_scores = [
                float(r["score"])
                for r in rows
                if r.get("kind") == "trained"
                and r["trait"] == trait
                and r.get("arm") == "system"
                and r.get("eval_context") == ctx
            ]
            role_scores = [
                float(r["score"])
                for r in rows
                if r.get("kind") == "trained"
                and r["trait"] == trait
                and r.get("arm") == "role"
                and r.get("eval_context") == ctx
            ]
            m_s, se_s = _mean_se(sys_scores)
            m_r, se_r = _mean_se(role_scores)
            sys_means.append(m_s)
            sys_ses.append(se_s)
            role_means.append(m_r)
            role_ses.append(se_r)
        ax.bar(
            [x - width / 2 for x in x_pos],
            sys_means,
            width,
            yerr=sys_ses,
            color="#d95f02",
            label="system",
        )
        ax.bar(
            [x + width / 2 for x in x_pos],
            role_means,
            width,
            yerr=role_ses,
            color="#1f78b4",
            label="role",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace("_", "\n") for c in contexts], fontsize=7)
        ax.set_ylim(0, 5)
        ax.set_title(trait.replace("_", " "))
        if i == 0:
            ax.legend(fontsize=7)
    axes2[0].set_ylabel("Likert score (1-5)")
    fig2.suptitle("H2 — segmentation across own + off-target contexts")
    _save(
        fig2,
        "hero2_h2_segmentation",
        meta={
            "kind": "hero_h2_segmentation",
            "git_commit": _git(),
            "ts": datetime.utcnow().isoformat() + "Z",
        },
    )

    # ---------- Hero 3: joint on-target vs off-target ----------
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    for arm, color in (("system", "#d95f02"), ("role", "#1f78b4")):
        xs, ys = [], []
        for trait in traits:
            for seed in (42, 137, 1337):
                on_scores = [
                    float(r["score"])
                    for r in rows
                    if r.get("kind") == "trained"
                    and r["trait"] == trait
                    and r.get("arm") == arm
                    and r.get("seed") == seed
                    and r.get("eval_context") == "own_scenario"
                ]
                off_scores = [
                    float(r["score"])
                    for r in rows
                    if r.get("kind") == "trained"
                    and r["trait"] == trait
                    and r.get("arm") == arm
                    and r.get("seed") == seed
                    and r.get("eval_context")
                    in ("sibling_1", "sibling_2", "sibling_3", "default_assistant")
                ]
                if not on_scores or not off_scores:
                    continue
                xs.append(sum(on_scores) / len(on_scores))
                ys.append(sum(off_scores) / len(off_scores))
        ax3.scatter(xs, ys, color=color, alpha=0.7, label=arm)
    lo, hi = 1, 5
    ax3.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax3.set_xlim(lo, hi)
    ax3.set_ylim(lo, hi)
    ax3.set_xlabel("On-target Likert (own scenario)")
    ax3.set_ylabel("Off-target Likert (4-context mean)")
    ax3.set_title("H2 — points below y=x indicate cleaner segmentation")
    ax3.legend()
    _save(
        fig3,
        "hero3_joint_on_vs_off",
        meta={
            "kind": "hero_h2_joint",
            "git_commit": _git(),
            "ts": datetime.utcnow().isoformat() + "Z",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
