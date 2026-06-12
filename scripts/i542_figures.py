"""Issue #542 figures (plan §5): hero 1 composition dot plot, hero 2 count axis,
plus the exploratory dump (per-arm heatmap strip, stop-step vs count, replicate
scatter, logit-space twins of both heroes).

CPU-only; consumes ``analysis/registered_reads_542.json``, the per-arm G
tensors, and the stop-step files. Figures -> ``figures/issue_542/`` (PNG+PDF
with the repo paper-plots rcParams + commit-pinned meta.json).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


FIG_DIR = Path(os.environ.get("I542_FIG_DIR", str(REPO / "figures/issue_542")))
ARM_LABELS = {
    "arm1_xfam": "Cross-family panel\n(parent control)",
    "arm2_close": "Close-persona panel",
    "arm3_default": "Default-including panel",
    "c2": "Two negatives",
    "c8": "Eight negatives",
    "c16": "Sixteen negatives",
}
COUNTS = {"c2": 2, "arm1_xfam": 4, "c8": 8, "c16": 16}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _save(fig, name: str, meta: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    (FIG_DIR / f"{name}.meta.json").write_text(
        json.dumps(
            {
                "git_commit": _git_commit(),
                "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                **meta,
            },
            indent=2,
        )
    )
    plt.close(fig)
    print(f"[figures] wrote {FIG_DIR / name}.png")


def _floor_band(reads: dict, stat: str) -> float | None:
    pooled = reads.get("seed_noise", {}).get("pooled", {}).get(stat)
    return pooled["arm_level_floor"] if pooled else None


def hero_composition(reads: dict, sn: dict, *, space: str = "logp") -> None:
    """Hero 1: per-arm off-diag mean / default column / gradient slope dots."""
    arms = [a for a in ARM_LABELS if a in reads["per_arm"] and not a.startswith("repl_")]
    stats = [
        ("offdiag_mean", "Off-diagonal mean ΔlogP(※) (nat)"),
        ("default_col_broad_full_mask", "Default column, 10 broad rows (nat)"),
    ]
    fig, axes = plt.subplots(1, len(stats), figsize=(4.2 * len(stats), 3.6), sharey=False)
    for ax, (stat, label) in zip(axes, stats, strict=True):
        ys = [reads["per_arm"][a].get(stat) for a in arms]
        xs = np.arange(len(arms))
        ax.scatter(xs, ys, s=60, zorder=3)
        floor = _floor_band(
            {"seed_noise": sn}, "offdiag_mean" if stat == "offdiag_mean" else "default_col"
        )
        if floor is not None and ys[0] is not None:
            ref = ys[0]
            ax.axhspan(ref - floor, ref + floor, alpha=0.15, color="gray", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(label)
    fig.suptitle("Negative-panel composition: arm summaries with seed-noise band", fontsize=10)
    _save(
        fig,
        f"hero1_composition_{space}",
        {"arms": arms, "note": "band = ±1 arm-level seed-noise floor around the parent control"},
    )


def hero_count(reads: dict, sn: dict) -> None:
    """Hero 2: count-axis lines for diag / off-diag / default column."""
    arms = [a for a in COUNTS if a in reads["per_arm"]]
    if len(arms) < 3:
        print("[figures] hero2 skipped: <3 count levels present")
        return
    xs = [np.log2(COUNTS[a]) for a in arms]
    series = [
        ("diag_mean", "Diagonal mean (implant)"),
        ("offdiag_mean", "Off-diagonal mean (leakage)"),
        ("default_col_broad_full_mask", "Default column (broad rows)"),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for stat, label in series:
        ys = [reads["per_arm"][a].get(stat) for a in arms]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=label)
    floor = _floor_band({"seed_noise": sn}, "offdiag_mean")
    if floor is not None:
        ax.fill_between(
            sorted(xs),
            [reads["per_arm"]["arm1_xfam"]["offdiag_mean"] - floor] * len(xs),
            [reads["per_arm"]["arm1_xfam"]["offdiag_mean"] + floor] * len(xs),
            alpha=0.12,
            color="gray",
        )
    ax.set_xticks(sorted(xs))
    ax.set_xticklabels([str(COUNTS[a]) for a in sorted(arms, key=lambda a: COUNTS[a])])
    ax.set_xlabel("Negative-persona count (300 negative rows at every level)", fontsize=9)
    ax.set_ylabel("ΔlogP(※) trained - base (nat)")
    ax.legend(fontsize=7)
    _save(fig, "hero2_count_axis", {"arms": arms})


def heatmap_strip(eval_root: Path) -> None:
    arm_dirs = sorted(p.name for p in (eval_root / "G_arm").glob("*") if p.is_dir())
    arm_dirs = [a for a in arm_dirs if not a.startswith("repl_")]
    if not arm_dirs:
        return
    fig, axes = plt.subplots(1, len(arm_dirs), figsize=(3.2 * len(arm_dirs), 4.6))
    if len(arm_dirs) == 1:
        axes = [axes]
    for ax, arm in zip(axes, arm_dirs, strict=True):
        z = np.load(eval_root / f"G_arm/{arm}/G_tensor.npz", allow_pickle=True)
        im = ax.imshow(z["G"], aspect="auto", cmap="viridis", vmin=-2, vmax=14)
        ax.set_title(ARM_LABELS.get(arm, arm), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks(range(len(z["train_cids"])))
        ax.set_yticklabels([str(c) for c in z["train_cids"]], fontsize=5)
    fig.colorbar(im, ax=axes, shrink=0.7, label="ΔlogP(※) (nat)")
    _save(fig, "per_arm_heatmap_strip", {"arms": arm_dirs})


def stop_step_figure(reads: dict) -> None:
    ss = reads.get("stop_steps", {})
    arms = [a for a in COUNTS if a in ss]
    if len(arms) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for arm in arms:
        steps = list(ss[arm].values())
        ax.scatter([np.log2(COUNTS[arm])] * len(steps), steps, alpha=0.6, s=24)
    ax.set_xticks([np.log2(COUNTS[a]) for a in arms])
    ax.set_xticklabels([str(COUNTS[a]) for a in arms])
    ax.set_xlabel("Negative-persona count")
    ax.set_ylabel("Realized band-stop step")
    _save(fig, "stop_step_vs_count", {"arms": arms})


def replicate_scatter(eval_root: Path) -> None:
    pairs = []
    for repl, ref in (("repl_parent", "arm1_xfam"), ("repl_close", "arm2_close")):
        rp, fp = eval_root / f"G_arm/{repl}/G_tensor.npz", eval_root / f"G_arm/{ref}/G_tensor.npz"
        if not (rp.exists() and fp.exists()):
            continue
        zr, zf = np.load(rp, allow_pickle=True), np.load(fp, allow_pickle=True)
        tr = [str(c) for c in zr["train_cids"]]
        tf = [str(c) for c in zf["train_cids"]]
        for ci in tr:
            pairs.append(
                (
                    zf["G"][tf.index(ci), :],
                    zr["G"][tr.index(ci), :],
                    repl,
                )
            )
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for s42, s43, repl in pairs:
        ax.scatter(s42, s43, s=8, alpha=0.4, label=repl)
    lims = ax.get_xlim()
    ax.plot(lims, lims, ls="--", lw=0.8, color="gray")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=True))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7)
    ax.set_xlabel("ΔlogP(※), seed 42 (nat)")
    ax.set_ylabel("ΔlogP(※), seed 43 replicate (nat)")
    _save(fig, "replicate_scatter", {"n_rows": len(pairs)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=Path(os.environ.get("I542_EVAL_ROOT", str(REPO / "eval_results/issue_542"))),
    )
    args = ap.parse_args()
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    reads = json.loads((args.eval_root / "analysis/registered_reads_542.json").read_text())
    sn = json.loads((args.eval_root / "analysis/seed_noise_542.json").read_text())
    hero_composition(reads, sn)
    hero_count(reads, sn)
    heatmap_strip(args.eval_root)
    stop_step_figure(reads)
    replicate_scatter(args.eval_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
