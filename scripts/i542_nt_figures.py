"""Issue #542 follow-up (genuine-near-twin-negatives) hero figures (plan §6).

CPU-only; consumes ``analysis/registered_reads_542_nt.json`` (the PRIMARY
partial-correlation localization read) + the per-arm G tensors + the L22
last_prompt centroids. Figures -> ``figures/issue_542/genuine-near-twin-negatives/``
(PNG+PDF with the repo paper-plots rcParams + commit-pinned meta.json).

Heroes (plan §6):

- ``nt_hero1_partial_forest`` -- forest of the per-arm partial Spearman
  rho(leakage L, dist-to-nearest-near-twin-neg | dist-to-source) for nt_close
  vs xfam_long, with bootstrap CIs + the seed-43 noise-floor band. POSITIVE-rho
  convention: a reliably positive nt_close partial = near-twin localization.
- ``nt_hero2_scatter`` -- per-held-out-bystander leakage L vs
  dist-to-nearest-near-twin-neg, nt_close vs xfam_long, RAW alongside the
  dist-to-source-residualized scatter (the "show raw alongside processed" rule).

If a K1'' non-saturation abort landed (``analysis/k1prime_nt_abort.json``), the
hero figures are SKIPPED and the realized per-cell ``log P - base`` landing
table is plotted instead (``nt_k1prime_landing``) -- the abort-and-report
finding (plan §7 K1'').
"""

# ruff: noqa: RUF001  -- figure labels deliberately use the marker glyph and the
# Greek rho character (the project's marker-leakage measurement convention).
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

import i542_registered_reads as rr  # noqa: E402  (sibling-script reuse: ArmTensor + read helpers)

ARM_LABELS = {
    "nt_close": "Genuine near-twins\n(surface-perturbed sources)",
    "xfam_long": "Cross-family distant\n(matched control)",
}
ARM_COLORS = {"nt_close": "#1b9e77", "xfam_long": "#7570b3"}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _save(fig, fig_dir: Path, name: str, meta: dict) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    (fig_dir / f"{name}.meta.json").write_text(
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


def _hero1_forest(nt: dict, fig_dir: Path) -> None:
    """Forest of per-arm partial Spearman with bootstrap CIs + noise-floor band."""
    arms = [a for a in ("nt_close", "xfam_long") if a in nt.get("arms", {})]
    arms = [a for a in arms if "partial_spearman" in nt["arms"][a]]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ys = list(range(len(arms)))[::-1]
    for y, arm in zip(ys, arms, strict=True):
        a = nt["arms"][arm]
        pt = a["partial_spearman"]
        ci = a.get("cluster_bootstrap", {}).get("ci95", [None, None])
        lo = ci[0] if ci[0] is not None else pt
        hi = ci[1] if ci[1] is not None else pt
        ax.errorbar(
            [pt],
            [y],
            xerr=[[max(0.0, pt - lo)], [max(0.0, hi - pt)]],
            fmt="o",
            color=ARM_COLORS[arm],
            capsize=4,
            markersize=8,
        )
    # Noise-floor band (seed-43 repl_nt 2x |partial|).
    floor = nt.get("noise_floor")
    if isinstance(floor, dict) and floor.get("floor_2x_abs") is not None:
        f = floor["floor_2x_abs"]
        ax.axvspan(-f, f, color="0.85", alpha=0.6, label=f"seed-43 noise floor (±{f:.2f})")
    ax.axvline(0.0, color="0.3", lw=1, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([ARM_LABELS[a] for a in arms])
    ax.set_xlabel(
        "partial Spearman ρ(leakage L, distance-to-near-twin-negative | distance-to-source)"
    )
    ax.set_title("Near-twin localization: positive ρ = leak-suppression near a near-twin negative")
    if floor and isinstance(floor, dict) and floor.get("floor_2x_abs") is not None:
        ax.legend(loc="lower right", fontsize=8)
    _save(
        fig,
        fig_dir,
        "nt_hero1_partial_forest",
        {
            "figure": "partial-correlation forest (PRIMARY localization read)",
            "sign_convention": nt.get("sign_convention"),
            "per_arm": {a: nt["arms"][a].get("partial_spearman") for a in arms},
            "noise_floor": floor,
        },
    )


def _hero2_scatter(nt: dict, eval_root: Path, fig_dir: Path) -> None:
    """Leakage L vs dist-to-near-twin-neg scatter; RAW + residualized (per arm)."""
    from scipy.stats import rankdata

    from explore_persona_space.experiments.i542_panels import PANELS

    arm_dirs = sorted(p.name for p in (eval_root / "G_arm").glob("*") if p.is_dir())
    arms = {a: rr.ArmTensor(a, eval_root) for a in arm_dirs if a in ("nt_close", "xfam_long")}
    if not arms:
        return
    qmasks = {a: rr._quarantine_mask(t) for a, t in arms.items()}
    centroids = rr._centroid_cache(eval_root)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    plotted = False
    for arm, t in arms.items():
        vecs = rr._nt_dist_vectors(t, PANELS[arm], centroids, qmasks[arm])
        if vecs["n_cells"] < 4:
            continue
        plotted = True
        c = ARM_COLORS[arm]
        # RAW (left): L vs dist-to-neg.
        axes[0].scatter(
            vecs["dist_to_neg"], vecs["leak"], s=14, alpha=0.55, color=c, label=ARM_LABELS[arm]
        )
        # Residualized (right): rank-residualize both on dist-to-source.
        ry = rankdata(vecs["leak"])
        rx = rankdata(vecs["dist_to_neg"])
        rz = rankdata(vecs["dist_to_source"])
        Z = np.column_stack([np.ones(rz.size), rz])
        by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
        bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
        axes[1].scatter(rx - Z @ bx, ry - Z @ by, s=14, alpha=0.55, color=c, label=ARM_LABELS[arm])
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_xlabel("distance to nearest near-twin negative")
    axes[0].set_ylabel("bystander leakage L = Δ log P(※)")
    axes[0].set_title("Raw")
    axes[1].set_xlabel("dist-to-neg rank residual | dist-to-source")
    axes[1].set_ylabel("leakage-L rank residual | dist-to-source")
    axes[1].set_title("Residualized on distance-to-source")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Bystander leakage vs proximity to a near-twin negative (predicted: positive trend)"
    )
    _save(
        fig,
        fig_dir,
        "nt_hero2_scatter",
        {"figure": "leakage-vs-distance scatter (raw + residualized), nt_close vs xfam_long"},
    )


def _k1prime_landing_fig(abort: dict, fig_dir: Path) -> None:
    """Plot the realized per-cell log P - base landing table (K1'' abort finding)."""
    table = abort.get("landing_table", {})
    arms = [a for a in ("nt_close", "xfam_long") if a in table]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    band = abort.get("band", [5.0, 12.0])
    ax.axhspan(band[0], band[1], color="0.85", alpha=0.6, label=f"usable band {band}")
    for arm in arms:
        cids = list(table[arm].keys())
        vals = [table[arm][c]["delta_logp"] for c in cids]
        xs = np.arange(len(cids))
        ax.scatter(xs, vals, label=ARM_LABELS[arm], color=ARM_COLORS[arm], s=18)
    ax.set_xticks(np.arange(len(table[arms[0]])))
    ax.set_xticklabels(list(table[arms[0]].keys()), rotation=90, fontsize=6)
    ax.set_ylabel("source log P − base (nat)")
    ax.set_title("K1″ non-saturation abort: realized per-cell landing (the finding)")
    ax.legend(fontsize=8)
    _save(
        fig,
        fig_dir,
        "nt_k1prime_landing",
        {
            "figure": "K1'' realized per-cell log P - base landing table (abort finding)",
            "n_violators": abort.get("n_violators"),
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=Path(os.environ.get("I542_EVAL_ROOT", str(REPO / "eval_results/issue_542"))),
    )
    args = ap.parse_args()
    eval_root = args.eval_root
    fig_dir = Path(
        os.environ.get(
            "I542_NT_FIG_DIR", str(REPO / "figures/issue_542/genuine-near-twin-negatives")
        )
    )

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()

    abort_p = eval_root / "analysis/k1prime_nt_abort.json"
    if abort_p.exists():
        # The longer budget saturated -> the landing table is the deliverable.
        abort = json.loads(abort_p.read_text())
        _k1prime_landing_fig(abort, fig_dir)
        print(f"[nt-figures] K1'' abort -> landing table figure only ({fig_dir})")
        return 0

    nt_p = eval_root / "analysis/registered_reads_542_nt.json"
    assert nt_p.exists(), (
        f"NT read missing: {nt_p} (run i542_registered_reads.py --nt-proximity first)"
    )
    nt = json.loads(nt_p.read_text())
    _hero1_forest(nt, fig_dir)
    _hero2_scatter(nt, eval_root, fig_dir)
    print(f"[nt-figures] wrote hero figures -> {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
